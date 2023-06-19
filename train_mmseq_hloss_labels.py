import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

exp_id = '_hloss_gt'

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = args['dataset']
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
BAM_PATH = f'{DEPTH_PATH}/bam_sorted'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_semisupervised_mmseq_{DATASET}{exp_id}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'
GT_PATH = f'gt_tax_{DATASET}.csv'

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers

# df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)
# df_mmseq_genus = df_mmseq[(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]
# df_mmseq[8] = df_mmseq[8].fillna('d_Bacteria')
# df_mmseq.loc[df_mmseq[8].isin(['-_root']), 8] = 'd_Bacteria'

# df_mmseq_family = df_mmseq[(df_mmseq[2] == 'family') |(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]
df_gt = pd.read_csv(GT_PATH)
df_gt['tax'] = df_gt['tax'].str.replace('Bacteria', 'd_Bacteria', regex=False)
df_gt['tax'] = df_gt['tax'].str.replace('Archaea', 'd_Archaea', regex=False)

classes_order = list(df_gt['tax'].str.split(';').str[-1])

ind_map = {c: i for i, c in enumerate(contignames)}
indices_mmseq = [ind_map[c] for c in df_gt['contigs']]

nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(df_gt['tax'].unique())

targets = [ind_nodes[i] for i in classes_order]

# vae = vamb.semisupervised_encode.VAELabels(
#      len(set(classes_order)), 
#      cuda=CUDA,
#      logfile=sys.stdout,
# )

vae = vamb.h_loss.VAELabelsHLoss(
     len(nodes), 
     table_indices, 
     table_true, 
     table_walkdown, 
     nodes, 
     table_parent,
     cuda=CUDA,
     logfile=sys.stdout,
)

with open(f'indices_mmseq_genus_{DATASET}{exp_id}.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
# dataloader_labels, mask = vamb.semisupervised_encode.make_dataloader_labels(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], classes_order)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], targets, len(nodes), table_parent)

with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader_labels,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
    )
    print('training')

latent_both = vae.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_lengths_mmseq_both_{DATASET}{exp_id}.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent_both)

composition.metadata.filter_mask(mask_vamb)
names = composition.metadata.identifiers

with open(f'clusters_mmseq_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_both)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)
