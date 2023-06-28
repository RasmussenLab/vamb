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

exp_id = '_hloss_jointtrain'

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = args['dataset']
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
BAM_PATH = f'{DEPTH_PATH}/bam_sorted'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'
GT_PATH = f'gt_tax_{DATASET}.csv'

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers

df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)

df_mmseq = df_mmseq[~(df_mmseq[2] == 'no rank')]
# df_mmseq_genus = df_mmseq[(df_mmseq[2] == 'genus') | (df_mmseq[2] == 'species')]
# df_mmseq_genus = df_mmseq_genus[df_mmseq_genus[7] > 0.8]
# classes_order = list(df_mmseq_genus[8].str.split(';').str[-1])

# df_gt = pd.read_csv(GT_PATH)
# df_gt['tax'] = df_gt['tax'].str.replace('Bacteria', 'd_Bacteria', regex=False)
# df_gt['tax'] = df_gt['tax'].str.replace('Archaea', 'd_Archaea', regex=False)
# df_gt['gt_genus'] = df_gt['tax'].str.split(';').str[:6].map(lambda x: ';'.join(x))
# df_gt['gt_species'] = df_gt['tax'].str.split(';').str[:7].map(lambda x: ';'.join(x))

# df_merged = pd.merge(df_mmseq, df_gt, left_on=0, right_on='contigs')
# df_merged['gt_tree'] = df_merged['gt_species']
# df_merged.loc[(df_merged[2] == 'genus'), 'gt_tree'] = df_merged.loc[(df_merged[2] == 'genus'), 'gt_genus']
# df_merged['len'] = df_merged['gt_tree'].str.split(';').str[:].map(lambda x: len(x))
# print(df_merged['len'].value_counts(), flush=True)

# df_mmseq_genus = df_merged[(df_merged[2] == 'genus') | (df_merged[2] == 'species')]

ind_map = {c: i for i, c in enumerate(contignames)}
indices_mmseq = [ind_map[c] for c in df_mmseq[0]]

graph_column = df_mmseq[8]
nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(graph_column.unique())

# indices_mmseq = [ind_map[c] for c in df_gt['contigs']]
classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

vae = vamb.h_loss.VAEVAEHLossPredict(
     rpkms.shape[1], 
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
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], targets, len(nodes), table_parent)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], targets, len(nodes), table_parent)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
dataloader = vamb.h_loss.make_dataloader_semisupervised_hloss(dataloader_joint, dataloader_vamb, dataloader_labels, len(nodes), table_parent, shapes)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[25, 75, 150],
        # batchsteps=[],
    )
    print('training')

latent_vamb = vae.VAEVamb.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_vamb_{DATASET}{exp_id}.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent_vamb)

latent_labels = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_labels_{DATASET}{exp_id}.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent_labels)

latent_both = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_both_{DATASET}{exp_id}.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent_both)

inds = np.array(range(latent_vamb.shape[0]))
# new_indices = [np.argwhere(inds == i)[0][0] for i in indices_mmseq]
# print(latent_both.shape, len(new_indices), len(indices_mmseq))
latent_vamb_new = latent_vamb.copy()
latent_vamb_new[indices_mmseq] = latent_both

composition.metadata.filter_mask(mask_vamb)
names = composition.metadata.identifiers

latent_vamb_new_labels = latent_vamb.copy()
latent_vamb_new_labels[indices_mmseq] = latent_labels

with open(f'clusters_mmseq_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_vamb_new)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_vamb_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_vamb)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_labels_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_vamb_new_labels)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)