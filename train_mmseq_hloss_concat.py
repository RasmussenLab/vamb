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

exp_id = '_hloss_mmseq_predict'

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

df_gt = pd.read_csv(GT_PATH)
graph_column = df_gt[f'predictions{exp_id}']
nodes, ind_nodes, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

vae = vamb.h_loss.VAEConcatHLoss(
     rpkms.shape[1], 
     len(nodes), 
     nodes, 
     table_parent,
     cuda=CUDA,
     logfile=sys.stdout,
)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader_joint,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=set(),
    )
    print('training')

latent_both = vae.encode(dataloader_joint)
LATENT_PATH = f'latent_concat_both_{DATASET}{exp_id}.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent_both)

composition.metadata.filter_mask(mask_vamb)
names = composition.metadata.identifiers

with open(f'clusters_concat_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_both)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)
