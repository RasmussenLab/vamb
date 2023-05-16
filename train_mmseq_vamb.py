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

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers

vae = vamb.encode.VAE(nsamples=rpkms.shape[1], cuda=CUDA)

dataloader_vamb, mask = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader_vamb,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
    )
    print('training')

latent = vae.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_lengths_vamb_{DATASET}.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent)

composition.metadata.filter_mask(mask)
names = composition.metadata.identifiers

with open(f'clusters_vamb_{DATASET}_v4.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)