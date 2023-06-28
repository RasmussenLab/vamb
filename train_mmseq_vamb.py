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

exp_id = '_longread'
SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = 'longread'
DEPTH_PATH = f'/home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout'
COMPOSITION_PATH = f'{DEPTH_PATH}/composition.npz'
ABUNDANCE_PATH = f'{DEPTH_PATH}/abundance.npz'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}{exp_id}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy_2023.tsv'

rpkms = np.load(ABUNDANCE_PATH, mmap_mode='r')['matrix']
composition = np.load(COMPOSITION_PATH, mmap_mode='r', allow_pickle=True)
tnfs, lengths = composition['matrix'], composition['lengths']
contignames = composition['identifiers']

vae = vamb.encode.VAE(nsamples=rpkms.shape[1], cuda=CUDA)

dataloader_vamb, mask = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader_vamb,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[25, 75, 150],
    )
    print('training')

latent = vae.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_lengths_vamb_{DATASET}{exp_id}.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent)

names = contignames

with open(f'clusters_vamb_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)