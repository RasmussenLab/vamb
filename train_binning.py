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
parser.add_argument("--exp", type=str, default='')
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
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'

# PATH_CONTIGS = f'/home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz'
# ABUNDANCE_PATH = f'/home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz'


# exp_id = args['exp']
exp_id = '_species_filter'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}{exp_id}.pt'
# MODEL_PATH = f'model_semisupervised_mmseq_almeida.pt'

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers

dataloader, mask = vamb.encode.make_dataloader(
    rpkms,
    composition.matrix,
    composition.metadata.lengths,
)

# latent_vamb = np.load(f'latent_trained_semisupervised_mmseq_genus_vamb_{DATASET}{exp_id}.npy')
# latent_both = np.load(f'latent_trained_semisupervised_mmseq_genus_both_{DATASET}{exp_id}.npy')
# latent_labels = np.load(f'latent_trained_semisupervised_mmseq_genus_labels_{DATASET}{exp_id}.npy')

latent_vamb = np.load(f'latent_trained_lengths_mmseq_genus_vamb_{DATASET}{exp_id}.npy')
latent_both = np.load(f'latent_trained_lengths_mmseq_genus_both_{DATASET}{exp_id}.npy')
latent_labels = np.load(f'latent_trained_lengths_mmseq_genus_labels_{DATASET}{exp_id}.npy')
inds = np.array(range(latent_vamb.shape[0]))
with open(f'indices_mmseq_genus_{DATASET}{exp_id}.pickle', 'rb') as fp:
    indices_both = pickle.load(fp)
new_indices = [np.argwhere(inds == i)[0][0] for i in indices_both]
print(latent_both.shape, len(new_indices), len(indices_both))
latent_vamb_new = latent_vamb.copy()
latent_vamb_new[new_indices] = latent_both

composition.metadata.filter_mask(mask)
names = composition.metadata.identifiers

with open(f'clusters_mmseq_species_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_vamb_new)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_species_vamb_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_vamb)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_species_labels_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_labels)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)
