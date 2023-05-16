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
# PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
BAM_PATH = f'{DEPTH_PATH}/bam_sorted'
# ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'

PATH_CONTIGS = f'/home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz'
ABUNDANCE_PATH = f'/home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz'


exp_id = args['exp']
# MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}{exp_id}.pt'
MODEL_PATH = f'model_semisupervised_mmseq_almeida.pt'

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers
lengthof = dict(zip(contignames, lengths))

with open(f'clusters_mmseq_genus_{DATASET}_v4{exp_id}.tsv') as file:
    clusters = vamb.vambtools.read_clusters(file)
filtered_clusters = dict()

for cluster, contigs in clusters.items():
    size = sum(lengthof[contig] for contig in contigs)
    if size >= 200000:
        filtered_clusters[cluster] = clusters[cluster]

with vamb.vambtools.Reader(PATH_CONTIGS) as file:
    vamb.vambtools.write_bins(
        f'bins_almeida',
        filtered_clusters,
        file,
        maxbins=None,
        separator='C',
    )
