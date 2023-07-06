#!/usr/bin/bash
dataset=$1

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out \
    --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/${dataset}_taxonomy_2023.tsv \
    -l 64 \
    -e 500 \
    -q 25 75 150 \
    -pe 100 \
    -pq 25 75 \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_out_reclustering \
    --minfasta 200000
