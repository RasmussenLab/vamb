#!/usr/bin/bash
dataset=$1
run_id=$2

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reassembled_${run_id} \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/contigs.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/paupie/vaevae/mmseq2_annotations/ptracker/${dataset}_taxonomy.tsv \
    -l 32 \
    -e 500 \
    -q 25 75 150 \
    -pe 100 \
    -pq 25 75 \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reassembled_${run_id}/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reassembled_${run_id}/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reclustering_reassembled_${run_id} \
    --algorithm kmeans \
    --minfasta 200000
