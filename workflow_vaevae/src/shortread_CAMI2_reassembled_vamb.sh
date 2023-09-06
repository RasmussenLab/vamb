#!/usr/bin/bash
dataset=$1
run_id=$2
    # --taxonomy_predictions /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reassembled_1/results_taxonomy_predictor.csv \

# vamb \
#     --model vae \
#     --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_vamb_${run_id} \
#     --fasta /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/contigs.fna \
#     --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/vambout/abundance.npz \
#     -o C \
#     --cuda \
#     --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_vamb_${run_id}/latent.npz \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_${dataset}_vamb_${run_id}/vae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/contigs.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_reclustering_vamb_${run_id} \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/cami_reassembled_${dataset}.hmmout \
    --algorithm kmeans \
    --minfasta 200000
