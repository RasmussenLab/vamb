#!/usr/bin/bash


# vamb \
#     --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout64_20102023 \
#     --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
#     --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
#     -l 64 \
#     -e 500 \
#     -q 25 75 150 \
#     -o C \
#     --cuda \
#     --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout64_20102023/latent.npz \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout64_20102023/vae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout64_20102023_reclustering \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_sludge.hmmout \
    --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevae_full_predictor_0.5/results_taxonomy_predictor.csv \
    --algorithm dbscan \
    --minfasta 200000
