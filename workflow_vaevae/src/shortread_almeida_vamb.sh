#!/usr/bin/bash

vamb \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_vamb \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fna \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    -l 32 \
    -e 300 \
    -q 25 75 150 \
    -o C \
    --cuda \
    --minfasta 200000

# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
#     --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
#     --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam_reclustering \
#     --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
#     --minfasta 200000
