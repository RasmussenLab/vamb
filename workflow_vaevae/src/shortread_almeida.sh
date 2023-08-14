#!/usr/bin/bash
# --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
# --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
    -l 32 \
    -t 512 \
    -e 60 \
    -q \
    -pe 50 \
    -pq \
    -o C \
    --n_species 100 \
    --cuda \
    --minfasta 200000


vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam_reclustering \
    --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
    --minfasta 200000
