#!/usr/bin/bash

# --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
# --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_flatsoftmax \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fna \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/almeida_taxonomy_metabuli_otu.tsv \
    --no_predictor \
    -l 32 \
    -t 512 \
    -e 200 \
    -q \
    -o C \
    --cuda \
    --minfasta 200000


vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_flatsoftmax/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_flatsoftmax/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fna \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_flatsoftmax_reclustering \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_almeida.hmmout \
    --algorithm kmeans \
    --minfasta 200000
