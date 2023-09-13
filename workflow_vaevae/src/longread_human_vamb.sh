#!/usr/bin/bash

vamb \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout64 \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    -l 64 \
    -e 1000 \
    -q 25 75 150 500 \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout64/latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout64/vae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout64_reclustering \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_human.hmmout \
    --algorithm dbscan \
    --minfasta 200000
