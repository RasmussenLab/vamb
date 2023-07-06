#!/usr/bin/bash

# --taxonomy /home/projects/cpr_10006/people/paupie/vaevae/mmseq2_annotations/long_read_sludge/lr_sludge_taxonomy.tsv \

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --taxonomy_predictions /home/projects/cpr_10006/people/svekut/vamb/results_taxonomy_predictor.csv \
    -l 64 \
    -e 500 \
    -q 150 \
    -pe 100 \
    -pq 25 75 \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_reclustering \
    --minfasta 200000
