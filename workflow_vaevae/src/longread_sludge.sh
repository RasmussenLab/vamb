#!/usr/bin/bash

    # --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_sp100/results_taxonomy_predictor.csv \


vamb \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/paupie/vaevae/mmseq2_annotations/long_read_sludge/lr_sludge_taxonomy.tsv \
    -l 64 \
    -e 500 \
    -q  \
    -pe 100 \
    -pq  \
    --n_species 100 \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam_reclustering \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_sludge.hmmout \
    --taxonomy_predictions /home/projects/cpr_10006/people/svekut/vamb/results_taxonomy_predictor_sludge.csv \
    --algorithm dbscan \
    --minfasta 200000
