#!/usr/bin/bash

    # --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_sp100/results_taxonomy_predictor.csv \


vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_metabuli_flatsoftmax_500 \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/sludge_taxonomy_metabuli_otu.tsv \
    --no_predictor \
    -l 64 \
    -e 500 \
    -t 512 \
    -q  \
    -o C \
    --cuda \
    --minfasta 200000

# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
#     --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
#     --outdir /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vaevaeout_dadam_reclustering \
#     --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_sludge.hmmout \
#     --taxonomy_predictions /home/projects/cpr_10006/people/svekut/vamb/results_taxonomy_predictor_sludge.csv \
#     --algorithm dbscan \
#     --minfasta 200000
