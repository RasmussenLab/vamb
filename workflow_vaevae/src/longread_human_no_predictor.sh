#!/usr/bin/bash
    # --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/results_taxonomy_predictor.csv \
    # --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/longread_taxonomy_2023.tsv \

    # --cuda \


vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_flat_softmax \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/human_longread_taxonomy_metabuli_otu.tsv \
    --no_predictor \
    -l 64 \
    -e 500 \
    -t 1024 \
    -q  \
    -o C \
    --cuda \
    --minfasta 200000

# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/raw_dbscan_full/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/raw_dbscan_full/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
#     --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
#     --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/raw_dbscan_reclustering \
#     --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_human.hmmout \
#     --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/results_taxonomy_predictor.csv \
#     --algorithm dbscan \
#     --minfasta 200000

