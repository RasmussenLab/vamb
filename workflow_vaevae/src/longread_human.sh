#!/usr/bin/bash
    # --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/results_taxonomy_predictor.csv \


vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout_dadam \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/longread_taxonomy_2023.tsv \
    -l 64 \
    -e 1000 \
    -q  \
    -pe 100 \
    -pq  \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout_dadam/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout_dadam/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout_dadam_reclustering \
    --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_human.hmmout \
    --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/results_taxonomy_predictor.csv \
    --algorithm dbscan \
    --minfasta 200000

