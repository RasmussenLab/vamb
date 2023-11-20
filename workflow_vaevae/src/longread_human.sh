#!/usr/bin/bash
annotator=$1
thres=$2
    # --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/results_taxonomy_predictor.csv \


vamb bin taxvamb \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_${annotator}_predictor_${thres} \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/human_longread_taxonomy_${annotator}.tsv \
    -l 64 \
    -e 1000 \
    -t 1024 \
    -q  \
    -pe 100 \
    -pt 1024 \
    -pq  \
    -pthr ${thres} \
    -o C \
    --cuda \
    --minfasta 200000

# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_${annotator}_predictor_${thres}/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_${annotator}_predictor_${thres}/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
#     --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
#     --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_${annotator}_predictor_${thres}_reclustering \
#     --hmmout_path /home/projects/cpr_10006/projects/semi_vamb/data/marker_genes/markers_human.hmmout \
#     --taxonomy_predictions /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevae_${annotator}_predictor_${thres}/results_taxonomy_predictor.csv \
#     --algorithm dbscan \
#     --minfasta 200000
