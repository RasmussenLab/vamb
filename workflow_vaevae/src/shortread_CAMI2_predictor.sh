#!/usr/bin/bash
dataset=$1
run_id=$2
keyword=$3

    # --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/${dataset}_taxonomy.tsv \
    # --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/${dataset}_taxonomy_${run_id}.tsv \
# 
vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_predictor_${keyword}_${run_id}_abs_in \
    --fasta /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/contigs_2kbp.fna.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/data/datasets/cami2_${dataset}/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/${dataset}_taxonomy_${run_id}.tsv \
    -pe 100 \
    -pq  \
    -pt 1024 \
    --cuda \
    -ploss ${keyword}
