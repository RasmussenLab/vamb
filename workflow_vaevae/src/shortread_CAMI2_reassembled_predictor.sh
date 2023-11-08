#!/usr/bin/bash
dataset=$1
run_id=$2

vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_${dataset}_predictor_${run_id} \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/contigs.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/cami_errorfree/${dataset}/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/paupie/vaevae/mmseq2_annotations/ptracker/${dataset}_taxonomy.tsv \
    -pe 100 \
    -pq 25 75 \
    -pthr 0.7 \
    --cuda
