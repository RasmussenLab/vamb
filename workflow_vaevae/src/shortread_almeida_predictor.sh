#!/usr/bin/bash
# --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
# --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
run_id=$1

vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/almeida_kfold_predictor_flat_softmax_${run_id} \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fna \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/almeida_taxonomy_${run_id}.tsv \
    -pe 100 \
    -pq  \
    -pt 1024 \
    --cuda \
    -ploss flat_softmax
