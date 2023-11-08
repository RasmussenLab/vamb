#!/usr/bin/bash
keyword=$1

vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_marine_predictor_taxometer_${keyword} \
    --fasta /home/projects/cpr_10006/data/CAMI2/marine/vamb_output/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/data/CAMI2/marine/vamb_output/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/marine_taxonomy_${keyword}.tsv \
    -pe 100 \
    -pq  \
    -pt 1024 \
    --cuda \
    -ploss flat_softmax
