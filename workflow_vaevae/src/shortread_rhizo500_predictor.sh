#!/usr/bin/bash
vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_rhizo_500_predictor \
    --fasta /home/projects/cpr_10006/people/svekut/rhizo_data/rhizo.fna.gz \
    --rpkm /home/projects/cpr_10006/people/svekut/rhizo_data/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/rhizo_500_taxonomy_metabuli_otu.tsv \
    -pe 100 \
    -pq  \
    -pt 1024 \
    --cuda \
    -ploss flat_softmax
