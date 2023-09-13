#!/usr/bin/bash

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
    -l 32 \
    -t 512 \
    -e 200 \
    -q \
    -pe 50 \
    -pq \
    --n_species 100 \
    --cuda \
    --minfasta 200000
