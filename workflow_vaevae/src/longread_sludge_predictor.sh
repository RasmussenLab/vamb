#!/usr/bin/bash
# --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
# --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
run_id=$1

vamb \
    --model taxonomy_predictor \
    --outdir /home/projects/cpr_10006/people/svekut/long_read_sludge_kfold_predictor_lengths_${run_id} \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/long_read_sludge_taxonomy_${run_id}.tsv \
    -pe 100 \
    -pq 25 75 \
    --cuda
