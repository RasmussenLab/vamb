#!/usr/bin/bash
folder=$(awk -v var="$PBS_ARRAYID" 'NR==var {print; exit}' /home/projects/cpr_10006/people/svekut/almeida_N_samples/folder_list_10.txt)

# vamb bin default \
#     --outdir /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/vambout \
#     --fasta /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/contigs.fna.gz \
#     --bamfiles /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/bam_files/*.bam \
#     -l 32 \
#     -e 300 \
#     -q 25 75 150 \
#     -o C \
#     --cuda \
#     --minfasta 200000

vamb bin taxvamb \
    --outdir /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/taxvambout \
    --fasta /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/contigs.fna.gz \
    --bamdir /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/bam_files \
    --taxonomy /home/projects/cpr_10006/people/svekut/almeida_N_samples/data_folders/${folder}/taxonomy_mmseqs.tsv \
    -l 32 \
    -e 300 \
    -t 1024 \
    -q  \
    -pe 100 \
    -pt 1024 \
    -pq  \
    -pthr 0.95 \
    -o C \
    --cuda \
    --minfasta 200000

