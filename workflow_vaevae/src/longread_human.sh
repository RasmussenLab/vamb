#!/usr/bin/bash

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vambout/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/longread_taxonomy_2023.tsv \
    -l 64 \
    -e 1000 \
    -q 25 75 150 500 \
    -pe 100 \
    -pq 25 75 \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/projects/semi_vamb/data/sludge/contigs_2kbp.fna \
    --rpkm /home/projects/cpr_10006/projects/semi_vamb/data/sludge/vambout/abundance.npz \
    --outdir /home/projects/cpr_10006/projects/semi_vamb/data/human_longread/vaevaeout_reclustering \
    --minfasta 200000
