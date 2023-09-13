#!/usr/bin/bash

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10 \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/almeida_10_samples/almeida_10_samples_taxonomy.tsv \
    -l 32 \
    -e 200 \
    -q 25 75 150 \
    -pe 100 \
    -pq 25 75 \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_almeida10/vaevae_latent.npy \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_almeida10/vaevae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_reclustering \
    --algorithm kmeans \
    --minfasta 200000
