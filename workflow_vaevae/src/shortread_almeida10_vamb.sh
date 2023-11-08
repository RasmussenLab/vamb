#!/usr/bin/bash

# vamb \
#     --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb_20102023 \
#     --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
#     --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
#     -l 32 \
#     -e 300 \
#     -q 25 75 150 \
#     -o C \
#     --cuda \
#     --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb_20102023/latent.npz \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb_20102023/vae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb_20102023_reclustering \
    --algorithm kmeans \
    --minfasta 200000
