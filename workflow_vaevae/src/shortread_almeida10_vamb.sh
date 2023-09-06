#!/usr/bin/bash

vamb \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    -o C \
    --cuda \
    --minfasta 200000

vamb \
    --model reclustering \
    --latent_path /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb/latent.npz \
    --clusters_path /home/projects/cpr_10006/people/svekut/cami2_almeida10_vamb/vae_clusters.tsv  \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_reclustering_vamb \
    --algorithm kmeans \
    --minfasta 200000
