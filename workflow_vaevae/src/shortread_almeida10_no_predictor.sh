#!/usr/bin/bash
annotator=$1
thres=$2

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/people/svekut/almeida10_${annotator}_fix \
    --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
    --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/04_mmseq2/taxonomy_cami_kfold/almeida_10_samples_taxonomy_${annotator}.tsv\
    --no_predictor \
    -l 32 \
    -e 300 \
    -t 1024 \
    -q  \
    -o C \
    --cuda \
    --minfasta 200000


# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/people/svekut/cami2_almeida10/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/people/svekut/cami2_almeida10/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/people/paupie/vaevae/almeida_10_samples/03_abundances/abundances/contigs.flt.fna.gz \
#     --rpkm /home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/almeida10/abundance.npz \
#     --outdir /home/projects/cpr_10006/people/svekut/cami2_almeida10_reclustering \
#     --algorithm kmeans \
#     --minfasta 200000
