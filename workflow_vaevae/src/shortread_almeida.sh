#!/usr/bin/bash
thres=$1

vamb \
    --model vaevae \
    --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_predictor_${thres}_mem \
    --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fna \
    --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
    --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/almeida_taxonomy.tsv \
    -l 32 \
    -e 300 \
    -t 1024 \
    -q  \
    -pe 100 \
    -pt 1024 \
    -pq  \
    -pthr ${thres} \
    -o C \
    --cuda \
    --minfasta 200000

# vamb \
#     --model reclustering \
#     --latent_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
#     --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
#     --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_e50_dadam_reclustering \
#     --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
#     --minfasta 200000
