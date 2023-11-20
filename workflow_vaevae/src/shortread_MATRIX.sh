#!/usr/bin/bash
thres=$1

vamb bin default \
    --outdir /home/projects/cpr_10006/projects/vamb/MATRIX_vamb_no_split \
    --fasta /home/projects/ku_00200/data/matrix/mgx22_assembly/avamb_outdir/contigs.flt.fna.gz \
    --rpkm /home/projects/ku_00200/data/matrix/mgx22_assembly/avamb_outdir/abundance.npz \
    -l 32 \
    -e 300 \
    -t 256 \
    -q 25 75 100 150 \
    --cuda \
    --minfasta 200000

# vamb bin taxvamb \
#     --outdir /home/projects/cpr_10006/projects/vamb/MATRIX_taxvamb_${thres} \
#     --fasta /home/projects/ku_00200/data/matrix/mgx22_assembly/avamb_outdir/contigs.flt.fna.gz \
#     --rpkm /home/projects/ku_00200/data/matrix/mgx22_assembly/avamb_outdir/abundance.npz \
#     --taxonomy /home/projects/cpr_10006/people/svekut/mmseq2/MATRIX_taxonomy_v207.tsv \
#     -l 32 \
#     -e 300 \
#     -t 1024 \
#     -q  \
#     -pe 100 \
#     -pt 1024 \
#     -pq  \
#     -pthr ${thres} \
#     -o C \
#     --cuda \
#     --minfasta 200000

# vamb recluster \
#     --latent_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_predictor_${thres}_mem/vaevae_latent.npy \
#     --clusters_path /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_predictor_${thres}_mem/vaevae_clusters.tsv  \
#     --fasta /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.fa.gz \
#     --rpkm /home/projects/cpr_10006/projects/vamb/analysis/almeida/data/almeida.jgi.depth.npz \
#     --outdir /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout_predictor_${thres}_mem_reclustering \
#     --taxonomy_predictions /home/projects/cpr_10006/projects/vamb/almeida_vaevaeout/results_taxonomy_predictor.csv \
#     --minfasta 200000
