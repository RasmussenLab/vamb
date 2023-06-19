#!/usr/bin/bash
module load tools computerome_utils/2.0
module load anaconda3/2023.03
module unload gcc
module load gcc/11.1.0
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate /home/projects/cpr_10006/people/svekut/.conda/vamb

nepoch=500
dataset=$1
PATH='/home/projects/ku_00200/data/matrix/mgx22_assembly/vamb'
/home/projects/cpr_10006/people/svekut/.conda/vamb/bin/vamb --outdir test_knud2 --fasta ${PATH}/outdir/contigs.flt.fna.gz -p 36 --rpkm ${PATH}/outdir/abundance.npz -m 2000 --minfasta 200000  --cuda -o C -m 2000 --minfasta 200000 --model vae-aae
