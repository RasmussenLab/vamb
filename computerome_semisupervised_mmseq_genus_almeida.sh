#!/usr/bin/bash
module load tools computerome_utils/2.0
module unload gcc
module load gcc/11.1.0
module load anaconda3/4.4.0
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate vamb_env

nepoch=2

python3 train_mmseq_almeida.py  --nepoch $nepoch --cuda  --dataset almeida