#!/usr/bin/bash
module load tools computerome_utils/2.0
module unload gcc
module load gcc/11.1.0
module unload anaconda3/4.4.0
module load anaconda3/2023.03
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate /home/projects/cpr_10006/people/svekut/.conda/vamb

nepoch=1000

python3 train_mmseq_hloss_vamb2label_longread.py  --nepoch $nepoch --cuda