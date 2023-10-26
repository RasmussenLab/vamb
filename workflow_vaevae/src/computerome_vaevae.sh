#!/usr/bin/bash
module load tools computerome_utils/2.0
module load anaconda3/2023.03
module unload gcc
module load gcc/11.1.0
module load minimap2/2.17r941 samtools/1.10

source ~/.bashrc
conda init bash
conda activate /home/projects/cpr_10006/people/svekut/.conda/vamb

./$1 $2 $3 $4
