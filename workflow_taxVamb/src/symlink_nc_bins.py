import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", type=str, help="outdir where symlinks will be stores ")
parser.add_argument(
    "--cs_d", type=str, help="cluster_score dictionary"
)
parser.add_argument(
    "--bp_d", type=str, help="bin_path dictionary"
)

parser.add_argument("-x", type=str,default='.fna', help="bins extensions ")

parser.add_argument("--min_comp", type=float, help="Minimum bin completeness")
parser.add_argument("--max_cont", type=float, help="Maximum bin contamination")

opt = parser.parse_args()

with open(opt.cs_d,'r') as f:
    cs_d = json.load(f)


with open(opt.bp_d,'r') as f:
    bp_d = json.load(f)

try:
    os.mkdir(opt.o)
except:
    pass

bin_format = opt.x
min_comp = opt.min_comp * 100
max_cont = opt.max_cont * 100

for cluster,comp_cont in cs_d.items():
    comp,cont = comp_cont
    bin_ = cluster+bin_format
    if (comp >= min_comp and cont <= max_cont ):
        os.symlink(os.path.join(os.getcwd(),bp_d[bin_]),os.path.join(opt.o,bin_))

 
