import sys
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="""Command-line bin abundance estimator.
Print the median RPKM abundance for each bin in each sample to STDOUT.
Will read the RPKM file into memory - beware.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)

parser.add_argument("rpkmpath", help="Path to RPKM file")
parser.add_argument("clusterspath", help="Path to clusters.tsv")
parser.add_argument("headerpath", help="Path to list of headers")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

# Check files
for infile in (args.rpkmpath, args.clusterspath, args.headerpath):
    if not os.path.isfile(infile):
        raise FileNotFoundError(infile)

# Load Vamb
sys.path.append("../vamb")
import vamb

# Load in files
with open(args.headerpath) as file:
    indexof = {line.strip(): i for i, line in enumerate(file)}

with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)

# Check that all clusters names are in headers:
for cluster in clusters.values():
    for header in cluster:
        if header not in indexof:
            raise KeyError("Header not found in headerlist: {}".format(header))

# Load RPKM and check it
rpkm = vamb.vambtools.read_npz(args.rpkmpath)
nsamples = rpkm.shape[1]

if len(indexof) != len(rpkm):
    raise ValueError("Not the same number of headers as rows in RPKM file")

# Now estimate abundances
for clustername, cluster in clusters.items():
    depths = np.empty((len(cluster), nsamples), dtype=np.float32)

    for row, header in enumerate(cluster):
        index = indexof[header]
        depths[row] = rpkm[index]

    median_depths = np.median(depths, axis=0)

    print(clustername, end="\t")
    print("\t".join([str(i) for i in median_depths]))
