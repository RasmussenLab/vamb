import sys
import os
import argparse

parser = argparse.ArgumentParser(
    description="""Command-line bin creator.
Will read the entire content of the FASTA file into memory - beware.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False)

parser.add_argument('fastapath', help='Path to FASTA file')
parser.add_argument('clusterspath', help='Path to clusters.tsv')
parser.add_argument('outdir', help='Directory to create')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

sys.path.append('../vamb')
import vamb

with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)

with vamb.vambtools.Reader(args.fastapath, 'rb') as file:
    fastadict = vamb.vambtools.loadfasta(file)

vamb.vambtools.write_bins(args.outdir, clusters, fastadict, maxbins=None)
