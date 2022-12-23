#!/usr/bin/python
import sys
import argparse
import vamb
import os

parser = argparse.ArgumentParser(
    description="""Command-line benchmark utility.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)

parser.add_argument("clusterspath", help="Path to clusters.tsv")
parser.add_argument("refpath", help="Path to reference file")
parser.add_argument("--tax", dest="taxpath", help="Path to taxonomic maps")
parser.add_argument(
    "-m",
    dest="min_bin_size",
    metavar="",
    type=int,
    default=200000,
    help="Minimum size of bins [200000]",
)
parser.add_argument("-s", dest="separator", help="Binsplit separator", default=None)
parser.add_argument("--disjoint", action="store_true", help="Enforce disjoint clusters")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

# Check that files exist
for path in args.clusterspath, args.refpath, args.taxpath:
    if path is not None and not os.path.isfile(path):
        raise FileNotFoundError(path)

with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)

with open(args.refpath) as file:
    reference = vamb.benchmark.Reference.from_file(file)

if args.taxpath is not None:
    with open(args.taxpath) as file:
        reference.load_tax_file(file)

binning = vamb.benchmark.Binning(
    clusters,
    reference,
    minsize=args.min_bin_size,
    disjoint=args.disjoint,
    binsplit_separator=args.separator,
)

for rank in range(len(binning.counters)):
    binning.print_matrix(rank)
    print("")
