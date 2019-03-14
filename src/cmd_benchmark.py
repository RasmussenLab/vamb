#!/usr/bin/python
import sys
import argparse

parser = argparse.ArgumentParser(
    description="""Command-line benchmark utility.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False)

parser.add_argument('vambpath', help='Path to vamb directory')
parser.add_argument('clusterspath', help='Path to clusters.tsv')
parser.add_argument('refpath', help='Path to reference file')
parser.add_argument('--disjoint', action='store_true', help='Enforce disjoint clusters')
parser.add_argument('-m', dest='min_bin_size', metavar='', type=int,
                    default=200000, help='Minimum genome size')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

sys.path.append(args.vambpath)
import vamb
import os
from collections import defaultdict

def split(clusters, sep='C'):
    """Split bins according to the label before the given separator"""
    result = dict()
    for binname, contignames in clusters.items():
        by_label = defaultdict(set)

        for contigname in contignames:
            label, _, name = contigname.partition(sep)
            by_label[label].add(contigname)

        for label, splitbin in by_label.items():
            result[str(binname) + '_' + label] = splitbin

    return result

def filtercontigs(clusters, reference, minsize=args.min_bin_size):
    filtered = dict()
    for binname, contignames in clusters.items():
        length = 0
        for contigname in contignames:
            contig = reference.contigs.get(contigname)
            if contig is None:
                raise ValueError('Contig {} not in reference'.format(contigname))
            length += contig.end - contig.start + 1

        if length >= minsize:
            filtered[binname] = contignames

    return filtered

# Check that files exist
for path in args.clusterspath, args.refpath:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

with open(args.clusterspath) as file:
    clusters = vamb.cluster.read_clusters(file)

with open(args.refpath) as file:
    reference = vamb.benchmark.Reference.from_file(file)

split_bins = split(clusters)
filtered_bins = filtercontigs(split_bins, reference)

bins = vamb.benchmark.Binning(filtered_bins, reference, disjoint=args.disjoint)

# Create dict of intersections for each bin
intersectionsof = defaultdict(dict)
for genome, bindict in bins.intersectionsof.items():
    for binname, intersection in bindict.items():
        intersectionsof[binname][genome] = intersection

print('# binname', 'genome', 'recall', 'precision', 'F1', 'binsize', sep='\t')
for binname, contigs in bins.contigsof.items():
    bestgenome, bestf1, recall, precision = None, None, None, None
    binsize = bins.breadthof[binname]

    for genome, intersection in intersectionsof[binname].items():
        tp, tn, fp, fn = bins.confusion_matrix(genome, binname)
        f1 = 2*tp / (2*tp + fp + fn)
        if bestf1 is None or f1 > bestf1:
            bestf1 = f1
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            bestgenome = genome.name

    if bestf1 is None:
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(binname, bestgenome, recall, precision, bestf1, binsize))
    else:
            print('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(binname, bestgenome, recall, precision, bestf1, binsize))
