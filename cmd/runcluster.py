import argparse

usage = "python cluster.py [OPTIONS ...] INPUT OUTPUT"
parser = argparse.ArgumentParser(
    description=__cmd_doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    usage=usage)

# create the parser
parser.add_argument('input', help='input dataset')
parser.add_argument('output', help='output clusters')

parser.add_argument('-i', dest='inner',
                    help='inner distance threshold [No = Auto]', type=float)
parser.add_argument('-o', dest='outer',
                    help='outer distance threshold [No = Auto]', type=float)
parser.add_argument('-c', dest='max_clusters',
                    help='stop after creating N clusters [None, i.e. infinite]', type=int)
parser.add_argument('-s', dest='min_size',
                    help='minimum cluster size to output [1]',default=1, type=int)
parser.add_argument('--precluster', help='precluster first [False]', action='store_true')

# Print help if no arguments are given
if len(_sys.argv) == 1:
    parser.print_help()
    _sys.exit()

args = parser.parse_args()

inner, outer = _check_inputs(15, args.inner, args.outer)

if not _os.path.isfile(args.input):
    raise FileNotFoundError(args.input)

if _os.path.isfile(args.output):
    raise FileExistsError(args.output)

directory = _os.path.dirname(args.output)
if directory and not _os.path.isdir(directory):
    raise NotADirectoryError(directory)

if args.max_clusters is not None and args.precluster:
    raise ValueError('Cannot pass max clusters with precluster.')

matrix = _np.loadtxt(args.input, delimiter='\t', dtype=_np.float32)
_vambtools.zscore(matrix, axis=1, inplace=True)
labels, inner, outer = _check_params(matrix, inner, outer, None, 1000, 2500)

if args.precluster:
    contigsof = tandemcluster(matrix, labels, inner, outer, normalized=True)

else:
    contigsof = cluster(matrix, labels, inner, outer, normalized=True)

# Allow the garbage collector to clean the original matrix
del matrix

header = "clustername\tcontigindex(1-indexed)"

with open(args.output, 'w') as filehandle:
    writeclusters(filehandle, contigsof, args.max_clusters, args.min_size, header=header)
