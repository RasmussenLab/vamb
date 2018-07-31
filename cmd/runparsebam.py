import argparse

usage = "python parsebam.py [OPTION ...] OUTFILE BAMFILE [BAMFILE ...]"
parser = _argparse.ArgumentParser(
    description=__doc__,
    formatter_class=_argparse.RawDescriptionHelpFormatter,
    usage=usage)

# Get the maximal number of extra processes to spawn
cpus = min(8, _os.cpu_count())

# Positional arguments
parser.add_argument('outfile', help='path to output file')
parser.add_argument('bamfiles', help='relative path to find bam files', nargs='+')

# Optional arguments
parser.add_argument('-a', dest='minscore', type=int, default=50,
                    help='minimum alignment score [50]')
parser.add_argument('-m', dest='minlength', type=int, default=100,
                    help='discard any references shorter than N [100]')
parser.add_argument('-p', dest='processes', type=int, default=cpus,
                    help=('number of extra processes to spawn '
                          '[min(' + str(cpus) + ', nfiles)]'))

# If no arguments, print help
if len(_sys.argv) == 1:
    parser.print_help()
    _sys.exit()

args = parser.parse_args()

# Check number of cores
if args.processes < 1:
    raise argsparse.ArgumentTypeError('Zero or negative processes requuested.')

if args.minlength < 4:
    raise ValueError('Minlength must be at least 4')

# Check presence of output file
if _os.path.exists(args.outfile):
    raise FileExistsError('Output file: ' + args.outfile)

directory = _os.path.dirname(args.outfile)
if directory and not _os.path.isdir(directory):
    raise NotADirectoryError(directory)

# Check presence of all BAM files
for bamfile in args.bamfiles:
    if not _os.path.isfile(bamfile):
        raise FileNotFoundError('Bam file: ' + bamfile)

# If more CPUs than files, scale down CPUs
if args.processes > len(args.bamfiles):
    args.processes = len(args.bamfiles)
    print('More processes given than files, scaling to {}.'.format(args.processes))

print('Number of files:', len(args.bamfiles))
print('Number of parallel processes:', args.processes)

sample_rpkms, contignames = read_bamfiles(args.bamfiles, args.minscore,
                                          args.minlength, args.processes)

with open(args.outfile, 'w') as file:
    write_rpkms(file, sample_rpkms, args.bamfiles)
