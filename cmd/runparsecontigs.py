import argparse

parserkws = {'prog': 'calculate_tnf.py',
             'formatter_class': argparse.RawDescriptionHelpFormatter,
             'usage': 'parsecontigs.py contigs.fna(.gz) tnfout lengthsout',
             'description': __doc__}

# Create the parser
parser = argparse.ArgumentParser(**parserkws)

parser.add_argument('contigs', help='FASTA file of contigs')
parser.add_argument('tnfout', help='TNF output path')

parser.add_argument('-m', dest='minlength', type=int, default=100,
                    help='minimum length of contigs [100]')

# Print help if no arguments are given
if len(_sys.argv) == 1:
    parser.print_help()
    _sys.exit()

args = parser.parse_args()

if args.minlength < 4:
    raise ValueError('Minlength must be at least 4')

if not _os.path.isfile(args.contigs):
    raise FileNotFoundError(args.contigs)

if _os.path.exists(args.tnfout):
    raise FileExistsError(args.tnfout)

directory = _os.path.dirname(args.tnfout)
if directory and not _os.path.isdir(directory):
    raise NotADirectoryError(directory)

with _vambtools.Reader(args.contigs, 'rb') as filehandle:
    tnfs, contignames, lengths = read_contigs(filehandle, args.minlength)

_vambtools.write_tsv(args.tnfout, tnfs)
