__doc__ = """Filter contigs by length

Input: Path to FASTA file with contigs
Output: Writes FASTA file of filtered contigs
"""

import argparse as _argparse

parserkws = {'prog': 'calculate_tnf.py',
             'formatter_class': _argparse.RawDescriptionHelpFormatter,
             'usage': 'filtercontigs.py inpath outpath',
             'description': __doc__}

# Create the parser
parser = _argparse.ArgumentParser(**parserkws)

parser.add_argument('inpath', help='input FASTA file')
parser.add_argument('outpath', help='output FASTA file')

parser.add_argument('-m', dest='minlength', type=int, default=2000,
                    help='minimum length of contigs [2000]')

# Print help if no arguments are given
if len(_sys.argv) == 1:
    parser.print_help()
    _sys.exit()

args = parser.parse_args()

if args.minlength < 4:
    raise ValueError('Minlength must be at least 4')

if not _os.path.isfile(args.inpath):
    raise FileNotFoundError(args.inpath)

if _os.path.exists(args.outpath):
    raise FileExistsError(args.outpath)

directory = _os.path.dirname(args.outpath)
if directory and not _os.path.isdir(directory):
    raise NotADirectoryError(directory)

with open(args.inpath) as infile, open(args.outpath, 'w') as outfile:
    filtercontigs(infile, outfile, args.minlength)