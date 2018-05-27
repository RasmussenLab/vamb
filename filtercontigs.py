
__doc__ = """Filter contigs by length

Input: Path to FASTA file with contigs
Output: Writes FASTA file of filtered contigs
"""



import sys as _sys
import os as _os
import argparse as _argparse

if __package__ is None or __package__ == '':
    import vambtools as _vambtools
    
else:
    import vamb.vambtools as _vambtools



def filtercontigs(infile, outfile, minlength=2000):
    fasta_entries = _vambtools.byte_iterfasta(infile)

    for entry in fasta_entries:
        if len(entry) > minlength:
            print(entry.format(), file=outfile)



if __name__ == '__main__':
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

