
__doc__ = """Filtering contigs by length

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



def filtercontigs(inpath, outpath, minlength=2000):
    with open(inpath, 'rb') as infile, open(outpath, 'w') as outfile:
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
        
    filtercontigs(args.inpath, args.outpath, args.minlength)
