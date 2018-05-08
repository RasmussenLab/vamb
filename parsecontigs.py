
__doc__ = """Calculate z-normalized tetranucleotide frequency from a FASTA file.

Input: A fasta file containing only A, C, G, T and N sequences
Output: A (n_sequences * 136) matrix of tetranucleotide frequencies,
z-normalized through axis 0.
"""



import sys
import os
import argparse
import vamb.tools

import numpy as _np



def main(contigpath, lengthspath, tnfpath):
    tnf_list = list()

    with vamb.tools.Reader(contigpath, 'rb') as contigfile, open(lengthspath, 'w') as lengthsfile:
        print('#contig\tlength', file=lengthsfile)
        
        entries = vamb.tools.byte_iterfasta(contigfile)

        for entry in entries:
            tnf_list.append(entry.fourmer_freq())
            
            print(entry.header, len(entry), sep='\t', file=lengthsfile)
            
    tnfs = _np.array(tnf_list, dtype=_np.float32)
    
    del tnf_list
    
    vamb.tools.zscore(tnfs, axis=0, inplace=True)
    
    _np.savetxt(tnfpath, tnfs, delimiter='\t', fmt='%.4f')



if __name__ == '__main__':
    parserkws = {'prog': 'calculate_tnf.py',
                 'formatter_class': argparse.RawDescriptionHelpFormatter,
                 'usage': 'calculate_tnf.py contigs.fna(.gz) tnfout lengthsout',
                 'description': __doc__}

    # Create the parser
    parser = argparse.ArgumentParser(**parserkws)

    parser.add_argument('contigs', help='FASTA file of contigs')
    parser.add_argument('tnfout', help='TNF output path')
    parser.add_argument('lengthsout', help='lengths output path')

    # Print help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if not os.path.isfile(args.contigs):
        raise FileNotFoundError(args.contigs)

    if os.path.exists(args.tnfout):
        raise FileExistsError(args.tnfout)
        
    if os.path.exists(args.lengthsout):
        raise FileExistsError(args.lengthsout)
        
    main(args.contigs, args.tnfout, args.lengthsout)

