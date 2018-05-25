
__doc__ = """Calculate z-normalized tetranucleotide frequency from a FASTA file.

Input: A fasta file containing only A, C, G, T and N sequences
Output: A (n_sequences * 136) matrix of tetranucleotide frequencies,
z-normalized through axis 0.
"""



import sys as _sys
import os as _os
import argparse as _argparse
import numpy as _np
import gzip as _gzip

if __package__ is None or __package__ == '':
    import vambtools as _vambtools
    
else:
    import vamb.vambtools as _vambtools



TNF_HEADER = '#contigheader\t' + '\t'.join([
'AAAA/TTTT', 'AAAC/GTTT', 'AAAG/CTTT', 'AAAT/ATTT',
'AACA/TGTT', 'AACC/GGTT', 'AACG/CGTT', 'AACT/AGTT',
'AAGA/TCTT', 'AAGC/GCTT', 'AAGG/CCTT', 'AAGT/ACTT',
'AATA/TATT', 'AATC/GATT', 'AATG/CATT', 'AATT',
'ACAA/TTGT', 'ACAC/GTGT', 'ACAG/CTGT', 'ACAT/ATGT',
'ACCA/TGGT', 'ACCC/GGGT', 'ACCG/CGGT', 'ACCT/AGGT',
'ACGA/TCGT', 'ACGC/GCGT', 'ACGG/CCGT', 'ACGT',
'ACTA/TAGT', 'ACTC/GAGT', 'ACTG/CAGT', 'AGAA/TTCT',
'AGAC/GTCT', 'AGAG/CTCT', 'AGAT/ATCT', 'AGCA/TGCT',
'AGCC/GGCT', 'AGCG/CGCT', 'AGCT',      'AGGA/TCCT',
'AGGC/GCCT', 'AGGG/CCCT', 'AGTA/TACT', 'AGTC/GACT',
'AGTG/CACT', 'ATAA/TTAT', 'ATAC/GTAT', 'ATAG/CTAT',
'ATAT',      'ATCA/TGAT', 'ATCC/GGAT', 'ATCG/CGAT',
'ATGA/TCAT', 'ATGC/GCAT', 'ATGG/CCAT', 'ATTA/TAAT',
'ATTC/GAAT', 'ATTG/CAAT', 'CAAA/TTTG', 'CAAC/GTTG',
'CAAG/CTTG', 'CACA/TGTG', 'CACC/GGTG', 'CACG/CGTG',
'CAGA/TCTG', 'CAGC/GCTG', 'CAGG/CCTG', 'CATA/TATG',
'CATC/GATG', 'CATG',      'CCAA/TTGG', 'CCAC/GTGG',
'CCAG/CTGG', 'CCCA/TGGG', 'CCCC/GGGG', 'CCCG/CGGG',
'CCGA/TCGG', 'CCGC/GCGG', 'CCGG',      'CCTA/TAGG',
'CCTC/GAGG', 'CGAA/TTCG', 'CGAC/GTCG', 'CGAG/CTCG',
'CGCA/TGCG', 'CGCC/GGCG', 'CGCG',      'CGGA/TCCG',
'CGGC/GCCG', 'CGTA/TACG', 'CGTC/GACG', 'CTAA/TTAG',
'CTAC/GTAG', 'CTAG',      'CTCA/TGAG', 'CTCC/GGAG',
'CTGA/TCAG', 'CTGC/GCAG', 'CTTA/TAAG', 'CTTC/GAAG',
'GAAA/TTTC', 'GAAC/GTTC', 'GACA/TGTC', 'GACC/GGTC',
'GAGA/TCTC', 'GAGC/GCTC', 'GATA/TATC', 'GATC',
'GCAA/TTGC', 'GCAC/GTGC', 'GCCA/TGGC', 'GCCC/GGGC',
'GCGA/TCGC', 'GCGC',      'GCTA/TAGC', 'GGAA/TTCC',
'GGAC/GTCC', 'GGCA/TGCC', 'GGCC', 'GGGA/TCCC',
'GGTA/TACC', 'GTAA/TTAC', 'GTAC', 'GTCA/TGAC',
'GTGA/TCAC', 'GTTA/TAAC', 'TAAA/TTTA', 'TACA/TGTA',
'TAGA/TCTA', 'TATA',      'TCAA/TTGA', 'TCCA/TGGA',
'TCGA',      'TGAA/TTCA', 'TGCA', 'TTAA']) + '\n'



def read_contigs(byte_iterator, minlength=2000):
    """Parses a FASTA file and produces a list of headers and a matrix of TNF.
    
    Input:
        contigpath: Path to a FASTA file with contigs
        min_length[2000]: Minimum length of contigs
    
    Outputs:
        tnfs: A (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        
    """
    
    tnf_list = list()
    contignames = list()
    lengths = list()
    
    if byte_iterator is not iter(byte_iterator):
        raise ValueError('byte_iterator is not a byte iterator')
        
    entries = _vambtools.byte_iterfasta(byte_iterator)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnf_list.append(entry.fourmer_freq())
        contignames.append(entry.header)
        lengths.append(len(entry))
            
    tnfs = _np.array(tnf_list, dtype=_np.float32)
    del tnf_list
    
    _vambtools.zscore(tnfs, axis=0, inplace=True)
    
    return tnfs, contignames, lengths



def write_tnf(tnfpath, contignames, tnfs):
    """Writes a TNF table to specified output path.
    
    Input:
        tnfpath: Path to write output
        contignames: A list of contig headers
        tnfs: A (n_FASTA_entries x 136) matrix of tetranucleotide freq.
    
    Outputs: None
    """
    
    formatstring = '{}\t' + '\t'.join(['{:.4f}']*136) + '\n'
    
    with _gzip.open(tnfpath, 'w') as file:
        file.write(TNF_HEADER.encode())
        for contigname, tnf in zip(contignames, tnfs):
            file.write(formatstring.format(contigname, *tnf).encode())



if __name__ == '__main__':
    parserkws = {'prog': 'calculate_tnf.py',
                 'formatter_class': _argparse.RawDescriptionHelpFormatter,
                 'usage': 'parsecontigs.py contigs.fna(.gz) tnfout lengthsout',
                 'description': __doc__}

    # Create the parser
    parser = _argparse.ArgumentParser(**parserkws)

    parser.add_argument('contigs', help='FASTA file of contigs')
    parser.add_argument('tnfout', help='TNF output path')
    
    parser.add_argument('-m', dest='minlength', type=int, default=2000,
                        help='minimum length of contigs [2000]')

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
    
    with _vambtools.Reader(args.contigs, 'rb') as filehandle:
        contignames, tnfs, lengths = read_contigs(filehandle, args.minlength)
        
    write_tnf(args.tnfout, contignames, tnfs)

