
__doc__ = """Calculate z-normalized tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""



import sys as _sys
import os as _os
import numpy as _np
import gzip as _gzip
import vamb.vambtools as _vambtools



def read_contigs(byte_iterator, minlength=100):
    """Parses a FASTA file open in binary reading mode.
    
    Input:
        byte_iterator: Iterator of binary lines of a FASTA file
        minlength[100]: Ignore any references shorter than N bases 
    
    Outputs:
        tnfs: A (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A lNumpy array of contig headers
        lengths: A Numpy array of contig lengths
    """
    
    tnf_list = list()
    contignames = list()
    lengths = list()
    
    if byte_iterator is not iter(byte_iterator):
        raise ValueError('byte_iterator is not an iterator')
        
    entries = _vambtools.byte_iterfasta(byte_iterator)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnf_list.append(entry.fourmer_freq())
        contignames.append(entry.header)
        lengths.append(len(entry))
    
    tnfs = _np.array(tnf_list, dtype=_np.float32)
    del tnf_list
    
    lengths = _np.array(lengths, dtype=_np.int)
    contignames = _np.array(contignames)

    return tnfs, contignames, lengths

