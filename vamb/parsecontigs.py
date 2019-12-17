__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import sys as _sys
import numpy as _np
import vamb.vambtools as _vambtools

def read_contigs(filehandle, minlength=100, preallocate=False):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]
        preallocate: [DEPRECATED] Read contigs twice, saving memory [False]

    Outputs:
        tnfs: An (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    if preallocate:
        print("Warning: Argument 'preallocate' in function read_contigs is deprecated."
        " read_contigs is now always memory efficient.",
        file=_sys.stderr)

    tnfs = _vambtools.PushArray(_np.float32)
    lengths = _vambtools.PushArray(_np.int)
    contignames = list()

    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs.extend(entry.fourmer_freq())
        lengths.append(len(entry))
        contignames.append(entry.header)

    tnfs_arr = tnfs.take().reshape(-1, 136)
    lengths_arr = lengths.take()

    return tnfs_arr, contignames, lengths_arr
