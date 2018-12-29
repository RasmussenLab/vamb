
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

def _read_contigs_online(filehandle, minlength):
    tnfs = list()
    contignames = list()
    lengths = list()

    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs.append(entry.fourmer_freq())
        contignames.append(entry.header)
        lengths.append(len(entry))

    lengths = _np.array(lengths, dtype=_np.int)
    tnfs = _np.array(tnfs, dtype=_np.float32)

    return tnfs, contignames, lengths

def _read_contigs_preallocated(filehandle, minlength):
    n_entries = 0
    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
         if len(entry) >= minlength:
             n_entries += 1

    tnfs = _np.empty((n_entries, 136), dtype=_np.float32)
    contignames = [None] * n_entries
    lengths = _np.empty(n_entries, dtype=_np.int)

    filehandle.seek(0) # Go back to beginning of file
    entries = _vambtools.byte_iterfasta(filehandle)
    entrynumber = 0
    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs[entrynumber] = entry.fourmer_freq()
        contignames[entrynumber] = entry.header
        lengths[entrynumber] = len(entry)

        entrynumber += 1

    return tnfs, contignames, lengths

def read_contigs(filehandle, minlength=100, preallocate=True):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]
        preallocate: Read contigs twice, saving memory [True]

    Outputs:
        tnfs: An (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4')

    if preallocate:
        return _read_contigs_preallocated(filehandle, minlength)
    else:
        return _read_contigs_online(filehandle, minlength)
