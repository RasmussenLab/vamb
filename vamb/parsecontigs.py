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
import itertools as _itertools

def _reverse_complement(kmer, table=str.maketrans("ACGT", "TGCA")):
    return kmer.translate(table)[::-1]

def _canonical_kmer_freq(kmers, k):
    assert kmers.ndim == 2
    assert kmers.shape[1] == 4**k
    assert k in range(1, 6)
    ncan = [0, 2, 10, 32, 136, 512][k]


    # Create lookup table to reverse complement
    rc_indexof = dict()
    rc_indices = [0] * (4**k)
    for i, _kmer in enumerate(_itertools.product("ACGT", repeat=k)):
        kmer = ''.join(_kmer)
        canonical = min(kmer, _reverse_complement(kmer))

        if canonical in rc_indexof:
            rc_index = rc_indexof[canonical]
        else:
            rc_index = len(rc_indexof)
            rc_indexof[canonical] = rc_index

        rc_indices[i] = rc_index

    result = _np.zeros((len(kmers), ncan), dtype=_np.float32)
    for src, dst in enumerate(rc_indices):
        result[:, dst] += kmers[:, src]

    nkmers = kmers.sum(axis=1).reshape((-1, 1))
    assert _np.all(nkmers > 1)
    result /= nkmers

    return result

def _read_contigs_preallocated(filehandle, minlength, k):
    "Reads file twice wasting time, but does no copying, saving memory."
    n_entries = 0
    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
         if len(entry) >= minlength:
             n_entries += 1


    kmers = _np.empty((n_entries, 4**k), dtype=_np.int32)
    contignames = [None] * n_entries
    lengths = _np.empty(n_entries, dtype=_np.int)

    filehandle.seek(0) # Go back to beginning of file
    entries = _vambtools.byte_iterfasta(filehandle)
    entrynumber = 0
    for entry in entries:
        if len(entry) < minlength:
            continue

        kmers[entrynumber] = entry.kmercounts(k)
        contignames[entrynumber] = entry.header
        lengths[entrynumber] = len(entry)

        entrynumber += 1

    freq = _canonical_kmer_freq(kmers, k)

    return freq, contignames, lengths

def read_contigs(filehandle, k, minlength=100, preallocate=True):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        k: kmer size
        minlength: Ignore any references shorter than N bases [100]
        preallocate: Read contigs twice, saving memory [True]

    Outputs:
        tnfs: An (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    if preallocate:
        return _read_contigs_preallocated(filehandle, minlength, k)
    else:
        return _read_contigs_online(filehandle, minlength)
