__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import sys as _sys
import os as _os
import numpy as _np
import vamb.vambtools as _vambtools

# This kernel is created in src/create_kernel.py. See that file for explanation
_KERNEL = _vambtools.read_npz(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "kernel.npz"))

def _project(fourmers, kernel=_KERNEL):
    "Project fourmers down in dimensionality"
    s = fourmers.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    fourmers *= 1/s
    fourmers += -(1/256)
    return _np.dot(fourmers, kernel)

def _convert(raw, projected):
    "Move data from raw PushArray to projected PushArray, converting it."
    raw_mat = raw.take().reshape(-1, 256)
    projected_mat = _project(raw_mat)
    projected.extend(projected_mat.ravel())
    raw.clear()

def read_contigs(filehandle, minlength=100):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]

    Outputs:
        tnfs: An (n_FASTA_entries x 103) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    raw = _vambtools.PushArray(_np.float32)
    projected = _vambtools.PushArray(_np.float32)
    lengths = _vambtools.PushArray(_np.int)
    contignames = list()

    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        raw.extend(entry.kmercounts(4))

        if len(raw) > 256000:
            _convert(raw, projected)

        lengths.append(len(entry))
        contignames.append(entry.header)

    # Convert rest of contigs
    _convert(raw, projected)
    tnfs_arr = projected.take()

    # Don't use reshape since it creates a new array object with shared memory
    tnfs_arr.shape = (len(tnfs_arr)//103, 103)
    lengths_arr = lengths.take()

    return tnfs_arr, contignames, lengths_arr
