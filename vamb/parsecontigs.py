__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import sys as _sys
import numpy as _np
import vamb.vambtools as _vambtools

class PushArray:
    """Data structure that allows efficient appending and extending a Numpy array"""
    __slots__ = ['data', 'capacity', 'length']
    def __init__(self, dtype, start_capacity=1<<16):
        self.capacity = start_capacity
        self.data = _np.empty(self.capacity, dtype=dtype)
        self.length = 0

    def grow(self):
        "Grow capacity by power of two between 1/8 and 1/4 of current capacity"
        toadd = max(32, int(self.capacity * 0.125))
        newcap = self.capacity + (1 << toadd.bit_length())
        self.capacity = newcap
        self.data.resize(self.capacity, refcheck=False)

    def append(self, value):
        if self.length == self.capacity:
            self.grow()

        self.data[self.length] = value
        self.length += 1

    def extend(self, values):
        if self.length + len(values) > self.capacity:
            self.grow()

        self.data[self.length:self.length+len(values)] = values
        self.length += len(values)

    def take(self):
        self.data.resize(self.length)
        self.capacity = self.length
        return self.data

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

    tnfs = PushArray(_np.float32)
    lengths = PushArray(_np.int)
    contignames = list()

    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs.extend(entry.fourmer_freq())
        lengths.append(len(entry))
        contignames.append(entry.header)

    lengths_arr = lengths.take()
    tnfs_arr = tnfs.take().reshape(-1, 136)

    return tnfs_arr, contignames, lengths_arr
