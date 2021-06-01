__doc__ = """Estimate RPKM (depths) from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = read_bamfiles(bampaths)
"""

import pycoverm as _pycoverm
import os as _os
import numpy as _np
import vamb.vambtools as _vambtools
from typing import List, Optional

_ncpu = _os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else _ncpu

def read_bamfiles(
    paths: List[str],
    refhash: Optional[bytes],
    minlength: Optional[int],
    minid: float,
    lengths: Optional[_np.ndarray],
    nthreads: int
) -> _np.ndarray:
    "Placeholder docstring - replaced after this func definition"
    # Verify values:
    if (minlength is not None) ^ (lengths is None):
        raise ValueError("Either none or both of minlength and lengths can be None")

    if minid < 0 or minid > 1:
        raise ValueError(f"minid must be between 0 and 1, not {minid}")

    # Coverage cuts off 75 bp in both sides, so if this is much lower,
    # numbers will be thrown wildly off.
    if minlength is not None and minlength < 250:
        raise ValueError(f"minlength must be at least 250, not {minlength}")

    for path in paths:
        if not _pycoverm.is_bam_sorted(path):
            raise ValueError(f"Path {path} is not sorted by reference.")

    headers, coverage = _pycoverm.get_coverages_from_bam(
        paths, threads=nthreads, min_identity=minid,
        trim_upper=0.1, trim_lower=0.1
    )

    # Filter length
    headers = list(headers)
    if lengths is not None:
        if len(lengths) != len(headers):
            raise ValueError("Lengths of headers and lengths does not match.")
        mask = _np.array(list(map(lambda x: x >= minlength, lengths)), dtype=bool)
        headers = [h for (h,m) in zip(headers, mask) if m]
        _vambtools.numpy_inplace_maskarray(coverage, mask)

    # Check refhash
    if refhash is not None:
        _vambtools.verify_refhash(headers, refhash)

    return coverage

read_bamfiles.__doc__ = """Calculates mean coverage for BAM files.

Input:
    paths: List or tuple of paths to BAM files
    refhash: Check all BAM references md5-hash to this (None = no check)
    minlength: Ignore any references shorter than N bases
    lengths: Array of lengths of each contig. (None = no length filtering)
    minid: Discard any reads with nucleotide identity less than this
    nthreads [{}]: Use this number of threads for coverage estimation

Output: A (n_contigs x n_samples) Numpy array with RPKM
""".format(DEFAULT_THREADS)
