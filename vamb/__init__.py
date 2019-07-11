"""Variational Autoencoder for Metagenomic Binning

Vamb does what it says on the tin - bins metagenomes using a variational autoencoder.

Vamb contains the following modules:
vamb.vambtools
vamb.parsecontigs
vamb.parsebam
vamb.encode
vamb.cluster
vamb.benchmark

To get it running, you need:
- A FASTA file of contigs or genes (contigs are better)
- BAM files of reads mapped to the FASTA file
- Python v >= 3.5 with some packages installed:
    pytorch
    numpy
    pysam

General workflow:
1) Filter contigs by size using vamb.vambtools.filtercontigs
2) Map reads to contigs to obtain BAM file
3) Calculate TNF of contigs using vamb.parsecontigs
4) Create RPKM table from BAM files using vamb.parsebam
5) Train autoencoder using vamb.encode
6) Cluster latent representation using vamb.cluster
"""

__authors__ = 'Jakob Nybo Nissen', 'Simon Rasmussen'
__licence__ = 'MIT'
__version__ = (2, 0, 1)

import sys as _sys
if _sys.version_info[:2] < (3, 5):
    raise ImportError('Python version must be >= 3.5')

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import benchmark
from . import encode
