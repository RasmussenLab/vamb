"""Vamb - Variational Autoencoders for Metagenomic Binning
Documentation: https://github.com/RasmussenLab/vamb/

Vamb contains the following modules:
vamb.vambtools
vamb.parsecontigs
vamb.parsebam
vamb.encode
vamb.cluster
vamb.benchmark

General workflow:
1) Filter contigs by size using vamb.vambtools.filtercontigs
2) Map reads to contigs to obtain BAM files
3) Calculate a Composition of contigs using vamb.parsecontigs
4) Create Abundance object from BAM files using vamb.parsebam
5) Train autoencoder using vamb.encode
6) Cluster latent representation using vamb.cluster
7) Split bins using vamb.vambtools
"""

__version__ = (4, 1, 0)

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import benchmark
from . import encode
from . import aamb_encode

__all__ = [
    "vambtools",
    "parsebam",
    "parsecontigs",
    "cluster",
    "benchmark",
    "encode",
    "aamb_encode",
]
