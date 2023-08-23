"""Vamb - Variational Autoencoders for Metagenomic Binning
Documentation: https://github.com/RasmussenLab/vamb/
"""

__version__ = (4, 1, 3)

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import parsemarkers
from . import cluster
from . import encode
from . import aamb_encode

__all__ = [
    "vambtools",
    "parsebam",
    "parsecontigs",
    "parsemarkers",
    "cluster",
    "encode",
    "aamb_encode",
]
