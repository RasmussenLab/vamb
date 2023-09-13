"""Vamb - Variational Autoencoders for Metagenomic Binning
Documentation: https://github.com/RasmussenLab/vamb/
"""

__version__ = (4, 1, 3)

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import encode
from . import aamb_encode
from . import semisupervised_encode
from . import h_loss
from . import hier
from . import hlosses_fast
from . import infer
from . import reclustering

__all__ = [
    "vambtools",
    "parsebam",
    "parsecontigs",
    "cluster",
    "encode",
    "aamb_encode",
    "semisupervised_encode",
    "h_loss",
    "hier",
    "hlosses_fast",
    "infer",
    "reclustering",
]
