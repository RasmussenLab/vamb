"""Vamb - Variational Autoencoders for Metagenomic Binning
Documentation: https://github.com/RasmussenLab/vamb/
"""

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import encode
from . import aamb_encode
from . import semisupervised_encode
from . import hloss_misc
from . import taxvamb_encode
from . import reclustering

from importlib.metadata import version as get_version
from loguru import logger

__version_str__ = get_version("vamb")
logger.remove()

__all__ = [
    "vambtools",
    "parsebam",
    "parsecontigs",
    "cluster",
    "encode",
    "aamb_encode",
    "semisupervised_encode",
    "taxvamb_encode",
    "hloss_misc",
    "reclustering",
]
