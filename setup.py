import sys
from setuptools import setup, find_packages
from setuptools import Extension
import os

SETUP_METADATA = \
               {
    "name": "vamb",
    "description": "Variational autoencoder for metagenomic binning",
    "url": "https://github.com/jakobnissen/vamb",
    "author": "Jakob Nybo Nissen and Simon Rasmussen",
    "author_email": "jakni@dtu.dk",
    "version": "1.1",
    "license": "MIT",
    "packages": find_packages(),
    "entry_points": {'console_scripts': [
        'vamb = vamb.__main__:main'
        ]
    },
    "ext_modules": [Extension("vamb._vambtools",
                               sources=["src/_vambtools.pyx"],
                               language="c++")],
    "install_requires": ["numpy>=1.15", "torch>=0.4", "pysam>=0.14"],
    "setup_requires": ['Cython>=0.25.2', "setuptools>=38.6.0"],
    "python_requires":">=3.5",
    }

setup(**SETUP_METADATA)
