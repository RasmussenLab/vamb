import sys
from setuptools import setup, find_packages
from setuptools import Extension
import os

SETUP_METADATA = \
               {
    "name": "vamb",
    "description": "Variational autoencoder for metagenomic binning",
    "url": "https://github.com/RasmussenLab/vamb",
    "author": "Jakob Nybo Nissen and Simon Rasmussen",
    "author_email": "jakobnybonissen@gmail.com",
    "version": "4.0.0",
    "license": "MIT",
    "packages": find_packages(),
    "package_data": {"vamb": ["kernel.npz"]},
    "entry_points": {'console_scripts': [
        'vamb = vamb.__main__:main'
        ]
    },
    "scripts": ['src/concatenate.py'],
    "ext_modules": [Extension("vamb._vambtools",
                               sources=["src/_vambtools.pyx"],
                               language="c")],
    "install_requires": ["numpy~=1.20", "torch~=1.8", "pycoverm~=0.2"],
    "setup_requires": ['Cython>=0.29.0', "setuptools~=56.0.0"],
    "python_requires": "~=3.6",
    "classifiers":[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    }

setup(**SETUP_METADATA)
