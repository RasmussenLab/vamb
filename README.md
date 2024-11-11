# Vamb
[![Read the Docs](https://readthedocs.org/projects/vamb/badge/?version=latest)](https://vamb.readthedocs.io/en/latest/)

Vamb is a metagenomic binner which feeds sequence composition from a FASTA file of contigs, and abundance information from e.g. BAM files into a variational autoencoder and clusters the latent representation.
It performs excellently with multiple samples, and pretty good on single-sample data.

* __New:__ _For benchmarking binnings with a known ground truth, see our tool [BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl)_
* __New:__ _The new semi-supervised [TaxVamb binning mode](TODO) achieves state-of-the-art binning_

## Programs in Vamb
The Vamb package contains several programs, including three binners:
* __Vamb__: The original binner based on variational autoencoders. [Article](https://doi.org/10.1038/s41587-020-00777-4).
  This has been upgraded significantly since its original release.
* __Avamb__: An ensemble model based on Vamb and adversarial autoencoders. [Article](https://doi.org/10.1038/s42003-023-05452-3).
  Avamb produces better bins than Vamb, but is a more complex and computationally demanding pipeline.
  See the [Avamb README page](https://github.com/RasmussenLab/avamb/tree/avamb_new/workflow_avamb) for more information.
* __TaxVamb__: A semi-supervised binner that uses taxonomy information from e.g. `mmseqs taxonomy`. [Article](https://doi.org/10.1101/2024.10.25.620172).
  TaxVamb produces superior bins, but requires you have run a taxonomic annotation workflow.

And a taxonomy predictor:
* __Taxometer__: This tool refines arbitrary taxonomy predictions (e.g. from `mmseqs taxonomy`) using kmer composition and co-abundance. Go to the [release branch](https://github.com/RasmussenLab/vamb/blob/taxometer_release/README_Taxometer.md) for the instructions. [Article](https://www.nature.com/articles/s41467-024-52771-y)

See also [our tool BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl) for evaluating metagenomic bins when a ground truth is available,
e.g. for simulated data or a mock microbiome.

# Installation
Vamb is in continuous development. Make sure to install the latest version for the best results.

### Installation for casual users:
Recommended: Vamb can be installed with pip (thanks to contribution from C. Titus Brown):
```
pip install vamb
```

Note: An active Conda environment can hijack your system's linker, causing an error during installation. Either deactivate `conda`, or delete the `~/miniconda/compiler_compats` directory before installing with pip.

### Installation for advanced users:
If you want to install the latest version from GitHub, or you want to change Vamb's source code, you should install it like this:

```
# clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b master
cd vamb
pip install -e .
```

__Note that the master branch is work-in-progress and is expected to have more bugs__

### Installing by compiling the Cython yourself

If you can't/don't want to use pip/Conda, you can do it the hard way: Install the dependencies listed in the `pyproject.toml` file. Compile `src/_vambtools.pyx` then move the resulting binary to the inner of the two `vamb` directories. Check if it works by importing `vamb` in a Python session.

# Running Vamb
First, figure out what program you want to run:
* If you want to bin, and are able to get taxonomic information, run `vamb bin taxvamb`
* Otherwise, if you want a good and simple binner, run `vamb bin default`
* If you want to bin, and don't mind a more complex, but performant workflow run the [Avamb Snakemake workflow](https://github.com/RasmussenLab/avamb/tree/avamb_new/workflow_avamb)
* If you want to refine existing taxonomic classification, run `vamb taxometer`

For more command-line options, see the command-line help menu:
```
vamb -h
```

For details about how to run Vamb, see [the documentation](https://vamb.readthedocs.io/en/latest/)
