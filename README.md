# Vamb
[![Read the Docs](https://readthedocs.org/projects/vamb/badge/?version=latest)](https://vamb.readthedocs.io/en/latest/)

Vamb is a metagenomic binner which feeds sequence composition from a FASTA file of contigs, and abundance information from e.g. BAM files into a variational autoencoder and clusters the latent representation.
It performs excellently with multiple samples, and pretty good on single-sample data.

* __New:__ _For benchmarking binnings with a known ground truth, see our tool [BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl)._
* __New:__ _The new semi-supervised [TaxVamb binning mode](https://www.biorxiv.org/content/10.1101/2024.10.25.620172v1) achieves state-of-the-art binning._

## Programs in Vamb
The Vamb package contains several programs, including three binners:
* __Vamb__: The original binner based on variational autoencoders.
  This has been upgraded significantly since its original release.
  [Link to article](https://doi.org/10.1038/s41587-020-00777-4).
* __Avamb__: An ensemble model based on Vamb and adversarial autoencoders. 
  Avamb produces better bins than Vamb, but is a more complex and computationally demanding pipeline.
  See the [Avamb README page](https://github.com/RasmussenLab/avamb/tree/avamb_new/ workflow_avamb) for more information.
  [Link to article](https://doi.org/10.1038/s42003-023-05452-3).
* __TaxVamb__: A semi-supervised binner that uses taxonomy information from e.g. `mmseqs taxonomy`.
  TaxVamb produces superior bins, but requires you have run a taxonomic annotation workflow.
  [Link to article](https://doi.org/10.1101/2024.10.25.620172).

And a taxonomy predictor:
* __Taxometer__: This tool refines arbitrary taxonomy predictions (e.g. from `mmseqs taxonomy`) using kmer composition and co-abundance.
  [Link to article](https://www.nature.com/articles/s41467-024-52771-y)

See also [our tool BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl) for evaluating metagenomic bins when a ground truth is available,
e.g. for simulated data or a mock microbiome.

## Installation
Vamb is in continuous development. Make sure to install the released latest version for the best results.

### Recommended: Install with `pip`:
Recommended: Vamb can be installed with `pip` (thanks to contribution from C. Titus Brown):
```shell
pip install vamb
```

Note: An active Conda environment can hijack your system's linker, causing an error during installation.
If you see issues, either deactivate `conda`, or delete the `~/miniconda/compiler_compats` directory before installing with pip.

### Installation for advanced users:
If you want to install the latest version from GitHub, or you want to change Vamb's source code, you should install it like this:

```shell
# Clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b master
cd vamb
# The `-e` flag will make Vamb change if the source code is changed after install
pip install -e .
```

__Note that the master branch is work-in-progress, have not been thoroughly tested, and is expected to have more bugs.__

### Installing by compiling the Cython yourself

If you can't/don't want to use pip/Conda, you can do it the hard way: Install the dependencies listed in the `pyproject.toml` file. Compile `src/_vambtools.pyx` then move the resulting binary to the inner of the two `vamb` directories. Check if it works by importing `vamb` in a Python session.

## Running Vamb
First, figure out what program you want to run:
* If you want to bin, and are able to get taxonomic information, run `vamb bin taxvamb`
* Otherwise, if you want a good and simple binner, run `vamb bin default`
* Otherwise, if you want to bin, and don't mind a more complex, but performant workflow run the [Avamb Snakemake workflow](https://github.com/RasmussenLab/avamb/tree/avamb_new/workflow_avamb).
* If you want to refine existing taxonomic classification, run `vamb taxometer`

For more command-line options, see the command-line help menu:
```shell
vamb -h
```

For details about how to run Vamb, see [the documentation](https://vamb.readthedocs.io/en/latest/)
