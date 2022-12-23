# Avamb

Created by Pau Piera, Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark and Novo Nordisk Foundation Center for Protein Research, University of Copenhagen.

Avamb is a metagenomics binning pipeline composed by aamb, an adversarial autoencoder for metagenomics binning, and vamb, a variational autoencoder for metagenomics binning, as well as a bin dereplication and quality filtering step that runs CheckM2. In the same way than vamb, avamb leverages sequence composition information from a contig catalogue and co-abundance information from BAM files by projecting those features into aamb and vamb latent spaces, where the metagenomics binning actually takes place. It performs excellently with multiple samples, and pretty good on single-sample data. Avamb is implemented in Python (with a little bit of Cython) and Snakemake.

Avamb pre-print can be found here [biorxiv](<insert biorXv link>)

# Installation
We expect avamb installation with pip and conda in the future. At the moment, Avamb can only be installed by clonning the repository:

```
# clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b v4 # might have to be updated
cd vamb
pip install -e .
```

# Run avamb
Avamb complete pipeline is implemented in Snakemake, which can be found and detailed [here](<insert avamb_workflow here>).
