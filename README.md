# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent representation. It performs excellently with many samples, well with 5-10 samples and poorly relying only on the nucleotide composition. Vamb is implemented purely in Python (with a little bit of Cython) and can be used both from commandline and from within a Python interpreter.

For more information on the background, context, and theory of Vamb, read [our paper on bioRxiv](https://www.biorxiv.org/content/early/2018/12/19/490078) (DOI: 10.1101/490078)

For more information about the implementation, methodological considerations, future directions, and advanced Python usage of Vamb, see the tutorial file (`doc/tutorial.html`)

# Installation
Vamb is most easily installed with pip - make sure your pip version is up to date, as it won't work with ancient versions (v. <= 9).

### Installation for casual users:

```
pip install https://github.com/jakobnissen/vamb/archive/v1.1.0.zip
```

### Installation for advanced users:

This is for developers who want to be able to edit Python files and have the changes show up directly in the running command:

```
# clone the desired branch from the repository, here master
git clone https://github.com/jakobnissen/vamb -b master

# dev install
cd vamb
pip install -e .
```

### Installing by compiling the Cython yourself

If you can't/don't want to use pip, you can do it the hard way: Get the most recent versions of the Python packages `cython`, `numpy`, `torch` and `pysam`. Compile `src/_vambtools.pyx` using Cython and move the resulting executable to the inner vamb directory. You can then run Vamb by invoking the Python interpreter as shown below.

# Running

After installation, you can run by passing the package to the Python interpreter:

```
python3 -m vamb
```

Which is useful if you want to specify which Python executable to use. If you've installed with pip, you can start Vamb by simply typing:

```
vamb
```

# Inputs and outputs

### Inputs

Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values.

* TNF is calculated from a regular fasta file of DNA sequences.
* Depth is calculated from BAM-files of mapping reads to that same fasta file.

The observed values for both of these measures become uncertain when the sequences is too short due to the law of large numbers. Therefore, Vamb works poorly on short sequences and on data with low depth. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced and thus can support hundreds of samples, but the results of working on contigs is better.

With fewer samples (up to 100), we recommend using contigs from an assembly with a minimum contig length cutoff of ~2000-ish basepairs. With many samples, the number of contigs can become overwhelming. The better approach is to split the dataset up into smaller chuncks and bin them independently.

There are situations where you can't just filter the fasta file, maybe because you have already spent tonnes of time getting those BAM files and you're not going to remap if your life depended on it, or because your fasta file contains genes and so removing all entries less than e.g. 2000 bps is a bit too much to ask. In those situations, you can still pass the argument `minlength` (`-m` on command line) if you want to have Vamb ignore the smaller contigs. This is not ideal, since the smaller, contigs will still have recruited some reads during mapping which are then not mapped to the larger contigs, but it can work alright.

### Outputs

Vamb produces the following output files:

- `log.txt` - a text file with information about the Vamb run.
- `tnf.npz`, `rpkm.npz`, `mask.npz` and `latent.npz` - Numpy .npz files with TNFs, abundances, which sequences were successfully encoded, and the latent encoding of the sequences.
- `model.pt` - containing a PyTorch model object of the trained VAE. You can load the VAE from this file using `vamb.VAE.load` from Python.
- `clusters.tsv` - a two-column text file with one row per sequence: Left column for the cluster name, right column for the sequence name. You can create the FASTA-file bins themselves using `vamb.vambtools.write_bins` (see `doc/tutorial.html` for more details).

### Recommended preparation

__1) Preprocess the reads and check their quality__

We use AdapterRemoval combined with FastQC for this.

__2) Assemble each sample individually OR co-assemble and get the contigs out__

We recommend using metaSPAdes on each sample individually.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique__

We recommend prepending the sample name to each contig header from that sample.

__4) Remove all small contigs from the FASTA file__

There's a tradeoff here between a too low cutoff, retaining hard-to-bin contigs which adversely affects the binning of all contigs, and throwing out good data. We use a length cutoff of 2000 bp as default but haven't actually run tests for the optimal value.

__5) Map the reads to the FASTA file to obtain BAM files__

We have used BWA MEM for mapping, fully aware that it is not well suited for metagenomic mapping. In theory, any mapper that produces a BAM file.

# Troubleshooting

__Installation: Compilation fails on Arch Linux in a Conda environment__

This is a known issue affecting multiple (all?) Cython-dependent packages at the moment of writing (2019-01-25). There is no easy fix to make the pip installation work, but you can compile the Cython file manually. [Link to pacman issue](https://bbs.archlinux.org/viewtopic.php?id=242682).

### Parsing the FASTA file

No trouble seen so far.

### Parsing the BAM files

No trouble seen so far.

### Autoencoding

No trouble seen so far.

### Clustering

__It warns: Only [N]% of contigs has well-separated threshold__

See the issue below

__It warns: Only [N]% of contigs has *any* observable threshold__

This happens if the inter-contig distances do not neatly separate into close and far contigs, which can happen if:

* There is little data, e.g. < 50k contigs
* The VAE has not trained sufficiently
* There is, for some reason, little signal in your data (low read abundances, similar bacterial strains etc.)

Make sure the assembly and mapping went well. If it did, there is not much to do about it. We do not know at what percentage of well-separated or observable threshold Vamb becomes unusable. I would still use Vamb unless the following error is thrown:

__It warns: Too little data: [ERROR]. Setting threshold to 0.08__

This happens if less than 5 of the Pearson samples (default of 2500 samples) return any threshold. If this happens for a reasonable number of Pearson samples, e.g. > 100, I would not trust Vamb to deliver good results, and would use another binner instead.
