*__IMPORTANT__: A critical bug in clustering [was fixed January 24th 2019](https://github.com/jakobnissen/vamb/commit/f7f9ceb788086a6a80a7ffe0c6209658126902a9) - if you have run Vamb at any previous time, your results will likely have been terrible.*

# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent representation. It performs excellently with > 5 samples and poorly relying only on the nucleotide composition. Vamb is implemented purely in Python (with a little bit of Cython) and can be used both from command line and from within a Python interpreter.

For more information on the background, context, and theory of Vamb, read [our paper on bioRxiv](https://www.biorxiv.org/content/early/2018/12/19/490078) (DOI: 10.1101/490078)

For more information about the implementation, methodological considerations, and advanced Python usage of Vamb, see the tutorial file (`doc/tutorial.html`)

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

If you can't/don't want to use pip, you can do it the hard way: Get the most recent versions of the Python packages `cython`, `numpy`, `torch` and `pysam`. Compile `src/_vambtools.pyx` using Cython and move the resulting executable to the inner of the two `vamb` directories.

# Running

## Invoking Vamb

After installation with pip, Vamb will show up in your PATH variable, and you can simply run:

```
vamb
```

To run Vamb with another Python executable (say, if you want to run with `python3.7`) than the default, you can run:

```
python3.7 -m vamb
```

You can also run the inner `vamb` directory as a script. This will work even if you did not install with pip:

```
python my_scripts/vamb/vamb
```

## Example command to run Vamb

Vamb with default inputs (a FASTA file and some BAM files) can be executed like so:

```
vamb --outdir vambout --fasta /path/to/file.fasta --bamfiles /path/to/bamfiles/*.bam
```

For a detailed explanation of the parameters of Vamb, or different inputs, see the tutorial in the `doc` directory.

# Inputs and outputs

### Inputs

Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values.

* TNF is calculated from a regular FASTA file of DNA sequences.
* Depth is calculated from BAM-files of mapping reads to that same FASTA file.

Remember that the quality of Vamb's bins are no better than the quality of the input files. If your BAM file is constructed carelessly, for example by allowing reads from distinct species to crossmap, the BAM file will not contain information with which Vamb can separate the crossmapping species. In general, you want reads to map only to contigs within the same phylogenetic distance that you want Vamb to bin together.

The observed values for both TNF and RPKM are statistically uncertain when the sequences are too short. Therefore, Vamb works poorly on short sequences and on data with low depth. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced and thus can support hundreds of samples, but the results of working on (larger) contigs are better.

With fewer samples (up to about 100), we recommend using contigs from an assembly with a minimum contig length cutoff of ~2000-ish basepairs. With many samples, the number of contigs can become overwhelming. The better approach is to split the dataset up into groups of similar samples and bin them independently.

### Outputs

Vamb produces the following output files:

- `log.txt` - a text file with information about the Vamb run. Look here (and at stderr) if you experience errors.
- `tnf.npz`, `rpkm.npz`, `mask.npz` and `latent.npz` - Numpy .npz files with TNFs, abundances, which sequences were successfully encoded, and the latent encoding of the sequences.
- `model.pt` - containing a PyTorch model object of the trained VAE. You can load the VAE from this file using `vamb.VAE.load` from Python.
- `clusters.tsv` - a two-column text file with one row per sequence: Left column for the cluster (i.e bin) name, right column for the sequence name. You can create the FASTA-file bins themselves using `vamb.vambtools.write_bins` (see `doc/tutorial.html` for more details).

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

We have used BWA MEM for mapping, fully aware that it is not well suited for metagenomic mapping. Be careful to choose proper parameters for your aligner - in general, if a read from contig A will map to contig B, then Vamb will bin A and B together. So your aligner should map reads with the same level of discrimination that you want Vamb to use. We run the following command to create our BAM files:

`bwa mem filtered_contigs.fna sample1.forward.fastq.gz sample1.reverse.fastq.gz -t 8 | samtools view -F 3584 -b --threads 8 > sample.bam`
