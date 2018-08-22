# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Please contact jakni@bioinformatics.dtu.dk for bug fixes and feature requests.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent 
representation. It performs excellently with many samples, well with 5-10 samples and poorly relying only on the nucleotide composition. Vamb is implemented purely in Python (with a little bit of Cython) 
and can be used both from commandline and from within a Python interpreter.

# Installation

Vamb requires the following Python packages to run:

* PyTorch
* Numpy
* pysam

So make sure you have those installed. When you have that:

__If you have `git` installed__:

    [jakni@nissen:scripts]$ git clone https://github.com/jakobnissen/vamb vamb
    
__If you don't__

You then presumably have access to the vamb directory with this notebook, so just put it wherever:

    [jakni@nissen:scripts]$ cp -r /path/to/vamb/directory vamb

# Quickstart

Take a brief look on the options with:

    [jakni@nissen:scripts]$ python vamb/run.py --help

Do the defaults look alright? They probably do, but you might want to check number of processes to launch, GPU acceleration and whether you want the faster `tandemclustering` option enabled.

Then just do:

    [jakni@nissen:scripts]$ python vamb/runvamb.py outdir contigs.fna path/to/bamfiles/*.bam


# Advanced usage

If you want to extend Vamb, run only a part of the pipeline or run with altered inputs, you'll need to import it as Python module and run it from within the Python interpreter. See the tutorials in the `doc` directory.

# Troubleshooting

### Parsing the fasta file
__`parsecontigs.read_contigs` fails with ValueError: Non-ACGTN in line [LINE]: [CHAR]__

Vamb can only parse contigs consisting of any bytes in b'acgtnACGTN', so IUPAC ambigious DNA, uracils (U), and anything else will cause an error. You will have to clean the FASTA file and change all the non-ACTGN bases to N. For convenience, you can use the `vamb.vambtools.maskbases` function:

    with vamb.vambtools.Reader('/path/to/contigs.fna', 'rb') as filehandle:
        masked_lines = vamb.vambtools.maskbases(filehandle)
        tnfs, contignames, lengths = vamb.parsecontigs.read_contigs(masked_lines)

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
* There is, for some reason, little signal in your data

There is not much to do about it. We do not know at what percentage of well-separated or observable threshold Vamb becomes unusable. I would still use Vamb unless the following error is thrown:

__It warns: Too little data: [ERROR]. Setting threshold to 0.08__

This happens if less than 5 of the samples (default of 1000 samples) return any threshold. If this happens for a reasonable number of samples, e.g. > 100, I would not trust Vamb to deliver good results, 
and would use another binner instead.


# Prerequisites

Like other metagenomic binners, Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values.

* TNF is calculated from a regular fasta file of DNA sequences.
* Depth is calculated from BAM-files of mapping reads to that same fasta file.

The observed values for both of these measures become uncertain when the sequences is too short due to the law of large numbers. Therefore, Vamb works poorly on short sequences and on data with low depth. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced and thus can support hundreds of samples. 

With fewer samples (up to 100), we recommend using contigs from an assembly with a minimum contig length cutoff of ~2000-ish basepairs. With many samples, the number of contigs can become overwhelming. 
The better approach is to split the dataset up into smaller chuncks and bin them independently.

There are situations where you can't just filter the fasta file, maybe because you have already spent tonnes of time getting those BAM files and you're not going to remap if your life depended on it, or because your fasta file contains genes and so removing all entries less than e.g. 2000 bps is a bit too much to ask.

In those situations, you can still pass the argument `minlength` if you want to have Vamb ignore the smaller contigs. This is not ideal, since the smaller, contigs will still have recruited some reads during mapping which are then not mapped to the larger contigs, but it can work alright.

### Recommended preparation

__1) Preprocess the reads and check their quality__

We recommend AdapterRemoval combined with FastQC for this.

__2) Assemble each sample individually OR co-assemble and get the contigs out__

We recommend using metaSPAdes on each sample individually.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique__

We recommend prepending the sample name to each contig header from that sample.

__4) Remove all small contigs from the FASTA file__

There's a tradeoff here between a too low cutoff, retaining hard-to-bin contigs which adversely affects the binning of all contigs, and throwing out good data. We use a length cutoff of ~2000 bp but 
haven't tested for the optimal value.

__5) Map the reads to the FASTA file to obtain BAM files__

We have used BWA MEM for mapping, fully aware that it is not well suited for metagenomic mapping. In theory, any mapper that produces a BAM file with an alignment score tagged `AS:i` and multiple 
secondary hits tagged `XA:Z` can work.
