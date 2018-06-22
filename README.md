# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Please contact jakni@bioinformatics.dtu.dk for bug fixes and feature requests.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent 
representation. It performs excellently with many samples, well with 5-10 samples and poorly relying only on the nucleotide composition. Vamb is implemented almost purely in Python and can be used both 
from commandline and from within a Python interpreter.

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

    [jakni@nissen:scripts]$ python vamb/runvamb.py --help

Do the defaults look alright? They probably do, but you might want to check number of processes to launch, GPU acceleration and whether you want the faster `tandemclustering` option enabled.

Then just do:

    [jakni@nissen:scripts]$ python vamb/runvamb.py outdir contigs.fna path/to/bamfiles/*.bam

# Running from command line

You can run either the entire pipeline from commandline, or each module independently.

---

__For the entire pipeline__, you need to use the `runvamb.py` script:

    [jakni@nissen:~]$ python Documents/scripts/vamb/runvamb.py --help
    usage: python runvamb.py OUTPATH FASTA BAMPATHS [OPTIONS ...]

    Run the Vamb pipeline.

    Creates a new direcotry and runs each module of the Vamb pipeline in the
    new directory. Does not yet support resuming stopped runs - in order to do so,
    
    [ lines elided ]

You use it like this:

    [jakni@nissen:~] python path/to/vamb/runvamb.py output_directory contig.fna path/to/bamfiles/*.bam
    
__For each module__, you find the relevant script:

    [jakni@nissen:~]$ python Documents/scripts/vamb/parsecontigs.py --help
    usage: parsecontigs.py contigs.fna(.gz) tnfout lengthsout

    Calculate z-normalized tetranucleotide frequency from a FASTA file.
    
    [ lines elided ]

# Walkthrough from within Python interpreter

See the notebooks in the `doc` directory for an in-depth walkthrough of the Vamb package.

# Troubleshooting

See the Markdown file in the `doc` directory.

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
