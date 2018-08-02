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

Do the defaults look alright? They probably do, but you might want to check number of subprocesses to launch, GPU acceleration and whether you want the faster but less accurate `tandemclustering` option enabled.

Then just do:

    [jakni@nissen:scripts]$ python vamb/runvamb.py outdir contigs.fna path/to/bamfiles/*.bam

# Prerequisites

Like other metagenomic binners, Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values.

* TNF is calculated from a regular fasta file of DNA sequences.
* Depth is calculated from BAM-files of mapping reads to that same fasta file.

The observed values for both of these measures become uncertain when the sequences is too short due to the law of large numbers. Therefore, Vamb works poorly on short sequences. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced and thus can support hundreds of samples. 

With fewer samples (up to 100), we recommend using contigs from an assembly with a minimum contig length cutoff of ~2000-ish basepairs. With many samples, the number of contigs become overwhelming. The better approach is to split the dataset up into smaller chuncks and bin them independently.

There are situations where you can't just filter the fasta file, maybe because you have already spent tonnes of time getting those BAM files and you're not going to remap if your life depended on it, or because your fasta file contains genes and so removing all entries less than e.g. 2000 bps is a bit too much to ask. In those situations, you can still pass the argument `minlength` if you want to have Vamb ignore the smaller contigs. This is not ideal, since the smaller, contigs will still have recruited some reads during mapping which are then not mapped to the larger contigs, but it can work alright.

### Recommended preparation

__1) Preprocess the reads and check their quality__

We recommend AdapterRemoval combined with FastQC for this.

__2) Assemble each sample individually OR co-assemble and get the contigs out__

We recommend using metaSPAdes on each sample individually.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique__

We recommend prepending the sample name to each contig header from that sample.

__4) Remove all small contigs from the FASTA file__

There's a tradeoff here between a too low cutoff, retaining hard-to-bin contigs which adversely affects the binning of all contigs, and throwing out good data. We recommend choosing a length cutoff of ~2000 bp.

__5) Map the reads to the FASTA file to obtain BAM files__

We have used BWA MEM for mapping, fully aware that it is not suited for this task. In theory, any mapper that produces a BAM file with an alignment score tagged `AS:i` and multiple secondary hits tagged `XA:Z` can work.

# Advanced usage

If you want to extend Vamb, run only a part of the pipeline, run with altered inputs, you'll need to import it as Python module and run it from within the Python interpreter. See the tutorials in the `doc` directory.