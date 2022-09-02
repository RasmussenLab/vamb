# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark and Novo Nordisk Foundation Center for Protein Research, University of Copenhagen.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent representation. It performs excellently with multiple samples, and pretty good on single-sample data. Vamb is implemented purely in Python (with a little bit of Cython) and can be used both from command line and from within a Python interpreter.

:star: Vamb is benchmarked in CAMI2. Read our interpretations of the results [here.](https://github.com/RasmussenLab/vamb/blob/master/doc/CAMI2.md) :star:

Vamb has changed a lot since the pre-print at [biorxiv](https://www.biorxiv.org/content/early/2018/12/19/490078) so we really recommend reading it at [Nature Biotechnology](https://doi.org/10.1038/s41587-020-00777-4). For instance, the current version of Vamb uses "multi-split binning" which is not mentioned in the pre-print. Furthermore, we in the pre-print mention that Vamb can bin gene catalogues - this did not make it in the Nature Biotech paper, but it can still do that. For instance, on the Integrated Gene Catalogue Vamb generated 675 near complete gene-bins compared to 558 and 429 from MSPminer and Canopy, respectively.

For more information about the implementation, methodological considerations, and advanced usage of Vamb, see the tutorial file (`doc/tutorial.html`)

**If you are interested in phages you can use Phamb after running Vamb to identify them! [Paper](https://www.nature.com/articles/s41467-022-28581-5) and [GitHub](https://github.com/RasmussenLab/phamb).**

# Installation
Vamb is most easily installed with pip - make sure your pip version is up to date, as it won't work with ancient versions (v. <= 9).

### Installation for casual users:

Recommended: Vamb can be installed with pip (thanks to contribution from C. Titus Brown):
```
pip install https://github.com/RasmussenLab/vamb/archive/v3.0.8.zip
```

or using [Bioconda's package](https://anaconda.org/bioconda/vamb) (thanks to contribution from Antônio Pedro Camargo).
(note that the BioConda package does not include GPU support).
 
```
conda install -c pytorch pytorch torchvision cudatoolkit=10.2
conda install -c bioconda vamb
```

### Installation for advanced users:

If you want to install the latest version from GitHub you can clone and install it using:

```
# clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b master
cd vamb
pip install -e .
```

### Installing by compiling the Cython yourself

If you can't/don't want to use pip/Conda, you can do it the hard way: Get the most recent versions of the Python packages `cython`, `numpy`, `torch` and `pysam`. Compile `src/_vambtools.pyx`, (see `src/build_vambtools.py`) then move the resulting binary to the inner of the two `vamb` directories. Check if it works by importing `vamb` in a Python session.

### Windows

Vamb does currently not compile on Windows because it is dependent on `pysam`, but a Windows-friendly version is hopefully on the way (see branch `v4`). Let us know if you want to run Vamb on Windows.

# Running

For a detailed explanation of the parameters of Vamb, or different inputs, see the tutorial in the `doc` directory. 

**Updated in 3.0.2: for a snakemake pipeline see `workflow` directory.**

For more command-line options, see the command-line help menu:
```
vamb -h
```

## TL;DR: Here's how to run Vamb

For this example, let us suppose you have a directory of short (e.g. Illumina) reads in a
directory `/path/to/reads`, and that _you have already quality controlled them_.

1. Run your favorite metagenomic assembler on each sample individually:

```
spades.py --meta /path/to/reads/sample1.fw.fq.gz /path/to/reads/sample1.rv.fq.gz
-k 21,29,39,59,79,99 -t 24 -m 100gb -o /path/to/assemblies/sample1
```

2. Use Vamb's `concatenate.py` to make the FASTA catalogue of all your assemblies:

```
concatenate.py /path/to/catalogue.fna.gz /path/to/assemblies/sample1/contigs.fasta
/path/to/assemblies/sample2/contigs.fasta  [ ... ]
```

3. Use your favorite short-read aligner to map each your read files back to the resulting FASTA file:

```
minimap2 -d catalogue.mmi /path/to/catalogue.fna.gz; # make index
minimap2 -t 8 -N 5 -ax sr catalogue.mmi --split-prefix mmsplit /path/to/reads/sample1.fw.fq.gz /path/to/reads/sample1.rv.fq.gz | samtools view -F 3584 -b --threads 8 > /path/to/bam/sample1.bam
```

4. Run Vamb:

```
vamb --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C --minfasta 200000
```

Note that we have found that MetaBAT2's `jgi_summarize_bam_contig_depths` program estimates BAM depths more accurate than Vamb's `parsebam` module (see below). If you want to use this approach instead we provide an easy to use `snakemake` workflow which will do this for you.

5. Apply the relevant postprocessing filters

Vamb outputs every input contig, even those not binned with other contigs. Depending on your use case, the large number of small bins may not be relevant. Make sure to filter the contigs either using a binning quality control tool, by size, or by aligning the resulting bins to references.

## Snakemake workflow

To make it even easier to run Vamb in the best possible way, we have created a [Snakemake](https://snakemake.readthedocs.io/en/stable/#) workflow that will run steps 2-4 above using MetaBAT2's `jgi_summarize_bam_contig_depths` program for improved counting. Additionally it will run [CheckM](https://ecogenomics.github.io/CheckM/) to estimate completeness and contamination of the resulting bins. It can run both on a local machine, a workstation and a HPC system using `qsub` - it is included in the `workflow` folder.

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

# Inputs and outputs

### Inputs
*Also see the section: Recommended workflow*

Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values:

* TNF is calculated from a regular FASTA file of DNA sequences.
* Depth is calculated from BAM-files of a set of reads from each sample mapped to that same FASTA file.

:warning: *Important:* Vamb can use information from multi-mapping reads, but all alignments of a single read __must__ be consecutive in the BAM files. See section 4 of *Recommended workflow*.

Remember that the quality of Vamb's bins are no better than the quality of the input files. If your BAM files are constructed carelessly, for example by allowing reads from distinct species to crossmap indiscriminately, your BAM files will not contain information with which Vamb can separate those species. In general, you want reads to map only to contigs within the same phylogenetic distance that you want Vamb to bin together.

Estimation of TNF and RPKM is subject to statistical uncertainty. Therefore, Vamb works less well on short sequences and on data with low depth. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced. However, we recommend *not* using homology reduction on the input sequences, and instead prevent duplicated strains by using binsplitting (see section: _recommended workflow_.)

### Outputs

Vamb produces the following output files:

- `log.txt` - a text file with information about the Vamb run. Look here (and at stderr) if you experience errors.
- `tnf.npz`, `lengths.npz` `rpkm.npz`, `mask.npz` and `latent.npz` - [Numpy .npz](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) files with TNFs, contig lengths. RPKM, which sequences were successfully encoded, and the latent encoding of the sequences.
- `model.pt` - containing a PyTorch model object of the trained VAE. You can load the VAE from this file using `vamb.encode.VAE.load` from Python.
- `clusters.tsv` - a two-column text file with one row per sequence: Left column for the cluster (i.e bin) name, right column for the sequence name. You can create the FASTA-file bins themselves using `vamb.vambtools.write_bins`, or using the function `vamb.vambtools.write_bins` (see `doc/tutorial.html` for more details).

# Recommended workflow

__1) Preprocess the reads and check their quality__

We use AdapterRemoval combined with FastQC for this - but you can use whichever tool you think gives the best results.

__2) Assemble each sample individually and get the contigs out__

We recommend using metaSPAdes on each sample individually. You can also use scaffolds or other nucleotide sequences instead of contigs as input sequences to Vamb. Assemble each sample individually, as single-sample assembly followed by samplewise binsplitting gives the best results.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique, and filter away small contigs__

You can use the function `vamb.vambtools.concatenate_fasta` for this or the script `src/concatenate.py`. 

:warning: *Important:* Vamb uses a neural network to encode sequences, and neural networks overfit on small datasets. We have tested that Vamb's neural network does not overfit too badly on all datasets we have worked with, but we have not tested on any dataset with fewer than 50,000 contigs.

You should not try to bin very short sequences. When deciding the length cutoff for your input sequences, there's a tradeoff here between choosing a too low cutoff, retaining hard-to-bin contigs which adversely affects the binning of *all* contigs, and choosing a too high one, throwing out good data. We use a length cutoff of 2000 bp as default but haven't actually run tests for the optimal value.

Your contig headers must be unique. Furthermore, if you want to use binsplitting (and you should!), your contig headers must be of the format {Samplename}{Separator}{X}, such that the part of the string before the *first* occurrence of {Separator} gives a name of the sample it originated from. For example, you could call contig number 115 from sample number 9 "S9C115", where "S9" would be {Samplename}, "C" is {Separator} and "115" is {X}.

Vamb is faily memory efficient, and we have run Vamb with 1000 samples and 5.9 million contigs using <30 GB of RAM. If you have a dataset too large to fit in RAM and feel the temptation to bin each sample individually, you can instead use a tool like MASH to group similar samples together in smaller batches, bin these batches individually. This way, you can still leverage co-abundance. **NB:** We have a version using memory-mapping that is much more RAM-efficient but 10-20% slower. Here we have processed a dataset of 942 samples with 30M contigs (total of 117Gbp contig sequence) in 40Gb RAM - see branch `mmap`.

__4) Map the reads to the FASTA file to obtain BAM files__

:warning: *Important:* If you allow reads to map to multiple contigs, the abundance estimation will be more accurate. However, all BAM records for a single read *must* be consecutive in the BAM file, or else Vamb will miscount these alignments. This is the default order in the output of almost all aligners, but if you use BAM files sorted by alignment position and have multi-mapping reads, you must sort them by read name first.

Be careful to choose proper parameters for your aligner - in general, if reads from contig A align to contig B, then Vamb will bin A and B together. So your aligner should map reads with the same level of discrimination that you want Vamb to use. Although you can use any aligner that produces a specification-compliant BAM file, we prefer using `minimap2` (though be aware of [this annoying bug in minimap2](https://github.com/lh3/minimap2/issues/15)):

```
minimap2 -d catalogue.mmi /path/to/catalogue.fna.gz; # make index
minimap2 -t 28 -N 5 -ax sr catalogue.mmi --split-prefix mmsplit sample1.forward.fastq.gz sample1.reverse.fastq.gz | samtools view -F 3584 -b --threads 8 > sample1.bam
```

:warning: *Important:* Do *not* filter the aligments for mapping quality as specified by the MAPQ field of the BAM file. This field gives the probability that the mapping position is correct, which is influenced by the number of alternative mapping locations. Filtering low MAPQ alignments away removes alignments to homologous sequences which biases the depth estimation.

If you are using BAM files where you do not trust the validity of every alignment in the file, you can filter the alignments for minimum nucleotide identity using the `-z` flag (uses the `NM` optional field of the alignment, we recommend setting it to `0.95`), and/or filter for minimum alignments score using the `-s` flag (uses the `AS` optional field of the alignment.)

We have found that MetaBAT2's `jgi_summarize_bam_contig_depths` program estimates BAM depths more accurate than Vamb's `parsebam` module. For the best results, we recommend [downloading MetaBAT2](https://bitbucket.org/berkeleylab/metabat/src/master/), using `jgi_summarize_bam_contig_depths` to estimate depths, and then running Vamb with `--jgi` instead of `--bamfiles`. Also consider using the `snakemake` workflow which will do this for you. 

__5) Run Vamb__

By default, Vamb does not output any FASTA files of the bins. In the examples below, the option `--minfasta 200000` is set, meaning that all bins with a size of 200 kbp or more will be output as FASTA files.
If you trust the alignments in your BAM files, use:

`vamb -o SEP --outdir OUT --fasta FASTA --bamfiles BAM1 BAM2 [...] --minfasta 200000`,

where `SEP` in the {Separator} chosen in step 3, e.g. `C` in that example, `OUT` is the name of the output directory to create, `FASTA` the path to the FASTA file and `BAM1` the path to the first BAM file. You can also use shell globbing to input multiple BAM files: `my_bamdir/*bam`.

If you don't trust your alignments, set the `-z` and `-s` flag as appropriate, depending on the properties of your aligner. For example, if I used the aligner BWA MEM, I would use:

`vamb -o SEP -z 0.95 -s 30 --outdir OUT --fasta FASTA --bamfiles BAM1 BAM2 [...] --minfasta 200000`

__6) Postprocess the results__

Vamb will bin every input contig. Contigs that cannot be binned with other contigs will result in single-bin contigs. Therefore, the output of Vamb includes many tiny bins, which may be relevant if you are looking for e.g. plasmids or viruses, but not if you are looking for cellular genomes. These small bins may account for the large majority of bins. When using the output of Vamb, it is therefore critical to filter the output bins based on whatever criteria is relevant for your particular analysis. You could use a binning quality control tool such as CheckM, or align the bins to reference genomes, or simply filter away all bins smaller than, say, 500 kbp.

## Parameter optimisation (optional)

The default hyperparameters of Vamb will provide good performance on any dataset. However, since running Vamb is fast (especially using GPUs) it is possible to try to run Vamb with different hyperparameters to see if better performance can be achieved (note that here we measure performance as the number of near-complete bins assessed by CheckM). We recommend to try to increase and decrease the size of the neural network and have used Vamb on datasets where increasing the network resulted in more near-complete bins and other datasets where decreasing the network resulted in more near-complete bins. To do this you can run Vamb as (default for multiple samples is `-l 32 -n 512 512`)`:

```
vamb -l 24 -n 384 384 --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C --minfasta 200000
vamb -l 40 -n 768 768 --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C --minfasta 200000
```

It is possible to try any combination of latent and hidden neurons as well as other sizes of the layers. Number of near-complete bins can be assessed using CheckM and compared between the methods. Potentially see the snakemake folder `workflow` for an automated way to run Vamb with multiple parameters.
