# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent representation. It performs excellently with multiple samples, and pretty good on single-sample data. Vamb is implemented purely in Python (with a little bit of Cython) and can be used both from command line and from within a Python interpreter.

For more information on the background, context, and theory of Vamb, read [our paper on bioRxiv](https://www.biorxiv.org/content/early/2018/12/19/490078) (DOI: 10.1101/490078)

For more information about the implementation, methodological considerations, and advanced Python usage of Vamb, see the tutorial file (`doc/tutorial.html`)

# Installation
Vamb is most easily installed with pip - make sure your pip version is up to date, as it won't work with ancient versions (v. <= 9).

### Installation for casual users:

Recommended: Vamb can be installed with pip (thanks to contribution from C. Titus Brown):
```
pip install https://github.com/RasmussenLab/vamb/archive/2.1.0.zip
```

or using [Bioconda's package](https://anaconda.org/bioconda/vamb) (thanks to contribution from Antônio Pedro Camargo):
```
conda install -c bioconda vamb
```

### Installation for advanced users:

This is for developers who want to be able to edit Python files and have the changes show up directly in the running command:

```
# clone the desired branch from the repository, here master, but you can clone a release as well
git clone https://github.com/jakobnissen/vamb -b master
cd vamb
pip install -e .
```

### Installing by compiling the Cython yourself

If you can't/don't want to use pip/Conda, you can do it the hard way: Get the most recent versions of the Python packages `cython`, `numpy`, `torch` and `pysam`. Compile `src/_vambtools.pyx`, (see `src/build_vambtools.py`) then move the resulting binary to the inner of the two `vamb` directories. Check if it works by importing `vamb` in a Python session.

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

You probably want to split your bins by sample to deduplicate them and preserve strain variation, if so you should use a binsplitting separator (see the section on the FASTA file preparation under "Recommended workflow"). If your separator is "C", add `-o C`:

```
vamb --outdir vambout --fasta /path/to/file.fasta --bamfiles /path/to/bamfiles/*.bam -o C
```

For more command-line options, see the command-line help menu:
```
vamb -h
```

For a detailed explanation of the parameters of Vamb, or different inputs, see the tutorial in the `doc` directory.

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

:warning: *Important:* Vamb works best on datasets with more than 100,000 contigs. If you have fewer than 50,000 contigs, we recommend you use another binner like MetaBAT2.

Remember that the quality of Vamb's bins are no better than the quality of the input files. If your BAM file is constructed carelessly, for example by allowing reads from distinct species to crossmap, the BAM file will not contain information with which Vamb can separate the crossmapping species. In general, you want reads to map only to contigs within the same phylogenetic distance that you want Vamb to bin together.

Estimation of TNF and RPKM is subject to statistical uncertainty. Therefore, Vamb works poorly on short sequences and on data with low depth. Vamb *can* work on shorter sequences such as genes, which are more easily homology reduced, but the results of working on (larger) contigs are better.

With fewer samples (up to about 100), we recommend using contigs from a concatenation of single-sample assemblies with a minimum contig length cutoff of ~2000-ish basepairs, and then split your bins according to the sample of origin by using the `-o` option. With many samples, the number of contigs can become too large for the data to fit in RAM. The better approach is to split the dataset up into groups of similar samples and bin these groups, instead of binning each individual sample.

### Outputs

Vamb produces the following output files:

- `log.txt` - a text file with information about the Vamb run. Look here (and at stderr) if you experience errors.
- `tnf.npz`, `lengths.npz` `rpkm.npz`, `mask.npz` and `latent.npz` - [Numpy .npz](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) files with TNFs, contig lengths. RPKM, which sequences were successfully encoded, and the latent encoding of the sequences.
- `model.pt` - containing a PyTorch model object of the trained VAE. You can load the VAE from this file using `vamb.encode.VAE.load` from Python.
- `clusters.tsv` - a two-column text file with one row per sequence: Left column for the cluster (i.e bin) name, right column for the sequence name. You can create the FASTA-file bins themselves using `vamb.vambtools.write_bins`, or using the function `vamb.vambtools.write_bins` (see `doc/tutorial.html` for more details).

# Recommended workflow

__1) Preprocess the reads and check their quality__

We use AdapterRemoval combined with FastQC for this.

__2) Assemble each sample individually OR co-assemble and get the contigs out__

We recommend using metaSPAdes on each sample individually. Single-sample assembly followed by samplewise binsplitting gives the best results.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique, and filter away small contigs__

You can use the function `vamb.vambtools.concatenate_fasta` for this.

:warning: *Important:* If you have fewer contigs than about 100,000, Vamb's autoencoder is prone to overfitting, and below about 50,000 contigs Vamb performs worse than MetaBAT2. So count your sequences at this point and use MetaBAT2 instead if you have too few contigs.

There's a tradeoff here between a too low cutoff, retaining hard-to-bin contigs which adversely affects the binning of all contigs, and throwing out good data. We use a length cutoff of 2000 bp as default but haven't actually run tests for the optimal value.

Your contig headers must be unique. Furthermore, if you want to use binsplitting (and you should!), your contig headers must be of the format {Samplename}{Separator}{X}, such that the part of the string before the *first* occurrence of {Separator} gives a name of the sample it originated from. For example, you could call contig number 115 from sample number 9 "S9C115", where "S9" would be {Samplename}, "C" is {Separator} and "115" is {X}.

If you have more than about 100 samples, the autoencoding and clustering steps can take a long time, and the mapping may be error-prone depending on the aligner. We have successfully run Vamb with 1000 samples and ~6 million contigs, but we recommend clustering your samples by similarity (for example, using MASH), and split your dataset into groups of ~100 samples, and then bin these independently.

__4) Map the reads to the FASTA file to obtain BAM files__

:warning: *Important:* If you allow reads to map to multiple contigs, the abundance estimation will be more accurate. However, all BAM records for a single read *must* be consecutive in the BAM file, or else Vamb will miscount these alignments. This is the default order in the output of almost all aligners, but if you use BAM files sorted by alignment position and have multi-mapping reads, you must sort them by read name first.

Be careful to choose proper parameters for your aligner - in general, if reads from contig A aligns to contig B, then Vamb will bin A and B together. So your aligner should map reads with the same level of discrimination that you want Vamb to use. We have used following commands to create our BAM files:

BWA MEM for smaller datasets: `bwa mem filtered_contigs.fna sample1.forward.fastq.gz sample1.reverse.fastq.gz -t 8 | samtools view -F 3584 -b --threads 8 > sample1.bam`

Minimap2 for larger ones: `minimap2 -t 28 -N 50 -ax sr almeida.mmi sample1.forward.fastq.gz sample1.reverse.fastq.gz | samtools view -F 3584 -b --threads 8 > sample1.bam`

:warning: *Important:* If you are using BAM files where you do not trust the validity of every alignment in the file, you can filter the alignments for minimum nucleotide identity using the `-z` flag (uses the `NM` optional field of the alignment, we recommend setting it to `0.95`), and/or filter for minimum alignments score using the `-s` flag (uses the `AS` optional field of the alignment.)

__5) Run Vamb__

If you trust the alignments in your BAM files, use:

`vamb -o SEP --outdir OUT --fasta FASTA --bamfiles BAM1 BAM2 [...]`,

where `SEP` in the {Separator} chosen in step 3, e.g. `C` in that example, `OUT` is the name of the output directory to create, `FASTA` the path to the FASTA file and `BAM1` the path to the first BAM file. You can also use shell globbing to input multiple BAM files: `my_bamdir/*bam`.

If you don't trust your alignments, set the `-z` and `-s` flag as appropriate, depending on the properties of your aligner. For example, with BWA MEM, I would use:

`vamb -o SEP -z 0.95 -s 30 --outdir OUT --fasta FASTA --bamfiles BAM1 BAM2 [...]`
