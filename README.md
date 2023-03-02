# Vamb

Created by Jakob Nybo Nissen and Simon Rasmussen, Technical University of Denmark and Novo Nordisk Foundation Center for Protein Research, University of Copenhagen.

Vamb is a metagenomic binner which feeds sequence composition information from a contig catalogue and co-abundance information from BAM files into a variational autoencoder and clusters the latent representation. It performs excellently with multiple samples, and pretty good on single-sample data. Vamb is implemented purely in Python (with a little bit of Cython) and can be used both from command line and from within a Python interpreter.

:star: Vamb is benchmarked in CAMI2. Read our interpretations of the results [here.](https://github.com/RasmussenLab/vamb/blob/master/doc/CAMI2.md) :star:

For more information about the implementation, methodological considerations, and advanced usage of Vamb, see the [Vamb paper in Nature Biotechnology](https://doi.org/10.1038/s41587-020-00777-4), a [blog post](https://go.nature.com/2JzYUvI) by Jakob on the development of Vamb, and the [Vamb tutorial](https://github.com/github/RasmussenLab/blob/master/doc/tutorial.md).

:mega: Interested on maximizing the number of high quality bins from your samples?
Please check out Avamb.
Avamb is a metagenomics binning pipeline composed by Vamb and Aamb, an adversarial autoencoder for metagenomics binning, as well as a bin dereplication and quality filtering step that runs CheckM2.
Similar to Vamb, Avamb leverages sequence composition information from a contig catalogue and co-abundance information from BAM files by projecting those features into  latent spaces, where the metagenomics binning actually takes place.
Avamb has shown to reconstruct 23% more high quality bins as well as to increase the bins quality.
For more information, please check the [Avamb snakemake page](https://github.com/RasmussenLab/avamb/tree/avamb_new/workflow_avamb).

The Avamb pre-print can be found here [biorxiv](https://doi.org/10.1101/2023.02.27.527078). 


# Installation

### Installation for casual users:

Recommended: Vamb can be installed with pip (thanks to contribution from C. Titus Brown):
```
pip install vamb
```

:bangbang: An active Conda environment can hijack your system's linker, causing an error during installation. Either deactivate `conda`, or delete the `~/miniconda/compiler_compats` directory before installing with pip.

Alternatively, it can be installed as a [Bioconda's package](https://anaconda.org/bioconda/vamb) (thanks to contribution from Antônio Pedro Camargo).
Currently, the Conda version lags behind the pip version, so we recommend using pip. The BioConda package does not include GPU support.
 
```
conda install -c pytorch pytorch torchvision cudatoolkit=10.2
conda install -c bioconda vamb
```

### Installation for advanced users:

If you want to install the latest version from GitHub, or you want to change Vamb's source code, you should install it like this:

```
# clone the desired branch from the repository, here master
git clone https://github.com/RasmussenLab/vamb -b master
cd vamb
pip install -e .
```

### Installing by compiling the Cython yourself

If you can't/don't want to use pip/Conda, you can do it the hard way: Get the most recent versions of the Python packages `cython`, `numpy`, `torch` and `pycoverm`. Compile `src/_vambtools.pyx` then move the resulting binary to the inner of the two `vamb` directories. Check if it works by importing `vamb` in a Python session.

# Running
For more detailed description of the recommended workflow, see the tutorial in the `doc` directory. 

For more command-line options, see the command-line help menu:
```
vamb -h
```

To run, Vamb needs a FASTA file with contigs from one or more samples, and a set of BAM files, one from each sample, with reads mapped to the same FASTA file.
You can either create the input files using your own workflow before calling Vamb, or you can use the Snakemake workflow included in Vamb to automatically create the input files before running Vamb.

## How to run: Using your own input files

For this example, let us suppose you have a directory of  reads in a
directory `/path/to/reads`, and that _you have already quality controlled them_.

1. Run your favorite metagenomic assembler on each sample individually:

```
spades.py --meta /path/to/reads/sample1.fw.fq.gz /path/to/reads/sample1.rv.fq.gz
-k 21,29,39,59,79,99 -t 24 -m 100gb -o /path/to/assemblies/sample1
```

2. Concatenate the input contigs to a single FASTA file discarding very short contigs (e.g. < 2 kbp), e.g. using Vamb's `concatenate.py` script:

```
python concatenate.py /path/to/catalogue.fna.gz /path/to/assemblies/sample1/contigs.fasta /path/to/assemblies/sample2/contigs.fasta  [ ... ]
```

3. Use your favorite aligner to map each your read files back to the FASTA file:

```
minimap2 -d catalogue.mmi /path/to/catalogue.fna.gz; # make index
minimap2 -t 8 -N 5 -ax sr catalogue.mmi --split-prefix mmsplit /path/to/reads/sample1.fw.fq.gz /path/to/reads/sample1.rv.fq.gz | samtools view -F 3584 -b --threads 8 > /path/to/bam/sample1.bam
```

4. Run Vamb:

```
vamb --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C
```

5. Apply any desired postprocessing to Vamb's output.

## How to run: Using the Vamb Snakemake workflow

To make it even easier to run Vamb, we have created a [Snakemake](https://snakemake.readthedocs.io/en/stable/#) workflow.
This workflow runs steps 2-5 above using `minimap2` to align, and [CheckM](https://ecogenomics.github.io/CheckM/) to estimate completeness and contamination of the resulting bins.
The workflow can run both on a local machine, a workstation and a HPC system using `qsub`. It can be found in the `workflow` folder - see the file `workflow/README.md` for details.

# Detailed user instructions
See the tutorial in `doc/tutorial.md` for even more detailed instructions.

### Inputs
Vamb relies on two properties of the DNA sequences to be binned:

* The kmer-composition of the sequence (here tetranucleotide frequency, *TNF*) and
* The abundance of the contigs in each sample (the *depth* or the *RPKM*).

So before you can run Vamb, you need to have files from which Vamb can calculate these values:

* TNF is calculated from a regular FASTA file of DNA sequences.
* Depth is calculated from _sorted_ BAM-files of a set of reads from each sample mapped to that same FASTA file.

Remember that the quality of Vamb's bins are no better than the quality of the input files. If the assembly is poor quality, or if your BAM files are constructed carelessly, for example by allowing reads from distinct species to crossmap indiscriminately, your BAM files will not contain information with which Vamb can separate those species. In general, you want reads to map only to contigs within the same phylogenetic distance that you want Vamb to bin together.

Estimation of TNF and RPKM is subject to statistical uncertainty. Therefore, Vamb works less well on short sequences and on data with low depth. Therefore, you get the best results using contigs (or scaffolds) with the longest possible length. Do not attempt to deduplicate or homology reduce your input sequences.

### Outputs

Vamb produces the following output files:

- `log.txt` - A text file with information about the Vamb run. Look here (and at stderr) if you experience errors.
- `composition.npz`: A Numpy .npz file that contain all kmer composition information computed by Vamb from the FASTA file. This can be provided to another run of Vamb to skip the composition calculation step.
- `abundance.npz`: Similar to `composition.npz`, but this file contains information calculated from the BAM files. Using this as input instead of BAM files will skip re-parsing the BAM files, which take a significant amount of time.
- `model.pt`: A file containing the trained VAE model. When running Vamb from a Python interpreter, the VAE can be loaded from this file to skip training.
- `latent.npz`: This contains the output of the VAE model.
- `clusters.tsv` - A two-column text file with one row per sequence: Left column for the cluster (i.e bin) name, right column for the sequence name. You can create the FASTA-file bins themselves using the script in `src/create_fasta.py`

# Recommended workflow
Garbage in, garbage out. To get good results with Vamb, you must make sure the inputs are of high quality.

__1) Preprocess the reads and check their quality__

We use `fastp` for this - but you can use whichever tool you think gives the best results.

__2) Assemble each sample individually and get the contigs out__

We recommend using metaSPAdes on each sample individually. MEGAHIT also works well. You can also use scaffolds or other nucleotide sequences instead of contigs as input sequences to Vamb. Assemble each sample individually, as single-sample assembly followed by samplewise binsplitting gives the best results.

__3) Concatenate the FASTA files together while making sure all contig headers stay unique, and filter away small contigs__

You can use the function `vamb.vambtools.concatenate_fasta` for this or the script `src/concatenate.py`. 

:warning: *Important:* Vamb uses a neural network to encode sequences, and neural networks overfit on small datasets. We have designed Vamb to underfit most datasets, but it may still overfit if you have less than e.g. 20,000 contigs.

You should not try to bin very short sequences. When deciding the length cutoff for your input sequences, there's a tradeoff here between on one hand choosing a too low cutoff, retaining short contigs that are hard to bin AND which adversely affects the binning of *all* contigs, and on the other hand choosing a too high cutoff, throwing out good data.
We use a length cutoff of 2000 bp as default but haven't actually run tests for the optimal value. The optimal tradeoff will depend on your specific dataset and the biological information you are interested in.

Your contig headers must be unique. Furthermore, if you want to use binsplitting (and you should!), your contig headers must be of the format {Samplename}{Separator}{X}, such that the part of the string before the *first* occurrence of {Separator} gives a name of the sample it originated from.
For example, you could call contig number 115 from sample number 9 "S9C115", where "S9" would be {Samplename}, "C" is {Separator} and "115" is {X}.
The script `src/concatenate.py` will automatically rename your sequences to this format.

Vamb is fairly memory efficient, and we have run Vamb with 1000 samples and 5.9 million contigs using <30 GB of RAM.
If you have a dataset too large to fit in RAM and feel the temptation to bin each sample individually, you can instead use a tool like `sourmash` to group similar samples together in smaller batches, bin these batches individually.
This way, you can still leverage co-abundance, and will get much better results.

__4) Map the reads to the FASTA file to obtain BAM files__

:warning: *Important:* Vamb only accepts BAM files sorted by coordinate. You can sort BAM files with `samtools sort`.

Be careful to choose proper parameters for your aligner - in general, if reads from contig A align to contig B, then Vamb will bin A and B together. So your aligner should map reads with the same level of discrimination that you want Vamb to use. Although you can use any aligner that produces a specification-compliant BAM file, we prefer using `minimap2` (though be aware of [this annoying bug in minimap2](https://github.com/lh3/minimap2/issues/15) - see our Snakemake script for how to work around it):

```
minimap2 -d catalogue.mmi /path/to/catalogue.fna.gz; # make index
minimap2 -t 28 -N 5 -ax sr catalogue.mmi --split-prefix mmsplit sample1.forward.fastq.gz sample1.reverse.fastq.gz | samtools view -F 3584 -b --threads 8 > sample1.bam
```

:warning: *Important:* Do *not* filter the aligments for mapping quality as specified by the MAPQ field of the BAM file. This field gives the probability that the mapping position is correct, which is influenced by the number of alternative mapping locations. Filtering low MAPQ alignments away removes alignments to homologous sequences which biases the depth estimation.

If you are using BAM files where you do not trust the validity of every alignment in the file, you can filter the alignments for minimum nucleotide identity using the `-z` option.

__5) Run Vamb__

By default, Vamb does not output any FASTA files of the bins. In the examples below, the option `--minfasta 200000` is set, meaning that all bins with a size of 200 kbp or more will be output as FASTA files.
Run Vamb with:

`vamb -o SEP --outdir OUT --fasta FASTA --bamfiles BAM1 BAM2 [...] --minfasta 200000`,

where `SEP` in the {Separator} chosen in step 3, e.g. `C` in that example, `OUT` is the name of the output directory to create, `FASTA` the path to the FASTA file and `BAM1` the path to the first BAM file. You can also use shell globbing to input multiple BAM files: `my_bamdir/*bam`.

Note that if you provide insufficient memory (RAM) to Vamb, it will run very slowly due to [thrashing](https://en.wikipedia.org/wiki/Thrashing_(computer_science)). Make sure you don't run out of RAM!

__6) Postprocess the results__

Vamb will bin every input contig. Contigs that cannot be binned with other contigs will result in single-bin contigs. Therefore, the output of Vamb includes many tiny bins, which may be relevant if you are looking for e.g. plasmids or viruses, but not if you are looking for cellular genomes. These small bins may account for the large majority of bins. When using the output of Vamb, it is therefore CRITICAL to filter the output bins based on whatever criteria is relevant for your particular analysis. You could use a binning quality control tool such as CheckM, or align the bins to reference genomes, or simply filter away all bins smaller than, say, 500 kbp.

## Parameter optimisation (optional)

The default hyperparameters of Vamb will provide good performance on any dataset. However, since running Vamb is fast (especially using GPUs) it is possible to try to run Vamb with different hyperparameters to see if better performance can be achieved (note that here we measure performance as the number of near-complete bins assessed by CheckM). We recommend to try to increase and decrease the size of the neural network and have used Vamb on datasets where increasing the network resulted in more near-complete bins and other datasets where decreasing the network resulted in more near-complete bins. To do this you can run Vamb as (default for multiple samples is `-l 32 -n 512 512`):

```
vamb -l 24 -n 384 384 --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C --minfasta 200000
vamb -l 40 -n 768 768 --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam -o C --minfasta 200000
```

It is possible to try any combination of latent and hidden neurons as well as other sizes of the layers. Number of near-complete bins can be assessed using CheckM and compared between the methods. Potentially see the snakemake folder `workflow` for an automated way to run Vamb with multiple parameters.
