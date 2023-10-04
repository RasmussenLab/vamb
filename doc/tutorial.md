# Walkthough of Vamb from the Python interpreter

__NOTE: This uses Vamb internals subject to change!__.

Vamb's user interface is defined by its command-line invokation,
so when using Vamb from the Python interpreter, you are interacting with internals in Vamb which may be changed at any moment. Nonetheless, it is useful to do so for understanding how Vamb works under the hood.

For the same reason, this following tutorial may not work with Vamb's master branch or the current latest release.
This tutorial has been run with Vamb 4.2.0.

### Introduction
The Vamb pipeline consist of a series of tasks each which have a dedicated module:

1) Parse fasta file and get TNF of each sequence, as well as sequence length and names (module `parsecontigs`)

2) Parse the BAM files and get abundance estimate for each sequence in the fasta file (module `parsebam`)

3) Train a VAE with the abundance and TNF matrices, and encode it to a latent representation (module `encode`)

4) Cluster the encoded inputs to metagenomic bins (module `cluster`)

In the following chapters of this walkthrough, we will show how to use Vamb by example, what each step does, some of the theory behind the actions, and the different parameters that can be set.
With this knowledge, you should be able to extend or alter the behaviour of Vamb more easily.

For the examples, we will assume the following relevant prerequisite files exists in the directory `/home/jakob/Downloads/vambdata/`:

* `contigs.fna.gz` - FASTA contigs which were mapped against, and
* `bam/*.bam` - 3 sample BAM files from reads mapped to the contigs above

For details of what exactly these filed contain, and how to create these files correctly, see Vamb's README.md file.

## Table of contents:

### [Introduction to metagenomic binning and best practices](#introduction)

### [Calculate the sequence tetranucleotide frequencies](#parsecontigs)

### [Calculate the abundance matrix](#parsebam)

### [Train the autoencoder and encode input data](#encode)

### [Binning the encoding](#cluster)

### [Postprocessing the bins](#postprocessing)

### [Summary of full workflow](#summary)

### [Running Vamb with low memory (RAM)](#memory)

### [Optional: Benchmarking your bins](#benchmark)

<a id="introduction"></a>
## Introduction to metagenomic binning and best practices

When we began working on the Vamb project, we had been doing reference-free metagenomics for years using MetaBAT2, and thought we had a pretty good idea of how metagenomic binning worked and what to watch out for.
Turns out, there's more to it than we had thought about.
During this project, our group have revised the way we use binners multiple times as we've learned the dos and don'ts. 
In this introduction I will cover some facts of binning that is useful for the end user.

__1. Binning is a hard problem__

Various binning papers (including ours), typically brag about how many high-quality genomes they've been able to reconstruct from data.
However, examining the actual number of genomes reconstructed compared to the estimated diversity of the samples can be sobering:
Only a very small fraction of the organisms are binned decently.
In fact, only a small fraction of the *sequenced* organisms are binned decently!

Information is lost at every step along the way:
In the sequencing itself, during assembly, when estimating contig abundance, and during the binning.
For now, this is just a fact to accept, but we hope it will improve in the future, especially with better long-read sequencing technologies.

__2. There is a tradeoff in what contigs to discard__

Typically, the inputs to the binners are sequence composition and/or sequence abundance.
Both these metrics are noisy for smaller contigs, as the kmer composition is not stable for small contigs, and the estimated depth across a genome varies wildly.
This causes small contigs to be hard to handle for binners, as the noise swamps the signal.

Worse, the presence of hard-to-bin contigs may adversely affect the binning of easy-to-bin contigs depending on the clustering algorithm used.
One solution is to discard small contigs, but here lies a tradeoff: When are you just throwing away good data? This is especially bitter when the large majority of metagenomic assemblies usually lies in short contigs.
With current (2023) assembly and mapping technology, we are using a threshold of around 2000 bp.

__3. Garbage in, garbage out__

Binners like MaxBin, MetaBAT2 and Vamb uses kmer frequencies and estimated depths to bin sequences.
While the kmer frequency is easy to calculate (but sensitive to small contigs - see above), this is not the case for depth.
Depths are estimated by mapping reads back to the contigs and counting where they map to.
Here especially, it's worth thinking about exactly how you map:

* In general, if reads are allowed to map to subjects with a nucleotide identity of X %, then it is not possible to distinguish genomes with a higher nucleotide identity than X % using co-abundance, and these genomes will be binned together.
  This means you want to tweak your alignment tool to only output alignments with the desired minimum query/subject identity - for example 97%.

* The MAPQ field of a BAM/SAM file is the probability that the mapping position is correct.
  Typically, aligners outputs a low MAPQ score if a read can map to multiple sequences.
  However, because we expect to have multiple similar sequences in metagenomics (e.g. the same strain from different samples), filtering low MAPQ-alignments away using e.g. Samtools will give bad results.

So when creating your BAM files, make sure to think of the above.

__4. It's hard to figure out if a binner is doing well__

What makes a good binner? This question quickly becomes subjective or context dependent:

* What's the ideal precision/recall tradeoff?
  One researcher might prefer pure but fragmented genomes, whereas another might want complete genomes, and care less about the purity.

* What's the ideal cutoff in bin quality where the binner should not output the bin?
  Is it better to only produce good bins, or produce all bins, including the dubious ones?

* Which taxonomic level is the right level to bin at?
  If nearly identical genomes from different samples go in different bins, is that a failure of the binner, or successfully identitfying microdiversity? What about splitting up strains? Species?
  Since the concept of a bacterial species is arbitrary, should the binner ideally group along nucleotide identity levels, regardless of the taxonomic annotation?

* Are plasmids, prophages etc. considered a necessary part of a bin?
  Should a binner strive to bin plasmids with their host?
  What if it comes at a cost of genomic recall or precision?

* If a genome is split equally into three bins, with the genome being a minority in each bin does that mean the genome is present at a recall of 0%, because each bin by majority vote will be assigned to another genome, a recall of 33%, because that's the maximal recall in any particular bin, or at 100%, because the whole genome is present in the three bins?

* Should contigs be allowed to be present in multiple bins?
  If so, any ambiguity in binning can be trivially resolved by producing multiple bins where there is ambiguity (in extrema: Outputting the powerset of input contigs) - but surely, this just pushes the problems to the next point in the pipeline.
  If not, then conserved genomic regions, where reads from multiple species can be assembled into a single contig, cannot be binned in the multiple bins where it is actually present.

* What postprocessing can you assume people are doing after the binning?
  Given e.g. a bin deduplication postprocessing step, creating multiple similar bins does not matter so much.
  However, if no postprocessing is done, doing so will give misleading results. 

Unfortunately, all these choices have a significant impact on how you asses performance of a binner - and thus, *how you create a binner in the first place*.
That means, given different definitions of binning quality, different binners might to worse or better.

<a id="importing"></a>
## Importing Vamb

First step is to get Vamb imported.
If you installed with `pip`, it should be directly importable.
Else, you might have to add the path of Vamb to `sys.path`

```python
import vamb
```

<a id="parsecontigs"></a>
## Calculate the sequence tetranucleotide frequencies

I use the suggested `vamb.parsecontigs.Composition.read_file` with the inputs and outputs as written:

```python
path = "/home/jakob/Downloads/vambdata/contigs.fna.gz"

# Use Reader to open plain or zipped files. File must be opened in binary mode
with vamb.vambtools.Reader(path) as filehandle:
    composition = vamb.parsecontigs.Composition.from_file(filehandle)
```

__The `vamb.vambtools.Reader`__, object will automatically and transparently reads plaintext, gzipped, bzip2 and .xz files, and return an opened file object.
You can also directly pass an ordinary file handle (or indeed any iterable of lines as `bytes` objects) to the `Commposition.from_files` method.

Note that reading zipped files will slow down the FASTA parsing quite a bit.
But the time spent parsing the FASTA file will likely still be insignificant compared to the other steps of Vamb.

__The rationale for parsing the contigs__ is that it turns out that related organisms tend to share a similar kmer-distribution across most of their genome.
The reason for that is not understood, even though it's believed that common functional motifs, GC-content and presence/absence of endonucleases explains some of the observed similary.

In Vamb, we care about the "TNF" - the tetranucleotide frequency - it's the frequency of the canonical kmer of each 4mer in the contig.
We use 4-mers because there are 256 4-mers, which is an appropriate number of features to cluster - not so few that there's no signal and not so many it becomes unwieldy and the estimates of the frequencies become uncertain.
We could also have used 3-mers.
In tests we have made, 3-mers are *almost*, but not quite as good as 4-mers for separating different species.
You could probably switch tetranucleotide frequency to trinucleotide frequency in Vamb without any significant drop of accuracy.
However, there are 1024 5-mers, that would be too many features to handle comfortably, and it could easily cause memory issues.

We do not work with tetranucleotide frequencies (TNF) directly.
TNFs are highly correlated, for example, we would expect the frequency of `AAGA` to be very similar to `AGAA`.
Some of these correlations are due to e.g.
GC content which is a valid signal.
But some of these correlations are due to interdependencies between the TNFs which does NOT constitute a signal and is pure redundancy in the data.
Removing this saves RAM, computing resources, and may also make the VAE converge better.
There are three such dependencies:

* Being frequencies, they must sum to 1.
* The frequency of a kmer must be the same as the reverse complement (RC), because the RC is simply present on the other strand.
  We only observe one strand, which one is arbitrary.
  So we average between kmers and RC.
  For example, if we see 31 AAGA and 24 TCTT, we will note (31+24)/2 = 27.5 of each instead.
* Every time we observe a kmer with the bases ABCD, we *must* also observe the immediately following kmer, BCDE.
  Hence, for every 3-mer XYZ, it is true that (AXYZ + CXYZ + GXYZ + TXYZ) = (XYZA + XYZC + XYZG + XYZT).
  This is not always true for finite contigs because the 4mer at the end has no "next" 4mer.
  But that fact is a tiny measurement error in the first place, so we do not care about that.

Based on these constraints, we can create a set of linear equations, and then find the kernel matrix L which projects any matrix of 4mer frequencies T to a matrix P with fewer features (103 in this case):

$$\textbf{P} = \textbf{TL}$$

We enforce constraint 2 by tallying the mean of each kmer and its frequency.
This reverse complement averaging can be done using a matrix multiply on the raw tetranucleotide frequencies F with a reverse-complementing matrix R:

$$\textbf{T} = \textbf{FR}$$

And since matrix multiplication is associative, we can create a single kernel K which does both the reverse complementation averaging and projects down to 103 features in one matrix multiply:

$$\textbf{P} = \textbf{TL} = (\textbf{FR})\textbf{L} = \textbf{F}(\textbf{RL}) = \textbf{FK}$$

The kernel K is calculated in `src/create_kernel.py`, but comes pre-calculated with Vamb.
See the method section of [Kislyuk et al. 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2765972/) for more details.

__`Composition.from_files` has an optional argument `minlength`__ which sets the filter removing any contigs shorter than this.
We don't know exactly what the sweet spot is, but it's probably somewhere around ~2000 bp.

The problem with filtering contigs using `minlength` is that the smaller contigs which are thrown away will still recruit reads during the mapping that creates the BAM files, thus removing information from those reads.
For that reason, we recommend filtering the contigs *before* mapping.
Therefore, Vamb doesn't default its `minlength` to 2000.

__The memory consumption of Vamb can be an issue__, so at this point, you should probably consider whether you have enough RAM.
This is a small dataset, so there's no problem.
With hundreds of samples and millions of contigs however, this becomes a problem, even though Vamb is fairly memory-friendly.
If you think memory might be an issue, see the [Running Vamb with low memory (RAM)](#memory) section.
If invoked from command-line, Vamb will attempt to be memory friendly.

---
The `Composition` object we computed above is an internal object, meaning its contents might change in future releases of Vamb.

Nonetheless, let's see how the information above is stored in the `Composition` object at the time of writing:


```python
print('Data fields of Composition:', composition.__slots__)
print('\tType of metadata:', type(composition.metadata))
print('\tType of matrix:', type(composition.matrix), "with shape", composition.matrix.shape, "and element type", composition.matrix.dtype)
```

    Data fields of Composition: ['metadata', 'matrix']
    	Type of metadata: <class 'vamb.parsecontigs.CompositionMetaData'>
    	Type of matrix: <class 'numpy.ndarray'> with shape (10000, 103) and element type float32


The `composition.matrix` represents the projected TNF data with 103 dimensions as described above.
Hence, it's an N x 103 matrix of 32-bit floats, such that each row represents the frequency of the 103 TNF features.

What's in the metadata?

```python
metadata = composition.metadata
print("Data fields of CompositionMetaData:", metadata.__slots__)
print("\tidentifiers", type(metadata.identifiers), metadata.identifiers[:3])
print("\tlengths", type(metadata.lengths), metadata.lengths[:3])
print("\tmask", type(metadata.mask), metadata.mask[:3])
print("\trefhash", type(metadata.refhash))
print("\tminlength", metadata.minlength)
```

```
Data fields of CompositionMetaData: ['identifiers', 'lengths', 'mask', 'refhash', 'minlength']
    identifiers <class 'numpy.ndarray'> ['S27C175628' 'S27C95602' 'S27C25358']
    lengths <class 'numpy.ndarray'> [2271 3235 3816]
    mask <class 'numpy.ndarray'> [ True  True  True]
    refhash <class 'bytes'>
    minlength 100
```

* The `identifiers` field is a Numpy array with the identifier (i.e. name before any whitespace) of each kept sequence in the FASTA file
* The `lengths` give the number of DNA bases in each sequence
* The `mask` keeps track of which entries in the original FASTA file was kept, and which were filtered away by being below `minlength`.
  It's `True` if kept.
  Hence, `len(mask)` is the number of total sequences in the FASTA file, and `sum(mask) == len(identifiers) == len(lengths)`
* The `refhash` is a hash of the identifiers.
It it used to verify that the Abundance object (see below) was calculated with the same set of sequences as the Composition.
* Lastly, `minlength` stores the `minlength` set on construction.

---
<a id="parsebam"></a>
## Calculate the abundance matrix

Next, we need to compute the abundance of each sequence:

```python
bamfiles = ['6.sorted.bam', '7.sorted.bam', '8.sorted.bam']
bamfiles = ['/home/jakob/Downloads/vambdata/bam/' + p for p in bamfiles]

# This step takes some time for large files.
abundance = vamb.parsebam.Abundance.from_files(
    bamfiles,
    None, # temp directory, if needed
    composition.metadata,
    True, # check identifiers are same as metadata's
    0.0, # minimum alignment identity of read
    3 # number of threads
)
```
---
The idea here is that two contigs from the same genome will always be physically present together, and so they should have a similar abundance in all samples.

Vamb uses `pycoverm` to estimate abundances, which itself uses `CoverM`.

We pass in the `Composition.metadata` object when constructing the Abundance.
This will enable a check that the `Composition` and `Abundance` was made from the same sequences, and also automatically filter the BAM files by the same `minlength` setting as the `Composition`.

We set the minimum read ID to `0.0`, meaning no reads are discarded by mapping too poorly.
In general, it's preferrable to supply a good mapping to Vamb instead of making a poor mapping and then filtering it afterwards.

In this case, we don't use a temporary directory, setting it to `None`.
When loading more BAM files, or larger BAM files, a temporary directory is needed to avoid loading all the BAM files into memory at once which may cause memory issues.
When Vamb is run from command line, it will use such a temporary directory.

Like the `Composition` object, let's have a look at the `Abundance` object:

```python
print('Data fields of Abundance:', abundance.__slots__)
print('\tmatrix:', type(abundance.matrix), "with shape", abundance.matrix.shape, "and element type", abundance.matrix.dtype)
print('\tsamplenames:', type(abundance.samplenames), abundance.samplenames)
print('\tminid:', abundance.minid)
print('\tType of refhash:', type(abundance.refhash))
```

```
# Output
Data fields of Abundance: ['matrix', 'samplenames', 'minid', 'refhash']
        matrix: <class 'numpy.ndarray'> with shape (10056, 3) and element type float32
        samplenames: <class 'numpy.ndarray'> ['/home/jakob/Downloads/vambdata/bam/6.sorted.bam'
 '/home/jakob/Downloads/vambdata/bam/7.sorted.bam'
 '/home/jakob/Downloads/vambdata/bam/8.sorted.bam']
        minid: 0.001
        Type of refhash: <class 'bytes'>
```
* The `matrix` field contains an N_sequences x N_samples matrix with the estimated abundance for that sequence in that sample.
* The `samplenames` field store the path of each input file. This is the same order as the columns of the matrix.
* `minid` stores the input minimum ID. Currently, this is set to minimum 0.001, because pycoverm does not work properly with a `minid` of 0.0
* `refhash` is exactly like `CompositionMetaData`'s.

---

Now, I tend to be a bit ~~paranoid~~<sup>careful</sup>, so if I loaded in 500 GB of BAM files, I'd want to save the work I have now in case something goes wrong - and we're about to fire up the VAE so lots of things can go wrong. So, I save the `Composition` and `Abundance` to files using their `.save` methods, so they can be loaded later:


```python
composition.save('/tmp/composition.npz')
abundance.save('/tmp/abundance.npz')
```
---
<a id="encode"></a>
## Train the autoencoder and encode input data

Now that we have the features that Vamb uses for binning, we need to use the titular variational autoencoders to create an embedding, which we can then cluster.
This happens in three steps:
1. Create the VAE itself
2. Create a `DataLoader` (and a mask) representing the features the VAE will use
3. Use the `DataLoader` to train the VAE.

Training networks always take some time.
If you have a GPU and CUDA installed, you can pass `cuda=True` to the VAE to train on your GPU for increased speed.
With a beefy GPU, this can make quite a difference.
I run this on my laptop (with a puny Intel GPU), so I'll just use my CPU.
And I'll run just 10 epochs rather than the more suitable 500:

```python
import sys

vae = vamb.encode.VAE(nsamples=abundance.nsamples)
dataloader = vamb.encode.make_dataloader(
    abundance.matrix,
    composition.matrix,
    composition.metadata.lengths,
)

with open('/tmp/model.pt', 'wb') as modelfile:
    vae.trainmodel(
        dataloader,
        nepochs=10,
        modelfile=modelfile,
        batchsteps=None,
        logfile=sys.stdout
    )
```

```
    Network properties:
    CUDA: False
    Alpha: 0.15
    Beta: 200.0
    Dropout: 0.2
    N hidden: 512, 512
    N latent: 32

    Training properties:
    N epochs: 10
    Starting batch size: 256
    Batchsteps: None
    Learning rate: 0.001
    N sequences: 10056
    N samples: 3

    Time: 09:59:32  Epoch:   1  Loss: 6.91338e-01  CE: 4.63212e-01  AB: 1.21466e-01  SSE: 1.00855e-01  KLD: 5.28434e-03  Batchsize: 256
    Time: 09:59:34  Epoch:   2  Loss: 5.07434e-01  CE: 3.96126e-01  AB: 4.06000e-02  SSE: 6.07153e-02  KLD: 1.02072e-02  Batchsize: 256
    Time: 09:59:35  Epoch:   3  Loss: 4.66900e-01  CE: 3.72669e-01  AB: 2.89287e-02  SSE: 5.45222e-02  KLD: 1.11823e-02  Batchsize: 256
    Time: 09:59:36  Epoch:   4  Loss: 4.47042e-01  CE: 3.60973e-01  AB: 2.38930e-02  SSE: 5.04224e-02  KLD: 1.15812e-02  Batchsize: 256
    Time: 09:59:37  Epoch:   5  Loss: 4.34523e-01  CE: 3.54323e-01  AB: 2.00695e-02  SSE: 4.83254e-02  KLD: 1.20165e-02  Batchsize: 256
    Time: 09:59:38  Epoch:   6  Loss: 4.26110e-01  CE: 3.49559e-01  AB: 1.75491e-02  SSE: 4.67316e-02  KLD: 1.22742e-02  Batchsize: 256
    Time: 09:59:40  Epoch:   7  Loss: 4.21156e-01  CE: 3.45925e-01  AB: 1.71833e-02  SSE: 4.55252e-02  KLD: 1.25041e-02  Batchsize: 256
    Time: 09:59:41  Epoch:   8  Loss: 4.18073e-01  CE: 3.44946e-01  AB: 1.60090e-02  SSE: 4.45488e-02  KLD: 1.25530e-02  Batchsize: 256
    Time: 09:59:42  Epoch:   9  Loss: 4.14682e-01  CE: 3.43305e-01  AB: 1.48536e-02  SSE: 4.40206e-02  KLD: 1.25180e-02  Batchsize: 256
    Time: 09:59:43  Epoch:  10  Loss: 4.11646e-01  CE: 3.42163e-01  AB: 1.38400e-02  SSE: 4.32769e-02  KLD: 1.24808e-02  Batchsize: 256
```

First we create the VAE, then we create the dataloader and the mask.

In the `make_dataloader` function, the features (TNF and abundances) are transformed and normalized, such that the input arrays are transformed into arrays suitable for the VAE.
Currently, the VAE uses the following derived features
* TNF which are z-score normalized across each tetranucleotide.
  The z-score normalization improves the learning when using a SSE (sum of squared error) loss.
* Absolute abundances of each sequence, summed across samples.
  First, the depths are normalized by the total depth of each sample, in order to control for varying sequencing depth in each sample.
  Then, the depths are summed across the samples, then log-transformed (with depths of zero set to a small pseudocount).
  The log transformation is because we expect depths of sequences to be approxiamtely exponentially distributes.
  Then, these log transformed counts are zscore normalized.
* The relative abundances between samples.
  First, abundances are normalized by sample depths, as for the absolute abundances.
  Then, the relative abundance is computed, such that they sum to 1 across samples.
* The loss weight of the contigs.
  We want the network to prioritize long contigs, as they contain more of the target genomes.
  Hence, we compute a weight proportional to the log of the sequence length.

The data loader also shuffles the contigs to avoid any bias in the training stemming from the order of the contigs which is rarely random.

Here, we kept the default value `False` to the `destroy` keyword of `make_dataloader`.
If this is set to `True`, the input arrays are normalized and masked in-place, modifying them.
This prevents another in-memory copy of the data, which can be critical for large array sizes.

I passed a few keywords to the `trainmodel` function: 
* `nepochs` should be obvious - the number of epochs trained
* `modelfile` is a filehandle to where the final, trained VAE model will be saved
* `logfile` is the file to which the progress text will be printed.
  Here, I passed `sys.stdout` for demonstration purposes.
* `batchsteps` should be `None`, or an iterable of integers.
  If `None`, it does nothing.
  Else, the batch size will double when it reaches an epoch in `batchsteps`.
Increasing batch size helps find better minima; see the paper "Don't Decay the Learning Rate, Increase the Batch Size".

The VAE encodes the high-dimensional (n_samples + 105 features) input data in a lower dimensional space (`nlatent` features).
When training, it learns an encoding scheme, with which it encodes the input data to a series of normal distributions, and a decoding scheme, in which it uses one value sampled from each normal distribution to reconstruct the input data.

The theory here is that if the VAE learns to reconstruct the input, the distributions must be a more efficient encoding of the input data, since the same information is contained in fewer neurons.
If the input data for the contigs indeed do fall into bins, an efficient encoding would be to simply encode the bin they belong to, then use the "bin identity" to reconstruct the data.
We force it to encode to *distributions* rather than single values because this makes it more robust - it will not as easily overfit to interpret slightly different values as being very distinct if there is an intrinsic noise in each encoding.

### The loss function
The loss of the VAE consists of a few terms:

* The absolute abundance loss measures the error when reconstructing the absolute abundances.
  This is computed as sum of squared error (SSE).
* The relative abundace loss measures the error of reconstructing the relative abundances.
  This is computed as cross entropy (CE)
* The TNF loss is also computed as SSE.
* The kullback-leibler divergence (KLD) measures the dissimilarity between the encoded distributions and its prior (the standard gaussian distribution N(0, 1)).
  This penalizes learning.

All terms are important:
Abundance and TNF losses are necessary, because we believe the VAE can only learn to effectively reconstruct the input if it learns to encode the signal from the input into the latent layers.
In other words, these terms incentivize the network to learn something.
KLD is necessary because we care that the encoding is *efficient*, viz. it is contained in as little information as possible.
The entire point of encoding is to encode a majority of the signal while shedding the noise, and this is only achieved if we place contrains on how much the network is allowed to learn.
Without KLD, the network can theoretically learn an infinitely complex encoding, and the network will learn to encode both noise and signal.

In normal autoencoders, people use binary cross-entropy rather than crossentropy.
We believe crossentropy is more correct here, since we normalize by letting the depths sum to one across a contig.
In this way, you can view the depths distribution across samples as a probability distribution that a random mapping read will come from each sample.

In `encode.py`, the loss function $L$ is written as the following terms:

$$L_A = (A_{in} - A_{out})^2 \frac{1}{\alpha N}$$
$$L_R = -log(R_{out} + 10^{-9}) R_{in} \frac{(1 - \alpha)(N - 1)}{N log(N)}$$
$$L_T = (T_{in} - T_{out})^2 \frac{\alpha}{103}$$
$$L_K = -\frac{1}{2L\beta} \sum (1 + log(\sigma)^2 - \mu^2 - \sigma^2)$$
$$L = L_A + L_R + L_T + L_K$$

Where:
* $N$ is the number of samples
* $L$ is the number of dimensions of the latent space
* $\alpha$ and $\beta$ are hyperparameters, explained below
* The latent embedding is a Gaussian parameterized by $\mu$ and $\sigma$

The loss is composed of four parts as explained above:
* $L_A$, the loss of reconstructing the absolute abundances $A_{in}$
* $L_R$, the loss of reconstructing the relative abundances $R_{in}$
* $L_T$, the loss of reconstructing the tetranucleotide frequencies $T_{in}$
* $L_K$, Kullback-Leibler Divergence, i.e. regularization loss


### Explaining scaling factors
#### Constant scaling factors
To make it easier to optimise hyperparameters for the network, we want a single variable to control the ratio between abundance and TNF reconstruction losses - we call this $\alpha$.
Similarly, we should be able to control the ratio of reconstruction loss to regularization loss with a single variable $\beta$.
Ideally, these hyperparameters should be insensitive to the number of samples.

But here comes a problem.
While we want to scale abundance and TNF loss we can't know *beforehand* what these losses will be for any given dataset.
And, in any rate, these values changes across the training run (that's the point of training!).

What we *can* reason about is the values of CE and SSE in a totally *naive*, network which had *no knowledge* of the input dataset.
This represents the state of the network before *any* learning is done.
What would such a network predict?

The absolute depths and the TNFs are normalized to follow an approximately normal distribution with mean around 0.
This means the expected sum of squared error (SSE) for these two losses is 1 per input TNF neuron.
This means the expected SSE for TNFs is 103, and the expected SSE for absolute abundances is 1.

For the relative abundances, we assume that the baseline prediction of an untrained network is close to $[N^{-1}, N^{-1} ... N^{-1}]^{T}$, which, by the definition of cross entropy, would yield a unweigted relative abundance loss of of $log(N)$.

Therefore, we divide the absolute abundance, relative abundance, and TNF abundances by 1, $log(N)$ and 103, respectively.

The regularization loss $L_K$ is a sum over the $L$ dimensions of the latent space.
So, we multiply a factor $\frac{1}{L}$ to $K_L$.

The small constant factor $10^{-9}$ used when computing the relative abundance loss $L_R$ is to prevent computing the undefined $log(0)$ when the network happens to predict exactly 0.

#### Relative vs absolute abundances
With N samples, the relative abundances contain exactly $\frac{N-1}{N}$ of the total abundance signal, whereas the absolute abundances contain $\frac{1}{N}$ of the signal.
This should be clear, since the relative abundances has $N-1$ degrees of freedom, while the absolute ones has 1 degree of freedom.

So, we scale the relative and absolute abundances by $\frac{N-1}{N}$ and $\frac{1}{N}$, respectively

#### Factors $\alpha$ and $\beta$
With the above factors in place, the expected reconstruction losses are all 1, and we can scale the TNF versus abundance losses, and the reconstruction losses vs regularization losses.

Hence, we simply define $\alpha$ as the fraction of the loss from TNF.
So, we scale TNF loss with $\alpha$ and the two abundance losses with $1 - \alpha$.

To balance regularization and reconstruction losses, we multiply the regularization loss $L_K$ by $\frac{1}{\beta}$.

The values of $\alpha$ and $\beta$ have been empirically found.

### Back to Python
If you look at the outputs from training, you can see the KL-divergence rises the first epoch as it learns the dataset and the latent layer drifts away from its prior.
Eventually, the penalty associated with KL-divergence outweighs the reconstruction loss.
At this point, the KLD will stall, and then fall.
This point depends on $\beta$ and the complexity of the dataset.

Okay, so now we have the trained `vae` and gotten the `dataloader`.
Let's feed the dataloader to the VAE in order to get the latent representation.

---
```python
# No need to pass gpu=True to the encode function to encode on GPU
# If you trained the VAE on GPU, it already resides there
latent = vae.encode(dataloader)

print(latent.shape)
```
```
(10056, 32)
```
---
That's the 10,056 contigs each represented by the mean of their latent distribution in `n_latent` = 32 dimensions.

Sometimes, you'll want to reuse a VAE you have already trained.
For this, I've added the `VAE.save` method of the VAE class, as well as a `VAE.load` method.
Defining `modelfile` when training as I did above calls `vae.save(modelfile)`.
We can always use that file to recreate the VAE and have a pretrained model.
But remember - a trained VAE only works on the dataset it's been trained on, and not necessarily on any other!

I want to **show** that we get the exact same network back that we trained, so here I encode the first contig, delete the VAE, reload the VAE and encode the first contig again.
The two encodings should be identical.

---


```python
import torch

(rel_in, tnf_in, abs_in, weights_in) = next(iter(dataloader))

# Put VAE in testing mode - strictly not necessary here, since it goes in test
# mode when encoding latent, as we did above
vae.eval()

# Calling the VAE as a function encodes and decodes the arguments,
# returning the outputs and the two distribution layers
rel_out, tnf_out, abs_out, mu = vae(rel_in, tnf_in, abs_in)

# The mu layer is the encoding itself
print(mu[0])

# Now, delete the VAE
del vae

# And reload it from the model file we created:
# Here we can specify whether to put it on GPU and whether it should start
# in training or evaluation (encoding) mode. By default, it's not on GPU and 
# in testing mode
vae = vamb.encode.VAE.load('/tmp/model.pt')
rel_out, tnf_out, abs_out, mu = vae(rel_in, tnf_in, abs_in)
print(mu[0])
```

```
# Note: The precise values here are subject to change.
# with new releases of Vamb
tensor([  6.6391,  -1.6709,   9.4311,   8.6112,  -5.3148, -19.5604,  23.9720,
         -6.5217,   3.1008, -18.2250, -19.0971,  13.8964,   7.4714,  -4.2162,
          4.6839,   8.4110, -22.4917,  -0.9834,  -2.9969,  -0.3610,  -2.1918,
         12.8971,  -1.2768,  -4.7298,  -6.9836, -19.1941,  15.2329,  -8.4965,
         25.6173,   4.7288, -11.1059,  -8.7754], grad_fn=<SelectBackward0>)
tensor([  6.6391,  -1.6709,   9.4311,   8.6112,  -5.3148, -19.5604,  23.9720,
         -6.5217,   3.1008, -18.2250, -19.0971,  13.8964,   7.4714,  -4.2162,
          4.6839,   8.4110, -22.4917,  -0.9834,  -2.9969,  -0.3610,  -2.1918,
         12.8971,  -1.2768,  -4.7298,  -6.9836, -19.1941,  15.2329,  -8.4965,
         25.6173,   4.7288, -11.1059,  -8.7754], grad_fn=<SelectBackward0>)
```
We get the same values back, meaning the saved network is the same as the loaded network!

---
<a id="cluster"></a>
## Binning the encoding

__The role of clustering in Vamb__

Fundamentally, the process of binning is just clustering sequences based on some of their properties, with the expectation that sequences from the same organism cluster together.

In the previous step, we encoded each sequence to a vector of real numbers.
The idea behind encoding is that, because the neural network attempts to strike a balance between having low reconstruction error (i.e. high fidelity) and having low Kullback-Leibler divergence (i.e. containing as little information as possible), then the network will preferentially encode signal over noise.
In this way, the VAE act like a denoiser.
This has been explored in other papers by people before us.
Furthermore, because the latent space has fewer dimensions than the input space, clustering is quicker and more memory efficient.

With the latent representation conveniently represented by an (n_contigs x n_features) matrix, you could use any clustering algorithm to cluster them (such as the ones in the Python package `sklearn.cluster`).
In practice though, you have perhaps a million contigs and prior constrains on the diameter, shape and size of the clusters, so non-custom clustering algorithms will probably be slow and inaccurate.

The module `vamb.cluster` implements a iterative medoid clustering algorithm.
The algorithm is complicated and not very elegant, so I will go through it here in painstaking detail.

### Overview of Vamb clustering algorithm

In very broad strokes, the algorithm works like this:

```
A) Choose the next sequence P still remaining in the dataset, which have not been
   chosen before as P.
B) Sample points close to P, settling on the point with highest local density.
   We assume this is the center of a cluster, and call it the medoid M.
C) Find the distance d from M where the density of other points drops according to certain criteria. This represents the egde of the current cluster.
D) If no d can be found, restart from A).
    If it has restarted too often the last X tries, relax criteria in B).
    If criteria has been maximally relaxed, choose an predetermined value of d.
E) Output all points within d of M, and remove them from dataset.
F) Repeat from A) until no points remain in the dataset.
```

The idea behind this algorithm is that we expect the clusters to be roughly spherical, have a high density near the middle, and be separed form other clusters by an area with lower density.
Hence, we should be able to move from P to M in step B) by moving towards higher density.
And we should be able to find the radius of the cluster from M by considering a distance from M where the density of points drops.

This algorithm consists of several steps, discussed below.
Most of the steps unfortunately have several parameters which have been chosen by a combination of guessing, gut feeling and testing.

As a distance measure, we are using cosine distance.
This is particularly useful for a couple of reasons:

First, it is extremely quick to calculate.
Given two points represented by vectors $\textbf{a}$ and $\textbf{b}$, we define cosine distance as:

$$
\frac{1}{2} - \frac{\textbf{a} \cdot \textbf{b}}{2|\textbf{a}| |\textbf{b}|}
$$

This is slightly different from the ordinary definition, because it is squeezed between 0 and 1, rather than -1 and 1.

Now if we preprocess all points by calculating:

$$
\bar{\textbf{a}} = \textbf{a} \circ \frac{1}{\sqrt{2} \textbf{|a|}}
$$

Where $\circ$ is elementwise multiplication, then the cosine distance is simply:

$$
\frac{1}{2} - \bar{\textbf{a}} \cdot \bar{\textbf{b}}
$$

And since the points are represented by a matrix $\bar{\textbf{M}}$, with each row vector being a point, finding the distance to from the i'th point of $\bar{\textbf{M}}$ to all other points is then a series of vector products as in the equation above, which can be expressed in a single matrix multiply and subtraction:

$$
\frac{1}{2} - \bar{\textbf{M}} \times \bar{\textbf{M}}_{i}^{T}
$$

Second, with all the $N_{latent}$ dimensions being used by the neural network the surface of an $N_{latent}$-dimensional hyperphere is so large that it's not likely that multiple clusters are far in euclidian space but close in cosine space.

And third, it works better than the other alternatives we tried in practice, although I don't understand why for instance Euclidian distance is not better.

### Finding the medoid
First, we order the sequences from longest to shortest.
This is because we believe that long contigs
* Have more stable features, and so are more likely to be placed correctly in latent space
* Are more important to bin correctly, so they have larger weight in both training and clustering

Therefore, they are the best candidates for cluster centers.

We pick "seed sequences" S in order from longest to shortest.

Then, we use the following algorithm to find the medoid.
The idea is that, as you move nearer to a cluster's centre, the length-weighed mean distance to close contigs will decreate (i.e. the local density will increase)

Currently, the default parameters DC and MAX_STEPS are 0.05 and 25, respectively.

```
Function: Wander medoid
1. Pick next point S as the seed. If last point has been reached, begin from the longest sequence again.
2. Let C be the set of contigs within a small distance DC of S.
3. Let D be the distances from S to C, and L be their lengths.
4. Compute the M(S) = mean of L*(DC-D).
5. Sample a new point P in C to check if M(P) < M(S).
6. If M(P) < M(S), let S <- P, go to 2.
   If not, continue until there are no points in C that haven't been checked, or P has been sampled MAX_STEPS consecutive times.
7. Return S as medoid
```

### Finding the right distance
Having found a medoid above, we need to find out if it's the center of a suitable cluster, and if so, what the radius of that cluster is.

In order to see how this is done, it is useful to look at these graphs first:
    
![png](tutorial_files/figA.png)
        
![png](tutorial_files/figB.png)
    
When picking an arbitrary contig as medoid (here contig 'S4C11236' in the CAMI2 toy human Airways short read dataset), and calculating the distances to all other contigs in the dataset, it follows a distribution as seen in Figure A, when the contribution of each contig is weighed by the contig length.
Unsurprisingly, the latent representation of most other contigs are mostly uncorrelated with the chosen contig, so the large majority of points are at a distance of around 0.5.
No contigs have a larger distance than 0.8. 

But look at the lower left corner of Figure A: A small group of contigs appear to have a smaller distance to the medoid than most others - those of the cluster it belongs to.
If the cluster is well-separated, this small initial peak should be mostly isolated from the main peak around 0.5 by a band of empty space, with a low density.
Figure B confirms this is true.

Vamb groups the observed distances from the medoid M to a histogram similar to that in Figure B, and then iterates over the y-values of the histogram attempting to find first an initial peak, then a valley.
At the bottom of the valley, the cutoff distance threshold is set, and all points closer to M than this point is clustered out.
This threshold is depicted in Figure B by the dotted line.
Vamb smoothes the histogram in order to mitigate the influence of random variation on the minima/maxima of the function.
The smoothed hisotgram is visible in Figure B as a blue line.

The algorithm for detecting the threshold is highly heuristic, but in short, it reads the histogram from left to right and tries to find the pattern of a small peak, then a valley, then a larger peak.
This is taken to signify the presence of a well-defined cluster separated from the other points.

The algorithm has a "strictness criteria" that determines how strict the cluster must be separated from the other points.
Any medoids which produce a distance histogram with an insufficiently separated cluster are ignored, and the next medoid is considered.

Initially, the clustering algorithm is strict.
As the clustering algorithm progresses, it measures how often medoids are successful in producing clusters.
When too few medoids are successful, it will lower the strictness parameter.
Eventually, all points will be clustered, no matter how poorly separated they are.

### Clustering
Let's have a look at the actual Python function that does the clustering in Vamb:

```python
clusters = dict()
lengths = composition.metadata.lengths
for (n, cluster) in enumerate(vamb.cluster.ClusterGenerator(latent, lengths)):
    medoid, members = cluster.as_tuple()
    clusters[medoid] = members
    
    if n + 1 == 1000: # Stop after 1000 clusters
        break
```
You can just put it directly in a dictionary:

```python
clusters = dict(vamb.cluster.ClusterGenerator(latent, lengths))
```

<a id="postprocessing"></a>
## Postprocessing the clusters

We haven't written any dedicated postprocessing modules because how to postprocess really depends on what you're looking for in your data.

One of the greatest weaknesses of Vamb - probably of metagenomic binners in general - is that the bins tend to be highly fragmented.
Unlike other binners, Vamb does not hide this fact, and will bin *all* input contigs.
This means you'll have lots of tiny bins, some of which are legitimate (viruses, plasmids), but most are parts of larger genomes that didn't get binned properly - about 2/3 of the bins here, for example, are 1-contig bins. 

---
<a id="summary"></a>
## Summary of full workflow

This is an example of the full workflow from start to end.
Calling Vamb from command line does, roughly speaking, this, except with some more bells and whistles.

```python
import sys
import os
import vamb

contig_path = "/home/jakob/Downloads/vambdata/contigs.fna.gz"
bamfiles = ['/home/jakob/Downloads/vambdata/bam/' + p for p in ['6.sorted.bam', '7.sorted.bam', '8.sorted.bam']]
out_path = "/tmp/out_bins.tsv"

# Calculate composition
with vamb.vambtools.Reader(contig_path) as filehandle:
    composition = vamb.parsecontigs.Composition.from_file(filehandle)

# Calculate abundance
abundance = vamb.parsebam.Abundance.from_files(
    bamfiles,
    None,
    composition.metadata,
    True,
    0.0,
    3
)

# Encode
vae = vamb.encode.VAE(nsamples=abundance.nsamples)
dataloader = vamb.encode.make_dataloader(
    abundance.matrix,
    composition.matrix,
    composition.metadata.lengths,
)

vae.trainmodel(dataloader)
latent = vae.encode(dataloader)

clusters: Iterable[tuple[str, Iterable[str]]], separator: str

# Cluster and output clusters
names = composition.metadata.identifiers
with open(out_path 'w') as binfile:
    clusterer = vamb.cluster.ClusterGenerator(latent, composition.metadata.lengths)
    binsplit_clusters = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in clusterer
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)
```

<a id="memory"></a>
## Running Vamb with low memory (RAM)

In the Vamb software, a series of tradeoffs can be taken to decrease RAM consumption, usually to the detriment of some other property like speed or convenience (though not accuracy).
With all of those tradeoffs taken, Vamb is relatively memory efficient.
By default, several of these tradeoffs are are not taken when running from a Python interpreter, however they are all enabled when running Vamb from command line.

The memory consumption of the encoding step is usually the bottleneck.
With all memory-saving options enabled, this uses approximately $4 * N_{contigs}*(103+N_{latent}+N_{samples})$ bytes, plus some hundreds of megabytes of overhead.
As a rule of thumb, if you don't have at least two times that amount of memory, you might want to enable some or all of these memory optimizations.

Here's a short list of all the available memory saving options:

- Pass a path to the `cache_directory` in the `vamb.parsebam.Abundance.from_files` function.
- In the `encode.make_dataloader` function, set `destroy` to True.
This will in-place modify your RPKM and TNF array, deleting unusable rows and normalizing them.
This prevents the creation of a copy of the data.
- Similarly, set `destroy=True` when instantiating the `cluster.ClusterGenerator` class.
Again, the input (latent) datasets are modified in-place, saving a copy of the data.
This function will completely consume the input array.
- When clustering, instead of generating clusters in memory, instantiate the cluster iterator and pass it directly to `cluster.write_clusters`.
This way, each cluster will be written to disk after creation and not accumulate in memory.
This obviously requires disk space, and you will have to reload the clusters if you are planning on using them.

<a id="benchmark"></a>
## Optional: Benchmarking your bins

For benchmarking, you can use our tool VambBenchmarks.jl, which is hosted at https://github.com/jakobnissen/VambBenchmarks.jl.
This tool is not yet published or thoroughly documented, so please contact us if you wish to use it.