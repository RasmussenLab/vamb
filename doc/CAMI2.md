# Vamb results in the CAMI2 challenge
The Critical Assesment of Metagenome Interpretations second challenge (CAMI2)
has been published. CAMI2 benchmarks metagenomics tools against each other in
a fair competition, and as such enables the field to progress in a measurable
manner, and enables downstream users to pick the best tool for the job.

In CAMI2, the Vamb (commit fa045c0) results were highly anomalous. In this text,
I (the first author of Vamb), attempts to interpret the results.

### TL;DR
Vamb's results were poor in CAMI2. This is explained by Vamb being the only binner
that included preprocessing but no postprocessing. This combination is disaterous,
and not recommended for any binner, including Vamb. Based on measures
not affected by postprocessing, Vamb was the best non-ensemble binner. If Vamb's
best practises had been used (binsplitting), it would presumably be better still.

### Vamb's observed results in CAMI2
In short, here's how the results appears:

* Vamb is - by far - the binner with the lowest mean bin completeness
* Vamb is also - by far - the binner with the highest mean bin purity
* Vamb does medium-well in adjusted Rand index compared to the gold standard binning,
except on the "strain madness gsa" dataset.
* Vamb is mediocre in % binned bp.
* Vamb appears to do better binning MEGAHIT assembled contigs than binning artificial
gold standard contigs, compared to other binners
* Vamb does better on unique strains (genomes without close relatives in the dataset)
than on non-unique, compared to other binners
* Vamb is far better than others at binning plasmids and high-copy circular elements
* Overall, Vamb is ranked 9/12, 5/7 and 7/10 in the three datasets.
* Vamb was significantly slower to run than MetaBAT by a factor of ~600 for one dataset.

### So... what happened?
On the surface, Vamb appears to weigh purity much higher than other binners, which
leads to poor average performance. However, digging a little deeper, the true
reason for Vamb's strange results appear - and it turns out that Vamb's results
are mostly incomparable with the other binners'.

### Unlike the other binners, Vamb's bins were subject to no postprocessing
Unlike other binners, Vamb outputs every single input contig. Contigs that could
not be binned with other contigs are simply output as a single-contig bin. This means
that the large majority of bins will be composed of one or a few hard-to-bin contigs.
Other binners like MetaBAT and MaxBin does not output bins below a chosen output size.

The presence of an overwhelming amount of tiny bins explains most of Vamb's score
in CAMI2. Purity is so high (99.9% on the harder "strain madness" dataset)
because most bins only consists of a single, or a few small contigs. By definition,
a single-contig bin has a purity of 100%. Conversely, recall is so abysmal (0.1%)
because a single, small contig from a bacterial genome has a recall near 0%.

We made the choice of building Vamb to keep all bins because we, as tool builders,
cannot predict the context in which the user wants to use Vamb. For example, when
working with Vamb, a co-author of Vamb discovered that it was excellent at binning
viral genomes. These viral bins would have been filtered away by MetaBAT or MaxBin.
If the researcher wants to work with larger genomes, small bins can simply
be filtered away. If not, we won't make that choice on your behalf.

This choice is vindicated by CAMI2's result. When binning plasmids, Vamb far
exceeds all other binners (F1 = 70.8 vs the next best's 12.7).
Plasmids are simply filtered away by the other binners, but not by Vamb!

### Preprocessing without postprocessing gave the worst of both worlds
In the CAMI2 paper, besides precision and recall, the two measures of binners are:

1. ARI: adjusted Rand index, a measure of the similarity between a binner's
output and the gold standard
2. Binned bp: The fraction of input basepairs that appears in the
output.

The rationale for including binned bp is that any binner could do well on the first
metric if it choses to only output the most certain bins. In contrast, a binner
which outputs _everything_ must necessarily also output the hard-to-bin contigs.

You would think that Vamb, without any bin filtering, would max out on binned bp.
However, unfortunately for us, Vamb was run with strict _preprocessing_, which
removed all contigs that were too short, or with too low depth, to easily bin.
This in turn lowered the binned bp to mediocre values. The lack of postprocessing
caused the low ARI.

If Vamb had been run without preprocessing, it would have uniquely maxed out on
binned bp, which would have given a hint something was amiss. Conversely, if both
pre- and postprocessing had been applied, it would have done much better at ARI.
The results in CAMI2, as included, represents the worst possible combination,
and does not represent any recommended workflow with Vamb.

### Because the input data was co-assembled, binsplitting was not used
Vamb's README recommends users to use _binsplitting_. This simple technique applies
to binning multiple samples from similar environments. Using it means assembling each
sample individually, binning all contigs together across samples, splitting the
resulting bins by input sample, and finally deduplicating the bins if necessary.
Contrast this to the most common workflow, where samples are co-assembled across
samples before binning. We argue in the Vamb paper that binsplitting leads to
superior results, and have built Vamb with binsplitting in mind.

However, the CAMI2 challenge input data was co-assembled. As such, binsplitting
was not possible. I suspect this led to poorer performance of Vamb, because Vamb,
by being designed for binsplitting, may be designed to purposefully bin similar
strains from different together, which would usually be separated by binsplitting.

### What would a more useful comparison look like?
CAMI2's supplementary material include a table where they count the number of
high-quality genomes recovered using each binner. Using a >90% recall, <5% contamination
threshold, and looking at the MEGAHIT assembled input data (more realistic), Vamb
is the best single binner for the "marine" and "plant-associated" datasets, only
beat by ensemble methods (which would undoubtably be improved by including Vamb).
Vamb recovers 8 bins from the "strain madness" dataset, whereas CONCOCT and MetaWRAP
recovers 9 - here too, Vamb is near the top.

Again, this is _without_ binsplitting. With binsplitting, I am convinced Vamb would
dominate the other binners. In fairness though, binsplitting is a simple technique
that could be used with every binner. It just happens that Vamb is the only current
binner to include it by default. I would imagine that, if all binners used binsplitting,
the results would be similar to the ones shown above: Vamb would be good, but not
outstanding.

### A note on time and memory consumption
One of the more surprising findings is that Vamb was so slow and memory-hungry
compared to other binners. It was second-fastest behind MetaBAT, but by quite
a big margin. In my own experience, Vamb is the fastest binner around.

More precisely, Vamb _scales the best_ with input size, although it has high
baseline time and memory usage. The 5 hours for the marine dataset is somewhat
expected, with Vamb's high up-front cost and all. However, CAMI2's tests showed
Vamb taking 31 hours for a dataset MetaBAT finished in 3 minutes.
This surprises me. Any dataset big enough to take 31 hours for Vamb should take
days for MetaBAT.

Digging through the old logs from our own CAMI2 submission, Vamb took 1.5 hours
for the marine dataset (compared to 5.4 reported by CAMI2), and 0.8 hours for the
strain madness dataset (compared to 31.8 reported by CAMI2). That's 35 times
slower than what we measured! Stanger still, the relative time spent on the
two datasets are inversed.

I don't know as to what happened there, but can speculate:
* Perhaps CAMI2 ran Vamb on a computer with a poor BLAS library. If the fallback
BLAS library of numpy is used, it can slow down the program quite a bit.
* It is possible Vamb's performance was dragged down by its use of pysam.
Even though this is not recommended usage according to Vamb's README, this is a
legitimate concern, and we will work on transitioning from use of pysam
to CoverM in the future.

### The requirements for CAMI2 versus Vamb
The CAMI2 competition benchmarked 10 binners, as well as assemblers and taxonomic
binners. Undoubtably no two authors of these 10 binners would fully agree to what
even constitutes a good set of bins, which assumptions a good binning tool are
allowed to make, or how to measure the output of a binner.

I recognize the difficulty of CAMI2 establishing a common benchmark suite
for such disparate tools. As such, this text should not be taken to mean I
disapprove of the methodology of CAMI2. To make a level playing field, some
rules have to be set, which will inevitably bias the results.

Here, I simply argue that your conclusions reflect what you have chosen to measure.
CAMI2's binning challenge measures the quality of all output bins, indiscriminately.
Vamb requires the researcher to make sure both input and output meaningfully address
the use case of the individual user.

This philosophy runs deep in Vamb. We do not require any specific assembler, just
sequences in FASTA format, be they contigs, scaffolds, genes or anything else.
Vamb's default preprocessing of input is _extremely_ lax. We accept any aligner,
as long as it produces BAM files, no matter its paramters or assumptions.
And we do not postprocess the bins.

The drawback of this philosophy is that, if you do not check the inputs and outputs
are meaningful for your application, you will get rubbish results. The advantages are
manyfold:
* We do not lock you in to any other particular programs or workflows. If a new
great assembler appears tomorrow, you can use it with Vamb
* Vamb is extremely flexible. Want to use it for plasmids? Sure. Homology-reduced genes?
Why not. Eukaryotes? Go for it. You can even disable depths-based binning by providing
dummy depths data. And you can do all these things, and more, by importing Vamb in
a Python session and playing around with its internals.
* Vamb is modular. Its contig parsing and encoding is distinct from its depth
calculation, which is distinct from its variational encoding, which is distinct from
its clustering. This is only possible because each part of Vamb is not opinionated
about _how_ the input was created, only _what_ the input is.