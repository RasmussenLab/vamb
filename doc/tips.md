# Tips for running Vamb

## Use the latest released version
Vamb generally gets faster and more accurate over time, so it's worth it to get the latest version.
Note that Conda releases are typically (far) behind pip releases, so I recommend installation using pip.

```{image} ../benchmark/benchmark.png
:alt: Vamb gets better over time
:width: 600px
```
_Figure 1: Newer Vamb releases are faster and more accurate.
Results are from binning the CAMI2 toy human microbiome short read gold standard assembly datasets, using the recommended multi-split workflow, and binned with Vamb using default settings._

## Garbage in, garbage out
For the best results when running Vamb, make sure the inputs to Vamb are as good as they can be.
In particular, the assembly process is a main bottleneck in the total binning workflow, so improving assembly
by e.g. preprocessing reads, using a better assembler, or switching to long read technology can make a big difference.

## Postprocess your bins
On principle, Vamb will bin every single input contig.
Currently, Vamb's bins are also _disjoint_, meaning each contig is present in only one bin.

Having to place every contig into a big, even those with a weak binning signal,
means that a large number of contigs will be binned poorly.
Often, these poor-quality contigs are put in a bin of their own, or with just one or two smaller contigs.
Practically speaking, this means _most bins produces by Vamb will be of poor quality_.

Hence, to use bins you can rely on, you will need to postprocess your bins:
* You may filter the bins by size, if you are only looking for organisms
  and not e.g. plasmids.
  For example, removing all bins < 250,000 bp in size will remove most poor quality bins,
  while keeping all bacterial genomes with a reasonable level of completeness.
* Using tools such as CheckM2 to score your bins, you can keep only the bins
  that pass some scoring criteria
* You may use the information in the `vae_clusters_metadata.tsv` file (see Output),
  and e.g. remove all clusters marked as "Fallback", below a certain size, or with a too
  high peak-valley ratio. However, this is only recommended for advanced users.

## How binsplitting works
In the recommended workflow, each sample is assembled independently, then the contigs are pooled
and binning together.
After Vamb have encoded the input features into the embedding (latent space), the embedding is clustered
to clusters.
The clusters thus may contain contigs from multiple samples, and may represent the same genome assembled
in different samples.
To obtain mono-sample bins from the clusters, the clusters then split by their sample of origin in a process we call binsplitting.
This reduces duplication in the output bins, and better preserves inter-sample diversity.

Binsplitting is done by looking at the identifiers (headers) of the contigs in the FASTA file:
They are assumed to be named according to the scheme `<sample identifier><separator><contig identifier>`,
where:
* The sample identifier uniquely identifies the same that the contig came from,
* The separator separates the sample- and contig identifier, and is guaranteed to not be contained in the sample identifier
* The contig identifier uniquely identifies the contig within the sample.
When using the provided `src/concatenate.py` script, the names conform to this scheme, being named e.g.
`S5C1042`, for sample 5, contig 1042. In this case, the binsplit separator is 'C'.

The separator can be set on command-line with the flag `-o`.
It defaults to 'C', if all headers contain a 'C'.
To disable binsplitting, pass `-o` without an argument.
