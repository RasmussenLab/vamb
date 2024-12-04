# Changelog

## v5.0.0 [UNRELEASED]
Version 5 is a major release that includes several breaking changes to the API,
as well as new types of models, improved binning accuracy, and more user
friendliness.

### Added
* Added the TaxVamb binner - a semi-supervised model that can augment binning
  using taxonomic assignment from e.g. mmseqs2 of some of the input contigs.
  TaxVamb is state-of-the-art, and significantly outperforms all other Vamb
  models when the taxonomic assignment is reasonably good.
  TaxVamb is available from command-line using `vamb bin taxvamb`
* Added the Taxometer annotation refiner. This program enhances taxonomic
  assignment of metagenomic contigs using composition and abundance.
  TaxVamb will automatically run Taxometer to increase accuracy.
  Taxometer is available from command-line using `vamb taxometer`
* [EXPERIMENTAL] Added reclustering functionality, which reclusters an existing
  binning using single-copy genes, using a technique inspired by the SemiBin2
  binner. This improves bacterial bins.
  We may remove this feature in future versions of Vamb.

### Breaking changes
* The command-line interface of Vamb has been changed, such that the different
  functionality should be used through subcommands. For example, the binners in
  Vamb are accesible through `vamb bin`.
  Also, a few command-line flags have been removed.
* All output files ending in `.tsv` is now actually in TSV format. Previously,
  Vamb did not include a header in the file, as the TSV format requires.
  In version 5, the header is included.
* The file `mask.npz` is no longer output, because the encoder no longer masks
  any sequences.
* The name of the output clusters files have been changed. When binsplitting is
  used, Vamb now outputs both the split, and the unsplit clusters.
  The name of the output files are now:
  	- `vae_clusters_split.tsv`
  	- `vae_clusters_unsplit.tsv`
  And similarly for e.g. `vaevae_clusters_split.tsv`.
  When binsplitting is not used, only the unsplit clusters are output.
* The `benchmark` module of Vamb has been removed, as it is superseded by our
  new benchmarking tool https://github.com/jakobnissen/BinBencher.jl
  
### Other changes
* Several details of the clustering algorithm has been rehauled.
  It now returns more accurate clusters and may be faster in some circumstances.
  However, GPU clustering may be significantly slower. (#198)
* Vamb now uses both relative and absolute abundances in the encoder, compared
  to only the relative ones before. This improves binning, especially when using
  a low number of samples (#210)
* Vamb now binsplits with `-o C` by default.
	- To disable binsplitting, pass `-o` without an argument
* Vamb now supports passing abundances in TSV format. This TSV can created very
  efficiently using the `strobealign` aligner with the `--aemb` flag.
* If passing abundances in BAM format, it is now recommended to pass in a
  directory with all the BAM files using the --bamdir flag, instead of using
  the old --bamfiles flag.
* Vamb no longer errors when the batch size is too large.
* Several errors and warnings have been improved:
	- The user is warned if any sequences are filtered away for falling below
	  the contig size cutoff (flag `-m`).
	- Improved the error message when the FASTA and BAM headers to not match.
	- Vamb now errors early if the binsplit separator (flag `-o`) is not found
	  in the parsed contig identifiers.
	  If the binsplit separator is not set explicitly and defaults to `-o C`,
	  Vamb will instead warn the user and disable binsplitting. 
* Vamb now writes its log to both stderr and to the logfile. Every line in the
  log is now timestamped, and formatted better.
* Vamb now outputs metadata about the unsplit clusters in the output TSV file
  `vae_clusters_metadata.tsv`.
* Vamb now correctly uses a random seed on each invokation (#213)
* Fixed various bugs and undoubtedly introduced some fresh ones.

## v4.1.3
* Fix a bug that resulting in poor clustering results (#179)

## v4.1.2
* Fix a bug in src/create_fasta.py
* Bugfix: Make seeding the RNG work from command line
* Bump compatible Cython version

## v4.1.1
* Create tmp directory in parsebam if needed for pycoverm (issue # 167)

## v4.1.0
* Fix typo in output AAE_Z cluster names. They are now called e.g. "aae_z_1"
  instead of "aae_z1"
* Clean up the directory structure of Avamb workflow.
* Fix the CheckM2 dependencies to allow CheckM2 to be installed
* Allow the Avamb workflow to be run on Slurm clusters
* Fix issue #161: Mismatched refhash when spaces in FASTA headers
* Allow setting the RNG seed from command line

## v4.0.1
* Fix Random.choice for Tensor on Python 3.11. See issue #148

## v4.0.0
Version 4 is a thorough rewrite of major parts of Vamb that has taken more than a year.
Vamb now ships with with an upgraded dual variational autoencoder (VAE) and
adversatial autoencoder (AAE) model, usable in a CheckM based workflow.
The code quality and test suite has gotten significant upgrades, making Vamb
more stable and robust to bugs.
Vamb version is slightly faster and produces better bins than v3.
The user interface has gotten limited changes.

### Breaking changes
* The official API of Vamb is now defined only in terms of its command-line
  interface. This means that from now on, Vamb can freely change and modify its
  internal functions, even in minor releases or patch releases.
  If you are using Vamb as a Python package, it means you should precisely
  specify the full version of Vamb used in order to ensure reproducibility.
* Benchmark procedure has been changed, so benchmark results are incompatible
  with results from v3. Benchmarking is now considered an implementation detail,
  and is not stable across releases.
* Vamb no longer outputs TNF, sequence names and sequence lengths as .npz files.
  Instead, it produces a `composition.npz` that contains all this information
  and more.
  As a consequence command-line options `--tnfs`, `--names` and `--lengths`
  have been removed, and replaced with the single `--composition` option.
* The output .npz array `rpkm.npz` has been changed in a backwards incompatible
  way. From version 4, the content of the output .npz files are considered an
  implementation detail.
* The depths input option `--jgi` has been removed. To use depths computed by
  an external program, construct an instance of the `Abundance` class from your
  depths and save it using its `.save` method to an `rpkm.npz` file.
  (though read the Notable changes section below).
  
### New features
* Vamb now included an optional AAE model along the VAE model.
  Users may run the VAE model, where it behaves similarly to v3, or run the mixed
  VAE/AAE model, in which both models will be run on the same dataset.
* The Snakemake workflow has been rehauled, and how defaults to using
  the VAE/AAE combined model, using CheckM to dereplicate.
* Vamb is now more easily installed via pip: `pip install vamb`. We have fixed
  a bunch of issues that caused installation problems.
* By default, Vamb gzip compresses FASTA files written using the `--minfasta`
  flag.

### Notable other changes
* Using the combined VAE-AAE workflow, the user can get significantly better bins.
* Vamb now uses `CoverM` internally to calculate abundances. This means it is
  significantly faster and more accurate than before.
  Thus, we no longer recommend users computing depths with MetaBAT2's JGI tool.
* Lots of bugfixes may have changed Vamb's behaviour in a backwards incompatible
  way for certain edge cases. For example, FASTA identifiers are now required to
  match the name specification in the SAM format to ensure the identifiers are
  the same in FASTA and BAM files.
