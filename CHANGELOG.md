# Changelog

## v4.2.0
* Several details of the clustering algorithm has been rehauled.
  It now returns more accurate clusters and may be faster in some circumstances.
  However, GPU clustering may be significantly slower. (#198)
* Vamb now uses both relative and absolute abundances in the encoder, compared
  to only the relative ones before. This improves binning, especially when using
  a low number of samples (#210)
* Vamb now correctly uses a random seed on each invokation (#213)
* The `benchmark` module of Vamb has been removed, as it is superseded by our
  new benchmarking tool https://github.com/jakobnissen/VambBenchmarks.jl
* Improved error message when the reference hash doesn't match (#191)
* Various bugfixes
* The printing of epoch losses has been changed (#211)

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
