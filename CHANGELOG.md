# Changelog

## Unreleased - v4.0.0-DEV
Version 4 is a thorough rewrite of major parts of Vamb.
The code quality and test suite has gotten significant upgrades, making Vamb
more stable and robust to bugs.
Vamb version is slightly faster and produces slightly better bins than v3.
The user interface has only gotten slight changes.

### Breaking changes
* The official API of Vamb is now defined only in terms of its command-line
  interface. This means that from now on, Vamb can freely change and modify its
  internal functions, even in minor releases or patch releases.
  If you are using Vamb as a Python package, it means you should precisely
  specify the full version of Vamb used in order to ensure reproducibility.
* Benchmark procedure has been changed, so benchmark results are incompatible
  with results from v3.
  In v3, a complete bin was defined as the total set of covered basepairs in any
  contig from the input assembly. In v4, it's defined as the genome of origin,
  from where contigs are sampled.
  This new procedure is more fair, more intuitive and easier to compute.
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
* Vamb is now more easily installed via pip: `pip install vamb`. We have fixed
  a bunch of issues that caused installation problems.
* Added new flag: `--noencode`. With this flag, Vamb stops after producing the
  composition and depth outputs, and does not encode nor cluster.
  This can be used to produce the input data of Vamb to other clustering models.
* By default, Vamb gzip compresses FASTA files written using the `--minfasta`
  flag.

### Notable changes
* Vamb now uses `CoverM` internally to calculate abundances. This means it is
  significantly faster and more accurate than before.
  Thus, we no longer recommend users computing depths with MetaBAT2's JGI tool.
* Lots of bugfixes may have changed Vamb's behaviour in a backwards incompatible
  way for certain edge cases. For example, FASTA identifiers are now required to
  match the name specification in the SAM format to ensure the identifiers are
  the same in FASTA and BAM files.
