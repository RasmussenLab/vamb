# Variational Autoencoders for Metagenomic Binning (Vamb)

Vamb is a family of metagenomic binners which feeds kmer composition and abundance into a variational autoencoder and clusters the embedding to form bins.
Its binners perform excellently with multiple samples, and pretty good on single-sample data.

## Programs in Vamb
The Vamb package contains several programs, including three binners:

* __TaxVamb__: A semi-supervised binner that uses taxonomy information from e.g. `mmseqs taxonomy`.
  TaxVamb produces the best results, but requires you have run a taxonomic annotation workflow.
  [Link to article](https://doi.org/10.1101/2024.10.25.620172).
* __Vamb__: The original binner based on variational autoencoders.
  This has been upgraded significantly since its original release.
  Vamb strikes a good balance between speed and accuracy.
  [Link to article](https://doi.org/10.1038/s41587-020-00777-4).
* __Avamb__: An obsolete ensemble model based on Vamb and adversarial autoencoders. 
  Avamb has an accuracy in between Vamb and TaxVamb, but is more computationally demanding than either.
  We don't recommend running Avamb: If you have the compute to run it, you should instead run TaxVamb
  See the [Avamb README page](https://github.com/RasmussenLab/vamb/tree/master/workflow_avamb) for more information.
  [Link to article](https://doi.org/10.1038/s42003-023-05452-3).

And a taxonomy predictor:
* __Taxometer__: This tool refines arbitrary taxonomy predictions (e.g. from `mmseqs taxonomy`) using kmer composition and co-abundance.
  [Link to article](https://www.nature.com/articles/s41467-024-52771-y)

See also [our tool BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl) for evaluating metagenomic bins when a ground truth is available,
e.g. for simulated data or a mock microbiome.
