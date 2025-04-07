# Vamb
[![Read the Doc](https://readthedocs.org/projects/vamb/badge/?version=latest)](https://vamb.readthedocs.io/en/latest/)

Read the documentation on how to use Vamb here: https://vamb.readthedocs.io/en/latest/

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
  See the [Avamb README page](https://github.com/RasmussenLab/avamb/tree/avamb_new/workflow_avamb) for more information.
  [Link to article](https://doi.org/10.1038/s42003-023-05452-3).

And a taxonomy predictor:
* __Taxometer__: This tool refines arbitrary taxonomy predictions (e.g. from `mmseqs taxonomy`) using kmer composition and co-abundance.
  [Link to article](https://www.nature.com/articles/s41467-024-52771-y)

See also [our tool BinBencher.jl](https://github.com/jakobnissen/BinBencher.jl) for evaluating metagenomic bins when a ground truth is available,
e.g. for simulated data or a mock microbiome.

## Quickstart
For more details, and how to run on an example dataset [see the documentation.](https://vamb.readthedocs.io/en/latest/)

```shell
# Assemble your reads, one assembly per sample, e.g. with SPAdes
for sample in 1 2 3; do
    spades.py --meta ${sample}.{fw,rv}.fq.gz -t 24 -m 100gb -o asm_${sample};
done    

# Concatenate your assemblies, and rename the contigs to the naming scheme
# S{sample}C{original contig name}. This can be done with a script provided by Vamb
# in the vamb/src directory
python src/concatenate.py contigs.fna.gz asm_{1,2,3}/contigs.fasta

# Estimate sample-wise abundance by mapping reads to the contigs.
# Any mapper will do, but we recommend strobealign with the --aemb flag
mkdir aemb
for sample in 1 2 3; do
    strobealign -t 8 --aemb contigs.fna.gz ${sample}.{fw,rv}.fq.gz > aemb/${sample}.tsv;
done

# Create an abundance TSV file from --aemb outputs using the script in vamb/src dir
python src/merge_aemb.py aemb abundance.tsv

# Run Vamb using the contigs and the directory with abundance files
vamb bin default --outdir vambout --fasta contigs.fna.gz --abundance_tsv abundance.tsv
```
