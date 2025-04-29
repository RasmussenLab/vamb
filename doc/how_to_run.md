# Running Vamb
Most users will want to copy and change the commands from the quickstart section below.
Users with more advanced data, or who really wants to dig into Vamb to get the most out of Vamb should read the in-depth sections below.

First figure out what you want to run:
* Do you have contigs plus reads plus a taxonomic annotation of the contigs? Use __TaxVamb__
* Do you only have contigs plus reads and want a decent, fast binner? Use __Vamb__

We also support the now-obsolete __AVAMB__ binner. Its performance is in between TaxVamb and Vamb,
but it requires more compute than either.
I recommend new users run either TaxVamb or Vamb.

## Quickstart
The general workflow looks like this.
For more detailed information, see the documentation page on Vamb's inputs and outputs, as well as the page with tips on how to run Vamb.

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

## Running with test data
We provide example data under the "releases" section on the Vamb Github repository: https://github.com/RasmussenLab/vamb/releases/download/input_data/inputs.tar.gz

After downloading, extract its content:
```shell
$ tar -xzf inputs.tar.gz
```

This data is only for demonstrating the Vamb commands, and test running Vamb, and does not reflect a realistic metagenome. It is not suitable for benchmarking the accuracy of any binner.

The following commands makes use of these example files. You can substitute those files with your own in the commands.


### Vamb
Default command:

```shell
$ vamb bin default --outdir out1 --fasta contigs.fna.gz --abundance_tsv abundances.tsv
```

### TaxVamb
For TaxVamb, it's almost the same, but we also provide the taxonomy file:

```shell
$ vamb bin taxvamb --outdir out2 --fasta contigs.fna.gz --abundance_tsv abundances.tsv --taxonomy taxonomy.tsv
```

### Taxometer
Same default arguments as TaxVamb:

```shell
$ vamb taxometer --outdir out3 --fasta contigs.fna.gz --abundance_tsv abundances.tsv --taxonomy taxonomy.tsv
```

### AVAMB
See the README.md file in the `workflow_avamb` directory.

### Reducing the number of epochs for testing
For testing purposes, e.g. when running on the test data, it may be useful to reduce the number of training epochs, so Vamb finishes faster.
This will cause Vamb's models to be severely underfitted and perform terribly, so doing it is only recommended for testing.

* For Vamb: Add flags `-e 5 -q 2 3`
* For TaxVamb: Add flags `-e 5 -q 2 3 -pe 5`
* For Taxometer: Add flags `-pe 5`

## Explanation of command-line options
Each program in Vamb only has a subset of the following options.

* `-h, --help`: Print help and exit
* `--version`: Print version to stdout and exit
* `--outdir`: Output directory to create. Must not exist. Parent directory must exist.
* `-m`: Ignore contigs shorter than this value. Too short contigs have an unstable kmer composition
  and abundance signal, and therefore adds too much noise to the binning process.
* `-p` Number of threads to use. Note that Vamb has limited control over the number of threads used by
  its underlying libraries such as PyTorch, NumPy and BLAS. Although Vamb tries its best to limit the
  number of threads to the number specified, that might not always work.
* `--norefcheck`: Disable reference hash checking between composition, abundance and taxonomic inputs.
  See the section on reference hash checking in the input section.
* `--cuda`: Use a graphical processing unit for model training and clustering.
  Must have a CUDA-compatible version of PyTorch installed, and an NVIDIA GPU which supports CUDA. 
* `--seed`: Pass an integer seed for the random number generation. Vamb will use this seed to attempt reproducibility. Note that PyTorch does not support reproducible training of models, so passing this seed does not guarantee that Vamb will produce the same results from the same data.
* `--minfasta`: Output all bins with a total size (sum of contig lengths) greater than or equal to this
  number. The bins will be output in a directory called `bins` under the output directory, and each bin
  will be a FASTA file with the same name as the bin, suffixed by ".fna".
* `-o` Set binsplit separator. See the section on binsplitting in "tips for running Vamb" section for its meaning.
  If not passed, defaults to `C` if 'C' is present in all identifiers.
  To disable binsplitting, pass `-o` without an argument.
* `--no_predictor`: When running TaxVamb, if this flag is not set, TaxVamb will automatically run
  Taxometer when given an unrefined input taxonomy to refine it.
  Using a refined taxonomy usually improves the accuracy of TaxVamb.
* `--fasta`: FASTA input file. See section on Vamb inputs and outputs.
* `--composition`: NPZ composition input file. See section on Vamb inputs and outputs.
* `--bamdir`: Directory with BAM files to use for abundance. See section on Vamb inputs and outputs.
* `--abundance_tsv`: TSV file with precomputed abundances. See section on Vamb inputs and outputs.
* `--abundance`: NPZ abundance input file. See section on Vamb inputs and outputs.
* `--taxonomy`: TSV file with refined or unrefined taxonomy. See section on Vamb inputs and outputs.
