## Running Vamb
Most users will want to copy and change the commands from the quickstart section below.
Users with more advanced data, or who really wants to dig into Vamb to get the most out of Vamb should read the in-depth sections below.

### Quickstart
#### Vamb
```shell
$ # Assemble your reads, one assembly per sample, e.g. with SPAdes
$ for sample in 1 2 3; do
      spades.py --meta ${sample}.{fw,rv}.fq.gz  -t 24 -m 100gb -o asm_${sample};
  done    

$ # Concatenate your assemblies, and rename the contigs to the naming scheme
$ # S{sample}C{original contig name}. This can be done with a script provided by Vamb:
$ python concatenate.py contigs.fna.gz asm_{1,2,3}/contigs.fasta

$ # Estimate sample-wise abundance by mapping reads to the contigs.
$ # Any mapper will do, but we recommend strobealign with the --aemb flag
$ mkdir aemb
$ for sample in 1 2 3; do
      strobealign -t 8 --aemb contigs.fna.gz ${sample}.{fw,rv}.fq.gz > aemb/${sample}.tsv;
  done

$ # Paste the aemb files together to make a TSV file with given header
$ cat <(echo -e "contigname\t1\t2\t3") paste aemb/1.tsv <(cut -f 2 aemb/2.tsv) <(cut -f 2 aemb/3.tsv) > abundance.tsv

$ # Run Vamb using the contigs and the directory with abundance files
$ vamb bin default --outdir vambout --fasta contigs.fna.gz --abundance_tsv abundance.tsv
```

#### TaxVamb

The basic usecase of running TaxVAMB is the following:
```
vamb bin taxvamb --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam --taxonomy taxonomy.tsv
```

The FASTA and BAM files preprocessing intructions are the same as for running default VAMB. BAM files must be sorted by coordinate. You can sort BAM files with `samtools sort`.

In addition to that, TaxVAMB requires a taxonomic annotation file for all the contigs. 

### Taxonomy

Taxonomy annotations are text labels (e.g. "s__Alteromonas hispanica" or "Gammaproteobacteria") on all levels, sorted from domain to species, concatenated with ";". The taxonomic labels do not have to correspond to a particular version of any database, the only thing that is important is that the taxonomy levels are consistent within the dataset. E.g. if one contig has a taxonomic annotation
```
Bacteria;Bacillota;Clostridia;Eubacteriales;Lachnospiraceae;Roseburia;Roseburia hominis
```
then other contig annotations can be missing lower taxonomic levels, like
```
Bacteria;Bacillota;Clostridia
Bacteria;Bacillota;Bacilli;Bacillales
Bacteria;Pseudomonadota;Gammaproteobacteria;Moraxellales;Moraxellaceae;Acinetobacter;Acinetobacter sp. TTH0-4
```
but no annotations should violate the order that starts with the domain, e.g. that is incorrect:
```
Clostridia;Eubacteriales;Lachnospiraceae;Roseburia;Roseburia hominis
```

The taxonomy file is a tab-separated file with two named columns: `contigs` and `predictions`. For example, your `taxonomy.tsv` file could look like this:
```
contigs predictions
S18C13  Bacteria;Bacillota;Clostridia;Eubacteriales
S18C25  Bacteria;Pseudomonadota
S18C67  Bacteria;Bacillota;Bacilli;Bacillales;Staphylococcaceae;Staphylococcus
```

To use TaxVAMB and Taxometer on the output of the different classifiers, you need to convert their way to present taxonomic annotations to a format described above. One way to do it, is to use our tool __Taxconverter__ https://github.com/RasmussenLab/taxconverter. Taxconverter accepts MMSeqs2, Centrifuge, Kraken2, Metabuli or MetaMaps output files and returns the taxonomy file as described above.

#### AVAMB
[TODO]

### The different Vamb inputs
All modes of Vamb takes various _inputs_ and produces various _outputs_.
Currently, all modes take the following two central inputs:

* The kmer-composition of the sequence (the _composition_).
* The abundance of the contigs in each sample (the _abundance_).

For inputs that take significant time to produce, Vamb will serialize the parsed input to a file, such that future runs of Vamb can use that instead of re-computing it.

#### Composition
The composition is computed from the input contig file in FASTA format (the 'catalogue').
From command line, this looks like:

```
--fasta contigs.fna.gz
```

Where the catalogue may be either gzipped or a plain file.
When parsed, Vamb will write the composition in the output file `composition.npz`.
Future runs can then instead use:

```
--composition composition.npz
```

#### Abundance
The abundance may be computed from:
* A directory of BAM files generated the same way, except using any aligner that
  produces a BAM file, e.g. `minimap2`
* A TSV file with the header being "contigname" followed by one samplename per sample,
  and the values in the TSV file being precomputed abundances.
  These may be derived from `paste`ing together outputs from the tool `strobealign --aemb`

Thus, it can be specified as:
```
--abundance_tsv abundance.tsv
```
or
```
--bamdir dir_with_bam_files
```

When parsed, Vamb will produce the file `abundance.npz`, which can be used for future
Vamb runs instead:
```
--abundance abundance.npz
```

__Note:__ Vamb will check that the sequence names in the TSV / BAM files correspond
to the names in the composition.
