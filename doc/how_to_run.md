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

$ # Run Vamb using the contigs and the directory with abundance files
$ vamb bin default --outdir vambout --fasta contigs.fna.gz --aemb aemb
```

#### TaxVamb
[TODO]

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
* [recommended ]A directory containing TSV files obtained by mapping reads from
  each individual sample against the contig catalogue using `strobealign --aemb`
* A directory of BAM files generated the same way, except using any aligner that
  produces a BAM file, e.g. `minimap2`

Thus, it can be specified as:
```
--aemb dir_with_aemb_files
```
or
```
--bamdir dir_with_bam_files
```

When parsed, Vamb will produce the file `abundance.npz`, which can be used for future
Vamb runs instead:
```
--rpkm abundance.npz
```

__Note:__ Vamb will check that the sequence names in the TSV / BAM files correspond
to the names in the composition.

#### Taxonomy
[TODO]
