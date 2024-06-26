# Taxometer
Taxometer is a tool that improves taxonomic annotations from any taxonomic classifier for the set of contigs. Taxometer does not use a database or map contigs to reference sequences. Instead, the underlying neural network uses tetra-nucleotide frequencies and contigs abundances from a multi-sample metagenomic experiment to identify the contigs of the same origin and then completes and refines taxonomic annotations provided by any classifier using this information. 

For more explanation, motivation and benchmarks see the Taxometer preprint https://www.biorxiv.org/content/10.1101/2023.11.23.568413v1.

Any questions, bug reports or feature requests are most welcome.

## Installation
The code is a part of VAMB codebase and should be installed via cloning the Taxometer release branch:

```
git clone https://github.com/RasmussenLab/vamb -b taxometer_release
cd vamb
pip install -e .
```

In the future, it should become a part of the VAMB PyPi distribution. The updates will appear here and in the VAMB main README.

## Usage
The basic usecase of refining MMseqs2 annotations is the following:
```
vamb taxometer --outdir path/to/outdir --fasta /path/to/catalogue.fna.gz --bamfiles /path/to/bam/*.bam --taxonomy mmseqs_taxonomy.tsv
```

The FASTA and BAM files preprocessing intructions can be found here https://github.com/RasmussenLab/vamb#how-to-run-using-your-own-input-files 

Some important properties of the input files:
* BAM files must be sorted by coordinate. You can sort BAM files with `samtools sort`.
* For the command above, the default taxonomy format follows MMseqs2 output, which means that the file is tab-separated, has no headers, the 1st column contains contigs identifiers and the 9th column contains the taxonomy annotations. Taxonomy annotations are text labels (e.g. "s__Alteromonas hispanica" or "Gammaproteobacteria") on all levels, sorted from domain to species, concatenated with ";". In the paper we benchmarked annotations that use GTDB and NCBI identifiers. 

## Taxonomic labels
The taxonomic labels do not have to correspond to a particular version of any database, the only thing that is important is that the taxonomy levels are consistent within the dataset. E.g. if one contig has a taxonomic annotation
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

In the paper, besides MMseqs2, we benchmarked Centrifuge, Kraken2 and Metabuli tools. To use Taxometer on these classifiers, you need to convert their way to present taxonomic annotations to a format described above. One way to do it, is to use our tool __Taxconverter__ https://github.com/RasmussenLab/taxconverter. Taxconverter accepts Centrifuge, Kraken2 or Metabuli output files and returns the Taxometer-compatible, MMseqs2-like taxonomy file.

The taxonomy file can also be formatted differently. There has to be a column with contigs identifiers and a column with taxonomic labels in a format described above. For example, your `another_taxonomy.csv` file looks like this:
```
contigs,taxonomy
S18C13,Bacteria;Bacillota;Clostridia;Eubacteriales
S18C25,Bacteria;Pseudomonadota
S18C67,Bacteria;Bacillota;Bacilli;Bacillales;Staphylococcaceae;Staphylococcus
```

In this case, add the following arguments:
```
vamb taxometer --outdir path/to/outdir \ 
  --fasta /path/to/catalogue.fna.gz \
  --bamfiles /path/to/bam/*.bam \
  --taxonomy another_taxonomy.csv \
  --column_contigs contigs \
  --column_taxonomy taxonomy \
  --delimiter_taxonomy ,
```

For more information about the different arguments, run `vamb taxometer -h`

## Output
The `--outdir` folder will contain the `result_taxometer.csv` file of the following structure:
```
contigs,lengths,predictions,scores
S0C93,6527,d_Bacteria;p_Myxococcota;c_Polyangia;o_Polyangiales;f_Polyangiaceae;g_Sorangium;s_Sorangium cellulosum B,1.0;1.0;1.0;1.0;1.0;1.0;0.9999716
S0C112,5109,d_Bacteria;p_Proteobacteria;c_Gammaproteobacteria;o_Burkholderiales;f_Burkholderiaceae;g_Massilia;s_Massilia yuzhufengensis,0.9999998;0.9999998;0.9999998;0.9999998;0.9999998;0.9999957;0.9722711
S0C133,2128,d_Bacteria;p_Proteobacteria;c_Gammaproteobacteria;o_Burkholderiales;f_Burkholderiaceae;g_Variovorax;s_Variovorax paradoxus,1.0;1.0;1.0;0.9999999;0.9999999;0.9999999;0.9999908
```

Columns:
* __contigs__ - contig identifiers
* __lengths__ - contig lengths
* __predictions__ - taxonomic annotations predicted by Taxometer
* __scores__ - assigned scores on each taxonomic level, in the range [0.5, 1]

## Example with data

Taxometer can be tested on the pre-processed data (~4800 contigs from one sample of CAMI2 toy Oral dataset) provided as a part of this repository. After installing Taxometer, run from the current folder:

```
vamb taxometer --outdir taxometer_test --composition test/data/taxometer_data/composition.npz --rpkm test/data/taxometer_data/abundance.npz --taxonomy test/data/taxometer_data/taxonomy_oral_sample0.tsv
```

Running this command will take a few minutes on a personal computer. A neural network will be trained from scratch, followed by inference performed and the predictions stored. GPU access is not required to run this example. The `taxometer_test` folder will be created. The predicted taxonomy labels are at `taxometer_test/result_taxometer.csv`. The example output is also at `test/data/taxometer_data/result_taxometer_oral_sample0.csv`, but since the network is trained on the fly, the output can slightly vary between the runs.

## Links and references

To cite Taxometer, use the following reference:
```
@article {Kutuzova2023.11.23.568413,
	author = {Svetlana Kutuzova and Mads Nielsen and Pau Piera Lindez and Jakob Nybo Nissen and Simon Rasmussen},
	title = {Taxometer: Improving taxonomic classification of metagenomics contigs},
	elocation-id = {2023.11.23.568413},
	year = {2023},
	doi = {10.1101/2023.11.23.568413},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/11/23/2023.11.23.568413},
	eprint = {https://www.biorxiv.org/content/early/2023/11/23/2023.11.23.568413.full.pdf},
	journal = {bioRxiv}
}
```

Benchmarked taxonomic classifiers:
* __MMseqs2__ [Article](https://academic.oup.com/bioinformatics/article/37/18/3029/6178277?login=true) [Github](https://github.com/soedinglab/MMseqs2)
* __Metabuli__ [Article](https://www.biorxiv.org/content/10.1101/2023.05.31.543018v2) [Github](https://github.com/steineggerlab/Metabuli)
* __Centrifuge__ [Article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5131823/) [Github](https://github.com/infphilo/centrifuge)
* __Kraken2__ [Article](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1891-0) [Github](https://github.com/DerrickWood/kraken2)
* __MetaMaps__ [Article](https://www.nature.com/articles/s41467-019-10934-2) [Github](https://github.com/DiltheyLab/MetaMaps)
