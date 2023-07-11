# taxVamb snakemake workflow

This is a snakemake workflow that performs all the necessary steps to run TaxVamb, as input it takes quality control filtered paired end reads and individual sample _de novo_ assemblies, process them through best practice pipeline (multi-split), obtains taxonomic annotations with MMseqs2, runs vaevae, and runs CheckM2 to assess the quality of the bins.

In short it will:

```
1. Filter contigs for 2000bp and rename them to conform with the multi-split workflow
2. Index resulting contig-file with minimap2 and create a sequence dictionary
3. Map reads with minimap2 to the combined contig set
4. Sort bam-files
5. Run MMeqs2
6. Run vaevae, and reclustering.
7. Determine completeness and contamination of the bins using CheckM2
```

The nice thing about using snakemake for this is that it will keep track of which jobs have finished and it allows the workflow to be run on different hardware such as a laptop, a linux workstation and a HPC facility (currently with qsub). Keep in mind that there are three different paths, (named directed acyclic graphs in snakemake), that can be executed by snakemake depending on the outputs generated during the workflow and complicating a bit the interpretation of the snakemake file. That's why we added some comments for each rule briefily explaining their purpose. Feel free to reach us if you encounter any problems.  

## Installation 
To run the workflow first install a Python3 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [mamba](https://mamba.readthedocs.io/en/latest/installation.html#fresh-install). TaxVamb uses CheckM2 to score the bins, unfortunately, due to some dependencies conflicts, CheckM2 can not be installed in the same environment than taxVamb, therefore a specific environment should be created for CheckM2 and for taxVamb:

```
 # Install taxVamb and mmseqs2 in taxVamb environment
 git clone https://github.com/sgalkina/vamb.git --branch mmseq-vae
 mamba create -n taxVamb python=3.9.16
 mamba env update -n taxVamb --file vamb/workflow_vaevae/envs/taxVamb.yaml
 conda activate taxVamb
 cd vamb && pip install -e . && cd ..
 # mmseqs relies on the gtdb_tk database, so if not present already, it has to be downloaded. 
 mmseqs databases GTDB databases/
 
# Install CheckM2 in checkm2 environment
 git clone https://github.com/chklovski/CheckM2.git 
 mamba create -n checkm2 python=3.8.15
 mamba env update -n checkm2 --file vamb/workflow_taxVamb/envs/checkm2.yml
 conda activate checkm2
 cd CheckM2  && git checkout e563159 && python setup.py install && cd ..
 checkm2 database --download
 conda deactivate
```
However, despite taxVamb and CheckM2 being in different environments, snakemake will be taking care of which is the right environment for each task. So now we should be ready to move forward and configure the input data to run our workflow. Installation should not take more than 30 minutes on a normal laptop, depending on your internet connexion and computer.

## Set up configuration with your data

To run the snakemake workflow you need to set up three files: the configuration file (`config.json`), a file with paths to your contig-files (`contigs.txt`) and a file with paths to your reads (`samples2data.tsv`). Example files are included and described here for an example dataset of four samples: 

`contigs.txt` contains paths to each of the per-sample assemblies:
```
assemblies/contigs.sample_1.fna.gz
assemblies/contigs.sample_2.fna.gz
assemblies/contigs.sample_3.fna.gz
assemblies/contigs.sample_4.fna.gz
```

`samples2data.tsv` contains sample name, path to read-pair1 and path to read-pair2 (tab-separated):
```
sample_1    reads/sample_1.r1.fq.gz    reads/sample_1.r2.fq.gz
sample_2    reads/sample_2.r1.fq.gz    reads/sample_2.r2.fq.gz
sample_3    reads/sample_3.r1.fq.gz    reads/sample_3.r2.fq.gz
sample_4    reads/sample_4.r1.fq.gz    reads/sample_4.r2.fq.gz

```

Then the configuration file (`config.json`). The first two lines points to the files containing contigs and read information; `index_size` is size of the minimap2 index(es); `min_contig_size` defines the minimum contig length considered for binning; `min_bin_size` defines the minimum bin length that will be written as fasta file; `min_identity` referes to the minimum read alignment threshold for the contig abundances calculation; `mem` and `ppn` shows the amount of memory and cores set aside for minimap2, vaevae, CheckM2 and CheckM2 dereplicate, `vaevae_params` gives the parameters used by vaevae, finally `outdir` points to the directory that wiil be created and where all output files will be stored. Here we tell it to use the multi-split approach (`-o C`) and to run the vaevae model. With regard to `index_size`  this is the amount of Gbases that minimap will map to at the same time. You can increase the size of this if you have more memory available on your machine (a value of `12G` can be run using `"minimap_mem": "12gb"`).

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.tsv",
   "index_size": "3G",
   "min_contig_size": "2000",
   "min_bin_size": "200000",
   "min_identity": "0.95",
   "minimap_mem": "15GB",
   "minimap_ppn": "15",
   "vaevae_mem": "50GB",
   "vaevae_ppn": "30",
   "checkm2_mem": "15GB",
   "checkm2_ppn": "15",
   "mmseq_mem": "260GB",
   "mmseq_ppn": "30",
   "vaevae_params": "  -l 64 -e 500 -q 25 75 150 -pe 100 -pq 25 75   --model vaevae  -o  C  ",
   "vaevae_preload": "",
   "outdir": "taxVamb_outdir",
   "min_comp": "0.9",
   "max_cont": "0.05"
}

```
## Example run

Workflow can be executed and taxVamb installation tested with our exampla input data, available in taxVamb zenodo's [upload](https://zenodo.org/record/7941203).

When running the workflow use snakemake, give it the maximum number of cores you want to use and the path to the configfile as well as the snakemake file. You can then run snakemake on a laptop or workstation as below - remember to activate the conda environment first before running snakemake.

```
conda activate taxVamb
snakemake --cores 20 --configfile /path/to/vamb/workflow_taxVamb/config.json --snakefile /path/to/vamb/workflow_taxVamb/taxVamb.snake.conda.smk --use-conda
```

If you want to use snakemake on a compute cluster using `qsub` we add the following `--cluster` option below. 
```

conda activate taxVamb
snakemake --jobs 20 --configfile /path/to/vamb/workflow_taxVamb/config.json --snakefile /path/to/vamb/workflow_taxVamb/taxVamb.snake.conda.smk --latency-wait 60 --use-conda --cluster "qsub -l walltime={params.walltime} -l nodes=1:ppn={params.ppn} -l mem={resources.mem} -e {log.e} -o {log.o}" 
```

Or if you want to use snakemake on a compute cluster using `slurm`, we add the following `--slurm` option below.
```

conda activate taxVamb
snakemake --slurm   --cores 20 --jobs 10 --configfile /path/to/vamb/workflow_taxVamb/config.json --snakefile /path/to/vamb/workflow_taxVamb/taxVamb.snake.conda.smk --latency-wait 60 --use-conda  
```

Note 1: If you want to re-run with different parameters of taxVamb you can change `vaevae_params` in the config-file, but remember to rename the  `outdir` configuration file entry, otherwise it will overwrite it.

## Outputs
taxVamb produces the following output files:
- `Final_bins`: folder containing the final set of bins by running CheckM2 on taxVamb bins per sample. This folder also contains a `quality_report.tsv` file with the completeness and contamination of the final set of near complete bins determined by CheckM2. Bins contamination and completeness thresholds can be modified by setting the `max_cont` and `min_comp` in the `config.json` file. 
- `Final_clusters.tsv`: final set of clusters file product of running CheckM2 on taxVamb bins per sample. `Final_bins` and `Final_clusters.tsv` have the same contigs per bin/cluster. Clusters contamination and completeness thresholds can be modified by setting the `max_cont` and `min_comp` in the `config.json` file.
-  `tmp`: folder containing the intermediate files and directories generated during the workflow:
   -  `mapped`: folder containing the sorted BAM files per sample, those BAM files were used by TaxVamb to generate the `abundance.npz` file, which expresses the contig abudnaces along samples. Those files might be quite large, so they can be deleted to free space, on the other hand they take a while to be generated, so might be worth keeping them.
   - `abundance.npz`: file aggregating contig abundances along samples from the BAM files. Using this as input instead of BAM files will skip re-parsing the BAM files, which take a significant amount of time.
   -  `contigs.flt.fna.gz`: a gunzipped fasta file containing contigs aggregated from all samples, renamed and filtered by the minimum contig length defined by `min_contig_size` in the `config.json`.
   -  `snakemake_tmp`: folder containing all log files from the snakemake workflow, specially from the binning part. Those files are the evidences that each workflow rule has been executed, and some of them contain the actual `stderr` `stdout` of the commands executed by the rules. If encountering any problem with snakemake, check them folder for debugging.   
-  `log`: folder containing all the log files from running the snakemake workflow in high computing cluster as well as the log files from the non binning part of the snakemake workflow, i.e. mapping, contig filtering, etc. If any snakemake rule fails to be executed, check the rule log for debugging.
- `vaevae`: folder containing the actual binning output files:
   - `bins`: folder containing the bins per sample obtained after clustering the VAEVAE latents and before reclustering.
   - `bins_reclustered`: folder containing the bins per sample obtained after clustering the VAEVAE latents and reclustering them accounting for single copy genes.
   - `composition.npz`: a Numpy .npz file that contain all kmer composition information computed by TaxVamb from the FASTA file. This can be provided to another run of TaxVamb to skip the composition calculation step.
   - `contignames`: text file containing a list of the contigs remaining after the minimum contig size allowed, and defined on the `min_contig_size` in the `config.json` file.
   - `lengths.npz`: Numpy object that contains the contig length, same order than the contignames.
   - `log.txt`: a text file with information about the TaxVamb run. Look here (and at stderr) if you experience errors.
   - `mask.npz`: considering the contigs abundances and tetra nucleotide frequencies computed per contig, some contigs might have been filtered out before binning, this numpy boolean object contains this masking.
   - `model.pt`: a file containing the trained VAEVAE model. When running TaxVamb from a Python interpreter, the VAEVAE can be loaded from this file to skip training.
   - `vae_clusters.tsv`: file generated by clustering the VAEVAE latent space, where each row is a sequence: Left column for the cluster (i.e bin) name, right column for the sequence name. You can create the FASTA-file bins themselves using the script in `src/create_fasta.py`



## Using a GPU to speed up taxVamb

Using a GPU can speed up taxVamb considerably - especially when you are binning millions of contigs. In order to enable it you need to make a couple of changes to the configuration file. Basically we need to add `--cuda` to the `vaevae_params` to tell taxVamb to use the GPU. Then if you are using the `--cluster` option, you also need to update `vaevae_ppn` accordingly - e.g. on our system (qsub) we exchange `"vaevae_ppn": "10"` to `"vaevae_ppn": "10:gpus=1"`. Therefore the `config.json` file looks like this if I want to use GPU acceleration:

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.tsv",
   "index_size": "3G",
   "min_contig_size": "2000",
   "min_bin_size": "200000",
   "min_identity": "0.95",
   "minimap_mem": "15GB",
   "minimap_ppn": "15",
   "vaevae_mem": "50GB",
   "vaevae_ppn": "30",
   "checkm2_mem": "15GB",
   "checkm2_ppn": "15",
   "mmseq_mem": "260GB",
   "mmseq_ppn": "30",
   "vaevae_params": "  -l 64 -e 500 -q 25 75 150 -pe 100 -pq 25 75   --model vaevae  -o  C  ",
   "vaevae_preload": "",
   "outdir": "taxVamb_outdir",
   "min_comp": "0.9",
   "max_cont": "0.05"
}

```

Note that I could not get `taxVamb` to work with `cuda` on our cluster when installing from bioconda. Therefore I added a line to preload cuda toolkit to the configuration file that will load this module when running `taxVamb`. 

Please let us know if you have any issues and we can try to help out.


