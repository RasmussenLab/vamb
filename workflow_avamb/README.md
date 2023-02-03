# AVAMB snakemake workflow

This is a snakemake workflow that performs all the necessary steps to run Avamb, as input it takes quality control filtered paired end reads and individual sample _de novo_ assemblies, process them through best practice pipeline (multi-split), runs Vamb and Aamb, runs CheckM2 to assess the quality of the bins as well as to dereplicate the bins per sample and binner.

In short it will:

```
1. Filter contigs for 2000bp and rename them to conform with the multi-split workflow
2. Index resulting contig-file with minimap2 and create a sequence dictionary
3. Map reads with minimap2 to the combined contig set
4. Sort bam-files
5. Run Vamb and Aamb to bin the contigs, generating 3 sets of bins: vae_bins, aae_z_bins and aae_y_bins
6. Determine completeness and contamination of the bins using CheckM2
7. Dereplicate near complete bins, assuring contigs are present in one bin only.
```

The nice thing about using snakemake for this is that it will keep track of which jobs have finished and it allows the workflow to be run on different hardware such as a laptop, a linux workstation and a HPC facility (currently with qsub).

## Installation 
To run the workflow first install a Python3 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [mamba](https://mamba.readthedocs.io/en/latest/installation.html#fresh-install). Avamb uses CheckM2 to score the bins, unfortunately, due to some dependencies conflicts, CheckM2 can not be installed in the same environment than vamb, therefore a specific environment should be created for CheckM2 and for Avamb:

```
 # Install CheckM2 in checkm2 environment
 git clone https://github.com/chklovski/CheckM2.git 
 mamba create -n checkm2 
 mamba env update -n checkm2 --file CheckM2/checkm2.yml
 conda activate checkm2
 cd  CheckM2  && python setup.py install && cd ..
 checkm2 database --download
 conda deactivate
 # Install Avamb in avamb environment
 conda config --add channels conda-forge
 conda config --add channels bioconda
 mamba create -n avamb python=3.9
 mamba install -n avamb snakemake pip biopython netowrkx
 conda activate avamb
 pip install ordered-set 
 #pip install vamb # uncomment once it's merged with avamb
 git clone https://github.com/RasmussenLab/avamb.git -b avamb_new # remove once it's merged with vamb
 cd avamb && pip install -e . && cd .. # remove once it's merged with avamb
```
However, despite avamb and CheckM2 being in different environments, snakemake will be taking care of which is the right environment for each task. So now we should be ready to move forward and configure the input data to run our workflow.

## Set up configuration with your data

To run the snakemake workflow you need to set up three files: the configuration file (`config.json`), a file with paths to your contig-files (`contigs.txt`) and a file with paths to your reads (`samples2data.txt`). Example files are included and described here for an example dataset of four samples: 

`contigs.txt` contains paths to each of the per-sample assemblies:
```
assemblies/contigs.sample_1.fna.gz
assemblies/contigs.sample_2.fna.gz
assemblies/contigs.sample_3.fna.gz
assemblies/contigs.sample_4.fna.gz
```

`samples2data.txt` contains sample name, path to read-pair1 and path to read-pair2 (tab-separated):
```
sample_1    reads/sample_1.r1.fq.gz    reads/sample_1.r2.fq.gz
sample_2    reads/sample_2.r1.fq.gz    reads/sample_2.r2.fq.gz
sample_3    reads/sample_3.r1.fq.gz    reads/sample_3.r2.fq.gz
sample_4    reads/sample_4.r1.fq.gz    reads/sample_4.r2.fq.gz

```

Then the configuration file (`config.json`). The first two lines points to the files containing contigs and read information; `index_size` is size of the minimap2 index(es); `mem` and `ppn` shows the amount of memory and cores set aside for minimap2, AVAMB, CheckM2 and CheckM2 dereplicate, `avamb_params` gives the parameters used by AVAMB, finally `outdir` points to the directory that wiil be created and where all output files will be stored. Here we tell it to use the multi-split approach (`-o C`), to skip contigs smaller than 2kb (`-m 2000`) and to write all bins larger than 500kb as fasta (`--minfasta 500000`). With regard to `index_size`  this is the amount of Gbases that minimap will map to at the same time. You can increase the size of this if you have more memory available on your machine (a value of `12G` can be run using `"minimap_mem": "12gb"`).

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.txt",
   "index_size": "3G",
   "min_contig_size": "2000",
   "minimap_mem": "15gb",
   "minimap_ppn": "10",
   "avamb_mem": "10gb",
   "avamb_ppn": "10",
   "checkm2_mem": "30gb",
   "checkm2_ppn": "30",
   "checkm2_mem_r": "15gb",
   "checkm2_ppn_r": "15",
   "avamb_params": "-o C -m 2000 --minfasta 500000",
   "avamb_preload": "",
   "outdir":"avamb_outdir"
}

```
## Example run

When running the workflow use snakemake, give it the maximum number of cores you want to use and the path to the configfile as well as the snakemake file. You can then run snakemake on a laptop or workstation as below - remember to activate the conda environment first before running snakemake.

```
conda activate avamb
snakemake --cores 20 --configfile /path/to/vamb/workflow_avamb/config.json --snakefile /path/to/vamb/workflow_avamb/avamb.snake.conda.smk --use-conda
```

If you want to use snakemake on a compute cluster using `qsub` we add the following `--cluster` option below. 
```

conda activate avamb
snakemake --jobs 20 --configfile /path/to/vamb/workflow_avamb/config.json --snakefile /path/to/vamb/workflow_avamb/avamb.snake.conda.smk --latency-wait 60 --use-conda --cluster "qsub -l walltime={params.walltime} -l nodes=1:ppn={params.ppn} -l mem={params.mem}" 
```

Note 1: If you want to re-run with different parameters of AVAMB you can change  `avamb_params` in the config-file, but remember to rename the  `outdir` configuration file entry, otherwise it will overwrite it.


## Using a GPU to speed up Avamb

Using a GPU can speed up Avamb considerably - especially when you are binning millions of contigs. In order to enable it you need to make a couple of changes to the configuration file. Basically we need to add `--cuda` to the `avamb_params` to tell Avamb to use the GPU. Then if you are using the `--cluster` option, you also need to update `avamb_ppn` accordingly - e.g. on our system (qsub) we exchange `"avamb_ppn": "10"` to `"avamb_ppn": "10:gpus=1"`. Therefore the `config.json` file looks like this if I want to use GPU acceleration:

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.txt",
   "index_size": "3G",
   "minimap_mem": "15gb",
   "minimap_ppn": "10",
   "avamb_mem": "10gb",
   "avamb_ppn": "10:gpus=1",
   "checkm_mem": "25gb",
   "checkm_ppn": "10",   
   "avamb_params": "-o C -m 2000 --minfasta 500000 --outdir avamb --cuda",
   "avamb_preload": "module load cuda/toolkit/10.2.89;",
   "outdir":"avamb_outdir"
}
```

Note that I could not get `avamb` to work with `cuda` on our cluster when installing from bioconda. Therefore I added a line to preload cuda toolkit to the configuration file that will load this module when running `avamb`. 

Please let us know if you have any issues and we can try to help out.

