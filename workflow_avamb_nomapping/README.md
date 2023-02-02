# AVAMB snakemake workflow

This is a snakemake workflow that performs all the necessary steps to run Avamb, as input it takes sorted bam files per sample and the contigs file, process them through best practice pipeline (multi-split), runs Vamb and Aamb, runs CheckM2 to assess the quality of the bins as well as to dereplicate the bins per sample and binner.

In short it will:

```
1. Filter contigs for 2000bp and rename them to conform with the multi-split workflow
2. Run Vamb and Aamb to bin the contigs, generating 3 sets of bins: vae_bins, aae_z_bins and aae_y_bins
3. Determine completeness and contamination of the bins using CheckM2
4. Dereplicate near complete bins, assuring contigs are present in one bin only.
```

The nice thing about using snakemake for this is that it will keep track of which jobs have finished and it allows the workflow to be run on different hardware such as a laptop, a linux workstation and a HPC facility (currently with qsub).

## Installation 
To run the workflow first install a Python3 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html), make a conda environment named `avamb` and herefter install all the dependencies needed.

```
 conda config --append channels conda-forge
 conda config --append channels bioconda
 conda config --append channels pytorch
 mamba create -n avamb python=3.10
 conda install mamba
 mamba install -n avamb snakemake pytorch torchvision cudatoolkit=10.2 networkx # we might have to add checkm2 whenever it's possible to install it through conda (mamba)
 conda activate avamb
 pip install https://github.com/RasmussenLab/vamb/archive/v3.0.2.zip # that might to be updated 
 conda deactivate
```

We only have access to `avamb` and the other programs when we are inside this specific conda environment. We can use `conda activate avamb` to activate the environment (the name of shell changes and should say something with avamb) and `conda deactivate` to get out of it again. Therefore, if you find an error further down it could simply be because you have not activated the environment.  


## Set up configuration with your data

To run the snakemake workflow you need to set up three files: the configuration file (`config.json`), a contigs file wich contains all the contigs from all samples, and the sorted bam files per sample. CONTINUE FROM HERE Example files are included and described here for an example dataset of four samples: 

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
snakemake --cores 20 --configfile config.json --snakefile /path/to/avamb/workflow/avamb.snake.conda.py
```

If you want to use snakemake on a compute cluster using `qsub` we add the following `--cluster` and `--use-conda` options below. The latter is added because when running on the cluster we need to build the different program environments before starting the runs. Therefore, it can a little bit of time before it actually starts running the first jobs. NOT AVAILABLE SINCE CHECKM2 CANNOT BE INSTALLED WITH CONDA AT THE MOMENT (20/12/2022).

```
snakemake --jobs 20 --configfile config.json --snakefile /path/to/avamb/workflow/avamb.snake.conda.py --latency-wait 60 --use-conda --cluster "qsub -l walltime={params.walltime} -l nodes=1:ppn={params.ppn} -l mem={params.mem}" 
```

Note 1: If you installed in a conda environment (option 2 above), remember to activate your conda environment using `conda activate avamb` before running the above commands.

Note 2: If you want to re-run with different parameters of AVAMB you can change  `avamb_params` in the config-file, but remember to rename the  `outdir` configuration file entry, otherwise it will overwrite it.


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
