# VAMB snakemake workflow

This is a snakemake workflow that performs all the necessary steps to run Vamb. As input it takes paired end reads and individual sample _de novo_ assemblies, process them through best practice pipeline (multi-split), runs Vamb and finally CheckM to assess the quality of the bins. 

In short it will:

```
1. Filter contigs for 2000bp and rename them to conform with the multi-split workflow
2. Index resulting contig-file with minimap2 and create a sequence dictionary
3. Map reads with minimap2 to the combined contig set
4. Sort bam-files
5. Run Vamb to bin the contigs
6. Determine completeness and contamination of the bins using CheckM
```

The nice thing about using snakemake for this is that it will keep track of which jobs have finished and it allows the workflow to be run on different hardware such as a laptop, a linux workstation and a HPC facility (currently with qsub).

## Installation 
To run the workflow first install a Python3 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html), make a conda environment named `vamb` and herefter install all the dependencies needed.

```
 conda config --append channels conda-forge
 conda config --append channels bioconda
 conda config --append channels pytorch
 mamba create -n vamb python=3.10
 conda install mamba
 mamba install -n vamb snakemake "samtools>=1.8" minimap2 checkm-genome pytorch torchvision cudatoolkit=10.2
 conda activate vamb
 pip install https://github.com/RasmussenLab/vamb/archive/v3.0.2.zip
 conda deactivate
```

We only have access to `vamb` and the other programs when we are inside this specific conda environment. We can use `conda activate vamb` to activate the environment (the name of shell changes and should say something with vamb) and `conda deactivate` to get out of it again. Therefore, if you can an error further down it could simply be because you have not activated the environment.  


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

Then the configuration file (`config.json`). The first two lines points to the files containing contigs and read information; `index_size` is size of the minimap2 index(es); `mem` and `ppn` shows the amount of memory and cores set aside for minimap2 and VAMB and finally `vamb_params` gives the parameters used by VAMB. Here we tell it to use the multi-split approach (`-o C`), to skip contigs smaller than 2kb (`-m 2000`) and to write all bins larger than 500kb as fasta (`--minfasta 500000`). With regard to `index_size`  this is the amount of Gbases that minimap will map to at the same time. You can increase the size of this if you have more memory available on your machine (a value of `12G` can be run using `"minimap_mem": "35gb"`).

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.txt",
   "index_size": "3G",
   "minimap_mem": "15gb",
   "minimap_ppn": "10",
   "vamb_mem": "10gb",
   "vamb_ppn": "10",
   "checkm_mem": "25gb",
   "checkm_ppn": "10",   
   "vamb_params": "-o C -m 2000 --minfasta 500000 --outdir vamb"
}
```

## Example run

When running the workflow use snakemake, give it the maximum number of cores you want to use and the path to the configfile as well as the snakemake file. You can then run snakemake on a laptop or workstation as below - remember to activate the conda environment first before running snakemake.

```
conda activate vamb
snakemake --cores 20 --configfile config.json --snakefile /path/to/vamb/workflow/vamb.snake.conda.py
```

If you want to use snakemake on a compute cluster using `qsub` we add the following `--cluster` and `--use-conda` options below. The latter is added because when running on the cluster we need to build the different program environments before starting the runs. Therefore, it can a little bit of time before it actually starts running the first jobs.

```
snakemake --jobs 20 --configfile config.json --snakefile /path/to/vamb/workflow/vamb.snake.conda.py --latency-wait 60 --use-conda --cluster "qsub -l walltime={params.walltime} -l nodes=1:ppn={params.ppn} -l mem={params.mem}" 
```

Note 1: If you installed in a conda environment (option 2 above), remember to activate your conda environment using `conda activate vamb` before running the above commands.

Note 2: If you want to re-run with different parameters of VAMB you can change  `vamb_params` in the config-file, but remember to rename the existing `vamb` folder as it will overwrite existing `vamb` folder.


## Using a GPU to speed up Vamb

Using a GPU can speed up Vamb considerably - especially when you are binning millions of contigs. In order to enable it you need to make a couple of changes to the configuration file. Basically we need to add `--cuda` to the `vamb_params` to tell Vamb to use the GPU. Then if you are using the `--cluster` option, you also need to update `vamb_ppn` accordingly - e.g. on our system (qsub) we exchange `"vamb_ppn": "10"` to `"vamb_ppn": "10:gpus=1"`. Therefore the `config.json` file looks like this if I want to use GPU acceleration:

```
{
   "contigs": "contigs.txt",
   "sample_data": "samples2data.txt",
   "index_size": "3G",
   "minimap_mem": "15gb",
   "minimap_ppn": "10",
   "vamb_mem": "10gb",
   "vamb_ppn": "10:gpus=1",
   "checkm_mem": "25gb",
   "checkm_ppn": "10",   
   "vamb_params": "-o C -m 2000 --minfasta 500000 --outdir vamb --cuda",
   "vamb_preload": "module load cuda/toolkit/10.2.89;"
}
```

Note that I could not get `vamb` to work with `cuda` on our cluster when installing from bioconda. Therefore I added a line to preload cuda toolkit to the configuration file that will load this module when running `vamb`. 

Please let us know if you have any issues and we can try to help out.

