# VAMB snakemake workflow

This is a snakemake workflow that performs all the necessary steps to run Vamb. As input it takes paired end reads and individual sample _de novo_ assemblies, process them through best practice pipeline (multi-split), runs Vamb and finally CheckM to assess the quality of the bins. 

In short it will:

```
1. Filter contigs for 2000bp and rename them to conform with the multi-split workflow
2. Index resulting contig-file with minimap2 and create a sequence dictionary
3. Map reads with minimap2 to the combined contig set
4. Sort bam-files and create a jgi-count matrix by running individual samples in parallel 
5. Run Vamb to bin the contigs
6. Determine completeness and contamination of the bins using CheckM
```

The nice thing about using snakemake for this is that it will keep track of which jobs have finished and it allows the workflow to be run on different hardware such as a laptop, a linux workstation and a HPC facility (currently with qsub).

## Installation 
To run the workflow first install a Python3 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and herefter install the dependencies of Vamb.

Option 1 (global install):
```
 conda install -c conda-forge mamba
 mamba install -c conda-forge -c bioconda snakemake 
 mamba install -c pytorch pytorch torchvision cudatoolkit=10.2
 mamba install -c bioconda vamb=3.0.2
```

Option 2 (conda environment): If you rather want a specific conda environment for your Vamb installation you can do this. Here we make a conda environment called `vamb` and installs the required packages. To activate the environment use `conda activate vamb` and use `conda deactivate` to get back to your normal shell.

```
 conda install -n vamb -c conda-forge mamba
 mamba create -n vamb python=3.7
 mamba install -n vamb -c conda-forge -c bioconda snakemake
 mamba install -n vamb -c pytorch pytorch torchvision cudatoolkit=10.2
 mamba install -n vamb -c bioconda vamb=3.0.2
```

## Set up configuration with your data

To run the snakemake workflow you need to set up three files: the configuration file (`config.json`), a file with paths to your contig-files (`contigs.txt`) and a file with paths to your reads (`samples2data.txt`). Example files are included and described here for an example dataset of four samples: 

`contigs.txt` contains paths to each of the per-sample assemblies:
```
assemblies/contigs.sample_1.fna.gz
assemblies/contigs.sample_2.fna.gz
assemblies/contigs.sample_3.fna.gz
assemblies/contigs.sample_4.fna.gz
```

_`samples2data.txt` contains sample name, path to read-pair1 and path to read-pair2 (tab-separated):
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
   "vamb_params": "-o C -m 2000 --minfasta 500000"
}
```

For GPU usage: add `--cuda` to the `vamb_params`. If you are using qsub then also update `vamb_ppn` accordingly - e.g. on our system we exchange `"vamb_ppn": "10"` to `"vamb_ppn": "10:gpus=1"`.


## Example run

When running the workflow use snakemake, give it the maximum number of cores you want to use and the path to the configfile as well as the snakemake file. We add the flag `--use-conda` to make snakemake install the dependencies (Minimap2, Samtools, MetaBAT2, CheckM and Vamb) for you during runtime. You might have noticed we already installed Vamb above, but it is needed here as well if one installed a into a conda environment (option 2) and are using `--cluster`.  

```
snakemake --cores 20 --configfile config.json --snakefile /path/to/vamb/workflow/vamb.snake.conda.py --use-conda
```

If you want to use snakemake on a compute cluster using `qsub` we add the following `--cluster` option below.

```
snakemake --jobs 20 --configfile config.json --snakefile /path/to/vamb/workflow/vamb.snake.conda.py --use-conda --latency-wait 60 --cluster "qsub -l walltime={params.walltime} -l nodes=1:ppn={params.ppn} -l mem={params.mem}" 
```

Note 1: If you installed in a conda environment (option 2 above), remember to activate your conda environment using `conda activate vamb` before running the above commands.

Note 2: If you want to re-run with different parameters of VAMB you can change the _vamb_params_ in the config-file, but remember to rename the existing `vamb` folder as it will overwrite existing _vamb_ folder.

Please let us know if you have any issues and we can try to help out.

