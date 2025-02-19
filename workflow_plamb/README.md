# PLAMB 
<information about the tool>

## Quick Start :rocket:
Clone the repository and install the package using conda
```
git clone <repository> -b <branch>
cd <repository>/workflow_plamb
conda env create -n workflow_plamb --file=envs/pipeline_conda.yaml
conda activate workflow_plamb
```

 To run the entire pipeline including assembly pass in a whitespace separated file containing the reads:
```
workflow_plamb --reads <read_file>  --output <output_directory>
```
The <read_file> could look like:

``` 
read1                          read2
im/a/path/to/sample_1/read1    im/a/path/to/sample_1/read2
im/a/path/to/sample_2/read1    im/a/path/to/sample_2/read2
```
:heavy_exclamation_mark: Notice the header names are required to be: read1 and read2  

To dry run the pipeline before pass in the --dryrun flag

To run the pipeline from allready assembled reads pass in a whitespace separated file containing the reads and the path to the spades assembly directories for each read pair.
```
workflow_plamb --reads_and_assembly_dir <reads_and_assembly_dir>  --output <output_directory>
```
The `reads_and_assembly_dir` file could look like:
``` 
read1                          read2                         assembly_dir                                           
im/a/path/to/sample_1/read1    im/a/path/to/sample_1/read2   path/sample_1/Spades_output  
im/a/path/to/sample_2/read1    im/a/path/to/sample_2/read2   path/sample_2/Spades_output          
```
 :heavy_exclamation_mark: Notice the header names are required to be: read1, read2 and assembly_dir  

The path to the SPades output directory (under the assembly_dir column in the above example) must contain the following 3 files which Spades produces: 
| Description                         | File Name from Spades                               |
|:------------------------------------|:----------------------------------------|
| The assembled contigs               | `contigs.fasta`                         |
| The simplified assembly graphs      | `assembly_graph_after_simplification.gfa` |
| A metadata file                     | `contigs.paths`                         |

## Output files
(Description of output files, maybe a nice tree)

## Advanced
### Using an allready downloaded geNomad database
To use an allready downloaded database, pass in a path to the genomad database with the --genomad_db argument

### Resources 

The pipeline can be configurated in: ``` config/config.yaml ```
Here the resources for each rule can be configurated as follows
```
spades:
  walltime: "15-00:00:00"
  threads: 16
  mem_gb: 60
```
if no resourcess are configurated for a rule the defaults will be used which are also defined in: ``` config/config.yaml ```  as
```
default_walltime: "48:00:00"
default_threads: 16
default_mem_gb: 50
```
If these exceed the resourcess available they will be scaled down to match the hardware available. 

### Running on cluster [! needs to be tested also w.r.t to specific snakemake version as they changed how to submit to a cluster through snakemake !]
You can extend the arguments passed to snakemake by the '--snakemake_arguments' flag
This can then be used to have snakemake submit jobs to a cluster.
on PBS this could like:
```
./cli.py <arguments> --snakemake_arguments \
    '--jobs 16 --max-jobs-per-second 5 --max-status-checks-per-second 5 --latency-wait 60 \
    --cluster "sbatch  --output={rule}.%j.o --error={rule}.%j.e \
    --time={resources.walltime} --job-name {rule} --cpus-per-task {threads} --mem {resources.mem_gb}G "'
```
### Running using snakemake CLI directly 
The pipeline can be run without using the CLI wrapper around snakemake. 
For using snakemake refer to the snakemake documentation: <https://snakemake.readthedocs.io/en/stable/>

#### Running from Reads using snakemake directly
To run the entire pipeline including assembly pass in a whitespace separated file containing the reads to snakemake using to the config flag in the snakemake CLI:
```
snakemake --use-conda --cores <number_of_cores> --snakefile <path_to_snakefile> --config read_file=<read_file>
```
The <read_file> could look like:

``` 
read1                          read2
im/a/path/to/sample_1/read1    im/a/path/to/sample_1/read2
im/a/path/to/sample_2/read1    im/a/path/to/sample_2/read2
```
:heavy_exclamation_mark: Notice the header names are required to be: read1 and read2  

#### Running from assembled reads using snakemake directly
To run the pipeline from allready assembled reads pass in a whitespace separated file containing the reads and the path to the spades assembly directories for each read pair to the config flag in snakemake.
```
snakemake --use-conda --cores <number_of_cores> --snakefile <path_to_snakefile> --config read_assembly_dir=<reads_and_assembly_dir_file>
```

The reads_and_assembly_dir_file could look like:
``` 
read1                          read2                         assembly_dir                                           
im/a/path/to/sample_1/read1    im/a/path/to/sample_1/read2   path/sample_1/Spades_output  
im/a/path/to/sample_2/read1    im/a/path/to/sample_2/read2   path/sample_2/Spades_output          
```
 :heavy_exclamation_mark: Notice the header names are required to be: read1, read2 and assembly_dir  

This assembly_dir directory filepath must contain the following 3 files which Spades produces: 
| Description                         | File Name from Spades                               |
|:------------------------------------|:----------------------------------------|
| The assembled contigs               | `contigs.fasta`                         |
| The simplified assembly graphs      | `assembly_graph_after_simplification.gfa` |
| A metadata file                     | `contigs.paths`                         |

#### Ressources for the different snakemake rules when using snakemake directly
To define resources for the specific snakemake rules either edit the snakemake.smk file directly or pass in a YAML config-file describing the resource use for each rule. 
```
snakemake <arguments> --configfile <config_file.yaml>
```
For a specification of the configfile refer to the ``` config/config.yaml ``` file.
Here the resources for each rule can be configurated as follows
```
spades:
  walltime: "15-00:00:00"
  threads: 16
  mem_gb: 60
```
if no resourcess are configurated for a rule the defaults will be used which are also defined in: ``` config/config.yaml ```  as
```
default_walltime: "48:00:00"
default_threads: 16
default_mem_gb: 50
```
#### Using an allready downloaded geNomad database 
To use an allready downloaded database, pass in a path to the genomad database with using the config flag
```
snakemake <arguments> --config genomad_database=<path_to_genomad_database>
```
