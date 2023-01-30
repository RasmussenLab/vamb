#!/usr/bin/env python

import re
import os
from vamb.vambtools import concatenate_fasta

def get_config(name, default, regex):
    res = config.get(name, default).strip()
    m = re.match(regex, res)
    if m is None:
        raise ValueError(
            f"Config option \"{name}\" is \"{res}\", but must conform to regex \"{regex}\"")
    return res

SNAKEDIR = os.path.dirname(workflow.snakefile)

# set configurations
CONTIGS = get_config("contigs", "contigs.txt", r".*") # each line is a contigs path from a given sample
SAMPLE_DATA = get_config("sample_data", "samples2data.txt", r".*") # each line is composed by 3 elements: sample id, forward_reads_path , backward_reads_path
INDEX_SIZE = get_config("index_size", "12G", r"[1-9]\d*[GM]$")
MIN_CONTIG_SIZE = get_config("min_contig_size", "2000", r"[1-9]\d*$")

MM_MEM = get_config("minimap_mem", "35gb", r"[1-9]\d*gb$")
MM_PPN = get_config("minimap_ppn", "10", r"[1-9]\d*$")
AVAMB_MEM = get_config("avamb_mem", "20gb", r"[1-9]\d*gb$")
AVAMB_PPN = get_config("avamb_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")

CHECKM_MEM = get_config("checkm2_mem", "10gb", r"[1-9]\d*gb$")
CHECKM_PPN = get_config("checkm2_ppn", "10", r"[1-9]\d*$")
CHECKM_MEM_r = get_config("checkm2_mem_r", "30gb", r"[1-9]\d*gb$")
CHECKM_PPN_r = get_config("checkm2_ppn_r", "30", r"[1-9]\d*$")


AVAMB_PARAMS = get_config("avamb_params","  -o C --minfasta 200000  -e 2 -q 1 --e_aae 2 --q_aae 2 -m 2000 --minfasta 200000 ", r".*")
AVAMB_PRELOAD = get_config("avamb_preload", "", r".*")

OUTDIR= get_config("outdir", "outdir_avamb", r".*")

try:
    os.makedirs(os.path.join(OUTDIR,"log"), exist_ok=True)
except FileExistsError:
    pass
except:
    raise


# parse if GPUs is needed #
avamb_threads, sep, avamb_gpus = AVAMB_PPN.partition(":gpus=")
AVAMB_PPN = avamb_threads
CUDA = len(avamb_gpus) > 0

## read in sample information ##

# read in sample2path
IDS = []
sample2path = {}
fh_in = open(SAMPLE_DATA, 'r')
for line in fh_in:
    line = line.rstrip()
    fields = line.split('\t')
    IDS.append(fields[0])
    sample2path[fields[0]] = [fields[1], fields[2]]

# read in list of per-sample assemblies
contigs_list = []
fh_in = open(CONTIGS, 'r')
for line in fh_in:
    line = line.rstrip()
    contigs_list.append(line)

# target
rule target_rule:
    input:
        os.path.join(OUTDIR,'avamb/tmp/workflow_finished_avamb.log')

checkpoint samples_with_bins:
    input:
        os.path.join(OUTDIR,"avamb/tmp/avamb_finished.log")
    output:
        os.path.join(OUTDIR,"avamb/tmp/samples_with_bins.txt")
    params:
        walltime="300",
        nodes="1",
        ppn="1",
        mem="1gb"
    threads:
        1
    shell:
        "find {OUTDIR}/avamb/bins/*/ -type d ! -empty |sed 's=.*bins/==g'  |sed 's=/==g'  > {output}"

def samples_with_bins_f(wildcards):
    # decision based on content of output file
    with checkpoints.samples_with_bins.get().output[0].open() as f:
        samples_with_bins = [sample.strip() for sample in f.readlines()]
        samples_with_bins_paths=expand(os.path.join(OUTDIR,"avamb/tmp/checkm2_all_{sample}_bins_finished.log"),sample=samples_with_bins)
        return samples_with_bins_paths

        
rule run_checkm2_per_sample_all_bins:
    input:
        bins_dir_sample=os.path.join(OUTDIR,"avamb/bins/{sample}"),
        out_dir_checkm2=os.path.join(OUTDIR,"avamb/tmp/checkm2_all")
    output:
        out_log_file=os.path.join(OUTDIR,"avamb/tmp/checkm2_all_{sample}_bins_finished.log")
    params:
        walltime="86400",
        nodes="1",
        ppn=CHECKM_PPN,
        mem=CHECKM_MEM
    threads:
        int(CHECKM_PPN)
    conda:
        "checkm2" 
    shell:
        #"checkm2  --help "
        "checkm2  predict --threads {threads} --input {input.bins_dir_sample}/*.fna --output-directory {input.out_dir_checkm2}/{wildcards.sample}" #2> {output.out_log_file}"

rule cat_checkm2_all:
    input:
        samples_with_bins_f
    output: 
        os.path.join(OUTDIR,"avamb/tmp/checkm2_finished.txt")
    params:
        walltime="86400",
        nodes="1",
        ppn="2",
        mem="5gb"
    threads:
        1
    shell:
        "touch {output}"
            

rule workflow_finished:
    input:
        os.path.join(OUTDIR,"avamb/tmp/checkm2_finished.txt")         
    output:
        os.path.join(OUTDIR,"avamb/tmp/workflow_finished_avamb.log")
    params:
        walltime="86400",
        nodes="1",
        ppn="1",
        mem="1gb"
    threads:
        1
    shell:
        "touch {output}"
