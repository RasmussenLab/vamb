#!/usr/bin/env python
import re
import os
import from vamb.vambtools import concatenate_fasta

def get_config(name, default, regex):
    res = config.get(name, default).strip()
    m = re.match(regex, res)
    if m is None:
        raise ValueError(
            f"Config option \"{name}\" is \"{res}\", but must conform to regex \"{regex}\"")
    return res

# set configurations
INDEX_SIZE = get_config("index_size", "12G", r"[1-9]\d*[GM]$")
MM_MEM = get_config("minimap_mem", "35gb", r"[1-9]\d*gb$")
MM_PPN = get_config("minimap_ppn", "10", r"[1-9]\d*$")
MIN_CONTIG_SIZE = get_config("min_contig_size", "2000", r"[1-9]\d*$")
VAMB_MEM = get_config("vamb_mem", "20gb", r"[1-9]\d*gb$")
VAMB_PPN = get_config("vamb_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")
CHECKM_MEM = get_config("checkm_mem", "25gb", r"[1-9]\d*gb$")
CHECKM_PPN = get_config("checkm_ppn", "10", r"[1-9]\d*$")
SAMPLE_DATA = get_config("sample_data", "samples2data.txt", r".*")
CONTIGS = get_config("contigs", "contigs.txt", r".*")
VAMB_PARAMS = get_config("vamb_params", "-o C -m 2000 --minfasta 500000", r".*")
VAMB_PRELOAD = get_config("vamb_preload", "", r".*")

# Get output dir
OUTDIR_NAME = "vamb"
while os.path.exists(OUTDIR_NAME):
    n = 1 if OUTDIR_NAME == "vamb" else int(OUTDIR_NAME[4:]) + 1
    OUTDIR_NAME = "vamb" + str(n)

# parse if GPUs is needed #

vamb_threads, sep, vamb_gpus = VAMB_PPN.partition(":gpus=")
VAMB_PPN = vamb_threads
CUDA = len(vamb_gpus) > 0

## read in sample information ##

# read in sample2path
IDS = []
sample2path = {}
with open(SAMPLE_DATA) as file:
    for (id, fw, rv) in map(lambda line: line.strip().split('\t'), file):
        IDS.append(id)
        sample2path[id] = [fw, rv]

# read in list of per-sample assemblies
with open(CONTIGS) as file:
    contigs_list = list(filter(None, map(str.strip, file)))

## start of snakemake rules ##

# targets
rule all:
    input:
        "vamb/clusters.tsv",
        "vamb/checkm.results"

rule cat_contigs:
    input:
        contigs_list
    output:
        "contigs.flt.fna.gz"
    params:
        walltime="864000",
        nodes="1",
        ppn="1",
        mem="10gb"
    threads:
        1
    log:
        "log/contigs/catcontigs.log"
    conda:
        "envs/vamb.yaml"
    run:
        for path in input:
            if not os.path.isfile()
                raise FileNotFoundError(path)

        if os.path.exists(output[0]):
            raise FileExistsError(output[0])

        with gzip.open(output[0], "wt", compresslevel=3) as file:
            concatenate_fasta(
                file, input, minlength=int(MIN_CONTIG_SIZE), rename=True
            )

rule index:
    input:
        contigs = "contigs.flt.fna.gz"
    output:
        mmi = "contigs.flt.mmi"
    params:
        walltime="864000",
        nodes="1",
        ppn="1",
        mem="90gb"
    threads:
        1
    log:
        "log/contigs/index.log"
    conda: 
        "envs/minimap2.yaml"
    shell:
        "minimap2 -I {INDEX_SIZE} -d {output} {input} 2> {log}"

# This rule creates a SAM header from a FASTA file.
# We need it because minimap2 for truly unknowable reasons will write
# SAM headers INTERSPERSED in the output SAM file, making it unparseable.
# To work around this mind-boggling bug, we remove all header lines from
# minimap2's SAM output by grepping, then re-add the header created in this
# rule.
rule dict:
    input:
        contigs = "contigs.flt.fna.gz"
    output:
        dict = "contigs.flt.dict"
    params:
        walltime="864000",
        nodes="1",
        ppn="1",
        mem="10gb"
    threads:
        1
    log:
        "log/contigs/dict.log"
    conda:
        "envs/samtools.yaml"
    shell:
        "samtools dict {input} | cut -f1-3 > {output} 2> {log}"

rule minimap:
    input:
        fq = lambda wildcards: sample2path[wildcards.sample],
        mmi = "contigs.flt.mmi",
        dict = "contigs.flt.dict"
    output:
        bam = temp("mapped/{sample}.bam")
    params:
        walltime="864000",
        nodes="1",
        ppn=MM_PPN,
        mem=MM_MEM
    threads:
        int(MM_PPN)
    log:
        "log/map/{sample}.minimap.log"
    conda:
        "envs/minimap2.yaml"
    shell:
        # See comment over rule "dict" to understand what happens here
        "minimap2 -t {threads} -ax sr {input.mmi} {input.fq} -N 5"
        " | grep -v '^@'"
        " | cat {input.dict} - "
        " | samtools view -F 3584 -b - " # supplementary, duplicate read, fail QC check
        " > {output.bam} 2> {log}"

rule sort:
    input:
        "mapped/{sample}.bam"
    output:
        temp("mapped/{sample}.sort.bam")
    params:
        walltime="864000",
        nodes="1",
        ppn="2",
        mem="15gb",
        prefix="mapped/tmp.{sample}"
    threads:
        2
    log:
        "log/map/{sample}.sort.log"
    conda:
        "envs/samtools.yaml"
    shell:
        "samtools sort {input} -T {params.prefix} --threads 1 -m 3G -o {output} 2>{log}"

rule vamb:
    input:
        contigs = "contigs.flt.fna.gz",
        bamfiles=expand("mapped/{sample}.sort.bam", sample=IDS)
    output:
        "vamb/composition.npz",
        "vamb/abundance.npz",
        "vamb/clusters.tsv",
        "vamb/latent.npz",
        "vamb/log.txt",
        "vamb/model.pt"
    params:
        walltime="86400",
        nodes="1",
        ppn=VAMB_PPN,
        mem=VAMB_MEM,
        cuda="--cuda " if CUDA else ""
    log:
        "log/vamb/vamb.log"
    threads:
        int(VAMB_threads)
    conda:
        "envs/vamb.yaml"
    shell:
        "{VAMB_PRELOAD}"
        "vamb --outdir {OUTDIR_NAME} "
        "--fasta {input.contigs} "
        "--bamfiles {input.bamfiles} "
        "{params.cuda}"
        "-m {MIN_CONTIG_SIZE} {VAMB_PARAMS} 2> {log}"

rule checkm:
    input:
        "vamb/clusters.tsv"
    output:
        "vamb/checkm.results"
    params:
        walltime="86400",
        nodes="1",
        ppn=CHECKM_PPN,
        mem=CHECKM_MEM,
        bins="vamb/bins",
        outdir="vamb/checkm.outdir"
    log:
        "log/vamb/checkm.log"
    threads:
        int(CHECKM_PPN)
    conda:
        "envs/checkm.yaml"
    shell:
        "checkm lineage_wf -f {output} -t {threads} -x fna {params.bins} {params.outdir} 2> {log}"
