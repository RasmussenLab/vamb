import re
import os
import sys
from vamb.vambtools import concatenate_fasta, hash_refnames
import numpy as np
SNAKEDIR = os.path.dirname(workflow.snakefile)

sys.path.append(os.path.join(SNAKEDIR, 'src'))


def get_config(name, default, regex):
    res = config.get(name, default).strip()
    m = re.match(regex, res)
    if m is None:
        raise ValueError(
            f"Config option \"{name}\" is \"{res}\", but must conform to regex \"{regex}\"")
    return res


##### set configurations

CONTIGS = get_config("contigs", "contigs.txt", r".*") # each line is a contigs path from a given sample
SAMPLE_DATA = get_config("sample_data", "samples2data.txt", r".*") # each line is composed by 3 elements: sample id, forward_reads_path , backward_reads_path
INDEX_SIZE = get_config("index_size", "12G", r"[1-9]\d*[GM]$")
MIN_CONTIG_SIZE = int(get_config("min_contig_size", "2000", r"[1-9]\d*$"))
MIN_BIN_SIZE = int(get_config("min_bin_size", "200000", r"[1-9]\d*$"))
MIN_IDENTITY = float(get_config("min_identity", "0.95", r".*"))
MM_MEM = get_config("minimap_mem", "35gb", r"[1-9]\d*GB$")
MM_PPN = get_config("minimap_ppn", "10", r"[1-9]\d*$")
MMSEQ_MEM = get_config("mmseq_mem", "260gb", r"[1-9]\d*GB$")
MMSEQ_PPN = get_config("mmseq_ppn", "30", r"[1-9]\d*$")
MMSEQ_DB = get_config("mmseq_db", "",  r".*")
VAEVAE_PARAMS = get_config("vaevae_params"," -o C --minfasta 200000  ", r".*")
VAEVAE_PRELOAD = get_config("vaevae_preload", "", r".*")
VAEVAE_MEM = get_config("vaevae_mem", "20gb", r"[1-9]\d*GB$")
VAEVAE_PPN = get_config("vaevae_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")
RECLUST_MEM = get_config("reclust_mem", "20gb", r"[1-9]\d*GB$")
RECLUST_PPN = get_config("reclust_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")
CHECKM_MEM = get_config("checkm2_mem", "10gb", r"[1-9]\d*GB$")
CHECKM_PPN = get_config("checkm2_ppn", "10", r"[1-9]\d*$")
MIN_COMP = get_config("min_comp", "0.9", r".*")
MAX_CONT = get_config("max_cont", "0.05", r".*")
OUTDIR = get_config("outdir", "taxVamb_outdir", r".*")

try:
    os.makedirs(os.path.join(OUTDIR,"log"), exist_ok=True)
except FileExistsError:
    pass


# parse if GPUs is needed #
vaevae_threads, sep, vaevae_gpus = VAEVAE_PPN.partition(":gpus=")
VAEVAE_PPN = vaevae_threads
CUDA = len(vaevae_gpus) > 0

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

# target rule
rule all:
    input:
        os.path.join(OUTDIR,"log/workflow_finished_taxvamb.log")

# Filter contigs for 2000bp and rename them to conform with the multi-split workflow 
rule cat_contigs:
    input:
        contigs_list
    output:
        os.path.join(OUTDIR,"contigs.flt.fna.gz")
    params:
        path=os.path.join(os.path.dirname(SNAKEDIR), "src", "concatenate.py"),
        walltime="864000",
        nodes="1",
        ppn="1",
    resources:
        mem="5GB"
    threads:
        1
    log:
        o = os.path.join(OUTDIR,"log/contigs/catcontigs.o"),
        e = os.path.join(OUTDIR,"log/contigs/catcontigs.e")
    
    conda:
        "taxVamb"
    shell: "python {params.path} {output} {input} -m {MIN_CONTIG_SIZE}"





#  Run mmseq2 over the contigs
rule mmseq2:
    input:
        os.path.join(OUTDIR,"contigs.flt.fna.gz")
    output:
        taxonomytsv = os.path.join(OUTDIR,"tmp/mmseq/taxonomy.tsv"),
        out_log_file = os.path.join(OUTDIR,"tmp/mmseq_finished.log")
    
    params:
        walltime="86400",
        nodes="1",
        ppn=MMSEQ_PPN
    resources:
        mem=MMSEQ_MEM
    threads:
        int(MMSEQ_PPN)
    log:
        o=os.path.join(OUTDIR,'log','mmseq.out'),
        e=os.path.join(OUTDIR,'log','mmseq.err')

    conda: 
        "taxVamb" 
    shell:
        """
        mmseqs createdb  {input} {OUTDIR}/tmp/mmseq/qdb
        mmseqs taxonomy  {OUTDIR}/tmp/mmseq/qdb  {MMSEQ_DB}  {OUTDIR}/tmp/mmseq/taxonomy  {OUTDIR}/tmp/mmseq/tmp  --threads {threads}  --tax-lineage 1
        mmseqs createtsv {OUTDIR}/tmp/mmseq/qdb {OUTDIR}/tmp/mmseq/taxonomy {output.taxonomytsv}
        touch {output.out_log_file}
        """



# Index resulting contig-file with minimap2
rule index:
    input:
        contigs = os.path.join(OUTDIR,"contigs.flt.fna.gz")
    output:
        mmi = os.path.join(OUTDIR,"contigs.flt.mmi")
    params:
        walltime="864000",
        nodes="1",
        ppn="1"
    resources:
        mem="90GB"
    threads:
        1
    log:
        out_ind = os.path.join(OUTDIR,"log/contigs/index.log"),
        o = os.path.join(OUTDIR,"log/contigs/index.o"),
        e = os.path.join(OUTDIR,"log/contigs/index.e")
 

    conda: 
        "envs/minimap2.yaml"
    shell:
        "minimap2 -I {INDEX_SIZE} -d {output} {input} 2> {log.out_ind}"

# This rule creates a SAM header from a FASTA file.
# We need it because minimap2 for truly unknowable reasons will write
# SAM headers INTERSPERSED in the output SAM file, making it unparseable.
# To work around this mind-boggling bug, we remove all header lines from
# minimap2's SAM output by grepping, then re-add the header created in this
# rule.
rule dict:
    input:
        contigs = os.path.join(OUTDIR,"contigs.flt.fna.gz")
    output:
        dict = os.path.join(OUTDIR,"contigs.flt.dict")  
    params:
        walltime="864000",
        nodes="1",
        ppn="1"
    resources:
        mem="10GB"
    threads:
        1
    log:
        out_dict= os.path.join(OUTDIR,"log/contigs/dict.log"),
        o = os.path.join(OUTDIR,"log/contigs/dict.o"),
        e = os.path.join(OUTDIR,"log/contigs/dict.e")

    conda:
        "envs/samtools.yaml"
    shell:
        "samtools dict {input} | cut -f1-3 > {output} 2> {log.out_dict}"

# Generate bam files 
rule minimap:
    input:
        fq = lambda wildcards: sample2path[wildcards.sample],
        mmi = os.path.join(OUTDIR,"contigs.flt.mmi"),
        dict = os.path.join(OUTDIR,"contigs.flt.dict")
    output:
        bam = temp(os.path.join(OUTDIR,"mapped/{sample}.bam"))
    params:
        walltime="864000",
        nodes="1",
        ppn=MM_PPN
    resources:
        mem=MM_MEM
    threads:
        int(MM_PPN)
    log:
        out_minimap = os.path.join(OUTDIR,"log/map/{sample}.minimap.log"),
        o = os.path.join(OUTDIR,"log/map/{sample}.minimap.o"),
        e = os.path.join(OUTDIR,"log/map/{sample}.minimap.e")

    conda:
        "envs/minimap2.yaml"
    shell:
        # See comment over rule "dict" to understand what happens here
        "minimap2 -t {threads} -ax sr {input.mmi} {input.fq} -N 5"
        " | grep -v '^@'"
        " | cat {input.dict} - "
        " | samtools view -F 3584 -b - " # supplementary, duplicate read, fail QC check
        " > {output.bam} 2> {log.out_minimap}"

# Sort bam files
rule sort:
    input:
        os.path.join(OUTDIR,"mapped/{sample}.bam")
    output:
        os.path.join(OUTDIR,"mapped/{sample}.sort.bam")
    params:
        walltime="864000",
        nodes="1",
        ppn="2",
        prefix=os.path.join(OUTDIR,"mapped/tmp.{sample}")
    resources:
        mem="15GB"
    threads:
        2
    log:
        out_sort = os.path.join(OUTDIR,"log/map/{sample}.sort.log"),
        o = os.path.join(OUTDIR,"log/map/{sample}.sort.o"),
        e = os.path.join(OUTDIR,"log/map/{sample}.sort.e")
       
    conda:
        "envs/samtools.yaml"
    shell:
        "samtools sort {input} -T {params.prefix} --threads 1 -m 3G -o {output} 2> {log.out_sort}"

# Extract header lengths from a BAM file in order to determine which headers
# to filter from the abundance (i.e. get the mask)
rule get_headers:
    input:
        os.path.join(OUTDIR, "mapped", f"{IDS[1]}.sort.bam")
    output:
        os.path.join(OUTDIR,"abundances/headers.txt")
    params:
        walltime = "86400",
        nodes = "1",
        ppn = "1"
    resources:
        mem = "4GB"
    threads:
        1
    conda:
        "envs/samtools.yaml"
    log:
        head = os.path.join(OUTDIR,"log/abundance/headers.log"),
        o = os.path.join(OUTDIR,"log/abundance/get_headers.o"),
        e = os.path.join(OUTDIR,"log/abundance/get_headers.e")

    shell:
        "samtools view -H {input}"
        " | grep '^@SQ'"
        " | cut -f 2,3"
        " > {output} 2> {log.head} "
 
# Using the headers above, compute the mask and the refhash
rule abundance_mask:
    input:
        os.path.join(OUTDIR,"abundances/headers.txt")
    output:
        os.path.join(OUTDIR,"abundances/mask_refhash.npz")

    log:
        mask = os.path.join(OUTDIR,"log/abundance/mask.log"),
        o = os.path.join(OUTDIR,"log/abundance/mask.o"),
        e = os.path.join(OUTDIR,"log/abundance/mask.e")
    params:
        path = os.path.join(SNAKEDIR, "src", "abundances_mask.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "taxVamb"

    shell:
        """
        python {params.path} --h {input} --msk {output} --minsize {MIN_CONTIG_SIZE} 2> {log.mask}
        """


# For every sample, compute the abundances given the mask and refhash above
rule bam_abundance:
    input:
        bampath=os.path.join(OUTDIR,"mapped/{sample}.sort.bam"),
        mask_refhash=os.path.join(OUTDIR,"abundances/mask_refhash.npz")
    output:
        os.path.join(OUTDIR,"abundances/{sample}.npz")
    params:
        path = os.path.join(SNAKEDIR, "src", "write_abundances.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "taxVamb"
    log:
        bam = os.path.join(OUTDIR,"log/abundance/bam_abundance_{sample}.log"),
        o = os.path.join(OUTDIR,"log/abundance/bam_abundance.{sample}.o"),
        e = os.path.join(OUTDIR,"log/abundance/bam_abundance.{sample}.e")

    shell:
        """
        python {params.path} --msk {input.mask_refhash} --b {input.bampath} --min_id {MIN_IDENTITY} --out {output} 2> {log.bam}
        """
    
# Merge the abundances to a single Abundance object and save it
rule create_abundances:
    input:
        npzpaths=expand(os.path.join(OUTDIR,"abundances","{sample}.npz"), sample=IDS),
        mask_refhash=os.path.join(OUTDIR,"abundances","mask_refhash.npz")
    output:
        os.path.join(OUTDIR,"abundance.npz")
    params:
        path = os.path.join(SNAKEDIR, "src", "create_abundances.py"),
        abundance_dir = os.path.join(OUTDIR, "abundances"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "taxVamb"
    log:
        create_abs = os.path.join(OUTDIR,"log/abundance/create_abundances.log"),
        o = os.path.join(OUTDIR,"log/abundance/create_abundances.o"),
        e = os.path.join(OUTDIR,"log/abundance/create_abundances.e")

    shell:
        "python {params.path} --msk {input.mask_refhash} --ab {input.npzpaths} --min_id {MIN_IDENTITY} --out {output} 2> {log.create_abs} && "
        "rm -r {params.abundance_dir}"

# run vaevae
rule run_vaevae:
    input:
        contigs = os.path.join(OUTDIR,"contigs.flt.fna.gz"),
        abundance = os.path.join(OUTDIR,"abundance.npz"),
        taxonomytsv = os.path.join(OUTDIR,"tmp/mmseq/taxonomy.tsv"),
        mmsq_log = os.path.join(OUTDIR,"tmp/mmseq_finished.log")
    output:
        clusters = os.path.join(OUTDIR,"vaevae/vaevae_clusters.tsv"),
        latents = os.path.join(OUTDIR,"vaevae/vaevae_latent.npy"),
        vaevae_log = os.path.join(OUTDIR,"tmp/vaevae_finished.log")
    params:
        walltime="86400",
        nodes="1",
        ppn=VAEVAE_PPN,
        cuda="--cuda" if CUDA else ""
    resources:
        mem=VAEVAE_MEM
    threads:
        int(vaevae_threads)
    conda:
        "taxVamb" 
    log:
        o = os.path.join(OUTDIR,'log','run_vaevae.out'),
        e = os.path.join(OUTDIR,'log','run_vaevae.err'),
        
    shell:
        """
        #rm -rf {OUTDIR}/abundances
        #rm -rf {OUTDIR}/contigs.flt.dict
        #rm -f {OUTDIR}/contigs.flt.mmi
        rm -rf {OUTDIR}/vaevae 
        {VAEVAE_PRELOAD}
        mkdir -p {OUTDIR}/Final_bins
        vamb \
            --outdir {OUTDIR}/vaevae  \
            --fasta {input.contigs} \
            --rpkm {input.abundance} \
            --taxonomy {input.taxonomytsv} \
            --minfasta {MIN_BIN_SIZE} \
            -p {threads} \
            -m {MIN_CONTIG_SIZE} \
            {params.cuda} \
            {VAEVAE_PARAMS}
        
        touch {output.vaevae_log}
        
        """

rule reclustering:
    input:
        clusters = os.path.join(OUTDIR,"vaevae/vaevae_clusters.tsv"),
        latents = os.path.join(OUTDIR,"vaevae/vaevae_latent.npy"),
        contigs = os.path.join(OUTDIR,"contigs.flt.fna.gz"),
        abundance = os.path.join(OUTDIR,"abundance.npz"),
        vaevae_log = os.path.join(OUTDIR,"tmp/vaevae_finished.log")
        
        
    output:
        #reclustered_bins = directory(os.path.join(OUTDIR,"vaevae/bins_reclustered")),
        #hmm_markers = os.path.join(OUTDIR,"tmp/markers.hmmout"),
        recluster_log = os.path.join(OUTDIR,"tmp/reclustering_finished.log")

    resources:
        mem = RECLUST_MEM
    
    threads:
        int(RECLUST_PPN)

    conda:
        "taxVamb"
    log:
        o = os.path.join(OUTDIR,'log','run_reclustering.out'),
        e = os.path.join(OUTDIR,'log','run_reclustering.err')
    shell:
        """
        rm -rf {OUTDIR}/vaevae/bins_reclustered 
        vamb \
            --model reclustering \
            --latent_path {input.latents} \
            --clusters_path {input.clusters}  \
            --fasta {input.contigs} \
            --rpkm {input.abundance} \
            --outdir {OUTDIR}/vaevae/bins_reclustered \
            --minfasta {MIN_BIN_SIZE}    
        touch {output.recluster_log}        
        """
        # --hmmout_path {output.hmm_markers} \
# Evaluate in which samples bins were reconstructed
checkpoint samples_with_bins:
    input:        
        os.path.join(OUTDIR,"tmp/reclustering_finished.log")
    output:
        os.path.join(OUTDIR,"tmp/samples_with_bins.txt")
    params:
        walltime="300",
        nodes="1",
        ppn="1"
    resources:
        mem="1GB"
    log:
        o=os.path.join(OUTDIR,'log','samples_with_bins.out'),
        e=os.path.join(OUTDIR,'log','samples_with_bins.err')

    threads:
        1
    shell:
        "find {OUTDIR}/vaevae/bins_reclustered/bins/*/ -type d ! -empty | sed 's=.*bins/==g'  |sed 's=/==g'  > {output}"


def samples_with_bins_f(wildcards):
    # decision based on content of output file
    with checkpoints.samples_with_bins.get().output[0].open() as f:
        samples_with_bins = [sample.strip() for sample in f.readlines()]
        samples_with_bins_paths=expand(os.path.join(OUTDIR,"tmp/checkm2_all_{sample}_bins_finished.log"),sample=samples_with_bins)
        return samples_with_bins_paths

# Run CheckM2 for each sample with bins        
rule run_checkm2_per_sample_all_bins:
    output:
        out_log_file=os.path.join(OUTDIR,"tmp/checkm2_all_{sample}_bins_finished.log")
    params:
        walltime="86400",
        nodes="1",
        ppn=CHECKM_PPN
    resources:
        mem=CHECKM_MEM
    threads:
        int(CHECKM_PPN)
    log:
        o=os.path.join(OUTDIR,'log','checkm2_{sample}.out'),
        e=os.path.join(OUTDIR,'log','checkm2_{sample}.err')

    conda: 
        "checkm2" 
    shell:
        "checkm2 predict --threads {threads} --input {OUTDIR}/vaevae/bins_reclustered/bins/{wildcards.sample}/*.fna --output-directory {OUTDIR}/tmp/checkm2_all/{wildcards.sample} > {output.out_log_file}"

# this rule will be executed when all CheckM2 runs per sample finish, so it can move to the next step 
rule cat_checkm2_all:
    input:
        samples_with_bins_f
    output: 
        os.path.join(OUTDIR,"tmp/checkm2_finished.txt")
    params:
        walltime="86400",
        nodes="1",
        ppn="1"
    resources:
        mem="1GB"
    threads:
        1
    log:
        o=os.path.join(OUTDIR,'log','cat_checkm2.out'),
        e=os.path.join(OUTDIR,'log','cat_checkm2.err')

    shell:
        "touch {output}"

# Generate a 2 python dictionaries stored in json files:
#   - {bin : [completeness, contamination]}             
#   - {bin : bin_path}             
rule create_cluster_scores_bin_path_dictionaries:
    input:
        checkm2_finished_log_file = os.path.join(OUTDIR,"tmp/checkm2_finished.txt")
    output:
        cluster_score_dict_path = os.path.join(OUTDIR,"tmp/cs_d.json"),
        bin_path_dict_path = os.path.join(OUTDIR,"tmp/bp_d.json"),
    params:
        path = os.path.join(SNAKEDIR, "src", "create_cluster_scores_bin_path_dict.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "taxVamb"
    log:
        o=os.path.join(OUTDIR,'log','cs_bp_dicts.out'),
        e=os.path.join(OUTDIR,'log','cs_bp_dicts.err')

    shell:
        "python {params.path}  --s {OUTDIR}/tmp/checkm2_all --b {OUTDIR}/vaevae/bins_reclustered/bins --cs_d {output.cluster_score_dict_path} --bp_d {output.bin_path_dict_path} "




rule sym_NC_bins:
    input:
        cluster_score_dict_path = os.path.join(OUTDIR,"tmp/cs_d.json"),
        bin_path_dict_path = os.path.join(OUTDIR,"tmp/bp_d.json")
    output:
        nc_bins_dir = directory(os.path.join(OUTDIR,"Final_bins"))
    params:
        path = os.path.join(SNAKEDIR, "src", "symlink_nc_bins.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "taxVamb"
    log:
        ncs_syml=os.path.join(OUTDIR,'tmp','sym_final_bins_finished.log'),
        o=os.path.join(OUTDIR,'log','sym_NC_bins.out'),
        e=os.path.join(OUTDIR,'log','sym_NC_bins.err')

    shell:
        """
        python {params.path} --cs_d {input.cluster_score_dict_path} --bp_d {input.bin_path_dict_path} --min_comp {MIN_COMP}  --max_cont {MAX_CONT} -o {output.nc_bins_dir}
        touch {log.ncs_syml}
        """


# # Write final clusters from the Final_bins folder # NC_bins contains symlinks and bins from all samples in the same dir
# rule write_clusters_from_nc_folders:
#     input:
#         ncs_syml = os.path.join(OUTDIR,'tmp','sym_final_bins_finished.log')

#     output:
#         os.path.join(OUTDIR,"Final_clusters.tsv")
#     log:
#         log_fin = os.path.join(OUTDIR,"tmp/final_taxvamb_clusters_written.log"),
#         o=os.path.join(OUTDIR,'log','create_final_clusters.out'),
#         e=os.path.join(OUTDIR,'log','create_final_clusters.err')

#     params:
#         path = os.path.join(SNAKEDIR, "src", "write_clusters_from_final_bins.sh"),
#         walltime = "86400",
#         nodes = "1",
#         ppn = "1"
#     resources:
#         mem = "1GB"
#     threads:
#         1
#     conda:
#         "taxVamb"

#     shell:
#         "sh {params.path} -d {OUTDIR}/Final_bins -o {output} ;"
#         "touch {log.log_fin} "


# Rename and move some files and folders 
rule workflow_finished:
    input:
        #log_fin = os.path.join(OUTDIR,'tmp','final_taxvamb_clusters_written.log')
        ncs_syml=os.path.join(OUTDIR,'tmp','sym_final_bins_finished.log')
    output:
        os.path.join(OUTDIR,"log/workflow_finished_taxvamb.log")
    params:
        walltime = "86400",
        nodes = "1",
        ppn = "1"
    resources:
        mem = "1GB"
    threads:
        1
    log:
        o=os.path.join(OUTDIR,'log','workflow_finished_taxvamb.out'),
        e=os.path.join(OUTDIR,'log','workflow_finished_taxvamb.err')
    shell:
        """
        #rm -r {OUTDIR}/tmp/checkm2_all/*/protein_files
        
        #mkdir {OUTDIR}/tmp/snakemake_tmp/
        #mv {OUTDIR}/tmp/*log  {OUTDIR}/tmp/snakemake_tmp/
        #mv {OUTDIR}/tmp/*json  {OUTDIR}/tmp/snakemake_tmp/
        #mv {OUTDIR}/tmp/*tsv  {OUTDIR}/tmp/snakemake_tmp/
        #mv {OUTDIR}/tmp/*txt  {OUTDIR}/tmp/snakemake_tmp/
        #mv {OUTDIR}/tmp/checkm2_all  {OUTDIR}/tmp/snakemake_tmp/
        
        mv {OUTDIR}/abundance.npz {OUTDIR}/tmp/
        mv {OUTDIR}/mapped {OUTDIR}/tmp/
        mv {OUTDIR}/contigs.flt.fna.gz {OUTDIR}/tmp/
        touch {output}
        """
    

