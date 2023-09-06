import re
import os
import sys
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


# set configurations
CONTIGS = get_config("contigs", "contigs.txt", r".*") # each line is a contigs path from a given sample
SAMPLE_DATA = get_config("sample_data", "samples2data.txt", r".*") # each line is composed by 3 elements: sample id, forward_reads_path , backward_reads_path
INDEX_SIZE = get_config("index_size", "12G", r"[1-9]\d*[GM]$")
MIN_CONTIG_SIZE = int(get_config("min_contig_size", "2000", r"[1-9]\d*$"))
MIN_BIN_SIZE = int(get_config("min_bin_size", "200000", r"[1-9]\d*$"))

MIN_IDENTITY = float(get_config("min_identity", "0.95", r".*"))

MM_MEM = get_config("minimap_mem", "35gb", r"[1-9]\d*GB$")
MM_PPN = get_config("minimap_ppn", "10", r"[1-9]\d*$")
AVAMB_MEM = get_config("avamb_mem", "20gb", r"[1-9]\d*GB$")
AVAMB_PPN = get_config("avamb_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")

CHECKM_MEM = get_config("checkm2_mem", "10gb", r"[1-9]\d*GB$")
CHECKM_PPN = get_config("checkm2_ppn", "10", r"[1-9]\d*$")
CHECKM_MEM_r = get_config("checkm2_mem_r", "30gb", r"[1-9]\d*GB$")
CHECKM_PPN_r = get_config("checkm2_ppn_r", "30", r"[1-9]\d*$")


AVAMB_PARAMS = get_config("avamb_params"," -o C --minfasta 200000  ", r".*")
AVAMB_PRELOAD = get_config("avamb_preload", "", r".*")

MIN_COMP = get_config("min_comp", "0.9", r".*")
MAX_CONT = get_config("max_cont", "0.05", r".*")

OUTDIR= get_config("outdir", "outdir_avamb", r".*")

try:
    os.makedirs(os.path.join(OUTDIR,"log"), exist_ok=True)
except FileExistsError:
    pass


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

# target rule
rule all:
    input:
        os.path.join(OUTDIR,'log/workflow_finished_avamb.log')

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
        "avamb"
    shell: "python {params.path} {output} {input} -m {MIN_CONTIG_SIZE}"

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
        "avamb"

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
        "avamb"
    log:
        bam = os.path.join(OUTDIR,"log/abundance/bam_abundance_{sample}.log"),
        o = os.path.join(OUTDIR,"log/abundance/{sample}.bam_abundance.o"),
        e = os.path.join(OUTDIR,"log/abundance/{sample}.bam_abundance.e")

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
        "avamb"
    log:
        create_abs = os.path.join(OUTDIR,"log/abundance/create_abundances.log"),
        o = os.path.join(OUTDIR,"log/abundance/create_abundances.o"),
        e = os.path.join(OUTDIR,"log/abundance/create_abundances.e")

    shell:
        "python {params.path} --msk {input.mask_refhash} --ab {input.npzpaths} --min_id {MIN_IDENTITY} --out {output} 2> {log.create_abs} && "
        "rm -r {params.abundance_dir}"

# Generate the 3 sets of clusters and bins
rule run_avamb:
    input:
        contigs=os.path.join(OUTDIR,"contigs.flt.fna.gz"),
        abundance=os.path.join(OUTDIR,"abundance.npz")
    output:
        outdir_avamb=directory(os.path.join(OUTDIR,"avamb")),
        clusters_aae_z=os.path.join(OUTDIR,"avamb/aae_z_clusters.tsv"),
        clusters_aae_y=os.path.join(OUTDIR,"avamb/aae_y_clusters.tsv"),
        clusters_vamb=os.path.join(OUTDIR,"avamb/vae_clusters.tsv"),
        contignames=os.path.join(OUTDIR,"avamb/contignames"),
        contiglenghts=os.path.join(OUTDIR,"avamb/lengths.npz")
    params:
        walltime="86400",
        nodes="1",
        ppn=AVAMB_PPN,
        cuda="--cuda" if CUDA else ""
    resources:
        mem=AVAMB_MEM
    threads:
        int(avamb_threads)
    conda:
        "avamb" 
    log:
        vamb_out=os.path.join(OUTDIR,"tmp/avamb_finished.log"),
        o=os.path.join(OUTDIR,'log','run_avamb.out'),
        e=os.path.join(OUTDIR,'log','run_avamb.err')
    shell:
        """
        rm -rf {OUTDIR}/abundances
        rm -rf {OUTDIR}/contigs.flt.dict
        rm -f {OUTDIR}/contigs.flt.mmi
        rm -rf {output.outdir_avamb} 
        {AVAMB_PRELOAD}
        vamb --outdir {output.outdir_avamb} --fasta {input.contigs} -p {threads} --rpkm {input.abundance} -m {MIN_CONTIG_SIZE} --minfasta {MIN_BIN_SIZE}  {params.cuda}  {AVAMB_PARAMS}
        mkdir -p {OUTDIR}/Final_bins
        mkdir -p {OUTDIR}/tmp/checkm2_all
        mkdir -p {OUTDIR}/tmp/ripped_bins
        touch {log.vamb_out}
        """

# Evaluate in which samples bins were reconstructed
checkpoint samples_with_bins:
    input:        
        os.path.join(OUTDIR,"tmp/avamb_finished.log")
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
        "find {OUTDIR}/avamb/bins/*/ -type d ! -empty |sed 's=.*bins/==g'  |sed 's=/==g'  > {output}"


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
        """
        checkm2 predict --threads {threads} --input {OUTDIR}/avamb/bins/{wildcards.sample}/*.fna --output-directory {OUTDIR}/tmp/checkm2_all/{wildcards.sample}
        touch {output.out_log_file}
        """

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
        cluster_score_dict_path_avamb = os.path.join(OUTDIR,"tmp/cs_d_avamb.json"),
        bin_path_dict_path_avamb = os.path.join(OUTDIR,"tmp/bp_d_avamb.json"),
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
        "avamb"
    log:
        o=os.path.join(OUTDIR,'log','cs_bp_dicts.out'),
        e=os.path.join(OUTDIR,'log','cs_bp_dicts.err')

    shell:
        "python {params.path}  --s {OUTDIR}/tmp/checkm2_all --b {OUTDIR}/avamb/bins --cs_d {output.cluster_score_dict_path_avamb} --bp_d {output.bin_path_dict_path_avamb} "

# Filter bins within the completeness and contamination thresholds defined and dereplicate them (min default cov=75% of the smallest bin, 100% identity)
rule run_drep_manual_vamb_z_y:
    input:
        cluster_score_dict_path_avamb=os.path.join(OUTDIR,"tmp/cs_d_avamb.json"),
        contignames=os.path.join(OUTDIR,"avamb/contignames"),
        contiglengths=os.path.join(OUTDIR,"avamb/lengths.npz"),
        clusters_aae_z=os.path.join(OUTDIR,"avamb/aae_z_clusters.tsv"),
        clusters_aae_y=os.path.join(OUTDIR,"avamb/aae_y_clusters.tsv"),
        clusters_vamb=os.path.join(OUTDIR,"avamb/vae_clusters.tsv")

    output:
        clusters_avamb_manual_drep=os.path.join(OUTDIR,"tmp/avamb_manual_drep_clusters.tsv")
    params:
        path=os.path.join(SNAKEDIR, "src", "manual_drep_JN.py"),
        walltime="86400",
        nodes="1",
        ppn="5"
    resources:
        mem="5GB"
    threads:
        5
    conda:
        "avamb"
    log:
        o=os.path.join(OUTDIR,'log','dereplication.out'),
        e=os.path.join(OUTDIR,'log','dereplication.err')

    shell:
        """
        python {params.path}  --cs_d  {input.cluster_score_dict_path_avamb} --names {input.contignames}\
        --lengths {input.contiglengths}  --output {output.clusters_avamb_manual_drep}\
        --clusters {input.clusters_aae_z} {input.clusters_aae_y} {input.clusters_vamb}\
        --comp {MIN_COMP} --cont {MAX_CONT}  --min_bin_size {MIN_BIN_SIZE} 
        """

# Evaluate if after the dereplication step (previous rule), still there are contigs present in more than one bin.
# If that's the case, for each pair of bins sharing a contig(s), create 2 extra bins that miss the intersecting contig(s), 
# which are called ripped bins since we ripped off contig(s) 
# If a contig is shared among three bins, it will automatically be removed from the bin where the contig represents the shortes
# length relative to the entire bin length.
checkpoint create_ripped_bins_avamb:
    input:
        path_avamb_manually_drep_clusters = os.path.join(OUTDIR,"tmp/avamb_manual_drep_clusters.tsv"),
        bin_path_dict_path = os.path.join(OUTDIR,"tmp/bp_d_avamb.json")
        
    output:
        path_avamb_manually_drep_clusters_ripped = os.path.join(OUTDIR,"tmp/avamb_manual_drep_not_ripped_clusters.tsv"),
        name_bins_ripped_file = os.path.join(OUTDIR,"tmp/bins_ripped_avamb.log")
    params:
        path = os.path.join(SNAKEDIR, "src", "rip_bins.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "5"
    resources:
        mem = "10GB"
    threads:
        5
    conda:
        "avamb"
    log:
        o=os.path.join(OUTDIR,'log','ripping.out'),
        e=os.path.join(OUTDIR,'log','ripping.err')

    shell: 
        """
        python {params.path} -r {OUTDIR}/avamb/ --ci {input.path_avamb_manually_drep_clusters}\
        --co  {output.path_avamb_manually_drep_clusters_ripped}  -l {OUTDIR}/avamb/lengths.npz\
        -n {OUTDIR}/avamb/contignames --bp_d {input.bin_path_dict_path} --br {OUTDIR}/tmp/ripped_bins\
        --bin_separator C --log_nc_ripped_bins {output.name_bins_ripped_file} 
        """
# If after run_drep_manual_vamb_z_y rule (dereplication rule), there are no contigs present in more than one bin,
# the dereplication is finished, so we can create the final clusters and the final bins.
rule nc_clusters_and_bins_from_mdrep_clusters_avamb:
    input:
        clusters_avamb_manual_drep = os.path.join(OUTDIR,"tmp/avamb_manual_drep_clusters.tsv"),   
        cluster_score_dict_path_avamb = os.path.join(OUTDIR,"tmp/cs_d_avamb.json")
    output:
        clusters_avamb_after_drep_disjoint = os.path.join(OUTDIR,"avamb/avamb_manual_drep_disjoint_clusters.tsv")
    log:
        log_fin=os.path.join(OUTDIR,"tmp/avamb_nc_clusters_and_bins_from_mdrep_clusters.log"),
        o=os.path.join(OUTDIR,'log','dereplicated_nc_bins.out'),
        e=os.path.join(OUTDIR,'log','dereplicated_nc_bins.err')

    params:
        path = os.path.join(SNAKEDIR, "src", "mv_bins_from_mdrep_clusters.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "avamb"
    shell:
        """
        python {params.path} --c {input.clusters_avamb_manual_drep} \
        --cf  {output.clusters_avamb_after_drep_disjoint}  --b {OUTDIR}/avamb/bins \
        --cs_d  {input.cluster_score_dict_path_avamb} --d  {OUTDIR}/Final_bins \
        --bin_separator C --comp {MIN_COMP}   --cont  {MAX_CONT}  2> {log.log_fin} 
        """
        
    
    
def ripped_bins_avamb_check_output_f(wildcards):
    # decision based on content of output file
    with checkpoints.create_ripped_bins_avamb.get().output[1].open() as f:
        n_ripped_bins=int(f.readline())
        if n_ripped_bins == 0 :
            return os.path.join(OUTDIR,"tmp/avamb_nc_clusters_and_bins_from_mdrep_clusters.log")
        else:
            return os.path.join(OUTDIR,"tmp/final_avamb_clusters_written.log")

# If any ripped bins have been generated on the create_ripped_bins_avamb rule, run CheckM2 on the ripped_bins
rule run_checkm2_ripped_bins_avamb:
    input:
        os.path.join(OUTDIR,"tmp/bins_ripped_avamb.log")
    output:
        os.path.join(OUTDIR,"tmp/ripped_bins/checkm2_out/quality_report.tsv")
    log:
        log_fin=os.path.join(OUTDIR,"tmp/checkm2_ripped_avamb_run_finished.log"),
        o=os.path.join(OUTDIR,'log','checkm2_ripped.out'),
        e=os.path.join(OUTDIR,'log','checkm2_ripped.err')

    params:
        walltime="86400",
        nodes="1",
        ppn=CHECKM_PPN_r
    resources:
        mem=CHECKM_MEM_r
    threads:
        int(CHECKM_PPN_r)
    conda:
        "checkm2" 
    shell:
        """
        checkm2 predict --threads {CHECKM_PPN_r} --input {OUTDIR}/tmp/ripped_bins  \
         --output-directory {OUTDIR}/tmp/ripped_bins/checkm2_out 
        touch {log.log_fin}
        """
# As explained in rule create_ripped_bins_avamb, contigs present in three bins were removed from the larger bin.
# This rule updates the quality scores after removing such contig(s).
rule update_cs_d_avamb:
    input:
        chck2_finished = os.path.join(OUTDIR,"tmp/checkm2_ripped_avamb_run_finished.log"),
        cluster_score_dict_path_avamb = os.path.join(OUTDIR,"tmp/cs_d_avamb.json")
    output:
        os.path.join(OUTDIR,"tmp/cs_d_avamb_updated.json")
    params:
        path = os.path.join(SNAKEDIR, "src", "update_cluster_scores_dict_after_ripping.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "4"
    resources:
        mem = "1GB"
    threads:
        4
    conda:
        "avamb" 
    log:
        o=os.path.join(OUTDIR,'log','update_cs_bp_dicts.out'),
        e=os.path.join(OUTDIR,'log','update_cs_bp_dicst.err'),
        log_fin = os.path.join(OUTDIR,"tmp/cs_d_avamb_updated.log")
    shell:
        """
        python {params.path} --s {OUTDIR}/tmp/ripped_bins/checkm2_out/quality_report.tsv \
         --cs_d {input.cluster_score_dict_path_avamb} --cs_d_o {output} > {log.log_fin}
         """

# For each pair of bins sharing a contig(s), determine which bin keeps the contig(s) by evaluating the decrease of the 
# bin scores. Move all dereplicated bins to the Final_bins folder. 
rule aggregate_nc_bins_avamb:
    input:
        cs_updated_log = os.path.join(OUTDIR,"tmp/cs_d_avamb_updated.log"),
        drep_clusters = os.path.join(OUTDIR,"tmp/avamb_manual_drep_clusters.tsv"),
        drep_clusters_not_ripped = os.path.join(OUTDIR,"tmp/avamb_manual_drep_not_ripped_clusters.tsv"),
        cluster_scores_dict_path_avamb = os.path.join(OUTDIR,"tmp/cs_d_avamb_updated.json"),
        bin_path_dict_path_avamb = os.path.join(OUTDIR,"tmp/bp_d_avamb.json"),
        checkm_finished_file = os.path.join(OUTDIR,"tmp/checkm2_ripped_avamb_run_finished.log")
    output:
        os.path.join(OUTDIR,"tmp/contigs_transfer_finished_avamb.log")
    params:
        path = os.path.join(SNAKEDIR, "src", "transfer_contigs_and_aggregate_all_nc_bins.py"),
        walltime = "86400",
        nodes = "1",
        ppn = "5"
    resources:
        mem = "10GB"
    threads:
        5
    conda:
        "avamb" 

    log:
        o=os.path.join(OUTDIR,'log','aggregate_final_ncs.out'),
        e=os.path.join(OUTDIR,'log','aggregate_final_ncs.err')

    shell:
        """
	    python {params.path} -r {OUTDIR} --c {input.drep_clusters} \
        --cnr {input.drep_clusters_not_ripped} --sbr {OUTDIR}/tmp/ripped_bins/checkm2_out/quality_report.tsv \
        --cs_d {input.cluster_scores_dict_path_avamb} --bp_d {input.bin_path_dict_path_avamb} \
        --br {OUTDIR}/tmp/ripped_bins -d Final_bins --bin_separator C \
        --comp {MIN_COMP}  --cont {MAX_CONT}  >  {output}
        rm -r {OUTDIR}/tmp/ripped_bins/checkm2_out/protein_files >>  {output}
        """

# Write final clusters from the Final_bins folder
rule write_clusters_from_nc_folders:
    input:
        contigs_transfered_log = os.path.join(OUTDIR,"tmp/contigs_transfer_finished_avamb.log")
               
    output:
        os.path.join(OUTDIR,"avamb/avamb_manual_drep_disjoint_clusters.tsv")
    log:
        log_fin=os.path.join(OUTDIR,"tmp/final_avamb_clusters_written.log"),
        o=os.path.join(OUTDIR,'log','create_final_clusters.out'),
        e=os.path.join(OUTDIR,'log','create_final_clusters.err')

    params:
        path = os.path.join(SNAKEDIR, "src", "write_clusters_from_dereplicated_and_ripped_bins.sh"),
        walltime = "86400",
        nodes = "1",
        ppn = "1"
    resources:
        mem = "1GB"
    threads:
        1
    conda:
        "avamb"
   
    shell:
        "sh {params.path} -d {OUTDIR}/Final_bins -o {OUTDIR} ;"
        "touch {log.log_fin} "
        
# Rename and move some files and folders 
rule workflow_finished:
    input:
        ripped_bins_avamb_check_output_f
    output:
        os.path.join(OUTDIR,"log/workflow_finished_avamb.log")
    params:
        walltime = "86400",
        nodes = "1",
        ppn = "1"
    resources:
        mem = "1GB"
    threads:
        1
    log:
        o=os.path.join(OUTDIR,'log','workflow_finished.out'),
        e=os.path.join(OUTDIR,'log','workflow_finished.err')
    shell:
        """
        rm -r {OUTDIR}/tmp/checkm2_all/*/protein_files
        
        rm -r {OUTDIR}/avamb/bins
        
        mkdir {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/tmp/*log  {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/tmp/*json  {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/tmp/*tsv  {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/tmp/*txt  {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/tmp/ripped_bins {OUTDIR}/tmp/snakemake_tmp/
        
        mv {OUTDIR}/tmp/checkm2_all  {OUTDIR}/tmp/snakemake_tmp/
        mv {OUTDIR}/avamb/avamb_manual_drep_disjoint_clusters.tsv  {OUTDIR}/Final_clusters.tsv
        mv {OUTDIR}/abundance.npz {OUTDIR}/tmp/
        mv {OUTDIR}/mapped {OUTDIR}/tmp/
        mv {OUTDIR}/contigs.flt.fna.gz {OUTDIR}/tmp/

        

        touch {output}
        """

