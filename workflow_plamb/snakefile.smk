import pandas as pd
import collections
import os
from pathlib import Path
import sys

THIS_FILE_DIR =config.get("src_dir")
THIS_FILE_DIR = Path("") if THIS_FILE_DIR is None else Path(THIS_FILE_DIR)

CONFIG_PATH = THIS_FILE_DIR / "config/config.yaml"
if CONFIG_PATH.exists():
    configfile: CONFIG_PATH

# The src directory for the files used in the snakemake workflow
PAU_SRC_DIR = THIS_FILE_DIR / "files_used_in_snakemake_workflow"  

# Get thee output_directory defined by the user
output_directory = config.get("output_directory") # Returns None if `output_directory` is not defined through CLI
# Set as empty path if not defined 
output_directory = Path("") if output_directory is None else Path(output_directory)
OUTDIR= output_directory / "outdir_plamb" 

# Define default threads/walltime/mem_gb based on configfile
default_walltime = config.get("default_walltime", "48:00:00")
default_threads = config.get("default_threads", 16)
default_mem_gb = config.get("default_mem_gb", 50)

# Functions to get the config-defined threads/walltime/mem_gb for a rule and if not defined the default
threads_fn = lambda rulename: config.get(rulename, {"threads": default_threads}).get("threads", default_threads) 
walltime_fn  = lambda rulename: config.get(rulename, {"walltime": default_walltime}).get("walltime", default_walltime) 
mem_gb_fn  = lambda rulename: config.get(rulename, {"mem_gb": default_mem_gb}).get("mem_gb", default_mem_gb) 

## Contig parameters
MIN_CONTIG_LEN = int(config.get("min_contig_len", 2000)) #get_config('min_contig_len', '2000', r'[1-9]\d*$'))

## TODO MISING
MAX_INSERT_SIZE_CIRC = 50 # 50 is deafult

## N2V parameters
N2V_NZ= config.get("n2v_nz", "weight") 
N2V_ED= config.get("n2v_ed", 128) 
N2V_WL= config.get("n2v_wl", 10) 
N2V_NW= config.get("n2v_nw", 50) 
N2V_WS= config.get("n2v_ws", 10) 
N2V_P= config.get("n2v_p", 0.1) 
N2V_Q= config.get("n2v_q", 2.0) 

NEIGHS_R=config.get("neighs_r", '0.05') 

## Binning parameters
PLAMB_PARAMS = config.get("plamb_params", ' -o C --minfasta 200000  ') 
PLAMB_PRELOAD = config.get("plamb_preload", "") 

# is GPU used ? #
CUDA = config.get("cuda", False)

# Set default paths for the SPades outputfiles 
# These need to be defined here as running the pipeline from allready assembled reads overwrite these values
contigs =  OUTDIR / "data/sample_{key}/spades_{id}/contigs.fasta"
contigs_paths =  OUTDIR / "data/sample_{key}/spades_{id}/contigs.paths"
assembly_graph = OUTDIR / "data/sample_{key}/spades_{id}/assembly_graph_after_simplification.gfa"

sample_id = dict()               
sample_id_path= dict() 
sample_id_path_assembly = dict()

if config.get("read_file") == None and config.get("read_assembly_dir") == None and config.get("should_install_genomad") == None:
    print("ERROR: read_file or read_assembly_dir not passed to snakemake as config. Define either. Eg. snakemake <arguments> --config read_file=<read file>. If in doubt refer to the README.md file")
    sys.exit()

# If the read_file is defined the pipeline will also run SPades and assemble the reads
if config.get("read_file") != None:
    df = pd.read_csv(config["read_file"], sep=r"\s+", comment="#")
    sample_id = collections.defaultdict(list)
    sample_id_path = collections.defaultdict(dict)
    for id, (read1, read2) in enumerate(zip(df.read1, df.read2)):
        id = f"sample{str(id)}"
        sample = "Plamb_Ptracker"
        sample_id[sample].append(id)
        sample_id_path[sample][id] = [read1, read2]

# If read_assembly dir is defined the pipeline will run user defined SPades output files
if config.get("read_assembly_dir") != None:
    df = pd.read_csv(config["read_assembly_dir"], sep=r"\s+", comment="#")
    sample_id = collections.defaultdict(list)
    sample_id_path = collections.defaultdict(dict)
    sample_id_path_assembly = collections.defaultdict(dict)
    for id, (read1, read2, assembly) in enumerate(zip( df.read1, df.read2, df.assembly_dir)):
        id = f"sample{str(id)}"
        sample = "Plamb_Ptracker"
        sample_id[sample].append(id)
        sample_id_path[sample][id] = [read1, read2]
        sample_id_path_assembly[sample][id] = [assembly]

    # Setting the output paths for the user defined SPades files
    contigs =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "contigs.fasta"
    assembly_graph  =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "assembly_graph_after_simplification.gfa"
    contigs_paths  =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "contigs.paths"

read_fw_after_fastp = lambda wildcards: sample_id_path[wildcards.key][wildcards.id][0]
read_rv_after_fastp =  lambda wildcards: sample_id_path[wildcards.key][wildcards.id][1]


try: # TODO why?
    os.makedirs(os.path.join(OUTDIR,'log'), exist_ok=True)
except FileExistsError:
    pass

rule all:
    input:
        expand(os.path.join(OUTDIR, "{key}", 'log/run_vamb_asymmetric.finished'), key=sample_id.keys()),
        expand(os.path.join(OUTDIR,"{key}",'vamb_asymmetric','vae_clusters_graph_thr_0.75_candidate_plasmids.tsv'),key=sample_id.keys()),
        expand(os.path.join(OUTDIR,"{key}",'log/run_geNomad.finished'), key=sample_id.keys()),


# If the genomad database is given as an argument don't download it again
# This works boths from when the tool is called from the CLI wrapper and from snakemake if the config is extended setting the genomad_database variable
if config.get("genomad_database") is not None:
    geNomad_db = config.get("genomad_database")
else: 
    geNomad_db = protected(THIS_FILE_DIR / "genomad_db" / "genomad_db"),

    rulename = "download_genomad_db"
    rule download_genomad_db:
        output:
            geNomad_db_path = directory(geNomad_db),
            dir_geNomad_db = directory(THIS_FILE_DIR / "genomad_db"),
            geNomad_db = protected(THIS_FILE_DIR / "genomad_db" / "genomad_db" / "genomad_db"),
        threads: threads_fn(rulename)
        resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
        conda: THIS_FILE_DIR / "envs/genomad.yaml"
        shell:
            """
            mkdir -p {output.dir_geNomad_db}
            genomad download-database {output.dir_geNomad_db}
            """

rulename = "spades"
rule spades:
    input:
       fw = read_fw_after_fastp, 
       rv = read_rv_after_fastp, 
    output:
       outdir = directory(OUTDIR / "data/sample_{key}/spades_{id}"),
       outfile = OUTDIR / "data/sample_{key}/spades_{id}/contigs.fasta",
       graph = OUTDIR / "data/sample_{key}/spades_{id}/assembly_graph_after_simplification.gfa",
       graphinfo  = OUTDIR / "data/sample_{key}/spades_{id}/contigs.paths",
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", "log/") + "{key}_{id}_" + rulename
    conda: THIS_FILE_DIR / "envs/spades_env.yaml"
    shell:
       "spades.py --meta "
       "-t {threads} -m 180 "
       "-o {output.outdir} -1 {input.fw} -2 {input.rv} " 
       "-t {threads} --memory {resources.mem_gb} > {log} " 

rulename = "rename_contigs"
rule rename_contigs:
    input:
        contigs,
    output:
        OUTDIR / "data/sample_{key}/spades_{id}/contigs.renamed.fasta"
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", "log/") + "{key}_{id}_" + rulename
    shell:
        """
        sed 's/^>/>S{wildcards.id}C/' {input} > {output} 2> {log}
        """

rulename="cat_contigs"
rule cat_contigs:
    input: lambda wildcards: expand(OUTDIR / "data/sample_{key}/spades_{id}/contigs.renamed.fasta", key=wildcards.key, id=sample_id[wildcards.key]),
    output: OUTDIR / "data/sample_{key}/contigs.flt.fna.gz"
    threads: threads_fn(rulename)
    params: script =  PAU_SRC_DIR / "concatenate.py"
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell: 
        "python {params.script} {output} {input} --keepnames 2> {log} "  # TODO should filter depending on size????

rulename = "get_contig_names"
rule get_contig_names:
    input:
        OUTDIR / "data/sample_{key}/contigs.flt.fna.gz"
    output: 
        OUTDIR / "data/sample_{key}/contigs.names.sorted"
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    shell:
        "zcat {input} | grep '>' | sed 's/>//' > {output} 2> {log} "

rulename = "Strobealign_bam_default"
rule Strobealign_bam_default:
        input: 
            fw = read_fw_after_fastp,
            rv = read_rv_after_fastp,
            contig = OUTDIR /"data/sample_{key}/contigs.flt.fna.gz",
            # fw = lambda wildcards: sample_id_path[wildcards.key][wildcards.value][0],
            # rv = lambda wildcards: sample_id_path[wildcards.key][wildcards.value][1],
            # contig = "results/{key}/contigs.flt.fna",
        output:
            OUTDIR / "data/sample_{key}/mapped/{id}.bam"
        threads: threads_fn(rulename)
        resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
        benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
        log: config.get("log", "log/") + "{key}_{id}_" + rulename
        conda: THIS_FILE_DIR / "envs/strobe_env.yaml"
        shell:
            """
            strobealign -t {threads} {input.contig} {input.fw} {input.rv} > {output} 2> {log}
            """

 # Sort bam files
rulename="sort"
rule sort:
    input:
        OUTDIR / "data/sample_{key}/mapped/{id}.bam",
    output:
        OUTDIR / "data/sample_{key}/mapped_sorted/{id}.bam.sort",
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", "log/") + "{key}_{id}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
	samtools sort --threads {threads} {input} -o {output} 
	samtools index {output}
	"""

## The next part of the pipeline is composed of the following steps:
# (Despite the steps are numerated, some of the order might change)
#   0. Look for contigs circularizable 
#   1. Align contigs all against all 
#   2. Generate per sample assembly graphs from gfa assembly graphs
#   3. Run n2v on the per sample assembly graphs 
#   4. Extract hoods from assembly graphs n2v embeddings per sample 
#   5. Connect hoods with alignment graphs
#   6. Run vamb to merge the hoods
#   7. Classify bins/clusters into plasmid/organism/virus bins/clusters


# 0.Look for contigs circularizable 
rulename = "circularize"
rule circularize:
    input:
        # dir_bams=BAMS_DIR
        bamfiles = lambda wildcards: expand(str(OUTDIR) +  "/data/sample_{key}/mapped_sorted/{id}.bam.sort", key=f"{wildcards.key}", id=sample_id[wildcards.key]),
    output:
        os.path.join(OUTDIR,'{key}','tmp','circularisation','max_insert_len_%i_circular_clusters.tsv.txt'%MAX_INSERT_SIZE_CIRC),
        os.path.join(OUTDIR,'{key}','log/circularisation/circularisation.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'circularisation.py'),
        dir_bams = lambda wildcards: expand(OUTDIR / "data/sample_{key}/mapped_sorted", key=wildcards.key)
    threads: threads_fn(rulename),
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    log: config.get("log", "log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --dir_bams {params.dir_bams} --outcls {output[0]} --max_insert {MAX_INSERT_SIZE_CIRC}
        touch {output[1]}
        """        

# Align contigs all against all
rulename = "align_contigs"
rule align_contigs:
    input:
        OUTDIR / "data/sample_{key}/contigs.flt.fna.gz",
    output:
        os.path.join(OUTDIR,"{key}",'tmp','blastn','blastn_all_against_all.txt'),
        os.path.join(OUTDIR,"{key}",'log/blastn/align_contigs.finished')
    params:
        db_name=os.path.join(OUTDIR,'tmp', "{key}",'blastn','contigs.db'), # TODO should be made?
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    log: config.get("log", "log/") + "{key}_" + rulename
    shell:
        """
        gunzip -c {input} |makeblastdb -in - -dbtype nucl -out {params.db_name} -title contigs.db 
        gunzip -c {input} |blastn -query - -db {params.db_name} -out {output[0]}.redundant -outfmt 6 -perc_identity 95 -num_threads {threads} -max_hsps 1000000 >> {log}
        awk '$1 != $2 && $4 >= 500' {output[0]}.redundant > {output[0]} 
        touch {output[1]}
        """
        
## Generate nx graph per sample for graphs from gfa assembly graphs
# for each "sample"
rulename = "weighted_assembly_graphs"
rule weighted_assembly_graphs:
    input:
        graph = assembly_graph,
        graphinfo  = contigs_paths
    output:
        os.path.join(OUTDIR,"{key}",'tmp','assembly_graphs','{id}.pkl'),
        os.path.join(OUTDIR,"{key}", 'log','assembly_graph_processing','weighted_assembly_graphs_{id}.finished'),
    params:
        path = os.path.join(PAU_SRC_DIR, 'process_gfa.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", "log/") + "{key}_{id}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --gfa {input[0]} --paths {input[1]} -s {wildcards.id} -m {MIN_CONTIG_LEN}  --out {output[0]} 2> {log} \
        && touch {output[1]}
        """

# TODO Why does this exist?
rulename = "weighted_assembly_graphs_all_samples"
rule weighted_assembly_graphs_all_samples:
    input: lambda wildcards: expand(os.path.join(OUTDIR,"{key}",'tmp','assembly_graphs','{id}.pkl'),key=wildcards.key, id=sample_id[wildcards.key]),
    output:
        os.path.join(OUTDIR,"{key}",'log','assembly_graph_processing','weighted_assembly_graphs_all_samples.finished')
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    shell:
        """
        touch {output}
        """

## 3. Genereate nx graph from the alignment graph
rulename = "weighted_alignment_graph"
rule weighted_alignment_graph:
    input:
        os.path.join(OUTDIR,"{key}",'tmp','blastn','blastn_all_against_all.txt'),
        os.path.join(OUTDIR,"{key}",'log/blastn/align_contigs.finished')
    output:
        os.path.join(OUTDIR,"{key}",'tmp','alignment_graph','alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}",'log','alignment_graph_processing','weighted_alignment_graph.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'process_blastout.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --blastout {input[0]} --out {output[0]} --minid 98 2> {log}
        touch {output[1]}
        """


## 4. Merge assembly graphs an alignment graph into a unified graph
rulename = "create_assembly_alignment_graph"
rule create_assembly_alignment_graph:
    input:
        alignment_graph_file = os.path.join(OUTDIR,"{key}",'tmp','alignment_graph','alignment_graph.pkl'),
        # assembly_graph_files = expand_dir(os.path.join(OUTDIR,"[key]",'tmp','assembly_graphs','[value].pkl'), sample_id), # TODO might be funky 
        assembly_graph_files = lambda wildcards: expand(os.path.join(OUTDIR,"{key}",'tmp','assembly_graphs','{id}.pkl'), key=wildcards.key, id=sample_id[wildcards.key]),
        weighted_alignment_graph_finished_log = os.path.join(OUTDIR,"{key}",'log','alignment_graph_processing','weighted_alignment_graph.finished'),
        weighted_assembly_graphs_all_samples_finished_log = os.path.join(OUTDIR,"{key}", 'log','assembly_graph_processing','weighted_assembly_graphs_all_samples.finished')
    output:
        os.path.join(OUTDIR,"{key}",'tmp','assembly_alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}", 'log','create_assembly_alignment_graph.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'merge_assembly_alignment_graphs.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --graph_alignment {input.alignment_graph_file}  --graphs_assembly {input.assembly_graph_files} --out {output[0]}  2> {log}
        touch {output[1]}
        """

## 3. Run n2v on the per sample assembly graphs
rulename = "n2v_assembly_alignment_graph"
rule n2v_assembly_alignment_graph:
    input:
        os.path.join(OUTDIR,"{key}",'tmp','assembly_alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}",'log','create_assembly_alignment_graph.finished'), 
        contig_names_file = OUTDIR / "data/sample_{key}/contigs.names.sorted"
    output:
        directory(os.path.join(OUTDIR,"{key}",'tmp','n2v','assembly_alignment_graph_embeddings')),
        os.path.join(OUTDIR,"{key}",'tmp','n2v','assembly_alignment_graph_embeddings','embeddings.npz'),
        os.path.join(OUTDIR,"{key}",'tmp','n2v','assembly_alignment_graph_embeddings','contigs_embedded.txt'),
        os.path.join(OUTDIR,"{key}",'log','n2v','n2v_assembly_alignment_graph.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'fastnode2vec_args.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} -G {input[0]} --ed {N2V_ED} --nw {N2V_NW} --ws {N2V_WS} --wl {N2V_WL}\
         -p {N2V_P} -q {N2V_Q} --outdirembs {output[0]} --normE {N2V_NZ} --contignames {input.contig_names_file}
        touch {output[3]}
        """


## 4. Extract hoods from assembly graphs n2v embeddings per sample 
rulename = "extract_neighs_from_n2v_embeddings"
rule extract_neighs_from_n2v_embeddings:
    input:
        os.path.join(OUTDIR,"{key}",'tmp','n2v','assembly_alignment_graph_embeddings','embeddings.npz'),
        os.path.join(OUTDIR,"{key}",'tmp','n2v','assembly_alignment_graph_embeddings','contigs_embedded.txt'),
        os.path.join(OUTDIR,"{key}",'log','n2v','n2v_assembly_alignment_graph.finished'),
        os.path.join(OUTDIR,"{key}",'tmp','assembly_alignment_graph.pkl'),
        contig_names_file = OUTDIR / "data/sample_{key}/contigs.names.sorted"
    output:
        directory(os.path.join(OUTDIR,"{key}",'tmp','neighs')),
        os.path.join(OUTDIR,'{key}','tmp','neighs','neighs_intraonly_rm_object_r_%s.npz'%NEIGHS_R),
        os.path.join(OUTDIR,"{key}",'log','neighs','extract_neighs_from_n2v_embeddings.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'embeddings_to_neighs.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --embs {input[0]} --contigs_embs {input[1]}\
         --contignames {input.contig_names_file} -g {input[3]} -r {NEIGHS_R} --neighs_outdir {output[0]}
        touch {output[2]}
        """

## 5. Run vamb to merge the hoods
rulename = "run_vamb_asymmetric"
rule run_vamb_asymmetric:
    input:
        notused = os.path.join(OUTDIR,"{key}",'log','neighs','extract_neighs_from_n2v_embeddings.finished'), # TODO why is this not used?
        contigs = OUTDIR /  "data/sample_{key}/contigs.flt.fna.gz",
        bamfiles = lambda wildcards: expand(OUTDIR / "data/sample_{key}/mapped_sorted/{id}.bam.sort", key=wildcards.key, id=sample_id[wildcards.key]),
        nb_file = os.path.join(OUTDIR,'{key}','tmp','neighs','neighs_intraonly_rm_object_r_%s.npz'%NEIGHS_R)
    output:
        directory = directory(os.path.join(OUTDIR,"{key}", 'vamb_asymmetric')),
        bins = os.path.join(OUTDIR,"{key}",'vamb_asymmetric','vae_clusters_unsplit.tsv'),
        finished = os.path.join(OUTDIR,"{key}",'log/run_vamb_asymmetric.finished'),
        lengths = os.path.join(OUTDIR,"{key}",'vamb_asymmetric','lengths.npz'),
        vae_clusters = os.path.join(OUTDIR, '{key}','vamb_asymmetric/vae_clusters_community_based_complete_unsplit.tsv'),
        compo = os.path.join(OUTDIR, '{key}','vamb_asymmetric/composition.npz'),
    params:
        walltime='86400',
        cuda='--cuda' if CUDA else ''
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        rmdir {output.directory}
        {PLAMB_PRELOAD}
        vamb bin vae_asy --outdir {output.directory} --fasta {input.contigs} -p {threads} --bamfiles {input.bamfiles}\
        --seed 1 --neighs {input.nb_file}  -m {MIN_CONTIG_LEN} {PLAMB_PARAMS}\
         {params.cuda}  
        touch {output}
        """

rulename = "run_geNomad"
rule run_geNomad:
    input:
        OUTDIR / "data/sample_{key}/contigs.flt.fna.gz",
        geNomad_db = geNomad_db
    output:
        directory(os.path.join(OUTDIR,"{key}",'tmp','geNomad')),
        os.path.join(OUTDIR,"{key}",'log/run_geNomad.finished'),
        os.path.join(OUTDIR,"{key}",'tmp','geNomad','contigs.flt_aggregated_classification','contigs.flt_aggregated_classification.tsv')
    # params:
    #     db_geNomad= THIS_FILE_DIR / "genomad_db" / "genomad_db", 
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/genomad.yaml"
    shell:
        """
        genomad end-to-end --cleanup {input} {output[0]} {input.geNomad_db} --threads {threads}
        touch {output[1]}
        """

## 7. Merge graph clustes with circular clusters
rulename = "merge_circular_with_graph_clusters"
rule merge_circular_with_graph_clusters:
    input:
        os.path.join(OUTDIR,'{key}','log/run_vamb_asymmetric.finished'),
        os.path.join(OUTDIR,'{key}','log/circularisation/circularisation.finished'),
        os.path.join(OUTDIR,'{key}','tmp','circularisation','max_insert_len_%i_circular_clusters.tsv.txt'%MAX_INSERT_SIZE_CIRC),
        vae_clusters = os.path.join(OUTDIR, '{key}','vamb_asymmetric/vae_clusters_community_based_complete_unsplit.tsv')
    output:
        os.path.join(OUTDIR,'{key}','vamb_asymmetric','vae_clusters_community_based_complete_and_circular_unsplit.tsv'),
        os.path.join(OUTDIR,'{key}','log/merge_circular_with_graph_clusters.finished')
    params:
        path=os.path.join(PAU_SRC_DIR, 'merge_circular_plamb_clusters.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --cls_plamb {input.vae_clusters} --cls_circular {input[2]} --outcls {output[0]}
        touch {output[1]}
        """

## 7. Classify bins/clusters into plasmid/organism/virus bins/clusters
rulename = "classify_bins_with_geNomad"
rule classify_bins_with_geNomad:
    input:
        os.path.join(OUTDIR,"{key}",'tmp','geNomad','contigs.flt_aggregated_classification','contigs.flt_aggregated_classification.tsv'),
        os.path.join(OUTDIR,"{key}",'log/run_vamb_asymmetric.finished'),
        os.path.join(OUTDIR,"{key}",'log/run_geNomad.finished'),
        os.path.join(OUTDIR,"{key}",'vamb_asymmetric','vae_clusters_unsplit.tsv'),
        contignames = OUTDIR / "data/sample_{key}/contigs.names.sorted",
        lengths = os.path.join(OUTDIR,"{key}",'vamb_asymmetric','lengths.npz'),
        comm_clusters = os.path.join(OUTDIR,'{key}','vamb_asymmetric','vae_clusters_community_based_complete_and_circular_unsplit.tsv'),
        composition = os.path.join(OUTDIR,'{key}','vamb_asymmetric','composition.npz'),
    output:
        os.path.join(OUTDIR,"{key}",'vamb_asymmetric','vae_clusters_graph_thr_0.75_candidate_plasmids.tsv'),
        os.path.join(OUTDIR,"{key}",'log','classify_bins_with_geNomad.finished')
    params:
        path = os.path.join(PAU_SRC_DIR, 'classify_bins_with_geNomad_strict_circular.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", "log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/pipeline_conda.yaml"
    shell:
        """
        python {params.path} --clusters {input.comm_clusters} \
         --dflt_cls {OUTDIR}/{wildcards.key}/vamb_asymmetric/vae_clusters_density_unsplit.tsv --scores {input[0]} --outp {output[0]} \
         --composition {input.composition}
         # --lengths {input.lengths} --contignames {input.contignames} --composition {input.composition}
        touch {output[1]}
        """









