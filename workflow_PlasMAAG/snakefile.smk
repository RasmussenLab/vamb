import pandas as pd
import collections
import os
from pathlib import Path
import sys

# Set the directory the snakefile exists in. This makes us able to call the pipeline with the relevant src files from other directories.
THIS_FILE_DIR = Path(workflow.basedir)

# If the configfile is not set explicit fall back to the default
CONFIG_PATH = THIS_FILE_DIR / "config/config.yaml"
# TODO should go to configfile
if CONFIG_PATH.exists():
    configfile: CONFIG_PATH

# Define the src directory for the files used in the snakemake workflow
SRC_DIR = THIS_FILE_DIR / "files_used_in_snakemake_workflow"  

# Get the output_directory defined by the user or fallback to current directory, which is the default way snakemake handles output directories
OUTDIR = Path("") if config.get("output_directory") is None else Path(config.get("output_directory"))

#### Setting parameters from the config file ####
##  For a more throughout description of what the different config options mean see the /config/config.yaml file

# Default resources used
default_walltime = config.get("default_walltime", "48:00:00")
default_threads = config.get("default_threads", 16)
default_mem_gb = config.get("default_mem_gb", 50)

# Minimum contig length used
MIN_CONTIG_LEN = int(config.get("min_contig_len", 2000)) 

# N2V parameters
N2V_NZ= config.get("n2v_nz", "weight") 
N2V_ED= config.get("n2v_ed", 32) 
N2V_WL= config.get("n2v_wl", 10) 
N2V_NW= config.get("n2v_nw", 50) 
N2V_WS= config.get("n2v_ws", 10) 
N2V_P= config.get("n2v_p", 0.1) 
N2V_Q= config.get("n2v_q", 2.0) 

# Binning parameters
PLAMB_PARAMS = config.get("plamb_params", ' -o C ') 
PLAMB_PRELOAD = config.get("plamb_preload", "") 

# Other options
CUDA = True if config.get("cuda") ==  "True" else False
NEIGHS_R=config.get("neighs_r", '0.1') 
MAX_INSERT_SIZE_CIRC = int(config.get("max_insert_size_circ", 50))
GENOMAD_THR = config.get("genomad_thr", "0.75")
GENOMAD_THR_CIRC = config.get("genomad_thr_circ", "0.5")
 
## ----------- ##

# Assert that input files are actually passed to snakemake
if config.get("read_file") == None and config.get("read_assembly_dir") == None and config.get("should_install_genomad") == None:
    print("ERROR: read_file or read_assembly_dir not passed to snakemake as config. Define either. Eg. snakemake <arguments> --config read_file=<read file>. If in doubt refer to the README.md file")
    sys.exit()

# Set default paths for the SPades outputfiles - running the pipeline from allready assembled reads overwrite these values
contigs =  OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.fasta"
contigs_paths =  OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.paths"
assembly_graph = OUTDIR / "{key}/assembly_mapping_output/spades_{id}/assembly_graph_after_simplification.gfa"

# Set default values for dictonaries containg information about the input information
# The way snakemake parses snakefiles means we have to define them even though they will always be present
sample_id = dict()               
sample_id_path= dict() 
sample_id_path_assembly = dict()

# If the read_file is defined the pipeline will also run SPades and assemble the reads
if config.get("read_file") != None:
    df = pd.read_csv(config["read_file"], sep=r"\s+", comment="#")
    sample_id = collections.defaultdict(list)
    sample_id_path = collections.defaultdict(dict)
    for id, (read1, read2) in enumerate(zip(df.read1, df.read2)):
        id = f"sample{str(id)}"
        # Earlier version of the pipeline could handle passing several samples at the same time which would be processed separatly.
        # For easier user input this was removed. Therefore the "sample" is always set to the same. 
        # By parsing different sample names from the input this can be implemented again - this is nice eg. for benchmarking
        sample = "intermidiate_files" 
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
        sample = "intermidiate_files"
        sample_id[sample].append(id)
        sample_id_path[sample][id] = [read1, read2]
        sample_id_path_assembly[sample][id] = [assembly]

    # Setting the output paths for the user defined SPades files
    contigs =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "contigs.fasta"
    assembly_graph  =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "assembly_graph_after_simplification.gfa"
    contigs_paths  =  lambda wildcards: Path(sample_id_path_assembly[wildcards.key][wildcards.id][0]) / "contigs.paths"

read_fw = lambda wildcards: sample_id_path[wildcards.key][wildcards.id][0]
read_rv =  lambda wildcards: sample_id_path[wildcards.key][wildcards.id][1]

# Functions to get the config-defined threads/walltime/mem_gb for a rule and if not defined the default
threads_fn = lambda rulename: config.get(rulename, {"threads": default_threads}).get("threads", default_threads) 
walltime_fn  = lambda rulename: config.get(rulename, {"walltime": default_walltime}).get("walltime", default_walltime) 
mem_gb_fn  = lambda rulename: config.get(rulename, {"mem_gb": default_mem_gb}).get("mem_gb", default_mem_gb) 

try: 
    os.makedirs(os.path.join(OUTDIR,'log'), exist_ok=True)
except FileExistsError:
    pass

rulename = "all"
rule all:
    input:
        candidate_plasmids = expand(os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_graph_thr_' + GENOMAD_THR + '_candidate_plasmids.tsv'),key=sample_id.keys()),
        candidate_genomes = expand(os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_' + GENOMAD_THR + '_thrcirc_' + GENOMAD_THR_CIRC + '.tsv'),key=sample_id.keys()), #
        assert_genomad_finished = expand(os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_geNomad.finished'), key=sample_id.keys()),
        assert_vamb_finished = expand(os.path.join(OUTDIR, "{key}",'rule_completed_checks/run_contrastive_VAE.finished'), key=sample_id.keys()),
        candidate_genomes_scores =expand(os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_' + GENOMAD_THR + '_thrcirc_' + GENOMAD_THR_CIRC + '_gN_scores.tsv'), key=sample_id.keys()),
        candidate_plasmids_scores = expand(os.path.join(OUTDIR,"{key}",'contrastive_VAE',f'vae_clusters_graph_thr_' + GENOMAD_THR + '_candidate_plasmids_gN_scores.tsv'), key=sample_id.keys()),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    output:
        candidate_plasmids = OUTDIR / "results/candidate_plasmids.tsv",
        candidate_genomes = OUTDIR / "results/candidate_genomes.tsv",
        combined_scores = OUTDIR / "results/scores.tsv",
    shell: 
        """
        cat {input.candidate_plasmids} > {output.candidate_plasmids}
        cat {input.candidate_genomes} > {output.candidate_genomes}
        # Remove header from one of the files
        tail -n+2  {input.candidate_genomes_scores} | cat {input.candidate_plasmids_scores} - > {output.combined_scores}
        """

# If the genomad database is given as an argument don't download it again
# This works boths from when the tool is called from the CLI wrapper and from snakemake if the config is extended setting the genomad_database variable
if config.get("genomad_database") is not None:
    geNomad_db = config.get("genomad_database")
else: 
    geNomad_db = THIS_FILE_DIR / "genomad_db" / "genomad_db",

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

# If only reads are passed run metaspades to assemble the reads
rulename = "spades"
rule spades:
    input:
       fw = read_fw, 
       rv = read_rv, 
    output:
       outdir = directory(OUTDIR / "{key}/assembly_mapping_output/spades_{id}"),
       outfile = OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.fasta",
       graph = OUTDIR / "{key}/assembly_mapping_output/spades_{id}/assembly_graph_after_simplification.gfa",
       graphinfo  = OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.paths",
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_{id}_" + rulename
    conda: THIS_FILE_DIR / "envs/spades_env.yaml"
    shell:
       "spades.py --meta "
       "-t {threads} -m 180 "
       "-o {output.outdir} -1 {input.fw} -2 {input.rv} " 
       "-t {threads} --memory {resources.mem_gb} &> {log} " 

# Rename the contigs to keep sample information for later use 
rulename = "rename_contigs"
rule rename_contigs:
    input:
        contigs,
    output:
        OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.renamed.fasta"
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_{id}_" + rulename
    shell:
        """
        sed 's/^>/>S{wildcards.id}C/' {input} > {output} 2> {log}
        """

# Cat the contigs together in one file to later map each pair of reads against all the contigs together
rulename="cat_contigs"
rule cat_contigs:
    input: lambda wildcards: expand(OUTDIR / "{key}/assembly_mapping_output/spades_{id}/contigs.renamed.fasta", key=wildcards.key, id=sample_id[wildcards.key]),
    output: OUTDIR / "{key}/assembly_mapping_output/contigs.flt.fna.gz"
    threads: threads_fn(rulename)
    params: script =  SRC_DIR / "concatenate.py"
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell: 
        "python {params.script} {output} {input} --keepnames -m {MIN_CONTIG_LEN} &> {log} "  

# Extract the contigs names for later use
rulename = "get_contig_names"
rule get_contig_names:
    input:
        OUTDIR / "{key}/assembly_mapping_output/contigs.flt.fna.gz"
    output: 
        OUTDIR / "{key}/assembly_mapping_output/contigs.names.sorted"
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        "zcat {input} | grep '>' | sed 's/>//' > {output} 2> {log} "

# Run strobealign to get the abundances  
rulename = "Strobealign_bam_default"
rule Strobealign_bam_default:
        input: 
            fw = read_fw,
            rv = read_rv,
            contig = OUTDIR /"{key}/assembly_mapping_output/contigs.flt.fna.gz",
        output:
            OUTDIR / "{key}/assembly_mapping_output/mapped/{id}.bam"
        threads: threads_fn(rulename)
        resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
        benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
        log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_{id}_" + rulename
        conda: THIS_FILE_DIR / "envs/strobe_env.yaml"
        shell:
            """
            strobealign -t {threads} {input.contig} {input.fw} {input.rv} > {output} 2> {log}
            """

# Sort the bam files and index them
rulename="sort"
rule sort:
    input:
        OUTDIR / "{key}/assembly_mapping_output/mapped/{id}.bam",
    output:
        OUTDIR / "{key}/assembly_mapping_output/mapped_sorted/{id}.bam.sort",
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_{id}_" + rulename
    shell:
        """
	samtools sort --threads {threads} {input} -o {output} 2> {log}
	samtools index {output} 2>> {log}
	"""

## The next part of the pipeline is composed of the following steps:
# 0. Look for contigs circularizable 
# 1. Align contigs all against all 
# 2. Generate per sample assembly graphs from gfa assembly graphs
# 3. Generate alignment graph from 1. output
# 4. Produce assembly-alignment graph from merging assembly graph and alignment graph.
# 5. Run n2v on the per sample assembly graphs 
# 6. Extract neighbourhoods from assembly-alignment graph n2v embeddings
# 7. Run vamb to merge,split, and expand the hoods
# 8. Merge graph clusters with circular clusters
# 9. Run geNomad to get contig plasmid, chromosome, and virus classificaiton scores
# 10. Classify bins/clusters into plasmid/organism/virus bins/clusters

# 0. Look for contigs circularizable 
rulename = "circularize"
rule circularize:
    input:
        bamfiles = lambda wildcards: expand(OUTDIR /  "{key}/assembly_mapping_output/mapped_sorted/{id}.bam.sort", key=f"{wildcards.key}", id=sample_id[wildcards.key]),
    output:
        os.path.join(OUTDIR,"{key}",'circularisation','max_insert_len_%i_circular_clusters.tsv.txt'%MAX_INSERT_SIZE_CIRC),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/circularisation/circularisation.finished')
    params:
        path = os.path.join(SRC_DIR, 'circularisation.py'),
        dir_bams = lambda wildcards: expand(OUTDIR / "{key}/assembly_mapping_output/mapped_sorted", key=wildcards.key)
    threads: threads_fn(rulename),
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --dir_bams {params.dir_bams} --outcls {output[0]} --max_insert {MAX_INSERT_SIZE_CIRC} &> {log}
        touch {output[1]}
        """        

# 1. Align contigs all against all
rulename = "align_contigs"
rule align_contigs:
    input:
        OUTDIR / "{key}/assembly_mapping_output/contigs.flt.fna.gz",
    output:
        os.path.join(OUTDIR,"{key}",'blastn','blastn_all_against_all.txt'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/blastn/align_contigs.finished'),
    params:
        db_name=os.path.join(OUTDIR, "{key}",'blastn','contigs.db'), # TODO should be made?
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        gunzip -c {input} |makeblastdb -in - -dbtype nucl -out {params.db_name} -title contigs.db 2> {log}
        gunzip -c {input} |blastn -query - -db {params.db_name} -out {output[0]}.redundant -outfmt 6 -perc_identity 95 -num_threads {threads} -max_hsps 1000000 2>> {log}
        awk '$1 != $2 && $4 >= 500' {output[0]}.redundant > {output[0]} 2>> {log}
        touch {output[1]}
        """
        
## 2. Generate nx graph per sample for graphs from gfa assembly graphs for each "sample"
rulename = "weighted_assembly_graphs"
rule weighted_assembly_graphs:
    input:
        graph = assembly_graph,
        graphinfo  = contigs_paths
    output:
        os.path.join(OUTDIR,"{key}",'assembly_graphs','{id}.pkl'),
        os.path.join(OUTDIR,"{key}", 'rule_completed_checks','assembly_graph_processing','weighted_assembly_graphs_{id}.finished'),
    params:
        path = os.path.join(SRC_DIR, 'process_gfa.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_{id}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_{id}_" + rulename
    shell:
        """
        python {params.path} --gfa {input[0]} --paths {input[1]} -s {wildcards.id} -m {MIN_CONTIG_LEN}  --out {output[0]} &> {log} \
        && touch {output[1]}
        """

rulename = "weighted_assembly_graphs_all_samples"
rule weighted_assembly_graphs_all_samples:
    input: lambda wildcards: expand(os.path.join(OUTDIR,"{key}",'assembly_graphs','{id}.pkl'),key=wildcards.key, id=sample_id[wildcards.key]),
    output:
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','assembly_graph_processing','weighted_assembly_graphs_all_samples.finished')
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        touch {output}
        """

# 3. Genereate nx graph from the alignment graph
rulename = "weighted_alignment_graph"
rule weighted_alignment_graph:
    input:
        os.path.join(OUTDIR,"{key}",'blastn','blastn_all_against_all.txt'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/blastn/align_contigs.finished')
    output:
        os.path.join(OUTDIR,"{key}",'alignment_graph','alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','alignment_graph_processing','weighted_alignment_graph.finished')
    params:
        path = os.path.join(SRC_DIR, 'process_blastout.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --blastout {input[0]} --out {output[0]} --minid 98 &> {log}
        touch {output[1]}
        """

# 4. Produce assembly-alignment graph from merging assembly graph and alignment graph.
rulename = "create_assembly_alignment_graph"
rule create_assembly_alignment_graph:
    input:
        alignment_graph_file = os.path.join(OUTDIR,"{key}",'alignment_graph','alignment_graph.pkl'),
        assembly_graph_files = lambda wildcards: expand(os.path.join(OUTDIR,"{key}",'assembly_graphs','{id}.pkl'), key=wildcards.key, id=sample_id[wildcards.key]),
        weighted_alignment_graph_finished_log = os.path.join(OUTDIR,"{key}",'rule_completed_checks','alignment_graph_processing','weighted_alignment_graph.finished'),
        weighted_assembly_graphs_all_samples_finished_log = os.path.join(OUTDIR,"{key}", 'rule_completed_checks','assembly_graph_processing','weighted_assembly_graphs_all_samples.finished')
    output:
        os.path.join(OUTDIR,"{key}",'assembly_alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}", 'rule_completed_checks','create_assembly_alignment_graph.finished')
    params:
        path = os.path.join(SRC_DIR, 'merge_assembly_alignment_graphs.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --graph_alignment {input.alignment_graph_file}  --graphs_assembly {input.assembly_graph_files} --out {output[0]}  &> {log}
        touch {output[1]}
        """

## 5. Run n2v on the per sample assembly graphs
rulename = "n2v_assembly_alignment_graph"
rule n2v_assembly_alignment_graph:
    input:
        os.path.join(OUTDIR,"{key}",'assembly_alignment_graph.pkl'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','create_assembly_alignment_graph.finished'), 
        contig_names_file = OUTDIR / "{key}/assembly_mapping_output/contigs.names.sorted"
    output:
        directory(os.path.join(OUTDIR,"{key}",'n2v','assembly_alignment_graph_embeddings')),
        os.path.join(OUTDIR,"{key}",'n2v','assembly_alignment_graph_embeddings','embeddings.npz'),
        os.path.join(OUTDIR,"{key}",'n2v','assembly_alignment_graph_embeddings','contigs_embedded.txt'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','n2v','n2v_assembly_alignment_graph.finished')
    params:
        path = os.path.join(SRC_DIR, 'fastnode2vec_args.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/node2vec.yaml"
    shell:
        """
        python {params.path} -G {input[0]} --ed {N2V_ED} --nw {N2V_NW} --ws {N2V_WS} --wl {N2V_WL}\
         -p {N2V_P} -q {N2V_Q} --outdirembs {output[0]} --normE {N2V_NZ} --contignames {input.contig_names_file} &> {log}
        touch {output[3]}
        """

# 6. Extract neighbourhoods from assembly-alignment graph n2v embeddings
rulename = "extract_neighs_from_n2v_embeddings"
rule extract_neighs_from_n2v_embeddings:
    input:
        os.path.join(OUTDIR,"{key}",'n2v','assembly_alignment_graph_embeddings','embeddings.npz'),
        os.path.join(OUTDIR,"{key}",'n2v','assembly_alignment_graph_embeddings','contigs_embedded.txt'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','n2v','n2v_assembly_alignment_graph.finished'),
        os.path.join(OUTDIR,"{key}",'assembly_alignment_graph.pkl'),
        contig_names_file = OUTDIR / "{key}/assembly_mapping_output/contigs.names.sorted"
    output:
        directory(os.path.join(OUTDIR,"{key}",'neighs')),
        os.path.join(OUTDIR,"{key}",'neighs','neighs_intraonly_rm_object_r_%s.npz'%NEIGHS_R),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','neighs','extract_neighs_from_n2v_embeddings.finished')
    params:
        path = os.path.join(SRC_DIR, 'embeddings_to_neighs.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --embs {input[0]} --contigs_embs {input[1]}\
         --contignames {input.contig_names_file} -g {input[3]} -r {NEIGHS_R} --neighs_outdir {output[0]} &> {log}
        touch {output[2]}
        """

# 7. Run vamb to merge, split, and expand the hoods
rulename = "run_contrastive_VAE"
rule run_contrastive_VAE:
    input:
        notused = os.path.join(OUTDIR,"{key}",'rule_completed_checks','neighs','extract_neighs_from_n2v_embeddings.finished'), # TODO why is this not used?
        contigs = OUTDIR /  "{key}/assembly_mapping_output/contigs.flt.fna.gz",
        bamfiles = lambda wildcards: expand(OUTDIR / "{key}/assembly_mapping_output/mapped_sorted/{id}.bam.sort", key=wildcards.key, id=sample_id[wildcards.key]),
        nb_file = os.path.join(OUTDIR,"{key}",'neighs','neighs_intraonly_rm_object_r_%s.npz'%NEIGHS_R)
    output:
        directory = directory(os.path.join(OUTDIR,"{key}", 'contrastive_VAE')),
        bins = os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_unsplit.tsv'),
        finished = os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_contrastive_VAE.finished'),
        lengths = os.path.join(OUTDIR,"{key}",'contrastive_VAE','lengths.npz'),
        vae_clusters = os.path.join(OUTDIR, '{key}','contrastive_VAE/vae_clusters_community_based_complete_unsplit.tsv'),
        compo = os.path.join(OUTDIR, '{key}','contrastive_VAE/composition.npz'),
    params:
        cuda='--cuda' if CUDA else ''
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        rmdir {output.directory}
        {PLAMB_PRELOAD}
        vamb bin contr_vamb --outdir {output.directory} --fasta {input.contigs} -p {threads} --bamfiles {input.bamfiles}\
        --neighs {input.nb_file}  -m {MIN_CONTIG_LEN} {PLAMB_PARAMS}\
         {params.cuda} &> {log}
        touch {output}
        """


# 8. Merge graph clustes with circular clusters
rulename = "merge_circular_with_graph_clusters"
rule merge_circular_with_graph_clusters:
    input:
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_contrastive_VAE.finished'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/circularisation/circularisation.finished'),
        os.path.join(OUTDIR,"{key}",'circularisation','max_insert_len_%i_circular_clusters.tsv.txt'%MAX_INSERT_SIZE_CIRC),
        vae_clusters = os.path.join(OUTDIR, '{key}','contrastive_VAE/vae_clusters_community_based_complete_unsplit.tsv')
    output:
        os.path.join(OUTDIR,'{key}','contrastive_VAE','vae_clusters_community_based_complete_and_circular_unsplit.tsv'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/merge_circular_with_graph_clusters.finished')
    params:
        path=os.path.join(SRC_DIR, 'merge_circular_plamb_clusters.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --cls_plamb {input.vae_clusters} --cls_circular {input[2]} --outcls {output[0]} &> {log}
        touch {output[1]}
        """

# 9. Run geNomad to get contig plasmid, chromosome, and virus classificaiton scores
rulename = "run_geNomad"
rule run_geNomad:
    input:
        contigs = OUTDIR / "{key}/assembly_mapping_output/contigs.flt.fna.gz",
        geNomad_db = geNomad_db
    output:
        directory(os.path.join(OUTDIR,"{key}",'geNomad')),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_geNomad.finished'),
        os.path.join(OUTDIR,"{key}",'geNomad','contigs.flt_aggregated_classification','contigs.flt_aggregated_classification.tsv')
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    conda: THIS_FILE_DIR / "envs/genomad.yaml"
    shell:
        """
        genomad end-to-end --cleanup {input.contigs} {output[0]} {input.geNomad_db} --threads {threads} &> {log}
        touch {output[1]}
        """

# 10. Classify bins/clusters into plasmid/organism/virus bins/clusters
rulename = "classify_bins_with_geNomad"
rule classify_bins_with_geNomad:
    input:
        os.path.join(OUTDIR,"{key}",'geNomad','contigs.flt_aggregated_classification','contigs.flt_aggregated_classification.tsv'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_contrastive_VAE.finished'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks/run_geNomad.finished'),
        os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_unsplit.tsv'),
        contignames = OUTDIR / "{key}/assembly_mapping_output/contigs.names.sorted",
        lengths = os.path.join(OUTDIR,"{key}",'contrastive_VAE','lengths.npz'),
        comm_clusters = os.path.join(OUTDIR,'{key}','contrastive_VAE','vae_clusters_community_based_complete_and_circular_unsplit.tsv'),
        composition = os.path.join(OUTDIR,'{key}','contrastive_VAE','composition.npz'),
    output:
        os.path.join(OUTDIR,"{key}",'contrastive_VAE',f'vae_clusters_graph_thr_' + GENOMAD_THR + '_candidate_plasmids.tsv'),
        os.path.join(OUTDIR,"{key}",'rule_completed_checks','classify_bins_with_geNomad.finished'),
        candidate_genomes =os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_' + GENOMAD_THR + '_thrcirc_' + GENOMAD_THR_CIRC + '.tsv'),
        candidate_genomes_scores =os.path.join(OUTDIR,"{key}",'contrastive_VAE','vae_clusters_density_unsplit_geNomadplasclustercontigs_extracted_thr_' + GENOMAD_THR + '_thrcirc_' + GENOMAD_THR_CIRC + '_gN_scores.tsv'),
        candidate_plasmids_scores = os.path.join(OUTDIR,"{key}",'contrastive_VAE',f'vae_clusters_graph_thr_' + GENOMAD_THR + '_candidate_plasmids_gN_scores.tsv'),
    params:
        path = os.path.join(SRC_DIR, 'classify_bins_with_geNomad.py'),
    threads: threads_fn(rulename)
    resources: walltime = walltime_fn(rulename), mem_gb = mem_gb_fn(rulename)
    benchmark: config.get("benchmark", "benchmark/") + "{key}_" + rulename
    log: config.get("log", f"{str(OUTDIR)}/log/") + "{key}_" + rulename
    shell:
        """
        python {params.path} --clusters {input.comm_clusters} \
         --dflt_cls {OUTDIR}/{wildcards.key}/contrastive_VAE/vae_clusters_density_unsplit.tsv --scores {input[0]} --outp {output[0]} \
         --composition {input.composition} --thr {GENOMAD_THR} --thr_circ {GENOMAD_THR_CIRC} &> {log}
        touch {output[1]}
        """
