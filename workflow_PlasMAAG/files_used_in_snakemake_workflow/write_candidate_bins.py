import argparse
import os

from Bio import SeqIO
from git_commit import get_git_commit
import gzip
import numpy as np 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write plasmid and non plasmid bins")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--cls_pl",
        required=True,
        type=str,
        help="Path to the plasmid clusters",
    )

    parser.add_argument(
        "--cls_nonpl",
        required=True,
        type=str,
        help="Path to the nonplasmid clusters",
    )

    parser.add_argument(
        "--contigs",
        required=True,
        type=str,
        help="Path to the fasta contigs file",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Path to the results directory",
    )
    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print("Git commit hash: " + commit_hash)

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # 
    plcl_cs_d = { cl:set() for cl,c in np.loadtxt(args.cls_pl,dtype=object,skiprows=1)}
    for cl,c in np.loadtxt(args.cls_pl,dtype=object,skiprows=1):
        plcl_cs_d[cl].add(c)
    c2plcl={ c:cl for cl,cs in plcl_cs_d.items() for c in cs}
    pl_cs = set(c2plcl.keys())
    
    
    nonplcl_cs_d = { cl:set() for cl,c in np.loadtxt(args.cls_nonpl,dtype=object,skiprows=1)}
    for cl,c in np.loadtxt(
        args.cls_nonpl,dtype=object,skiprows=1):
        nonplcl_cs_d[cl].add(c)
    c2nonplcl={ c:cl for cl,cs in nonplcl_cs_d.items() for c in cs}
    nonpl_cs = set(c2plcl.keys())
    
    os.makedirs(os.path.join(args.outdir,"candidate_plasmids"))
    os.makedirs(os.path.join(args.outdir,"candidate_genomes"))
    
    with gzip.open(args.contigs, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            contig_name= record.id
            
            bin_file = contig_name+".fna"
            bin_dir =  "candidate_plasmids" if contig_name in pl_cs else "candidate_genomes"
            bin_name= c2plcl[contigname] if bin_dir == "candidate_plasmids" else c2nonplcl
            bin_path = os.path.join(args.outdir,"%s/%s"%(bin_dir,bin_file))
            with open(bin_path, "a") as out:
                SeqIO.write(record, out, "fasta")

        