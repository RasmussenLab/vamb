import os
import pickle
import time

import numpy as np
import networkx as nx
import argparse
from git_commit import get_git_commit


def merge_graphs(graphs_list):

    merged_graph = nx.Graph()

    for graph in graphs_list:
        merged_graph.add_edges_from(graph.edges(data=True))

    return merged_graph


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Merge networkx graph files from assembly graph samples and from alignment graph."
    )

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--graphs_assembly",
        required=True,
        nargs="+",
        help="Path to the graphs generated from the assembly graphs.",
    )
    parser.add_argument(
        "--graph_alignment",
        required=True,
        type=str,
        help="Path to the graph generated from the alignment of contigs.",
    )

    parser.add_argument(
        "--out",
        dest="outfile",
        required=True,
        type=str,
        help="File where the merged networkx graph file will be stored.",
    )
    # Parse the arguments
    args = parser.parse_args()

    ## Print git commit so we can debug    
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    print(args)
    t0 = time.time()

    graphs_to_merge = []
    for assembly_graph_f in args.graphs_assembly:
        with open(assembly_graph_f, "rb") as picklefile:
            graphs_to_merge.append(pickle.load(picklefile))

    with open(args.graph_alignment, "rb") as picklefile:
        graphs_to_merge.append(pickle.load(picklefile))

    t1 = time.time()
    print("Graphs loaded in %.2f seconds" % (t1 - t0))
    merged_graph = merge_graphs(graphs_to_merge)

    with open(args.outfile, "wb") as picklefile:
        pickle.dump(merged_graph, picklefile)

    print(
        "Merged graph has %i(%i) nodes(edges)"
        % (len(merged_graph.nodes()), len(merged_graph.edges()))
    )
    t2 = time.time()
    print("Graphs merged in %.2f seconds" % (t2 - t1))
