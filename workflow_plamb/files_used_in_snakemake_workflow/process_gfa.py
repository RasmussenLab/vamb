import argparse
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from git_commit import get_git_commit


def load_gfa(contigs_file: str, gfa_file: str):
    """
    Given the contigs.paths file and the gfa assembly graph file, generate a networkx graph object:
        - graph_objec: undirected graph, each contig is a node, create edge if the starting of ending segment of the contig is
        present in any other contig
    """
    paths = {}  # dictionary {contig_id: [first_segment, last_segment]}
    segment_contigs = {}  # dictionary  {segment:{contig_n,contig_n}} considering any segment part of the path
    node_count = 0
    # Get contig paths from contigs.paths
    with open(contigs_file) as file:
        name = file.readline()
        path = file.readline()
        while name != "" and path != "":
            while ";" in path:
                path = path[:-2] + "," + file.readline()

            start = "NODE_"
            end = "_length_"
            # Extract the integer between NODE_ and _length_
            contig_num = str(
                int(re.search("%s(.*)%s" % (start, end), name).group(1)) - 1
            )

            # Those are all the segments that connect all sequences composing the contig, the first and last one will be
            # used to define the beggining and end of the contig
            segments = path.rstrip().split(",")

            if contig_num not in paths:
                node_count += 1

                paths[contig_num] = [
                    segments[0],
                    segments[-1],
                ]  # first and last segment
            # add the the contig name to the segment key, so we can know into which contig a segment appears
            for segment in segments:
                if segment not in segment_contigs:
                    segment_contigs[segment] = set([contig_num])
                else:
                    segment_contigs[segment].add(contig_num)

            name = file.readline()
            path = file.readline()

    links = []  # ["46463 - 34345 +","652 + 8238 -",...]
    links_map = defaultdict(
        set
    )  # dictionary {segment:{segment_i,segment_j}}, indicates mapping between segments
    links_map_e2s = defaultdict(
        set
    )  # same than links_map but order consistent, end 2 start
    links_map_s2e = defaultdict(
        set
    )  # same than links_map but order consistent, start 2 end

    # Get gfa file from
    with open(gfa_file) as file:
        line = file.readline()
        while line != "":
            # Identify lines with link information
            if "L" in line:
                strings = line.split("\t")
                f1, f2 = strings[1] + strings[2], strings[3] + strings[4]
                links_map[f1].add(f2)
                links_map[f2].add(f1)
                links.append(strings[1] + strings[2] + " " + strings[3] + strings[4])

                links_map_e2s[f1].add(f2)
                links_map_s2e[f2].add(f1)

            line = file.readline()
    # Create graph
    graph_object = nx.Graph()

    # Add vertices
    for i in range(node_count):
        graph_object.add_node(i)  # change it later so it matches

    for i in range(len(paths)):
        # for each contig, retrieve the first and last segment, which define the beginning and end of the contig
        segments = paths[str(i)]

        start = segments[0]
        start_rev = ""
        # if the start segment is not complement (+), the start of the reverse will be the same but with (-)
        if start.endswith("+"):
            start_rev = start[:-1] + "-"
        else:
            start_rev = start[:-1] + "+"

        end = segments[1]
        end_rev = ""
        if end.endswith("+"):
            end_rev = end[:-1] + "-"
        else:
            end_rev = end[:-1] + "+"

        new_links = []  # list of segments that link to either my start or end segments, both in + and -

        if start in links_map:
            new_links.extend(
                list(links_map[start])
            )  # what maps to my starting segment ?
        # if the reverse complement of the starting segment is present in links_map then add it to the new_links list
        if start_rev in links_map:
            new_links.extend(
                list(links_map[start_rev])
            )  # what maps to my starting rev compl segment ?
        if end in links_map:
            new_links.extend(list(links_map[end]))  # what maps to my ending segment ?
        if end_rev in links_map:
            new_links.extend(
                list(links_map[end_rev])
            )  # what maps to my ending rev compl segment ?

        # for each segment in new_links
        for new_link in (
            new_links
        ):  # for all segment s_i that links to either my start or ending segment
            # if the segment exists as a key in segment_contigs, which indicates that is eihter part of an ending or starting segment
            if new_link in segment_contigs:  # if this segment s_i maps to a contig
                # for each contig that contains this segment (as starting or ending segment)
                for contig in segment_contigs[new_link]:
                    # if the contig that contains this segment is not itself
                    if i != int(contig):  # if this contig is not the same than my own
                        # add an edge between the contigs
                        if not graph_object.has_edge(int(contig), i):
                            graph_object.add_edge(i, int(contig), nlinks=0)

                        graph_object[int(contig)][i]["nlinks"] += 1

    return graph_object


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Generate nx graphs from gfa files")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--gfa",
        dest="gfa_file",
        required=True,
        type=str,
        help="Path to the .gfa file.",
    )
    parser.add_argument(
        "--paths",
        dest="contig_path",
        required=True,
        type=str,
        help="Path to the contig.paths file.",
    )
    parser.add_argument(
        "--out",
        dest="outfile",
        required=True,
        type=str,
        help="File for the networkx graph file will be stored.",
    )
    parser.add_argument(
        "-s",
        dest="sample",
        required=True,
        type=str,
        help="Sample name that will be prefixed to contig with separator",
    )
    parser.add_argument(
        "-m",
        dest="min_contig_size",
        type=int,
        default=2000,
        help="Min contig size, connected components without more than 1 contig larger than min_contig size will be removed from the graph.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="C",
        help="Separator that indicates the end of the sample name and the beggining of the contig name [C].",
    )

    # Parse the arguments
    args = parser.parse_args()

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    print(args)
    t0 = time.time()
    nx_graph = load_gfa(contigs_file=args.contig_path, gfa_file=args.gfa_file)

    # Rename nodes since now they are only named as integers
    t1 = time.time()
    print("Networkx file generated in %.2f seconds." % (t1 - t0))

    ## Weight the edges based with normalized_linkage
    # 1. Extract contig lengths from contig names
    # 2. Iterate over edges and defined edge weight
    # 3. Define a new weight scaled between 0 and 1

    # 1. Extract contig lengths from contig.paths  ########## and 2. Remove graph components if have less than 2 contigs above min_contig_size
    t2 = time.time()
    contig_len_d = dict()
    binning_contigs = set()
    contignames = []
    for line in np.loadtxt(args.contig_path, dtype=object, delimiter="\t"):
        if not line.startswith("NODE_"):
            continue
        c_id = line
        c_name = args.sample + args.separator + c_id
        c_length = int(line.split("length_")[1].split("_cov")[0])
        contig_len_d[c_name] = c_length
        if c_length >= args.min_contig_size:
            binning_contigs.add(c_name)

        if c_id.startswith("NODE_") and not c_id.endswith("'"):
            contignames.append(c_id)

    print(
        "%i/%i contigs will be considered for binning."
        % (len(binning_contigs), len(contig_len_d.keys()))
    )

    cid_originalcontignames_d = {
        i: contignames[i] for i in range(len(nx_graph.nodes()))
    }

    nx_graph_renamed = nx.relabel_nodes(
        nx_graph,
        {
            node: args.sample + args.separator + cid_originalcontignames_d[node]
            for node in nx_graph.nodes()
        },
    )

    print(args.sample, list(nx_graph_renamed.nodes())[:10])

    t3 = time.time()

    print("Sample added to node names in %.2f seconds." % (t3 - t2))

    # 2. Iterate over edges and defined edge weight

    for u, v in nx_graph_renamed.edges():
        nlinks_uv = nx_graph_renamed[u][v]["nlinks"]
        # nlinks_u = np.sum([u_d["nlinks"] for u_d in nx_graph_renamed[u].values()])
        # nlinks_v = np.sum([v_d["nlinks"] for v_d in nx_graph_renamed[v].values()])
        nx_graph_renamed[u][v]["weight_unscaled"] = nlinks_uv / np.min(
            [contig_len_d[u], contig_len_d[v]]
        )

    # 3. Define a new weight scaled between 0 and 1
    t4 = time.time()
    print("Edges weighted in %.2f seconds." % (t4 - t3))
    max_edge_weight = np.percentile(
        np.array(
            [
                nx_graph_renamed[u][v]["weight_unscaled"]
                for u, v in nx_graph_renamed.edges()
            ]
        ),
        99,
    )
    for u, v in nx_graph_renamed.edges():
        nx_graph_renamed[u][v]["weight"] = min(
            nx_graph_renamed[u][v]["weight_unscaled"] / max_edge_weight, 1.0
        )

    t5 = time.time()
    print("Edge weights scaled in %.2f seconds." % (t5 - t4))

    ## Save graph to pickle file
    with open(args.outfile, "wb") as pkl:
        pickle.dump(nx_graph_renamed, pkl)

    t6 = time.time()
    print("Graph saved in %.2f seconds." % (t6 - t5))

    print("\nNetworkx file created from gfa file in %.2f seconds." % (t6 - t0))
