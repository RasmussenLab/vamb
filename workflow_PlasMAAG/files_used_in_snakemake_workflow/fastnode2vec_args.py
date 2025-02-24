# import node2vec
import argparse
import os
import pickle
import time

import networkx as nx
import numpy as np
from fastnode2vec import Graph, Node2Vec
from git_commit import get_git_commit


def remove_singleton_ccs(graph, binning_contigs):
    """
    Remove graph components if have less than 2 contigs above min_contig_size and nodes without edges

    """
    nodes_to_remove = set(
        [
            c
            for cc in nx.connected_components(graph)
            if len(cc.intersection(binning_contigs)) < 2
            for c in cc
        ]
    )
    print(
        "%i/%i contigs will be removed from the graph since they are part of components that have less than 2 binning contigs."
        % (len(nodes_to_remove), len(graph.nodes()))
    )
    print(
        "%i(%i) nodes(edges) before removing components containing less than 2 binning contig."
        % (len(graph.nodes()), len(graph.edges()))
    )
    for c in nodes_to_remove:
        graph.remove_node(c)
    print(
        "%i(%i) nodes(edges) after removing components containing less than 2 binning contig."
        % (len(graph.nodes()), len(graph.edges()))
    )

    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)

    return graph


def run_n2v(G, emb_n, wl, nw, normalization_scheme, ws, p, q):
    # Create a node2vec object
    graph = Graph(
        [(u, v, G[u][v][normalization_scheme]) for (u, v) in G.edges()],
        directed=False,
        weighted=True,
    )
    # Node2Vec(graph: fastnode2vec.graph.Graph, dim: int, walk_length: int, window: int, p: float = 1.0, q: float = 1.0, workers: int = 1, batch_walks: Optional[int] = None, use_skipgram: bool = True, seed: Optional[int] = None, **kwargs)
    n2v = Node2Vec(graph, dim=emb_n, walk_length=wl, window=ws, p=p, q=q, workers=8)

    n2v.train(epochs=nw)

    # Learn embeddings

    return n2v


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Run n2v")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "-G",
        dest="graph_file",
        required=True,
        type=str,
        help="Path to the graphml object that contains the graph",
    )

    parser.add_argument(
        "--ed",
        dest="emb_n",
        required=True,
        type=int,
        help="Dimension of the embeddings",
    )

    parser.add_argument(
        "--nw", dest="nw", required=True, type=int, help="Number of walks"
    )

    parser.add_argument("--ws", dest="ws", required=True, type=int, help="Window size")

    parser.add_argument(
        "--wl", dest="wl", required=True, type=int, help="Walk lenghths"
    )

    parser.add_argument("-p", dest="p", type=float, default=1.0, help="p n2v parameter")

    parser.add_argument("-q", dest="q", type=float, default=1.0, help="q n2v parameter")

    parser.add_argument(
        "--normE",
        dest="normalization_scheme",
        type=str,
        help="defines the edge weights to be used",
        default="weight",
    )

    parser.add_argument(
        "--outdirembs",
        dest="embs_dir",
        required=True,
        type=str,
        help="Path were embeddings will and embedded contigs (only contigs with neighbours are embedded)  be dumped (directory)",
    )

    parser.add_argument(
        "--contignames",
        required=True,
        type=str,
        help="File containing the contignames.",
    )

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    # Parse the arguments
    args = parser.parse_args()
    print(args)
    # Use the accumulate function specified by either --sum or --multiply on the integers

    # Load graph

    # G = nx.read_graphml(args.graph_file)
    with open(args.graph_file, "rb") as picklefile:
        G = pickle.load(picklefile)

    ## Remove graph components that contain less than 2 binning contigs
    contignames = np.loadtxt(args.contignames, dtype=object)
    G_clean = remove_singleton_ccs(G, set(contignames))
    cc_l = list(nx.connected_components(G_clean))
    c_cc_map = {c: i for i, cc in enumerate(cc_l) for c in cc}
    cc_c_map = {i: set([c for c in cc]) for i, cc in enumerate(cc_l)}

    # define run_id
    id_run_g = (args.graph_file).split("/g_")[-1].replace(".pkl", "")

    sufix = "%s_nz_%s_en%i_wl%i_nw%i_Ws%i_p%.2f_q%.2f" % (
        id_run_g,
        args.normalization_scheme,
        args.emb_n,
        args.wl,
        args.nw,
        args.ws,
        args.p,
        args.q,
    )

    print("fastN2V run named %s" % sufix)

    t0 = time.time()

    # Identify and remove isolated nodes (nodes without edges)

    model = run_n2v(
        G,
        args.emb_n,
        args.wl,
        args.nw,
        args.normalization_scheme,
        args.ws,
        args.p,
        args.q,
    )

    t1 = time.time()
    ti = (t1 - t0) / 60

    print("fastN2V run %s finished in %.2f minutes" % (sufix, ti))

    # contignames files
    contigs_embedded = []
    embeddings = []

    for cc in nx.connected_components(G):
        for c in cc:
            embeddings.append(model.wv[c])
            contigs_embedded.append(c)

    # embs_dir_run = os.path.join(args.embs_dir, sufix + "/")
    embs_dir_run = args.embs_dir
    try:
        os.makedirs(embs_dir_run)
    except:
        pass

    embs_path = os.path.join(embs_dir_run, "embeddings.npz")

    contigsembs_path = os.path.join(embs_dir_run, "contigs_embedded.txt")

    np.savez(embs_path, np.array(embeddings))
    np.savetxt(contigsembs_path, np.array(contigs_embedded, dtype=object), fmt="%s")
    model.save(os.path.join(embs_dir_run, "word2vec.model"))

    print("Output directory: %s" % embs_dir_run)
