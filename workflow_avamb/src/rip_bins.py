from Bio import SeqIO
import os
import numpy as np
from collections import OrderedDict
import argparse
import networkx as nx
import vamb
import json

from collections.abc import Sequence
from typing import cast


def main(
    contig_names: Sequence[str],
    contig_lengths: Sequence[int],
    cluster_contigs: dict[str, set[str]],
    bin_path: dict[str, str],
    path_ripped: str,
    path_clusters_not_ripped: str,
) -> dict[str, str]:
    cluster_contigs_original = cluster_contigs.copy()

    contig_length = get_contig_length(contig_names, contig_lengths)
    cluster_length = get_cluster_length(cluster_contigs, contig_length)
    graph_clusters = get_graph_clusters(cluster_contigs, contig_length, cluster_length)
    (
        cluster_contigs,
        graph_clusters,
        cluster_length,
        clusters_changed_but_not_intersecting_contigs,
    ) = remove_meaningless_edges_from_pairs(
        graph_clusters, cluster_length, cluster_contigs, contig_length
    )

    (
        cluster_contigs,
        graph_clusters,
        cluster_length,
        clusters_changed_but_not_intersecting_contigs_2,
    ) = make_all_components_pair(
        graph_clusters, cluster_contigs, cluster_length, contig_length
    )

    ripped_clusters_set = create_ripped_clusters_and_write_ripped_bins(
        graph_clusters, cluster_contigs, bin_path, path_ripped
    )

    clusters_changed_but_not_intersecting_contigs_total = (
        clusters_changed_but_not_intersecting_contigs.copy()
    )
    clusters_changed_but_not_intersecting_contigs_total.update(
        clusters_changed_but_not_intersecting_contigs_2
    )

    (
        bin_path,
        cluster_contigs_not_ripped,
    ) = find_remaining_clusters_ripped_and_write_ripped_bins(
        cluster_contigs_original,
        clusters_changed_but_not_intersecting_contigs_total,
        bin_path,
        path_ripped,
    )

    # Remove the clusters that have been ripped because of removing the intersections
    for cluster in ripped_clusters_set:
        if cluster in cluster_contigs_not_ripped.keys():
            print("%s cluster removed from the set since it was ripped" % (cluster))
            del cluster_contigs_not_ripped[cluster]

    with open(path_clusters_not_ripped, "w") as file:
        for cluster, contigs_nr in cluster_contigs_not_ripped.items():
            contigs_o = cluster_contigs_original[cluster]
            if contigs_o != contigs_nr:
                raise ValueError(
                    "Clusters that have been ripped remain in the non ripped set of clusters"
                )
            for contig in contigs_nr:
                print(cluster, contig, sep="\t", file=file)
            file.flush()
    return bin_path


def get_contig_length(names: Sequence[str], lengths: Sequence[int]) -> dict[str, int]:
    return dict(zip(names, lengths))


def get_cluster_length(
    cluster_contigs: dict[str, set[str]], contig_length: dict[str, int]
) -> OrderedDict[str, int]:
    cluster_length: OrderedDict[str, int] = OrderedDict()
    for cluster_name, contigs in cluster_contigs.items():
        cluster_length[cluster_name] = 0
        for contig in contigs:
            cluster_length[cluster_name] += contig_length[contig]
    return cluster_length


def get_graph_clusters(
    cluster_contigs: dict[str, set[str]],
    contig_length: dict[str, int],
    cluster_length: OrderedDict[str, int],
):
    """Generate a graph where each cluster is a node and each edge connecting 2 nodes is
    created if clusters share some contigs, edges are weighted by the intersection length
    """
    graph_clusters = nx.Graph()
    for i, cluster_i in enumerate(cluster_contigs.keys()):
        for j, cluster_j in enumerate(cluster_contigs.keys()):
            if j <= i:
                j += 1
                continue

            cluster_i_contigs, cluster_j_contigs = (
                cluster_contigs[cluster_i],
                cluster_contigs[cluster_j],
            )
            cluster_i_j_contigs_intersection = cluster_i_contigs.intersection(
                cluster_j_contigs
            )

            if len(cluster_i_j_contigs_intersection) > 0:
                cluster_i_length, cluster_j_length = (
                    cluster_length[cluster_i],
                    cluster_length[cluster_j],
                )
                cluster_i_j_contigs_intersection_length = sum(
                    (
                        contig_length[contig_k]
                        for contig_k in cluster_i_j_contigs_intersection
                    )
                )
                intersection_length_ratio = (
                    cluster_i_j_contigs_intersection_length
                    / min(cluster_i_length, cluster_j_length)
                )
                graph_clusters.add_node(cluster_i)
                graph_clusters.add_node(cluster_j)
                graph_clusters.add_edge(
                    cluster_i, cluster_j, weight=intersection_length_ratio
                )
    return graph_clusters


def remove_meaningless_edges_from_pairs(
    graph_clusters: nx.Graph,
    cluster_length: OrderedDict[str, int],
    cluster_contigs: dict[str, set[str]],
    contig_length: dict[str, int],
    weight_thr: float = 0.001,
):
    """Iterate over all the components of the graph and remove components composed by 2 node/clusters
    according the following criteria:
        - For edges with weight smaller than threshold, just remove edge and remove intersecting contigs from
          larger cluster
    """
    components: list[set[str]] = list()

    for component in nx.connected_components(graph_clusters):
        components.append(component)

    clusters_changed_but_not_intersecting_contigs: dict[str, set[str]] = dict()
    for component in components:
        if len(component) == 2:
            cl_i = component.pop()
            cl_j = component - {cl_i}
            assert len(cl_j) == 1
            cl_j = cl_j.pop()
            cl_i_cl_j_edge_weight: float = graph_clusters.get_edge_data(cl_i, cl_j)[
                "weight"
            ]
            if cl_i_cl_j_edge_weight > weight_thr:
                pass
            else:
                cluster_updated = move_intersection_to_smaller_cluster(
                    graph_clusters,
                    cl_i,
                    cl_j,
                    cluster_contigs,
                    cluster_length,
                    contig_length,
                )
                print("Cluster ripped because of a meaningless edge ", cluster_updated)
                clusters_changed_but_not_intersecting_contigs[
                    cluster_updated
                ] = cluster_contigs[cluster_updated]

    components: list[set[str]] = list()
    for component in nx.connected_components(graph_clusters):
        components.append(component)

    for component in components:
        if len(component) == 1:
            cl_i = component.pop()
            graph_clusters.remove_node(cl_i)

    print("Meaningless edges removed from pairs")

    return (
        cluster_contigs,
        graph_clusters,
        cluster_length,
        clusters_changed_but_not_intersecting_contigs,
    )


def move_intersection_to_smaller_cluster(
    graph_bins: nx.Graph,
    cl_i: str,
    cl_j: str,
    cluster_contigs: dict[str, set[str]],
    cluster_length: OrderedDict[str, int],
    contig_length: dict[str, int],
) -> str:
    """Move the intersecting contigs to the shortest cluster"""
    cl_i_contigs, cl_j_contigs = cluster_contigs[cl_i], cluster_contigs[cl_j]
    cl_i_cl_j_intersection_contigs = cl_i_contigs.intersection(cl_j_contigs)
    cl_i_length, cl_j_length = cluster_length[cl_i], cluster_length[cl_j]

    if cl_i_length < cl_j_length:
        cluster_contigs[cl_j] = cl_j_contigs - cl_i_cl_j_intersection_contigs
        cluster_length[cl_j] = sum(
            [contig_length[contig_i] for contig_i in cluster_contigs[cl_j]]
        )
        cluster_updated = cl_j
    else:
        cluster_contigs[cl_i] = cl_i_contigs - cl_i_cl_j_intersection_contigs
        cluster_length[cl_i] = sum(
            [contig_length[contig_i] for contig_i in cluster_contigs[cl_i]]
        )
        cluster_updated = cl_i
    graph_bins.remove_edge(cl_i, cl_j)

    return cluster_updated


def make_all_components_pair(
    graph_clusters: nx.Graph,
    cluster_contigs: dict[str, set[str]],
    cluster_length: OrderedDict[str, int],
    contig_length: dict[str, int],
):
    """Iterate over all graph components that contain more than 2 nodes/clusters and break them down to
    2 nodes/clusters components according the following criteria:
        - Remove the weaker edge by removing the intersecting contigs from the larger cluster
    """

    components: list[set[str]] = list()
    for component in nx.connected_components(graph_clusters):
        components.append(component)

    clusters_changed_but_not_intersecting_contigs: dict[str, set[str]] = dict()
    for component in components:
        if len(component) > 2:
            weight_edges: dict[float, set[str]] = dict()
            for cl_i in component:
                for cl_j in component - {cl_i}:
                    if graph_clusters.has_edge(cl_i, cl_j):
                        cl_i_cl_j_edge_weight: float = graph_clusters.get_edge_data(
                            cl_i, cl_j
                        )["weight"]
                        # if the weight as key in the dict and is mapping to different clusters (nodes)
                        # then add a insignificant value to the weight so we can add the weight with
                        # different clusters (nodes)
                        if (
                            cl_i_cl_j_edge_weight in weight_edges.keys()
                            and weight_edges[cl_i_cl_j_edge_weight] != {cl_i, cl_j}
                        ):
                            cl_i_cl_j_edge_weight += 1 * 10**-30
                        weight_edges[cl_i_cl_j_edge_weight] = {cl_i, cl_j}

            dsc_sorted_weights: list[float] = sorted(list(weight_edges.keys()))

            i = 0
            component_len = len(component)

            print(
                dsc_sorted_weights,
                len(dsc_sorted_weights),
                len(component),
                weight_edges.keys(),
            )
            while component_len > 2:
                weight_i = dsc_sorted_weights[i]
                cl_i, cl_j = weight_edges[weight_i]
                print(weight_i, cl_i, cl_j)
                cluster_updated = move_intersection_to_smaller_cluster(
                    graph_clusters,
                    cl_i,
                    cl_j,
                    cluster_contigs,
                    cluster_length,
                    contig_length,
                )
                print("Cluster ripped because of a pairing component ", cluster_updated)
                clusters_changed_but_not_intersecting_contigs[
                    cluster_updated
                ] = cluster_contigs[cluster_updated]
                component_len = max(
                    [
                        len(nx.node_connected_component(graph_clusters, node_i))
                        for node_i in component
                    ]
                )

                i += 1

    components: list[set[str]] = list()
    for component in nx.connected_components(graph_clusters):
        components.append(component)

    for component in components:
        assert len(component) < 3

        if len(component) == 1:
            cl_i = component.pop()
            graph_clusters.remove_node(cl_i)
    print("All components are reduced to 2 nodes per component")
    return (
        cluster_contigs,
        graph_clusters,
        cluster_length,
        clusters_changed_but_not_intersecting_contigs,
    )


def create_ripped_clusters_and_write_ripped_bins(
    graph_clusters: nx.Graph,
    cluster_contigs: dict[str, set[str]],
    bin_path: dict[str, str],
    path_ripped: str,
) -> set[str]:
    """Given the components of the graph, write ripped bins where each bin misses the intersection with the other bin"""
    ripped_clusters_set: set[str] = set()
    try:
        os.mkdir(path_ripped)
    except FileExistsError:
        pass

    for component in nx.connected_components(graph_clusters):
        cl_i = next(iter(component))
        cl_j = component - {cl_i}
        assert len(cl_j) == 1
        cl_j = next(iter(cl_j))
        cl_i_contigs, cl_j_contigs = cluster_contigs[cl_i], cluster_contigs[cl_j]
        cluster_i_j_contigs_intersection = cl_i_contigs.intersection(cl_j_contigs)
        print(
            "Cluster %s and cluster %s share %i contigs "
            % (cl_i, cl_j, len(cluster_i_j_contigs_intersection))
        )

        bin_i_path = bin_path[cl_i + ".fna"]
        bin_j_path = bin_path[cl_j + ".fna"]

        bin_i_ripped_path = os.path.join(path_ripped, cl_i + "--" + cl_j + ".fna")
        bin_j_ripped_path = os.path.join(path_ripped, cl_j + "--" + cl_i + ".fna")
        print(bin_i_path, bin_i_ripped_path)
        write_ripped_bin(
            cl_i_contigs - cluster_i_j_contigs_intersection,
            bin_i_path,
            bin_i_ripped_path,
        )

        print(bin_j_path, bin_j_ripped_path)
        write_ripped_bin(
            cl_j_contigs - cluster_i_j_contigs_intersection,
            bin_j_path,
            bin_j_ripped_path,
        )

        ripped_clusters_set.add(cl_i)
        ripped_clusters_set.add(cl_j)

    if len(os.listdir(path_ripped)) == 0:
        os.rmdir(path_ripped)
    print("Ripped bins created from components")
    return ripped_clusters_set


def write_ripped_bin(
    contigs_cluster_ripped, bin_i_path: str, bin_i_ripped_path: str
) -> None:
    """Write ripped bin"""
    contig_sequences = []
    for contig in SeqIO.parse(bin_i_path, "fasta"):
        if contig.id in contigs_cluster_ripped:
            contig_sequences.append(contig)
    with open(bin_i_ripped_path, "w") as output_handle:
        SeqIO.write(contig_sequences, output_handle, "fasta")


def find_remaining_clusters_ripped_and_write_ripped_bins(
    cluster_contigs_original: dict[str, set[str]],
    clusters_changed_but_not_intersecting_contigs_total: dict[str, set[str]],
    bin_path: dict[str, str],
    path_ripped: str,
):
    """Compare cluster_contigs_ripped_only_by_meaningless_edges_or_not_ripped_at_all with cluster_contigs
    to find the clusters that remain different, so we can re evaluate them with checkM2, those clusters
    are the product of the meaningless ripping or the parition of n>2 components to generate
    all components n<=2"""
    cluster_contigs_original_copy = cluster_contigs_original.copy()

    try:
        os.mkdir(path_ripped)
    except FileExistsError:
        pass

    for (
        cluster,
        contigs_r,
    ) in clusters_changed_but_not_intersecting_contigs_total.items():
        contigs_o = cluster_contigs_original[cluster]

        assert contigs_r != contigs_o
        assert len(contigs_r) < len(contigs_o)
        bin_o_path = bin_path[cluster + ".fna"]
        bin_r_path = os.path.join(path_ripped, cluster + ".fna")
        write_ripped_bin(contigs_r, bin_o_path, bin_r_path)
        bin_path[cluster + ".fna"] = bin_r_path
        del cluster_contigs_original_copy[cluster]

    return bin_path, cluster_contigs_original_copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, help="path run generated by AVAMB")
    parser.add_argument("--ci", type=str, help="path dereplicated clusters file")
    parser.add_argument(
        "--co", type=str, help="path dereplicated clusters without ripped clusters"
    )
    parser.add_argument("-l", type=str, help="path to contig lengths")
    parser.add_argument("-n", type=str, help="path to contig names")
    parser.add_argument("--bp_d", type=str, help="bin_path dictionary path")
    parser.add_argument("--br", type=str, help="path ripped bins")
    parser.add_argument("--bin_separator", type=str, help="path ripped bins")
    parser.add_argument(
        "--log_nc_ripped_bins",
        type=str,
        help="path to og file that contains the # of ripped bins",
    )

    opt = parser.parse_args()

    path_run = (
        opt.r
    )  # '/Users/mbv396/aamb/vamb_aamb_on_almeida_hmp2/runs/vamb_aamb_on_almeida/'

    clusters_path = (
        opt.ci
    )  # os.path.join(path_run,'aae_l_y_vamb_manual_drep_85_10_clusters.tsv')
    clusters_not_ripped_path = (
        opt.co
    )  # os.path.join(path_run,'aae_l_y_vamb_manual_drep_85_10_not_ripped_clusters.tsv')

    with open(clusters_path) as file:
        cluster_contigs = vamb.vambtools.read_clusters(file)

    contig_lengths_file = opt.l
    contig_lengths = cast(Sequence[int], np.load(contig_lengths_file)["arr_0"])

    contignames_file = opt.n
    contig_names = cast(Sequence[str], np.loadtxt(contignames_file, dtype=object))

    path_ripped = cast(str, opt.br)

    with open(opt.bp_d) as f:
        bin_path = json.load(f)

    bin_separator = opt.bin_separator

    bin_path_ = main(
        contig_names,
        contig_lengths,
        cluster_contigs,
        bin_path,
        path_ripped,
        clusters_not_ripped_path,
    )

    with open(opt.bp_d, "w") as f:
        json.dump(bin_path_, f)

    n_bins_ripped_in_folder = 0
    for _file in os.listdir(path_ripped):
        if ".fna" in _file:
            n_bins_ripped_in_folder += 1

    with open(opt.log_nc_ripped_bins, "w") as file:
        file.write(str(n_bins_ripped_in_folder))
