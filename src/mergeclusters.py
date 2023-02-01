__doc__ = """Merge clusters according to overlap in contigs.

This module is obsolete in the current version of Vamb, as VAMB now creates
disjoint clusters. We have retained it here because in the future, it might be
of interest to return to producing overlapping clusters to improve accuracy
and/or speed up clustering.

This module contains four merging functions, of which one is commented out
because it requires the python-igraph package. For all functions, the set of
clusters are converted to a graph with each vertex being a cluster and two
clusters are connected with an edge if the intersection of contigs divided
by the smaller clusters exceeds a threshold.

The four merging functions are:

community_merge: (commented out, requires python-igraph):
    Merges nodes which form a community according to walktrap algorithm

clique_merge:
    Merges all maximal cliques as new clusters. Note that this may increase
    rather than reduce the number of clusters.

subgraph_merge:
    Merges sets of nodes which form a connected subgraph

hierarchical_merge:
    Merges pairs of nodes together from most to least connected
"""


import sys as _sys
import os as _os
import itertools as _itertools
import bisect as _bisect
from collections import defaultdict as _defaultdict

# Uncomment this (& install python-igraph) to get community (walktrap) merging.
# On sparse graphs this gives the same result as subgraph_merge, so no need.
# import igraph as _igraph


def _iter_overlapping_pairs(contigsof, threshold=0.5):
    """This creates a generator that yields (clst1, clst2), overlap
    for all pairs of clusters with nonzero overlap."""

    pairs = set()
    clustersof = _defaultdict(list)

    for clustername, contigs in contigsof.items():
        for contig in contigs:
            clustersof[contig].append(clustername)

    for clusterlist in clustersof.values():
        pairs.update(set(_itertools.combinations(clusterlist, 2)))

    del clustersof

    while pairs:
        cluster1, cluster2 = pairs.pop()
        contigs1 = contigsof[cluster1]
        contigs2 = contigsof[cluster2]
        intersection = contigs1.intersection(contigs2)
        overlap = len(intersection) / min(len(contigs1), len(contigs2))

        if overlap >= threshold:
            yield (cluster1, cluster2), overlap


# def comminuty_merge(contigsof, threshold=0.5, steps=4):
#     """Merges all communities of clusters using the Walktrap algorithm(1)
#     (1) http://arxiv.org/abs/physics/0512106

#     Inputs:
#         contigsof: A {clustername: set(contignames)} dict
#         threshold [0.5]: Minimum fraction of overlapping contigs to create edge
#         steps [4]: Number of random walking steps in the walktrap

#     Output: A {clustername: set(contignames)} dict
#     """

#     # Create a weighted graph
#     graph = _igraph.Graph()
#     graph.es["weight"] = True

#     # Add all the clusters as vertices
#     graph.add_vertices(list(contigsof))

#     for (cluster1, cluster2), overlap in _iter_overlapping_pairs(contigsof, threshold):
#         graph[cluster1, cluster2] = overlap

#     merged = dict()

#     # If *all* contigs are disjoint, just return input
#     if len(graph.es['weight']) == 0:
#         return contigsof

#     dendrogram = graph.community_walktrap(weights='weight', steps=steps)
#     clusters = dendrogram.as_clustering()
#     subgraphs = clusters.subgraphs()

#     for subgraphnumber, subgraph in enumerate(subgraphs):
#         mergedname = 'cluster_' + str(subgraphnumber + 1)
#         mergedcluster = set()

#         for cluster in subgraph.vs['name']:
#             mergedcluster.update(contigsof[cluster])

#         merged[mergedname] = mergedcluster

#     return merged


def _bron_kerbosch(r, p, x, cliques, edges):
    """Finds all maximal cliques in a graph"""

    # https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm#With_pivoting
    if p or x:
        pivot = max((p | x), key=lambda v: len(edges.get(v)))
        for v in p - edges[pivot]:
            _bron_kerbosch(r | {v}, p & edges[v], x & edges[v], cliques, edges)
            p.remove(v)
            x.add(v)

    else:
        cliques.append(r)


def clique_merge(contigsof, threshold=0.5):
    """Merges all maximal cliques of clusters.

    Inputs:
        contigsof: A {clustername: set(contignames)} dict
        threshold [0.5]: Minimum fraction of overlapping contigs to create edge

    Output: A {clustername: set(contignames)} dict
    """

    # Calculate all edges between the vertices
    edges = _defaultdict(set)
    for (cluster1, cluster2), overlap in _iter_overlapping_pairs(contigsof, threshold):
        edges[cluster1].add(cluster2)
        edges[cluster2].add(cluster1)

    # Find all maximal 2-cliques or larger w. Bron-Kerbosch algorithm
    cliques = list()
    _bron_kerbosch(set(), set(edges), set(), cliques, edges)

    # All maximal 1-cliques (i.e. vertices with degree zero) are added
    for loner in list(set(contigsof) - set(edges)):
        cliques.append({loner})

    del edges

    # Now simply add the results to a dictionary with new names for clusters.
    mergedclusters = dict()

    for i, clique in enumerate(cliques):
        mergedname = "cluster_" + str(i + 1)
        contigs = set()

        for cluster in clique:
            contigs.update(contigsof[cluster])

        mergedclusters[mergedname] = contigs

    return mergedclusters


def subgraph_merge(contigsof, threshold=0.5):
    """Merges all connected graphs of clusters together.

    Inputs:
        contigsof: A {clustername: set(contignames)} dict
        threshold [0.5]: Minimum fraction of overlapping contigs to create edge

    Output: A {clustername: set(contignames)} dict
    """

    # Calculate all edges between the vertices
    subgraphs = {cluster: {cluster} for cluster in contigsof}

    for (cluster1, cluster2), overlap in _iter_overlapping_pairs(contigsof, threshold):
        subgraph = subgraphs[cluster1] | subgraphs[cluster2]

        for cluster in subgraph:
            subgraphs[cluster] = subgraph

    deduplicated_subgraphs = {frozenset(subgraph) for subgraph in subgraphs.values()}

    del subgraphs

    mergedclusters = dict()

    for i, subgraph in enumerate(deduplicated_subgraphs):
        mergedname = "cluster_" + str(i + 1)
        contigs = set()

        for cluster in subgraph:
            contigs.update(contigsof[cluster])

        mergedclusters[mergedname] = contigs

    return mergedclusters


def hierarchical_merge(contigsof, threshold=0.5):
    """Merges pairs together from most to least connected.

    Inputs:
        contigsof: A {clustername: set(contignames)} dict
        threshold [0.5]: Minimum fraction of overlapping contigs to create edge

    Output: A {clustername: set(contignames)} dict
    """

    # Make clustersof: {contig: set(clusters)} and existing set
    clustersof = _defaultdict(set)
    existing = set()
    for cluster, contigs in contigsof.items():
        frozenset_ = frozenset(contigs)
        existing.add(frozenset_)
        for contig in contigs:
            clustersof[contig].add(frozenset_)

    # Make sorted list S of overlaps, (pair) of sets
    overlaps = list()
    for (name1, name2), overlap in _iter_overlapping_pairs(contigsof, threshold):
        set1 = frozenset(contigsof[name1])
        set2 = frozenset(contigsof[name2])
        overlaps.append(((overlap), (set1, set2)))

    overlaps.sort()

    # For each overlap, (set1, set2) in S
    while overlaps:
        overlap, (set1, set2) = overlaps.pop()

        # Skip if set1 or set2 does not exist
        if set1 not in existing or set2 not in existing:
            continue

        # Make newset and remove old from existing
        newset = set1 | set2

        for contig in set1:
            clustersof[contig].remove(set1)

        for contig in set2:
            clustersof[contig].discard(set2)  # no error for set(set1 & set2)

        existing.remove(set1)
        existing.discard(set2)  # don't raise error if set1 == set2

        # Use clustersof to get list of all sets which may have some overlap with newset
        # We only do this to avoid checking for overlap with *every* existing set
        overlapping = set.union(*(clustersof[contig] for contig in newset))

        for set_ in overlapping:
            intersection = newset.intersection(set_)
            overlap = len(intersection) / min(len(newset), len(set_))
            if overlap >= threshold:
                newelement = (overlap, (newset, set_))
                _bisect.insort(overlaps, newelement)

        # Update existing
        existing.add(newset)

        # Update clustersof
        for contig in newset:
            clustersof[contig].add(newset)

    del clustersof

    mergedclusters = dict()

    for number, cluster in enumerate(existing):
        mergedname = "cluster_" + str(number + 1)
        mergedclusters[mergedname] = set(cluster)

    return mergedclusters
