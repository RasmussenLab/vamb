
__doc__ = """Merge clusters with walktrap algorithm.

Input: Path to tab-sep rows of cluster, contig as produced by cluster.py
    
Output: Same as clusterpath, but with merged clusters

This program represents each cluster as a vertex in a graph. A pair of contigs
for which the size of their intersection of sets of contigs divided by the size
of the smaller of the contigs are connected by an edge.

By default, this scripts merges communities of clusters. Import the script as
a Python module to access clique merging and subgraph merging.
"""



import sys as _sys
import os as _os
from collections import defaultdict as _defaultdict
import itertools as _itertools
#import igraph as _igraph
import argparse as _argparse



def read_clusters_from_disk(path):
    """Get mappings from cluster to contigs and back from a clustering file.
    
    Input: Path of a clustering file as produced by clustering.py
    
    Output: A {clustername: set(contignames)} dict
    """
    
    contigsof = _defaultdict(set)
    
    with open(path) as clusterfile:
        for line in clusterfile:     
            line = line.rstrip()
        
            if line == '' or line[0] == '#':
                continue
                
            cluster, contig = line.split()
            contigsof[cluster].add(contig)
    
    return contigsof



def _iter_overlapping_pairs(contigsof, threshold=0.5):
    
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
        
        if overlap > threshold:
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
        u = max((p | x), key=lambda v: len(edges.get(v)))
        for v in p - edges[u]:
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
    loners = list(set(contigsof) - set(edges))
    
    for loner in loners:
        cliques.append({loner})
    
    del loners, edges
    
    # Now simply add the results to a dictionary with new names for clusters.
    mergedclusters = dict()
    
    for i, clique in enumerate(cliques):
        mergedname = 'cluster_' + str(i + 1)
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
        mergedname = 'cluster_' + str(i + 1)
        contigs = set()
        
        for cluster in subgraph:
            contigs.update(contigsof[cluster])
            
        mergedclusters[mergedname] = contigs
        
    return mergedclusters



def write_merged_clusters(path, mergedclusters):
    """Write the merged clusters to a file.
    
    Inputs:
        path: Path to write merged clusters to, tab-sep rows of cluster, contig
        subgraphs: A list of igraph.subgraphs
        
    Output: None
    """
        
    with open(path, 'w') as outfile:
        print('#clustername', '#contigindex(1-indexed)', sep='\t', file=outfile)
        
        for mergedname, contigs in mergedclusters.items():
            separator = '\n{}\t'.format(mergedname)

            print(mergedname, file=outfile, end='\t')
            print(separator.join([str(c) for c in contigs]), file=outfile)



if __name__ == '__main__':
    parserkws = {'prog': 'mergeclusters.py',
                 'formatter_class': _argparse.RawDescriptionHelpFormatter,
                 'usage': '%(prog)s CLUSTERFILE OUTPUTFILE',
                 'description': __doc__}

    # Create the parser
    parser = _argparse.ArgumentParser(**parserkws)

    parser.add_argument('clusterfile', help='clusters from clustering.py')
    parser.add_argument('outputfile', help='output path')
    
    parser.add_argument('-m', dest='min_overlap',
                        help='Minimum cluster overlap [0.5]', default=0.5, type=float)

    # Print help if no arguments are given
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()

    args = parser.parse_args()

    if not _os.path.isfile(args.clusterfile):
        raise FileNotFoundError(args.clusterfile)
        
    if _os.path.exists(args.outputfile):
        raise FileExistsError(args.outputfile)
        
    if args.min_overlap < 0 or args.min_overlap > 1:
        raise ValueError('Min overlap must be 0 < m < 1, not ' + str(args.min_overlap))
        
    contigsof = read_clusters_from_disk(args.clusterfile)
    merged = subgraph_merge(contigsof, args.min_overlap)
    write_merged_clusters(args.outputfile, merged)

