
__doc__ = """Merge clusters with walktrap algorithm.

Input: Path to tab-sep rows of cluster, contig as produced by cluster.py
    
Output: Same as clusterpath, but with merged clusters

This program represents each cluster as a vertex in a graph. Any two clusters
with more than <threshold> overlap in contigs as counter per basepair of length
is connected with an edge with weight equal to their overlap.
The walktrap algorithm then finds dense subgraphs. These are merged as new
clusters and printed to output file.

Reference for walktrap algorithm: http://arxiv.org/abs/physics/0512106
"""



import sys as _sys
import os as _os
from collections import defaultdict as _defaultdict
import itertools as _itertools
import igraph as _igraph
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



def graph_from_clusters(contigsof, threshold):
    """Creates the graph representing each cluster and their mututal overlaps.
    
    Input:
        contigsof: A {clustername: set(contignames)} dict
        threshold: Minimum fraction of overlapping contigs to create edge
        
    Output: Graph with each vertex being a cluster and each edge their overlap.
    """

    pairs = set()
    clustersof = _defaultdict(list)
    
    for clustername, contigs in contigsof.items():
        for contig in contigs:
            clustersof[contig].append(clustername)
            
    for clusterlist in clustersof.values():
        pairs.update(set(_itertools.combinations(clusterlist, 2)))
    
    del clustersof
    
    # Create a weighted graph
    graph = _igraph.Graph()
    graph.es["weight"] = True

    # Add all the clusters as vertices
    graph.add_vertices(list(contigsof))
    
    for cluster1, cluster2 in pairs:           
            contigs1 = contigsof[cluster1]
            contigs2 = contigsof[cluster2]
            intersection = contigs1.intersection(contigs2)
            overlap = len(intersection) / min(len(contigs1), len(contigs2))

            if overlap > threshold:
                graph[cluster1, cluster2] = overlap
    
    return graph



def walktrap(graph, contigsof):
    """Do the walktrap algorithm on the graph.
    
    Inputs:
        graph: Graph with each vertex being a cluster and each edge their overlap
        contigsof: A {clustername: set(contignames)} dict
    
    Output: A {clustername: set(contignames)} dict
    """
    
    merged = dict()
    
    # If *all* contigs are disjoint, just return input
    if len(graph.es['weight']) == 0:
        return contigsof
        
    dendrogram = graph.community_walktrap(weights='weight')
    clusters = dendrogram.as_clustering()
    subgraphs = clusters.subgraphs()

    for subgraphnumber, subgraph in enumerate(subgraphs):
        mergedname = 'cluster_' + str(subgraphnumber + 1)
        mergedcluster = set()
        
        for cluster in subgraph.vs['name']:
            mergedcluster.update(contigsof[cluster])
            
        merged[mergedname] = mergedcluster
    
    return merged



def write_merged_clusters(path, mergedclusters):
    """Write the merged clusters to a file.
    
    Inputs:
        path: Path to write merged clusters to, tab-sep rows of cluster, contig
        subgraphs: A list of igraph.subgraphs
        
    Output: None
    """
        
    with open(outputpath, 'w') as outfile:
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
    graph = graph_from_clusters(contigsof, args.min_overlap)
    merged = walktrap(graph, contigsof)
    write_merged_clusters(args.outputfile, merged)

