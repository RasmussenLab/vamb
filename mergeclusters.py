
__doc__ = """Merge clusters with walktrap algorithm.

Input: 
    clusterpath: Path to tab-sep rows of cluster, contig as produced by cluster.py
    lengthpath: Path to tab-sep rows of contig, length in same order as clusterpath
    
Output: Same as clusterpath, but with merged clusters

This program represents each cluster as a vertex in a graph. Any two clusters
with more than <threshold> overlap in contigs as counter per basepair of length
is connected with an edge with weight equal to their overlap.
The walktrap algorithm then finds dense subgraphs. These are merged as new
clusters and printed to output file.

Reference for walktrap algorithm: http://arxiv.org/abs/physics/0512106
"""



import sys
import os
from collections import defaultdict
import itertools
import igraph
import argparse



lengthpath = '/mnt/computerome/projects/deep_bin/data/datasets/metabat_errorfree/lengths.txt'
clusterfile = '/home/jakni/Downloads/clustered_0.08_0.10.tsv'



def calculate_overlap(contigs1, contigs2, clusterlength1, clusterlength2, contiglengths):
    """Calculates the overlap of two clusters.
    
    Input:
        contigs1/2: Sets of contigs,
        clusterlength1/2: number, total basepair length of cluster
        contiglengths: {contig: length_of_contig} dictionary
        
    Output:
        Number between 0 and 1, where 1 is maximal overlap.
    """
    
    intersection = contigs1.intersection(contigs2)
    overlaplength = sum(contiglengths[contig] for contig in intersection)
    
    return overlaplength / min(clusterlength1, clusterlength2)



def read_clusters(path):
    """Get mappings from cluster to contigs and back from a clustering file.
    
    Input: Path of a clustering file as produced by clustering.py
    
    Output:
        contigsof: a {clustername: set(contigindices-0-indexed)} dict
        clustersof: a {contigindex-0-indexed: list(clusters with contig)} dict
    """
    
    contigsof = defaultdict(set)
    clustersof = defaultdict(list)
    
    with open(path) as clusterfile:
        for line in clusterfile:
            line = line.rstrip()
        
            if not line:
                continue
                
            cluster, contig = line.split()
            
            # Output is 1-indexed, let's keep things 0-indexed internally
            contig = int(contig) - 1
            
            contigsof[cluster].add(contig)
            clustersof[contig].append(cluster)
    
    return contigsof, clustersof



def read_lengths(path):
    """Get a mapping from contig index to length.
    
    Input: Path of tab-sep clustername, length-of-contig lines.
    
    Output: {contigindex-0-indexed: length-of-contig} dict
    """
    
    lengthofcontig = dict()
    
    with open(path) as lengthfile:
        for linenumber, line in enumerate(lengthfile):
            line = line.rstrip()
        
            if not line:
                continue
                
            contig, length = line.split()
            length = int(length)
            
            # We've kept contig number 0-indexed in read_clusters function
            lengthofcontig[linenumber] = length
            
    return lengthofcontig



def load_graph(threshold, contigsof, clustersof, lengthofcontig, lengthofcluster):
    """Creates the graph representing each cluster and their mututal overlaps.
    
    Input:
        threshold: A float with minimum overlap for graph edge (0 ≤ x ≤ 1)
        contigsof: As returned by read_clusters
        clustersof: As returned by read_clusters
        lengthofcontig: As returned by read_lengths
        lengthofcluster: A {clustername: total-length-of-cluster} dict
        
    Output:
        Graph with each vertex being a cluster and each edge their overlap.
    """
    
    graph = igraph.Graph()
    graph.es["weight"] = True

    graph.add_vertices(list(contigsof))
    
    pairschecked = set()

    for clusterlist in clustersof.values():
        for clusterpair in itertools.combinations(clusterlist, 2):
            if clusterpair in pairschecked:
                continue

            cluster1, cluster2 = clusterpair
            pairschecked.add(clusterpair)
            
            contigs1 = contigsof[cluster1]
            contigs2 = contigsof[cluster2]
            
            clusterlength1 = lengthofcluster[cluster1]
            clusterlength2 = lengthofcluster[cluster2]

            overlap = calculate_overlap(contigs1, contigs2, clusterlength1, 
                                        clusterlength2, lengthofcontig)

            if overlap > threshold:
                graph[cluster1, cluster2] = overlap
    
    return graph



def walktrap(graph):
    """Do the walktrap algorithm on the graph.
    
    Input: The graph from load_graph
    
    Output: A list of subgraphs.
    """
    
    dendrogram = graph.community_walktrap(weights='weight')
    clusters = dendrogram.as_clustering()
    subgraphs = clusters.subgraphs()
    
    return subgraphs



def main(clusterpath, lengthspath, outputpath, min_overlap):
    """Do everything, see the script documentation."""
    
    contigsof, clustersof = read_clusters(clusterpath)
    lengthofcontig = read_lengths(lengthspath)
    
    lengthofcluster = dict()
    for cluster, contigs in contigsof.items():
        lengthofcluster[cluster] = sum(lengthofcontig[contig] for contig in contigs)
    
    graph = load_graph(min_overlap, contigsof, clustersof, lengthofcontig, lengthofcluster)
    subgraphs = walktrap(graph)
    
    with open(outputpath, 'w') as outfile:
        print('#clustername', '#contigindex(1-indexed)', sep='\t', file=outfile)
        
        for subgraph_number, subgraph in enumerate(subgraphs):
            contigs = set()

            for cluster in subgraph.vs['name']:
                contigs.update(set(contigsof[cluster]))
                
            clustername = 'C' + str(subgraph_number + 1)
            separator = '\n{}\t'.format(clustername)

            print(clustername, file=outfile, end='\t')
            print(separator.join([str(c) for c in contigs]), file=outfile)



if __name__ == '__main__':
    parserkws = {'prog': 'mergeclusters.py',
                 'formatter_class': argparse.RawDescriptionHelpFormatter,
                 'usage': '%(prog)s CLUSTERFILE LENGTHFILE OUTPUTFILE',
                 'description': __doc__}

    # Create the parser
    parser = argparse.ArgumentParser(**parserkws)

    parser.add_argument('clusterfile', help='clusters from clustering.py')
    parser.add_argument('lengthfile', help='file with length of contigs')
    parser.add_argument('outputfile', help='output path')
    
    parser.add_argument('-m', dest='min_overlap',
                        help='Minimum cluster overlap [0.5]', default=0.5, type=float)

    # Print help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if not os.path.isfile(args.clusterfile):
        raise FileNotFoundError(args.clusterfile)

    if not os.path.isfile(args.lengthfile):
        raise FileNotFoundError(args.lengthfile)
        
    if os.path.exists(args.outputfile):
        raise FileExistsError(args.outputfile)
        
    if args.min_overlap < 0 or args.min_overlap > 1:
        raise ValueError('Min overlap must be 0 < m < 1, not ' + str(args.min_overlap))

    main(args.clusterfile, args.lengthfile, args.outputfile, args.min_overlap)

