
#!/usr/bin/env python3

__doc__ = """Iterative medoid clustering of Numpy arrays.

Implements two core functions: cluster and tandemcluster, along with the helper
functions writeclusters and readclusters.
For all functions in this module, a collection of clusters are represented as
a {clustername, set(elements)} dict.

cluster algorithm:
(1): Pick random seed observation S
(2): Define inner_obs(S) = all observations with Pearson distance from S < INNER
(3): Sample MOVES observations I from inner_obs
(4): If any inner_obs(i) > inner_obs(S) for i in I: Let S be i, go to (2)
     Else: Outer_obs(S) = all observations with Pearson distance from S < OUTER
(5): Output outer_obs(S) as cluster, remove inner_obs(S) from observations
(6): If no more observations or MAX_CLUSTERS have been reached: Stop
     Else: Go to (1)

tandemcluster agorithm:
(1): Pick random observation S
(2): Define inner_obs(S) = 10,000 closest obs to S
(3): Define outer_obs(S) = 20,000 closest obs to S
(4): Remove inner_obs(S) from dataset
(5): Cluster outer_obs(S) with algorithm above, add result to clusters.
(6): If more than 20,000 observations remain: Go to (1)
(7): Assign each obs to the largest cluster it features in
"""

cmd_doc = """Iterative medoid clustering.

    Input: An observations x features matrix in text format.
    Output: Tab-sep lines with clustername, observation(1-indexed).
"""

import sys as _sys
import os as _os
import numpy as _np
from collections import defaultdict as _defaultdict

if __name__ == '__main__':
    import argparse



def _checklabels(labels, length):
    """If labels are None, returns an array of integers 1 to length.
    Else checks that labels are already of correct length and unique"""
    
    if labels is None:
        labels = _np.arange(length) + 1
        
    elif type(labels) != _np.ndarray or len(labels) != length:
        raise ValueError('labels must be a 1D Numpy array with same length as matrix')
        
    if len(set(labels)) != length:
        raise ValueError('Labels must be unique')
        
    return labels



def _normalize(matrix, spearman=False):
    """Returns a new, z-score normalized matrix across axis 1"""
    
    if spearman:
        # Two successive argsorts gives the elementwise ranking
        matrix = _np.argsort(matrix, axis=1) 
        matrix = _np.argsort(matrix, axis=1)
    
    normalized = matrix - matrix.mean(axis=1).reshape((len(matrix), 1))
    std = _np.std(normalized, axis=1).reshape((len(normalized), 1))
    std[std == 0] = 1
    normalized /= std
    
    return normalized



def _distances_to_vector(matrix, index):
    """Calculates the Pearson distances from row `index` to all rows
    in the matrix, including itself. Returns numpy array of distances"""
    
    # Distance D = (P - 1) / -2, where P is Pearson correlation coefficient.
    # For two vectors x and y with numbers xi and yi,
    # P = sum((xi-x_mean)*(yi-y_mean)) / (std(y) * std(x) * len(x)).
    # If we normalize matrix so x_mean = y_mean = 0 and std(x) = std(y) = 1,
    # this reduces to sum(xi*yi) / len(x) = x @ y.T / len(x) =>
    # D = ((x @ y.T) / len(x)) - 1) / -2 =>
    # D = (x @ y.T - len(x)) * (-1 / 2len(x))
    
    # Matrix should have already been zscore normalized by axis 1 (subtract mean, div by std)
    vectorlength = matrix.shape[1]
    result = _np.dot(matrix, matrix[index].T)
    result -= vectorlength
    result *= -1 / (2 * vectorlength)
    
    return result



def _getinner(matrix, point, inner_threshold):
    """Gets the distance vector, array of inner points and average distance
    to inner points from a starting point"""
    
    distances = _distances_to_vector(matrix, point)
    inner_points = _np.where(distances < inner_threshold)[0]
    
    if len(inner_points) == 1:
        average_distance = 0
    else:
        average_distance = _np.sum(distances[inner_points]) / (len(inner_points) - 1)

    return distances, inner_points, average_distance



def _sample_clusters(matrix, point, max_attempts, inner_threshold, outer_threshold, randomstate):
    """Keeps sampling new points within the inner points until it has sampled
    max_attempts without getting a new set of inner points with lower average
    distance"""
    
    futile_attempts = 0
    
    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}
    
    distances, inner_points, average_distance = _getinner(matrix, point, inner_threshold)
    
    while len(inner_points) - len(tried) > 0 and futile_attempts < max_attempts:
        sample = randomstate.choice(inner_points)
        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = randomstate.choice(inner_points)
            
        tried.add(sample)
        
        inner = _getinner(matrix, sample, inner_threshold)
        sample_dist, sample_inner, sample_average =  inner
        
        if sample_average < average_distance:
            point = sample
            inner_points = sample_inner
            average_distance = sample_average
            distances = sample_dist
            futile_attempts = 0
            tried = {point}
            
        else:
            futile_attempts += 1
            
    outer_points = _np.where(distances < outer_threshold)[0]
    
    return point, inner_points, outer_points



def _cluster(matrix, labels, inner_threshold, outer_threshold, max_steps):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""

    randomstate = _np.random.RandomState(324645)
    
    # This list keeps track of points to remove because they compose the inner circle
    # of clusters. We only remove points when we create a cluster with more than one
    # point, since one-point-clusters don't interfere with other clusters and the
    # point removal operations are expensive.
    toremove = list()
    
    # This index keeps track of which point we initialize clusters from.
    # It's necessary since we don't remove points after every cluster.
    seed_index = 0
    
    # We initialize clusters from most extreme to less extreme. This is
    # arbitrary and just to have some kind of reproducability.
    # Note to Simon: Sorting by means makes no sense with normalized rows.
    extremes = _np.max(matrix, axis=1)
    argextremes = _np.argsort(extremes)
    
    while len(matrix) > 0:           
        # Most extreme point (without picking same point twice)
        seed = argextremes[-seed_index -1]
        
        # Find medoid using iterative sampling function above
        sampling = _sample_clusters(matrix, seed, max_steps, inner_threshold, outer_threshold, randomstate)
        medoid, inner_points, outer_points = sampling
        
        # Write data to output
        yield labels[medoid], set(labels[outer_points])
            
        seed_index += 1
        
        for point in inner_points:
            toremove.append(point)

        # Only remove points if we have more than 1 point in cluster
        # Note that these operations are really expensive.
        if len(inner_points) > 1 or len(argextremes) == seed_index:
            matrix = _np.delete(matrix, toremove, 0)
            labels = _np.delete(labels, toremove, 0)
            extremes = _np.delete(extremes, toremove, 0)
            argextremes = _np.argsort(extremes)
            seed_index = 0
            toremove.clear()



def _precluster(matrix, labels, randomstate, nremove=10000, nextract=20000):
    """Does rough preclustering, splits matrix and labels into multiple,
    overlapping matrixes/labels. Uses a version of Canopy clustering:
    
    1) Pick random seed observation, calculate distances P to all other obs
    2) Inner threshold is `nremove`th smallest dist, other is `nextract`th
    3) Create new matrix, labels pair for all obs within outer threshold
    4) Remove all obs within inner dist from set
    5) Continue from 1) until max `nextract` observations are left
    """
    
    if nextract < nremove:
        raise ValueError('nextract must exceed or be equal to nremove')
    
    while len(matrix) > nextract:
        seed = randomstate.randint(len(matrix))
        distances = _distances_to_vector(matrix, seed)
        
        sorted_distances = _np.sort(distances)
        innerdistance = sorted_distances[nremove]
        outerdistance = sorted_distances[nextract]
        del sorted_distances
        
        innerindices = _np.where(distances <= innerdistance)[0]
        outerindices = _np.where(distances <= outerdistance)[0]
        
        yield matrix[outerindices], labels[outerindices]
        
        matrix = _np.delete(matrix, innerindices, 0)
        labels = _np.delete(labels, innerindices, 0)
    
    yield matrix, labels



def _mergeclusters(contigsof):
    """Given a cluster dict, for each obs in multiple clusters, only keep
    it in the largest cluster it appears in. Modifies dict in-place."""
    
    clustersof = _defaultdict(list)
    
    for cluster, contigs in contigsof.items():
        for contig in contigs:
            clustersof[contig].append(cluster)
            
    # We don't have to do anything about contigs in only one cluster
    clustersof = {co: cl for co, cl in clustersof.items() if len(cl) > 1}
    
    # Here, we assign each contig to the largest cluster it appears in
    while clustersof:
        contig, clusters = clustersof.popitem()
        largestcluster = max(clusters, key=lambda x: len(contigsof[x]))
        
        for cluster in clusters:
            if cluster is not largestcluster:
                contigsof[cluster].remove(contig)
                
    # Finally, we remove all clusters with less than minclusters members
    toremove = [cl for cl, co in contigsof.items() if len(co) == 0]
    
    for key in toremove:
        contigsof.pop(key)    



def tandemcluster(matrix, labels, inner, outer=None, max_steps=15, spearman=False):
    """Splits the datasets, then clusters each partition before merging
    the resulting clusters. This is faster, especially on larger datasets, but
    less accurate than normal clustering.
    
    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        labels: Numpy array with labels for matrix rows. None or 1-D array
        inner: Optimal medoid search within this distance from medoid
        outer: Radius of clusters extracted from medoid. If None, same as inner
        max_steps: Stop searching for optimal medoid after N futile attempts
        spearman: Use Spearman, not Pearson correlation
    
    Output: {medoid: set(labels_in_cluster) dictionary}
    """
    
    if outer is None:
        outer = inner
        
    elif outer < inner:
        raise ValueError('outer must exceed or be equal to inner')
    
    labels = _checklabels(labels, len(matrix))
    matrix = _normalize(matrix, spearman=spearman)
    
    randomstate = _np.random.RandomState(324645) 
    contigsof = dict()
    
    for submatrix, sublabels in _precluster(matrix, labels, randomstate, nremove=10000, nextract=20000):
        for medoid, cluster in _cluster(submatrix, sublabels, inner, outer, max_steps):
            contigsof[medoid] = cluster
        
    _mergeclusters(contigsof)
    
    return contigsof



def cluster(matrix, labels, inner, outer=None, max_steps=15, spearman=False):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.
    
    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        labels: Numpy array with labels for matrix rows or None.
        inner: Optimal medoid search within this distance from medoid
        outer: Radius of clusters extracted from medoid. If None, same as inner
        max_steps: Stop searching for optimal medoid after N futile attempts
        spearman: Use Spearman, not Pearson correlation
    
    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """

    if outer is None:
        outer = inner
        
    elif outer < inner:
        raise ValueError('outer must exceed or be equal to inner')
    
    labels = _checklabels(labels, len(matrix))
    matrix = _normalize(matrix, spearman=spearman)
    
    return _cluster(matrix, labels, inner, outer, max_steps)



def writeclusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None):
    """Writes clusters to an open filehandle.
    
    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or dict
        min_size: Don't output clusters smaller than N contigs
        
    Output: None
    """
    
    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')
        
    if iter(clusters) is not clusters:
        clusters = clusters.items()
    
    if header is not None and len(header) > 0:
        print('# ' + header, file=filehandle)
    
    clusternumber = 0
    for clustername, contigs in clusters:
        if clusternumber == max_clusters:
            break
        
        if len(contigs) < min_size:
            continue
        
        clustername = 'cluster_' + str(clusternumber + 1)
        
        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)
            
        clusternumber += 1



def readclusters(filehandle, min_size=1):
    """Read clusters from a file as created by function `writeclusters`.
    
    Inputs:
        filehandle: An open filehandle that can be read from
        min_size: Minimum number of contigs in cluster to be kept
    
    Output: A {clustername: set(contigs)} dict"""
    
    contigsof = _defaultdict(set)
    
    for line in filehandle:
        stripped = line.strip()
        
        if stripped[0] == '#':
            continue
            
        clustername, contigname = stripped.split('\t')
        
        contigsof[clustername].add(contigname)
        
    contigsof = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}
        
    return contigsof



if __name__ == '__main__':
    usage = "python cluster.py [OPTIONS ...] INPUT OUTPUT"
    parser = argparse.ArgumentParser(
        description=cmd_doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage)
   
    # create the parser
    parser.add_argument('input', help='input dataset')
    parser.add_argument('output', help='output clusters')
    parser.add_argument('inner', help='inner distance threshold', type=float)
    parser.add_argument('outer', help='outer distnace threshold', type=float)
    
    
    parser.add_argument('-c', dest='max_clusters',
                        help='stop after creating N clusters [None, i.e. infinite]', type=int)
    parser.add_argument('-m', dest='max_steps',
                        help='stop searchin for optimal medoid after N attempts [15]',
                        default=15, type=int)
    parser.add_argument('-s', dest='min_size',
                        help='minimum cluster size to output [1]',default=1, type=int)
    parser.add_argument('--precluster', help='precluster first [False]', action='store_true')
    parser.add_argument('--spearman', help='use Spearman, not Pearson distance [False]', action='store_true')
    
    # Print help if no arguments are given
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()

    args = parser.parse_args()
    
    if args.outer is None:
        args.outer = args.inner
    
    elif args.outer < args.inner:
        raise ValueError('outer threshold must exceed or be equal to inner threshold')
        
    if args.inner <= 0:
        raise ValueError('inner threshold must be larger than 1.')
        
    if args.minsize < 1:
        raise ValueError('Minimum size must be 1 or above.')
        
    if args.max_steps < 1:
        raise ValueError('Max steps must be 1 or above.')
        
    if args.max_clusters is not None and args.precluster:
        raise ValueError('Conflicting arguments: precluster and max_clusters')
    
    if not _os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    
    if _os.path.isfile(args.output):
        raise FileExistsError(args.output)
        
    directory = _os.path.dirname(args.output)
    if directory and not _os.path.isdir(directory):
        raise NotADirectoryError(directory)
   
    matrix = _np.loadtxt(args.input, delimiter='\t', dtype=_np.float32)
    
    if args.precluster:
        contigsof = tandemcluster(matrix, None, args.inner, outer=None,
                               max_steps=args.max_steps, spearman=args.spearman)
        
    else:
        contigsof = cluster(matrix, None, args.inner, outer=None,
                                max_steps=args.max_steps, spearman=args.spearman)
    
    header = "clustername\tcontigindex(1-indexed)"
    
    with open(args.output, 'w') as filehandle:
        writeclusters(filehandle, contigsof, args.max_clusters, args.min_size,
                     header=header)

