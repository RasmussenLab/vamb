# # To do:
# 
# Precluster using Wikipedia's Canopy to break into smaller pieces, then cluster.
# 
# Save all unclustered points (i.e. also the ones in clusters lower than the minsize threshold)
# 
# Maybe first check if any of sklearns clustering algos work better than the it. medoid in here.


#!/usr/bin/env python3

__doc__ = """Iterative medoid clustering.

Input: An observations x features matrix in text format.

Output: Tab-sep lines with clustername, observation(1-indexed).

Algorithm:
(1): Pick random seed observation S
(2): Define inner_obs(S) = all observations with Pearson distance from S < INNER
(3): Sample MOVES observations I from inner_obs
(4): if any inner_obs(I) > inner_obs(S): I is new S, go to (2)
     else: outer_obs(S) = all observations with Pearson distance from S < OUTER
(5): Output outer_obs(S) as cluster, remove inner_obs(S) from set
(6): If no more points or MAX_CLUSTERS have been reached: Stop, else go to (1)
"""

import sys as _sys
import os as _os
import argparse as _argparse
import numpy as _np



def _distances_to_vector(matrix, index):
    
    # Distance D = (P - 1) / -2, where P is Pearson correlation coefficient.
    # For two vectors x and y with numbers xi and yi,
    # P = sum((xi-x_mean)*(yi-y_mean)) / (std(y) * std(x) * len(x)).
    # If we normalize matrix so x_mean = y_mean = 0 and std(x) = std(y) = 1,
    # this reduces to sum(xi*yi) / len(x) = x @ y.T / len(x) =>
    # D = ((x @ y.T) / len(x)) - 1) / -2 =>
    # D = (x @ y.T - len(x)) * (-1 / 2len(x))
    
    # Matrix should have already been zscore normalized by axis 1 (subtract mean, div by std)
    vectorlength = matrix.shape[1]
    result = matrix @ matrix[index].T
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



def _sample_clusters(matrix, point, max_attempts, inner_threshold, outer_threshold):
    """Keeps sampling new points within the inner points until it has sampled
    max_attempts without getting a new set of inner points with lower average
    distance"""
    
    futile_attempts = 0
    
    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}
    
    distances, inner_points, average_distance = _getinner(matrix, point, inner_threshold)
    
    while len(inner_points) - len(tried) > 0 and futile_attempts < max_attempts:
        sample = _np.random.choice(inner_points)
        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = _np.random.choice(inner_points)
            
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



def clusters(matrix, inner_threshold, outer_threshold,
             max_clusters=30000, max_steps=15, min_size=5):
    """Yields (medoid, points) pairs from a (obs x features) matrix
    
    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        inner_threshold: Optimal medoid search within this distance from medoid
        outer_threshold: Radius of clusters extracted from medoid
        max_clusters [30k]: Stop generating clusters after N output clusters
        max_steps [15]: Stop searching for optimal medoid after N futile attempts
        min_size [5]: Don't output clusters with fewer than N elements
        
    Output: A generator yielding (medoid_0_indexed, points_0_indexed) pairs
    """
    
    _np.random.seed(324645) # Reproducability even when it's random.
    
    # This list keeps track of points to remove because they compose the inner circle
    # of clusters. We only remove points when we create a cluster with more than one
    # point, since one-point-clusters don't interfere with other clusters and the
    # point removal operations are expensive.
    toremove = list()
    
    # This index keeps track of which point we initialize clusters from.
    # It's necessary since we don't remove points after every cluster.
    seed_index = 0
    
    # Normalize - this simplifies calculating the Pearson distance
    # and boosts speed tremendously
    matrix = matrix - matrix.mean(axis=1).reshape((len(matrix), 1))
    std = _np.std(matrix, axis=1).reshape((len(matrix), 1))
    std[std == 0] = 1
    matrix /= std
    del std
    
    # We initialize clusters from most extreme to less extreme. This is
    # arbitrary and just to have some kind of reproducability.
    # Note to Simon: Sorting by means makes no sense with normalized rows.
    extremes = _np.max(matrix, axis=1)
    argextremes = _np.argsort(extremes)
    
    # This is to keep track of the original order of the points, even when we
    # remove points as the clustering proceeds.
    indices = _np.arange(len(matrix))
    
    clusters_completed = 0
    
    while len(matrix) > 0:
        if clusters_completed == max_clusters:
            break
            
        # Most extreme point (without picking same point twice)
        seed = argextremes[-seed_index -1]
        
        # Find medoid using iterative sampling function above
        sampling = _sample_clusters(matrix, seed, max_steps, inner_threshold, outer_threshold)
        medoid, inner_points, outer_points = sampling
        
        # Write data to output if the cluster is not too small
        if len(outer_points) >= min_size:
            yield indices[medoid], indices[outer_points]
            clusters_completed += 1
            
        seed_index += 1
        
        for point in inner_points:
            toremove.append(point)

        # Only remove points if we have more than 1 point in cluster
        # Note that these operations are really expensive.
        if len(inner_points) > 1 or len(argextremes) == seed_index:
            matrix = _np.delete(matrix, toremove, 0)
            indices = _np.delete(indices, toremove, 0)
            extremes = _np.delete(extremes, toremove, 0)
            argextremes = _np.argsort(extremes)
            seed_index = 0
            toremove.clear()



def writeclusters(filehandle, clusters, contignames=None):
    """Writes clusters to an open filehandle.
    
    Inputs:
        clusters: An iterator generated by function `clusters`
        filehandle: An open filehandle that can be written to.
        contignames: None or an indexable array of contignames.
        
    Output: None
    """
    
    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')
        
    print('#clustername', 'contigindex(1-indexed)', sep='\t', file=filehandle)
        
    for clusternumber, (medoid, cluster) in enumerate(clusters):
        clustername = 'cluster_' + str(clusternumber + 1)

        for contigindex in cluster:
            if contignames is None:
                contigname = contigindex + 1
            else:
                contigname = contignames[contigindex]
            
            print(clustername, contigname, sep='\t', file=filehandle)



if __name__ == '__main__':
    usage = "python cluster.py [OPTIONS ...] INPUT OUTPUT"
    parser = _argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_argparse.RawDescriptionHelpFormatter,
        usage=usage)
   
    # create the parser
    parser.add_argument('input', help='input dataset')
    parser.add_argument('output', help='output clusters')
    parser.add_argument('-i', dest='inner', help='inner threshold [0.08]', default=0.08, type=float)
    parser.add_argument('-o', dest='outer', help='outer threshold [0.10]', default=0.10, type=float)
    parser.add_argument('-c', dest='max_clusters', help='max seeds before exiting [30000]', default=30000, type=int)
    parser.add_argument('-m', dest='moves', help='Moves before cluster is stable [15]', default=15, type=int)
    parser.add_argument('-s', dest='min_size', help='Minimum cluster size to output [1]', default=1, type=int)
    
    # Print help if no arguments are given
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()

    args = parser.parse_args()
    
    if not _os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    
    if _os.path.isfile(args.output):
        raise FileExistsError(args.output)
   
    matrix = _np.loadtxt(args.input, delimiter='\t', dtype=_np.float32)
    
    clusters = cluster(matrix, args.inner, args.outer, args.max_clusters, 
                       args.moves, args.min_size)
    
    with open(args.output, 'w') as filehandle:
        writeclusters(filehandle, clusters, contignames=None)

