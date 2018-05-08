
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

import sys
import os
import argparse
import numpy as np



def distances_to_vector(matrix, index):
    
    # Distance D = (P - 1) / -2, where P is Pearson correlation coefficient.
    # For two vectors x and y with numbers xi and yi,
    # P = sum((xi-x_mean)*(yi-y_mean)) / (std(y) * std(x) * len(x)).
    # If we normalize matrix so x_mean = y_mean = 0 and std(x) = std(y) = 1,
    # this reduces to sum(xi*yi) / len(x) = x @ y.T / len(x) =>
    # D = ((x @ y.T) / len(x)) - 1) / -2 =>
    # D = (x @ y.T - len(x)) * (-1 / 2len(x))
    
    # Matrix should have already been normalized (subtract mean, div by std)
    vectorlength = matrix.shape[1]
    result = matrix @ matrix[index].T
    result -= vectorlength
    result *= -1 / (2 * vectorlength)
    
    return result



def getinner(matrix, point, inner_threshold):
    """Gets the distance vector, array of inner points and average distance
    to inner points from a starting point"""
    
    distances = distances_to_vector(matrix, point)
    inner_points = np.where(distances < inner_threshold)[0]
    
    if len(inner_points) == 1:
        average_distance = 0
    else:
        average_distance = np.sum(distances[inner_points]) / (len(inner_points) - 1)

    return distances, inner_points, average_distance



def sample_clusters(matrix, point, max_attempts, inner_threshold, outer_threshold):
    """Keeps sampling new points within the inner points until it has sampled
    max_attempts without getting a new set of inner points with lower average
    distance"""
    
    futile_attempts = 0
    
    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}
    
    distances, inner_points, average_distance = getinner(matrix, point, inner_threshold)
    
    while len(inner_points) - len(tried) > 0 and futile_attempts < max_attempts:
        sample = np.random.choice(inner_points)
        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = np.random.choice(inner_points)
            
        tried.add(sample)
        
        inner = getinner(matrix, sample, inner_threshold)
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
            
    outer_points = np.where(distances < outer_threshold)[0]
    
    return point, inner_points, outer_points



def clusters(matrix, inner_threshold, outer_threshold, max_clusters, max_steps, min_size):
    """Yields (medoid, points) pairs from a obs x features matrix"""
    
    np.random.seed(324645) # Reproducability even when it's random.
    
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
    matrix -= matrix.mean(axis=1).reshape((len(matrix), 1))
    matrix /= np.std(matrix, axis=1).reshape((len(matrix), 1))
    
    # We initialize clusters from most extreme to less extreme. This is
    # arbitrary and just to have some kind of reproducability.
    # Note to Simon: Sorting by means makes no sense with normalized rows.
    extremes = np.max(matrix, axis=1)
    argextremes = np.argsort(extremes)
    
    # This is to keep track of the original order of the points, even when we
    # remove points as the clustering proceeds.
    indices = np.arange(len(matrix))
    
    clusters_completed = 0
    
    while len(matrix) > 0:
        if clusters_completed == max_clusters:
            break
            
        # Most extreme point (without picking same point twice)
        seed = argextremes[-seed_index -1]
        
        # Find medoid using iterative sampling function above
        sampling = sample_clusters(matrix, seed, max_steps, inner_threshold, outer_threshold)
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
            matrix = np.delete(matrix, toremove, 0)
            indices = np.delete(indices, toremove, 0)
            extremes = np.delete(extremes, toremove, 0)
            argextremes = np.argsort(extremes)
            seed_index = 0
            toremove.clear()



def main(input, output, inner, outer, max_clusters, moves, min_size):
    '''Canopy clustering'''

    matrix = np.loadtxt(input, delimiter='\t', dtype=np.float32)
    
    with open(output, 'w') as filehandle:
        print('#clustername', 'contigindex(1-indexed)', sep='\t', file=filehandle)
        
        clusters = cluster(matrix, inner, outer, max_clusters, moves, min_size)
        
        for clusternumber, (medoid, cluster) in enumerate(clusters):
            clustername = 'cluster_' + str(clusternumber + 1)
            
            for clusterindex in cluster:
                print(clustername, clusterindex + 1, sep='\t', file=filehandle)



if __name__ == '__main__':
    
    parserkws = {'prog': 'medoid_clustering.py',
                 'formatter_class': lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=130),
                 'usage': '%(prog)s [options]',
                 'description': __doc__}
   
    # create the parser
    parser = argparse.ArgumentParser(**parserkws)
   
    parser.add_argument('input', help='input dataset')
    parser.add_argument('output', help='output clusters')
    parser.add_argument('-i', dest='inner', help='inner threshold [0.08]', default=0.08, type=float)
    parser.add_argument('-o', dest='outer', help='outer threshold [0.10]', default=0.10, type=float)
    parser.add_argument('-c', dest='max_clusters', help='max seeds before exiting [30000]', default=30000, type=int)
    parser.add_argument('-m', dest='moves', help='Moves before cluster is stable [15]', default=15, type=int)
    parser.add_argument('-s', dest='min_size', help='Minimum cluster size to output [1]', default=1, type=int)
    
    # Print help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    
    if os.path.isfile(args.output):
        raise FileExistsError(args.output)
   
    main(args.input, args.output, args.inner, args.outer, args.max_clusters,
         args.moves, args.min_size)

