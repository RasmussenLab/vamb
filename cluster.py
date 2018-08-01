
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
(4): If any mean(inner_obs(i)) < mean(inner_obs(S)) for i in I: Let S be i, go to (2)
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

import sys as _sys
import os as _os
import numpy as _np
from collections import defaultdict as _defaultdict
import vamb.vambtools as _vambtools
import vamb.threshold as _threshold



def _pearson_distances(matrix, index):
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
    # Also make sure that no rows are [0, 0, 0 ... ]
    vectorlength = matrix.shape[1]
    result = _np.dot(matrix, matrix[index].T)
    result -= vectorlength
    result *= -1 / (2 * vectorlength)
    
    return result



def _getinner(matrix, point, inner_threshold):
    """Gets the distance vector, array of inner points and average distance
    to inner points from a starting point"""
    
    distances = _pearson_distances(matrix, point)
    inner_points = _np.where(distances < inner_threshold)[0]
    
    # This happens if std(matrix[points]) == 0, then all pearson distances
    # become 0.5, even the distance to itself.
    if len(inner_points) == 0:
        inner_points = _np.array([point])
        average_distance = 0   
    
    elif len(inner_points) == 1:
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
            
    if inner_threshold == outer_threshold:
        outer_points = inner_points
    else:
        outer_points = _np.where(distances < outer_threshold)[0]
    
    return point, inner_points, outer_points



def _cluster(matrix, labels, inner_threshold, outer_threshold, max_steps):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""

    randomstate = _np.random.RandomState(324645)

    # The seed keeps track of which point we initialize clusters from.
    # It's necessary since we don't remove points after every cluster.
    # We don't do that since removing points is expensive. If there's only
    # one point in the outer_points, by definition, the point to be removed
    # will never be present in any other cluster anyway.
    seed = 0
    keepmask = _np.ones(len(matrix), dtype=_np.bool)
    
    while len(matrix) > 0:           
        # Find medoid using iterative sampling function above
        sampling = _sample_clusters(matrix, seed, max_steps, inner_threshold, outer_threshold, randomstate)
        medoid, inner_points, outer_points = sampling
        seed += 1
        
        # Write data to output
        yield labels[medoid], set(labels[outer_points])
        
        for point in inner_points:
            keepmask[point] = False

        # Only remove points if we have more than 1 point in cluster
        if len(outer_points) > 1 or seed == len(matrix):
            matrix = matrix[keepmask]
            labels = labels[keepmask]
            
            # This is quicker than changing existing mask to True
            keepmask = _np.ones(len(matrix), dtype=_np.bool)
            
            seed = 0



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
        distances = _pearson_distances(matrix, seed)
        
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



def _collapse(clustername, cluster, contigsof, clusterof):
    """When a new cluster is created among other clusters which might share
    its points, assigns the points to the largest cluster.
    
    Inputs:
        clustername: some hashable identifier of cluster
        cluster: Set of points
        contigsof: {identifier: set(points) dict for all clusters}
        clusterof: {point: identifier} for all points
        
    Output: None
    """

    # Get list of contigs sorted by length of set they're in
    # We sort to minimize contig reassignment
    membership = list()
    for contig in cluster:
        if contig in clusterof:
            length = len(contigsof[clusterof[contig]])

        else:
            length = 0

        membership.append((contig, length))

    membership.sort(key=lambda x: x[1], reverse=True)

    # From contig in largest set to that in smallest:
    for contig, length in membership:
        # If it's already in a larger set, remove it from this proposed set
        if length > len(cluster):
            cluster.remove(contig)

        # Else, since they're sorted, we're done here
        else:
            break

    # All remaining contigs now truly belong to this proposed set
    for contig in cluster:
        # Remove it from other clusters if they are in another cluster
        # and delete the cluster all its contigs are now gone
        if contig in clusterof:
            othername = clusterof[contig]
            othercluster = contigsof[othername]
            othercluster.remove(contig)

            if len(othercluster) == 0:
                contigsof.pop(othername)

        # Finally, add this new proposed cluster to set of clusters
        clusterof[contig] = clustername

    if len(cluster) > 0:
        contigsof[clustername] = cluster



def _check_inputs(max_steps, inner, outer):
    """Checks whether max_steps, inner and outer are okay.
    Can be run before loading matrix into memory."""
    
    if max_steps < 1:
        raise ValueError('maxsteps must be a positive integer')
    
    if inner is None:
        if outer is not None:
            raise ValueError('If inner is None, outer must be None')
        
    elif outer is None:
        outer = inner
        
    elif outer < inner:
        raise ValueError('outer must exceed or be equal to inner')
        
    return inner, outer



def _check_params(matrix, inner, outer, labels, nsamples, maxsize, logfile):
    """Checks matrix, labels, nsamples, maxsize and estimates inner if necessary."""
    
    if len(matrix) < 1:
        raise ValueError('Matrix must have at least 1 observation.')
    
    if labels is None:
        labels = _np.arange(len(matrix)) + 1
        
    elif type(labels) != _np.ndarray or len(labels) != len(matrix):
        raise ValueError('labels must be a 1D Numpy array with same length as matrix')
        
    if len(set(labels)) != len(matrix):
        raise ValueError('Labels must be unique')
    
    if inner is None:
        if len(matrix) < 1000 and inner is None:
            raise ValueError('Cannot estimate from less than 1000 contigs')

        if len(matrix) < nsamples:
            raise ValueError('Specified more samples than available contigs')

        if maxsize < 1:
            raise ValueError('maxsize must be positive number')
        
        try:
            _gt = _threshold.getthreshold(matrix, _pearson_distances, nsamples, maxsize)
            inner, support, separation = _gt
            outer = inner
            
            if separation < 0.25:
                sep = round(separation * 100, 1)
                wn = '\tWarning: Only {}% of contigs has well-separated threshold'
                print(wn.format(sep), file=logfile)

            if support < 0.50:
                sup = round(support * 100, 1)
                wn = '\tWarning: Only {}% of contigs has *any* observable threshold'
                print(wn.format(sup), file=logfile)
                
        except _threshold.TooLittleData as error:
            wn = '\tWarning: Too little data: {}. Setting threshold to 0.08'
            print(wn.format(error.args[0]), file=logfile)
            inner = outer = 0.08
        
    return labels, inner, outer



def cluster(matrix, labels=None, inner=None, outer=None, max_steps=15,
            normalized=False, nsamples=2000, maxsize=2500, logfile=None):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.
    
    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        labels: None or Numpy array with labels for matrix rows [None = ints]
        inner: Optimal medoid search within this distance from medoid [None = auto]
        outer: Radius of clusters extracted from medoid. [None = inner]
        max_steps: Stop searching for optimal medoid after N futile attempts [15]
        normalized: Matrix is already zscore-normalized [False]
        nsamples: Estimate threshold from N samples [1000]
        maxsize: Discard sample if more than N contigs are within threshold [2500]
    
    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """
    
    if not normalized:
        matrix = _vambtools.zscore(matrix, axis=1)
        
    inner, outer = _check_inputs(max_steps, inner, outer)
    labels, inner, outer = _check_params(matrix, inner, outer, labels, nsamples, maxsize, logfile)
    
    if logfile is not None:
        print('Clustering threshold:', inner, file=logfile)
    
    return _cluster(matrix, labels, inner, outer, max_steps)



def tandemcluster(matrix, labels=None, inner=None, outer=None, max_steps=15,
            normalized=False, nsamples=2000, maxsize=2500, logfile=None):
    """Splits the datasets, then clusters each partition before merging
    the resulting clusters. This is faster, especially on larger datasets, but
    less accurate than normal clustering.
    
    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        labels: None or Numpy array with labels for matrix rows [None = ints]
        inner: Optimal medoid search within this distance from medoid [None = auto]
        outer: Radius of clusters extracted from medoid. [None = inner]
        max_steps: Stop searching for optimal medoid after N futile attempts [15]
        normalized: Matrix is already zscore-normalized [False]
        nsamples: Estimate threshold from N samples [1000]
        maxsize: Discard sample if more than N contigs are within threshold [2500]
    
    Output: {(partition, medoid): set(labels_in_cluster) dictionary}
    """
    
    if not normalized:
        matrix = _vambtools.zscore(matrix, axis=1)
        
    inner, outer = _check_inputs(max_steps, inner, outer)
    labels, inner, outer = _check_params(matrix, inner, outer, labels, nsamples, maxsize, logfile)
    
    if logfile is not None:
        print('\tClustering threshold:', inner, file=logfile)
    
    randomstate = _np.random.RandomState(324645) 
    
    contigsof = dict()
    clusterof = dict()
    
    partitions = _precluster(matrix, labels, randomstate, nremove=10000, nextract=30000)
    
    for partition, (submatrix, sublabels) in enumerate(partitions):
        for medoid, cluster in _cluster(submatrix, sublabels, inner, outer, max_steps):
            _collapse((partition, medoid), cluster, contigsof, clusterof)
        
        matrix = None
            
    return contigsof



def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None):
    """Writes clusters to an open filehandle.
    
    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add
        
    Output: None
    """
    
    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')
        
    if iter(clusters) is not clusters:
        clusters = clusters.items()
    
    if header is not None and len(header) > 0:
        if '\n' in header:
            raise ValueError('Header cannot contain newline')
        
        if header[0] != '#':
            header = '# ' + header
        
        print(header, file=filehandle)
    
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



def read_clusters(filehandle, min_size=1):
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

