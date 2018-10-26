
#!/usr/bin/env python3

__doc__ = """Iterative medoid clustering.

Usage:
>>> cluster_iterator = cluster(rpkms, tnfs, labels=contignames)
>>> clusters = dict(cluster_iterator)

Implements one core function, cluster, along with the helper
functions write_clusters and read_clusters.
For all functions in this module, a collection of clusters are represented as
a {clustername, set(elements)} dict.

cluster algorithm:
(1): Pick random seed observation S
(2): Define cluster(S) = all observations with Pearson distance from S < THRESHOLD
(3): Sample MOVES observations I from cluster(S)
(4): If any mean(cluster(i)) < mean(cluster(S)) for i in I: Let S be i, go to (2)
     Else: cluster(S) = all observations with Pearson distance from S < THRESHOLD
(5): Output cluster(S) as cluster, remove cluster(S) from observations
(6): If no more observations or MAX_CLUSTERS have been reached: Stop
     Else: Go to (1)
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
    vectorlength = matrix.shape[1]
    result = _np.dot(matrix, matrix[index].T) # 70% clustering time spent on this line
    result -= vectorlength
    result *= -1 / (2 * vectorlength)

    return result

def _getcluster(matrix, point, threshold):
    """Gets the distance vector, array of cluster points and average distance
    to cluster points from a starting point"""

    distances = _pearson_distances(matrix, point)
    cluster = _np.where(distances < threshold)[0]

    # This happens if std(matrix[points]) == 0, then all pearson distances
    # become 0.5, even the distance to itself.
    if len(cluster) == 0:
        cluster = _np.array([point])
        average_distance = 0

    elif len(cluster) == 1:
        average_distance = 0

    else:
        average_distance = _np.sum(distances[cluster]) / (len(cluster) - 1)

    return distances, cluster, average_distance

def _sample_clusters(matrix, point, max_attempts, threshold, randomstate):
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0

    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}

    distances, cluster, average_distance = _getcluster(matrix, point, threshold)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sample = randomstate.choice(cluster)
        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = randomstate.choice(cluster)

        tried.add(sample)

        sample_dist, sample_cluster, sample_avg =  _getcluster(matrix, sample, threshold)

        if sample_avg < average_distance:
            point = sample
            cluster = sample_cluster
            average_distance = sample_avg
            distances = sample_dist
            futile_attempts = 0
            tried = {point}

        else:
            futile_attempts += 1

    return point, cluster

def _cluster(matrix, labels, threshold, max_steps):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""

    randomstate = _np.random.RandomState(324645)

    # The seed keeps track of which point we initialize clusters from.
    # It's necessary since we don't remove points after every cluster.
    # We don't do that since removing points is expensive. If there's only
    # one point in the cluster, by definition, the point to be removed
    # will never be present in any other cluster anyway.
    seed = 0
    keepmask = _np.ones(len(matrix), dtype=_np.bool)

    while len(matrix) > 0:
        # Find medoid using iterative sampling function above
        sampling = _sample_clusters(matrix, seed, max_steps, threshold, randomstate)
        medoid, cluster = sampling
        seed += 1

        # Write data to output
        yield labels[medoid], set(labels[cluster])

        for point in cluster:
            keepmask[point] = False

        # Only remove cluster if we have more than 1 point in cluster
        if len(cluster) > 1 or seed == len(matrix):
            matrix = matrix[keepmask]
            labels = labels[keepmask]

            # This is quicker than changing existing mask to True
            keepmask = _np.ones(len(matrix), dtype=_np.bool)
            seed = 0

def _check_params(matrix, threshold, labels, nsamples, maxsize, maxsteps, logfile):
    """Checks matrix, labels, nsamples, maxsize and estimates threshold if necessary."""

    if maxsteps < 1:
        raise ValueError('maxsteps must be a positive integer')

    if len(matrix) < 1:
        raise ValueError('Matrix must have at least 1 observation.')

    if labels is None:
        labels = _np.arange(len(matrix)) + 1

    elif type(labels) != _np.ndarray or len(labels) != len(matrix):
        raise ValueError('labels must be a 1D Numpy array with same length as matrix')

    if len(set(labels)) != len(matrix):
        raise ValueError('Labels must be unique')

    if threshold is None:
        if len(matrix) < 1000 and threshold is None:
            raise ValueError('Cannot estimate from less than 1000 contigs')

        if len(matrix) < nsamples:
            raise ValueError('Specified more samples than available contigs')

        if maxsize < 1:
            raise ValueError('maxsize must be positive number')

        try:
            if logfile is not None:
                print('\tEstimating threshold with {} samples'.format(nsamples), file=logfile)

            _gt = _threshold.getthreshold(matrix, _pearson_distances, nsamples, maxsize)
            threshold, support, separation = _gt

            if logfile is not None:
                print('\tClustering threshold:', threshold, file=logfile)
                print('\tThreshold support:', support, file=logfile)
                print('\tThreshold separation:', separation, file=logfile)

        except _threshold.TooLittleData as error:
            warningfile = sys.stderr if logfile is None else logfile
            wn = '\tWarning: Too little data: {}. Setting threshold to 0.08'
            print(wn.format(error.args[0]), file=warningfile)
            threshold = 0.08

    return labels, threshold

def cluster(matrix, labels=None, threshold=None, maxsteps=25,
            normalized=False, nsamples=2500, maxsize=2500, logfile=None):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.

    Inputs:
        matrix: A (obs x features) Numpy matrix of values
        labels: None or Numpy array with labels for matrix rows [None = indices]
        threshold: Optimal medoid search within this distance from medoid [None = auto]
        maxsteps: Stop searching for optimal medoid after N futile attempts [25]
        normalized: Matrix is already zscore-normalized [False]
        nsamples: Estimate threshold from N samples [2500]
        maxsize: Discard sample if more than N contigs are within threshold [2500]
        logfile: Print threshold estimates and certainty to file [None]

    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """

    if not normalized is True:
        matrix = _vambtools.zscore(matrix, axis=1)

    labels, threshold = _check_params(matrix, threshold, labels, nsamples, maxsize, maxsteps, logfile)

    return _cluster(matrix, labels, threshold, maxsteps)


def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None):
    """Writes clusters to an open filehandle.

    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add

    Outputs:
        clusternumber: Number of clusters written
        ncontigs: Number of contigs written
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
    ncontigs = 0

    for clustername, contigs in clusters:
        if clusternumber == max_clusters:
            break

        if len(contigs) < min_size:
            continue

        clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)

        clusternumber += 1
        ncontigs += len(contigs)

    return clusternumber, ncontigs



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
