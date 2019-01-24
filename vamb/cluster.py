#!/usr/bin/env python3

# Note on this source code:
# Clustering is the timewise bottle neck of VAMB for large datasets, as it
# scales quadratically with number of sequences. Because clustering time is
# dominated by operation on dense matrices, GPU acceleration can achieve
# tremendous effects (80x speedup in our test). So there is a great incentive
# for writing GPU-friendly code.
# Unfortunately, most people don't have good GPUs. Even worse, it turns out
# that fast GPU code is slow CPU code and vice versa - for example, CPU
# clustering is greatly sped up by removing sequences from the matrix, whereas
# this slows down GPU clustering.
# So we have to implement BOTH a numpy-based CPU clustering and a torch-based
# GPU clustering, duplicating all our code.

__doc__ = """Iterative medoid clustering.

Usage:
>>> cluster_iterator = cluster(matrix, labels=contignames)
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
import random as _random
import numpy as _np
import torch as _torch
from collections import defaultdict as _defaultdict
import vamb.vambtools as _vambtools
import vamb.threshold as _threshold

    # Distance D = (P - 1) / -2, where P is Pearson correlation coefficient.
    # For two vectors x and y with numbers xi and yi,
    # P = sum((xi-x_mean)*(yi-y_mean)) / (std(y) * std(x) * len(x)).
    # If we normalize matrix so x_mean = y_mean = 0 and std(x) = std(y) = 1,
    # this reduces to sum(xi*yi) / len(x) = x @ y.T / len(x) =>
    # D = ((x @ y.T) / len(x)) - 1) / -2 =>
    # D = (x @ y.T - len(x)) * (-1 / 2len(x))
def _numpy_pearson_distances(matrix, index):
    """Calculates the Pearson distances from row `index` to all rows
    in the matrix, including itself. Returns numpy array of distances.
    Input matrix must have been zscore normalized across axis 1"""

    vectorlength = matrix.shape[1]
    result = _np.dot(matrix, matrix[index].T) # 70% clustering time spent on this line
    result -= vectorlength
    result *= -1 / (2 * vectorlength)
    return result

def _torch_pearson_distances(tensor, index):
    """Calculates the Pearson distances from row `index` to all rows
        in the matrix, including itself. Returns torch array of distances.
        Input matrix must have been zscore normalized across dim 1"""

    vectorlength = tensor.shape[1]
    result = _torch.matmul(tensor, tensor[index])
    result -= vectorlength
    result *= -1 / (2 * vectorlength)
    return result

def _numpy_getcluster(matrix, medoid, threshold):
    """
    Returns:
    - A vector of indices to each of these inner points
    - The mean distance from medoid to the other inner points
    """

    distances = _numpy_pearson_distances(matrix, medoid)
    cluster = _np.where(distances <= threshold)[0]

    # This happens if std(matrix[points]) == 0, then all pearson distances
    # become 0.5, even the distance to itself.
    if len(cluster) < 2:
        cluster = _np.array([medoid])
        average_distance = 0
    else:
        average_distance = _np.sum(distances[cluster]) / (len(cluster) - 1)

    return cluster, average_distance

def _torch_getcluster(tensor, kept_mask, medoid, threshold):
    """
    Returns:
    - A boolean mask of which points are within the threshold
    - A vector of the indices of all points within threshold
    - The mean distance from medoid to all other points within threshold
    """

    distances = _torch_pearson_distances(tensor, medoid)
    mask = (distances <= threshold) & kept_mask
    inner_dists = distances[mask]

    # If the standard deviation of a row of the latent matrix is zero, all
    # distances becomes 0.5 - even to itself, so we need to cover both special
    # cases of len(inner_dists) being 0 and 1.
    if len(inner_dists) < 2:
        average_distance = 0
        mask[medoid] = 1
    else:
        average_distance = inner_dists.sum().item() / (len(inner_dists) - 1)

    cluster = mask.nonzero().reshape(-1)
    return mask, cluster, average_distance

def _numpy_wander_medoid(matrix, medoid, max_attempts, threshold, rng):
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0
    tried = {medoid} # keep track of already-tried medoids
    cluster, average_distance = _numpy_getcluster(matrix, medoid, threshold)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = rng.choice(cluster)

         # Prevent sampling same medoid multiple times.
        while sampled_medoid in tried:
            sampled_medoid = rng.choice(cluster)

        tried.add(sampled_medoid)

        sample_cluster, sample_avg =  _numpy_getcluster(matrix, sampled_medoid, threshold)

        # If the mean distance of inner points of the sample is lower,
        # we move the medoid and reset the futile_attempts count
        if sample_avg < average_distance:
            medoid = sampled_medoid
            cluster = sample_cluster
            average_distance = sample_avg
            futile_attempts = 0
            tried = {medoid}

        else:
            futile_attempts += 1

    return medoid, cluster

def _torch_wander_medoid(tensor, kept_mask, medoid, max_attempts, threshold, rng):
    """Keeps sampling new medoids within `threshold` distance to the medoid,
      until it has sampled `max_attempts` times without getting a new set of
      inner points with lower average distance to medoid."""

    futile_attempts = 0
    tried = {medoid} # keep track of already-tried medoids
    mask, cluster, average_distance = _torch_getcluster(tensor, kept_mask, medoid, threshold)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = rng.choice(cluster).item()

         # Prevent sampling same medoid multiple times.
        while sampled_medoid in tried:
            sampled_medoid = rng.choice(cluster).item()

        tried.add(sampled_medoid)

        _ = _torch_getcluster(tensor, kept_mask, sampled_medoid, threshold)
        sample_mask, sample_cluster, sample_average = _

        # If the mean distance of cluster of the sample is lower,
        # we move the medoid and reset the futile_attempts count
        if sample_average < average_distance:
            medoid = sampled_medoid
            mask = sample_mask
            cluster = sample_cluster
            average_distance = sample_average
            futile_attempts = 0
            tried = {medoid}

        else:
            futile_attempts += 1

    return medoid, mask

def _numpy_cluster(matrix, labels, indices, threshold, max_steps):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""
    seed = 0
    kept_mask = _np.ones(len(matrix), dtype=_np.bool)
    rng = _random.Random(0)

    while len(matrix) > 0:
        # Find medoid using iterative sampling function above
        sampling = _numpy_wander_medoid(matrix, seed, max_steps, threshold, rng)
        medoid, cluster = sampling
        seed += 1

        # Write data to output. We use indices instead of labels directly to
        # prevent masking and thereby duplicating large string arrays
        if labels is None:
            yield indices[medoid]+1, {i+1 for i in indices[cluster]}
        else:
            yield labels[indices[medoid]], {labels[indices[i]] for i in cluster}

        for point in cluster:
            kept_mask[point] = False

        # Only remove cluster if we have more than 1 point in cluster
        # because masking takes time
        if len(cluster) > 1 or seed == len(matrix):
            _vambtools.inplace_maskarray(matrix, kept_mask)
            indices = indices[kept_mask] # no need to inplace mask small array
            kept_mask.resize(len(matrix), refcheck=False)
            kept_mask[:] = True
            seed = 0

def _torch_cluster(tensor, labels, indices, threshold, max_steps):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""
    kept_mask = _torch.ones(len(tensor), dtype=_torch.uint8)
    rng = _random.Random(0)

    # Move tensors to GPU
    tensor = tensor.cuda()
    kept_mask = kept_mask.cuda()
    indices = indices.cuda()

    # The seed is the first index that has not yet been clustered
    possible_seeds = kept_mask.nonzero().reshape(-1)
    while len(possible_seeds) > 0:
        seed = possible_seeds[0]

        # Find the most optimal medoid by letting it wander around
        medoid, mask = _torch_wander_medoid(tensor, kept_mask, seed, max_steps, threshold, rng)
        kept_mask &= ~mask # And add these new points to the set of removed points
        cluster = indices[mask]
        cluster = cluster.cpu().numpy()

        # Write data to output. We use indices instead of labels directly to
        # prevent masking and thereby duplicating large string arrays
        if labels is None:
            yield indices[medoid].item() + 1, {i+1 for i in cluster}
        else:
            yield labels[indices[medoid].item()], {labels[i] for i in cluster}

        possible_seeds = kept_mask.nonzero().reshape(-1)

def _check_params(matrix, threshold, labels, nsamples, maxsize, maxsteps, logfile):
    """Checks matrix, labels, nsamples, maxsize and estimates threshold if necessary."""

    if maxsteps < 1:
        raise ValueError('maxsteps must be a positive integer')

    if len(matrix) < 1:
        raise ValueError('Matrix must have at least 1 observation.')

    if labels is not None:
        if len(labels) != len(matrix):
            raise ValueError('labels must have same length as matrix')

        if len(set(labels)) != len(matrix):
            raise ValueError('Labels must be unique')

    if matrix.dtype != _np.float32:
        raise ValueError('Matrix must be of data type np.float32')

    if threshold is None:
        if len(matrix) < nsamples and threshold is None:
            raise ValueError('Cannot estimate from less than nsamples contigs')

        if len(matrix) < nsamples:
            raise ValueError('Specified more samples than available contigs')

        if maxsize < 1:
            raise ValueError('maxsize must be positive number')

        try:
            if logfile is not None:
                print('\tEstimating threshold with {} sampled sequences'.format(nsamples), file=logfile)

            _gt = _threshold.getthreshold(matrix, _numpy_pearson_distances, nsamples, maxsize)
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

    return threshold

def cluster(matrix, labels=None, threshold=None, maxsteps=25,
            normalized=False, nsamples=2500, maxsize=2500, cuda=False, logfile=None):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.

    Inputs:
        matrix: A (obs x features) Numpy matrix of data type numpy.float32
        labels: None or Numpy array/list with labels for seqs [None = indices+1]
        threshold: Pearson distance "radius" of output clusters. [None = auto]
        maxsteps: Stop searching for optimal medoid after N futile attempts [25]
        normalized: Matrix is already zscore-normalized across axis 1 [False]
        nsamples: Estimate threshold from N samples [2500]
        maxsize: Discard sample if more than N contigs are within threshold [2500]
        cuda: Use CUDA (GPU acceleration) based on PyTorch. [False]
        logfile: Print threshold estimates and certainty to file [None]

    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """

    # Shuffle matrix and labels in unison to prevent seed sampling bias.
    # It would be nice if this could be done inplace, however Numpy.shuffle
    # does not produce same results for 1D and 2D arrays, and random.shuffle does
    # not work for 2D arrays.
    indices = _np.random.RandomState(0).permutation(len(matrix))
    matrix = matrix[randomindices]

    if not normalized:
        _vambtools.zscore(matrix, axis=1, inplace=True)

    threshold = _check_params(matrix, threshold, labels, nsamples, maxsize, maxsteps, logfile)

    if cuda:
        tensor = _torch.from_numpy(matrix)
        indices = _torch.from_numpy(indices)
        return _torch_cluster(tensor, labels, indices, threshold, maxsteps)
    else:
        return _numpy_cluster(matrix, labels, indices, threshold, maxsteps)

def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None):
    """Writes clusters to an open filehandle.

    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or a dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add

    Outputs:
        clusternumber: Number of clusters written
        ncontigs: Number of contigs written
    """

    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')

    if iter(clusters) is not clusters: # Is True if clusters is not iterator
        clusters = clusters.items()

    if max_clusters is not None and max_clusters < 1:
        raise ValueError('max_clusters must be at least 1.')

    if header is not None and len(header) > 0:
        if '\n' in header:
            raise ValueError('Header cannot contain newline')

        if header[0] != '#':
            header = '# ' + header

        print(header, file=filehandle)

    clusternumber = 0
    ncontigs = 0

    for clustername, contigs in clusters:
        if len(contigs) < min_size:
            continue

        clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)
        filehandle.flush()

        clusternumber += 1
        ncontigs += len(contigs)

        if clusternumber + 1 == max_clusters:
            break

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

        if not stripped or stripped[0] == '#':
            continue

        clustername, contigname = stripped.split('\t')
        contigsof[clustername].add(contigname)

    contigsof = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}

    return contigsof
