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

__cmd_doc__ = """Iterative medoid clustering.

    Input: An observations x features matrix in text format.
    Output: Tab-sep lines with clustername, observation(1-indexed).
"""

# To do here:
# Run on GPU w. benchmark

import sys as _sys
import os as _os
import torch as _torch
import random as _random
from collections import defaultdict as _defaultdict
import vamb.threshold as _threshold

def _pearson_distances(tensor, index):
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
    vectorlength = tensor.shape[1]
    result = tensor @ tensor[index]
    result -= vectorlength
    result *= -1 / (2 * vectorlength)

    return result

def _tensor_normalize(tensor, inplace=False):
    ncontigs = tensor.shape[0]
    mean = tensor.mean(dim=1).view((ncontigs, 1))
    std = tensor.std(dim=1).view((ncontigs, 1))
    std[std == 0] = 1

    if inplace:
        tensor -= mean
    else:
        tensor = tensor - mean

    tensor /= std
    return tensor

def _getinner(tensor, point, inner_threshold):
    """Gets the distance vector, array of inner points and average distance
    to inner points from a starting point"""

    distances = _pearson_distances(tensor, point)
    mask = distances <= inner_threshold
    inner_dists = distances[mask]

    if len(inner_dists) == 1:
        average_distance = 0
    else:
        average_distance = inner_dists.sum().item() / (len(inner_dists) - 1)

    return distances, mask, len(inner_dists), average_distance

def _sample_clusters(tensor, point, max_attempts, inner_threshold, outer_threshold, randomstate):
    """Keeps sampling new points within the inner points until it has sampled
    max_attempts without getting a new set of inner points with lower average
    distance"""

    futile_attempts = 0

    # Keep track of tried points to avoid sampling the same more than once
    tried = {point}

    distances, inner_mask, ninner, average_distance = _getinner(tensor, point, inner_threshold)

    while ninner - len(tried) > 0 and futile_attempts < max_attempts:
        inner_points = inner_mask.nonzero().squeeze()

        sample = randomstate.choice(inner_points).item()
        while sample in tried: # Not sure there is a faster way to prevent resampling
            sample = randomstate.choice(inner_points).item()

        tried.add(sample)

        inner = _getinner(tensor, sample, inner_threshold)
        sample_dist, sample_mask, sample_ninner, sample_average = inner

        if sample_average < average_distance:
            point = sample
            inner_mask = sample_mask
            ninner = sample_ninner
            average_distance = sample_average
            distances = sample_dist
            futile_attempts = 0
            tried = {point}

        else:
            futile_attempts += 1

    outer_mask = distances <= outer_threshold

    return point, inner_mask, outer_mask, ninner

def _cluster(tensor, labels, inner_threshold, outer_threshold, max_steps, cuda):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""

    randomstate = _random.Random(324645)
    device = 'cuda' if cuda else 'cpu'

    # This list keeps track of points to remove because they compose the inner circle
    # of clusters. We only remove points when we create a cluster with more than one
    # point, since one-point-clusters don't interfere with other clusters and the
    # point removal operations are expensive.
    toremove = list()

    # This index keeps track of which point we initialize clusters from.
    # It's necessary since we don't remove points after every cluster.
    seed = 0
    removemask = _torch.zeros(len(tensor), dtype=_torch.uint8)
    if cuda:
        removemask = removemask.cuda()

    while len(tensor) > 0:
        # Most extreme point (without picking same point twice)

        # Find medoid using iterative sampling function above
        sampling = _sample_clusters(tensor, seed, max_steps, inner_threshold, outer_threshold, randomstate)
        medoid, inner_mask, outer_mask, ninner = sampling
        removemask |= inner_mask

        # Write data to output
        yield labels[medoid].item(), set(labels[outer_mask].cpu().numpy())
        seed += 1

        # Only remove points if we have more than 1 point in cluster
        # Note that these operations are really expensive.
        if ninner > 1 or len(tensor) == seed:
            keepmask = ~removemask

            tensor = tensor[keepmask]
            labels = labels[keepmask]

            removemask &= keepmask
            removemask = removemask[:len(tensor)]


            seed = 0

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

def _check_params(tensor, inner, outer, labels, nsamples, maxsize, cuda):
    """Checks tensor, labels, nsamples, maxsize and estimates inner if necessary."""

    if len(tensor) < 1:
        raise ValueError('Matrix must have at least 1 observation.')

    if labels is None:
        labels = _torch.arange(1, len(tensor) + 1, dtype=_torch.int32)
        if cuda:
            labels = labels.cuda()

    elif type(labels) != _torch.tensor or len(labels) != len(tensor):
        raise ValueError('labels must be a 1D Numpy array with same length as tensor')

    if len(set(labels)) != len(tensor):
        raise ValueError('Labels must be unique')

    if len(tensor) < 1000:
        raise ValueError('Cannot estimate from less than 1000 contigs')

    if len(tensor) < nsamples:
        raise ValueError('Specified more samples than available contigs')

    if maxsize < 1:
        raise ValueError('maxsize must be positive number')

    if inner is None:
        try:
            _gt = _threshold.getthreshold(tensor, _pearson_distances, nsamples, maxsize)
            inner, support, separation = _gt
            outer = inner

            if separation < 0.25:
                sep = round(separation * 100, 1)
                wn = 'Warning: Only {}% of contigs has well-separated threshold'
                print(wn.format(sep), file=_sys.stderr)

            if support < 0.50:
                sup = round(support * 100, 1)
                wn = 'Warning: Only {}% of contigs has *any* observable threshold'
                print(wn.format(sup), file=_sys.stderr)

        except _threshold.TooLittleData as error:
            wn = 'Warning: Too little data: {}. Setting threshold to 0.08'
            print(wn.format(error.args[0]), file=_sys.stderr)
            inner = outer = 0.08

    return labels, inner, outer

def cluster(tensor, labels=None, inner=None, outer=None, max_steps=15,
            normalized=False, nsamples=1000, maxsize=2500, cuda=False):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.

    Inputs:
        tensor: A (obs x features) PyTorch tensor of values
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
        tensor = _tensor_normalize(tensor)

    inner, outer = _check_inputs(max_steps, inner, outer)
    labels, inner, outer = _check_params(tensor, inner, outer, labels, nsamples, maxsize, cuda)

    return _cluster(tensor, labels, inner, outer, max_steps, cuda)

def writeclusters(filehandle, clusters, max_clusters=None, min_size=1,
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
