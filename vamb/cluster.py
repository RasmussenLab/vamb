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
(2): Define cluster(S) = all observations with cosine distance from S < THRESHOLD
(3): Sample MOVES observations I from cluster(S)
(4): If any mean(cluster(i)) < mean(cluster(S)) for i in I: Let S be i, go to (2)
     Else: cluster(S) = all observations with cosine distance from S < THRESHOLD
(5): Output cluster(S) as cluster, remove cluster(S) from observations
(6): If no more observations or MAX_CLUSTERS have been reached: Stop
     Else: Go to (1)
"""

import sys as _sys
import os as _os
import random as _random
import numpy as _np
import torch as _torch
from collections import defaultdict as _defaultdict, deque as _deque
import vamb.vambtools as _vambtools

_DELTA_X = 0.005
_XS = _np.arange(0, 0.3, _DELTA_X)

# This is the PDF of normal with µ=0, s=0.01 from -0.075 to 0.075 with intervals
# of DELTA_X, for a total of 31 values. We multiply by _DELTA_X so the density
# of one point sums to approximately one
_NORMALPDF = _DELTA_X * _np.array(
      [2.43432053e-11, 9.13472041e-10, 2.66955661e-08, 6.07588285e-07,
       1.07697600e-05, 1.48671951e-04, 1.59837411e-03, 1.33830226e-02,
       8.72682695e-02, 4.43184841e-01, 1.75283005e+00, 5.39909665e+00,
       1.29517596e+01, 2.41970725e+01, 3.52065327e+01, 3.98942280e+01,
       3.52065327e+01, 2.41970725e+01, 1.29517596e+01, 5.39909665e+00,
       1.75283005e+00, 4.43184841e-01, 8.72682695e-02, 1.33830226e-02,
       1.59837411e-03, 1.48671951e-04, 1.07697600e-05, 6.07588285e-07,
       2.66955661e-08, 9.13472041e-10, 2.43432053e-11])

# Distance within which to search for medoid point
_MEDOID_RADIUS = 0.05

def _calc_histogram(distances, bins=_XS):
    """Calculates number of OTHER points within the bins given by bins."""
    histogram, _ = _np.histogram(distances, bins=bins)

    # compensate for self-correlation of chosen contig. This is not always true
    # due to floating point errors, yielding negative distance to self.
    if histogram[0] > 0:
        histogram[0] -= 1

    return histogram

def _calc_densities(histogram, pdf=_NORMALPDF):
    """Given an array of histogram, smoothes the histogram."""
    pdf_len = len(pdf)
    densities = _np.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(densities) - pdf_len + 1):
        densities[i:i+pdf_len] += pdf * histogram[i]

    densities = densities[15:-15]

    return densities

def _find_threshold(distances, peak_valley_ratio, default, xs=_XS[1:]):
    """Find a threshold distance, where where is a dip in point density
    that separates an initial peak in densities from the larger bulk around 0.5.
    Returns (threshold, success), where succes is False if no threshold could
    be found, True if a good threshold could be found, and None if the point is
    alone, or the default threshold has been used.
    """
    peak_density = 0
    peak_over = False
    minimum_x = None
    density_at_minimum = None
    threshold = None
    success = False
    histogram = _calc_histogram(distances)

    # If the point is a loner, immediately return a threshold in where only
    # that point is contained.
    if histogram[:6].sum() == 0:
        return 0.025, None

    densities = _calc_densities(histogram)

    # Else we analyze the point densities to find the valley
    for x, density in zip(xs, densities):
        # Define the first "peak" in point density. That's simply the max until
        # the peak is defined as being over.
        if not peak_over and density > peak_density:
            # Do not accept first peak to be after x = 0.1
            if x > 0.1:
                break
            peak_density = density

        # Peak is over when density drops below 60% of peak density
        if not peak_over and density < 0.6 * peak_density:
            peak_over = True
            density_at_minimum = density

        # If another peak is detected, we stop
        if peak_over and density > 1.5 * density_at_minimum:
            break

        # Now find the minimum after the peak
        if peak_over and density < density_at_minimum:
            minimum_x, density_at_minimum = x, density

            # If this minimum is below ratio * peak, it's accepted as threshold
            if density < peak_valley_ratio * peak_density:
                threshold = minimum_x
                success = True

    # Don't allow a threshold too high - this is relaxed with p_v_ratio
    if threshold is not None and threshold > 0.14 + 0.1 * peak_valley_ratio:
        threshold = None
        success = False

    # If ratio has been set to 0.6, we do not accept returning no threshold.
    if threshold is None and peak_valley_ratio > 0.55:
        threshold = default
        success = None

    return threshold, success

def _normalize(matrix, inplace=False):
    """Preprocess the matrix to make distance calculations faster.
    The distance functions in this module assumes input has been normalized
    and will not work otherwise.
    """
    if not inplace:
        matrix = matrix.copy()

    # If next line is uncommented, result is Pearson distance, not cosine
    # matrix -= matrix.mean(axis=1).reshape((-1, 1))
    denominator = _np.linalg.norm(matrix, axis=1).reshape((-1, 1)) * (2 ** 0.5)
    denominator[denominator == 0] = 1
    matrix /= denominator
    return matrix

# Returns cosine or Pearson distance when matrix is normalized as above.
def _numpy_distances(matrix, index):
    "Return vector of distances from rows of normalized matrix to given row."
    return 0.5 - _np.dot(matrix, matrix[index].T)

def _torch_distances(tensor, index):
    "Return vector of distances from rows of normalized matrix to given row."
    return 0.5 - _torch.matmul(tensor, tensor[index])

def _numpy_getcluster(matrix, medoid, threshold):
    """Returns:
    - A vector of indices to points within threshold
    - A vector of distances to these points
    - The mean distance from medoid to the other inner points
    """

    distances = _numpy_distances(matrix, medoid)
    cluster = _np.where(distances <= threshold)[0]

    # This happens if std(matrix[points]) == 0, then all distances
    # become 0.5, even the distance to itself.
    if len(cluster) < 2:
        cluster = _np.array([medoid])
        average_distance = 0
    else:
        average_distance = _np.sum(distances[cluster]) / (len(cluster) - 1)

    return cluster, distances, average_distance

def _torch_getcluster(tensor, kept_mask, medoid, threshold):
    """
    Returns:
    - A boolean mask of which points are within the threshold
    - A vector of the indices of all points within threshold
    - The mean distance from medoid to all other points within threshold
    """

    distances = _torch_distances(tensor, medoid)
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

def _numpy_wander_medoid(matrix, medoid, max_attempts, rng):
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0
    tried = {medoid} # keep track of already-tried medoids
    cluster, distances, average_distance = _numpy_getcluster(matrix, medoid, _MEDOID_RADIUS)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = rng.choice(cluster)

         # Prevent sampling same medoid multiple times.
        while sampled_medoid in tried:
            sampled_medoid = rng.choice(cluster)

        tried.add(sampled_medoid)

        sampling = _numpy_getcluster(matrix, sampled_medoid, _MEDOID_RADIUS)
        sample_cluster, sample_distances, sample_avg = sampling

        # If the mean distance of inner points of the sample is lower,
        # we move the medoid and reset the futile_attempts count
        if sample_avg < average_distance:
            medoid = sampled_medoid
            cluster = sample_cluster
            average_distance = sample_avg
            futile_attempts = 0
            tried = {medoid}
            distances = sample_distances

        else:
            futile_attempts += 1

    return medoid, distances

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

def _numpy_findcluster(matrix, seed, peak_valley_ratio, max_steps, minsuccesses, default, rng, attempts):
    """Finds a cluster to output.
    """
    threshold = None
    successes = sum(attempts)

    while threshold is None:
        seed = (seed + 1) % len(matrix)
        medoid, distances = _numpy_wander_medoid(matrix, seed, max_steps, rng)
        threshold, success = _find_threshold(distances, peak_valley_ratio, default)

        # If success is not None, either threshold detection failed or succeded.
        if success is not None:
            # Keep accurately track of successes if we exceed maxlen
            if len(attempts) == attempts.maxlen:
                successes -= attempts.popleft()

            # Add the current success to count
            successes += success
            attempts.append(success)

            # If less than minsuccesses of the last maxlen attempts were successful,
            # we relax the clustering criteria and reset counting successes.
            if len(attempts) == attempts.maxlen and successes < minsuccesses:
                peak_valley_ratio += 0.1
                attempts.clear()
                successes = 0

    cluster = _np.where(distances <= threshold)[0]
    return cluster, medoid, seed, peak_valley_ratio

def _numpy_cluster(matrix, labels, indices, max_steps, windowsize, minsuccesses, default):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""
    seed = -1
    kept_mask = _np.ones(len(matrix), dtype=_np.bool)
    rng = _random.Random(0)
    peak_valley_ratio = 0.1
    attempts = _deque(maxlen=windowsize)

    while len(matrix) > 0:
        _ = _numpy_findcluster(matrix, seed, peak_valley_ratio, max_steps, minsuccesses, default, rng, attempts)
        cluster, medoid, seed, peak_valley_ratio = _

        # Write data to output. We use indices instead of labels directly to
        # prevent masking and thereby duplicating large string arrays
        if labels is None:
            yield indices[medoid]+1, {i+1 for i in indices[cluster]}
        else:
            yield labels[indices[medoid]], {labels[indices[i]] for i in cluster}

        for point in cluster:
            kept_mask[point] = False

        # To do: Can we sometimes skip doing this? It's CPU intensive.
        _vambtools.inplace_maskarray(matrix, kept_mask)
        indices = indices[kept_mask] # no need to inplace mask small array
        kept_mask.resize(len(matrix), refcheck=False)
        kept_mask[:] = True

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

def _check_params(matrix, labels, maxsteps, windowsize, minsuccesses, default, logfile):
    """Checks matrix, labels, and maxsteps."""

    if maxsteps < 1:
        raise ValueError('maxsteps must be a positive integer, not {}'.format(maxsteps))

    if windowsize < 1:
        raise ValueError('windowsize must be at least 1, not {}'.format(windowsize))

    if minsuccesses < 1 or minsuccesses > windowsize:
        raise ValueError('minsuccesses must be between 1 and windowsize, not {}'.format(minsuccesses))

    if default <= 0.0:
        raise ValueError('default threshold must be a positive float, not {}'.format(default))

    if len(matrix) < 1:
        raise ValueError('Matrix must have at least 1 observation.')

    if labels is not None:
        if len(labels) != len(matrix):
            raise ValueError('labels must have same length as matrix')

        if len(set(labels)) != len(matrix):
            raise ValueError('Labels must be unique')

    if matrix.dtype != _np.float32:
        raise ValueError('Matrix must be of data type np.float32')

def cluster(matrix, labels=None, maxsteps=25, windowsize=200, minsuccesses=20,
            default=0.09, destroy=False, normalized=False, cuda=False, logfile=None):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.

    Inputs:
        matrix: A (obs x features) Numpy matrix of data type numpy.float32
        labels: None or Numpy array/list with labels for seqs [None = indices+1]
        maxsteps: Stop searching for optimal medoid after N futile attempts [25]
        default: Fallback threshold if cannot be estimated [0.09]
        windowsize: Length of window to count successes [200]
        minsuccesses: Minimum acceptable number of successes [15]
        destroy: Save memory by destroying matrix while clustering [False]
        normalized: Matrix is already preprocessed [False]
        cuda: Use CUDA (GPU acceleration) based on PyTorch. [False]
        logfile: Print threshold estimates and certainty to file [None]

    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """
    if not destroy:
        matrix = matrix.copy()

    # Shuffle matrix in unison to prevent seed sampling bias. Indices keeps
    # track of which points are which.
    _np.random.RandomState(0).shuffle(matrix)
    indices = _np.random.RandomState(0).permutation(len(matrix))

    if not normalized:
        _normalize(matrix, inplace=True)

    _check_params(matrix, labels, maxsteps, windowsize, minsuccesses, default, logfile)

    if cuda:
        tensor = _torch.from_numpy(matrix)
        indices = _torch.from_numpy(indices)
        raise NotImplementedError('CUDA clustering not yet implemented.')
        return _torch_cluster(tensor, labels, indices, threshold, maxsteps)
    else:
        return _numpy_cluster(matrix, labels, indices, maxsteps, windowsize, minsuccesses, default)

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

    # If clusters is not iterator it must be a dict - transform it to iterator
    if iter(clusters) is not clusters:
        clusters = clusters.items()

    if max_clusters is not None and max_clusters < 1:
        raise ValueError('max_clusters must None or at least 1, not {}'.format(max_clusters))

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

        if clusternumber == max_clusters:
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
