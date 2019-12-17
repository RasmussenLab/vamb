__doc__ = """Iterative medoid clustering.

Usage:
>>> cluster_iterator = cluster(matrix, labels=contignames)
>>> clusters = dict(cluster_iterator)

Implements one core function, cluster, along with the helper
functions write_clusters and read_clusters.
For all functions in this module, a collection of clusters are represented as
a {clustername, set(elements)} dict.
"""

import random as _random
import numpy as _np
import torch as _torch
from collections import defaultdict as _defaultdict, deque as _deque
from math import ceil as _ceil
import vamb.vambtools as _vambtools

_DELTA_X = 0.005
_XMAX = 0.3

# This is the PDF of normal with µ=0, s=0.01 from -0.075 to 0.075 with intervals
# of DELTA_X, for a total of 31 values. We multiply by _DELTA_X so the density
# of one point sums to approximately one
_NORMALPDF = _DELTA_X * _torch.Tensor(
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

def _calc_densities(histogram, cuda, pdf=_NORMALPDF):
    """Given an array of histogram, smoothes the histogram."""
    pdf_len = len(pdf)

    if cuda:
        histogram = histogram.cpu()

    densities = _torch.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(densities) - pdf_len + 1):
        densities[i:i+pdf_len] += pdf * histogram[i]

    densities = densities[15:-15]

    return densities

def _find_threshold(histogram, peak_valley_ratio, default, cuda):
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
    delta_x = _XMAX / len(histogram)

    # If the point is a loner, immediately return a threshold in where only
    # that point is contained.
    if histogram[:6].sum().item() == 0:
        return 0.025, None

    densities = _calc_densities(histogram, cuda)

    # Else we analyze the point densities to find the valley
    x = 0
    for density in densities:
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

        x += delta_x

    # Don't allow a threshold too high - this is relaxed with p_v_ratio
    if threshold is not None and threshold > 0.14 + 0.1 * peak_valley_ratio:
        threshold = None
        success = False

    # If ratio has been set to 0.6, we do not accept returning no threshold.
    if threshold is None and peak_valley_ratio > 0.55:
        threshold = default
        success = None

    return threshold, success

def _smaller_indices(tensor, kept_mask, threshold, cuda):
    """Get all indices where the tensor is smaller than the threshold.
    Uses Numpy because Torch is slow - See https://github.com/pytorch/pytorch/pull/15190"""

    # If it's on GPU, we remove the already clustered points at this step.
    if cuda:
        return _torch.nonzero((tensor <= threshold) & kept_mask).flatten()
    else:
        arr = tensor.numpy()
        indices = (arr <= threshold).nonzero()[0]
        torch_indices = _torch.from_numpy(indices)
        return torch_indices

def _normalize(matrix, inplace=False):
    """Preprocess the matrix to make distance calculations faster.
    The distance functions in this module assumes input has been normalized
    and will not work otherwise.
    """
    if isinstance(matrix, _np.ndarray):
        matrix = _torch.from_numpy(matrix)

    if not inplace:
        matrix = matrix.clone()

    # If any rows are kept all zeros, the distance function will return 0.5 to all points
    # inclusive itself, which can break the code in this module
    zeromask = matrix.sum(dim=1) == 0
    matrix[zeromask] = 1/matrix.shape[1]
    matrix /= (matrix.norm(dim=1).reshape(-1, 1) * (2 ** 0.5))
    return matrix


def _calc_distances(matrix, index):
    "Return vector of cosine distances from rows of normalized matrix to given row."
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0 # avoid float rounding errors
    return dists

def _sample_medoid(matrix, kept_mask, medoid, threshold, cuda):
    """Returns:
    - A vector of indices to points within threshold
    - A vector of distances to all points
    - The mean distance from medoid to the other points in the first vector
    """

    distances = _calc_distances(matrix, medoid)
    cluster = _smaller_indices(distances, kept_mask, threshold, cuda)

    if len(cluster) == 1:
        average_distance = 0.0
    else:
        average_distance = distances[cluster].sum().item() / (len(cluster) - 1)

    return cluster, distances, average_distance


def _wander_medoid(matrix, kept_mask, medoid, max_attempts, rng, cuda):
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0
    tried = {medoid} # keep track of already-tried medoids
    cluster, distances, average_distance = _sample_medoid(matrix, kept_mask, medoid, _MEDOID_RADIUS, cuda)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = rng.choice(cluster).item()

         # Prevent sampling same medoid multiple times.
        while sampled_medoid in tried:
            sampled_medoid = rng.choice(cluster).item()

        tried.add(sampled_medoid)

        sampling = _sample_medoid(matrix, kept_mask, sampled_medoid, _MEDOID_RADIUS, cuda)
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


def _findcluster(matrix, kept_mask, histogram, seed, peak_valley_ratio, max_steps, minsuccesses, default, rng, attempts, cuda):
    """Finds a cluster to output."""
    threshold = None
    successes = sum(attempts)

    while threshold is None:
        # If on GPU, we need to take next seed which has not already been clusted out.
        # if not, clustered points have been removed, so we can just take next seed
        if cuda:
            seed = (seed + 1) % len(matrix)
            while kept_mask[seed] == False:
                seed = (seed + 1) % len(matrix)
        else:
            seed = (seed + 1) % len(matrix)

        medoid, distances = _wander_medoid(matrix, kept_mask, seed, max_steps, rng, cuda)

        # We need to make a histogram of only the unclustered distances - when run on GPU
        # these have not been removed and we must use the kept_mask
        if cuda:
            _torch.histc(distances[kept_mask], len(histogram), 0, _XMAX, out=histogram)
        else:
            _torch.histc(distances, len(histogram), 0, _XMAX, out=histogram)
        histogram[0] -= 1 # Remove distance to self

        threshold, success = _find_threshold(histogram, peak_valley_ratio, default, cuda)

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

    # This is the final cluster AFTER establishing the threshold used
    cluster = _smaller_indices(distances, kept_mask, threshold, cuda)
    return cluster, medoid, seed, peak_valley_ratio


def _cluster(matrix, labels, indices, max_steps, windowsize, minsuccesses, default, cuda):
    """Yields (medoid, points) pairs from a (obs x features) matrix"""
    seed = -1
    rng = _random.Random(0)
    peak_valley_ratio = 0.1
    attempts = _deque(maxlen=windowsize)

    if _torch.__version__ >= '1.2':
        # https://github.com/pytorch/pytorch/issues/20208 fixed in PyTorch 1.2
        cuda_hist_dtype = _torch.float
        # Masking using uint8 tensors deprecated in PyTorch 1.2
        kept_mask_dtype = _torch.bool
    else:
         cuda_hist_dtype = _torch.long
         kept_mask_dtype = _torch.uint8

    kept_mask = _torch.ones(len(indices), dtype=kept_mask_dtype)

    if cuda:
        histogram = _torch.empty(_ceil(_XMAX/_DELTA_X), dtype=cuda_hist_dtype).cuda()
        kept_mask = kept_mask.cuda()
    else:
        histogram = _torch.empty(_ceil(_XMAX/_DELTA_X))

    done = False
    while not done:
        _ = _findcluster(matrix, kept_mask, histogram, seed, peak_valley_ratio, max_steps,
                         minsuccesses, default, rng, attempts, cuda)
        cluster, medoid, seed, peak_valley_ratio = _

        # Write data to output. We use indices instead of labels directly to
        # prevent masking and thereby duplicating large string arrays
        if labels is None:
            yield indices[medoid].item(), {i.item() for i in indices[cluster]}
        else:
            yield labels[indices[medoid].item()], {labels[indices[i].item()] for i in cluster}

        for point in cluster:
            kept_mask[point] = 0

        # If we use CUDA and the arrays are on GPU, it's too slow to modify the arrays and
        # faster to simply mask.
        if not cuda:
            _vambtools.torch_inplace_maskarray(matrix, kept_mask)
            indices = indices[kept_mask] # no need to inplace mask small array
            kept_mask.resize_(len(matrix))
            kept_mask[:] = 1
            done = len(matrix) == 0
        else:
            done = not _torch.any(kept_mask).item()


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


def cluster(matrix, labels=None, maxsteps=25, windowsize=200, minsuccesses=20,
            default=0.09, destroy=False, normalized=False, logfile=None, cuda=False):
    """Iterative medoid cluster generator. Yields (medoid), set(labels) pairs.

    Inputs:
        matrix: A (obs x features) Numpy matrix of data type numpy.float32
        labels: None or Numpy array/list with labels for seqs [None = indices]
        maxsteps: Stop searching for optimal medoid after N futile attempts [25]
        default: Fallback threshold if cannot be estimated [0.09]
        windowsize: Length of window to count successes [200]
        minsuccesses: Minimum acceptable number of successes [15]
        destroy: Save memory by destroying matrix while clustering [False]
        normalized: Matrix is already preprocessed [False]
        logfile: Print threshold estimates and certainty to file [None]

    Output: Generator of (medoid, set(labels_in_cluster)) tuples.
    """

    if matrix.dtype != _np.float32:
        raise(ValueError("Matrix must be of dtype float32"))

    if not destroy:
        matrix = matrix.copy()

    # Shuffle matrix in unison to prevent seed sampling bias. Indices keeps
    # track of which points are which.
    _np.random.RandomState(0).shuffle(matrix)
    indices = _np.random.RandomState(0).permutation(len(matrix))

    indices = _torch.from_numpy(indices)
    matrix = _torch.from_numpy(matrix)

    if not normalized:
        _normalize(matrix, inplace=True)

    _check_params(matrix, labels, maxsteps, windowsize, minsuccesses, default, logfile)

    # Move to GPU
    if cuda:
        matrix = matrix.cuda()

    return _cluster(matrix, labels, indices, maxsteps, windowsize, minsuccesses, default, cuda)


def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None, rename=True):
    """Writes clusters to an open filehandle.

    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or a dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add
        rename: Rename clusters to "cluster_1", "cluster_2" etc.

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

        if rename:
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
