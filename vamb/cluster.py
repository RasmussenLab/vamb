__doc__ = """Iterative medoid clustering.

Usage:
>>> clusters = list(ClusterIterator(matrix))

Implements one core function, cluster, along with the helper
functions write_clusters and read_clusters.
"""

import random as _random
import numpy as _np
import torch as _torch
from collections import deque as _deque
from math import ceil as _ceil
from torch.functional import Tensor as _Tensor
import vamb.vambtools as _vambtools
from typing import Optional
from collections.abc import Sequence, Iterable

_DEFAULT_RADIUS = 0.06
# Distance within which to search for medoid point
_MEDOID_RADIUS = 0.05

_DELTA_X = 0.005
_XMAX = 0.3

# This is the PDF of normal with µ=0, s=0.01 from -0.075 to 0.075 with intervals
# of DELTA_X, for a total of 31 values. We multiply by _DELTA_X so the density
# of one point sums to approximately one
_NORMALPDF = _DELTA_X * _Tensor(
    [
        2.43432053e-11,
        9.13472041e-10,
        2.66955661e-08,
        6.07588285e-07,
        1.07697600e-05,
        1.48671951e-04,
        1.59837411e-03,
        1.33830226e-02,
        8.72682695e-02,
        4.43184841e-01,
        1.75283005e00,
        5.39909665e00,
        1.29517596e01,
        2.41970725e01,
        3.52065327e01,
        3.98942280e01,
        3.52065327e01,
        2.41970725e01,
        1.29517596e01,
        5.39909665e00,
        1.75283005e00,
        4.43184841e-01,
        8.72682695e-02,
        1.33830226e-02,
        1.59837411e-03,
        1.48671951e-04,
        1.07697600e-05,
        6.07588285e-07,
        2.66955661e-08,
        9.13472041e-10,
        2.43432053e-11,
    ]
)


class Cluster:
    __slots__ = [
        "medoid",
        "seed",
        "members",
        "pvr",
        "radius",
        "isdefault",
        "successes",
        "attempts",
    ]

    def __init__(
        self,
        medoid: int,
        seed: int,
        members: _np.ndarray,
        pvr: float,
        radius: float,
        isdefault: bool,
        successes: int,
        attempts: int,
    ):
        self.medoid = medoid
        self.seed = seed
        self.members = members
        self.pvr = pvr
        self.radius = radius
        self.isdefault = isdefault
        self.successes = successes
        self.attempts = attempts

    def __repr__(self) -> str:
        return f"<Cluster of medoid {self.medoid}, {len(self.members)} members>"

    def as_tuple(self) -> tuple[int, set[int]]:
        return (self.medoid, {i for i in self.members})

    def dump(self) -> str:
        return (
            f"{self.medoid}\t{self.seed}\t{self.pvr}\t{self.radius}\t{self.isdefault}"
            f"\t{self.successes}\t{self.attempts}\t"
        ) + ",".join([str(i) for i in self.members])

    def __str__(self) -> str:
        radius = f"{self.radius:.3f}"
        if self.isdefault:
            radius += " (fallback)"

        return f"""Cluster of medoid {self.medoid}
  N members: {len(self.members)}
  seed:      {self.seed}
  radius:    {radius}
  successes: {self.successes} / {self.attempts}
  pvr:       {self.pvr:.1f}
  """


class ClusterGenerator:
    """Iterative medoid cluster generator. Iterate this object to get clusters.

    Inputs:
        matrix: A (obs x features) Numpy matrix of data type numpy.float32
        maxsteps: Stop searching for optimal medoid after N futile attempts [25]
        windowsize: Length of window to count successes [200]
        minsuccesses: Minimum acceptable number of successes [15]
        destroy: Save memory by destroying matrix while clustering [False]
        normalized: Matrix is already preprocessed [False]
        cuda: Accelerate clustering with GPU [False]
    """

    __slots__ = [
        "maxsteps",
        "minsuccesses",
        "cuda",
        "rng",
        "matrix",
        "indices",
        "seed",
        "nclusters",
        "peak_valley_ratio",
        "attempts",
        "successes",
        "histogram",
        "kept_mask",
    ]

    def __repr__(self) -> str:
        return f"ClusterGenerator({len(self.matrix)} points, {self.nclusters} clusters)"

    def __str__(self) -> str:
        return f"""ClusterGenerator({len(self.matrix)} points, {self.nclusters} clusters)
  CUDA:         {self.cuda}
  maxsteps:     {self.maxsteps}
  minsuccesses: {self.minsuccesses}
  pvr:          {self.peak_valley_ratio}
  successes:    {self.successes}/{len(self.attempts)}
"""

    def _check_params(
        self, matrix: _np.ndarray, maxsteps: int, windowsize: int, minsuccesses: int
    ) -> None:
        """Checks matrix, and maxsteps."""

        if matrix.dtype != _np.float32:
            raise (ValueError("Matrix must be of dtype float32"))

        if maxsteps < 1:
            raise ValueError(f"maxsteps must be a positive integer, not {maxsteps}")

        if windowsize < 1:
            raise ValueError(f"windowsize must be at least 1, not {windowsize}")

        if minsuccesses < 1 or minsuccesses > windowsize:
            raise ValueError(
                f"minsuccesses must be between 1 and windowsize, not {minsuccesses}"
            )

        if len(matrix) < 1:
            raise ValueError("Matrix must have at least 1 observation.")

    def _init_histogram_kept_mask(self, N: int) -> tuple[_Tensor, _Tensor]:
        "N is number of contigs"

        kept_mask = _torch.ones(N, dtype=_torch.bool)
        if self.cuda:
            histogram = _torch.empty(_ceil(_XMAX / _DELTA_X), dtype=_torch.float).cuda()
            kept_mask = kept_mask.cuda()
        else:
            histogram = _torch.empty(_ceil(_XMAX / _DELTA_X))

        return histogram, kept_mask

    def __init__(
        self,
        matrix: _np.ndarray,
        maxsteps: int = 25,
        windowsize: int = 200,
        minsuccesses: int = 20,
        destroy: bool = False,
        normalized: bool = False,
        cuda: bool = False,
    ):
        self._check_params(matrix, maxsteps, windowsize, minsuccesses)
        if not destroy:
            matrix = matrix.copy()

        # Shuffle matrix in unison to prevent seed sampling bias. Indices keeps
        # track of which points are which.
        _np.random.RandomState(0).shuffle(matrix)
        indices = _np.random.RandomState(0).permutation(len(matrix))
        indices = _torch.from_numpy(indices)
        torch_matrix = _torch.from_numpy(matrix)

        if not normalized:
            _normalize(torch_matrix, inplace=True)

        # Move to GPU
        if cuda:
            torch_matrix = torch_matrix.cuda()

        self.maxsteps: int = maxsteps
        self.minsuccesses: int = minsuccesses
        self.cuda: bool = cuda
        self.rng: _random.Random = _random.Random(0)

        self.matrix = torch_matrix
        # This refers to the indices of the original matrix. As we remove points, these
        # indices do not correspond to merely range(len(matrix)) anymore.
        self.indices = indices
        self.seed = -1
        self.nclusters = 0
        self.peak_valley_ratio = 0.1
        self.attempts: _deque[bool] = _deque(maxlen=windowsize)
        self.successes = 0

        histogram, kept_mask = self._init_histogram_kept_mask(len(indices))
        self.histogram = histogram
        self.kept_mask = kept_mask

    # It's an iterator itself
    def __iter__(self):
        return self

    def __next__(self) -> Cluster:
        # Stop criterion. For CUDA, inplace masking the array is too slow, so the matrix is
        # unchanged. On CPU, we continually modify the matrix by removing rows.
        if self.cuda:
            if not _torch.any(self.kept_mask).item():
                raise StopIteration
        elif len(self.matrix) == 0:
            raise StopIteration

        cluster, _, points = self._findcluster()
        self.nclusters += 1

        for point in points:
            self.kept_mask[point] = 0

        # Remove all points that's been clustered away. Is slow it itself, but speeds up
        # distance calculation by having fewer points. Worth it on CPU, not on GPU
        if not self.cuda:
            _vambtools.torch_inplace_maskarray(self.matrix, self.kept_mask)
            # no need to inplace mask small array
            self.indices = self.indices[self.kept_mask]
            self.kept_mask.resize_(len(self.matrix))
            self.kept_mask[:] = 1

        return cluster

    def _findcluster(self) -> tuple[Cluster, int, _Tensor]:
        """Finds a cluster to output."""
        threshold, success, medoid = None, None, -1

        # Keep looping until we find a cluster
        distances = None
        while threshold is None:
            # If on GPU, we need to take next seed which has not already been clusted out.
            # if not, clustered points have been removed, so we can just take next seed
            if self.cuda:
                self.seed = (self.seed + 1) % len(self.matrix)
                while self.kept_mask[self.seed] == False:
                    self.seed = (self.seed + 1) % len(self.matrix)
            else:
                self.seed = (self.seed + 1) % len(self.matrix)

            medoid, distances = _wander_medoid(
                self.matrix,
                self.kept_mask,
                self.seed,
                self.maxsteps,
                self.rng,
                self.cuda,
            )

            # We need to make a histogram of only the unclustered distances - when run on GPU
            # these have not been removed and we must use the kept_mask
            if self.cuda:
                _torch.histc(
                    distances[self.kept_mask],
                    len(self.histogram),
                    0,
                    _XMAX,
                    out=self.histogram,
                )
            else:
                _torch.histc(
                    distances, len(self.histogram), 0, _XMAX, out=self.histogram
                )
            self.histogram[0] -= 1  # Remove distance to self

            threshold, success = _find_threshold(
                self.histogram, self.peak_valley_ratio, self.cuda
            )

            # If success is not None, either threshold detection failed or succeded.
            if success is not None:
                # Keep accurately track of successes if we exceed maxlen
                if len(self.attempts) == self.attempts.maxlen:
                    self.successes -= self.attempts.popleft()

                # Add the current success to count
                self.successes += success
                self.attempts.append(success)

                # If less than minsuccesses of the last maxlen attempts were successful,
                # we relax the clustering criteria and reset counting successes.
                if (
                    len(self.attempts) == self.attempts.maxlen
                    and self.successes < self.minsuccesses
                ):
                    self.peak_valley_ratio += 0.1
                    self.attempts.clear()
                    self.successes = 0

        # These are the points of the final cluster AFTER establishing the threshold used
        assert isinstance(distances, _Tensor)
        points = _smaller_indices(distances, self.kept_mask, threshold, self.cuda)
        isdefault = (
            success is None
            and threshold == _DEFAULT_RADIUS
            and self.peak_valley_ratio > 0.55
        )

        cluster = Cluster(
            int(self.indices[medoid].item()),  # type: ignore
            self.seed,
            self.indices[points].numpy(),
            self.peak_valley_ratio,
            threshold,
            isdefault,
            self.successes,
            len(self.attempts),
        )
        return cluster, medoid, points


def _calc_densities(
    histogram: _Tensor, cuda: bool, pdf: _Tensor = _NORMALPDF
) -> _Tensor:
    """Given an array of histogram, smoothes the histogram."""
    pdf_len = len(pdf)

    if cuda:
        histogram = histogram.cpu()

    densities = _torch.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(densities) - pdf_len + 1):
        densities[i : i + pdf_len] += pdf * histogram[i]

    densities = densities[15:-15]

    return densities


def _find_threshold(
    histogram: _Tensor, peak_valley_ratio: float, cuda: bool
) -> tuple[Optional[float], Optional[bool]]:
    """Find a threshold distance, where where is a dip in point density
    that separates an initial peak in densities from the larger bulk around 0.5.
    Returns (threshold, success), where succes is False if no threshold could
    be found, True if a good threshold could be found, and None if the point is
    alone, or the threshold has been used.
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
    if histogram[:10].sum().item() == 0:
        return 0.025, None

    densities = _calc_densities(histogram, cuda)

    # Else we analyze the point densities to find the valley
    x = 0
    density_at_minimum = 0.0
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
    if threshold is not None and threshold > 0.2 + peak_valley_ratio:
        threshold = None
        success = False

    # If ratio has been set to 0.6, we do not accept returning no threshold.
    if threshold is None and peak_valley_ratio > 0.55:
        threshold = _DEFAULT_RADIUS
        success = None

    return threshold, success


def _smaller_indices(
    tensor: _Tensor, kept_mask: _Tensor, threshold: float, cuda: bool
) -> _Tensor:
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


def _normalize(matrix: _Tensor, inplace: bool = False) -> _Tensor:
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
    matrix[zeromask] = 1 / matrix.shape[1]
    matrix /= matrix.norm(dim=1).reshape(-1, 1) * (2**0.5)
    return matrix


def _calc_distances(matrix: _Tensor, index: int) -> _Tensor:
    "Return vector of cosine distances from rows of normalized matrix to given row."
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0  # avoid float rounding errors
    return dists


def _sample_medoid(
    matrix: _Tensor, kept_mask: _Tensor, medoid: int, threshold: float, cuda: bool
) -> tuple[_Tensor, _Tensor, float]:
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


def _wander_medoid(
    matrix: _Tensor,
    kept_mask: _Tensor,
    medoid: int,
    max_attempts: int,
    rng: _random.Random,
    cuda: bool,
) -> tuple[int, _Tensor]:
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0
    tried = {medoid}  # keep track of already-tried medoids
    cluster, distances, average_distance = _sample_medoid(
        matrix, kept_mask, medoid, _MEDOID_RADIUS, cuda
    )

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = int(rng.choice(cluster).item())

        # Prevent sampling same medoid multiple times.
        while sampled_medoid in tried:
            sampled_medoid = int(rng.choice(cluster).item())

        tried.add(sampled_medoid)

        sampling = _sample_medoid(
            matrix, kept_mask, sampled_medoid, _MEDOID_RADIUS, cuda
        )
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


def pairs(
    clustergenerator: ClusterGenerator, labels: Sequence[str]
) -> Iterable[tuple[str, set[str]]]:
    """Create an iterable of (N, {label1, label2 ...}) for each
    cluster in a ClusterGenerator, where N is "1", "2", "3", etc.
    Useful to pass to e.g. vambtools.writer_clusters.

    Inputs:
        clustergenerator: A ClusterGenerator object
        labels: List or array of cluster labels
    Output:
        Generator yielding ("1", {label1, label2 ... }) for each cluster
    """
    maxindex = clustergenerator.indices.max()
    if len(labels) <= maxindex:
        raise ValueError(
            f"Cluster generator contains point with index {maxindex}, "
            f"but was given only {len(labels)} labels"
        )

    return (
        (str(i + 1), {labels[j] for j in c.as_tuple()[1]})
        for (i, c) in enumerate(clustergenerator)
    )
