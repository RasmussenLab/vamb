__doc__ = "Iterative medoid clustering"

import random as _random
import numpy as _np
import torch as _torch
from collections import deque as _deque, OrderedDict
from math import ceil as _ceil
from torch.functional import Tensor as _Tensor
import vamb.vambtools as _vambtools
from typing import TypeVar, Union, Optional, cast

_DEFAULT_RADIUS = 0.06
# Distance within which to search for medoid point
_MEDOID_RADIUS = 0.05

_DELTA_X = 0.005
_XMAX = 0.3

# We don't want to keep an indefinite amount of data points in the medoid cache,
# since it can be a memory hog.
MAX_CACHED_RESULTS = 64

Cl = TypeVar("Cl", bound="Cluster")


class Loner:
    __slots__ = []
    pass


class NoThreshold:
    __slots__ = []
    pass


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
        "maximal_pvr",
        "observed_pvr",
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
        maximal_pvr: float,
        # If loner or default: None
        observed_pvr: Optional[float],
        # If loner: None. If default, _DEFAULT_RADIUS
        radius: Optional[float],
        successes: int,
        attempts: int,
    ):
        self.medoid = medoid
        self.seed = seed
        self.members = members
        self.maximal_pvr = maximal_pvr
        self.observed_pvr = observed_pvr
        self.radius = radius
        self.successes = successes
        self.attempts = attempts

    @property
    def kind_str(self) -> str:
        if self.observed_pvr is not None:
            return "normal"
        else:
            if self.radius is None:
                return "loner"
            else:
                return "fallback"


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
        # Maximum number of futile steps taken in the wander_medoid function until it gives up
        # and emits the currently best medoid as the medoid
        "maxsteps",
        # Minimum number of successul clusters emitted of the last `attempts`, lest the
        # peak_valley_ratio be increased to prevent an infinite loop
        "minsuccesses",
        # Whether this clusterer runs on GPU
        "cuda",
        # Random number generator, currently used in wander_medoid function
        "rng",
        # Actual data to be clustered
        "matrix",
        # Lengths of input contigs
        "lengths",
        # This are the original indices of the rows of the matrix. Initially this is just 1..len(matrix),
        # but if not on GPU, we delete used rows of the matrix (and indices) to speed up subsequent computations.
        # Then, we can obtain the original row index by looking up in this array
        "indices",
        # Original indices are scored from most promising as seeds to worse, and ordered in the `order` array.
        # The best contig indices are first.
        "order",
        # The integer index in `order` which the clusterer on next iteration will try to use as medoid
        "order_index",
        "n_emitted_clusters",
        "n_remaining_points",
        # A float value which determines how strictly the clusterer rejects potential clusters.
        # The lower, the stricter. We increase this adaptively to avoid clustering for ever
        "peak_valley_ratio",
        # A deque to store whether the last N candicate clusters were rejected. See `minsuccesses`
        "attempts",
        # An integer storing the number of True values in `attempts` to avoid looping over it
        "successes",
        # A buffer in which a histrogram of distances from the current medoid is stored.
        # Is overwritten on each iteration, we just keep it here to avoid allocations.
        "histogram",
        "histogram_edges",
        # This bool array is False if the contig at the given index has been emitted in a previous iteration.
        # When matrix is on CPU, rows in this array (and the data matrix) is continuously deleted, and we use
        # this array to keep track of which rows to delete each iteration.
        # On GPU, deleting rows is not feasable because that requires us to copy the matrix from and to the GPU,
        # so we instead use this array to mask away any point emitted in an earlier iteration.
        "kept_mask",
        # This dict caches the computation of `sample_medoid`, which is the most compute heavy function
        "medoid_cache",
    ]

    def __repr__(self) -> str:
        return f"ClusterGenerator({len(self.matrix)} points, {self.n_emitted_clusters} clusters)"

    def __str__(self) -> str:
        return f"""ClusterGenerator({len(self.matrix)} points, {self.n_emitted_clusters} clusters)
  CUDA:         {self.cuda}
  maxsteps:     {self.maxsteps}
  minsuccesses: {self.minsuccesses}
  pvr:          {self.peak_valley_ratio}
  successes:    {self.successes}/{len(self.attempts)}
"""

    def _check_params(
        self,
        matrix: _np.ndarray,
        lengths: _np.ndarray,
        maxsteps: int,
        windowsize: int,
        minsuccesses: int,
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

        if len(lengths) != len(matrix):
            raise ValueError("N sequences in lengths and matrix do not match")

    def _init_histogram_kept_mask(self, N: int) -> tuple[_Tensor, _Tensor]:
        "N is number of contigs"

        kept_mask = _torch.ones(N, dtype=_torch.bool)
        if self.cuda:
            kept_mask = kept_mask.cuda()

        histogram = _torch.empty(_ceil(_XMAX / _DELTA_X))
        return histogram, kept_mask

    def __init__(
        self,
        matrix: _np.ndarray,
        lengths: _np.ndarray,
        maxsteps: int = 25,
        windowsize: int = 300,
        minsuccesses: int = 15,
        destroy: bool = False,
        normalized: bool = False,
        cuda: bool = False,
        rng_seed: int = 0,
    ):
        self._check_params(
            matrix,
            lengths,
            maxsteps,
            windowsize,
            minsuccesses,
        )
        if not destroy:
            matrix = matrix.copy()

        torch_matrix = _torch.from_numpy(matrix)
        if not normalized:
            _normalize(torch_matrix, inplace=True)

        # Move to GPU
        torch_lengths = _torch.Tensor(lengths)
        if cuda:
            torch_matrix = torch_matrix.cuda()
            torch_lengths = torch_lengths.cuda()

        self.maxsteps: int = maxsteps
        self.minsuccesses: int = minsuccesses
        self.cuda: bool = cuda
        self.rng: _random.Random = _random.Random(rng_seed)

        self.matrix = torch_matrix
        # This refers to the indices of the original matrix. As we remove points, these
        # indices do not correspond to merely range(len(matrix)) anymore.
        self.indices = _torch.arange(len(matrix))
        self.order = _np.argsort(lengths)[::-1]
        self.order_index = 0
        self.lengths = _torch.Tensor(lengths)
        if cuda:
            self.lengths = self.lengths.cuda()
        self.n_emitted_clusters = 0
        self.n_remaining_points = len(torch_matrix)
        self.peak_valley_ratio = 0.1
        self.attempts: _deque[bool] = _deque(maxlen=windowsize)
        self.successes = 0

        histogram, kept_mask = self._init_histogram_kept_mask(len(self.indices))
        self.histogram = histogram
        self.histogram_edges = _torch.linspace(0.0, _XMAX, round(_XMAX / _DELTA_X) + 1)
        self.kept_mask = kept_mask
        self.medoid_cache: OrderedDict[int, tuple[_Tensor, _Tensor, float]] = (
            OrderedDict()
        )

    # It's an iterator itself
    def __iter__(self):
        return self

    def __next__(self) -> Cluster:
        if self.n_remaining_points == 0:
            raise StopIteration
        assert self.n_remaining_points > 0  # not negative

        cluster, _, points = self.find_cluster()
        self.medoid_cache.clear()
        self.n_emitted_clusters += 1
        self.n_remaining_points -= len(points)

        for point in points:
            self.kept_mask[point] = 0

        # Remove all points that's been clustered away. Is slow it itself, but speeds up
        # distance calculation by having fewer points. Worth it on CPU, not on GPU
        if not self.cuda:
            self.pack()

        return cluster

    def pack(self):
        "Remove all used points from the matrix and indices, and reset kept_mask."
        if self.cuda:
            cpu_kept_mask = self.kept_mask.cpu()
            self.matrix = _vambtools.torch_inplace_maskarray(
                self.matrix.cpu(), cpu_kept_mask
            ).cuda()
            self.indices = self.indices[cpu_kept_mask]

        else:
            _vambtools.torch_inplace_maskarray(self.matrix, self.kept_mask)
            self.indices = self.indices[self.kept_mask]

        self.lengths = self.lengths[self.kept_mask]
        self.kept_mask.resize_(len(self.matrix))
        self.kept_mask[:] = 1
        # The medoid cache is invalidated when the indices shifts
        self.medoid_cache.clear()

    def pack_order(self):
        "Remove all used points from self.order"
        self.order = self.order[self.order > -1]
        assert len(self.order) > 0

    def get_next_seed(self) -> int:
        "Get the next seed index for a new medoid search"
        n_original_contigs = len(self.order)
        i = self.order_index - 1  # we increment by 1 in beginning of loop
        while True:
            # Get the order: That's the original index of the contig.
            i = (i + 1) % n_original_contigs

            # When we loop back to the first index after having passed over all indices before,
            # we potentially have many used up -1 values to skip, so we remove these
            # Since the clustering algorithm may loop over self.order many times, we can potentially
            # save time.
            # When running on GPU, we also take this (rare-ish) chance to pack the on-GPU matrix.
            # This speeds up future distance calculation by making the matrix smaller, but requires
            # moving the matrix from and to GPU, so is slow.
            # Doing it here, which we assume is relatively rarely may be a good compromise
            if i == 0 and self.n_emitted_clusters > 0:
                if self.cuda:
                    self.pack()
                self.pack_order()
                n_original_contigs = len(self.order)

            order = self.order[i]
            # -1 signify an index which has previously been used up
            if order == -1:
                continue

            # Find the new index of this old index
            new_index = cast(int, _torch.searchsorted(self.indices, order).item())
            # It's possible the seed contig `order` is part of a previously emitted cluster,
            # in which case it's not to be found in self.indices. In that case, mark the order
            # as -1, and go to the next one
            if (
                new_index >= len(self.indices)
                or self.indices[new_index].item() != order
                or (self.cuda and not self.kept_mask[new_index].item())
            ):
                self.order[i] = -1
                continue

            # Move to next index for the next time this is called
            self.order_index = i + 1
            return new_index

    def update_successes(self, success: bool):
        """Keeps track of how many clusters have been rejected (False) and accepted (True).
        When sufficiently many False has been seen, the peak_valley_ratio is bumped, which relaxes
        the criteria for rejection.
        This prevents the clusterer from getting stuck in an infinite loop.
        """

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

            # After relaxing criteria, start over from the best candidate
            # seed contigs which may have been skipped the first time around
            self.order_index = 0

    def wander_medoid(self, seed) -> tuple[int, _Tensor]:
        """Keeps sampling new points within the cluster until it has sampled
        max_attempts without getting a new set of cluster with lower average
        distance"""

        medoid = seed
        tried: set[int] = {medoid}
        cluster, distances, local_density = self.sample_medoid(seed)
        candidates: list[int] = [i for i in cluster.tolist() if i not in tried]
        candidates = self.rng.sample(candidates, k=min(len(candidates), self.maxsteps))
        i = 0

        while i < len(candidates):
            sampled_medoid = candidates[i]
            tried.add(sampled_medoid)
            sample_cluster, sample_distances, sample_density = self.sample_medoid(
                sampled_medoid
            )

            # If the mean distance of inner points of the sample is lower,
            # we move the medoid and start over
            if sample_density > local_density:
                medoid = sampled_medoid
                distances = sample_distances
                local_density = sample_density
                candidates: list[int] = [
                    i for i in sample_cluster.tolist() if i not in tried
                ]
                candidates = self.rng.sample(
                    candidates, k=min(len(candidates), self.maxsteps)
                )
                i = 0
            else:
                i += 1

        return (medoid, distances)

    def find_threshold(
        self, distances: _Tensor
    ) -> Union[Loner, NoThreshold, tuple[float, float]]:
        # If the point is a loner, immediately return a threshold in where only
        # that point is contained.
        if _torch.count_nonzero(distances < 0.05) == 1:
            return Loner()

        # We need to make a histogram of only the unclustered distances - when run on GPU
        # these have not been removed and we must use the kept_mask
        if self.cuda:
            below_xmax = (distances <= _XMAX) & self.kept_mask
            picked_distances = distances[below_xmax].cpu()
            picked_lengths = self.lengths[below_xmax].cpu()
        else:
            below_xmax = distances <= _XMAX
            picked_distances = distances[below_xmax]
            picked_lengths = self.lengths[below_xmax]

        # TODO: https://github.com/pytorch/pytorch/issues/69519
        # Currently, this function does not run on GPU. This means we must
        # copy over the lengths and distances to CPU each time, which is very slow.
        # If the issue is resolved, there can be large speedups on GPU
        _torch.histogram(
            input=picked_distances,
            bins=len(self.histogram),
            range=(0.0, _XMAX),
            out=((self.histogram, self.histogram_edges)),
            weight=picked_lengths,
        )

        peak_density = 0.0
        peak_over = False
        minimum_x = 0.0
        threshold = None
        delta_x = _XMAX / len(self.histogram)
        pdf_len = len(_NORMALPDF)

        if self.cuda:
            histogram = self.histogram.cpu()
        else:
            histogram = self.histogram

        # This smoothes out the histogram, so we can more reliably detect peaks
        # and valleys.
        densities = _torch.zeros(len(histogram) + pdf_len - 1)
        for i in range(len(densities) - pdf_len + 1):
            densities[i : i + pdf_len] += _NORMALPDF * histogram[i]
        densities = densities[15:-15]

        # Analyze the point densities to find the valley
        x = 0
        density_at_minimum = 0.0
        for density_ in densities:
            density = cast(float, density_.item())
            # Define the first "peak" in point density. That's simply the max until
            # the peak is defined as being over.
            if not peak_over and density > peak_density:
                # Do not accept first peak to be after x = 0.1
                if x > 0.1:
                    return NoThreshold()
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
                if density < self.peak_valley_ratio * peak_density:
                    threshold = minimum_x

            x += delta_x

        # If we have not detected a threshold, we can't return one.
        if threshold is None:
            return NoThreshold()
        # Else, we check whether the threshold is too high. If not, we return it.
        else:
            if threshold > 0.2 + self.peak_valley_ratio:
                return NoThreshold()
            else:
                observed_pvr = density_at_minimum / peak_density
                return (threshold, observed_pvr)

    def find_cluster(self) -> tuple[Cluster, int, _Tensor]:
        while True:
            seed = self.get_next_seed()
            medoid, distances = self.wander_medoid(seed)
            threshold = self.find_threshold(distances)
            if isinstance(threshold, Loner):
                cluster = Cluster(
                    int(self.indices[medoid].item()),  # type: ignore
                    seed,
                    _np.array([self.indices[medoid].item()]),
                    self.peak_valley_ratio,
                    None,
                    None,
                    self.successes,
                    len(self.attempts),
                )
                points = _torch.IntTensor([medoid])
                return (cluster, medoid, points)
            elif isinstance(threshold, NoThreshold):
                # When the peak_valley_ratio is too high, we need to return something to not get caught
                # in an infinite loop.
                if self.peak_valley_ratio > 0.55:
                    points = _smaller_indices(
                        distances, self.kept_mask, _DEFAULT_RADIUS, self.cuda
                    )
                    cluster = Cluster(
                        int(self.indices[medoid].item()),  # type: ignore
                        seed,
                        self.indices[points].numpy(),
                        self.peak_valley_ratio,
                        None,
                        _DEFAULT_RADIUS,
                        self.successes,
                        len(self.attempts),
                    )
                    return (cluster, medoid, points)
                else:
                    self.update_successes(False)

            elif isinstance(threshold, tuple):
                (threshold, observed_pvr) = threshold
                points = _smaller_indices(
                    distances, self.kept_mask, threshold, self.cuda
                )
                cluster = Cluster(
                    int(self.indices[medoid].item()),  # type: ignore
                    seed,
                    self.indices[points].numpy(),
                    self.peak_valley_ratio,
                    observed_pvr,
                    threshold,
                    self.successes,
                    len(self.attempts),
                )
                if self.peak_valley_ratio < 0.55:
                    self.update_successes(True)
                return (cluster, medoid, points)

            else:  # No more types
                assert False

    def sample_medoid(
        self,
        medoid: int,
    ) -> tuple[_Tensor, _Tensor, float]:
        """Returns:
        - A vector of indices to points within threshold
        - A vector Xf distances to all points
        - The mean distance from medoid to the other points in the first vector
        """
        existing = self.medoid_cache.get(medoid, None)
        if existing is not None:
            return existing

        distances = _calc_distances(self.matrix, medoid)

        if self.cuda:
            within_threshold = (distances <= _MEDOID_RADIUS) & self.kept_mask
            cluster = _torch.nonzero(within_threshold).flatten().cpu()
        else:
            within_threshold = distances.numpy() <= _MEDOID_RADIUS
            cluster = _torch.from_numpy(within_threshold.nonzero()[0])

        closeness = _MEDOID_RADIUS - distances[within_threshold]
        local_density = (self.lengths[within_threshold] * closeness).sum().item()
        result = (cluster, distances, local_density)

        # Make room for a new element in the cache if full, by dropping the oldest entry
        if len(self.medoid_cache) == MAX_CACHED_RESULTS:
            self.medoid_cache.popitem(last=False)
        self.medoid_cache[medoid] = result

        return result


def _smaller_indices(
    tensor: _Tensor, kept_mask: _Tensor, threshold: float, cuda: bool
) -> _Tensor:
    """Get all indices where the tensor is smaller than the threshold."""

    # If it's on GPU, we remove the already clustered points at this step
    # and move to CPU
    if cuda:
        return _torch.nonzero((tensor <= threshold) & kept_mask).flatten().cpu()
    else:
        return _torch.nonzero(tensor <= threshold).flatten()


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
    zeromask = (matrix == 0).all(dim=1)
    matrix[zeromask] = 1 / matrix.shape[1]
    matrix /= matrix.norm(dim=1).reshape(-1, 1) * (2**0.5)
    return matrix


def _calc_distances(matrix: _Tensor, index: int) -> _Tensor:
    "Return vector of cosine distances from rows of normalized matrix to given row."
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0  # avoid float rounding errors
    return dists
