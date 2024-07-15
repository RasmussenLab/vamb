"""
The following code is based on the k-means based reclustering algorithm first published at https://github.com/BigDataBiology/SemiBin
The original code is distributed under MIT License.
"""

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from vamb.vambtools import ContigTaxonomy
from vamb.parsemarkers import Markers, MarkerID
from collections.abc import Sequence, Iterable
from typing import Callable, NewType, Optional
import heapq

# We use these aliases to be able to work with integers, which is faster.
ContigId = NewType("ContigId", int)
BinId = NewType("BinId", int)

EPS_VALUES = np.arange(0.01, 0.35, 0.02)


def deduplicate(
    scoring: Callable[[set[ContigId]], float],
    bins: dict[BinId, set[ContigId]],
) -> list[tuple[float, set[ContigId]]]:
    """
        deduplicate(f, bins)

    Given a bin scoring function `f(x: set[ContigId]) -> float`, where better bins have a higher score,
    and `bins: dict[BinId, set[ContigId]]` containing intersecting bins,
    greedily returns the best bin according to the score, removing any contigs from a bin that have
    previously been returned from any other bin.
    Returns `list[tuple[float, set[ContigId]]]` with scored, disjoint bins.
    """
    # We continuously mutate `bins`, so we need to make a copy here, so the caller
    # doesn't have `bins` mutated inadvertently
    bins = {b: s.copy() for (b, s) in bins.items()}

    result: list[tuple[float, set[ContigId]]] = []
    # Use a heap, such that `heappop` will return the best bin
    # (Heaps in Python are min-heaps, so we use the negative score.)
    heap: list[tuple[float, BinId]] = [(-scoring(c), b) for (b, c) in bins.items()]
    heapq.heapify(heap)

    # When removing the best bin, the contigs in that bin must be removed from all
    # other bins, which causes some bins's score to be invalidated.
    # We store the set of bins that needs to be recomputed in this set.
    to_recompute: set[BinId] = set()

    bins_of_contig: dict[ContigId, set[BinId]] = defaultdict(set)
    for b, c in bins.items():
        for ci in c:
            bins_of_contig[ci].add(b)

    while len(heap) > 0:
        (neg_score, best_bin) = heapq.heappop(heap)
        contigs = bins[best_bin]
        result.append((-neg_score, contigs))

        to_recompute.clear()
        for contig in contigs:
            # Delete `contig` from `bins_of_contig`, since we never need to
            # refer to this entry in the dict again
            bins_of_this_contig = bins_of_contig.pop(contig)
            for bin in bins_of_this_contig:
                if bin != best_bin:
                    # Remove this contig from the other bin
                    other_bin_contigs = bins[bin]
                    other_bin_contigs.remove(contig)
                    # If the other bin is now empty, we remove the whole bin.
                    if len(other_bin_contigs) == 0:
                        del bins[bin]
                    to_recompute.add(bin)

            # Remove the bin we picked as the best
            del bins[best_bin]

        # Check this here to skip recomputing scores and re-heapifying, since that
        # takes time.
        if len(to_recompute) > 0:
            heap = [(s, b) for (s, b) in heap if b not in to_recompute]
            for bin in to_recompute:
                # We could potentially have added some bins in `to_recompute` which have had
                # all their members removed.
                # These empty bins should be discarded
                c = bins.get(bin, None)
                if c is not None:
                    heap.append((-scoring(c), bin))
            heapq.heapify(heap)

    return result


class KmeansAlgorithm:
    "Arguments needed specifically when using the KMeans algorithm"

    def __init__(self, clusters: list[set[ContigId]], random_seed: int):
        self.clusters = clusters
        self.random_seed = random_seed


class DBScanAlgorithm:
    "Arguments needed specifically when using the DBScan algorithm"

    def __init__(self, taxonomy: list[Optional[ContigTaxonomy]], n_processes: int):
        self.taxonomy = taxonomy
        self.n_processes = n_processes


def recluster_bins(
    markers: Markers,
    contiglengths: np.ndarray,
    latent: np.ndarray,
    algorithm: KmeansAlgorithm | DBScanAlgorithm,
):
    assert len(contiglengths) == markers.n_seqs == len(latent)

    # Simply dispatch to the right implementation based on the algorithm used
    if isinstance(algorithm, KmeansAlgorithm):
        return recluster_kmeans(
            algorithm.clusters,
            latent,
            contiglengths,
            markers,
            algorithm.random_seed,
        )
    elif isinstance(algorithm, DBScanAlgorithm):
        assert len(algorithm.taxonomy) == markers.n_seqs
        return recluster_dbscan(
            algorithm.taxonomy,
            latent,
            contiglengths,
            markers,
            algorithm.n_processes,
        )


def recluster_kmeans(
    clusters: list[set[ContigId]],
    latent: np.ndarray,
    contiglengths: np.ndarray,
    markers: Markers,
    random_seed: int,
) -> list[set[ContigId]]:
    assert len(latent) == len(contiglengths) == markers.n_seqs

    result: list[set[ContigId]] = []
    indices_by_medoid: dict[int, set[ContigId]] = defaultdict(set)
    # We loop over all existing clusters, and determine if they should be split,
    # by looking at the median number of single-copy genes in the cluster
    for cluster in clusters:
        # All clusters with 1 contig by definition cannot have multiple single-copy
        # genes (because SCGs are deduplicated within a single contig)
        if len(cluster) == 1:
            result.append(cluster)
            continue
        # Get a count of each marker and compute the median count of SCGs
        counts = count_markers(cluster, markers)
        cp = counts.copy()
        cp.sort()
        median_counts: int = cp[len(cp) // 2]
        # If we have less than 2 SCGs on average, the cluster should not be split,
        # and we emit it unchanged
        if median_counts < 2:
            result.append(cluster)
            continue

        # Run K-means with the median number of SCGs to split the contig.
        # We weigh the contigs by length.
        seeds = get_kmeans_seeds(
            cluster,
            markers,
            contiglengths,  # type: ignore
            counts,
            median_counts,
        )

        cluster_indices = np.ndarray(list(cluster))
        cluter_latent = latent[cluster_indices]
        cluster_lengths = contiglengths[cluster_indices]
        seed_latent = latent[seeds]
        kmeans = KMeans(
            n_clusters=median_counts,
            init=seed_latent,
            n_init=1,
            random_state=random_seed,
        )
        kmeans.fit(cluter_latent, sample_weight=cluster_lengths)
        indices_by_medoid.clear()
        for cluster_label, index in zip(kmeans.labels_, cluster_indices):
            indices_by_medoid[cluster_label].add(ContigId(index))
        result.extend(indices_by_medoid.values())

    return result


# Get a vector of counts, of each SCG,
# where if MarkerID(5) is seen 9 times, then counts[5] == 9.
def count_markers(
    contigs: Iterable[ContigId],
    markers: Markers,
) -> np.ndarray:
    counts = np.zeros(markers.n_markers, dtype=np.int32)
    for contig in contigs:
        m = markers.markers[contig]
        if m is not None:
            for i in m:
                counts[i] += 1
    return counts


# This is not very effectively implemented, but I assume it does not matter.
# This function looks at all markers that occur exactly `median` times, each of these
# markers corresponding to a list of `median` number of contigs.
# It picks the marker for which the smallest contig that contains it is largest.
# The idea here is that long contigs, which contain one of the SCGs that exist exactly
# `median` times are most likely to be close to the actual medoid
# that Kmeans needs to find.
# This is just one possible seeding strategy. We could also plausibly choose e.g.
# the trio of contigs that have the most SCGs.
def get_kmeans_seeds(
    contigs: Iterable[ContigId],
    markers: Markers,
    contiglengths: Sequence[int],
    counts: np.ndarray,
    median: int,
) -> list[ContigId]:
    considered_markers = {MarkerID(i) for (i, c) in enumerate(counts) if c == median}
    contigs_of_markers: dict[MarkerID, list[ContigId]] = defaultdict(list)
    for contig in contigs:
        m = markers.markers[contig]
        if m is None:
            continue
        for mid in m:
            if mid not in considered_markers:
                continue
            contigs_of_markers[MarkerID(mid)].append(contig)

    candidate_list = list(contigs_of_markers.items())
    pair = max(candidate_list, key=lambda x: min(contiglengths[i] for i in x[1]))
    result = pair[1]
    assert len(result) == median
    return result


# An arbitrary score of a bin, where higher numbers is better.
# completeness - 5 * contamination is used by the CheckM group as a heuristic.
def score_bin(bin: set[ContigId], markers: Markers) -> float:
    counts = count_markers(bin, markers)
    n_total = counts.sum()
    n_unique = (counts > 0).sum()
    completeness = n_unique / len(counts)
    contamination = (n_total - n_unique) / len(counts)
    return completeness - 5 * contamination


def recluster_dbscan(
    taxonomy: list[Optional[ContigTaxonomy]],
    latent: np.ndarray,
    contiglengths: np.ndarray,
    markers: Markers,
    num_processes: int,
) -> list[set[ContigId]]:
    # Since DBScan is computationally expensive, and scales poorly with the number
    # of contigs, we use taxonomy to only cluster within each genus
    indices_by_genus = group_indices_by_genus(taxonomy)
    result: list[set[ContigId]] = []
    for indices in indices_by_genus.values():
        genus_latent = latent[indices]
        genus_clusters = dbscan_genus(
            genus_latent, indices, contiglengths[indices], markers, num_processes
        )
        result.extend(genus_clusters)

    return result


# DBScan within the subset of contigs that are annotated with a single genus
def dbscan_genus(
    latent_of_genus: np.ndarray,
    original_indices: np.ndarray,
    contiglengths_of_genus: np.ndarray,
    markers: Markers,
    num_processes: int,
) -> list[set[ContigId]]:
    assert len(latent_of_genus) == len(original_indices) == len(contiglengths_of_genus)
    redundant_bins: list[set[ContigId]] = []
    bins_this_iteration: dict[int, set[ContigId]] = defaultdict(set)
    # Precompute distance matrix. This is O(N^2), but DBScan is even worse,
    # so this pays off.
    # TODO: Maybe we should emit a warning if this function is called with too
    # many points such that this matrix becomes huge?
    distance_matrix = pairwise_distances(
        latent_of_genus, latent_of_genus, metric="cosine"
    )
    # The DBScan approach works by blindly clustering with different eps values
    # (a critical parameter for DBscan), and then using SCGs to select the best
    # subset of clusters.
    # It's ugly and wasteful, but it does work.
    for eps in EPS_VALUES:
        dbscan = DBSCAN(
            eps=eps,
            min_samples=5,
            n_jobs=num_processes,
            metric="precomputed",
        )
        dbscan.fit(distance_matrix, sample_weight=contiglengths_of_genus)
        bins_this_iteration.clear()
        for original_index, bin_index in zip(original_indices, dbscan.labels_):
            bins_this_iteration[bin_index].add(ContigId(original_index))
        redundant_bins.extend(bins_this_iteration.values())

    # Now we deduplicate the redundant bins, such that each contig is only
    # present in one output bin, using the scoring function to greedily
    # output the best of the redundant bins.
    bin_dict = {BinId(i): c for (i, c) in enumerate(redundant_bins)}
    scored_deduplicated = deduplicate(lambda x: score_bin(x, markers), bin_dict)
    return [b for (_, b) in scored_deduplicated]


def group_indices_by_genus(
    taxonomy: list[Optional[ContigTaxonomy]],
) -> dict[str, np.ndarray]:
    by_genus: dict[str, list[ContigId]] = defaultdict(list)
    for i, tax in enumerate(taxonomy):
        genus = None if tax is None else tax.genus
        # TODO: Should we also cluster the ones with unknown genus?
        # Currently, we just skip it here
        if genus is not None:
            by_genus[genus].append(ContigId(i))
    return {g: np.ndarray(i, dtype=np.int32) for (g, i) in by_genus.items()}
