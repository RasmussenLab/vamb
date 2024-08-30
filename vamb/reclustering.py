"""
The following code is based on the k-means based reclustering algorithm first published at https://github.com/BigDataBiology/SemiBin
The original code is distributed under MIT License.
"""

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from vamb.taxonomy import Taxonomy
from vamb.parsemarkers import Markers, MarkerID
from vamb.parsecontigs import CompositionMetaData
from vamb.vambtools import RefHasher
from collections.abc import Sequence, Iterable
from typing import NewType, Optional, Union

# TODO: We can get rid of this dep by using the trick from the heapq docs here:
# https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
# However, this might be a little more tricky to implement
from pqdict import pqdict

# We use these aliases to be able to work with integers, which is faster.
ContigId = NewType("ContigId", int)
BinId = NewType("BinId", int)

# TODO: We might want to benchmark the best value for this constant.
# Right now, we do too much duplicated work by clustering 18 times.
EPS_VALUES = np.arange(0.01, 0.35, 0.02)


def deduplicate(
    bins: dict[BinId, set[ContigId]],
    markers: Markers,
) -> list[set[ContigId]]:
    """
        deduplicate(f, bins)

    Given a bin scoring function `f(x: set[ContigId]) -> float`, where better bins have a higher score,
    and `bins: dict[BinId, set[ContigId]]` containing intersecting bins,
    greedily returns the best bin according to the score, removing any contigs from a bin that have
    previously been returned from any other bin.
    Returns `list[tuple[float, set[ContigId]]]` with scored, disjoint bins.
    """
    # This function is a bottleneck, performance wise. Hence, we begin by filtering away some clusters
    # that are trivial.
    contig_sets = remove_duplicate_bins(bins.values())
    scored_contig_sets = remove_badly_contaminated(contig_sets, markers)
    (to_deduplicate, result) = remove_unambigous_bins(scored_contig_sets)
    bins = {BinId(i): b for (i, (b, _)) in enumerate(to_deduplicate)}

    # This is a mutable priority queue. It allows us to greedily take the best cluster,
    # and then update all the clusters that share contigs with the removed best cluster.
    queue = pqdict.maxpq(((BinId(i), s) for (i, (_, s)) in enumerate(to_deduplicate)))

    # When removing the best bin, the contigs in that bin must be removed from all
    # other bins, which causes some bins's score to be invalidated.
    # We store the set of bins that needs to be recomputed in this set.
    to_recompute: set[BinId] = set()

    bins_of_contig: dict[ContigId, set[BinId]] = defaultdict(set)
    for b, c in bins.items():
        for ci in c:
            bins_of_contig[ci].add(b)

    while len(queue) > 0:
        best_bin = queue.pop()
        contigs = bins[best_bin]
        result.append(contigs)

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

        for bin_to_recompute in to_recompute:
            contigs_in_rec = bins.get(bin_to_recompute, None)
            # If the recomputed bin is now empty, we delete it from queue
            if contigs_in_rec is None:
                queue.pop(bin_to_recompute)
            else:
                # We could use the saturating version of the function here,
                # but it's unlikely to make a big difference performance wise,
                # since the truly terrible bins have been filtered away
                counts = count_markers(contigs_in_rec, markers)
                new_score = score_from_comp_cont(get_completeness_contamination(counts))
                queue.updateitem(bin_to_recompute, new_score)

    return result


def remove_duplicate_bins(sets: Iterable[set[ContigId]]) -> list[set[ContigId]]:
    seen_sets: set[frozenset[ContigId]] = set()
    # This is just for computational efficiency, so we don't instantiate frozen sets
    # with single elements.
    seen_singletons: set[ContigId] = set()
    for contig_set in sets:
        if len(contig_set) == 1:
            seen_singletons.add(next(iter(contig_set)))
        else:
            fz = frozenset(contig_set)
            if fz not in seen_sets:
                seen_sets.add(fz)

    result: list[set[ContigId]] = [{s} for s in seen_singletons]
    for st in seen_sets:
        result.append(set(st))
    return result


def remove_badly_contaminated(
    sets: Iterable[set[ContigId]],
    markers: Markers,
) -> list[tuple[set[ContigId], float]]:
    result: list[tuple[set[ContigId], float]] = []
    max_contamination = 1.0
    for contig_set in sets:
        counts = count_markers_saturated(contig_set, markers)
        # None here means that the contamination is already so high we stop counting.
        # this is a shortcut for efficiency
        if counts is None:
            continue
        (completeness, contamination) = get_completeness_contamination(counts)
        if contamination <= max_contamination:
            result.append(
                (contig_set, score_from_comp_cont((completeness, contamination)))
            )
    return result


def remove_unambigous_bins(
    sets: list[tuple[set[ContigId], float]],
) -> tuple[list[tuple[set[ContigId], float]], list[set[ContigId]]]:
    """Remove all bins for which all the contigs are only present in that one bin,
    and put them in the returned list.
    These contigs have a trivial, unambiguous assignment.
    """
    in_single_bin: dict[ContigId, bool] = dict()
    for contig_set, _ in sets:
        for contig in contig_set:
            existing = in_single_bin.get(contig)
            if existing is None:
                in_single_bin[contig] = True
            elif existing is True:
                in_single_bin[contig] = False
    to_deduplicate: list[tuple[set[ContigId], float]] = []
    unambiguous: list[set[ContigId]] = []
    for contig_set, scores in sets:
        if all(in_single_bin[c] for c in contig_set):
            unambiguous.append(contig_set)
        else:
            to_deduplicate.append((contig_set, scores))
    return (to_deduplicate, unambiguous)


class KmeansAlgorithm:
    "Arguments needed specifically when using the KMeans algorithm"

    def __init__(
        self, clusters: list[set[ContigId]], random_seed: int, contiglengths: np.ndarray
    ):
        assert np.issubdtype(contiglengths.dtype, np.integer)
        self.contiglengths = contiglengths
        self.clusters = clusters
        self.random_seed = random_seed


class DBScanAlgorithm:
    "Arguments needed specifically when using the DBScan algorithm"

    def __init__(
        self, comp_metadata: CompositionMetaData, taxonomy: Taxonomy, n_processes: int
    ):
        if not taxonomy.is_canonical:
            raise ValueError(
                "Can only run DBScan on a Taxonomy object with is_canonical set"
            )
        RefHasher.verify_refhash(
            taxonomy.refhash,
            comp_metadata.refhash,
            "taxonomy",
            "composition",
            None,
        )
        self.contiglengths = comp_metadata.lengths
        self.taxonomy = taxonomy
        self.n_processes = n_processes


def recluster_bins(
    markers: Markers,
    latent: np.ndarray,
    algorithm: Union[KmeansAlgorithm, DBScanAlgorithm],
) -> list[set[ContigId]]:
    assert np.issubdtype(algorithm.contiglengths.dtype, np.integer)
    assert np.issubdtype(latent.dtype, np.floating)

    if not (len(algorithm.contiglengths) == markers.n_seqs == len(latent)):
        raise ValueError(
            "Number of elements in contiglengths, markers and latent must match"
        )

    # Simply dispatch to the right implementation based on the algorithm used
    if isinstance(algorithm, KmeansAlgorithm):
        return recluster_kmeans(
            algorithm.clusters,
            latent,
            algorithm.contiglengths,
            markers,
            algorithm.random_seed,
        )
    elif isinstance(algorithm, DBScanAlgorithm):
        assert len(algorithm.taxonomy.contig_taxonomies) == markers.n_seqs
        return recluster_dbscan(
            algorithm.taxonomy,
            latent,
            algorithm.contiglengths,
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
    assert np.issubdtype(contiglengths.dtype, np.integer)
    assert np.issubdtype(latent.dtype, np.floating)
    assert latent.ndim == 2

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

        cluster_indices = np.array(list(cluster))
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
            counts[m] += 1
    return counts


# Same as above, but once we see a very high number of marker genes,
# we bail. This is because a large fraction of time spent in this module
# would otherwise be counting markers of huge clusters, long after we already
# know it's hopelessly contaminated
def count_markers_saturated(
    contigs: Iterable[ContigId],
    markers: Markers,
) -> Optional[np.ndarray]:
    counts = np.zeros(markers.n_markers, dtype=np.int32)
    # This implies contamination >= 2.0. The actual value depends on the unique
    # number of markers, which is too slow to compute in this hot function
    max_markers = 3 * markers.n_markers
    n_markers = 0
    for contig in contigs:
        m = markers.markers[contig]
        if m is not None:
            n_markers += len(m)
            counts[m] += 1
            if n_markers > max_markers:
                return None
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


def get_completeness_contamination(counts: np.ndarray) -> tuple[float, float]:
    n_total = counts.sum()
    n_unique = (counts > 0).sum()
    completeness = n_unique / len(counts)
    contamination = (n_total - n_unique) / len(counts)
    return (completeness, contamination)


# An arbitrary score of a bin, where higher numbers is better.
# completeness - 5 * contamination is used by the CheckM group as a heuristic.
def score_from_comp_cont(comp_cont: tuple[float, float]) -> float:
    return comp_cont[0] - 5 * comp_cont[1]


def recluster_dbscan(
    taxonomy: Taxonomy,
    latent: np.ndarray,
    contiglengths: np.ndarray,
    markers: Markers,
    num_processes: int,
) -> list[set[ContigId]]:
    # Since DBScan is computationally expensive, and scales poorly with the number
    # of contigs, we use taxonomy to only cluster within each genus
    result: list[set[ContigId]] = []
    for indices in group_indices_by_genus(taxonomy):
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
    return deduplicate(bin_dict, markers)


def group_indices_by_genus(
    taxonomy: Taxonomy,
) -> list[np.ndarray]:
    if not taxonomy.is_canonical:
        raise ValueError("Can only group by genus for a canonical taxonomy")
    by_genus: dict[Optional[str], list[ContigId]] = defaultdict(list)
    for i, tax in enumerate(taxonomy.contig_taxonomies):
        genus = None if tax is None else tax.genus
        by_genus[genus].append(ContigId(i))
    return [np.array(i, dtype=np.int32) for i in by_genus.values()]
