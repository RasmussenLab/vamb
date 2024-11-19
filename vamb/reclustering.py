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

# We use these aliases to be able to work with integers, which is faster.
ContigId = NewType("ContigId", int)
BinId = NewType("BinId", int)

# TODO: We might want to benchmark the best value for this constant.
# Right now, we do too much duplicated work by clustering 18 times.
EPS_VALUES = np.arange(0.01, 0.35, 0.02)


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
    n_markers = 0
    n_unique = 0
    # This implies contamination == 1.0
    max_duplicates = 1 * markers.n_markers
    for contig in contigs:
        m = markers.markers[contig]
        if m is not None:
            n_markers += len(m)
            for i in m:
                existing = counts[i]
                n_unique += existing == 0
                counts[i] = existing + 1

            if (n_markers - n_unique) > max_duplicates:
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


def recluster_dbscan(
    taxonomy: Taxonomy,
    latent: np.ndarray,
    contiglengths: np.ndarray,
    markers: Markers,
    num_processes: int,
) -> list[set[ContigId]]:
    # Since DBScan is computationally expensive, and scales poorly with the number
    # of contigs, we use taxonomy to only cluster within each genus
    n_worse_in_row = 0
    genera_indices = group_indices_by_genus(taxonomy)
    best_score = 0
    best_bins: list[set[ContigId]] = []
    for eps in EPS_VALUES:
        bins: list[set[ContigId]] = []
        for indices in genera_indices:
            genus_clusters = dbscan_genus(
                latent[indices], indices, contiglengths[indices], num_processes, eps
            )
            bins.extend(genus_clusters)

        score = count_good_genomes(bins, markers)
        if best_score == 0 or score > best_score:
            best_bins = bins
            best_score = score

        if score >= best_score:
            n_worse_in_row = 0
        else:
            n_worse_in_row += 1
            if n_worse_in_row > 2:
                break

    return best_bins


# DBScan within the subset of contigs that are annotated with a single genus
def dbscan_genus(
    latent_of_genus: np.ndarray,
    original_indices: np.ndarray,
    contiglengths_of_genus: np.ndarray,
    num_processes: int,
    eps: float,
) -> list[set[ContigId]]:
    assert len(latent_of_genus) == len(original_indices) == len(contiglengths_of_genus)
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
    dbscan = DBSCAN(
        eps=eps,
        min_samples=5,
        n_jobs=num_processes,
        metric="precomputed",
    )
    dbscan.fit(distance_matrix, sample_weight=contiglengths_of_genus)
    bins: dict[int, set[ContigId]] = defaultdict(set)
    for original_index, bin_index in zip(original_indices, dbscan.labels_):
        bins[bin_index].add(ContigId(original_index))
    return list(bins.values())


def count_good_genomes(binning: Iterable[Iterable[ContigId]], markers: Markers) -> int:
    max_contamination = 0.3
    min_completeness = 0.75
    result = 0
    for contigs in binning:
        count = count_markers_saturated(contigs, markers)
        if count is None:
            continue
        (comp, cont) = get_completeness_contamination(count)
        if comp >= min_completeness and cont <= max_contamination:
            result += 1

    return result


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
