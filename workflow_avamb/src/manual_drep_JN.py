import vamb
import numpy as np
import os
import itertools

from typing import NewType, Union, Optional
from collections.abc import Sequence, Mapping, Iterable
from pathlib import Path

import argparse
import json

ContigId = NewType("ContigId", int)
BinId = NewType("BinId", int)


def main(
    # Path to output clusters file. Will error if it already exists
    outpath: Path,
    # Path to names file of contig names. CHANGED
    names: Path,
    # Path to length.npz of contig length. Output by Vamb
    lengths_npz: Path,
    # Path to CheckM2 quality_report.tsv file
    quality_report: dict[str, list],
    # List of paths to clusters.tsv files as output by Vamb.
    # Names of clusters must match those in CheckM2 quality_report,
    # and those of contigs match those in names_npz
    binnings: Sequence[Path],
    # min coverage to consdier 2 bins the same with 100% above this coverage
    min_cov: float,
    # min completentes for bin to be included into the dereplication process
    min_comp: float,
    # max contamination for bin to be included into the dereplication process
    max_cont: float,
    bins_extension: str,
    min_bin_size: int,
) -> None:
    # Load contig names
    contig_names: list[str] = list(np.loadtxt(names, dtype=object))

    assert isinstance(contig_names, list)
    assert isinstance(contig_names[0], str)

    # Load lengths
    lengths: np.ndarray = vamb.vambtools.read_npz(lengths_npz)
    assert len(lengths) == len(contig_names)

    # Load CheckM2
    (bin_names, qualities, bin_by_name) = load_checkm2(
        quality_report, min_comp, max_cont, bins_extension
    )
    # Load bins
    (bin_lengths, union_bins) = load_binnings(
        binnings, contig_names, lengths, bin_by_name, min_bin_size
    )
    del bin_by_name

    dereplicated = dereplicate(union_bins, qualities, lengths, bin_lengths, min_cov)
    del bin_lengths

    if os.path.exists(outpath):
        raise FileExistsError(outpath)

    with open(outpath, "w") as file:
        for bin in dereplicated:
            bin_name = bin_names[bin]
            bin_name = bin_name.replace(".fna", "")
            for contig in union_bins[bin]:
                print(bin_name, contig_names[contig], sep="\t", file=file)


def load_checkm2(
    quality_report: dict[str, list],
    min_completeness: float,
    max_contamination: float,
    bins_extension: str,
) -> tuple[
    list[str],  # Bin names
    list[tuple[float, float]],  # Bin qualities
    dict[str, Optional[BinId]],  # Mapping to binid, if not skipped
]:
    """Extract all bin names and assign them either a BinId, or else None,
    if their completeness/contamination is so bad the bin should be discarded
    """
    # This is None if the bin is to be discarded
    bin_by_name: dict[str, Optional[BinId]] = dict()
    bin_names: list[str] = []
    qualities: list[tuple[float, float]] = []

    # The file looks like this:
    # Name    Completeness    Contamination   Completeness_Model_Used Translation_Table_Used  Additional_Notes
    # AAE_UC_Y_1980340Ccluster_501--AAE_UC_v3_1980340Ccluster_2599    5.18    0.0     Neural Network (Specific Model) 11      None

    for cluster, scores in quality_report.items():
        name = cluster + bins_extension
        comp, cont = scores
        completeness = float(comp) / 100
        contamination = float(cont) / 100
        assert 0.0 <= completeness <= 1.0
        assert 0.0 <= contamination  # can be unbounded

        if completeness >= min_completeness and contamination <= max_contamination:
            bin = BinId(len(bin_names))
            bin_names.append(name)
            qualities.append((completeness, contamination))
            bin_by_name[name] = bin
        else:
            bin_by_name[name] = None

    assert sum(1 for i in bin_by_name.values() if isinstance(i, int)) == len(bin_names)
    return (bin_names, qualities, bin_by_name)


def load_binnings(
    binnings: Sequence[Path],
    contig_names: Sequence[str],
    lengths: np.ndarray,
    bin_by_name: Mapping[str, Optional[BinId]],
    min_bin_size: int,
) -> tuple[list[int], list[set[ContigId]]]:
    """
    Load clusters.tsv files from each binning, and filter away those assigned to be discarded based on CheckM2 data.
    Return bin length and bins, each represented as a set of ContigId
    """
    id_len_of_contig_name: dict[str, tuple[ContigId, int]] = dict()
    for index, (name, length) in enumerate(zip(contig_names, lengths)):
        id_len_of_contig_name[name] = (ContigId(index), length)

    # Load binnings
    n_union_bins = sum(1 for i in bin_by_name.values() if i is not None)

    lengthof = dict(zip(contig_names, lengths))

    union_bins: list[Optional[set[ContigId]]] = [None] * n_union_bins
    for binning_path in binnings:
        with open(binning_path) as file:
            clusters = vamb.vambtools.read_clusters(file)
            clusters_filtered = filterclusters(clusters, lengthof, min_bin_size)
            # filter by clusters larger than 200kbs
            for bin_name, contigs in clusters_filtered.items():
                bin_name += ".fna"
                # None is a valid value, so we use -1 as sentinel for missing
                bin = bin_by_name.get(bin_name, -1)
                if bin == -1:
                    raise ValueError(
                        f"Bin {bin_name} found in binning {binning_path}, but is not scored by CheckM2"
                    )
                # Means: Below threshold, so skip it
                elif bin is None:
                    continue
                else:
                    ids: set[ContigId] = set()
                    for contig in contigs:
                        existing = id_len_of_contig_name.get(contig)
                        if existing is None:
                            raise KeyError(
                                f"Cluster file {binning_path} contain contig {contig}, "
                                "but that name is not present in provided names npz file"
                            )
                        ids.add(existing[0])
                    union_bins[bin] = ids

    bin_lengths: list[int] = []

    for i in union_bins:
        assert isinstance(i, set)
    union_bins_asserted: list[set[ContigId]] = union_bins  # type: ignore

    for contigs in union_bins_asserted:
        bin_lengths.append(sum(lengths[contig] for contig in contigs))

    return (bin_lengths, union_bins_asserted)


def filterclusters(
    clusters: Mapping[str, set], lengthof: Mapping[str, int], min_bin_size: int
) -> Mapping[str, set]:
    filtered_bins = dict()
    for medoid, contigs in clusters.items():
        binsize = sum(lengthof[contig] for contig in contigs)

        if binsize >= min_bin_size:
            filtered_bins[medoid] = contigs

    return filtered_bins


def dereplicate(
    union_bins: Sequence[set[ContigId]],
    qualities: Sequence[tuple[float, float]],
    contig_lengths: np.ndarray,
    bin_lengths: Sequence[int],
    threshold: float,
) -> list[BinId]:
    "Removes bins if they are too similar to another bin. Return list of kept bins"
    assert len(union_bins) == len(qualities) == len(bin_lengths)

    overlapping_pairs = get_overlapping_bin_pairs(get_binsof(union_bins), qualities)
    to_remove = compute_to_remove(
        union_bins, overlapping_pairs, contig_lengths, bin_lengths, threshold
    )
    return [BinId(i) for i in range(len(bin_lengths)) if BinId(i) not in to_remove]


def get_binsof(union_bins: Iterable[Iterable[ContigId]]) -> dict[ContigId, list[BinId]]:
    "Makes a dict from contig -> list of bins the contig is present in, if in multiple bins"
    binsof: dict[ContigId, Union[BinId, list[BinId]]] = dict()
    for bin_int, contigs in enumerate(union_bins):
        bin = BinId(bin_int)
        for contig in contigs:
            existing = binsof.get(contig)
            if existing is None:
                binsof[contig] = bin
            elif isinstance(existing, int):
                binsof[contig] = [existing, bin]
            else:
                assert isinstance(existing, list)
                existing.append(bin)
    return {k: v for (k, v) in binsof.items() if isinstance(v, list)}


def bin_score(completeness: float, contamination: float) -> float:
    return completeness - 5 * contamination


def get_overlapping_bin_pairs(
    binsof: Mapping[ContigId, list[BinId]], qualities: Sequence[tuple[float, float]]
) -> Sequence[tuple[BinId, BinId]]:
    "Get a list of pairs of bins that share at least one contig"
    pairs: set[tuple[BinId, BinId]] = set()
    for overlapping_bins in binsof.values():
        for a, b in itertools.combinations(overlapping_bins, r=2):
            # Order them so we don't have (a, b) and (b, a) as distinct pairs
            if a > b:
                (a, b) = (b, a)
            pairs.add((a, b))

    # Now be sure to order them as (worst, best) depending on score
    # If they tie, then use lexographic order (a, b) we added them
    # in above
    result: list[tuple[BinId, BinId]] = []
    for a, b in pairs:
        score_a = bin_score(*qualities[a])
        score_b = bin_score(*qualities[b])
        if score_a > score_b:
            result.append((b, a))
        else:
            result.append((a, b))

    return result


def compute_to_remove(
    union_bins: Sequence[set[ContigId]],
    overlapping_pairs: Iterable[tuple[BinId, BinId]],
    lengths: np.ndarray,
    bin_lengths: Sequence[int],
    threshold: float,
) -> set[BinId]:
    "Create a list of bins to remove because they overlap with another bin"
    result: set[BinId] = set()
    for bin_a, bin_b in overlapping_pairs:
        if bin_a in result or bin_b in result:
            continue

        intersection = union_bins[bin_a] & union_bins[bin_b]
        int_len = sum(lengths[i] for i in intersection)
        if int_len / min(bin_lengths[bin_a], bin_lengths[bin_b]) >= threshold:
            # We remove an arbitrary one
            result.add(bin_a)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs_d", type=str, help="path bins_scores dictionary")
    parser.add_argument("--names", type=str, help="Contig names txt file path ")
    parser.add_argument("--lengths", type=str, help="Contig lengths npz file path ")
    parser.add_argument(
        "--output",
        type=str,
        help="Path output clusters generated by dereplicating bins",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        nargs="*",
        help="Path input clusters generated by aamb and vamb",
    )
    parser.add_argument("--cov", type=float, default=0.75, help="Min coverage ")
    parser.add_argument("--comp", type=float, default=0.9, help="Min completeness ")
    parser.add_argument("--cont", type=float, default=0.05, help="Max contamination ")
    parser.add_argument(
        "--bins_extension", type=str, default=".fna", help="Extension of the bins  "
    )
    parser.add_argument(
        "--min_bin_size",
        type=int,
        help="Min bin length to be considered for dereplication ",
    )

    opt = parser.parse_args()
    args = vars(parser.parse_args())
    with open(opt.cs_d) as f:
        cluster_scores = json.load(f)

    main(
        outpath=opt.output,
        names=opt.names,
        lengths_npz=opt.lengths,
        quality_report=cluster_scores,
        binnings=opt.clusters,
        min_cov=opt.cov,
        min_comp=opt.comp,
        max_cont=opt.cont,
        bins_extension=opt.bins_extension,
        min_bin_size=opt.min_bin_size,
    )
