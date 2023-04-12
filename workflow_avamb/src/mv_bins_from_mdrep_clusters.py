import vamb
import argparse
import shutil
import os
import json

from typing import Optional


def main(
    cluster_scores: dict[str, tuple[float, float]],
    cluster_contigs: dict[str, set[str]],
    bin_separator: Optional[str],
    path_nc_bins_folder: str,
    path_bins_folder: str,
    path_nc_clusters: str,
    min_comp: float = 0.9,
    max_cont: float = 0.05,
):
    cluster_sample = get_cluster_sample(cluster_contigs, bin_separator)
    nc_cluster_scores = get_nc_cluster_scores(
        cluster_scores, cluster_sample, min_comp, max_cont
    )
    create_nc_sample_folders(nc_cluster_scores, cluster_sample, path_nc_bins_folder)
    write_nc_bins_from_mdrep_clusters(
        nc_cluster_scores, cluster_sample, path_nc_bins_folder, path_bins_folder
    )
    write_quality_report(nc_cluster_scores, path_nc_bins_folder)
    write_final_nc_clusters(nc_cluster_scores, cluster_contigs, path_nc_clusters)


def get_nc_cluster_scores(
    cluster_scores: dict[str, tuple[float, float]],
    cluster_sample: dict[str, str],
    min_comp: float,
    max_cont: float,
) -> dict[str, tuple[float, float]]:
    nc_cluster_scores: dict[str, tuple[float, float]] = dict()
    for cluster, scores in cluster_scores.items():
        comp, cont = scores
        comp, cont = float(comp), float(cont)
        comp, cont = comp / 100, cont / 100
        if cluster not in cluster_sample.keys():
            continue
        if comp >= min_comp and cont <= max_cont:
            nc_cluster_scores[cluster] = (comp, cont)

    return nc_cluster_scores


def get_cluster_sample(
    cluster_contigs: dict[str, set[str]], bin_separator: Optional[str]
) -> dict[str, str]:
    cluster_sample: dict[str, str] = dict()
    for cluster_ in cluster_contigs.keys():
        contigs = cluster_contigs[cluster_]
        contig_i = next(iter(contigs))
        sample = contig_i.split(bin_separator)[0]
        cluster_sample[cluster_] = sample

    return cluster_sample


def create_nc_sample_folders(
    cluster_scores: dict[str, tuple[float, float]],
    cluster_sample: dict[str, str],
    path_nc_bins_folder: str,
):
    nc_samples: set[str] = set()
    for cluster in cluster_scores.keys():
        sample = cluster_sample[cluster]
        nc_samples.add(sample)

    for sample in nc_samples:
        try:
            os.mkdir(os.path.join(path_nc_bins_folder, sample))
        except FileExistsError:
            pass


def write_nc_bins_from_mdrep_clusters(
    cluster_scores: dict[str, tuple[float, float]],
    cluster_sample: dict[str, str],
    path_nc_bins_folder: str,
    path_bins_folder: str,
):
    for cluster in cluster_scores.keys():
        sample = cluster_sample[cluster]
        src_bin = os.path.join(path_bins_folder, sample, cluster + ".fna")
        trg_bin = os.path.join(path_nc_bins_folder, sample, cluster + ".fna")
        shutil.move(src_bin, trg_bin)


def write_quality_report(
    cluster_scores: dict[str, tuple[float, float]], path_nc_bins_folder: str
):
    with open(os.path.join(path_nc_bins_folder, "quality_report.tsv"), "w") as file:
        print("Name completeness contamination", sep="\t", file=file)
        file.flush()
        for nc_cluster, (completeness, contaminaton) in cluster_scores.items():
            print(nc_cluster, completeness, contaminaton, sep="\t", file=file)
            file.flush()


def write_final_nc_clusters(
    cluster_scores: dict[str, tuple[float, float]],
    cluster_contigs: dict[str, set[str]],
    path_nc_clusters: str,
):
    with open(path_nc_clusters, "w") as file:
        for nc_cluster in cluster_scores.keys():
            nc_contigs = cluster_contigs[nc_cluster]
            for nc_contig in nc_contigs:
                print(nc_cluster, nc_contig, sep="\t", file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, help="path clusters file from tmp folder")
    parser.add_argument("--cf", type=str, help="path clusters file final")
    parser.add_argument("--cs_d", type=str, help="cluster_scores dictionary path ")
    parser.add_argument("--b", type=str, help="path all bins ")
    parser.add_argument(
        "--d", type=str, help="path to folder that will contain all nc bins"
    )
    parser.add_argument("--bin_separator", type=str, help="separator ")
    parser.add_argument("--comp", type=float, default=0.9, help="Min completeness ")
    parser.add_argument("--cont", type=float, default=0.05, help="Max contamination ")

    opt = parser.parse_args()

    with open(opt.c) as clusters_file:
        cluster_contigs = vamb.vambtools.read_clusters(clusters_file)

    with open(opt.cs_d) as f:
        cluster_scores = json.load(f)

    main(
        cluster_scores,
        cluster_contigs,
        opt.bin_separator,
        opt.d,
        opt.b,
        opt.cf,
        opt.comp,
        opt.cont,
    )
