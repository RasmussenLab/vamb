import numpy as np
import os
from typing import cast


def get_cluster_score_bin_path(
    path_checkm_all: str, path_bins: str, bins: set[str]
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """Given CheckM has been run for all samples, create 2 dictionaries:
    - {bin:path_bin}
    - {bin:[completeness, contamination]}"""
    cluster_score: dict[str, tuple[float, float]] = dict()
    bin_path: dict[str, str] = dict()
    for sample in os.listdir(path_checkm_all):
        path_quality_s = os.path.join(path_checkm_all, sample, "quality_report.tsv")
        c_com_con = np.loadtxt(
            path_quality_s,
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
            dtype=str,
            ndmin=2,
        )

        for row in c_com_con:
            cluster, com, con = row
            cluster = cast(str, cluster)
            com, con = float(com), float(con)
            bin_name = cluster + ".fna"
            if bin_name in bins:
                cluster_score[cluster] = (com, con)
                bin_path[cluster + ".fna"] = os.path.join(
                    path_bins, sample, cluster + ".fna"
                )
    return cluster_score, bin_path


def update_cluster_score_bin_path(
    path_checkm_ripped: str, cluster_score: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    c_com_con = np.loadtxt(
        path_checkm_ripped,
        delimiter="\t",
        skiprows=1,
        usecols=(0, 1, 2),
        dtype=str,
        ndmin=2,
    )
    for row in c_com_con:
        cluster, com, con = row
        if "--" in cluster:
            continue
        com, con = float(com), float(con)
        print(cluster, "scores were", cluster_score[cluster])

        cluster_score[cluster] = (com, con)
        print("and now are", cluster_score[cluster])
    return cluster_score
