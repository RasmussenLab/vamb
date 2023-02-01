import numpy as np
import os
import json
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, help="path checkm2 that contains all samples")
    parser.add_argument("--b", type=str, help="path all bins ")
    parser.add_argument(
        "--cs_d", type=str, help="cluster_score dictionary will be stored here"
    )
    parser.add_argument(
        "--bp_d", type=str, help="bin_path dictionary will be stored here "
    )

    opt = parser.parse_args()

    bins_set = set()
    for sample in os.listdir(opt.b):
        for bin_ in os.listdir(os.path.join(opt.b, sample)):
            if ".fna" in bin_:
                bins_set.add(bin_)

    cluster_score, bin_path = get_cluster_score_bin_path(opt.s, opt.b, bins_set)
    with open(opt.cs_d, "w") as f:
        json.dump(cluster_score, f)

    with open(opt.bp_d, "w") as f:
        json.dump(bin_path, f)
