import numpy as np
import json
import argparse


def update_cluster_score_bin_path(
    path_checkm_ripped: str, cluster_score: dict[str, tuple[float, float]]
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s",
        type=str,
        help="path checkm2 that contains quality_report.tsv file for ripped bins",
    )
    parser.add_argument(
        "--cs_d",
        type=str,
        help="cluster_score dictionary path  ",
    )
    parser.add_argument(
        "--cs_d_o",
        type=str,
        help="cluster_score dictionary path updated, updated with tthe information for clusters that where ripped either becuase of meaningless edges or when making the component lenght <= 2  ",
    )

    opt = parser.parse_args()

    with open(opt.cs_d) as f:
        cluster_score = json.load(f)

    cluster_score_ = update_cluster_score_bin_path(opt.s, cluster_score)

    with open(opt.cs_d_o, "w") as f:
        json.dump(cluster_score_, f)
