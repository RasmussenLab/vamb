import os
import numpy as np
import argparse
import vamb
import shutil
from ordered_set import OrderedSet
import json


def main(
    path_run,
    cluster_contigs,
    cluster_scores,
    cluster_not_r_contigs,
    ripped_bins_scores_ar,
    drep_folder,
    bin_path,
    path_bins_ripped,
    bin_separator,
    min_comp,
    max_cont,
):
    # {cluster:sample} dictionary for the manually drep clusters
    cluster_sample = get_cluster_sample(cluster_contigs, bin_separator)
    for sample in set(cluster_sample.values()):
        os.makedirs(os.path.join(path_run, drep_folder, sample), exist_ok=True)

    # Given all the clusters that have no connection (no intersection) with any other cluster
    # mv them to the final bins folder that contains the final set of NC bins if they are NC
    nc_clusters_unchanged = mv_nc_not_r_nc_bins(
        cluster_not_r_contigs,
        cluster_sample,
        cluster_scores,
        drep_folder,
        bin_path,
        min_comp,
        max_cont,
    )

    # Given the checkM2 scores for all ripped bins, create a dictionary
    # {ripped_bin:[comp,cont]} and also the ripped_bins pairs that share the interseciton
    cluster_r_scores, cluster_r_pairs = get_cluster_r_scores(ripped_bins_scores_ar)

    # Now, given the ripped_bins scores and the pairs, for each pair select one ripped and one original
    # per pair, and move them to the final NC folder
    nc_clusters_unchanged_2, nc_clusters_ripped = choose_best_ripped_bin_and_mv_if_nc(
        cluster_r_scores,
        cluster_scores,
        cluster_r_pairs,
        cluster_sample,
        bin_path,
        drep_folder,
        path_run,
        path_bins_ripped,
        min_comp,
        max_cont,
    )

    # for the bins that have been ripped because of meaningless edges and after the ripping present
    # no intersection with any other bin, also move them to the final set of NC bins if
    # they are NC
    nc_clusters_ripped_single = mv_single_ripped_nc_bins(
        ripped_bins_scores_ar,
        drep_folder,
        cluster_r_pairs,
        bin_path,
        cluster_scores,
        path_run,
        cluster_sample,
        min_comp,
        max_cont,
    )

    nc_clusters_unchanged = nc_clusters_unchanged.union(nc_clusters_unchanged_2)
    nc_clusters_ripped = nc_clusters_ripped.union(nc_clusters_ripped_single)
    quality_file = os.path.join(path_run, drep_folder, "quality_report.tsv")
    write_nc_bin_scores(
        cluster_scores,
        cluster_r_scores,
        nc_clusters_unchanged,
        nc_clusters_ripped,
        quality_file,
    )


def get_cluster_sample(cluster_contigs, bin_separator):
    """{cluster:sample} for all clusters"""
    cluster_sample = {}

    for cluster, contigs in cluster_contigs.items():
        contig_i = next(iter(contigs))
        sample = contig_i.split(bin_separator)[0]

        cluster_sample[cluster] = sample
    return cluster_sample


def mv_nc_not_r_nc_bins(
    cluster_not_r_contigs,
    cluster_sample,
    cluster_scores,
    drep_folder,
    bin_path,
    min_comp,
    max_cont,
):
    """mv not ripped NC bins from bins folder to drep folder"""
    nc_clusters_unchanged = set()
    for cluster in cluster_not_r_contigs.keys():
        comp, cont = cluster_scores[cluster]
        if comp >= min_comp and cont <= max_cont:
            # src_bin=os.path.join(path_bins,cluster_sample[cluster],cluster+'.fna')
            src_bin = bin_path[cluster + ".fna"]
            trg_bin = os.path.join(
                path_run, drep_folder, cluster_sample[cluster], cluster + ".fna"
            )
            print(
                "Bin %s has no intersection with any other bin so it is directly moved from %s to %s"
                % (cluster, src_bin, trg_bin)
            )
            shutil.move(src_bin, trg_bin)

            nc_clusters_unchanged.add(cluster)
    return nc_clusters_unchanged


# For ripped bins


def mv_single_ripped_nc_bins(
    ripped_bins_scores_ar,
    drep_folder,
    cluster_r_pairs,
    # path_bins_ripped,
    bin_path,
    cluster_score,
    path_run,
    cluster_sample,
    min_comp,
    max_cont,
):
    """Mv nc bins that were ripped only because they had meaningless edges"""
    nc_clusters_ripped_single = set()
    clusters_alredy_moved = set(
        [element for pair in cluster_r_pairs for element in pair]
    )
    for row in ripped_bins_scores_ar:
        cluster, comp, cont = row[:3]
        if "--" in cluster or cluster in clusters_alredy_moved:
            continue

        comp, cont = float(comp), float(cont)
        comp_, cont_ = cluster_score[cluster]

        assert comp == comp_
        assert cont == cont_

        if comp >= min_comp and cont <= max_cont:
            src_bin = bin_path[cluster + ".fna"]

            if os.path.isfile(src_bin):
                trg_bin = os.path.join(
                    path_run, drep_folder, cluster_sample[cluster], cluster + ".fna"
                )
                print(
                    "Bin %s was ripped because of meaningless edges or pairing and afterwards no intersection was shared with any other bin so it is moved from %s to %s"
                    % (cluster, src_bin, trg_bin)
                )
                shutil.move(src_bin, trg_bin)

                nc_clusters_ripped_single.add(cluster)
    return nc_clusters_ripped_single


def get_cluster_r_scores(ripped_bins_scores_ar):
    """{cluster:[comp,cont]} for ripped clusters"""

    cluster_r_pairs = []
    cluster_ripped_scores_dict = dict()

    for row in ripped_bins_scores_ar:
        bin_A_bin_B_name, comp, cont = row[:3]
        if "--" in bin_A_bin_B_name:
            bin_A_name, bin_B_name = bin_A_bin_B_name.split("--")
            cluster_A_name, cluster_B_name = bin_A_name, bin_B_name

            cluster_ripped_scores_dict[cluster_A_name] = [float(comp), float(cont)]

            cluster_r_pairs.append(OrderedSet([cluster_A_name, cluster_B_name]))
        else:
            cluster_A_name = bin_A_bin_B_name.replace(".fna", "")
            cluster_ripped_scores_dict[cluster_A_name] = [float(comp), float(cont)]

    return cluster_ripped_scores_dict, cluster_r_pairs


def choose_best_ripped_bin_and_mv_if_nc(
    cluster_r_scores,
    cluster_scores,
    cluster_r_pairs,
    cluster_sample,
    bin_path,
    drep_folder,
    path_run,
    path_bins_ripped,
    min_comp,
    max_cont,
):
    nc_clusters_unchanged, nc_clusters_ripped = set(), set()
    clusters_already_processed = set()
    """So given 2 ripped bins that miss the interseciton, we have to choose who keeps the contigs"""
    for cluster_A_r, cluster_B_r in cluster_r_pairs:
        if (
            cluster_A_r in clusters_already_processed
            or cluster_B_r in clusters_already_processed
        ):
            continue
        print("Bin %s and bin %s share some contigs " % (cluster_A_r, cluster_B_r))

        cluster_A_complet, cluster_A_cont = cluster_scores[cluster_A_r]
        cluster_A_r_complet, cluster_A_r_cont = cluster_r_scores[cluster_A_r]
        cluster_A_score = cluster_A_complet - 5 * cluster_A_cont
        cluster_A_r_score = cluster_A_r_complet - 5 * cluster_A_r_cont

        cluster_B_complet, cluster_B_cont = cluster_scores[cluster_B_r]
        cluster_B_r_complet, cluster_B_r_cont = cluster_r_scores[cluster_B_r]
        cluster_B_score = cluster_B_complet - 5 * cluster_B_cont
        cluster_B_r_score = cluster_B_r_complet - 5 * cluster_B_r_cont

        cluster_B_dif = cluster_B_score - cluster_B_r_score
        cluster_A_dif = cluster_A_score - cluster_A_r_score

        keeper = np.argmax([cluster_A_dif, cluster_B_dif])

        if keeper == 0:
            print(
                "%s keeps the intersecting contigs whereas %s does not "
                % (cluster_A_r, cluster_B_r)
            )

            # bin_A keeps the contigs
            if cluster_A_complet >= min_comp and cluster_A_cont <= max_cont:
                bin_A_name = cluster_A_r + ".fna"
                src_bin = bin_path[bin_A_name]
                trg_bin = os.path.join(
                    path_run, drep_folder, cluster_sample[cluster_A_r], bin_A_name
                )
                shutil.move(src_bin, trg_bin)
                nc_clusters_unchanged.add(cluster_A_r)
                print("%s keeps the contigs so src_path is %s " % (bin_A_name, src_bin))
            # and bin B not
            if cluster_B_r_complet >= min_comp and cluster_B_r_cont <= max_cont:
                bin_B_ripped_name = cluster_B_r + "--" + cluster_A_r + ".fna"
                bin_B_name = cluster_B_r + ".fna"
                src_bin = os.path.join(path_bins_ripped, bin_B_ripped_name)
                trg_bin = os.path.join(
                    path_run, drep_folder, cluster_sample[cluster_B_r], bin_B_name
                )
                shutil.move(src_bin, trg_bin)
                nc_clusters_ripped.add(cluster_B_r)
                print(
                    "%s looses the contigs so src_path is %s " % (bin_B_name, src_bin)
                )

        if keeper == 1:
            print(
                "%s keeps the intersecting contigs whereas %s does not "
                % (cluster_B_r, cluster_A_r)
            )

            # bin_B keeps the contigs
            if cluster_B_complet >= min_comp and cluster_B_cont <= max_cont:
                bin_B_name = cluster_B_r + ".fna"
                src_bin = bin_path[bin_B_name]
                trg_bin = os.path.join(
                    path_run, drep_folder, cluster_sample[cluster_B_r], bin_B_name
                )
                shutil.move(src_bin, trg_bin)
                nc_clusters_unchanged.add(cluster_B_r)
                print("%s keeps the contigs so src_path is %s " % (bin_B_name, src_bin))
            # and bin A not
            if cluster_A_r_complet >= min_comp and cluster_A_r_cont <= max_cont:
                bin_A_ripped_name = cluster_A_r + "--" + cluster_B_r + ".fna"
                bin_A_name = cluster_A_r + ".fna"
                src_bin = os.path.join(path_bins_ripped, bin_A_ripped_name)
                trg_bin = os.path.join(
                    path_run, drep_folder, cluster_sample[cluster_A_r], bin_A_name
                )
                shutil.move(src_bin, trg_bin)
                nc_clusters_ripped.add(cluster_A_r)
                print(
                    "%s looses the contigs so src_path is %s " % (bin_A_name, src_bin)
                )

        clusters_already_processed.add(cluster_A_r)
        clusters_already_processed.add(cluster_B_r)

    return nc_clusters_unchanged, nc_clusters_ripped


def write_nc_bin_scores(
    cluster_scores,
    cluster_r_scores,
    nc_clusters_unchanged,
    nc_clusters_ripped,
    quality_file,
):
    """Given that we know which bins were ripped and which not, write a final
    quality_score.tsv file that collect comp and cont for all NC final bins"""
    with open(quality_file, "a") as file_:
        print("Name", "completeness", "contamination", sep="\t", file=file_)
        file_.flush()
        for nc_cluster in nc_clusters_unchanged:
            comp, cont = cluster_scores[nc_cluster]
            print(nc_cluster, comp, cont, sep="\t", file=file_)
            file_.flush()
        for nc_cluster_r in nc_clusters_ripped:
            comp, cont = cluster_r_scores[nc_cluster_r]
            print(nc_cluster_r, comp, cont, sep="\t", file=file_)
            file_.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, help="path run generated by AVAMB")
    parser.add_argument("--c", type=str, help="path clusters file")
    parser.add_argument("--cnr", type=str, help="path dereplicated clusters non ripped")
    parser.add_argument("--sbr", type=str, help="path scores ripped bins ")
    parser.add_argument("--cs_d", type=str, help="cluster_scores dictionary path ")
    parser.add_argument("--bp_d", type=str, help="bin_path dictionary path")
    parser.add_argument("--br", type=str, help="path bins ripped ")
    parser.add_argument(
        "-d", type=str, help="folder name that will contain all nc bins"
    )
    parser.add_argument("--bin_separator", type=str, help="path ripped bins")
    parser.add_argument("--comp", type=float, default=0.9, help="Min completeness ")
    parser.add_argument("--cont", type=float, default=0.05, help="Max contamination ")

    opt = parser.parse_args()

    path_run = opt.r

    path_clusters = opt.c
    with open(path_clusters) as file:
        cluster_contigs = vamb.vambtools.read_clusters(file)

    with open(opt.cs_d) as f:
        cluster_scores = json.load(f)

    path_clusters_not_r = opt.cnr

    with open(path_clusters_not_r) as file:
        cluster_not_r_contigs = vamb.vambtools.read_clusters(file)

    path_bins_ripped_checkm_quality_report = opt.sbr

    ripped_bins_scores_ar = np.loadtxt(
        path_bins_ripped_checkm_quality_report, dtype=str, skiprows=1, ndmin=2
    )

    drep_folder = opt.d

    with open(opt.bp_d) as f:
        bin_path = json.load(f)

    path_bins_ripped = opt.br

    bin_separator = opt.bin_separator

    main(
        path_run,
        cluster_contigs,
        cluster_scores,
        cluster_not_r_contigs,
        ripped_bins_scores_ar,
        drep_folder,
        bin_path,
        path_bins_ripped,
        bin_separator,
        opt.comp * 100,
        opt.cont * 100,
    )
