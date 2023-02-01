import vamb
import argparse
import shutil
import os
import numpy as np
import json


def main(
    cluster_scores,
    cluster_contigs,
    bin_separator,
    path_nc_bins_folder,
    path_bins_folder,
    path_nc_clusters,
    min_comp=0.9,
    max_cont=0.05,
):

    cluster_sample = get_cluster_sample(cluster_contigs, bin_separator)

    nc_cluster_scores = get_nc_cluster_scores(cluster_scores, cluster_sample)

    create_nc_sample_folders(nc_cluster_scores, cluster_sample, path_nc_bins_folder)

    write_nc_bins_from_mdrep_clusters(
        nc_cluster_scores, cluster_sample, path_nc_bins_folder, path_bins_folder
    )

    write_quality_report(nc_cluster_scores, path_nc_bins_folder)

    write_final_nc_clusters(nc_cluster_scores, cluster_contigs, path_nc_clusters)


def get_nc_cluster_scores(cluster_scores, cluster_sample):
    nc_cluster_scores = dict()
    for cluster, scores in cluster_scores.items():
        comp, cont = scores
        comp, cont = float(comp), float(cont)
        comp, cont = comp / 100, cont / 100
        if cluster not in cluster_sample.keys():
            continue
        if comp >= min_comp and cont <= max_cont:
            nc_cluster_scores[cluster] = [comp, cont]

    return nc_cluster_scores


def get_cluster_sample(cluster_contigs, bin_separator):
    cluster_sample = dict()
    for cluster_ in cluster_contigs.keys():
        contigs = cluster_contigs[cluster_]
        contig_i = next(iter(contigs))
        sample = contig_i.split(bin_separator)[0]
        cluster_sample[cluster_] = sample

    return cluster_sample


def create_nc_sample_folders(cluster_scores, cluster_sample, path_nc_bins_folder):
    nc_samples = set()
    for cluster in cluster_scores.keys():

        sample = cluster_sample[cluster]
        nc_samples.add(sample)

    for sample in nc_samples:
        try:
            os.mkdir(os.path.join(path_nc_bins_folder, sample))
        except:
            pass


def write_nc_bins_from_mdrep_clusters(
    cluster_scores, cluster_sample, path_nc_bins_folder, path_bins_folder
):

    for cluster in cluster_scores.keys():
        sample = cluster_sample[cluster]
        src_bin = os.path.join(path_bins_folder, sample, cluster + ".fna")
        trg_bin = os.path.join(path_nc_bins_folder, sample, cluster + ".fna")
        shutil.move(src_bin, trg_bin)


def write_quality_report(cluster_scores, path_nc_bins_folder):

    with open(os.path.join(path_nc_bins_folder, "quality_report.tsv"), "w") as file_:
        print("Name completeness contamination", sep="\t", file=file_)
        file_.flush()
        for nc_cluster, scores in cluster_scores.items():
            print(nc_cluster, scores[0], scores[1], sep="\t", file=file_)
            file_.flush()


def write_final_nc_clusters(cluster_scores, cluster_contigs, path_nc_clusters):

    # nc_cluster_contigs=dict()
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
