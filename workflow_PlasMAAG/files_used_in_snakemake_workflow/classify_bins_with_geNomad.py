import argparse
import os

import numpy as np
from git_commit import get_git_commit

import vamb


def split_clusters_by_sample(clusters):
    nbhds_split = dict()
    for i, (nbhd_id, nbhd_cs) in enumerate(clusters.items()):
        samples_in_nbhd = set([c.split("C")[0] for c in nbhd_cs])

        nbhd_S_d = {S: set() for S in samples_in_nbhd}

        for c in nbhd_cs:
            nbhd_S_d[c.split("C")[0]].add(c)

        for S, cs in nbhd_S_d.items():
            nbhds_split["%s_%s" % (str(nbhd_id), S)] = cs

    return nbhds_split


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Classify clusters based upon the contig plasmid, virus, and Organism geNomad scores"
    )
    # Add optional arguments with `--flags`
    parser.add_argument(
        "--clusters",
        required=True,
        help="Clusters generated with the community-based clustering algorithm",
    )
    parser.add_argument(
        "--composition",
        required=True,
        help="Composition object containing contignames and contig lengths",
    )
    parser.add_argument(
        "--scores",
        required=True,
        type=str,
        help="Path to the file that contains the contig scores",
    )
    parser.add_argument(
        "--outp",
        required=True,
        help="Plasmid clusters file",
    )

    parser.add_argument(
        "--dflt_cls",
        required=True,
        help="clusters generated with the default clustering algorithm, where plasmid contigs will be removed",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="if set, contig score aggregation is applied per bin",
    )
    parser.add_argument(
        "--thr",
        default=0.7,
        type=float,
        help="Threshold set to classify bins",
    )
    parser.add_argument(
        "--thr_circ",
        default=0.5,
        type=float,
        help="Threshold set to classify circular contigs as plasmids",
    )

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    # Parse the arguments
    args = parser.parse_args()
    ## Load contignames aand lenghts
    composition = np.load(args.composition, allow_pickle=True)
    contignames = composition["identifiers"]
    contiglengths = composition["lengths"]
    c_len_d = {c: l for c, l in zip(contignames, contiglengths)}

    ## Load contig scores
    c_genomad_d = {}
    for row in np.loadtxt(args.scores, skiprows=1, dtype=object):
        contig, org_scr, pla_scr, vir_scr = row
        c_genomad_d[contig] = {
            "org": float(org_scr),
            "pla": float(pla_scr),
            "vir": float(vir_scr),
        }

    ## Load clusters and define a plasmid, virus, and organism score per cluster
    clusters_arr = np.loadtxt(args.clusters, skiprows=1, dtype=object)
    cl_cs_d = {cl: set() for cl in set(clusters_arr[:, 0])}
    for cl, c in clusters_arr:
        cl_cs_d[cl].add(c)

    if args.split:
        print("Splitting clusters per sample")
        cl_cs_d = split_clusters_by_sample(cl_cs_d)

    cl_genomadscores_d = {}
    thr_circ = min(args.thr_circ, args.thr)
    for cl, cs in cl_cs_d.items():
        if cl.endswith("_circ"):
            assert len(cs) == 1
            c = list(cs)[0]
            if c_genomad_d[c]["pla"] >= thr_circ:
                cl_genomadscores_d[cl] = {
                    "org": 0,
                    "pla": 1,
                    "vir": 0,
                }
            else:
                cl_genomadscores_d[cl] = {
                    "org": c_genomad_d[c]["org"],
                    "pla": c_genomad_d[c]["pla"],
                    "vir": c_genomad_d[c]["vir"],
                }

        else:
            denominator = np.sum([c_len_d[c] for c in cs])
            numerator_org = np.sum([c_genomad_d[c]["org"] * c_len_d[c] for c in cs])
            numerator_pla = np.sum([(c_genomad_d[c]["pla"]) * c_len_d[c] for c in cs])
            numerator_vir = np.sum([(c_genomad_d[c]["vir"]) * c_len_d[c] for c in cs])

            cl_genomadscores_d[cl] = {
                "org": numerator_org / denominator,
                "pla": numerator_pla / denominator,
                "vir": numerator_vir / denominator,
            }

    print("%i contigs with genomad scores" % (len(c_genomad_d.keys())))

    # Precompute the mask array for all contigs that meet the threshold
    mask_organism_cluster_contigs = np.zeros(len(c_len_d.keys()), dtype=bool)
    organism_cluster_contigs = set()

    # Filter clusters that pass the threshold
    below_threshold_clusters = {
        cl: cs
        for cl, cs in cl_cs_d.items()
        if cl_genomadscores_d[cl]["pla"] < float(args.thr)
    }

    # Use numpy's isin function to optimize masking in bulk
    all_contigs = np.array(list(c_len_d.keys()))
    below_threshold_contigs = set(
        c for cs in below_threshold_clusters.values() for c in cs
    )
    mask_organism_cluster_contigs = np.isin(all_contigs, list(below_threshold_contigs))

    # Add contigs to the set
    organism_cluster_contigs.update(below_threshold_contigs)

    # Write out clusters above the threshold to the file
    with open(args.outp, "w") as f_pla:
        f_pla.write("clustername\tcontigname\n")
        for cl, cs in cl_cs_d.items():
            if cl not in below_threshold_clusters:
                for c in cs:
                    f_pla.write(f"{cl}\t{c}\n")

    ## Load dflt clusters and extract plasmid contigs
    dfltclusters_arr = np.loadtxt(args.dflt_cls, skiprows=1, dtype=object)
    dfltcl_cs_d = {cl: set() for cl in set(dfltclusters_arr[:, 0])}
    for cl, c in dfltclusters_arr:
        if c in organism_cluster_contigs:
            dfltcl_cs_d[cl].add(c)

    ## save new clusters
    f_nonplasmid_cls = args.dflt_cls.replace(
        ".tsv",
        (
            "_geNomadplasclustercontigs_extracted_thr_%s_thrcirc_%s.tsv"
            % (str(args.thr), str(thr_circ))
        ),
    )
    with open(f_nonplasmid_cls, "w") as f:
        f.write("clustername\tcontigname\n")
        for cl, cs in dfltcl_cs_d.items():
            for c in cs:
                f.write("%s\t%s\n" % (cl, c))

    ## Write geNomad scores per plasmid candidate cluster
    with open(args.outp.replace(".tsv", "_gN_scores.tsv"), "w") as f:
        f.write(
            "%s\t%s\t%s\t%s\n"
            % ("clustername", "chromosome_score", "plasmid_score", "virus_score")
        )
        for cl in cl_cs_d.keys():
            if cl not in below_threshold_clusters:
                if "circ" not in cl:
                    f.write(
                        "%s\t%s\t%s\t%s\n"
                        % (
                            cl,
                            np.round(cl_genomadscores_d[cl]["org"], 3),
                            np.round(cl_genomadscores_d[cl]["pla"], 3),
                            np.round(cl_genomadscores_d[cl]["vir"], 3),
                        )
                    )
                else:
                    c = cl.replace("_circ", "")
                    f.write(
                        "%s\t%s\t%s\t%s\n"
                        % (
                            cl,
                            np.round(c_genomad_d[c]["org"], 3),
                            np.round(c_genomad_d[c]["pla"], 3),
                            np.round(c_genomad_d[c]["vir"], 3),
                        )
                    )

    ## Write geNomad scores per non-plasmid candidate cluster
    # First get the aggregated scores per cls
    dfltcl_genomadscores_d = dict()
    for cl, cs in dfltcl_cs_d.items():
        if len(cs) == 0:
            continue
        denominator = np.sum([c_len_d[c] for c in cs])
        numerator_org = np.sum([c_genomad_d[c]["org"] * c_len_d[c] for c in cs])
        numerator_pla = np.sum([(c_genomad_d[c]["pla"]) * c_len_d[c] for c in cs])
        numerator_vir = np.sum([(c_genomad_d[c]["vir"]) * c_len_d[c] for c in cs])

        dfltcl_genomadscores_d[cl] = {
            "org": numerator_org / denominator,
            "pla": numerator_pla / denominator,
            "vir": numerator_vir / denominator,
        }

    ## Write scores
    with open(f_nonplasmid_cls.replace(".tsv", "_gN_scores.tsv"), "w") as f:
        f.write(
            "%s\t%s\t%s\t%s\n"
            % ("clustername", "chromosome_score", "plasmid_score", "virus_score")
        )
        for cl in dfltcl_cs_d.keys():
            if len(dfltcl_cs_d[cl]) == 0:
                continue
            if "circ" not in cl:
                f.write(
                    "%s\t%s\t%s\t%s\n"
                    % (
                        cl,
                        np.round(dfltcl_genomadscores_d[cl]["org"], 3),
                        np.round(dfltcl_genomadscores_d[cl]["pla"], 3),
                        np.round(dfltcl_genomadscores_d[cl]["vir"], 3),
                    )
                )
            else:
                c = cl.replace("_circ", "")
                f.write(
                    "%s\t%s\t%s\t%s\n"
                    % (
                        cl,
                        np.round(c_genomad_d[c]["org"], 3),
                        np.round(c_genomad_d[c]["pla"], 3),
                        np.round(c_genomad_d[c]["vir"], 3),
                    )
                )
