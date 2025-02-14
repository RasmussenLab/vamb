import argparse
import os
import pickle

# from scipy.spatial.distance import cdist
import time

import networkx as nx
import numpy as np
import torch
from git_commit import get_git_commit

import vamb


# define functions and classess
def find_neighbours_optimized(
    embeddings_bincontigs,
    contignames,
    idxs_without_embeddings,
    c_cc_d,
    radius_clustering=0.2,
):
    """ """
    radius = radius_clustering / 2

    communities_g = nx.Graph()
    embeddings_bincontigs_nz = vamb.cluster._normalize(embeddings_bincontigs)
    neighs = []

    ## For each contig embeddings_bincontigs, find the contigs that are within a distance, and link them
    for i in range(len(embeddings_bincontigs)):
        neighs.append([])
        if i in idxs_without_embeddings:
            continue

        distances = vamb.cluster._calc_distances(embeddings_bincontigs_nz, i)
        ## can I use the triangle inequality to optimize this, so it does not compute all against all for every contig?s
        mask_ = torch.ones(len(embeddings_bincontigs), dtype=torch.bool)
        mask_[i] = False
        within_radius = (distances <= radius) & mask_

        if torch.sum(within_radius) == 0:  # in such case it will not have a neighbour
            continue
        else:
            for lat_neigh_idx in np.where(within_radius)[0]:
                if c_cc_d[contignames[lat_neigh_idx]] == c_cc_d[contignames[i]]:
                    communities_g.add_edge(contignames[i], contignames[lat_neigh_idx])
                    neighs[i].append(lat_neigh_idx)

    return (np.array(neighs, dtype=object), communities_g)


def split_neighs_per_sample(neighs, contignames):
    neighs_sorted_same_sample = []
    mask_cs_with_neighs = np.zeros(len(contignames), dtype=bool)
    neighs_removed_n = 0
    neighs_kept = 0
    for i, nn_idxs in enumerate(neighs):
        if len(nn_idxs) == 0:
            neighs_sorted_same_sample.append([])
            continue

        c_i = contignames[i]
        c_i_sample = c_i.split("C")[0]
        neighs_i_same_sample_idxs = []
        for neigh_idx in nn_idxs:
            if contignames[neigh_idx].split("C")[0] == c_i_sample:
                neighs_i_same_sample_idxs.append(neigh_idx)
                neighs_kept += 1
            else:
                neighs_removed_n += 1

        neighs_sorted_same_sample.append(neighs_i_same_sample_idxs)
        if len(neighs_i_same_sample_idxs) > 0:
            mask_cs_with_neighs[i] = True
    print(
        "%i/%i neighbour connections have been removed since neighs are part of different samples."
        % (neighs_removed_n, neighs_kept + neighs_removed_n)
    )
    print(np.where(mask_cs_with_neighs)[0])
    return (
        np.array(neighs_sorted_same_sample, dtype=object),
        np.where(mask_cs_with_neighs)[0],
    )


def get_neighbourhoods(neighs_obj, contignames, min_neighs=2):
    neighbourhoods_g = nx.Graph()
    for i, neigh_idxs in enumerate(neighs_obj):
        c = contignames[i]
        for neigh_idx in neigh_idxs:
            c_neigh = contignames[neigh_idx]
            if c_neigh == c:
                continue
            neighbourhoods_g.add_edge(c_neigh, c)

    hood_cs_d = {
        i: cc
        for i, cc in enumerate(nx.connected_components(neighbourhoods_g))
        if len(cc) >= min_neighs
    }

    return (
        hood_cs_d,
        neighbourhoods_g,
    )


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="From the embeddings, the list of contigs that have been embedded, and the binning contigs, generate a neighbours file, a contigs with neighbours mask, and a dictionary with neighbours and hoods."
    )
    # Add arguments
    parser.add_argument("--embs", help="Embeddings file.")
    parser.add_argument("--contigs_embs", help="Contigs embedded file.")
    parser.add_argument("--neighs_outdir", help="Neighbours outtdir.")
    parser.add_argument("-g", help="Graph files that was used as input for n2v")
    parser.add_argument(
        "--contignames",
        default=None,
        help="File containing the binning contignames, if not provided it will generate neighs with all contigs.",
    )
    parser.add_argument(
        "-r",
        default=[0.05],
        help="Radius within 2 contigs are considered neighbours.",
        nargs="+",
    )

    ## Taking the n2v embeddings from either the assembly graph or the alignment graph. Extract communities of contigs from them.
    # 1. Sort and mask so neighbours are generated according the order defined by the contignames file content
    # 2. Compute cosine distances all against all, only for considered contigs
    # 3. For each contig, define as neighbours the ones that are within a radius
    # 4. For each community of connected contigs, split community if belong to different graph componentnts

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    ## Parse arguments
    args = parser.parse_args()
    try:
        os.mkdir(args.neighs_outdir)
    except:
        pass

    with open(os.path.join(args.neighs_outdir, "log.txt"), "w") as f:
        f.write("%s\n" % (args))

    ## Load files
    # Load graph file
    with open(args.g, "rb") as pkl_file:
        graph = pickle.load(pkl_file)
    # Define which component each contig belongs to
    ccs_graph_d = {i: cc for i, cc in enumerate(nx.connected_components(graph))}
    c_cc_d = {c: i for i, cs in ccs_graph_d.items() for c in cs}

    # Define radiuses
    radiuses = [float(r) for r in args.r]

    # Load contignames
    if args.contignames != None:
        ## Load the names of the contigs that will be used for binning BE CAREFUL SINCE HE CONTIGNAMES USUALLY USED ONLY APPLUES FOR MIN CONTIG LEN 2000
        contignames = np.loadtxt(args.contignames, dtype=object)

    else:
        contignames = np.loadtxt(args.contigs_embs, dtype=object)

    print("len of contignames %i" % (len(contignames)))
    c_idx_d = {c: i for i, c in enumerate(contignames)}

    # Load the embeddings, and the contigs that are represented in those embeddings
    contigsembs = np.loadtxt(args.contigs_embs, dtype=object)
    embeddings = np.load(args.embs)["arr_0"]
    contig_emb_d = {c: e for c, e in zip(contigsembs, embeddings)}

    ## 1. Mask the embeddings and process them so they match the binning contigs
    embeddings_bincontigs = np.zeros((len(contignames), embeddings.shape[1]))
    embeddings_mask = np.ones(len(contignames), dtype=bool)
    for i, c in enumerate(contignames):
        if c not in contig_emb_d.keys():
            embeddings_mask[i] = False
            continue
        embeddings_bincontigs[c_idx_d[c], :] = contig_emb_d[c]

    fraction_embedded_contigs = np.sum(embeddings_mask) / len(contignames)
    print("Fraction of contigs with embeddings %.3f" % (fraction_embedded_contigs))

    ## 2. Compute cosine distances all against all, only for considered AND contigs and contigs taht are embedded
    t0 = time.time()

    ## 3. For each contig, define as neighbours the ones that are within a radius
    embs_d = dict()
    for radius in radiuses:
        print("Finding neighbours within radius %.3f" % radius)
        embs_d[radius] = dict()
        embs_d[radius]["neighs"] = find_neighbours_optimized(
            embeddings_bincontigs,
            contignames,
            np.where(embeddings_mask == False)[0],
            c_cc_d,
            radius,
        )

        ## extract also neighs split by sample
        embs_d[radius]["neighs_split_by_sample"] = split_neighs_per_sample(
            embs_d[radius]["neighs"][0], contignames
        )
        ## Define community of neighbours if a path of contigs within raidus can be walked
        # embs_d[radius]["nhbds"], embs_d[radius]["nhbds_g"], hoods_split_bool = (
        embs_d[radius]["nhbds"], embs_d[radius]["nhbds_g"] = get_neighbourhoods(
            embs_d[radius]["neighs"][0], contignames
        )

        c_hood_d = {c: hood for hood, cs in embs_d[radius]["nhbds"].items() for c in cs}

        ## also save neighs where hoods are removed if they are composed of only neighs from diff samples, i.e. hoood : {S1C1, S2C2}
        hoods_to_remove = set()
        for hood, cs in embs_d[radius]["nhbds"].items():
            samples_hood = set([c.split("C")[0] for c in cs])
            if len(samples_hood) == len(cs):
                hoods_to_remove.add(hood)
        contigs_to_clear_neighs = set(
            [c for h in hoods_to_remove for c in embs_d[radius]["nhbds"][h]]
        )
        print(
            "%i hoods with %i contigs will be removed since they contain only contigs from dif samples"
            % (len(hoods_to_remove), len(contigs_to_clear_neighs))
        )

        embs_d[radius]["neighs_cleared"] = find_neighbours_optimized(
            embeddings_bincontigs,
            contignames,
            set(np.where(embeddings_mask == False)[0]).union(
                set([c_idx_d[c] for c in contigs_to_clear_neighs])
            ),
            c_cc_d,
            radius,
        )

        embs_d[radius]["nhbds_cleared"], embs_d[radius]["nhbds_cleared_g"] = (
            get_neighbourhoods(embs_d[radius]["neighs_cleared"][0], contignames)
        )

        t1 = time.time()

        print(
            "Neighbours and neighbourhoods computed in %.2f minutes" % ((t1 - t0) / 60)
        )

        if args.contignames == None:
            print(
                "Generating only hoods files, not neighbours.npz since contignames were not provided"
            )
        else:
            print("Generating hoods, neighbours and mask neighbours files")

        # neighs, _ = embs_d[radius]["neighs"]

        contigs_w_neighs = np.sum(
            [1 for row in embs_d[radius]["neighs"][0] if len(row) > 0]
        )
        print("%i contigs with neighs" % contigs_w_neighs)

        if args.contignames != None:
            neighs_file = os.path.join(
                args.neighs_outdir, "neighs_object_r_%s.npz" % (str(radius))
            )
            np.savez_compressed(
                neighs_file,
                embs_d[radius]["neighs"][0],
            )
            print(f"Neighs saved in {neighs_file}")

            ## also save neighs where only intra edges hoods are removed
            neighs_file = os.path.join(
                args.neighs_outdir,
                "neighs_intraonly_rm_object_r_%s.npz" % (str(radius)),
            )
            np.savez_compressed(
                neighs_file,
                embs_d[radius]["neighs_cleared"][0],
            )
            print(
                f"Neighs where only intra edges hoods are removed by sample saved in {neighs_file}"
            )

            ## also save neighs split by sample
            neighs_file = os.path.join(
                args.neighs_outdir, "neighs_split_object_r_%s.npz" % (str(radius))
            )
            np.savez_compressed(
                neighs_file,
                embs_d[radius]["neighs_split_by_sample"][0],
            )
            print(f"Neighs split by sample saved in {neighs_file}")

        hoods_clusters_path = os.path.join(
            args.neighs_outdir, "hoods_clusters_r_%s.tsv" % (str(radius))
        )
        with open(hoods_clusters_path, "w") as f:
            f.write("clustername\tcontigname\n")
            for hood_i, cs in embs_d[radius]["nhbds"].items():
                for c in cs:
                    f.write("%s\t%s\n" % (hood_i, c))

        hoods_wointraonly_clusters_path = os.path.join(
            args.neighs_outdir, "hoods_intraonly_rm_clusters_r_%s.tsv" % (str(radius))
        )
        ## also save cleared hoods where only intra edges hoods are removed
        hoods_wointraonly_clusters_path = os.path.join(
            args.neighs_outdir, "hoods_intraonly_rm_clusters_r_%s.tsv" % (str(radius))
        )

        with open(hoods_wointraonly_clusters_path, "w") as f:
            f.write("clustername\tcontigname\n")
            for hood_i, cs in embs_d[radius]["nhbds_cleared"].items():
                for c in cs:
                    f.write("%s\t%s\n" % (hood_i, c))

        print(
            f"Hoods where only intra edges hoods are removed by sample clusters saved in {hoods_clusters_path}"
        )

        contignames_not_in_hoods = set(contignames) - set(
            [c for cs in embs_d[radius]["nhbds"].values() for c in cs]
        )

        hoods_clusters_complete_path = os.path.join(
            args.neighs_outdir, "hoods_clusters_r_%s_complete.tsv" % (str(radius))
        )
        with open(hoods_clusters_complete_path, "w") as f:
            f.write("clustername\tcontigname\n")
            for hood_i, cs in embs_d[radius]["nhbds"].items():
                for c in cs:
                    f.write("%s\t%s\n" % (hood_i, c))
            for i, c in enumerate(contignames_not_in_hoods):
                f.write("%i_\t%s\n" % (i, c))
        print(f"Hoods clusters complete saved in {hoods_clusters_complete_path}")

    # Save the dictionary to a Pickle file
    # path_dict = os.path.join(embs_dir, "tmp/embs_d_%s.pkl"%(date))
    path_dict = os.path.join("%s/embs_d.pkl" % args.neighs_outdir)

    with open(path_dict, "wb") as pickle_file:
        pickle.dump(embs_d, pickle_file)

    print(f"Dictionary saved to {path_dict}")
