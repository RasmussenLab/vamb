"""
The following code is a modification of a k-means based reclustering algorithm first published at https://github.com/BigDataBiology/SemiBin
The original code is distributed under MIT License.
"""

import os
import subprocess
import tempfile
import sys
import contextlib
from typing import IO
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import gzip


def log(string: str, logfile: IO[str], indent: int = 0):
    print(("\t" * indent) + string, file=logfile)
    logfile.flush()


def get_best_bin(results_dict, contig_to_marker, namelist, contig_dict, minfasta):
    # There is room for improving the loop below to avoid repeated computation
    # but it runs very fast in any case
    for max_contamination in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        max_F1 = 0
        weight_of_max = 1e9
        max_bin = None
        for res_labels in results_dict.values():
            res = defaultdict(list)
            for label, name in zip(res_labels, namelist):
                if label != -1:
                    res[label].append(name)
            for bin_contig in res.values():
                cur_weight = sum(len(contig_dict[contig]) for contig in bin_contig)
                if cur_weight < minfasta:
                    continue
                marker_list = []
                for contig in bin_contig:
                    marker_list.extend(contig_to_marker[contig])
                if len(marker_list) == 0:
                    continue
                recall = len(set(marker_list)) / 107
                contamination = (len(marker_list) - len(set(marker_list))) / len(
                    marker_list
                )
                if contamination <= max_contamination:
                    F1 = (
                        2
                        * recall
                        * (1 - contamination)
                        / (recall + (1 - contamination))
                    )
                    if F1 > max_F1:
                        max_F1 = F1
                        weight_of_max = cur_weight
                        max_bin = bin_contig
                    elif F1 == max_F1 and cur_weight <= weight_of_max:
                        weight_of_max = cur_weight
                        max_bin = bin_contig
        if max_F1 > 0:  # if there is a bin with F1 > 0
            return max_bin


def fasta_iter(fname, full_header=False):
    """Iterate over a (possibly gzipped) FASTA file

    Parameters
    ----------
    fname : str
        Filename.
            If it ends with .gz, gzip format is assumed
            If .bz2 then bzip2 format is assumed
            if .xz, then lzma format is assumerd
    full_header : boolean (optional)
        If True, yields the full header. Otherwise (the default), only the
        first word

    Yields
    ------
    (h,seq): tuple of (str, str)
    """
    header = None
    chunks = []
    if hasattr(fname, "readline"):

        def op(f, _):
            return f

    elif fname.endswith(".gz"):
        import gzip

        op = gzip.open
    elif fname.endswith(".bz2"):
        import bz2

        op = bz2.open
    elif fname.endswith(".xz"):
        import lzma

        op = lzma.open
    else:
        op = open
    with op(fname, "rt") as f:
        for line in f:
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(chunks)
                line = line[1:].strip()
                if not line:
                    header = ""
                elif full_header:
                    header = line.strip()
                else:
                    header = line.split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


normalize_marker_trans__dict = {
    "TIGR00388": "TIGR00389",
    "TIGR00471": "TIGR00472",
    "TIGR00408": "TIGR00409",
    "TIGR02386": "TIGR02387",
}


def replace_bin_names(data, clusters_labels_map):
    """
    The 'orf' column is formatted as 'bin.contig_id'
    but there is no need to search again if only the bins are different but the contigs are the same.
    This function replaces bin names in the 'orf' column
    with the new bin names from another cluster file given the same contigs.

    TODO: refactor the whole file so there is no need for that

    Input:
    - data: pd.DataFrame
    - clusters_labels_map: dict, {contig: bin}

    Returns a copy of the dataframe with the updated 'orf' column
    """
    data = data.copy()
    data["contig_number"] = data["orf"].str.split(pat=".", n=0, expand=True)[1]
    data["contig_only"] = data["contig_number"].map(lambda x: x.split("_")[0])
    data["old_bin"] = data["orf"].str.split(pat=".", n=0, expand=True)[0]
    data["new_bin"] = data["contig_only"].map(
        lambda x: f"bin{clusters_labels_map[x]:06}"
    )
    data["orf"] = data[["new_bin", "contig_number"]].agg(".".join, axis=1)
    return data


def get_marker(
    hmmout,
    fasta_path=None,
    min_contig_len=None,
    multi_mode=False,
    orf_finder=None,
    contig_to_marker=False,
    clusters_dict=None,
):
    """Parse HMM output file and return markers"""
    data = pd.read_table(
        hmmout,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=(0, 3, 5, 15, 16),
        names=["orf", "gene", "qlen", "qstart", "qend"],
    )
    if clusters_dict is not None:
        data = replace_bin_names(data, clusters_dict)
    if not len(data):
        return []
    data["gene"] = data["gene"].map(lambda m: normalize_marker_trans__dict.get(m, m))
    qlen = data[["gene", "qlen"]].drop_duplicates().set_index("gene")["qlen"]

    def contig_name(ell):
        if orf_finder in ["prodigal", "fast-naive"]:
            contig, _ = ell.rsplit("_", 1)
        else:
            contig, _, _, _ = ell.rsplit("_", 3)
        return contig

    data = data.query("(qend - qstart) / qlen > 0.4").copy()
    data["contig"] = data["orf"].map(contig_name)
    if min_contig_len is not None:
        contig_len = {h.split(".")[1]: len(seq) for h, seq in fasta_iter(fasta_path)}
        data = data[
            data["contig"].map(lambda c: contig_len[c.split(".")[1]] >= min_contig_len)
        ]
    data = data.drop_duplicates(["gene", "contig"])

    if contig_to_marker:
        from collections import defaultdict

        marker = data["gene"].values
        contig = data["contig"].str.split(".").str[-1].values
        sequence2markers = defaultdict(list)
        for m, c in zip(marker, contig):
            sequence2markers[c].append(m)
        return sequence2markers
    else:

        def extract_seeds(vs, sel):
            vs = vs.sort_values()
            median = vs[len(vs) // 2]

            # the original version broke ties by picking the shortest query, so we
            # replicate that here:
            candidates = vs.index[vs == median]
            c = qlen.loc[candidates].idxmin()
            r = list(sel[sel["gene"] == c]["contig"])
            r.sort()
            return r

        if multi_mode:
            data["bin"] = data["orf"].str.split(pat=".", n=0, expand=True)[0]
            counts = data.groupby(["bin", "gene"])["orf"].count()
            res = {}
            for b, vs in counts.groupby(level=0):
                cs = extract_seeds(
                    vs.droplevel(0), data.query("bin == @b", local_dict={"b": b})
                )
                res[b] = [c.split(".", 1)[1] for c in cs]
            return res
        else:
            counts = data.groupby("gene")["orf"].count()
            return extract_seeds(counts, data)


def run_prodigal(fasta_path, num_process, output):
    contigs = {}
    for h, seq in fasta_iter(fasta_path):
        contigs[h] = seq

    total_len = sum(len(s) for s in contigs.values())
    split_len = total_len // num_process

    cur = split_len + 1
    next_ix = 0
    out = None
    with contextlib.ExitStack() as stack:
        for h, seq in contigs.items():
            if cur > split_len and next_ix < num_process:
                if out is not None:
                    out.close()
                out = open(os.path.join(output, "contig_{}.fa".format(next_ix)), "wt")
                out = stack.enter_context(out)

                cur = 0
                next_ix += 1
            out.write(f">{h}\n{seq}\n")
            cur += len(seq)

    try:
        process = []
        for index in range(next_ix):
            with open(
                os.path.join(output, f"contig_{index}_log.txt") + ".out", "w"
            ) as prodigal_out_log:
                p = subprocess.Popen(
                    [
                        "prodigal",
                        "-i",
                        os.path.join(output, f"contig_{index}.fa"),
                        "-p",
                        "meta",
                        "-q",
                        "-m",  # See https://github.com/BigDataBiology/SemiBin/issues/87
                        "-a",
                        os.path.join(output, f"contig_{index}.faa"),
                    ],
                    stdout=prodigal_out_log,
                )
                process.append(p)

        for p in process:
            p.wait()

    except:
        sys.stderr.write("Error: Running prodigal fail\n")
        sys.exit(1)

    contig_output = os.path.join(output, "contigs.faa")
    with open(contig_output, "w") as f:
        for index in range(next_ix):
            f.write(
                open(os.path.join(output, "contig_{}.faa".format(index)), "r").read()
            )
    return contig_output


def cal_num_bins(
    fasta_path,
    binned_length,
    num_process,
    output,
    markers_path=None,
    clusters_dict=None,
    multi_mode=False,
    orf_finder="prodigal",
):
    """Estimate number of bins from a FASTA file

    Parameters
    fasta_path: path
    binned_length: int (minimal contig length)
    num_process: int (number of CPUs to use)
    multi_mode: bool, optional (if True, treat input as resulting from concatenating multiple files)
    """
    if markers_path is None:
        markers_path = os.path.join(output, "markers.hmmout")
    if os.path.exists(markers_path):
        return get_marker(
            markers_path,
            fasta_path,
            binned_length,
            multi_mode,
            orf_finder=orf_finder,
            clusters_dict=clusters_dict,
        )
    else:
        os.makedirs(output, exist_ok=True)
        target_dir = output

    contig_output = run_prodigal(fasta_path, num_process, output)

    hmm_output = os.path.join(target_dir, "markers.hmmout")
    try:
        with open(os.path.join(output, "markers.hmmout.out"), "w") as hmm_out_log:
            subprocess.check_call(
                [
                    "hmmsearch",
                    "--domtblout",
                    hmm_output,
                    "--cut_tc",
                    "--cpu",
                    str(num_process),
                    os.path.split(__file__)[0] + "/marker.hmm",
                    contig_output,
                ],
                stdout=hmm_out_log,
            )
    except:
        if os.path.exists(hmm_output):
            os.remove(hmm_output)
        sys.stderr.write("Error: Running hmmsearch fail\n")
        sys.exit(1)

    return get_marker(
        hmm_output, fasta_path, binned_length, multi_mode, orf_finder=orf_finder
    )


def recluster_bins(
    logfile,
    clusters_path,
    latents_path,
    contigs_path,
    contignames_all,
    minfasta,
    binned_length,
    num_process,
    out_dir,
    random_seed,
    algorithm,
    hmmout_path=None,
    predictions_path=None,
):
    contig_dict = {h: seq for h, seq in fasta_iter(contigs_path)}
    embedding = np.load(latents_path)
    if "arr_0" in embedding:
        embedding = embedding["arr_0"]
    df_clusters = pd.read_csv(clusters_path, delimiter="\t", header=None)
    if (algorithm == "dbscan") and predictions_path:
        df_gt = pd.read_csv(predictions_path)
        column = "predictions"
        df_gt["genus"] = df_gt[column].str.split(";").str[5].fillna("mock")
        species = list(set(df_gt["genus"]))
        species_ind = {s: i for i, s in enumerate(species)}
        clusters_labels_map = {
            k: species_ind[v] for k, v in zip(df_gt["contigs"], df_gt["genus"])
        }
    else:
        clusters_labels_map = {
            k: int(v.split("_")[1]) for k, v in zip(df_clusters[1], df_clusters[0])
        }
    labels_cluster_map = defaultdict(list)
    for k, v in clusters_labels_map.items():
        labels_cluster_map[v].append(k)
    indices_contigs = {c: i for i, c in enumerate(contignames_all)}
    log(f"Latent shape {embedding.shape}", logfile, 1)
    log(f"N contignames for the latent space {len(contignames_all)}", logfile, 1)
    log(f"N contigs in fasta files {len(contig_dict)}", logfile, 1)
    log(f"N contigs in the cluster file {len(clusters_labels_map)}", logfile, 1)

    contig_labels = np.array([clusters_labels_map[c] for c in contignames_all])

    assert len(contig_labels) == embedding.shape[0]

    total_size = defaultdict(int)
    for i, c in enumerate(contig_labels):
        total_size[c] += len(contig_dict[contignames_all[i]])

    cfasta = os.path.join(out_dir, "concatenated.fna")
    with open(cfasta, "wt") as concat_out:
        for ix, h in enumerate(contignames_all):
            bin_ix = contig_labels[ix]
            if total_size[bin_ix] < minfasta:
                continue
            concat_out.write(f">bin{bin_ix:06}.{h}\n")
            concat_out.write(contig_dict[contignames_all[ix]] + "\n")

    log("Starting searching for markers", logfile, 1)

    if hmmout_path is not None:
        log(f"hmmout file is provided at {hmmout_path}", logfile, 2)
        seeds = cal_num_bins(
            cfasta,
            binned_length,
            num_process,
            output=out_dir,
            markers_path=hmmout_path,
            clusters_dict=clusters_labels_map,
            multi_mode=True,
        )
    else:
        seeds = cal_num_bins(
            cfasta,
            binned_length,
            num_process,
            output=out_dir,
            multi_mode=True,
        )
    # we cannot bypass the orf_finder here, because of the renaming of the contigs
    if seeds == []:
        log("No bins found in the concatenated fasta file.", logfile, 1)
        return contig_labels
    log("Finished searching for markers", logfile, 1)
    log(f"Found {len(seeds)} seeds", logfile, 1)

    if algorithm == "dbscan":
        contig2marker = get_marker(
            hmmout_path,
            min_contig_len=binned_length,
            fasta_path=cfasta,
            orf_finder="prodigal",
            clusters_dict=clusters_labels_map,
            contig_to_marker=True,
        )
        log("Running DBSCAN with cosine distance", logfile, 1)
        extracted_all = []
        for k, v in labels_cluster_map.items():
            log(f"Label {k}, {len(v)} contigs", logfile, 1)
            indices = [indices_contigs[c] for c in v]
            contignames = contignames_all[indices]
            embedding_new = embedding[indices]
            DBSCAN_results_dict = {}
            distance_matrix = pairwise_distances(
                embedding_new, embedding_new, metric="cosine"
            )
            length_weight = np.array([len(contig_dict[name]) for name in contignames])
            for eps_value in np.arange(0.01, 0.35, 0.02):
                dbscan = DBSCAN(
                    eps=eps_value,
                    min_samples=5,
                    n_jobs=num_process,
                    metric="precomputed",
                )
                dbscan.fit(distance_matrix, sample_weight=length_weight)
                labels = dbscan.labels_
                log(f"epsilon {eps_value}, {len(set(labels))} labels", logfile, 2)
                DBSCAN_results_dict[eps_value] = labels.tolist()

            log("Integrating results", logfile, 1)

            extracted = []
            contignames_list = list(contignames)
            while (
                sum(len(contig_dict[contig]) for contig in contignames_list) >= minfasta
            ):
                if len(contignames_list) == 1:
                    extracted.append(contignames_list)
                    break

                max_bin = get_best_bin(
                    DBSCAN_results_dict,
                    contig2marker,
                    contignames_list,
                    contig_dict,
                    minfasta,
                )
                if not max_bin:
                    break

                extracted.append(max_bin)
                for temp in max_bin:
                    temp_index = contignames_list.index(temp)
                    contignames_list.pop(temp_index)
                    for eps_value in DBSCAN_results_dict:
                        DBSCAN_results_dict[eps_value].pop(temp_index)
            extracted_all.extend(extracted)

        contig2ix = {}
        log(f"{len(extracted_all)} extracted clusters", logfile, 1)
        for i, cs in enumerate(extracted_all):
            for c in cs:
                contig2ix[c] = i
        namelist = contignames_all.copy()
        contig_labels_reclustered = [contig2ix.get(c, -1) for c in namelist]
    elif algorithm == "kmeans":
        name2ix = {name: ix for ix, name in enumerate(contignames_all)}
        contig_labels_reclustered = np.empty_like(contig_labels)
        contig_labels_reclustered.fill(-1)
        next_label = 0
        for bin_ix in range(contig_labels.max() + 1):
            seed = seeds.get(f"bin{bin_ix:06}", [])
            num_bin = len(seed)

            if num_bin > 1 and total_size[bin_ix] >= minfasta:
                contig_indices = [
                    i for i, ell in enumerate(contig_labels) if ell == bin_ix
                ]
                re_bin_features = embedding[contig_indices]

                seed_index = [name2ix[s] for s in seed]
                length_weight = np.array(
                    [
                        len(contig_dict[name])
                        for name, ell in zip(contignames_all, contig_labels)
                        if ell == bin_ix
                    ]
                )
                seeds_embedding = embedding[seed_index]
                log(f"Starting K-means reclutering for bin {bin_ix}", logfile, 2)
                kmeans = KMeans(
                    n_clusters=num_bin,
                    init=seeds_embedding,
                    n_init=1,
                    random_state=random_seed,
                )
                kmeans.fit(re_bin_features, sample_weight=length_weight)
                log("Finished K-means reclutering", logfile, 2)
                for i, label in enumerate(kmeans.labels_):
                    contig_labels_reclustered[contig_indices[i]] = next_label + label
                next_label += num_bin
            else:
                contig_labels_reclustered[contig_labels == bin_ix] = next_label
                next_label += 1
        assert contig_labels_reclustered.min() >= 0
        os.remove(cfasta)
    else:
        raise AssertionError(
            "Reclustering algorithm must be one of ['dbscan', 'kmeans']"
        )
    return contig_labels_reclustered
