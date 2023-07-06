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


def log(string: str, logfile: IO[str], indent: int = 0):
    print(("\t" * indent) + string, file=logfile)
    logfile.flush()


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


def get_marker(
    hmmout,
    fasta_path=None,
    min_contig_len=None,
    multi_mode=False,
    orf_finder=None,
    contig_to_marker=False,
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
        contig_len = {h: len(seq) for h, seq in fasta_iter(fasta_path)}
        data = data[data["contig"].map(lambda c: contig_len[c] >= min_contig_len)]
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
            qlen.loc[candidates].idxmin()
            r = list(sel.query("gene == @c")["contig"])
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
    multi_mode=False,
    output=None,
    orf_finder="prodigal",
):
    """Estimate number of bins from a FASTA file

    Parameters
    fasta_path: path
    binned_length: int (minimal contig length)
    num_process: int (number of CPUs to use)
    multi_mode: bool, optional (if True, treat input as resulting from concatenating multiple files)
    """
    with tempfile.TemporaryDirectory() as tdir:
        if output is not None:
            if os.path.exists(os.path.join(output, "markers.hmmout")):
                return get_marker(
                    os.path.join(output, "markers.hmmout"),
                    fasta_path,
                    binned_length,
                    multi_mode,
                    orf_finder=orf_finder,
                )
            else:
                os.makedirs(output, exist_ok=True)
                target_dir = output
        else:
            target_dir = tdir

        contig_output = run_prodigal(fasta_path, num_process, tdir)

        hmm_output = os.path.join(target_dir, "markers.hmmout")
        try:
            with open(os.path.join(tdir, "markers.hmmout.out"), "w") as hmm_out_log:
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
    random_seed,
):
    contig_dict = {h: seq for h, seq in fasta_iter(contigs_path)}
    embedding = np.load(latents_path)
    df_clusters = pd.read_csv(clusters_path, delimiter="\t", header=None)
    clusters_labels_map = {
        k: int(v.split("_")[1]) for k, v in zip(df_clusters[1], df_clusters[0])
    }
    contignames = list(clusters_labels_map.keys())
    log(f"Latent shape {embedding.shape}", logfile, 1)
    log(f"N contignames for the latent space {len(contignames_all)}", logfile, 1)
    log(f"N contigs in fasta files {len(contig_dict)}", logfile, 1)
    log(f"N contigs in the cluster file {len(clusters_labels_map)}", logfile, 1)
    ind_map = {c: i for i, c in enumerate(contignames_all)}
    indices = [ind_map[c] for c in contignames]

    embedding_new = embedding[indices]
    contig_labels = np.array([clusters_labels_map[c] for c in contignames])

    assert len(contig_labels) == embedding_new.shape[0]

    total_size = defaultdict(int)
    for i, c in enumerate(contig_labels):
        total_size[c] += len(contig_dict[contignames[i]])
    with tempfile.TemporaryDirectory() as tdir:
        cfasta = os.path.join(tdir, "concatenated.fna")
        with open(cfasta, "wt") as concat_out:
            for ix, h in enumerate(contignames):
                bin_ix = contig_labels[ix]
                if total_size[bin_ix] < minfasta:
                    continue
                concat_out.write(f">bin{bin_ix:06}.{h}\n")
                concat_out.write(contig_dict[contignames[ix]] + "\n")

        log("Starting searching for markers", logfile, 1)

        seeds = cal_num_bins(
            cfasta,
            binned_length,
            num_process,
            multi_mode=True,
        )
        # we cannot bypass the orf_finder here, because of the renaming of the contigs
        if seeds == []:
            log("No bins found in the concatenated fasta file.", logfile, 1)
            return contig_labels
        log("Finished searching for markers", logfile, 1)

    name2ix = {name: ix for ix, name in enumerate(contignames)}
    contig_labels_reclustered = np.empty_like(contig_labels)
    contig_labels_reclustered.fill(-1)
    next_label = 0
    for bin_ix in range(contig_labels.max() + 1):
        seed = seeds.get(f"bin{bin_ix:06}", [])
        num_bin = len(seed)

        if num_bin > 1 and total_size[bin_ix] >= minfasta:
            contig_indices = [i for i, ell in enumerate(contig_labels) if ell == bin_ix]
            re_bin_features = embedding_new[contig_indices]

            seed_index = [name2ix[s] for s in seed]
            length_weight = np.array(
                [
                    len(contig_dict[name])
                    for name, ell in zip(contignames, contig_labels)
                    if ell == bin_ix
                ]
            )
            seeds_embedding = embedding_new[seed_index]
            log("Starting K-means reclutering", logfile, 1)
            kmeans = KMeans(
                n_clusters=num_bin,
                init=seeds_embedding,
                n_init=1,
                random_state=random_seed,
            )
            kmeans.fit(re_bin_features, sample_weight=length_weight)
            log("Finished K-means reclutering", logfile, 1)
            for i, label in enumerate(kmeans.labels_):
                contig_labels_reclustered[contig_indices[i]] = next_label + label
            next_label += num_bin
        else:
            contig_labels_reclustered[contig_labels == bin_ix] = next_label
            next_label += 1
    assert contig_labels_reclustered.min() >= 0
    return contig_labels_reclustered
