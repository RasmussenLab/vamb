import argparse
import os
import pickle

import networkx as nx
import numpy as np
from git_commit import get_git_commit


def graph_from_blastout(
    blastfile,
    min_id,
    min_al,
    restrictive_alignments,
    separator,
):
    """
    Given a blastfile that contains blast output between contigs,
    create a nx graph, where contigs are nodes and edges exist if blast alignment above min_id identity
    over at least min_al bps, [optional] allowing only matches that are restrictive, i.e. do not allow matches where a contig
    spans another contig completely.

    """
    if not restrictive_alignments:
        print("Allowing non restrictive alignments")

    def valid_alignment(c_q_len, c_s_len, q_s, q_e, s_s, s_e):
        """
        Check if alignment is consistent with both contigs belonging to the same original genome
        """

        if q_s > s_s and s_e < (c_q_len - q_s) or s_s > q_s and q_e < (c_s_len - s_s):
            return False

        else:
            return True

    def contig_len(contigname):
        return int(contigname.split("_length_")[1].split("_")[0])

    def merge_ranges(ranges_list):
        # Step 2: Merge overlapping and adjacent ranges
        merged_ranges = []
        current_start, current_end = ranges_list[0]

        for start, end in ranges_list[1:]:
            if start <= current_end:  # Overlapping or adjacent ranges
                current_end = max(current_end, end)
            else:  # No overlap
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end

        # Don't forget to add the last range
        merged_ranges.append((current_start, current_end))

        # Step 3: Calculate the total length of the merged ranges
        total_length = sum(end - start for start, end in merged_ranges)

        return merged_ranges, total_length

    cc_blast_g = nx.Graph()
    for blast_row in np.loadtxt(blastfile, dtype=object, delimiter="\t"):
        ci, cj = blast_row[0], blast_row[1]

        # WARNING LASSE I dont get this assert if they are the same crash?
        # But arent they aligned all against all?
        # Pehaps skip does which are the same ?
        # assert ci != cj:

        if (
            ci.split(separator)[0] == cj.split(separator)[0]
        ):  ## continue if matches from the same sample
            continue

        alig_len = int(blast_row[3])
        identity = float(blast_row[2])

        if alig_len < min_al:
            continue

        if alig_len >= min_al and identity >= min_id:
            ci_s, ci_e, cj_s, cj_e = [int(p) for p in blast_row[6:10]]
            valid_al = valid_alignment(
                contig_len(ci), contig_len(cj), ci_s, ci_e, cj_s, cj_e
            )

            if (ci, cj) not in cc_blast_g.edges():
                al_ranges_q = [(min(ci_s, ci_e), max(ci_s, ci_e))]
                al_ranges_s = [(min(cj_s, cj_e), max(cj_s, cj_e))]
                merged_ranges_q, al_len_q = (
                    al_ranges_q,
                    al_ranges_q[0][1] - al_ranges_q[0][0],
                )
                merged_ranges_s, al_len_s = (
                    al_ranges_s,
                    al_ranges_s[0][1] - al_ranges_s[0][0],
                )

                alig_len = min(al_len_q, al_len_s)

                weight = (
                    identity
                    * (
                        min(alig_len, contig_len(ci), contig_len(cj))
                        / min(contig_len(ci), contig_len(cj))
                    )
                    * (1 / 100)
                )
                cc_blast_g.add_edge(
                    ci,
                    cj,
                    al_len=alig_len,
                    identity=identity,
                    restrictive=valid_al,
                    weight=weight,
                    q=ci,
                    s=cj,
                )

                cc_blast_g[ci][cj]["al_ranges_%s" % ci] = merged_ranges_q
                cc_blast_g[ci][cj]["al_ranges_%s" % cj] = merged_ranges_s

            else:
                if cc_blast_g[ci][cj]["q"] == cj:
                    ci, cj = cj, ci
                    ci_s, ci_e, cj_s, cj_e = cj_s, cj_e, ci_s, ci_e

                al_ranges_q = cc_blast_g[ci][cj]["al_ranges_%s" % ci] + [
                    (min(ci_s, ci_e), max(ci_s, ci_e))
                ]
                al_ranges_s = cc_blast_g[ci][cj]["al_ranges_%s" % cj] + [
                    (min(cj_s, cj_e), max(cj_s, cj_e))
                ]
                al_ranges_q.sort()
                al_ranges_s.sort()

                merged_ranges_q, al_len_q = merge_ranges(al_ranges_q)
                merged_ranges_s, al_len_s = merge_ranges(al_ranges_s)
                alig_len = min(al_len_q, al_len_s)

                extended_alig_len = alig_len - cc_blast_g[ci][cj]["al_len"]

                cc_blast_g[ci][cj]["identity"] = (
                    cc_blast_g[ci][cj]["identity"] * cc_blast_g[ci][cj]["al_len"]
                    + identity * extended_alig_len
                ) / (alig_len)
                cc_blast_g[ci][cj]["al_len"] = alig_len
                weight = (
                    cc_blast_g[ci][cj]["identity"]
                    * (
                        min(
                            cc_blast_g[ci][cj]["al_len"],
                            contig_len(ci),
                            contig_len(cj),
                        )
                        / min(contig_len(ci), contig_len(cj))
                    )
                    * (1 / 100)
                )
                cc_blast_g[ci][cj]["weight"] = weight
                if not cc_blast_g[ci][cj]["restrictive"]:
                    cc_blast_g[ci][cj]["restrictive"] = valid_al

    if restrictive_alignments:
        edges_to_remove = [
            (u, v) for u, v in cc_blast_g.edges() if not cc_blast_g[u][v]["restrictive"]
        ]
        cc_blast_g.remove_edges_from(edges_to_remove)

    return cc_blast_g


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Generate nx graphs from gfa files")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--blastout",
        required=True,
        type=str,
        help="Path to the blsat output file.",
    )
    parser.add_argument(
        "--out",
        dest="outfile",
        required=True,
        type=str,
        help="File where the networkx graph file will be stored.",
    )

    parser.add_argument(
        "--minid",
        type=float,
        help="Min identity allowed.",
    )

    parser.add_argument(
        "--min_al_len",
        type=int,
        help="Min alignment length allowed.",
        default=500,
        required=False,
    )

    parser.add_argument(
        "--non_restrictive",
        action="store_true",
        help="if set, alignment graph will also contain edges from non restrictive alignments.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="C",
        help="Separator that indicates the end of the sample name and the beggining of the contig name [C].",
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    with open(args.outfile, "wb") as pkl:
        pickle.dump(
            graph_from_blastout(
                args.blastout,
                min_id=args.minid,
                min_al=args.min_al_len,
                restrictive_alignments=True if not args.non_restrictive else False,
                separator=args.separator,
            ),
            pkl,
        )
