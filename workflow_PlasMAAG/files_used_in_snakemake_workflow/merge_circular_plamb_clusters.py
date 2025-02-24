import argparse
import os

import numpy as np
from git_commit import get_git_commit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run n2v")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--cls_plamb",
        required=True,
        type=str,
        help="Path to the plamb clusters",
    )

    parser.add_argument(
        "--cls_circular",
        required=True,
        type=str,
        help="Path to the circular clusters",
    )

    parser.add_argument(
        "--outcls",
        required=True,
        type=str,
        help="Path were circular contigs will be dumped",
    )

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print("Git commit hash: " + commit_hash)

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    ## merge clusters
    circular_contigs_arr = np.loadtxt(args.cls_circular, dtype=object)
    if circular_contigs_arr.shape[0] == 1:
        circular_contigs = set()
    else:
        cls_plamb_arr = np.loadtxt(args.cls_plamb, skiprows=1, dtype=object)
        circular_contigs = set(circular_contigs_arr[1:, 1])

    with open(args.outcls, "w") as f:
        f.write("%s\t%s\n" % ("clustername", "contigname"))
        for c in circular_contigs:
            f.write("%s_circ\t%s\n" % (c, c))
        for cl, c in cls_plamb_arr:
            if c in circular_contigs:
                continue
            f.write("%s\t%s\n" % (cl, c))
