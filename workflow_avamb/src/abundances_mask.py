import numpy as np
import argparse
from vamb.vambtools import RefHasher
from pathlib import Path


def abundances_mask(headers: Path, mask_refhash: Path, min_contig_size: int):
    """# Using the headers above, compute the mask and the refhash"""

    mask = []
    identifiers = []

    with open(headers) as file:
        for line in file:
            # SN:S27C112075   LN:2239
            (sn, ln) = line.split("\t")
            if sn[:3] != "SN:" or ln[:3] != "LN:":
                raise ValueError("Unknown format")
            passed = int(ln[3:]) >= min_contig_size
            mask.append(passed)
            if passed:
                identifiers.append(sn[3:])

    np.savez_compressed(
        mask_refhash,
        mask=np.array(mask, dtype=bool),
        refhash=RefHasher.hash_refnames(identifiers),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=Path, help=" Headers file")
    parser.add_argument("--msk", type=Path, help="mask refhash")

    parser.add_argument("--minsize", type=int, help="min contig size")

    opt = parser.parse_args()

    abundances_mask(opt.h, opt.msk, opt.minsize)
