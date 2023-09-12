import numpy as np
import argparse
import vamb
from pathlib import Path


def write_abundances(
    mask_refhash: Path, bampath: Path, min_identity: float, outfile: Path
):
    """For every sample, compute the abundances given the mask and refhashes"""
    loadnpz = np.load(mask_refhash)
    refhash = loadnpz["refhash"]
    mask = loadnpz["mask"]
    refhash = refhash.reshape(1)[0]
    (abundance, _) = vamb.parsebam.Abundance.run_pycoverm(
        paths=[bampath],
        minid=min_identity,
        target_refhash=refhash,
        target_identifiers=None,
        mask=mask,
    )
    vamb.vambtools.write_npz(outfile, abundance.ravel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msk", type=Path, help="mask refhash")
    parser.add_argument("--b", type=Path, help=" bam path")
    parser.add_argument("--min_id", type=float, help="min identity for alignment")
    parser.add_argument("--out", type=Path, help="abundances outfile")

    opt = parser.parse_args()

    write_abundances(opt.msk, opt.b, opt.min_id, opt.out)
