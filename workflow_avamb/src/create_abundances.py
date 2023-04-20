import numpy as np
import argparse
import vamb
from pathlib import Path


def create_abundances(
    abundances: list[Path], mask_refhash: Path, min_id: float, outfile: Path
):
    """Merge the abundances to a single Abundance object and save it"""
    refhash = np.load(mask_refhash)["refhash"]

    n_samples = len(abundances)
    first = vamb.vambtools.read_npz(abundances[0])
    print(len(first), n_samples)
    print(first.shape)
    matrix = np.empty((len(first), n_samples), dtype=np.float32)
    matrix[:, 0] = first
    for i, path in enumerate(abundances[1:]):
        matrix[:, i + 1] = vamb.vambtools.read_npz(path)
    abundance = vamb.parsebam.Abundance(
        matrix, [str(i) for i in abundances], min_id, refhash
    )
    abundance.save(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msk", type=Path, help="mask refhash")
    parser.add_argument("--ab", type=Path, nargs="+", help=" abundancaes list of files")
    parser.add_argument("--min_id", type=float, help="min identity for alignment")
    parser.add_argument("--out", type=Path, help="abundances outfile")

    opt = parser.parse_args()

    create_abundances(opt.ab, opt.msk, opt.min_id, opt.out)
