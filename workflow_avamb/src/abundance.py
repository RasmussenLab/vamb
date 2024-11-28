import argparse
from pathlib import Path
from typing import TextIO
import numpy as np


def main(output: Path, input_dir: Path):
    files = list(input_dir.iterdir())
    files = [i for i in files if i.suffix == ".tsv"]
    names = [i.stem for i in files]
    with open(output, "w") as outfile:
        print("contigname", "\t".join(names), sep="\t", file=outfile)
        write_files(outfile, files)


def write_files(outfile: TextIO, files: list[Path]):
    if len(files) == 0:
        return
    index_of_contig: dict[str, int] = dict()
    names: list[str] = []
    first_file = files.pop()
    abundance: list[float] = []
    with open(first_file) as file:
        for line in file:
            (contigname, ab) = line.split("\t")
            index_of_contig[contigname] = len(index_of_contig)
            names.append(contigname)
            abundance.append(float(ab))
    array = np.empty((len(abundance), len(files) + 1), dtype=np.float32)
    array[:, 0] = abundance
    del abundance

    for col_minus_one, filename in enumerate(files):
        with open(filename) as file:
            for line in file:
                (contigname, ab) = line.split("\t")
                index = index_of_contig[contigname]
                array[index, col_minus_one + 1] = float(ab)

    del index_of_contig
    assert len(names) == len(array)
    for name, row in zip(names, array):
        print(name, "\t".join([f"{x:.6f}" for x in row]), sep="\t", file=outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Abundance TSV output path")
    parser.add_argument(
        "input_dir", help="Dir with single-sample abundance TSV files from strobealign"
    )

    args = parser.parse_args()
    main(Path(args.output), Path(args.input_dir))
