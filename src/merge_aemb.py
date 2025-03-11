import argparse
import numpy as np
import sys
from math import isinf, isnan
from pathlib import Path

parser = argparse.ArgumentParser(
    description="""Merge output files of `strobealign --aemb` to a single abundance TSV file.
The sample names will be the basenames of the paths in the input directory.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=True,
)

parser.add_argument("input_dir", help="Path to directory of --aemb output files")
parser.add_argument(
    "output_file", help="Path to write output TSV file (must not exist)"
)

args = parser.parse_args()


def exit_with(message: str):
    print(message, file=sys.stderr)
    exit(1)


# Check input directory exists
input = Path(args.input_dir)
output = Path(args.output_file)

if not input.is_dir():
    exit_with(f"Error: Input is not an existing directory: '{input}'")

# Check output file's parent is an existing directory.
if not output.parent.is_dir():
    exit_with(
        f"Error: Output file cannot be created: Parent directory '{output.parent}' is not an existing directory"
    )

if output.exists():
    exit_with(f"Error: Output file already exists: '{output}'")

files = sorted(input.iterdir())

for file in files:
    for char in ("\n", "\r", "\t", "\v"):
        if char in file.name:
            exit_with(
                f"Error: File name '{file.name}' contains a char {repr(char)}, which is not permitted in Vamb"
            )


def exit_on_line(path: Path, line: int, message: str):
    exit_with(f"Error: {message}, in file '{path}' on line {line}")


# We allow an empty directory, but let's warn the user since it's an easy mistake
# to make.
if len(files) == 0:
    # N.B: We don't use exitwith here, because we want the exit code to be 0
    # indicating this is not an error.
    print("Warning: No files in input directory", sys.stderr)
    exit(0)


# Parses an --aemb file, yielding (identifier, depth), where depth is a non-negative,
# non-inf, non-nan float.
def parse_lines(path: Path):
    with open(path) as file:
        for lineno_minus_one, line in enumerate(file):
            line = line.rstrip()

            # If line is empty or whitespace-only, it must be the last line.
            # --aemb does not produce trailing whitespace
            if not line:
                for next_line in file:
                    if next_line.rstrip():
                        exit_on_line(
                            path, lineno_minus_one + 1, "Found non-trailing empty line"
                        )
                return

            fields = line.split("\t")

            # Currently --aemb only outputs two columns, but they document explicitly
            # that they may add other columns in the future
            if len(fields) < 2:
                exit_on_line(
                    path, lineno_minus_one + 1, "Not at least two tab-separated columns"
                )

            (identifier, depth_str) = (fields[0], fields[1])
            try:
                depth = float(depth_str)
            except ValueError:
                exit_on_line(
                    path, lineno_minus_one + 1, "Depth cannot be parsed as float"
                )
            except:
                raise

            if isnan(depth) or isinf(depth) or depth < 0.0:
                exit_on_line(
                    path, lineno_minus_one + 1, "Depth is negative, NaN or infinite"
                )

            yield (identifier, depth)


# We allow the order of rows to differ between the files, so we need to be able
# to convert an identifier into a row index for subsequent files
identifier_to_index: dict[str, int] = dict()

# We store depths in a matrix, but we need to have parsed the first file to know
# how big to make the matrix
first_depths: list[float] = []
identifiers: list[str] = []
for identifier, depth in parse_lines(files[0]):
    length = len(identifier_to_index)
    identifier_to_index[identifier] = length
    # If the identifier has previously been seen, the dict entry will be overwritten
    if len(identifier_to_index) == length:
        exit_with(
            f"Duplicate sequence name found in file '{files[0]}': '{identifier}'",
        )
    first_depths.append(depth)
    identifiers.append(identifier)

# Initialize with -1, so we can search for it at the end and make sure no entries
# are uninitialized
matrix = np.full((len(identifiers), len(files)), -1.0, dtype=np.float32)
matrix[:, 0] = first_depths

del first_depths

# Fill in the rest of the files
for col_minus_one, file in enumerate(files[1:]):
    n_seen_identifiers = 0
    for identifier, depth in parse_lines(file):
        n_seen_identifiers += 1
        index = identifier_to_index.get(identifier)

        # Ensure all entries in this file have a known index (i.e. are also
        # in the first file)
        if index is None:
            exit_with(
                f"Error: Identifier '{identifier}' found in file '{file}' "
                "but not present in all files.",
            )

        if matrix[index, col_minus_one + 1] != -1.0:
            exit_with(
                f"Error: Identifier '{identifier}' present multiple times in file '{file}'"
            )

        matrix[index, col_minus_one + 1] = depth

    # Check that this file does not have a strict subset of identifiers from the first
    # file. After that, we know the set of identifiers is exactly the same
    if n_seen_identifiers != len(identifiers):
        exit_with(
            f"Error: File '{file}' does not have all identifiers of file '{files[0]}'."
        )

assert -1.0 not in matrix, (
    "Matrix not full; this is a bug in the script and should never happen"
)

with open(output, "w") as file:
    # We already checked this, but let's check it again
    assert len(matrix) == len(identifiers)
    print("contigname", "\t".join([p.name for p in files]), sep="\t", file=file)
    for identifier, row in zip(identifiers, matrix):
        print(identifier, "\t".join([str(i) for i in row]), sep="\t", file=file)
