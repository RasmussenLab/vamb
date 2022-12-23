#!/usr/bin/env python

import sys
import os
import argparse
import gzip
import vamb

parser = argparse.ArgumentParser(
    description="""Creates the input FASTA file for Vamb.
Input should be one or more FASTA files, each from a sample-specific assembly.
If keepnames is False, resulting FASTA can be binsplit with separator 'C'.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False,
)

parser.add_argument("outpath", help="Path to output FASTA file")
parser.add_argument("inpaths", help="Paths to input FASTA file(s)", nargs="+")
parser.add_argument(
    "-m",
    dest="minlength",
    metavar="",
    type=int,
    default=2000,
    help="Discard sequences below this length [2000]",
)
parser.add_argument(
    "--keepnames", action="store_true", help="Do not rename sequences [False]"
)
parser.add_argument("--nozip", action="store_true", help="Do not gzip output [False]")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

# Check inputs
for path in args.inpaths:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

if os.path.exists(args.outpath):
    raise FileExistsError(args.outpath)

parent = os.path.dirname(args.outpath)
if not os.path.isdir(parent):
    raise NotADirectoryError(
        f'Output file cannot be created: Parent directory "{parent}" is not an existing directory'
    )

# Run the code. Compressing DNA is easy, this is about 8% bigger than compresslevel 9,
# but ~15x faster.
filehandle = (
    open(args.outpath, "w")
    if args.nozip
    else gzip.open(args.outpath, "wt", compresslevel=3)
)
vamb.vambtools.concatenate_fasta(
    filehandle, args.inpaths, minlength=args.minlength, rename=(not args.keepnames)
)
filehandle.close()
