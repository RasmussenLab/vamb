__doc__ = """Estimate depths from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = Abundance.from_file(bampaths, metadata, True, 0.1, 3)
"""

import pycoverm
import os as _os
import numpy as _np
from math import isfinite
from vamb.parsecontigs import CompositionMetaData
from vamb import vambtools
from typing import Optional, TypeVar, Union, IO, Sequence, Iterable
from pathlib import Path
from itertools import zip_longest
import shutil

_ncpu = _os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else _ncpu

A = TypeVar("A", bound="Abundance")


class Abundance:
    "Object representing contig abundance. Contains a matrix and refhash."

    __slots__ = ["matrix", "samplenames", "minid", "refhash"]

    def __init__(
        self,
        matrix: _np.ndarray,
        samplenames: Sequence[str],
        minid: float,
        refhash: bytes,
    ):
        assert matrix.dtype == _np.float32
        assert matrix.ndim == 2
        assert matrix.shape[1] == len(samplenames)
        assert isfinite(minid) and 0.0 <= minid and minid <= 1.0

        self.matrix = matrix
        self.samplenames = _np.array(samplenames, dtype=object)
        self.minid = minid
        self.refhash = refhash

    @property
    def nseqs(self) -> int:
        return len(self.matrix)

    @property
    def nsamples(self) -> int:
        return len(self.samplenames)

    def save(self, io: Union[Path, IO[bytes]]):
        _np.savez_compressed(
            io,
            matrix=self.matrix,
            samplenames=self.samplenames,
            minid=self.minid,
            refhash=self.refhash,
        )

    @classmethod
    def load(
        cls: type[A], io: Union[str, Path, IO[bytes]], refhash: Optional[bytes]
    ) -> A:
        arrs = _np.load(io, allow_pickle=True)
        if "arr_0" in arrs.keys():
            return arrs["arr_0"]  # old format
        abundance = cls(
            vambtools.validate_input_array(arrs["matrix"]),
            arrs["samplenames"],
            arrs["minid"].item(),
            arrs["refhash"].item(),
        )
        if refhash is not None:
            vambtools.RefHasher.verify_refhash(
                abundance.refhash,
                refhash,
                "the loaded Abundance object",
                "the given refhash",
                None,
            )

        return abundance

    @classmethod
    def from_files(
        cls: type[A],
        paths: list[Path],
        cache_directory: Optional[Path],
        comp_metadata: CompositionMetaData,
        verify_refhash: bool,
        minid: float,
        nthreads: int,
    ) -> A:
        """Input:
        paths: List of paths to BAM files
        cache_directory: Where to store temp parts of the larger matrix, if reading multiple
           BAM files in chunks. Required if len(paths) > min(16, nthreads)
        comp_metadata: CompositionMetaData of sequence catalogue used to make BAM files
        verify_refhash: Whether to verify composition and BAM references are the same
        minid: Discard any reads with nucleotide identity less than this
        nthreads: Use this number of threads for coverage estimation
        """
        if minid < 0 or minid > 1:
            raise ValueError(f"minid must be between 0 and 1, not {minid}")

        # Workaround: Currently pycoverm has a bug where it filters contigs when mindid == 0
        # (issue #7). Can be solved by setting it to a low value
        minid = minid if minid > 0.001 else 0.001

        if nthreads < 1:
            raise ValueError(f"nthreads must be > 0, not {nthreads}")

        chunksize = min(nthreads, len(paths))

        # We cap it to 16 threads, max. This will prevent pycoverm from consuming a huge amount
        # of memory if given a crapload of threads, and most programs will probably be IO bound
        # when reading 16 files at a time.
        chunksize = min(chunksize, 16)

        # If it can be done in memory, do so
        if chunksize >= len(paths):
            (matrix, refhash) = cls.run_pycoverm(
                paths,
                minid,
                comp_metadata.refhash if verify_refhash else None,
                comp_metadata.identifiers if verify_refhash else None,
                comp_metadata.mask,
            )
            vambtools.mask_lower_bits(matrix, 12)
            return cls(matrix, [str(p) for p in paths], minid, refhash)
        # Else, we load it in chunks, then assemble afterwards
        else:
            if cache_directory is None:
                raise ValueError(
                    "If min(16, nthreads) < len(paths), cache_directory must not be None"
                )
            return cls.chunkwise_loading(
                paths,
                cache_directory,
                chunksize,
                minid,
                comp_metadata.refhash if verify_refhash else None,
                comp_metadata.identifiers if verify_refhash else None,
                comp_metadata.mask,
            )

    @classmethod
    def chunkwise_loading(
        cls: type[A],
        paths: list[Path],
        cache_directory: Path,
        nthreads: int,
        minid: float,
        target_refhash: Optional[bytes],
        target_identifiers: Optional[Iterable[str]],
        mask: _np.ndarray,
    ) -> A:
        _os.makedirs(cache_directory)

        chunks = [
            (i, min(len(paths), i + nthreads)) for i in range(0, len(paths), nthreads)
        ]
        filenames = [
            _os.path.join(cache_directory, str(i) + ".npz") for i in range(len(chunks))
        ]
        assert len(chunks) > 1

        # Load from BAM and store them chunkwise
        refhash = None
        for filename, (chunkstart, chunkstop) in zip(filenames, chunks):
            (matrix, refhash) = cls.run_pycoverm(
                paths[chunkstart:chunkstop],
                minid,
                target_refhash,
                target_identifiers,
                mask,
            )
            vambtools.write_npz(filename, matrix)

        # Initialize matrix, the load them chunkwise. Delete the temp files when done
        matrix = _np.empty((mask.sum(), len(paths)), dtype=_np.float32)
        for filename, (chunkstart, chunkstop) in zip(filenames, chunks):
            matrix[:, chunkstart:chunkstop] = vambtools.read_npz(filename)
        vambtools.mask_lower_bits(matrix, 12)

        shutil.rmtree(cache_directory)

        assert refhash is not None
        return cls(matrix, [str(p) for p in paths], minid, refhash)

    @staticmethod
    def run_pycoverm(
        paths: list[Path],
        minid: float,
        target_refhash: Optional[bytes],
        target_identifiers: Optional[Iterable[str]],
        mask: _np.ndarray,
    ) -> tuple[_np.ndarray, bytes]:
        (headers, coverage) = pycoverm.get_coverages_from_bam(
            [str(p) for p in paths],
            threads=len(paths),
            min_identity=minid,
            # Note: pycoverm's trim_upper=0.1 is same as CoverM trim-upper 90.
            trim_upper=0.1,
            trim_lower=0.1,
        )

        assert coverage.shape == (len(headers), len(paths))

        # Filter length, using comp_metadata's mask, which has been set by minlength
        if len(mask) != len(headers):
            raise ValueError(
                f"CompositionMetaData used to create Abundance object was created with {len(mask)} sequences, "
                f"but number of reference sequences in BAM files are {len(headers)}. "
                "Make sure the BAM files were created by mapping to the same FASTA file "
                "which you used to create the Composition object."
            )

        headers = [h for (h, m) in zip(headers, mask) if m]
        vambtools.numpy_inplace_maskarray(coverage, mask)
        refhash = vambtools.RefHasher.hash_refnames(headers)

        if target_identifiers is None:
            identifier_pairs = None
        else:
            identifier_pairs = (headers, target_identifiers)

        if target_refhash is not None:
            vambtools.RefHasher.verify_refhash(
                refhash, target_refhash, "FASTA file", "BAM", identifier_pairs
            )

        return (coverage, refhash)

    @classmethod
    def from_tsv(cls: type[A], path: Path, comp_metadata: CompositionMetaData) -> A:
        seen_identifiers: list[str] = []
        with open(path) as file:
            try:
                header = next(file)
            except StopIteration:
                err = ValueError(f"Found no TSV header in abundance file '{path}'")
                raise err from None
            columns = header.rstrip("\r\n").split("\t")
            if len(columns) < 2:
                raise ValueError(
                    f'Expected at least 2 columns in abundance TSV file at "{path}"'
                )
            if columns[0] != "contigname":
                raise ValueError('First column in header must be "contigname"')
            samples = columns[1:]
            n_samples = len(samples)
            matrix = _np.empty((comp_metadata.nseqs, n_samples), dtype=_np.float32)
            matrix_row = 0

            # Line number minus two since we already read header, and Python is zero-indexed
            for line_number_minus_two, (line, should_keep) in enumerate(
                zip_longest(file, comp_metadata.mask)
            ):
                if line is None:
                    # If line is none, there are too few lines in file
                    raise ValueError(
                        f'Too few rows in abundance TSV file "{path}", expected '
                        f"{len(comp_metadata.mask) + 1}, got {line_number_minus_two + 1}"
                    )

                line = line.rstrip()

                if not line:
                    for next_line in file:
                        if next_line.rstrip():
                            raise ValueError(
                                "Found an empty line not at end of abundance TSV file"
                                f'"{path}"'
                            )
                    break

                if should_keep is None:
                    raise ValueError(
                        f'Too many rows in abundance TSV file "{path}", expected '
                        f"{len(comp_metadata.mask) + 1} sequences, got at least "
                        f"{line_number_minus_two + 2}"
                    )

                if not should_keep:
                    continue

                fields = line.split("\t")
                if len(fields) != n_samples + 1:
                    raise ValueError(
                        f'In abundance TSV file "{path}", on line {line_number_minus_two + 2}'
                        f", expected {n_samples + 1} columns, found {len(fields)}"
                    )
                for i in range(n_samples):
                    matrix[matrix_row, i] = float(fields[i + 1])
                matrix_row += 1
                seen_identifiers.append(fields[0])

        vambtools.RefHasher.verify_refhash(
            vambtools.RefHasher.hash_refnames(seen_identifiers),
            comp_metadata.refhash,
            "abundance TSV",
            "composition",
            (seen_identifiers, comp_metadata.identifiers),
        )

        return cls(matrix, samples, 0.0, comp_metadata.refhash)
