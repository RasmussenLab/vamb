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
from typing import Optional, TypeVar, Union, IO, Sequence
from pathlib import Path
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

    @staticmethod
    def verify_refhash(refhash: bytes, target_refhash: bytes) -> None:
        if refhash != target_refhash:
            raise ValueError(
                f"At least one BAM file reference name hash to {refhash.hex()}, "
                f"expected {target_refhash.hex()}. "
                "Make sure all BAM and FASTA identifiers are identical "
                "and in the same order. "
                "Note that the identifier is the header before any whitespace."
            )

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
        abundance = cls(
            vambtools.validate_input_array(arrs["matrix"]),
            arrs["samplenames"],
            arrs["minid"].item(),
            arrs["refhash"].item(),
        )
        if refhash is not None:
            cls.verify_refhash(abundance.refhash, refhash)

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

        for path in paths:
            if not _os.path.isfile(path):
                raise FileNotFoundError(path)

            if not pycoverm.is_bam_sorted(str(path)):
                raise ValueError(f"Path {path} is not sorted by reference.")

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
                comp_metadata.mask,
            )
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
        mask: _np.ndarray,
    ) -> A:
        _os.mkdir(cache_directory)

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
                mask,
            )
            vambtools.write_npz(filename, matrix)

        # Initialize matrix, the load them chunkwise. Delete the temp files when done
        matrix = _np.empty((mask.sum(), len(paths)), dtype=_np.float32)
        for filename, (chunkstart, chunkstop) in zip(filenames, chunks):
            matrix[:, chunkstart:chunkstop] = vambtools.read_npz(filename)

        shutil.rmtree(cache_directory)

        assert refhash is not None
        return cls(matrix, [str(p) for p in paths], minid, refhash)

    @staticmethod
    def run_pycoverm(
        paths: list[Path],
        minid: float,
        target_refhash: Optional[bytes],
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
                f"CompositionMetaData was created with {len(mask)} sequences, "
                f"but number of refs in BAM files are {len(headers)}."
            )

        headers = [h for (h, m) in zip(headers, mask) if m]
        vambtools.numpy_inplace_maskarray(coverage, mask)
        refhash = vambtools.hash_refnames(headers)

        if target_refhash is not None:
            Abundance.verify_refhash(refhash, target_refhash)

        return (coverage, refhash)
