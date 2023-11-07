__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import os as _os
import numpy as _np
import vamb.vambtools as _vambtools
from collections.abc import Iterable, Sequence
from typing import IO, Union, TypeVar, Optional
from pathlib import Path
import warnings

# This kernel is created in src/create_kernel.py. See that file for explanation
_KERNEL: _np.ndarray = _vambtools.read_npz(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernel.npz")
)


class CompositionMetaData:
    """A class containing metadata of sequence composition.
    Current fields are:
    * identifiers: A Numpy array of objects, str identifiers of kept sequences
    * lengths: A Numpy vector of 32-bit uint lengths of kept sequences
    * mask: A boolean Numpy vector of which sequences were kept in original file
    * refhash: A bytes object representing the hash of the identifiers
    * minlength: The minimum contig length used for filtering
    """

    __slots__ = ["identifiers", "lengths", "mask", "refhash", "minlength"]

    def __init__(
        self,
        identifiers: _np.ndarray,
        lengths: _np.ndarray,
        mask: _np.ndarray,
        minlength: int,
    ):
        assert len(identifiers) == len(lengths)
        assert identifiers.dtype == _np.dtype("O")
        assert _np.issubdtype(lengths.dtype, _np.integer)
        assert mask.dtype == bool
        assert mask.sum() == len(lengths)
        assert lengths.min(initial=minlength) >= minlength

        if len(set(identifiers)) < len(identifiers):
            raise ValueError(
                "Sequence names must be unique, but are not. "
                "Vamb only uses the identifier (e.g. header before whitespace) as "
                "sequence identifiers. Verify identifier uniqueness."
            )

        self.identifiers = identifiers
        self.lengths = lengths
        self.mask = mask
        self.minlength = minlength
        self.refhash = _vambtools.RefHasher.hash_refnames(identifiers)

    @property
    def nseqs(self) -> int:
        "Number of sequences after filtering"
        return len(self.identifiers)

    def filter_mask(self, mask: Sequence[bool]):
        "Filter contigs given a mask whose length should be nseqs"
        assert len(mask) == self.nseqs
        ind = 0
        for i in range(len(self.mask)):
            if self.mask[i]:
                self.mask[i] &= mask[ind]
                ind += 1

        self.identifiers = self.identifiers[mask]
        self.lengths = self.lengths[mask]
        self.refhash = _vambtools.RefHasher.hash_refnames(self.identifiers)

    def filter_min_length(self, length: int):
        "Set or reset minlength of this object"
        if length <= self.minlength:
            return None

        self.filter_mask(self.lengths >= length)
        self.minlength = length


C = TypeVar("C", bound="Composition")


class Composition:
    """A class containing a CompositionMetaData and its TNF matrix.
    Current fields are:
    * metadata: A CompositionMetaData object
    * matrix: The composition matrix itself
    """

    __slots__ = ["metadata", "matrix"]

    def __init__(self, metadata: CompositionMetaData, matrix: _np.ndarray):
        assert matrix.dtype == _np.float32
        assert matrix.shape == (metadata.nseqs, 103)

        self.metadata = metadata
        self.matrix = matrix

    def count_bases(self) -> int:
        return self.metadata.lengths.sum()

    @property
    def nseqs(self) -> int:
        return self.metadata.nseqs

    def save(self, io: Union[str, Path, IO[bytes]]):
        _np.savez_compressed(
            io,
            matrix=self.matrix,
            identifiers=self.metadata.identifiers,
            lengths=self.metadata.lengths,
            mask=self.metadata.mask,
            minlength=self.metadata.minlength,
        )

    @classmethod
    def load(cls, io: Union[str, IO[bytes]]):
        arrs = _np.load(io, allow_pickle=True)
        metadata = CompositionMetaData(
            _vambtools.validate_input_array(arrs["identifiers"]),
            _vambtools.validate_input_array(arrs["lengths"]),
            _vambtools.validate_input_array(arrs["mask"]),
            arrs["minlength"].item(),
        )
        return cls(metadata, _vambtools.validate_input_array(arrs["matrix"]))

    def filter_min_length(self, length: int):
        if length <= self.metadata.minlength:
            return None

        mask = self.metadata.lengths >= length
        self.metadata.filter_mask(mask)
        self.metadata.minlength = length
        _vambtools.numpy_inplace_maskarray(self.matrix, mask)

    @staticmethod
    def _project(fourmers: _np.ndarray, kernel: _np.ndarray = _KERNEL) -> _np.ndarray:
        "Project fourmers down in dimensionality"
        s = fourmers.sum(axis=1).reshape(-1, 1)
        s[s == 0] = 1.0
        fourmers *= 1 / s
        fourmers += -(1 / 256)
        return _np.dot(fourmers, kernel)

    @staticmethod
    def _convert(raw: _vambtools.PushArray, projected: _vambtools.PushArray):
        "Move data from raw PushArray to projected PushArray, converting it."
        raw_mat = raw.take().reshape(-1, 256)
        projected_mat = Composition._project(raw_mat)
        projected.extend(projected_mat.ravel())
        raw.clear()

    @classmethod
    def from_file(
        cls: type[C],
        filehandle: Iterable[bytes],
        minlength: int = 2000,
        logfile: Optional[IO[str]] = None,
    ) -> C:
        """Parses a FASTA file open in binary reading mode, returning Composition.

        Input:
            filehandle: Filehandle open in binary mode of a FASTA file
            minlength: Ignore any references shorter than N bases [2000]
            logfile: Logfile to print warning to, if any
        """

        if minlength < 4:
            raise ValueError(f"Minlength must be at least 4, not {minlength}")

        raw = _vambtools.PushArray(_np.float32)
        projected = _vambtools.PushArray(_np.float32)
        lengths = _vambtools.PushArray(_np.int32)
        mask = bytearray()  # we convert to Numpy at end
        contignames: list[str] = list()
        minimum_seen_length = 2_000_000_000

        entries = _vambtools.byte_iterfasta(filehandle)

        for entry in entries:
            length = len(entry)
            minimum_seen_length = min(minimum_seen_length, length)
            skip = length < minlength
            mask.append(not skip)

            if skip:
                continue

            raw.extend(entry.kmercounts(4))

            if len(raw) > 256000:
                Composition._convert(raw, projected)

            lengths.append(len(entry))
            contignames.append(entry.identifier)

        # Convert rest of contigs
        Composition._convert(raw, projected)
        tnfs_arr = projected.take()

        # Don't use reshape since it creates a new array object with shared memory
        tnfs_arr.shape = (len(tnfs_arr) // 103, 103)
        lengths_arr = lengths.take()

        metadata = CompositionMetaData(
            _np.array(contignames, dtype=object),
            lengths_arr,
            _np.array(mask, dtype=bool),
            minlength,
        )

        if len(metadata.lengths) < 20_000:
            message = (
                f"WARNING: Parsed only {len(metadata.lengths)} sequences from FASTA file. "
                "We normally expect 20,000 sequences or more to prevent overfitting. "
                "As a deep learning model, VAEs are prone to overfitting with too few sequences. "
                "You may want to  bin more samples as a time, lower the beta parameter, "
                "or use a different binner altogether."
            )
            _vambtools.log_and_warn(message, logfile=logfile)

        # Warn the user if any contigs have been observed, which is smaller
        # than the threshold.
        if not _np.all(metadata.mask):
            message = (
                f"WARNING: The minimum sequence length has been set to {minlength}, but a contig with "
                f"length {minimum_seen_length} was seen. "
                "Better results are obtained if the sequence file is filtered to the minimum "
                "sequence length before mapping."
            )
            _vambtools.log_and_warn(message, logfile=logfile)

        return cls(metadata, tnfs_arr)
