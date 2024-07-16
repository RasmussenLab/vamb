# Overview
# We use pyrodigal to predict genes in every contigs not filtered away by
# the given mask, then use pyhmmer to predict single copy marker genes (SCGs)
# on the genes, hence getting a contig => list[SCG] mapping.
# Pyrodigal/pyhmmer is a bottleneck, so we run in parallel processes.
# To avoid inter-process communication overhead, we first split the input
# FASTA files to N files, then we have each process work on the files independently.

from vamb.vambtools import FastaEntry, Reader, RefHasher, byte_iterfasta
import pyrodigal
import pyhmmer
from multiprocessing.pool import Pool
import os
import itertools
from pathlib import Path
from typing import NewType, Sequence, Union, IO, Optional, Iterable
import shutil
from collections import defaultdict
import json
import numpy as np

MarkerID = NewType("Marker", int)
MarkerName = NewType("MarkerName", str)
ContigID = NewType("ContigID", int)
ContigName = NewType("ContigName", str)


class Markers:
    """
    The set of marker genes predicted for a collection of contigs.
    Instantiate using `Markers.from_files`, or load using `Markers.load`.
    Like Abundance objects, Markers carry a refhash to check that the markers correspond
    to the same sequences used to create the markers.
    Access the markers with `markers.markers`, a `list[Optional[np.array]]`, with one
    element for each contig. The element is `None` if there are no markers, else a list
    of marker genes present in the contig.
    The marker genes are stored as integers - the name of a marker `i` can be gotten using
    `markers.marker_names[i]`.
    In each contig, markers are deduplicated, so at most 1 of each marker is found
    in each contig.
    """

    __slots__ = ["markers", "marker_names", "refhash"]

    def __init__(
        self,
        markers: list[Optional[np.ndarray]],
        # Some IDs map to multiple names, if they act as the same SCG in the cell
        marker_names: list[list[MarkerName]],
        refhash: bytes,
    ):
        if len(set(itertools.chain.from_iterable(marker_names))) != sum(
            len(i) for i in marker_names
        ):
            raise ValueError("Marker names are not unique, but must be")

        self.markers = markers
        self.marker_names = marker_names
        self.refhash = refhash

    @property
    def n_markers(self):
        return len(self.marker_names)

    @property
    def n_seqs(self):
        return len(self.markers)

    def score_bin(self, indices: Iterable[int]) -> tuple[float, float]:
        counts = np.zeros(self.n_markers, dtype=np.uint8)
        for i in indices:
            mkrs = self.markers[i]
            if mkrs is None:
                continue
            for m in mkrs:
                counts[m] += 1

        n_unique = (counts > 0).sum()
        completeness = n_unique / self.n_markers
        contamination = (counts.sum() - n_unique) / self.n_markers
        return (completeness, contamination)

    def save(self, io: Union[Path, str, IO[str]]):
        representation = {
            "markers": [i if i is None else i.tolist() for i in self.markers],
            "marker_names": self.marker_names,
            "refhash": self.refhash.hex(),
        }
        # Check we didn't forget any fields
        assert len(representation) == len(self.__slots__)
        if isinstance(io, Path) or isinstance(io, str):
            with open(io, "w") as file:
                json.dump(representation, file)

        else:
            json.dump(representation, io)

    @classmethod
    def load(cls, io: Union[Path, str, IO[str]], refhash: Optional[bytes]):
        if isinstance(io, Path) or isinstance(io, str):
            with open(io, "rb") as file:
                representation = json.load(file)
        else:
            representation = json.load(io)
        observed_refhash = bytes.fromhex(representation["refhash"])
        if refhash is not None:
            RefHasher.verify_refhash(
                refhash=observed_refhash,
                target_refhash=refhash,
                observed_name="Loaded markers",
                target_name=None,
                identifiers=None,
            )
        markers_as_arrays = [
            i if i is None else np.array(i, dtype=np.uint8)
            for i in representation["markers"]
        ]

        return cls(markers_as_arrays, representation["marker_names"], observed_refhash)

    @classmethod
    def from_files(
        cls,
        contigs: Path,
        hmm_path: Path,
        tmpdir_to_create: Path,
        n_processes: int,
        fasta_entry_mask: Sequence[bool],
        target_refhash: Optional[bytes],
    ):
        """
        Create the Markers from input files:
        `contigs`: Path to a FASTA file with all contigs, gzipped or not.
        `hmm_path`: Path to a HMMER .hmm file with the markers. Note: Currently,
          this file can contain at most 256 markers, though this restriction can
          be lifted if necessary

        The `fasta_entry_mask` is a boolean mask of which contigs in the FASTA
        file to include. This affects the refhash which is only computed for
        the contigs not filtered away.
        If the target refhash is not None, and the computed reference hash does not
        match, an exception is thrown. See vamb.vambtools.RefHasher.
        """
        if n_processes < 1:
            raise ValueError(f"Must use at least 1 process, not {n_processes}")
        # Cap processes, because most OSs cap the number of open file handles,
        # and we need one file per process when splitting FASTA file
        elif n_processes > 64:
            print(f"Warning: Processes set to {n_processes}, capping to 64")
            n_processes = 64
        name_to_id: dict[MarkerName, MarkerID] = dict()

        # Create the list of marker names, translating those that are the same,
        # but appear under two marker names
        with open(hmm_path, "rb") as file:
            for hmm in pyhmmer.plan7.HMMFile(file):
                name = hmm.name.decode()
                if name in NORMALIZE_MARKER_TRANS_DICT:
                    continue
                name_to_id[MarkerName(name)] = MarkerID(len(name_to_id))
        for old_name, new_name in NORMALIZE_MARKER_TRANS_DICT.items():
            name_to_id[MarkerName(old_name)] = name_to_id[MarkerName(new_name)]
        id_to_names: defaultdict[MarkerID, list[MarkerName]] = defaultdict(list)
        for name, id in name_to_id.items():
            id_to_names[id].append(name)
        marker_names = [id_to_names[MarkerID(i)] for i in range(len(id_to_names))]

        # For safety: Verify that there are no more than 256 MarkerIDs, such that we can
        # store them in an uint8 array without overflow (which does not throw errors
        # in the current version of Numpy)
        assert len(marker_names) <= 256

        (refhash, paths) = split_file(
            contigs, tmpdir_to_create, n_processes, fasta_entry_mask
        )

        if target_refhash is not None:
            RefHasher.verify_refhash(
                refhash, target_refhash, "Markers FASTA file", None, None
            )

        marker_list: list[Optional[np.ndarray]] = [None] * sum(fasta_entry_mask)
        with Pool(n_processes) as pool:
            for sub_result in pool.imap_unordered(
                work_per_process,
                list(
                    zip(paths, itertools.repeat(hmm_path), itertools.repeat(name_to_id))
                ),
            ):
                for contig_id, markers in sub_result:
                    marker_list[contig_id] = markers

        shutil.rmtree(tmpdir_to_create)
        markers = cls(marker_list, marker_names, refhash)

        return markers


# Some markers have different names, but should be treated as the same SCG.
NORMALIZE_MARKER_TRANS_DICT = {
    "TIGR00388": "TIGR00389",
    "TIGR00471": "TIGR00472",
    "TIGR00408": "TIGR00409",
    "TIGR02386": "TIGR02387",
}


def filter_contigs(reader: Reader, mask: Sequence[bool]) -> Iterable[FastaEntry]:
    for record, keep in itertools.zip_longest(byte_iterfasta(reader), mask):
        if record is None or keep is None:
            raise ValueError(
                "The mask length does not match the length of FASTA records"
            )
        if keep:
            yield record


def split_file(
    input: Path, tmpdir_to_create: Path, n_splits: int, mask: Sequence[bool]
) -> tuple[bytes, list[Path]]:
    os.mkdir(tmpdir_to_create)
    paths = [tmpdir_to_create.joinpath(str(i)) for i in range(n_splits)]
    filehandles = [open(path, "w") for path in paths]
    refhasher = RefHasher()
    with Reader(input) as infile:
        records = filter_contigs(infile, mask)
        # We write the index to the record and store that instead of the name.
        for i, (outfile, record) in enumerate(
            zip(itertools.cycle(filehandles), records)
        ):
            refhasher.add_refname(record.identifier)
            record.identifier = str(i)
            print(record.format(), file=outfile)

    for filehandle in filehandles:
        filehandle.close()
    refhash = refhasher.digest()
    return (refhash, paths)


def process_chunk(
    chunk: list[FastaEntry],
    hmms: list[pyhmmer.plan7.HMM],
    name_to_id: dict[MarkerName, MarkerID],
    finder: pyrodigal.GeneFinder,
) -> list[tuple[ContigID, np.ndarray]]:
    # We temporarily store them as sets in order to deduplicate. While single contigs
    # may have duplicate markers, it makes no sense to count this as contamination,
    # because we are not about to second-guess the assembler's job of avoiding
    # chimeric sequences.
    markers: defaultdict[ContigID, set[MarkerID]] = defaultdict(set)
    alphabet = pyhmmer.easel.Alphabet.amino()
    digitized: list[pyhmmer.easel.DigitalSequence] = []
    for record in chunk:
        for gene in finder.find_genes(record.sequence):
            seq = pyhmmer.easel.TextSequence(
                name=record.identifier.encode(), sequence=gene.translate()
            ).digitize(alphabet)
            digitized.append(seq)

    for hmm, top_hits in zip(hmms, pyhmmer.hmmsearch(hmms, digitized)):
        marker_name = MarkerName(hmm.name.decode())
        marker_id = name_to_id[marker_name]
        # We need this score cutoff, which is stored in the HMM file to remove the large
        # number of false positives from HMMER
        score_cutoff = hmm.cutoffs.trusted1
        assert score_cutoff is not None
        for hit in top_hits:
            if hit.score >= score_cutoff:
                markers[ContigID(int(hit.name.decode()))].add(marker_id)

    return [
        (name, np.array(list(ids), dtype=np.uint8)) for (name, ids) in markers.items()
    ]


# We avoid moving data between processes as much as possible, so each process
# only needs these two paths, and this marker name dict which we assume to be small.
# The return type here is optimised for a small memory footprint.
def work_per_process(
    args: tuple[Path, Path, dict[MarkerName, MarkerID]],
) -> list[tuple[ContigID, np.ndarray]]:
    (contig_path, hmmpath, name_to_id) = args
    with open(hmmpath, "rb") as file:
        hmms = list(pyhmmer.plan7.HMMFile(file))

    # Chunk up the FASTA file for memory efficiency reasons, while still
    # allowing pyhmmer to scan multiple sequences at once for speed
    chunk: list[FastaEntry] = []
    result: list[tuple[ContigID, np.ndarray]] = []
    finder = pyrodigal.GeneFinder(meta=True)
    with open(contig_path, "rb") as file:
        for record in byte_iterfasta(file):
            chunk.append(record)
            if len(chunk) == 2048:
                result.extend(process_chunk(chunk, hmms, name_to_id, finder))
                chunk.clear()
        result.extend(process_chunk(chunk, hmms, name_to_id, finder))

    return result
