# The approach to benchmarking is this:

# A "subject" is a source sequence, e.g. a sequence directly from an underlying genome.
# A subject has only a name and length. Subject names are unique.

# A set of subjects is a Genome. The subjects are assumed to not be overlapping. Hence,
# the "breadth" of a genome is the sum of subject lengths in that genome. Genome names
# are assumed to be unique.

# A Contig is composed of a name, a source subject, and a start and end position within
# that subject, where the contig maps to. Contig names are assumed to be unique.
# Contigs always have a source. If the source is unknown, a dummy source can be used,
# i.e. a source not used by any other contig.

# A named set of Contigs is a Bin. Bin names are unique. The "breadth" of a Bin is the total
# number of basepairs (positions) covered by any contig in the bin. Contigs can overlap the
# same subject, but a position covered multiple times only count once towards breadth.

# A Bin/Genome pair has an "intersection". This is the number of basepairs (positions) in
# the Bin which come from a subject in the given Genome. In other words, it is the breadth
# of the subset of contigs in the Bin that originates from the Genome.

# Bin/Genome pairs also have other statistics:
# * True positives (TP) is the intersection
# * False positives (FP) is Bin breadth minus TP
# * False negatives (FN) is Genome breadth minus TP
# * Recall, precision and F-score is calculated from TP, FP and FN as usual.

# A binning result is described by two objects, the Reference and the Binning.

# A Reference object represents the information in the test dataset, and is independent of
# any concrete binning tool or binning result. Hence, a test dataset has one Reference object.
# It is composed of:
# * The total set of Contigs input to the binning
# * The total set of Genomes that the set of Contigs represent. This implies that all Contigs
#   in the Reference must derive from a subject which is also in a Genome in the Reference, i.e.
#   there must be no contigs where the source is not from a Genome (via a subject) in the Reference.
# * A list of "taxmaps". A taxmap describe taxonomic information about the Genomes of the
#   Reference. A taxmap is a dictionary that maps the name of a clade to its parent clade.
#   For example, the pair "Ecoli":"Escherichia" could be present.
#   The first taxmap has all Genome names as keys.
#   The Nth taxmap has as keys the values of the N-1th taxmap (except None).
#   A clade for which no higher level clade is known or defined maps to None.

# The Binning object contains the matching Reference and a collection of Bins.
# When instantiated, it can do a number of filters or transformations of the bins, e.g.
# binsplitting or filtering of bins by number of contigs.
# The only responsibility of the Binning object is to store the input bins, do this filtering
# and transformation, and do the benchmarking itself. In practice, this means computing a
# series of "counters", one for each Reference taxmap:

# The first counter is a map from minimum (recall, precision) to the number of genomes
# for which at least one bin in the Binning has a recall, precision level at or above the
# minimum recall/precision level.

# All the other counters are computed iteratively from the previous one based on the corresponding
# Reference taxmap. For a given bin B and a clade C where C  is not a Genome (i.e. not the lowest
# level of the taxmaps), the recall of the B/C pair is defined as the max of the recalls B/Cchild for
# all Cchild taxonomic children of C according to the taxmap.
# The precision is calculated as the SUM of all B/Cchild.


__doc__ = """Benchmark script

This benchmarks bins using number of covered sites.

First, a Reference is needed. This can be loaded from a JSON file, see
the Reference class source code for details.

You also need a file with the binning. This is simply the clusters.tsv file
as produced by Vamb, i.e. first column is the name of the bin of a contig,
second column the name of the contig.

Recall of a genome/bin pair is defined as the number of genome basepairs
covered by contigs in the bin divided by total number of basepairs of that
genome.
Precision is the number of bases in that genome covered by contigs in the
bin divided by the number of bases covered in any genome from that bin.

Usage:
>>> ref = Reference.from_file(open_reference_file_hande)
>>> bins = Binning.from_file(open_clusters_file_handle, ref)
>>> bins.print_matrix(rank=1)
"""

from collections import defaultdict
from itertools import product
import sys
import json
from math import isfinite
from vamb import vambtools
from collections.abc import Iterable, Sequence
from typing import Optional, TypeVar, IO, Any

C = TypeVar("C", bound="Contig")
G = TypeVar("G", bound="Genome")
Bn = TypeVar("Bn", bound="Bin")
R = TypeVar("R", bound="Reference")
Bs = TypeVar("Bs", bound="Binning")


class Contig:
    """An object representing a contig mapping to a subject at position start:end.
    Mapping positions use the half-open interval, like Python ranges and slices.

    Instantiate either with name, subject and mapping start/end:
        Contig('contig_41', 'subject_1', 600, 11, 510)
    Or with only name and length
        Contig.subjectless('contig_41', 499)
    A subjectless Contig uses itself as a subject (implying it only maps to itself).
    """

    __slots__ = ["name", "subject", "start", "end"]

    def __init__(self, name: str, subject: str, start: int, end: int):
        if start < 0:
            raise ValueError(f'Contig "{name}" has negative start index {start}')
        elif end <= start:
            raise ValueError(
                "Contig end must be higher than start, but "
                f'contig "{name}" spans {start}-{end}.'
            )

        self.name = name
        self.subject = subject
        self.start = start
        self.end = end

    @classmethod
    def subjectless(cls: type[C], name: str, length: int) -> C:
        "Instantiate with only name and length"
        return cls(name, name, 0, length)

    def __repr__(self) -> str:
        return f'Contig("{self.name}", "{self.subject}", {self.start}, {self.end})'

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Contig) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name) ^ 3458437981

    def __len__(self) -> int:
        return self.end - self.start


class Genome:
    """An object representing a set of subjects (or sources), i.e. genomes or source contigs
    that the binning contigs are drawn from.
    >>> g = Genome("Ecoli")
    >>> g.add("chrom", 5_300_000)
    """

    __slots__ = ["name", "sources", "breadth"]

    def __init__(self, name: str):
        self.name = name
        self.sources: dict[str, int] = dict()
        self.breadth = 0

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Genome) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name) ^ 4932511201

    def add(self, source: str, len: int) -> None:
        if len <= 0:
            raise ValueError(
                "Source sequence must have nonzero positive length, "
                f"but {source} has length {len}"
            )
        if source in self.sources:
            raise ValueError(f"Genome {self.name} already has source {source}.")

        self.sources[source] = len
        self.breadth += len

    def __repr__(self):
        return f'Genome("{self.name}")'


class Bin:
    """An object representing a set of Contigs.
    Should be instantiated with Bin.from_contigs. See that method for how to safely
    instantiate the object.
    """

    __slots__ = ["name", "contigs", "intersections", "breadth"]

    def __init__(self, name: str) -> None:
        self.name = name
        self.contigs: set[Contig] = set()
        self.intersections: dict[Genome, int] = dict()
        self.breadth: int = 0

    @property
    def ncontigs(self) -> int:
        return len(self.contigs)

    @classmethod
    def from_contigs(
        cls: type[Bn],
        name: str,
        contigs: Iterable[Contig],
        genomeof: dict[Contig, Genome],
    ) -> Bn:
        instance = cls(name)
        instance.contigs.update(contigs)
        instance._finalize(genomeof)  # remember to do this
        return instance

    def __repr__(self) -> str:
        return f'Bin("{self.name}")'

    @staticmethod
    def _intersection(contigs: list[Contig]) -> int:
        # Note: All contigs MUST come from same source, else the code is invalid
        contigs.sort(key=lambda x: x.start)
        result = 0
        rightmost_end = -1

        for contig in contigs:
            result += max(contig.end, rightmost_end) - max(contig.start, rightmost_end)
            rightmost_end = max(contig.end, rightmost_end)

        return result

    def _finalize(self, genomeof: dict[Contig, Genome]) -> None:
        self.intersections.clear()
        by_source: defaultdict[tuple[Genome, str], list[Contig]] = defaultdict(list)
        for contig in self.contigs:
            by_source[(genomeof[contig], contig.subject)].append(contig)
        for ((genome, _), contigs) in by_source.items():
            self.intersections[genome] = self.intersections.get(
                genome, 0
            ) + self._intersection(contigs)
        self.breadth = sum(self.intersections.values())

    def confusion_matrix(self, genome: Genome) -> tuple[int, int, int]:
        "Given a genome and a binname, returns TP, FP, FN"
        # This is None if it's not updated, we want it to throw type errors then
        d: dict[Genome, int] = self.intersections
        breadth: int = self.breadth
        tp = d.get(genome, 0)
        fp = breadth - tp
        fn = genome.breadth - tp
        return (tp, fp, fn)

    def recall_precision(self, genome: Genome) -> tuple[float, float]:
        (tp, fp, fn) = self.confusion_matrix(genome)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return (recall, precision)

    def fscore(self, b: float, genome: Genome) -> float:
        recall, precision = self.recall_precision(genome)

        # Some people say the Fscore is undefined in this case.
        # We define it to be 0.0
        if (recall + precision) == 0.0:
            return 0.0

        return (1 + b * b) * (recall * precision) / ((b * b * precision) + recall)

    def f1(self, genome: Genome) -> float:
        return self.fscore(1.0, genome)


class Reference:
    """An object that represent a set of Genome and Contigs, where the Contigs are sampled
    from the genomes. Also contain the phylogenetic tree for the contained genomes.
    Either instantiate directly and use self.add_contig and self.add_taxonomy, else
    use Reference.from_file
    """

    __slots__ = ["genomes", "genomeof", "contig_by_name", "taxmaps"]

    def __init__(self):
        self.genomes: set[Genome] = set()  # genome_name : genome dict
        self.genomeof: dict[Contig, Genome] = dict()  # contig : genome dict
        self.contig_by_name: dict[str, Contig] = dict()
        # This is a list of dicts: The first one maps genomename to name of next taxonomic level
        # The second maps name of second level to name of third level etc. None means it's the top level
        self.taxmaps: list[dict[str, Optional[str]]] = [dict()]

    def _add_genome(self, genome: Genome) -> None:
        if genome in self.genomes:
            raise ValueError(f'Genome "{genome.name}" already in reference.')
        self.genomes.add(genome)
        self.taxmaps[0][genome.name] = None

    def _add_contig(self, contig: Contig, genome: Genome) -> None:
        if contig in self.genomeof:
            raise ValueError(f'Reference already has Contig "{contig.name}"')
        if contig.subject not in genome.sources:
            raise ValueError(
                f'Attempted to add contig "{contig.name}" with source "{contig.subject}" '
                f'to genome "{genome.name}", but genome has no such source'
            )
        if genome not in self.genomes:
            raise ValueError(f'Genome "{genome.name}" is not in reference.')
        if genome.sources[contig.subject] < contig.end:
            raise IndexError(
                f'Attempted to add contig "{contig.name}" with mapping end {contig.end} '
                f'to subject "{contig.subject}", '
                f"but the subject only has length {genome.sources[contig.subject]}"
            )

        self.genomeof[contig] = genome
        self.contig_by_name[contig.name] = contig

    def _add_taxonomy(self, level: int, child: str, parent: str) -> None:
        existing = self.taxmaps[level][child]
        if existing is None:
            self.taxmaps[level][child] = parent
            # Add parent of parent, if not already present
            if level + 1 == self.nranks:
                self.taxmaps.append(dict())

            grandparent = self.taxmaps[level + 1].get(parent, 0)
            if grandparent == 0:
                self.taxmaps[level + 1][parent] = None

        elif existing != parent:
            raise ValueError(
                f'Clade "{child}" maps to both parent clade "{parent}" and "{existing}"'
            )

    @property
    def ngenomes(self) -> int:
        return len(self.genomes)

    @property
    def ncontigs(self) -> int:
        return len(self.genomeof)

    @property
    def nranks(self) -> int:
        return len(self.taxmaps)

    def __repr__(self) -> str:
        return f"<Reference with {self.ngenomes} genomes, {self.ncontigs} contigs and {self.nranks} ranks>"

    def parse_bins(
        self, io: Iterable[str], binsplit_sep: Optional[str] = None
    ) -> list[Bin]:
        clusters = vambtools.read_clusters(io).items()
        if binsplit_sep is not None:
            clusters = vambtools.binsplit(clusters, binsplit_sep)
        return self.load_bins(clusters)

    def load_bins(self, bins: Iterable[tuple[str, Iterable[str]]]) -> list[Bin]:
        """Convert a set of bin names to a list of Bins"""
        result: list[Bin] = list()
        for (binname, contignames) in bins:
            contigs = (self.contig_by_name[name] for name in contignames)
            result.append(Bin.from_contigs(binname, contigs, self.genomeof))

        return result

    @classmethod
    def from_file(cls: type[R], io: IO[str]) -> R:
        json_dict = json.load(io)
        return cls.from_dict(json_dict)

    @classmethod
    def from_dict(cls: type[R], json_dict: dict[str, Any]) -> R:
        instance = cls()
        for (genomename, sourcesdict) in json_dict["genomes"].items():
            genome = Genome(genomename)
            instance._add_genome(genome)
            for (sourcename, (sourcelen, contigdict)) in sourcesdict.items():
                genome.add(sourcename, sourcelen)
                for (contigname, (start, end)) in contigdict.items():
                    # JSON format is 1-indexed and includes endpoints, whereas
                    # Contig struct is not, so compensate.
                    contig = Contig(contigname, sourcename, start - 1, end)
                    instance._add_contig(contig, genome)

        for _ in range(len(json_dict["taxmaps"]) - 1):
            instance.taxmaps.append(dict())
        for (level, taxmap) in enumerate(json_dict["taxmaps"]):
            for (child, parent) in taxmap.items():
                instance._add_taxonomy(level, child, parent)

        return instance

    def save(self, io: IO[str]) -> None:
        json_dict: dict[str, Any] = {"genomes": dict(), "taxmaps": []}

        # "genomes": {genomename: [subject_len, {contigname: [start, stop]}]}
        genome_dict = json_dict["genomes"]
        for genome in self.genomes:
            source_dict: dict[str, list[Any]] = dict()
            genome_dict[genome.name] = source_dict
            for (sourcename, length) in genome.sources.items():
                source_dict[sourcename] = [length, dict()]

        for (contig, genome) in self.genomeof.items():
            # JSON format is 1-indexes and includes endpoints, whereas
            # Contig struct is not, so compensate.
            genome_dict[genome.name][contig.subject][1][contig.name] = [
                contig.start + 1,
                contig.end,
            ]

        for taxmap in self.taxmaps:
            d: dict[str, str] = dict()
            for (child, parent) in taxmap.items():
                if parent is not None:
                    d[child] = parent
            if len(d) > 0:
                json_dict["taxmaps"].append(d)

        json.dump(json_dict, io)


class Binning:
    """The result of a set of Bins applied to a Reference.
    See Binning.from_file for more usage.
    >>> with open("clusters.tsv") as file:
    ...     binning = Binning.from_file(file, reference)

    Properties:
    * reference: Reference
    * bins: list[Bin]

    Properties after self.benchmark()
    * counters: list[dict[tuple[float, float], int]]: Genomes at recall/prec thresholds
    * recalls: Recall thresholds used to compute counters
    * precisions: Precision thresholds used to compute counters

    Extra arguments to Binning.from_file:
    * disjoint: If True, do not allow same contig in multiple bins
    * binsplit_separator: If str and not None, split bins by separator
    * minsize: Filter away all bins with breadth less than this
    * mincontigs: Filter away bins with fewer contigs than this
    """

    __slots__ = ["reference", "bins", "counters", "recalls", "precisions"]
    _DEFAULTRECALLS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    _DEFAULTPRECISIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

    def __init__(
        self,
        bins: Sequence[Bin],
        reference: Reference,
        recalls: Sequence[float] = _DEFAULTRECALLS,
        precisions: Sequence[float] = _DEFAULTPRECISIONS,
        disjoint: bool = True,
    ) -> None:
        self.recalls: tuple[float] = self._validate_rec_prec(recalls)
        self.precisions: tuple[float] = self._validate_rec_prec(precisions)
        self.reference = reference
        self.bins: list[Bin] = list(bins)
        # This is initialized by self._benchmark
        self.counters: list[dict[tuple[float, float], int]] = list()

        if disjoint:
            seen_contigs: set[Contig] = set()
            for bin in self.bins:
                for contig in bin.contigs:
                    if contig in seen_contigs:
                        raise ValueError(
                            f'Contig "{contig.name}" seen twice in disjoint binning.'
                        )
                    seen_contigs.add(contig)
        self._benchmark()

    @classmethod
    def from_file(
        cls: type[Bs],
        filehandle: Iterable[str],
        reference: Reference,
        recalls: Sequence[float] = _DEFAULTRECALLS,
        precisions: Sequence[float] = _DEFAULTPRECISIONS,
        disjoint: bool = True,
        binsplit_separator: Optional[str] = None,
        minsize: int = 1,
        mincontigs: int = 1,
    ) -> Bs:
        bins = reference.parse_bins(filehandle, binsplit_separator)
        bins = cls.filter_bins(bins, minsize, mincontigs)
        instance = cls(bins, reference, recalls, precisions, disjoint)
        return instance

    @classmethod
    def gold_standard(
        cls: type[Bs],
        reference: Reference,
        recalls: Sequence[float] = _DEFAULTRECALLS,
        precisions: Sequence[float] = _DEFAULTPRECISIONS,
    ) -> Bs:
        "Return a Binning from a given Reference where each Genome is precisely one Bin"
        contigsof: defaultdict[Genome, list[Contig]] = defaultdict(list)
        for contig in reference.contig_by_name.values():
            contigsof[reference.genomeof[contig]].append(contig)
        bins = [
            Bin.from_contigs(genome.name, contigs, reference.genomeof)
            for (genome, contigs) in contigsof.items()
        ]
        instance = cls(bins, reference, recalls, precisions, disjoint=False)
        return instance

    def print_matrix(self, rank: int, file: IO[str] = sys.stdout) -> None:
        """Prints the recall/precision number of bins to STDOUT."""
        recalls: tuple[float] = self.recalls
        precisions: tuple[float] = self.precisions
        assert self.counters is not None

        if rank >= self.reference.nranks:
            raise IndexError("Taxonomic rank out of range")

        print("\tRecall", file=file)
        print("Prec.", "\t".join([str(r) for r in recalls]), sep="\t", file=file)

        for min_precision in precisions:
            row = [
                self.counters[rank][(min_recall, min_precision)]
                for min_recall in recalls
            ]
            print(min_precision, "\t".join([str(i) for i in row]), sep="\t", file=file)

    def __repr__(self) -> str:
        return (
            f"<Binning with {self.nbins} bins and reference {hex(id(self.reference))}>"
        )

    @property
    def nbins(self) -> int:
        return len(self.bins)

    @staticmethod
    def filter_bins(bins: Iterable[Bin], minsize: int, mincontigs: int) -> list[Bin]:
        def is_ok(bin: Bin) -> bool:
            breadth: int = bin.breadth
            return breadth >= minsize and bin.ncontigs >= mincontigs

        return list(filter(is_ok, bins))

    def _benchmark(self) -> None:
        counters: list[dict[tuple[float, float], int]] = list()
        # key here is name of genome (and later, other taxonomic ranks)
        rp_by_name: dict[str, dict[Bin, tuple[float, float]]] = {
            g.name: dict() for g in self.reference.genomes
        }
        for bin in self.bins:
            assert bin.intersections is not None
            for genome in bin.intersections:
                rp_by_name[genome.name][bin] = bin.recall_precision(genome)
        bitvectors = self._get_seen_bitvectors(rp_by_name)
        counters.append(self._counter_from_bitvectors(bitvectors))

        for rank in range(self.reference.nranks - 1):
            rp_by_name = self._uprank_rp_by_name(rank, rp_by_name)
            bitvectors = self._get_seen_bitvectors(rp_by_name)
            counters.append(self._counter_from_bitvectors(bitvectors))

        self.counters = counters

    @staticmethod
    def _validate_rec_prec(x: Iterable[float]) -> tuple[float]:
        s: set[float] = set()
        for i in x:
            if i in s:
                raise ValueError(f"Recall/precision value {i} present multiple times.")
            if not isfinite(i) or i <= 0.0 or i > 1:
                raise ValueError(
                    f"Recall/precision value {i} is not a finite value in (0;1]"
                )
            s.add(i)
        if len(s) == 0:
            raise ValueError("Must provide at least 1 recall/precision value")
        return tuple(sorted(s))

    def _get_seen_bitvectors(
        self, rp_by_name: dict[str, dict[Bin, tuple[float, float]]]
    ) -> dict[str, int]:
        recalls: tuple[float] = self.recalls
        precisions: tuple[float] = self.precisions
        bitvectors: dict[str, int] = dict()

        for (cladename, d) in rp_by_name.items():
            bitvector = 0
            for (recall, precision) in d.values():
                for (i, (min_recall, min_precision)) in enumerate(
                    product(recalls, precisions)
                ):
                    if recall >= min_recall and precision >= min_precision:
                        bitvector |= 1 << i
                    # Shortcut: Once we pass min recall, rest of values can be skipped
                    elif recall < min_recall:
                        break

            bitvectors[cladename] = bitvector

        return bitvectors

    def _counter_from_bitvectors(
        self, bitvectors: dict[str, int]
    ) -> dict[tuple[float, float], int]:
        recalls: tuple[float] = self.recalls
        precisions: tuple[float] = self.precisions
        result: dict[tuple[float, float], int] = {
            (r, p): 0 for (r, p) in product(recalls, precisions)
        }
        for (_, bitvector) in bitvectors.items():
            for (i, rp) in enumerate(product(recalls, precisions)):
                result[rp] += (bitvector >> i) & 1

        return result

    def _uprank_rp_by_name(
        self, fromrank: int, rp_by_name: dict[str, dict[Bin, tuple[float, float]]]
    ) -> dict[str, dict[Bin, tuple[float, float]]]:
        result: dict[str, dict[Bin, tuple[float, float]]] = dict()
        for (child, parent) in self.reference.taxmaps[fromrank].items():
            # If no parent clade, we just name the "parent" same as child
            if parent is None:
                parent = child

            if parent not in result:
                result[parent] = dict()

            parent_dict = result[parent]
            for (bin, (old_recall, old_prec)) in rp_by_name[child].items():
                (new_recall, new_prec) = parent_dict.get(bin, (0.0, 0.0))
                parent_dict[bin] = (max(old_recall, new_recall), old_prec + new_prec)

        return result
