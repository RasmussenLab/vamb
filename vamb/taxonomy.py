from typing import Optional, IO
from pathlib import Path
from vamb.parsecontigs import CompositionMetaData
from vamb.vambtools import strip_string_newline
import numpy as np
from typing import Union

TAXONOMY_HEADER = "contigs\tpredictions"
PREDICTED_TAXONOMY_HEADER = "contigs\tpredictions\tscores"


class ContigTaxonomy:
    """
    Hierarchical taxonomy of some contig.
    If `is_canonical`, the ranks are assumed to be domain, phylum, class,
    order, family, genus, species, in that order.
    The taxonomy may be arbitrarily truncated, e.g. ["Eukaryota", "Chordata"]
    is a valid (canonical) taxonomy for a human.
    """

    __slots__ = ["ranks"]

    def __init__(self, ranks: list[str], is_canonical: bool = False):
        if is_canonical and len(ranks) > 7:
            raise ValueError(
                "For a canonical ContigTaxonomy, there must be at most 7 ranks"
            )

        self.ranks = ranks

    @classmethod
    def from_semicolon_sep(cls, s: str, is_canonical: bool = False):
        if len(s) == 0:
            return cls([], is_canonical)
        else:
            return cls(s.split(";"), is_canonical)

    @property
    def genus(self) -> Optional[str]:
        if len(self.ranks) < 6:
            return None
        return self.ranks[5]


class Taxonomy:
    """
    * contig_taxonomies: An Optional[ContigTaxonomy] for every contig given by the
      CompositionMetaData used to instantiate
    * refhash: Refhash of CompositionMetaData used to instantiate
    * is_canonical: If the taxonomy uses the canonical seven ranks
      (domain, phylum, class, order, family, genus, species).
    """

    __slots__ = ["contig_taxonomies", "refhash", "is_canonical"]

    @property
    def nseqs(self) -> int:
        return len(self.contig_taxonomies)

    @classmethod
    def from_file(
        cls, tax_file: Path, metadata: CompositionMetaData, is_canonical: bool
    ):
        observed = cls.parse_tax_file(tax_file, is_canonical)
        return cls.from_observed(observed, metadata, is_canonical)

    @classmethod
    def from_refined_file(
        cls, tax_file: Path, metadata: CompositionMetaData, is_canonical: bool
    ):
        observed = PredictedTaxonomy.parse_tax_file(tax_file, is_canonical)
        observed = [(name, tax.contig_taxonomy) for (name, tax) in observed]
        return cls.from_observed(observed, metadata, is_canonical)

    @classmethod
    def from_observed(
        cls,
        observed_taxonomies: list[tuple[str, ContigTaxonomy]],
        metadata: CompositionMetaData,
        is_canonical: bool,
    ):
        index_of_contigname: dict[str, int] = {
            c: i for (i, c) in enumerate(metadata.identifiers)
        }
        contig_taxonomies: list[Optional[ContigTaxonomy]] = [None] * len(
            metadata.identifiers
        )
        n_found = 0
        for contigname, taxonomy in observed_taxonomies:
            index = index_of_contigname.get(contigname)
            if index is None:
                continue
            n_found += 1
            existing = contig_taxonomies[index]
            if existing is not None:
                raise ValueError(
                    f'Duplicate contigname when parsing taxonomy: "{contigname}"'
                )
            contig_taxonomies[index] = taxonomy

        if n_found != metadata.nseqs:
            raise ValueError(
                f"In taxonomy file, expected {metadata.nseqs} contigs that are "
                f"also present in the filtered FASTA file, but found {n_found}. "
                "Note that this might occur because some contigs in the taxonomy "
                "file falls under the minimum length threshold."
            )
        return cls(contig_taxonomies, metadata.refhash, is_canonical)

    def __init__(
        self,
        contig_taxonomies: list[Optional[ContigTaxonomy]],
        refhash: bytes,
        is_canonical: bool,
    ):
        self.contig_taxonomies = contig_taxonomies
        self.refhash = refhash
        self.is_canonical = is_canonical
        assert_unambiguous_ranks(self)

    @staticmethod
    def parse_tax_file(
        path: Path, force_canonical: bool
    ) -> list[tuple[str, ContigTaxonomy]]:
        with open(path) as file:
            result: list[tuple[str, ContigTaxonomy]] = []
            header = next(file, None)
            header = None if header is None else header.rstrip()
            if header is None or header != TAXONOMY_HEADER:
                raise ValueError(
                    f"In taxonomy file '{path}', expected header to be {repr(TAXONOMY_HEADER)}, "
                    f"but found {'no header' if header is None else repr(header)}"
                )
            # Minus two because we already read header, and because Python is zero-indexed
            for lineno_minus_two, line in enumerate(file):
                line = strip_string_newline(line)
                fields = line.split("\t")
                if len(fields) != 2:
                    raise ValueError(
                        f"In taxonomy file '{path}', on line {lineno_minus_two + 2}, "
                        f"expected 2 tab-separated columns, but found {len(fields)}."
                    )
                (contigname, taxonomy) = fields
                result.append(
                    (
                        contigname,
                        ContigTaxonomy.from_semicolon_sep(taxonomy, force_canonical),
                    )
                )

        return result


class PredictedContigTaxonomy:
    slots = ["contig_taxonomy", "probs"]

    def __init__(self, tax: ContigTaxonomy, probs: np.ndarray):
        if len(probs) != len(tax.ranks):
            raise ValueError("The length of probs must equal that of ranks")
        # Due to floating point errors, the probabilities may be slightly outside of 0 or 1.
        # We could perhaps validate the values, but that's not likely to be necessary.
        np.clip(probs, a_min=0.0, a_max=1.0, out=probs)
        self.contig_taxonomy = tax
        self.probs = probs


class PredictedTaxonomy:
    "Output of Taxometer"

    __slots__ = ["contig_taxonomies", "refhash", "is_canonical"]

    def __init__(
        self,
        taxonomies: list[PredictedContigTaxonomy],
        metadata: CompositionMetaData,
        is_canonical: bool,
    ):
        if len(taxonomies) != len(metadata.identifiers):
            raise ValueError("Length of taxonomies must match that of identifiers")

        self.contig_taxonomies = taxonomies
        self.refhash = metadata.refhash
        self.is_canonical = is_canonical
        assert_unambiguous_ranks(self)

    def to_taxonomy(self) -> Taxonomy:
        lst: list[Optional[ContigTaxonomy]] = [
            p.contig_taxonomy for p in self.contig_taxonomies
        ]
        return Taxonomy(lst, self.refhash, self.is_canonical)

    @property
    def nseqs(self) -> int:
        return len(self.contig_taxonomies)

    @staticmethod
    def parse_tax_file(
        path: Path, force_canonical: bool
    ) -> list[tuple[str, PredictedContigTaxonomy]]:
        with open(path) as file:
            result: list[tuple[str, PredictedContigTaxonomy]] = []
            lines = filter(None, map(str.rstrip, file))
            header = next(lines, None)
            if header is None or header != PREDICTED_TAXONOMY_HEADER:
                raise ValueError(
                    f"In predicted taxonomy file '{path}', "
                    f"expected header to be {repr(PREDICTED_TAXONOMY_HEADER)}, "
                    f"but found {'no header' if header is None else repr(header)}."
                )
            for linenum_minus_two, line in enumerate(lines):
                fields = line.split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"Expected 3 fields in line {linenum_minus_two + 2} of file '{path}', "
                        f"got {len(fields)}.\nLine: '{line}'"
                    )
                (contigname, taxonomy, scores) = fields
                contig_taxonomy = ContigTaxonomy.from_semicolon_sep(
                    taxonomy, force_canonical
                )
                probs = np.array([float(i) for i in scores.split(";")], dtype=float)
                result.append(
                    (
                        contigname,
                        PredictedContigTaxonomy(contig_taxonomy, probs),
                    )
                )

        return result

    def write_as_tsv(self, file: IO[str], comp_metadata: CompositionMetaData):
        if self.refhash != comp_metadata.refhash:
            raise ValueError(
                "Refhash of comp_metadata and predicted taxonomy must match"
            )
        assert self.nseqs == comp_metadata.nseqs
        print(PREDICTED_TAXONOMY_HEADER, file=file)
        for i in range(self.nseqs):
            tax = self.contig_taxonomies[i]
            ranks_str = ";".join(tax.contig_taxonomy.ranks)
            probs_str = ";".join([str(round(i, 5)) for i in tax.probs])
            print(
                comp_metadata.identifiers[i],
                ranks_str,
                probs_str,
                file=file,
                sep="\t",
            )


def assert_unambiguous_ranks(taxonomy: Union[Taxonomy, PredictedTaxonomy]):
    """
    Ensure that no rank appears at multiple levels in the taxonomy.
    This will mess up some of TaxVamb's algorithms since it's based on the names of
    taxons, and therefore, having a name on two ranks may cause it to be parsed
    as a graph which is not a tree.
    """
    seen_ranks: dict[str, int] = dict()
    parent_of: dict[str, str] = dict()
    for i in taxonomy.contig_taxonomies:
        # May be missing from Taxonomy
        if i is None:
            continue

        if isinstance(i, ContigTaxonomy):
            ranks = i.ranks
        else:
            ranks = i.contig_taxonomy.ranks

        for rank, name in enumerate(ranks):
            if seen_ranks.setdefault(name, rank) != rank:
                raise ValueError(
                    f'Taxonomy is ambiguous: "{name}" appears at multiple ranks'
                )

        for parent, child in zip(ranks, ranks[1:]):
            if parent_of.setdefault(child, parent) != parent:
                raise ValueError(
                    f'Taxonomy is ambiguous: "{child}" has multiple parents'
                )
