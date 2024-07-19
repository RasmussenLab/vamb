from typing import Optional
from pathlib import Path
from vamb.parsecontigs import CompositionMetaData


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
        return cls(observed, metadata, is_canonical)

    def __init__(
        self,
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
        for contigname, taxonomy in observed_taxonomies:
            index = index_of_contigname.get(contigname)
            if index is None:
                raise ValueError(
                    f'When parsing taxonomy, found contigname "{contigname}", '
                    "but no sequence of that name is in the FASTA file"
                )
            existing = contig_taxonomies[index]
            if existing is not None:
                raise ValueError(
                    f'Duplicate contigname when parsing taxonomy: "{contigname}"'
                )
            contig_taxonomies[index] = taxonomy
        self.refhash = metadata.refhash
        self.is_canonical = is_canonical

    @staticmethod
    def parse_tax_file(
        path: Path, force_canonical: bool
    ) -> list[tuple[str, ContigTaxonomy]]:
        with open(path) as file:
            result: list[tuple[str, ContigTaxonomy]] = []
            lines = filter(None, map(str.rstrip, file))
            header = next(lines, None)
            if header is None or not header.startswith("contigs\tpredictions"):
                raise ValueError(
                    'In taxonomy file, expected header to begin with "contigs\\tpredictions"'
                )
            for line in lines:
                (contigname, taxonomy, *_) = line.split("\t")
                result.append(
                    (
                        contigname,
                        ContigTaxonomy.from_semicolon_sep(taxonomy, force_canonical),
                    )
                )

        return result
