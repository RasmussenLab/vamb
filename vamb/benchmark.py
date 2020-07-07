# The approach to benchmarking is this:
#
# Most fundamentally, we have a Contig. A Contig is associated with a subject, where
# it comes from and aligns to. It thus has a start and end position in that subject.
# A set of contigs has a breadth. This is the number of basepairs covered by the
# contigs in their respective subjects, where basepairs covered by multiple Contigs are
# only counted once.
#
# A Genome is simply a collection of contigs. A set of Genomes constitute a Reference.
#
# A Binning is a collection of sets of contigs, each set representing a bin. It is
# required that all contigs in a Binning comes from Genomes in the same Reference.
# The breadth of a Binning is the sum of breadths of each bin. If the same subject are
# covered by multiple different bins, these are counted multiple times. This implies that
# the breadth of a Binning can exceed the total length of all the underlying Genomes.
#
# When creating a Binning, for each genome/bin pair, the following base statistics
# are calculated:
# * True positives (TP) is the breadth of the intersection of the bin and the genome contigs
# * False positives (FP) is the bin breadth minus TP
# * False negatives (FN) is the Genome breadth minus TP
# * True negatives (TN) is the Reference breadth minus TP+FP+FN
#
# From these values, the recall, precision, Matthew's correlation coefficient (MCC)
# and F1 statistics are calculated.
#
# Since each genome has a set of statistics for each bin, the MCC or F1 of a genome
# is the maximal MCC or F1 among all bins.  The mean MCC or F1 is then the mean of
# these maximal MCCs or F1.

__doc__ = """Benchmark script

This benchmarks bins using number of covered sites.

First, a Reference is needed. This can be generated from a tab-separated file:
contigname    genomename    subjectname    start    end

Where "contigname" is the name of the binned contig, which belongs to genome
"genomename", and aligns to the reference contig "subjectname" at positions
"start"-"end", both inclusive.

You also need a file with the binning. This is simply the clusters.tsv file
as produced by Vamb, i.e. first column is the name of the bin of a contig,
second column the name of the contig.

Recall of a genome/bin pair is defined as the number of bases in that genome
covered by contigs in the bin divided by the number of bases covered in that
genome from all bins total. Precision is the number of bases in that genome
covered by contigs in the bin divided by the number of bases covered in any
genome from that bin.

Usage:
>>> ref = Reference.from_file(open_reference_file_hande)
>>> ref.load_tax_file(open_tax_file_handle) # optional
>>> bins = Binning.from_file(open_clusters_file_handle, ref)
>>> bins.print_matrix(rank=1)
"""

import collections as _collections
from operator import add as _add
from itertools import product as _product
import sys as _sys
from math import sqrt as _sqrt
import vamb.vambtools as _vambtools

class Contig:
    """An object representing a contig mapping to a subject at position start:end.
    Mapping positions use the half-open interval, like Python ranges and slices.

    Instantiate either with name, subject and mapping start/end:
        Contig('contig_41', 'subject_1', 11, 510)
    Or with only name and length
        Contig.subjectless('contig_41', 499)
    A subjectless Contig uses itself as a subject (implying it only maps to itself).
    """
    __slots__ = ['name', 'subject', 'start', 'end']

    def __init__(self, name, subject, start, end):
        if end <= start:
            raise ValueError('Contig end must be higher than start')

        self.name = name
        self.subject = subject
        self.start = start
        self.end = end

    @classmethod
    def subjectless(cls, name, length):
        "Instantiate with only name and length"
        return cls(name, name, 0, length)

    def __repr__(self):
        return 'Contig({}, subject={}, {}:{})'.format(self.name, self.subject, self.start, self.end)

    def __len__(self):
        return self.end - self.start

class Genome:
    """A set of contigs known to come from the same organism.
    The breadth must be updated after adding/removing contigs with self.update_breadth(),
    before it is accurate.
    >>> genome = Genome('E. coli')
    >>> genome.add(contig)
    >>> genome.update_breadth()
    """
    __slots__ = ['name', 'breadth', 'contigs']

    def __init__(self, name):
        self.name = name
        self.contigs = set()
        self.breadth = 0

    def add(self, contig):
        self.contigs.add(contig)

    def remove(self, contig):
        self.contigs.remove(contig)

    def discard(self, contig):
        self.contigs.discard(contig)

    @property
    def ncontigs(self):
        return len(self.contigs)

    @staticmethod
    def getbreadth(contigs):
        "This calculates the total number of bases covered at least 1x in ANY Genome."
        bysubject = _collections.defaultdict(list)
        for contig in contigs:
            bysubject[contig.subject].append(contig)

        breadth = 0
        for contiglist in bysubject.values():
            contiglist.sort(key=lambda contig: contig.start)
            rightmost_end = float('-inf')

            for contig in contiglist:
                breadth += max(contig.end, rightmost_end) - max(contig.start, rightmost_end)
                rightmost_end = max(contig.end, rightmost_end)

        return breadth

    def update_breadth(self):
        "Updates the breadth of the genome"
        self.breadth = self.getbreadth(self.contigs)

    def __repr__(self):
        return 'Genome({}, ncontigs={}, breadth={})'.format(self.name, self.ncontigs, self.breadth)

class Reference:
    """A set of Genomes known to represent the ground truth for binning.
    Instantiate with any iterable of Genomes.

    >>> print(my_genomes)
    [Genome('E. coli'), ncontigs=95, breadth=5012521),
     Genome('Y. pestis'), ncontigs=5, breadth=46588721)]
    >>> Reference(my_genomes)
    Reference(ngenomes=2, ncontigs=100)

    Properties:
    self.genomes: {genome_name: genome} dict
    self.contigs: {contig_name: contig} dict
    self.genomeof: {contig: genome} dict
    self.breadth: Total length of all genomes
    self.ngenomes
    self.ncontigs
    """

    # Instantiate with any iterable of Genomes
    def __init__(self, genomes, taxmaps=list()):
        self.genomes = dict() # genome_name : genome dict
        self.contigs = dict() # contig_name : contig dict
        self.genomeof = dict() # contig : genome dict

        # This is a list of dicts: The first one maps genomename to name of next taxonomic level
        # The second maps name of second level to name of third level etc.
        self.taxmaps = taxmaps

        # Load genomes into list in case it's a one-time iterator
        genomes_backup = list(genomes) if iter(genomes) is genomes else genomes

        # Check that there are no genomes with same name
        if len({genome.name for genome in genomes_backup}) != len(genomes_backup):
            raise ValueError('Multiple genomes with same name not allowed in Reference.')

        for genome in genomes_backup:
            self.add(genome)

        self.breadth = sum(genome.breadth for genome in genomes_backup)

    def load_tax_file(self, line_iterator, comment='#'):
        """Load in a file with N+1 columns, the first being genomename, the next being
        the equivalent taxonomic annotation at different ranks
        Replaces the Reference's taxmaps list."""
        taxmaps = list()
        isempty = True

        for line in line_iterator:
            if line.startswith(comment):
                continue

            genomename, *clades = line[:-1].split('\t')

            if isempty:
                if not clades:
                    raise ValueError('Must have at least two columns')

                for i in clades:
                    taxmaps.append(dict())
                isempty = False

            if genomename in taxmaps[0]:
                raise KeyError("Genome name {} present more than once in taxfile".format(genomename))

            previousrank = genomename
            for nextrank, rankdict in zip(clades, taxmaps):
                existing = rankdict.get(previousrank, nextrank)
                if existing != nextrank:
                    raise KeyError("Rank {} mapped to both {} and {}".format(previousrank, existing, nextrank))

                rankdict[previousrank] = nextrank
                previousrank = nextrank

        self.taxmaps = taxmaps

    @property
    def ngenomes(self):
        return len(self.genomes)

    @property
    def ncontigs(self):
        return len(self.contigs)

    def __repr__(self):
        ranks = len(self.taxmaps) + 1
        return 'Reference(ngenomes={}, ncontigs={}, ranks={})'.format(self.ngenomes, self.ncontigs, ranks)

    @staticmethod
    def _parse_subject_line(line):
        "Returns contig, genome_name from a reference file line with subjects"
        contig_name, genome_name, subject, start, end = line[:-1].split('\t')
        start = int(start)
        end = int(end) + 1 # semi-open interval used in internals, like range()
        contig = Contig(contig_name, subject, start, end)
        return contig, genome_name

    @staticmethod
    def _parse_subjectless_line(line):
        "Returns contig, genome_name from a reference file line without subjects"
        contig_name, genome_name, length = line[:-1].split('\t')
        length = int(length)
        contig = Contig.subjectless(contig_name, length)
        return contig, genome_name

    @classmethod
    def _parse_file(cls, filehandle, subjectless=False):
        "Returns a list of genomes from a reference file"
        function = cls._parse_subjectless_line if subjectless else cls._parse_subject_line

        genomes = dict()
        for line in filehandle:
            # Skip comments
            if line.startswith('#'):
                continue

            contig, genome_name = function(line)
            genome = genomes.get(genome_name)
            if genome is None:
                genome = Genome(genome_name)
                genomes[genome_name] = genome
            genome.add(contig)

        # Update all genomes
        genomes = list(genomes.values())
        for genome in genomes:
            genome.update_breadth()

        return genomes

    @classmethod
    def from_file(cls, filehandle, subjectless=False):
        """Instantiate a Reference from an open filehandle.
        "subjectless" refers to the style of reference file: If true, assumes columns are
        [contig_name, genome_name, contig_length]. If false, assume
        [contig_name, genome_name, subject_name, mapping_start, mapping_end]

        >>> with open('my_reference.tsv') as filehandle:
            Reference.from_file(filehandle)
        Reference(ngenomes=2, ncontigs=100)
        """

        genomes = cls._parse_file(filehandle, subjectless=subjectless)
        return cls(genomes)

    def add(self, genome):
        "Adds a genome to this Reference. If already present, do nothing."
        if genome.name not in self.genomes:
            self.genomes[genome.name] = genome
            for contig in genome.contigs:
                if contig.name in self.contigs:
                    raise KeyError("Contig name '{}' multiple times in Reference.".format(contig.name))

                self.contigs[contig.name] = contig
                self.genomeof[contig] = genome

    def remove(self, genome):
        "Removes a genome from this Reference, raising an error if it is not present."
        del self.genomes[genome.name]

        for contig in genome.contigs:
            del self.contigs[contig.name]
            del self.genomeof[contig]

    def discard(self, genome):
        "Remove a genome if it is present, else do nothing."
        if genome.name in self.genomes:
            self.remove(genome)

class Binning:
    """The result of a set of clusters applied to a Reference.
    >>> ref
    (Reference(ngenomes=2, ncontigs=5)
    >>> b = Binning({'bin1': {contig1, contig2}, 'bin2': {contig3, contig4}}, ref)
    Binning(4/5 contigs, ReferenceID=0x7fe908180be0)
    >>> b[0.5, 0.9] # num. genomes 0.5 recall, 0.9 precision
    1

    Init arguments:
    ----------- Required ---------
    contigsof:     Dict of clusters, each sequence present in the Reference
    reference:     Associated Reference object
    ----------- Optional ---------
    recalls:       Iterable of minimum recall thresholds
    precisions:    Iterable of minimum precision thresholds
    checkpresence: Whether to raise an error if a sequence if not present in Reference
    disjoint:      Whether to raise an error if a sequence is in multiple bins
    binsplit_separator: Split bins according to prefix before this separator in seq name
    minsize:       Minimum sum of sequence lengths in a bin to not be ignored
    mincontigs:    Minimum number of sequence in a bin to not be ignored

    Properties:
    self.reference:       Reference object of this benchmark
    self.recalls:         Sorted tuple of recall thresholds
    self.precisions:      Sorted tuple of precision thresholds
    self.nbins:           Number of bins
    self.ncontigs:        Number of binned contigs
    self.contigsof:       {bin_name: {contig set}}
    self.binof:           {contig: bin_name(s)}, val is str or set
    self.breadthof:       {bin_name: breadth}
    self.intersectionsof: {genome: {bin:_name: intersection}}
    self.breadth:         Total breadth of all bins
    self.counters:        List of (rec, prec) Counters of genomes for each taxonomic rank
    """
    _DEFAULTRECALLS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    _DEFAULTPRECISIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    @property
    def nbins(self):
        return len(self.contigsof)

    @property
    def ncontigs(self):
        return len(self.binof)

    def _iter_intersections(self, genome):
        """Given a genome, return a generator of (bin_name, intersection) for
        all binning bins with a nonzero recall and precision.
        """
        # Get set of all binning bin names with contigs from that genome
        bin_names = set()
        for contig in genome.contigs:
            bin_name = self.binof.get(contig)
            if bin_name is None:
                continue
            elif isinstance(bin_name, set):
                bin_names.update(bin_name)
            else:
                bin_names.add(bin_name)

        for bin_name in bin_names:
            intersecting_contigs = genome.contigs.intersection(self.contigsof[bin_name])
            intersection = Genome.getbreadth(intersecting_contigs)
            yield bin_name, intersection

    def confusion_matrix(self, genome, bin_name):
        "Given a genome and a binname, returns TP, TN, FP, FN"

        true_positives = self.intersectionsof[genome].get(bin_name, 0)
        false_positives = self.breadthof[bin_name] - true_positives
        false_negatives = genome.breadth - true_positives
        true_negatives = self.reference.breadth - false_negatives - false_positives + true_positives

        return true_positives, true_negatives, false_positives, false_negatives

    def mcc(self, genome, bin_name):
        "Calculate Matthew's correlation coefficient between a genome and a bin."

        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        mcc_num = tp * tn - fp * fn
        mcc_den = (tp + fp) * (tp + fn)
        mcc_den *= (tn + fp) * (tn + fn)
        return 0 if mcc_den == 0 else mcc_num / _sqrt(mcc_den)

    def f1(self, genome, bin_name):
        "Calculate F1 score between genome and a bin"

        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        return 2*tp / (2*tp + fp + fn)

    def _getseen(self, recprecof):
        """Make a {genome: isseen} dict, where isseen is a boolean vector
        (implemented as an integer), 1 if a genome is seen at that recall, prec level,
        0 otherwise
        """
        isseen = dict()
        for genome, _dict in recprecof.items():
            seen = 0
            for binname, (recall, precision) in _dict.items():
                for i, (min_recall, min_precision) in enumerate(_product(self.recalls, self.precisions)):
                    if recall < min_recall:
                        break

                    if precision >= min_precision:
                        seen |= 1 << i
            isseen[genome] = seen
        return isseen

    def _accumulate(self, seen, counts):
        "Given a 'seen' dict, make a dict of counts at each threshold level"
        nsums = (len(self.recalls) * len(self.precisions))
        sums = [0] * nsums
        for v in seen.values():
            for i in range(nsums):
                sums[i] += (v >> i) & 1 == 1

        for i, (recall, precision) in enumerate(_product(self.recalls, self.precisions)):
            counts[(recall, precision)] = sums[i]

    def _get_prec_rec_dict(self):
        recprecof = _collections.defaultdict(dict)
        for genome, intersectiondict in self.intersectionsof.items():
            for binname in intersectiondict:
                tp, tn, fp, fn = self.confusion_matrix(genome, binname)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                recprecof[genome.name][binname] = (recall, precision)

        return recprecof

    def _getcounts(self):
        # One count per rank (+1 for inclusive "genome" rank)
        counts = [_collections.Counter() for i in range(len(self.reference.taxmaps) + 1)]
        recprecof = self._get_prec_rec_dict()
        seen = self._getseen(recprecof)
        # Calculate counts for each taxonomic level
        for counter, taxmap in zip(counts, self.reference.taxmaps):
            self._accumulate(seen, counter)
            newseen = dict()
            for clade, v in seen.items():
                newclade = taxmap[clade]
                newseen[newclade] = newseen.get(newclade, 0) | v
            seen = newseen

        self._accumulate(seen, counts[-1])

        return counts

    def __init__(self, contigsof, reference, recalls=_DEFAULTRECALLS,
              precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=True,
              binsplit_separator=None, minsize=None, mincontigs=None):
        # See class docstring for explanation of arguments

        # Checkpresence enforces that each contig in a bin is also in the reference,
        # disjoint enforces that each contig is only present in one bin.
        if not isinstance(reference, Reference):
            raise ValueError('reference must be a Reference')

        self.precisions = tuple(sorted(precisions))
        self.recalls = tuple(sorted(recalls))
        self.reference = reference

        self.contigsof = dict() # bin_name: {contigs} dict
        self.binof = dict() # contig: bin_name or {bin_names} dict
        self.breadthof = dict() # bin_name: int dict
        self._parse_bins(contigsof, checkpresence, disjoint, binsplit_separator, minsize, mincontigs)
        self.breadth = sum(self.breadthof.values())

        # intersectionsof[genome] = {genome: {binname: tp, binname: tp ... }}
        # for all bins with nonzero true positives
        intersectionsof = dict()
        for genome in reference.genomes.values():
            intersectionsof[genome] = dict()
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof

        # Set counts
        self.counters = self._getcounts()

    def _parse_bins(self, contigsof, checkpresence, disjoint, binsplit_separator, minsize, mincontigs):
        "Fills self.binof, self.contigsof and self.breadthof during instantiation"

        if binsplit_separator is not None:
            contigsof = _vambtools.binsplit(contigsof, binsplit_separator)

        if minsize is not None or mincontigs is not None:
            minsize = 1 if minsize is None else minsize
            mincontigs = 1 if mincontigs is None else mincontigs
            contigsof = filter_clusters(contigsof, self.reference, minsize, mincontigs, checkpresence=checkpresence)

        for bin_name, contig_names in contigsof.items():
            contigset = set()
            # This stores each contig by their true genome name.
            contigsof_genome = _collections.defaultdict(list)

            for contig_name in contig_names:
                contig = self.reference.contigs.get(contig_name)

                # Check that the contig is in the reference
                if contig is None:
                    if checkpresence:
                        raise KeyError('Contig {} not in reference.'.format(contig_name))
                    else:
                        continue

                # Check that contig is only present one time in input
                existing = self.binof.get(contig)
                if existing is None:
                    self.binof[contig] = bin_name
                else:
                    if disjoint:
                        raise KeyError('Contig {} found in multiple bins'.format(contig_name))
                    elif isinstance(existing, str):
                        self.binof[contig] = {existing, bin_name}
                    else:
                        self.binof[contig].add(bin_name)

                contigset.add(contig)
                genome = self.reference.genomeof[self.reference.contigs[contig_name]]
                contigsof_genome[genome.name].append(contig)

            self.contigsof[bin_name] = contigset

            # Now calculate breadth of bin. This is the sum of the number of covered
            # base pairs for all the genomes present in the bin.
            breadth = 0
            for contigs in contigsof_genome.values():
                breadth += Genome.getbreadth(contigs)
            self.breadthof[bin_name] = breadth

    @classmethod
    def from_file(cls, filehandle, reference, recalls=_DEFAULTRECALLS,
                  precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=True,
                  binsplit_separator=None, minsize=None, mincontigs=None):
        contigsof = dict()
        for line in filehandle:
            if line.startswith('#'):
                continue

            line = line.rstrip()
            bin_name, tab, contig_name = line.partition('\t')

            if bin_name not in contigsof:
                contigsof[bin_name] = [contig_name]
            else:
                contigsof[bin_name].append(contig_name)

        return cls(contigsof, reference, recalls, precisions, checkpresence, disjoint,
                   binsplit_separator, minsize, mincontigs)

    def print_matrix(self, rank, file=_sys.stdout):
        """Prints the recall/precision number of bins to STDOUT."""

        if rank >= len(self.counters):
            raise IndexError("Taxonomic rank out of range")

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self.counters[rank][(min_recall, min_precision)] for min_recall in self.recalls]
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)

    def __repr__(self):
        fields = (self.ncontigs, self.reference.ncontigs, hex(id(self.reference)))
        return 'Binning({}/{} contigs, ReferenceID={})'.format(*fields)

    def summary(self, precision=0.9, recalls=None):
        if recalls is None:
            recalls = self.recalls
        return [[counter[(recall, precision)] for recall in recalls] for counter in self.counters]

def filter_clusters(clusters, reference, minsize, mincontigs, checkpresence=True):
    """Creates a shallow copy of clusters, but without any clusters with a total size
    smaller than minsize, or fewer contigs than mincontigs.
    If checkpresence is True, raise error if a contig is not present in reference, else
    ignores it when counting cluster size.
    """

    filtered = dict()
    for binname, contignames in clusters.items():
        if len(contignames) < mincontigs:
            continue

        size = 0
        for contigname in contignames:
            contig = reference.contigs.get(contigname)

            if contig is not None:
                size += len(contig)
            elif checkpresence:
                raise KeyError('Contigname {} not in reference'.format(contigname))
            else:
                pass

        if size >= minsize:
            filtered[binname] = contignames.copy()

    return filtered
