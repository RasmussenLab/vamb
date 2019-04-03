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
# * True negatives (TN) is the Binning breadth minus TP+FP+FN
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
>>> bins = Binning.from_file(open_clusters_file_handle, ref)
>>> bins.print_matrix()
"""

import collections as _collections
from itertools import product as _product
import sys as _sys
from math import sqrt as _sqrt

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

    @property
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
    def __init__(self, genomes):
        self.genomes = dict() # genome_name : genome dict
        self.contigs = dict() # contig_name : contig dict
        self.genomeof = dict() # contig : genome dict

        # Load genomes into list in case it's a one-time iterator
        genomes_backup = list(genomes) if iter(genomes) is genomes else genomes

        # Check that there are no genomes with same name
        if len({genome.name for genome in genomes_backup}) != len(genomes_backup):
            raise ValueError('Multiple genomes with same name not allowed in Reference.')

        for genome in genomes_backup:
            self.add(genome)

        self.breadth = sum(genome.breadth for genome in genomes_backup)

    @property
    def ngenomes(self):
        return len(self.genomes)

    @property
    def ncontigs(self):
        return len(self.contigs)

    def __repr__(self):
        return 'Reference(ngenomes={}, ncontigs={})'.format(self.ngenomes, self.ncontigs)

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
        function =  cls._parse_subjectless_line if subjectless else cls._parse_subject_line

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

    def remove_small_genomes(self, *, minbreadth):
        "Removes any genomes smaller than the minimum breadth"
        # Copy this so we can iterate over the dict while removing values
        genomes_copy = list(self.genomes.values())
        for genome in genomes_copy:
            if genome.breadth < minbreadth:
                self.remove(genome)

class Binning:
    """The result of an Binning applied to a Reference.
    >>> ref
    (Reference(ngenomes=2, ncontigs=5)
    >>> b = Binning({'bin1': {contig1, contig2}, 'bin2': {contig3, contig4}}, ref)
    Binning(4/5 contigs, ReferenceID=0x7fe908180be0)
    >>> b[0.5, 0.9] # num. genomes 0.5 recall, 0.9 precision
    1

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
    self.mean_f1:         Mean F1 score among all Genomes
    self.mean_mcc         Mean Matthew's correlation coef. among all Genomes
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
            elif isinstance(bin_name, str):
                bin_names.add(bin_name)
            else:
                bin_names.update(bin_name)

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
        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        mcc_num = tp * tn - fp * fn
        mcc_den = (tp + fp) * (tp + fn)
        mcc_den *= (tn + fp) * (tn + fn)
        return 0 if mcc_den == 0 else mcc_num / _sqrt(mcc_den)

    def f1(self, genome, bin_name):
        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        return 2*tp / (2*tp + fp + fn)

    def _parse_bins(self, contigsof, checkpresence, disjoint):
        "Fills self.binof, self.contigsof and self.breadthof during instantiation"

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

    def __init__(self, contigsof, reference, recalls=_DEFAULTRECALLS,
              precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=True):
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
        self._parse_bins(contigsof, checkpresence, disjoint)
        self.breadth = sum(self.breadthof.values())

        # counts[r,p] is number of genomes w. recall >= r, precision >= p
        counts = _collections.Counter()
        # intersectionsof[genome] = {genome: {binname: tp, binname: tp ... }}
        # for all bins with nonzero true positives
        intersectionsof = dict()

        # Calculate intersectionsof
        for genome in reference.genomes.values():
            intersectionsof[genome] = dict()
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof

        # Calculate MCC score
        mccs = [max((self.mcc(genome, binname) for binname in self.intersectionsof[genome]),
                default=0.0) for genome in reference.genomes.values()]
        self.mean_mcc = 0 if len(mccs) == 0 else sum(mccs) / len(mccs)
        del mccs

        # Calculate F1 scores
        f1s = [max((self.f1(genome, binname) for binname in self.intersectionsof[genome]),
                default=0.0) for genome in reference.genomes.values()]
        self.mean_f1 = 0 if len(f1s) == 0 else sum(f1s) / len(f1s)
        del f1s

        # Calculate number of genomes above threshols
        # This could use some optimization
        for genome in reference.genomes.values():
            # Keep track of whether the genome has been found at those thresholds
            found = [False]*(len(self.recalls) * len(self.precisions))

            thresholds = list() # save to list to cache them for multiple iterations
            for bin_name in self.intersectionsof[genome]:
                tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                thresholds.append((recall, precision))

            for i, (min_recall, min_precision) in enumerate(_product(self.recalls, self.precisions)):
                for recall, precision in thresholds:
                    if recall >= min_recall and precision >= min_precision:
                        found[i] = True
            for i, thresholds in enumerate(_product(self.recalls, self.precisions)):
                if found[i]:
                    counts[thresholds] += 1

        self._counts = counts

    @classmethod
    def from_file(cls, filehandle, reference, recalls=_DEFAULTRECALLS,
                  precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=False):
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

        return cls(contigsof, reference, recalls, precisions, checkpresence, disjoint)

    def print_matrix(self, file=_sys.stdout):
        """Prints the recall/precision number of bins to STDOUT."""

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self._counts[(min_recall, min_precision)] for min_recall in self.recalls]
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)

    def __repr__(self):
        fields = (self.ncontigs, self.reference.ncontigs, hex(id(self.reference)))
        return 'Binning({}/{} contigs, ReferenceID={})'.format(*fields)

    def __getitem__(self, key):
        recall, precision = key
        if recall not in self.recalls or precision not in self.precisions:
            raise KeyError('Not initialized with recall={}, precision={}'.format(recall, precision))

        return self._counts.get(key, 0)

def filter_clusters(clusters, reference, minsize, checkpresence=True):
    filtered = dict()
    for binname, contignames in clusters.items():
        size = 0
        for contigname in contignames:
            contig = reference.contigs.get(contigname)

            if contig is not None:
                size += len(contig)
            elif checkpresence:
                raise KeyError('Contig {} not in reference'.format(contigname))
            else:
                pass

        if size >= minsize:
            filtered[binname] = contignames

    return filtered
