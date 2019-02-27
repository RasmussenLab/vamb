# The approach to benchmarking is this:
#
# Most fundamentally, we have a Contig. A contig is associated with a subject, where
# it comes from and aligns to. It thus has a start and end position in that subject.
# A set of contigs has a breadth. This is the number of basepairs covered by the
# contigs.
#
# A Genome is simply a collection of contigs. A set of Genomes constitute a Reference.
#
# A Binning is a collection of sets of contigs, each set representing a bin. It is
# required that all contigs in a Binning comes from Genomes in the same Reference.
# A bin's breadth is the total basepairs covered by the contigs of a bin, but since
# the contigs of a bin may come from multiple Genomes, the bin breadth is the sum
# of these per-Genome breadths.
# The breadth of a Binning is the sum of breadths of each bin. This implies that the
# breadth of a Binning can exceed the total length of all the underlying Genomes.
#
# When creating a Binning, for each genome/bin pair, the following base statistics
# are calculated:
# * True positives (TP) is the breadth of the Genome covered by contigs in the bin
# * False positives (FP) is the bin breadth minus TP
# * False negatives (FN) is the Genome length minus TP
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

Contig = _collections.namedtuple('Contig', ['name', 'subject', 'start', 'end'])

class Genome:
    """A set of clusters known to come from the same organism.
    >>> genome = Genome('E. coli')
    >>> genome.add(contig)
    """
    __slots__ = ['name', 'breadth', 'contigs']

    def __init__(self, name):
        self.name = name
        self.contigs = set()

    def add(self, contig):
        self.contigs.add(contig)

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

            for contig_name, subject, start, end in contiglist:
                breadth += max(end, rightmost_end) - max(start, rightmost_end)
                rightmost_end = max(end, rightmost_end)

        return breadth

    def update(self):
        self.breadth = self.getbreadth(self.contigs)

    def __repr__(self):
        return 'Genome({}, ncontigs={}, breadth={})'.format(self.name, self.ncontigs, self.breadth)

class Reference:
    """A set of Genomes known to represent the ground truth for binning.
    >>> print(my_genomes)
    [Genome('E. coli'), ncontigs=95, breadth=5012521),
     Genome('Y. pestis'), ncontigs=5, breadth=46588721)]
    >>> Reference(my_genomes)
    Reference(ngenomes=2, ncontigs=100)

    Properties:
    self.genome: {genome_name: genome} dict
    self.contigs: {contig_name: contig} dict
    self.genomeof: {contig_name: genome} dict
    self.ngenomes
    self.ncontigs
    """

    # Instantiate with any iterable of Genomes
    def __init__(self, genomes):
        self.genomes = dict() # genome_name : genome dict
        self.contigs = dict() # contig_name : contig dict
        self.genomeof = dict() # contig : genome dict

        for genome in genomes:
            self.genomes[genome.name] = genome
            for contig in genome.contigs:
                if contig.name in self.contigs:
                    raise KeyError("Contig name '{}' multiple times in Reference.".format(contig.name))

                self.contigs[contig.name] = contig
                self.genomeof[contig] = genome

    @property
    def ngenomes(self):
        return len(self.genomes)

    @property
    def ncontigs(self):
        return len(self.genomeof)

    def __repr__(self):
        return 'Reference(ngenomes={}, ncontigs={})'.format(self.ngenomes, self.ncontigs)

    @classmethod
    def from_file(cls, filehandle):
        """Instantiate a Reference from an open filehandle
        >>> with open('my_reference.tsv') as filehandle:
            Reference.from_file(filehandle)
        Reference(ngenomes=2, ncontigs=100)
        """

        genomes = dict()
        for line in filehandle:
            # Skip comments
            if line.startswith('#'):
                continue

            contig_name, genome_name, subject, start, end = line[:-1].split('\t')
            start = int(start)
            end = int(end) + 1 # semi-open interval used in internals, like range()
            contig = Contig(contig_name, subject, start, end)
            genome = genomes.get(genome_name)
            # Create a new Genome if we have not seen it before
            if genome is None:
                genome = Genome(genome_name)
                genomes[genome_name] = genome
            genome.add(contig)

        # Update all genomes
        for genome in genomes.values():
            genome.update()

        # Construct instance
        return cls(genomes.values())

class Binning:
    """The result of an Binning applied to a Reference.
    >>> ref
    (Reference(ngenomes=2, ncontigs=5)
    >>> b = Binning({'bin1': {contig1, contig2}, 'bin2': {contig3, contig4}}, ref)
    Binning(4/5 contigs, ReferenceID=0x7fe908180be0)
    >>> b[(0.5, 0.9)] # num. genomes 0.5 recall, 0.9 precision
    1

    Properties:
    self.reference:       Reference object of this benchmark
    self.recalls:         Sorted tuple of recall thresholds
    self.precisions:      Sorted tuple of precision thresholds
    self.nbins:           Number of bins
    self.ncontigs:        Number of binned contigs
    self.contigsof:       {bin_name: {contig set}}
    self.binof:           {contig: bin_name}
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
        bin_names = {self.binof.get(contig) for contig in genome.contigs}
        bin_names.discard(None)

        for bin_name in bin_names:
            intersecting_contigs = genome.contigs.intersection(self.contigsof[bin_name])
            intersection = Genome.getbreadth(intersecting_contigs)
            yield bin_name, intersection

    def confusion_matrix(self, genome, bin_name):
        true_positives = self.intersectionsof[genome].get(bin_name, 0)
        false_positives = self.breadthof[bin_name] - true_positives
        false_negatives = genome.breadth - true_positives
        true_negatives = self.breadth - true_positives - false_negatives - false_positives

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

    def _parse_bins(self, contigsof, checkpresence):
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
                if contig in self.binof:
                    raise KeyError('Contig {} more than once in contigsof'.format(contig))

                contigset.add(contig)
                self.binof[contig] = bin_name
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
              precisions=_DEFAULTPRECISIONS, checkpresence=True):
        if not isinstance(reference, Reference):
            raise ValueError('reference must be a Reference')

        self.precisions = tuple(sorted(precisions))
        self.recalls = tuple(sorted(recalls))
        self.reference = reference

        self.contigsof = dict() # bin_name: {contigs} dict
        self.binof = dict() # contig: bin_name dict
        self.breadthof = dict() # bin_name: int dict
        self._parse_bins(contigsof, checkpresence)
        self.breadth = sum(self.breadthof.values())

        # counts[r,p] is number of genomes w. recall >= r, precision >= p
        counts = _collections.Counter()
        # intersectionsof[genome_name] = [(bin_name, recall, precision) ... ]
        # for all bins with nonzero recall and precision
        intersectionsof = dict()

        # Calculate intersectionsof
        for genome in reference.genomes.values():
            intersectionsof[genome] = dict()
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof

        # Calculate MCC score
        mccs = [max(self.mcc(genome, binname) for binname in self.intersectionsof[genome])
                for genome in reference.genomes.values()]
        self.mean_mcc = 0 if len(mccs) == 0 else sum(mccs) / len(mccs)
        del mccs

        # Calculate F1 scores
        f1s = [max(self.f1(genome, binname) for binname in self.intersectionsof[genome])
                for genome in reference.genomes.values()]
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
                  precisions=_DEFAULTPRECISIONS, checkpresence=True):
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

        return cls(contigsof, reference, recalls, precisions, checkpresence)

    def print_matrix(self, file=_sys.stdout):
        """Prints the recall/precision number of bins to STDOUT."""

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self._counts[(min_recall, min_precision)] for min_recall in self.recalls]
            row.sort(reverse=True)
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)

    def __repr__(self):
        fields = (self.ncontigs, self.reference.ncontigs, hex(id(self.reference)))
        return 'Binning({}/{} contigs, ReferenceID={})'.format(*fields)

    def __getitem__(self, key):
        recall, precision = key
        if recall not in self.recalls or precision not in self.precisions:
            raise KeyError('Not initialized with that recall, precision pair')

        return self._counts.get(key, 0)
