"""
The approach to benchmarking is this:

The reference is given as a list of contignames, each with a genome and a start
and end position, for where the contig is located on the genome.

For each genome, the length is calculated as the number of basepairs
covered by at least one contig. Thus, any uncovered positions are not counted.

The observed is created with a set of bins, each bin being a set of contignames.
The length of a bin is the number of basepairs covered by the contigs, given that
their location is given by the Reference. Thus, a bin can contain contigs mapping
to different genomes, in which case the length is the sum of the bases covered in
each genome.

Each bin (in the observed) is then compared to each genome (in the reference).
For a bin/genome pair, the "intersection" is the number of basepairs in the
genome covered by both a reference contig and an observed contig.
The recall is thus intersection/genome_length, and precision is
intersection/bin_length.

The benchmark consists of counting how many genomes have at least one observed
bin which surpasses a given recall and threshold. Thus, an observed bin can be
counted twice, but not a genome.
"""

import collections as _collections
from itertools import product as _product
import sys as _sys
from math import sqrt as _sqrt

Contig = _collections.namedtuple('Contig', ['name', 'start', 'end'])

class Genome:
    """A set of clusters known to come from the same organism.
    >>> genome = Genome('E. coli')
    >>> genome.add(Contig)
    """
    __slots__ = ['name', 'contigs', 'length']

    def __init__(self, name):
        self.name = name
        self.contigs = set()
        self.length = 0

    def add(self, contig):
        self.contigs.add(contig)

    @property
    def ncontigs(self):
        return len(self.contigs)

    @staticmethod
    def getlength(contigs):
        "This calculates the total number of bases covered at least 1x."

        sorted_contigs = sorted(contigs, key=lambda contig: contig.start)
        rightmost_end = float('-inf')
        length = 0
        for contig_name, start, end in sorted_contigs:
            length += max(end, rightmost_end) - max(start, rightmost_end)
            rightmost_end = max(end, rightmost_end)
        return length

    def update(self):
        "This updates the length attribute of a Genome"
        self.length = self.getlength(self.contigs)

    def __repr__(self):
        return 'Genome({}, ncontigs={}, length={})'.format(self.name, self.ncontigs, self.length)

class Reference:
    """A set of Genomes known to represent the ground truth for binning.
    >>> print(my_genomes)
    [Genome('E. coli'), ncontigs=95, length=5012521),
     Genome('Y. pestis'), ncontigs=5, length=46588721)]
    >>> Reference(my_genomes)
    Reference(ngenomes=2, ncontigs=100)

    Properties:
    self.genome: {genome_name: genome} dict
    self.contigs: {contig_name: contig} dict
    self.genomeof: {contig_name: genome} dict
    self.length: Total length of all genomes
    self.ngenomes
    self.ncontigs
    """

    # Instantiate with any iterable of Genomes
    def __init__(self, genomes):
        self.genomes = dict() # genome_name : genome dict
        self.contigs = dict() # contig_name : contig dict
        self.genomeof = dict() # contig_name : genome dict
        self.length = 0

        for genome in genomes:
            self.length += genome.length
            self.genomes[genome.name] = genome
            for contig in genome.contigs:
                if contig.name in self.contigs:
                    raise KeyError("Contig name '{}' multiple times in Reference.".format(contig.name))

                self.contigs[contig.name] = contig
                self.genomeof[contig.name] = genome

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
        contig_names = set()

        for line in filehandle:
            # Skip comments
            if line.startswith('#'):
                continue

            contig_name, genome_name, start, end = line[:-1].split('\t')
            contig_names.add(contig_name)

            start = int(start)
            end = int(end) + 1 # semi-open interval used in internals, like range()
            contig = Contig(contig_name, start, end)

            genome = genomes.get(genome_name)
            # Create a new Genome if we have not seen it before
            if genome is None:
                genome = Genome(genome_name)
                genomes[genome_name] = genome
            genome.add(contig)

        # Update lengths of the genomes
        for genome in genomes.values():
            genome.update()

        # Construct instance
        return cls(genomes.values())

class Observed:
    """A set of bins, each being a set of contigs, as produced by a binner.
    This must be instantiated with a `contigsof` dict and a Reference.
    A `contigsof` dict is a {bin_name: {contig_names} ...} dict

    Properties:
    self.contigsof: {bin_name: {contigs ... }} dict
    self.binof: {contig: bin_name} dict
    self.bin_length = {bin_name: length} dict
    self.length: length of all bins combined
    self.nbins
    self.ncontigs
    """

    # Instantiate with a {bin_name: {contig_names}} dict and Reference
    def __init__(self, contigsof, reference, checkpresence=True):
        self.contigsof = dict() # bin_name: {contigs} dict
        self.binof = dict() # contig: bin_name dict
        self.bin_length = dict() # bin_name: int dict
        self.length = 0

        for bin_name, contig_names in contigsof.items():
            contigset = set()
            # This stores each contig by their true genome name.
            contigsof_genome = _collections.defaultdict(list)

            for contig_name in contig_names:
                contig = reference.contigs.get(contig_name)

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
                genome = reference.genomeof[contig_name]
                contigsof_genome[genome.name].append(contig)

            self.contigsof[bin_name] = contigset

            # Now calculate length of bin. This is the sum of the number of covered
            # base pairs for all the genomes present in the bin.
            bin_length = 0
            for contigs in contigsof_genome.values():
                bin_length += Genome.getlength(contigs)
            self.bin_length[bin_name] = bin_length
            self.length += bin_length

    @property
    def nbins(self):
        return len(self.contigsof)

    @property
    def ncontigs(self):
        return len(self.binof)

    def __repr__(self):
        return 'Observed(nbins={}, ncontigs={})'.format(self.nbins, self.ncontigs)

    @classmethod
    def from_file(cls, filehandle, reference):
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

        return cls(contigsof, reference)

class BenchMarkResult:
    """The result of an Observed applied to a Reference.
    >>> ref, obs
    (Reference(ngenomes=2, ncontigs=5), Observed(nbins=2, ncontigs=4))
    >>> benchmark = BenchMarkResult(ref, obs)
    BenchMarkResult(referenceID=0x7fe908180898, observedID=0x7fe908180be0)
    >>> benchmark[0.6, 0.9] # 0.6 recall, 0.9 precision
    1

    Properties:
    self.reference: Reference object of this benchmark
    self.observed: Observed object of this benchmark
    self.recalls: sorted tuple of recall thresholds
    self.precisions: sorted tuple of precision thresholds
    self.intersectionsof: {genome: {bin_name: intersection,  ...}} dict
    """
    _DEFAULTRECALLS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    _DEFAULTPRECISIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def _iter_intersections(self, genome):
        """Given a genome, return a generator of (bin_name, intersection) for
        all observed bins with a nonzero recall and precision.
        """
        # Get set of all observed bin names with contigs from that genome
        bin_names = {self.observed.binof.get(contig) for contig in genome.contigs}
        bin_names.discard(None)

        for bin_name in bin_names:
            intersecting_contigs = genome.contigs.intersection(self.observed.contigsof[bin_name])
            intersection = Genome.getlength(intersecting_contigs)
            yield bin_name, intersection

    def mcc(self, genome):
        "Get the best MCC for the given genome"
        maxmcc = 0 # default value of 0 for un-reconstructed genomes
        for bin_name, true_pos in self.intersectionsof[genome].items():
            false_pos = self.observed.bin_length[bin_name] - true_pos
            false_neg = genome.length - true_pos
            true_neg = self.reference.length - true_pos - false_neg - false_pos
            mcc_num = true_pos * true_neg - false_pos * false_neg
            mcc_den = (true_pos + false_pos) * (true_pos + false_neg)
            mcc_den *= (true_neg + false_pos) * (true_neg + false_neg)
            mcc = 0 if mcc_den == 0 else mcc_num / _sqrt(mcc_den)
            maxmcc = max(mcc, maxmcc)

        return maxmcc

    def f1(self, genome):
        "Get the best F1 score for the given genome"
        maxf1 = 0
        for bin_name, true_pos in self.intersectionsof[genome].items():
            false_pos = self.observed.bin_length[bin_name] - true_pos
            false_neg = genome.length - true_pos
            f1 = 2*true_pos / (2*true_pos + false_pos + false_neg)
            maxf1 = max(f1, maxf1)

        return maxf1


    def __init__(self, reference, observed, recalls=_DEFAULTRECALLS, precisions=_DEFAULTPRECISIONS):
        if not isinstance(reference, Reference):
            raise ValueError('reference must be a Reference')

        if not isinstance(observed, Observed):
            raise ValueError('observed must be a Observed')

        self.precisions = tuple(sorted(precisions))
        self.recalls = tuple(sorted(recalls))
        self.reference = reference
        self.observed = observed
        # counts[r,p] is number of genomes w. recall >= r, precision >= p
        counts = _collections.Counter()
        # intersectionsof[genome_name] = [(bin_name, recall, precision) ... ]
        # for all bins with nonzero recall and precision
        intersectionsof = _collections.defaultdict(dict)

        # Calculate intersectionsof
        for genome in reference.genomes.values():
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof

        # Calculate MCC score
        mccs = list()
        for genome in reference.genomes.values():
            mccs.append(self.mcc(genome))
        self.mean_mcc = 0 if len(mccs) == 0 else sum(mccs) / len(mccs)
        del mccs

        # Calculate F1 scores
        f1s = list()
        for genome in reference.genomes.values():
            f1s.append(self.f1(genome))
        self.mean_f1 = 0 if len(f1s) == 0 else sum(f1s) / len(f1s)
        del f1s

        # Calculate number of genomes above threshols
        # This could use some optimization
        for genome in reference.genomes.values():
            # Keep track of whether the genome has been found at those thresholds
            found = [False]*(len(self.recalls) * len(self.precisions))

            thresholds = list() # save to list to cache them for multiple iterations
            for bin_name, intersection in self.intersectionsof[genome].items():
                recall = intersection / genome.length
                precision = intersection / observed.bin_length[bin_name]
                thresholds.append((recall, precision))

            for i, (min_recall, min_precision) in enumerate(_product(self.recalls, self.precisions)):
                for recall, precision in thresholds:
                    if recall >= min_recall and precision >= min_precision:
                        found[i] = True
            for i, thresholds in enumerate(_product(self.recalls, self.precisions)):
                if found[i]:
                    counts[thresholds] += 1

        self._counts = counts

    def print_matrix(self, file=_sys.stdout):
        """Prints the recall/precision number of bins to STDOUT."""

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self._counts[(min_recall, min_precision)] for min_recall in self.recalls]
            row.sort(reverse=True)
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)

    def __repr__(self):
        refhex = hex(id(self.reference))
        obshex = hex(id(self.observed))
        return 'BenchMarkResult(referenceID={}, observedID={})'.format(refhex, obshex)

    def __getitem__(self, key):
        recall, precision = key
        if recall not in self.recalls or precision not in self.precisions:
            raise KeyError('Not initialized with that recall, precision pair')

        return self._counts(thresholds)
