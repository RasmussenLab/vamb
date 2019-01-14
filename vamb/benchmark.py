
__doc__ = """Benchmarks observed clusters given a reference.

Contains the objects Reference, Observed and BenchMarkResult.

type help(BenchMarkResult) to see what gets measured.

You can instantiate References and Observed from {cluster: set(contigs)} dicts
or from files (shown below)

>>> with open(reference_path) as reference_file:
>>>    reference = Reference.fromfile(reference_file)
>>>
>>> with open(observed_path) as observed_file
>>>     observed = Observed.fromfile(observed_file, reference)
>>>
>>> benchmarkresult = BenchMarkResult(reference=reference, observed=observed)

Observed file should be tab-sep lines of: cluster, contig
Reference file should be tab-sep lines of: cluster, contig, length

The cluster names need not be the same for obs. and ref. file, but the contig
names do.

NOTE: This version counts the number of reference bins detected at the
different levents of recall and precision.
"""

import sys
import os

from math import sqrt
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

def get_adjusted_rand_score(clusters1, clusters2):
    # First get an integer for each observation, which is the index in the
    # label arrays
    obs_number = dict()
    n = 0
    for d in clusters1, clusters2:
        for cluster in d.values():
            for obs in cluster:
                if obs not in obs_number:
                    obs_number[obs] = n
                    n += 1

    # Make label arrays. -1 represents unclustered observation
    v1, v2 = np.full(n, fill_value=-1), np.full(n, fill_value=-1)

    for d, v in ((clusters1, v1), (clusters2, v2)):
        # Set label of all clustered observations
        for label, cluster in enumerate(d.values()):
            for obs in cluster:
                v[obs_number[obs]] = label

        # Set label all unclustered observations to unique label
        label = len(d)
        for i, arraylabel in enumerate(v):
            if arraylabel == -1:
                v[i] = label
                label += 1

    return adjusted_rand_score(v1, v2)

class Reference:
    """Reference clusters.

    Init with {name: set_of_contig} dict (a) and {contigname: length} dict (b):
    >>> reference = Reference(a, b)

    Init with iterator of tab-sep lines of clustername, contigname, contiglength:
    >>> with open('/path/to/reference.tsv') as line_iterator:
    ...     filtered_lines = filter(str.startswith('HUMAN'), line_iterator)
    ...     reference = Reference.fromfile(filtered_lines) # or with filehandle

    Attributes:
        self.nbins: Number of bins
        self.ncontigs: Number of contigs
        self.contigsof: binname: set(contigsnames) dict
        self.binof: contigname: binname dict
        self.contiglength: contigname: contiglength dict
        self.binlength: binname: sum_of_contiglengths_for_bin dict
    """

    def __init__(self, contigsof, contiglength):
        self.contigsof = contigsof
        self.contiglength = contiglength
        self.ncontigs = len(self.contiglength)
        self.nbins = len(self.contigsof)

        self.binof = dict()
        self.binlength = dict()

        for cluster, contigs in self.contigsof.items():
            for contig in contigs:
                self.binof[contig] = cluster
                length = self.contiglength[contig]
                self.binlength[cluster] = self.binlength.get(cluster, 0) + length

        self.length = sum(self.binlength.values())

    @classmethod
    def fromfile(cls, filehandle):
        """Load a reference with tab-sep lines of: binid, contigname, length"""
        contigsof = defaultdict(set)
        contiglength = dict()

        for line in filehandle:
            stripped = line.rstrip()

            if line == '' or line[0] == '#':
                continue

            try:
                binid, contigname, length = stripped.split('\t')
            except ValueError as error:
                argument = error.args[0]
                if argument.startswith("not enough values to unpack"):
                    raise ValueError(argument + '. Did you pass it a filehandle?')

            length = int(length)
            contigsof[binid].add(contigname)
            contiglength[contigname] = length

        return cls(contigsof, contiglength)

    def flatten_lengths(self):
        for contig in self.contiglength:
            self.contiglength[contig] = 1

        for binid in self.binlength:
            self.binlength[binid] = len(self.contigsof[binid])

class Observed:
    """Observed clusters.

    Init with {name: set_of_contig} dict and Reference object (b):
    >>> observed = Observed(a, b)

    Init with file of tab-sep lines w. clustername, contigname and Reference:
    >>> with open('/path/to/observed.tsv') as line_iterator:
    ...     observed = Observed.fromfile(line_iterator, reference)
    """

    def __init__(self, contigsof, reference):
        """Load observed bins as tab-sep lines of: binind, contigname"""
        self.contigsof = contigsof
        self.ncontigs = sum(len(contigs) for contigs in self.contigsof.values())
        self.nbins = len(self.contigsof)

        self.binof = dict()
        self.binlength = dict()

        for cluster, contigs in self.contigsof.items():
            for contig in contigs:
                self.binof[contig] = cluster

                try:
                    contiglength = reference.contiglength[contig]
                except KeyError:
                    message = 'Contig {} not in reference'.format(contig)
                    raise KeyError(message) from None

                self.binlength[cluster] = self.binlength.get(cluster, 0) + contiglength

    @classmethod
    def fromfile(cls, filehandle, reference):
        """Load observed bins as tab-sep lines of: binind, contigname"""
        contigsof = defaultdict(set)

        for line in filehandle:
            stripped = line.rstrip()

            if line == '' or line[0] == '#':
                continue

            binid, contigname = stripped.split('\t')
            contigsof[binid].add(contigname)

        return cls(contigsof, reference)

    def flatten_lengths(self):
        for binid in self.binlength:
            self.binlength[binid] = len(self.contigsof[binid])



class BenchMarkResult:
    """An object holding some benchmark results:

    Init from Reference object and Observed object using keywords:
    >>> result = BenchmarkResult(reference=reference, observed=observed)

    recall_weight is the weight of recall relative to precision; to weigh
    precision higher than recall, use a value between 0 and 1.

    Get number of references found at recall, precision:
    >>> result[(recall, precision)]
    Get number of references found at recall or precision
    >>> result.atrecall/atprecision(recall/precision)
    Print number of references at all recalls and precisions:
    >>> result.printmatrix()

    Attributes:
        self.nreferencebins: Number of reference bins
        self.nobservedbins: Number of observed bins
        self.recalls: Tuple of sorted recalls used to init the object
        self.precisions: Tuple of sorted precisions used to init the object
        self.recall_weight: Weight of recall when computing Fn-score
        self.fscoreof: ref_bin_name: float dict of reference bin Fn-scores
        self.fmean: Mean fscore
        self.adj_rand: Adjusted Rand index (not size normalized)
        self.mccof: ref_bin_name: float dict of MCC values
        self.mccmean: Mean Matthew's Correlation Coefficient (MCC)
        self._binsfound: (recall, prec): n_bins Counter
    """

    _DEFAULTRECALLS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    _DEFAULTPRECISIONS = [0.7, 0.8, 0.9, 0.95, 0.99]

    @staticmethod
    def _iterpairs(observed, reference, true_binid):
        true_contigs = reference.contigsof[true_binid]

        ref_binsize = reference.binlength[true_binid]

        obs_bins = {observed.binof.get(contig, None) for contig in true_contigs}
        obs_bins.discard(None)

        for obs_bin in obs_bins:
            obs_binsize = observed.binlength[obs_bin]
            intersection = observed.contigsof[obs_bin] & true_contigs
            true_pos = sum(reference.contiglength[i] for i in intersection)
            true_neg = reference.length + true_pos - ref_binsize - obs_binsize
            false_pos = obs_binsize - true_pos
            false_neg = ref_binsize - true_pos

            yield obs_bin, true_pos, true_neg, false_pos, false_neg


    @staticmethod
    def _mcc(true_pos, true_neg, false_pos, false_neg, sqrt=sqrt):
        mcc_num = true_pos * true_neg - false_pos * false_neg
        mcc_den = (true_pos + false_pos) * (true_pos + false_neg)
        mcc_den *= (true_neg + false_pos) * (true_neg + false_neg)
        if mcc_den == 0:
            return 0

        mcc = mcc_num / sqrt(mcc_den)

        return mcc

    @staticmethod
    def _fscore(true_pos, true_neg, false_pos, false_neg, recall_weight):
        recall = true_pos / (true_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        fscore = (recall_weight*recall_weight + 1) * (precision*recall)
        fscore /= (recall_weight*recall_weight*precision + recall)

        return fscore

    def __init__(self, *fakeargs, reference=None, observed=None,
                 recalls=_DEFAULTRECALLS, precisions=_DEFAULTPRECISIONS,
                 recall_weight=1.0):

        if len(fakeargs) > 0:
            raise ValueError('Only allows keyword arguments for safety.')

        if reference is None or observed is None:
            raise ValueError('Must supply reference and observed')

        self.recall_weight = recall_weight
        self.nreferencebins = reference.nbins
        self.nobservedbins = observed.nbins
        self.precisions = tuple(sorted(precisions))
        self.recalls = tuple(sorted(recalls))
        self.fscoreof = dict()
        self.mccof = dict()
        self._binsfound = Counter()
        self.adj_rand = get_adjusted_rand_score(observed.contigsof, reference.contigsof)

        for true_binid, true_contigs in reference.contigsof.items():
            recalls_precisions = list()
            max_fscore = 0
            max_mcc = -2

            obs_bins = {observed.binof.get(contig, None) for contig in true_contigs}
            obs_bins.discard(None)
            ref_binsize = reference.binlength[true_binid]

            stats = self._iterpairs(observed, reference, true_binid)
            for obs_id, true_pos, true_neg, false_pos, false_neg in stats:
                mcc = self._mcc(true_pos, true_neg, false_pos, false_neg)
                recall = true_pos / (true_pos + false_neg)
                precision = true_pos / (true_pos + false_pos)
                fscore = self._fscore(true_pos, true_neg, false_pos, false_neg, recall_weight)

                if mcc > max_mcc:
                    max_mcc = mcc

                if fscore > max_fscore:
                    max_fscore = fscore

                recalls_precisions.append((recall, precision))

            self.fscoreof[true_binid] = max_fscore

            if max_mcc == -2:
                max_mcc = 0

            self.mccof[true_binid] = max_mcc

            for min_recall in recalls:
                for min_precision in precisions:
                    for recall, precision in recalls_precisions:
                        if recall >= min_recall and precision >= min_precision:
                            self._binsfound[(min_recall, min_precision)] += 1
                            break

        # Calculate mean fscore:
        if len(self.fscoreof) == 0:
            self.fmean = 0.0
        else:
            self.fmean = sum(self.fscoreof.values()) / len(self.fscoreof)

        # Calculate mean mcc:
        if len(self.mccof) == 0:
            self.mccmean = 0.0
        else:
            self.mccmean = sum(self.mccof.values()) / len(self.mccof)

    def __getitem__(self, key):
        recall, precision = key
        if recall not in self.recalls or precision not in self.precisions:
            raise KeyError('Not initialized with that recall, precision pair')

        return self._binsfound[key]

    def atrecall(self, recall):
        """Returns a list of counted bins with the different precisions at
        a given recall.

        >>> result.atrecall(0.9)
        [141, 130, 115, 85, 55]
        """

        if not recall in self.recalls:
            raise ValueError('Not initialized with that recall')

        return [self[(recall, p)] for p in self.precisions]

    def atprecision(self, precision):
        """Returns a list of counted bins with the different recalls at
        a given precision.

        >>> result.precision(0.9)
        [141, 130, 115, 85, 55, 30, 14]
        """
        if not precision in self.precisions:
            raise KeyError('Not initialized with that precision')

        return [self[(r, precision)] for r in self.recalls]

    def reference_fraction(self, key):
        """Get number of bins / total ref. bins given a (recall, prec) tuple

        >>> result.reference_fraction((0.5, 0.9))
        0.315542
        """

        return self[key] / self.nreferencebins

    def observed_fraction(self, key):
        """Get number of bins / total obs. bins given a (recall, prec) tuple

        >>> result.observed_fraction((0.5, 0.9))
        0.291176308742343
        """

        return self[key] / self.nobservedbins

    def printmatrix(self, file=sys.stdout):
        """Prints the recall/precision number of bins to STDOUT."""

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self._binsfound[(recall, min_precision)] for recall in self.recalls]
            row.sort(reverse=True)
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)
