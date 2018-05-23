
__doc__ = """Benchmarks observed clusters given a reference.
    
observed file should be tab-sep lines of: cluster, contig
reference should be tab-sep lines of: cluster, contig, length

The cluster names need not be the same for obs. and ref. file, but the contig
names do.

For references without lengths, or if you wish to not weigh contigs
by length, set length of all contigs in reference to 1.

NOTE: This version counts the number of reference bins detected at the
different levents of recall and precision."""



import sys
import os
import argparse

from collections import defaultdict, Counter



class Reference:
    """Reference clusters.

    Init with {name: set_of_contig} dict (a) and {contigname: length} dict (b):
    Reference(a, b)
    
    Init with file of tab-sep lines w. clustername, contigname, contiglength:
    Reference.fromfile(open_filehandle)
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
    
    @classmethod
    def fromfile(cls, filehandle):
        """Load a reference with tab-sep lines of: binid, contigname, length"""       
        contigsof = defaultdict(set)
        contiglength = dict()
        
        for line in filehandle:
            stripped = line.rstrip()

            if line == '' or line[0] == '#':
                continue

            binid, contigname, length = stripped.split('\t')
                
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
    Observed(a, b)
    
    Init with file of tab-sep lines w. clustername, contigname and Reference:
    Reference.fromfile(open_filehandle, reference_object)
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
    """An object holding some benchmarkresults:
    
    Init from Reference object and Observed object using keywords:
    result = BenchmarkResult(reference=reference, observed=observed)
    
    Get number of references found at recall, precision:
        result[(recall, precision)]
    Get number of references found at recall or precision
        result.atrecall/atprecision(recall/precision)
    Print number of references at all recalls and precisions:
        result.printmatrix()
    """
    
    _DEFAULTRECALLS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    _DEFAULTPRECISIONS = [0.7, 0.8, 0.9, 0.95, 0.99]
    
    def __init__(self, *fakeargs, reference=None, observed=None,
                 recalls=_DEFAULTRECALLS, precisions=_DEFAULTPRECISIONS):
        
        if len(fakeargs) > 0:
            raise ValueError('Only allows keyword arguments for safety.')
            
        if reference is None or observed is None:
            raise ValueError('Must supply reference and observed')
        
        self.nreferencebins = reference.nbins
        self.nobservedbins = observed.nbins
        self.precisions = sorted(precisions)
        self.recalls = sorted(recalls)
        
        self._binsfound = Counter()
        
        for true_binid, true_contigs in reference.contigsof.items():
            true_binlength = reference.binlength[true_binid]
    
            obs_bins = {observed.binof.get(contig, None) for contig in true_contigs}
            obs_bins.discard(None)
            
            recalls_precisions = list()
            for obs_bin in obs_bins:
                obs_binlength = observed.binlength[obs_bin]
                intersection = observed.contigsof[obs_bin] & true_contigs
                intersection_length = sum(reference.contiglength[i] for i in intersection)
            
                recall = intersection_length / true_binlength
                precision = intersection_length / obs_binlength
    
                recalls_precisions.append((recall, precision))

            for min_recall in recalls:
                for min_precision in precisions:
                    for recall, precision in recalls_precisions:
                        if recall >= min_recall and precision >= min_precision:
                            self._binsfound[(min_recall, min_precision)] += 1
                            break
                            
    def __getitem__(self, key):
        if key not in self._binsfound:
            raise KeyError('Not initialized with that recall, precision pair')
            
        return self._binsfound[key]
    
    def atrecall(self, recall):
        if not recall in self.recalls:
            raise ValueError('Not initialized with that recall')
            
        return [self[(recall, p)] for p in self.precisions]
    
    def atprecision(self, precision):
        if not precision in self.precisions:
            raise KeyError('Not initialized with that precision')
            
        return [self[(r, precision)] for r in self.recalls]
    
    def reference_fraction(self, key):
        """As fraction of existing bins"""
        
        return self[key] / self.nreferencebins
    
    def observed_fraction(self, key):
        """As fraction of existing bins"""
        
        return self[key] / self.nobservedbins
    
    def printmatrix(self):
        """Prints the result."""

        print('\tRecall')
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t')

        for min_precision in self.precisions:
            row = [self._binsfound[(recall, min_precision)] for recall in self.recalls]
            row.sort(reverse=True)
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t')        



if __name__ == '__main__':

    parserkws = {'prog': 'benchmark.py',
                 'formatter_class': argparse.RawDescriptionHelpFormatter,
                 'usage': '%(prog)s OBSERVED REFERENCE',
                 'description': __doc__}

    # Create the parser
    parser = argparse.ArgumentParser(**parserkws)

    parser.add_argument('observed', help='observed bins')
    parser.add_argument('reference', help='true bins')

    # Print help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if not os.path.isfile(args.observed):
        raise FileNotFoundError(args.observed)

    if not os.path.isfile(args.reference):
        raise FileNotFoundError(args.reference)
        
    with open(args.reference) as filehandle:
        reference = Reference.fromfile(filehandle)
        
    with open(args.observed) as filehandle:
        observed = Observed.fromfile(filehandle, reference)
        
    results = BenchMarkResult(reference=reference, observed=observed)
    
    results.printmatrix()

