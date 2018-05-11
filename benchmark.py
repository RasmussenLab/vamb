
__doc__ = """Benchmarks observed clusters given a reference.
    
observed file should be tab-sep lines of: cluster, contig
reference should be tab-sep lines of: cluster, contig, length

For references without lengths, or if you wish to not weigh contigs
by length, set length of all contigs in reference to 1.

NOTE: The numbers are the number of (observed, true) bin combinations
that pass the criteria. Thus, for relaxed criteria where recall < 0.5,
or where observed bins are not disjoint, the same true bin may be
counted more than once."""



import sys
import os
import argparse

from collections import defaultdict



def load_reference(path):
    """Load a reference with tab-sep lines of: binid, contigname, length"""
    
    true_lengthof = dict()
    true_binof = dict()
    lengthof = dict()
    
    with open(path) as filehandle:
        for line in filehandle:
            stripped = line.rstrip()
            
            if line == '' or line[0] == '#':
                continue
            
            binid, contig, length = stripped.split('\t')
            length = int(length)
            
            true_binof[contig] = binid
            true_lengthof[binid] = true_lengthof.get(binid, 0) + length
            lengthof[contig] = length
            
    return true_lengthof, true_binof, lengthof



def load_observed(path):
    """Load observed bins as tab-sep lines of: binind, contigname"""
    
    obs_contigsof = defaultdict(set)
    
    with open(path) as filehandle:
        for line in filehandle:
            stripped = line.rstrip()
            
            if line == '' or line[0] == '#':
                continue
                
            binid, contigname = stripped.split('\t')
            obs_contigsof[binid].add(contigname)
            
    return obs_contigsof



def compare_real_obs_bins(obs_contigsof, true_lengthof, true_binof, lengthof):
    """Calculates a {bin [(sens1, spec1), (sens2, spec2) ...], bin2 ...}
    for all true bins and their (sensitivity, specificity) to the observed contigs"""
    
    obs_sens_spec = defaultdict(list)
    
    # Compare observed and expected bins
    for obs_contigs in obs_contigsof.values():
        obs_length = 0
        
        for contig in obs_contigs:
            try:
                obs_length += lengthof[contig]
            except KeyError:
                errormsg = 'contig {} not present in reference'.format(contig)
                raise KeyError(errormsg) from None

        # Group contigs in observed bin by which bin they truly belong to
        groups = defaultdict(set)
        for contig in obs_contigs:
            truebin = true_binof[contig]
            groups[truebin].add(contig)

        # For each of these groups within the observed bin:
        for truebin, contigs in groups.items():
            length_of_group = sum(lengthof[contig] for contig in contigs)
            
            assert length_of_group <= obs_length

            sensitivity = length_of_group / true_lengthof[truebin]
            specificity = length_of_group / obs_length

            # Append specificity and sensitivity to the truebins
            obs_sens_spec[truebin].append((sensitivity, specificity))

    return obs_sens_spec



def print_sens_spec(obs_sens_spec):
    """Prints the result."""
    
    recalls = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    precisions = (0.7, 0.8, 0.9, 0.95, 0.99)

    print('\tRecall')
    print('Prec.', '\t'.join([str(r) for r in recalls]), sep='\t')

    for min_precision in precisions:
        passed_bins = [0] * len(recalls)

        for truebin, recall_precision_pairs in obs_sens_spec.items():
            for recall, precision in recall_precision_pairs:
                for i, min_recall in enumerate(recalls):
                    if precision >= min_precision and recall >= min_recall:
                        passed_bins[i] += 1

        print(min_precision, '\t'.join([str(c) for c in passed_bins]), sep='\t')



def main(referencepath, observedpath):
    true_lengthof, true_binof, lengthof = load_reference(referencepath)
    obs_contigsof = load_observed(observedpath)
    obs_sens_spec = compare_real_obs_bins(obs_contigsof, true_lengthof, true_binof, lengthof)
    
    print_sens_spec(obs_sens_spec)



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

    main(args.reference, args.observed)

