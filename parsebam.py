# # Changes to do
# 
# get_contig_rpkms should return an array.array('f').
# 
# split main to four functions:
# 
# read_bam_files: first part of current main, but instead saves the rpkms in a {path: array} dict.
# 
# save_arrays (or some name): makes a dir and saves all the arrays using array.array.tofile
# 
# load_as_np_array(paths, length): Inits a NP arr, then loads as columns using array.array.fromfile.
# 
# write_table(path, nparray): just use same lines as from parsecontigs.py's write function


# Metagenomic RPKM estimator

# Author: Jakob Nybo Nissen, DTU Bioinformatics, jakni@bioinformatics.dtu.dk
# Date: 2018-04-24
    
# This script calculates RPKM, when paired end reads are mapped to a contig
# catalogue with BWA MEM. It will not be accurate with single end reads or
# any other mapper than BWA MEM.

# Theory:
# We want a simple way to estimate abundance of redundant contig catalogues.
# Earlier we used to run analysis on deduplicated gene catalogues, but since 
# both depth and kmer composition are only stable for longer contigs, we have
# moved to contig catalogues. We have not found a way of deduplicating contigs.

# For this we have until now used two methods:
# 1) Only counting the primary hits. In this case the read will never be
# assigned to any contig which differ by just 1 basepair. Even for
# identical contigs, reads are assigned randomly which causes noise.

# 2) Using MetaBAT's jgi_summarize_bam_contig_depths, a script which is not
# documented and we cannot figure out how works. When testing with small
# toy data, it produces absurd results.

# This script is an attempt to take an approach as simple as possible while
# still being sound technically. We simply count the number of reads in a
# contig normalized by contig length and total number of reads.

# We look at all hits, including secondary hits. We do not discount partial
# alignments. Also, if a read maps to multiple contigs, we don't count each hit
# as less than if it mapped to both. The reason for all these decisions is that
# if the aligner believes it's a hit, we believe the contig is present.

# We do not take varying insert sizes into account. It is unlikely that
# any contig with enough reads to provide a reliable estimate of depth would,
# by chance, only recruit read pairs with short or long insert size. So this
# will average out over all contigs.

# We DO filter the input file for duplicate lines, as BWA MEM erroneously
# produces quite a few of them.

# We count each read independently, because BWA MEM often assigns mating reads
# to different contigs. If a read has their mate unmapped, we count it twice
# to compensate (one single read corresponds to two paired reads).



__doc__ = """Metagenomic RPKM estimator

Author: Jakob Nybo Nissen, DTU Bioinformatics, jakni@bioinformatics.dtu.dk
    
This script calculates reads per kilobase per million mapped reads (RPKM),
when paired end reads are mapped to a contig catalogue with BWA MEM.
Because of the idiosyncracities of BWA, it will not be accurate with single end
reads or any other mapper than BWA MEM.

Requires the module pysam to run."""



import pysam as _pysam

import sys as _sys
import os as _os
import multiprocessing as _multiprocessing
import argparse as _argparse
import numpy as _np
import gzip as _gzip



def _get_all_references(alignedsegment):
    """Given a pysam aligned segment, returns a list with the names of all
    references the read maps to, both primary and secondary hits.
    """
    
    references = [alignedsegment.reference_name]
    
    # Some reads don't have secondary hits
    if not alignedsegment.has_tag('XA'):
        return references
    
    # XA is a string contigname1,<other info>;contigname2,<other info>; ...
    secondary_alignment_string = alignedsegment.get_tag('XA')
    secondary_alignments = secondary_alignment_string.split(';')[:-1]
    
    for secondary_alignment in secondary_alignments:
        references.append(secondary_alignment.partition(',')[0])
        
    return references



def _filter_segments(segmentiterator, minscore):
    """Returns an iterator of AlignedSegment filtered for reads with low
    alignment score, and for any segments identical to the previous segment.
    This is necessary as BWA MEM produces dopplegangers.
    """
    
    # First get the first segment, so we in the loop can compare to the previous
    alignedsegment = next(segmentiterator)
    
    yield alignedsegment
        
    lastname = alignedsegment.query_name
    lastwasforward = alignedsegment.flag & 64 == 64
        
    for alignedsegment in segmentiterator:
        if alignedsegment.get_tag('AS') < minscore:
            continue
        
        # Depressingly, BWA somtimes outputs the same read multiple times.
        # We identify them by having same name as directionality as previous. 
        thisname = alignedsegment.query_name
        thisisforward = alignedsegment.flag & 64 == 64
        
        if thisisforward is not lastwasforward or thisname != lastname:
            yield alignedsegment
            
            lastname = thisname
            lastwasforward = thisisforward



def get_contig_rpkms(path, minscore):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.
    
    Inputs:
        path: Path to BAM file
        minscore: Minimum alignment score (AS field) to consider
        
    Outputs:
        path: Same as input path
        rpkms A float32-array with RPKM for each contig in BAM header
    """
    
    bamfile = _pysam.AlignmentFile(path, "rb")

    # We can only get secondary alignment reference names, not indices. So we must
    # make an idof dict to look up the indices.
    idof = {contig: i for i, contig in enumerate(bamfile.references)}
    contiglengths = bamfile.lengths
    halfreads = _np.zeros(len(contiglengths), dtype=_np.int32)
    
    nhalfreads = 0
    for segment in _filter_segments(bamfile, minscore):
        nhalfreads += 1
        
        # Read w. unmapped mates count twice as they represent a whole read
        value = 2 if segment.mate_is_unmapped else 1
        
        for reference in _get_all_references(segment):
            id = idof[reference]
            halfreads[id] += value
            
    bamfile.close()
    del idof
    
    rpkms = _np.zeros(len(contiglengths), dtype=_np.float32)
    
    # Compensate for having paired reads
    millionmappedreads = nhalfreads / 2e6
    
    for i, (contiglength, nhalfreads) in enumerate(zip(contiglengths, halfreads)):
        kilobases = contiglength / 1000
        rpkms[i] = nhalfreads / (kilobases * millionmappedreads)
    
    return path, rpkms



def read_bamfiles(paths, minscore, processors):
    """Spawns processes to parse BAM files and get contig rpkms.
    
    Input:
        path: Path to BAM file
        minscore: Minimum alignment score (AS field) to consider
        processors: Number of processes to spawn

    Outputs:
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        contignames: A list of contignames from first BAM header
    """
    
    # Get references and lengths from first BAM file.
    # We need these to print them in the output.
    # Might as well do it before spawning all those processes.
    firstfile = _pysam.AlignmentFile(paths[0], "rb")
    contignames = firstfile.references
    ncontigs = len(contignames)
    
    if ncontigs == 0:
        raise ValueError('Could not parse headers of first bam-file')
        
    del firstfile
    
    # Spawn independent processed to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=processors) as pool:
        for path in paths:
            arguments = (path, minscore)
            processresults.append(pool.apply_async(get_contig_rpkms, arguments))
        
        # For some reason, this is needed.
        pool.close()
        pool.join()
            
    sample_rpkms = dict()
    
    for processresult in processresults:
        if processresult.successful():
            path, rpkm = processresult.get()
            sample_rpkms[path] = rpkm
            
        else:
            return processresult
            raise _multiprocessing.ProcessError
            
    # Check results have same length
    wrong = [(path, len(rpkm)) for path, rpkm in sample_rpkms.items()
             if len(rpkm) != ncontigs]
    
    if wrong:
        message = ('Expected number of headers: ' +
                   str(ncontigs) +
                   '\nfollowing samples have wrong number of headers:\n' +
                   '\n'.join(p+'\t'+str(i) for p, i in wrong))
        raise ValueError(message)
            
    return sample_rpkms, contignames



def sample_rpkms_tonpz(sample_rpkms, path):
    """Saves an RPKM dictionary to a path as .npz file.
    
    Inputs:
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        path: Path to save to
        
    Output: None
    """
    
    _np.savez_compressed(path, **sample_rpkms)



def array_fromnpz(path, columns):
    """Creates a (n-contigs x n-bamfiles) array from an npz object as created
    by sample_rpkms_tonpz.
    
    Inputs: 
        path: Path to npz object
        columns: Names of BAM file paths in npz object in correct order
        
    Output: A (n-contigs x n-bamfiles) array with each column being the corre-
    sponding numpy array from the npz file, normalized so each row sums to 1
    """
    
    npz = _np.load(path)
    
    ncontigs = len(npz[columns[0]])
    ncolumns = len(columns)
    
    arr = _np.empty((ncontigs, ncolumns))
    
    for i, filename in enumerate(columns):
        column = npz[filename]
        
        if not len(column) == ncontigs:
            errormessage = 'Expected {} contigs, but {} has {}'
            raise ValueError(errormessage.format(ncontigs, filename, len(column)))
        
        arr[:, i] = column
        
    # Normalize to rows sum to 1
    rowsums = _np.sum(arr, axis=1)
    rowsums[rowsums == 0] = 1
    arr /= rowsums.reshape((-1, 1))
    
    return arr



def array_from_sample_rpkms(sample_rpkms, columns):
    """Creates a (n-contigs x n-bamfiles) array from a sample_rpkms
    (expected to be a {path: Numpy-32-float-RPKM} dictionary)
    
    Inputs: 
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        columns: Names of BAM file paths in sample_rpkms object in correct order
        
    Output: A (n-contigs x n-bamfiles) array with each column being the corre-
    sponding array from sample_rpkms, normalized so each row sums to 1
    """
    
    ncontigs = len(next(iter(sample_rpkms.values())))
    
    for filename, array in sample_rpkms.items():
        if not len(array) == ncontigs:
            errormessage = 'Expected {} contigs, but {} has {}'
            raise ValueError(errormessage.format(ncontigs, filename, len(array)))
    
    arr = _np.zeros((ncontigs, len(sample_rpkms)), dtype=_np.float32)
    
    for n, path in enumerate(columns):
        rpkm = sample_rpkms[path]
        arr[:, n] = rpkm
    
    # Normalize to rows sum to 1
    rowsums = _np.sum(arr, axis=1)
    rowsums[rowsums == 0] = 1
    arr /= rowsums.reshape((-1, 1))
    
    return arr



def write_rpkms(path, sample_rpkms, columns, contignames):
    """Writes an RPKMs table to specified output path.
    
    Input:
        path: Path to write output
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        columns: Names of BAM file paths in sample_rpkms object in correct order
        contignames: A list of contig headers in order
    Outputs: None
    """
    
    rpkms = [sample_rpkms[column] for column in columns]
    
    formatstring = '{}\t' + '\t'.join(['{:.4f}']*len(rpkms)) + '\n'
    header = '#contignames\t' + '\t'.join(columns) + '\n'
    
    with _gzip.open(path, 'w') as file:
        file.write(header.encode())
        for contigname, *values in zip(contignames, *rpkms):
            file.write(formatstring.format(contigname, *values).encode())



if __name__ == '__main__':
    usage = "python parsebam.py [OPTION ...] OUTFILE BAMFILE [BAMFILE ...]"
    parser = _argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_argparse.RawDescriptionHelpFormatter,
        usage=usage)
    
    # Get the maximal number of extra processes to spawn
    cpus = _os.cpu_count()
    
    # Positional arguments
    parser.add_argument('outfile', help='path to output file')
    parser.add_argument('bamfiles', help='relative path to find bam files', nargs='+')
    
    # Optional arguments
    parser.add_argument('-m', dest='minscore', type=int, default=50,
                        help='minimum alignment score [50]')
    parser.add_argument('-p', dest='processors', type=int, default=cpus,
                        help=('number of extra processes to spawn '
                              '[min(' + str(cpus) + ', nfiles)]'))
    parser.add_argument('--matrix', action='store_true',
        help='only print RPKM, not headers, contignames or lengths')
    
    # If no arguments, print help
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()
        
    args = parser.parse_args()
    
    # Check number of cores
    if args.processors < 1:
        raise argsparse.ArgumentTypeError('Zero or negative processors provided.')
        
    # Check presence of output file
    if _os.path.exists(args.outfile):
        raise FileExistsError('Output file: ' + args.outfile)
        
    # Check presence of all BAM files
    for bamfile in args.bamfiles:
        if not _os.path.isfile(bamfile):
            raise FileNotFoundError('Bam file: ' + bamfile)
            
    # If more CPUs than files, scale down CPUs
    if args.processors > len(args.bamfiles):
        args.processors = len(args.bamfiles)
        print('More processes given than files, scaling to {}.'.format(args.processors))
    
        print('Number of files:', len(args.bamfiles))
    print('Number of parallel processes:', args.processors)
    
    # Warn user if --matrix is given
    if args.matrix:
        print('No headers given. Columns ordered like file arguments:')
        for n, bamfile in enumerate(args.bamfiles):
            print('\tcol ', str(n+1), ': ', bamfile, sep='')
    
    sample_rpkms, contignames = read_bamfiles(args.bamfiles, args.minscore, args.processors)
    write_rpkms(args.outfile, sample_rpkms, args.bamfiles, contignames)

