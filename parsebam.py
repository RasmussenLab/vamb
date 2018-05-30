
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
import numpy as _np
import gzip as _gzip

if __name__ == '__main__':
    import argparse as _argparse



DEFAULT_PROCESSES = min(8, _os.cpu_count())



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



def _get_contig_rpkms(path, minscore=50, minlength=2000):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.
    
    Inputs:
        path: Path to BAM file
        minscore [50]: Minimum alignment score (AS field) to consider
        minlength [2000]: Discard any references shorter than N bases 
        
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
    
    millionmappedreads = nhalfreads / 1e6
    
    for i, (contiglength, nhalfreads) in enumerate(zip(contiglengths, halfreads)):
        kilobases = contiglength / 1000
        rpkms[i] = nhalfreads / (kilobases * millionmappedreads)
        
    # Now filter for small contigs
    lengthmask = _np.array(contiglengths, dtype=_np.int32) >= minlength
    rpkms = rpkms[lengthmask]
    
    return path, rpkms



def read_bamfiles(paths, minscore=50, minlength=100,
                  processes=DEFAULT_PROCESSES):
    "Placeholder docstring"
    
    # Get references and lengths from first BAM file.
    # We need these to print them in the output.
    # Might as well do it before spawning all those processes.
    firstfile = _pysam.AlignmentFile(paths[0], "rb")
    contignames = list()
    
    for name, length in zip(firstfile.references, firstfile.lengths):
        if length >= minlength:
            contignames.append(name)
    
    ncontigs = len(contignames)
    
    if ncontigs == 0:
        raise ValueError('No headers in first bam file after filtering')
        
    del firstfile
    
    # Spawn independent processed to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=processes) as pool:
        for path in paths:
            arguments = (path, minscore, minlength)
            processresults.append(pool.apply_async(_get_contig_rpkms, arguments))
        
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



read_bamfiles.__doc__ = """Spawns processes to parse BAM files and get contig rpkms.
    
Input:
    path: Path to BAM file
    minscore [50]: Minimum alignment score (AS field) to consider
    minlength [100]: Ignore any references shorter than N bases 
    processes [{}]: Number of processes to spawn

Outputs:
    sample_rpkms: A {{path: Numpy-32-float-RPKM}} dictionary
    contignames: A list of contignames from first BAM header
""".format(DEFAULT_PROCESSES)



def write_samplerpkm(path, sample_rpkms):
    """Saves an RPKM dictionary to a path as .npz file.
    
    Inputs:
        path: Path to save to.
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        
    Output: None
    """
    
    _np.savez_compressed(path, **sample_rpkms)



def read_npz(path, columns):
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
        
    npz.close()
        
    # Normalize to rows sum to 1
    rowsums = _np.sum(arr, axis=1)
    rowsums[rowsums == 0] = 1
    arr /= rowsums.reshape((-1, 1))
    
    return arr



def toarray(sample_rpkms, columns):
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



def write_rpkms(filehandle, sample_rpkms, columns):
    """Writes an RPKMs dict to specified output path.
    
    Input:
        filehandle: Open filehandle to write to
        sample_rpkms: A {path: Numpy-32-float-RPKM} dictionary
        columns: Names of BAM file paths in sample_rpkms object in correct order
    Outputs: None
    """
    
    rpkms = [sample_rpkms[column] for column in columns]
    
    formatstring = '\t'.join(['{:.4f}']*len(rpkms)) + '\n'
    header = '# ' + '\t'.join(columns) + '\n'
    
    filehandle.write(header)
    for values in zip(*rpkms):
        filehandle.write(formatstring.format(*values))



if __name__ == '__main__':
    usage = "python parsebam.py [OPTION ...] OUTFILE BAMFILE [BAMFILE ...]"
    parser = _argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_argparse.RawDescriptionHelpFormatter,
        usage=usage)
    
    # Get the maximal number of extra processes to spawn
    cpus = min(8, _os.cpu_count())
    
    # Positional arguments
    parser.add_argument('outfile', help='path to output file')
    parser.add_argument('bamfiles', help='relative path to find bam files', nargs='+')
    
    # Optional arguments
    parser.add_argument('-a', dest='minscore', type=int, default=50,
                        help='minimum alignment score [50]')
    parser.add_argument('-m', dest='minlength', type=int, default=100,
                        help='discard any references shorter than N [100]')
    parser.add_argument('-p', dest='processes', type=int, default=cpus,
                        help=('number of extra processes to spawn '
                              '[min(' + str(cpus) + ', nfiles)]'))
    
    # If no arguments, print help
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()
        
    args = parser.parse_args()
    
    # Check number of cores
    if args.processes < 1:
        raise argsparse.ArgumentTypeError('Zero or negative processes requuested.')
        
    if args.minlength < 4:
        raise ValueError('Minlength must be at least 4')
        
    # Check presence of output file
    if _os.path.exists(args.outfile):
        raise FileExistsError('Output file: ' + args.outfile)
        
    directory = _os.path.dirname(args.outfile)
    if directory and not _os.path.isdir(directory):
        raise NotADirectoryError(directory)
        
    # Check presence of all BAM files
    for bamfile in args.bamfiles:
        if not _os.path.isfile(bamfile):
            raise FileNotFoundError('Bam file: ' + bamfile)
            
    # If more CPUs than files, scale down CPUs
    if args.processes > len(args.bamfiles):
        args.processes = len(args.bamfiles)
        print('More processes given than files, scaling to {}.'.format(args.processes))
    
    print('Number of files:', len(args.bamfiles))
    print('Number of parallel processes:', args.processes)
    
    sample_rpkms, contignames = read_bamfiles(args.bamfiles, args.minscore,
                                              args.minlength, args.processes)
    
    with open(args.outfile, 'w') as file:
        write_rpkms(file, sample_rpkms, args.bamfiles)

