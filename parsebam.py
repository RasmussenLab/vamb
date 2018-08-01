
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
        # We identify them by having same name and directionality as previous. 
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
                  processes=DEFAULT_PROCESSES, logfile=None):
    "Placeholder docstring - replaced after this func definition"
    
    if logfile is not None:
        def _callback(result):
            path, rpkms = result
            print('\tProcessed ', path, file=logfile)

        def _error_callback(result):
            path, rpkms = result
            print('\tERROR WHEN PROCESSING ', path, file=logfile)
    else:
        def _callback(result):
            pass

        def _error_callback(result):
            pass
    
    # Get references and lengths from first BAM file.
    # We need these to print them in the output.
    # Might as well do it before spawning all those processes.
    firstfile = _pysam.AlignmentFile(paths[0], "rb")
    ncontigs = sum(1 for length in firstfile.lengths if length >= minlength)
    firstfile.close()
    del firstfile
    
    if ncontigs == 0:
        raise ValueError('No headers in first bam file after filtering')
    
    # Spawn independent processed to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=processes) as pool:
        for path in paths:
            arguments = (path, minscore, minlength)
            processresults.append(pool.apply_async(_get_contig_rpkms, arguments,
                                                   callback=_callback, error_callback=_error_callback))
        
        # For some reason, this is needed.
        pool.close()
        pool.join()
            
    columnof = {p:i for i, p in enumerate(paths)}
    rpkms = _np.zeros((ncontigs, len(paths)), dtype=_np.float32)
    
    for processresult in processresults:
        if processresult.successful():
            path, rpkm = processresult.get()
            
            if len(rpkm) != ncontigs:
                raise ValueError('Expected {} headers in {}, got {}.'.format(
                                 ncontigs, path, len(rpkm)))
            rpkms[:, columnof[path]] = rpkm
            
        else:
            return processresult
            raise _multiprocessing.ProcessError
            
    return rpkms



read_bamfiles.__doc__ = """Spawns processes to parse BAM files and get contig rpkms.
    
Input:
    path: Path to BAM file
    minscore [50]: Minimum alignment score (AS field) to consider
    minlength [100]: Ignore any references shorter than N bases 
    processes [{}]: Number of processes to spawn
    logfile: [None] File to print progress to

Output: A (n_contigs x n_samples) Numpy array with RPKM
""".format(DEFAULT_PROCESSES)

