
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



import pysam

import sys
import os
import multiprocessing
import argparse
import time

from collections import defaultdict



def get_all_references(alignedsegment):
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



def filter_segments(segmentiterator, minscore):
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



def get_contig_rpkms(identifier, path, minscore):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.
    """
    
    print('Parsing file:', path)
    
    bamfile = pysam.AlignmentFile(path, "rb")
    
    # We can only get secondary alignment reference names, not indices. So we must
    # make an idof dict to look up the indices.
    idof = {contig: i for i, contig in enumerate(bamfile.references)}
    contiglengths = bamfile.lengths
    halfreads = [0] * len(contiglengths)
    
    nhalfreads = 0
    for segment in filter_segments(bamfile, minscore):
        nhalfreads += 1
        
        # Read w. unmapped mates count twice as they represent a whole read
        value = 2 if segment.mate_is_unmapped else 1
        
        for reference in get_all_references(segment):
            id = idof[reference]
            halfreads[id] += value
            
    bamfile.close()
    
    print('Done parsing file:', path)
    
    rpkms = list()
    
    # Compensate for having paired reads
    millionmappedreads = nhalfreads / 2e6
    
    for contiglength, nhalfreads in zip(contiglengths, halfreads):
        kilobases = contiglength / 1000
        rpkms.append(nhalfreads / (kilobases * millionmappedreads))
    
    return identifier, rpkms



def main(paths, minscore, outfile, is_matrix):
    """Spawns processes to parse BAM files and get contig rpkms from them,
    then prints the resulting table to an output file"""
    
    # Get references and lengths from first BAM file.
    # We need these to print them in the output.
    # Might as well do it before spawning all those processes.
    firstfile = pysam.AlignmentFile(paths[0], "rb")
    references = firstfile.references
    lengths = firstfile.lengths
    
    if not len(references) == len(lengths):
        raise ValueError('Could not parse headers of first bam-file')
    
    # Spawn independent processed to calculate RPKM for each of the BAM files
    processresults = list()
    processes_done = 0
    
    # This is just to print to terminal when a process finishes. Not necessary.
    def callback(result, totalps=len(paths)):
        "Generator yielding processed"
        nonlocal processes_done
        processes_done += 1
        print('Files processed: {}/{}'.format(processes_done, totalps))
        return None

    # Queue all the processes
    with multiprocessing.Pool(processes=args.processors) as pool:
        for fileno, path in enumerate(paths):
            arguments = (fileno, path, args.minscore)
            processresults.append(pool.apply_async(get_contig_rpkms, arguments,
                                                  callback=callback, error_callback=callback))
        
        # For some reason, this is needed.
        pool.close()
        pool.join()
            
    print('All processes finished. Checking outputs')
    sample_rpkms = list()
    
    for processresult in processresults:
        if processresult.successful():
            sample_rpkms.append(processresult.get())
            
        else:
            raise multiprocessing.ProcessError
    
    # sample_rpkms now contain (identifier, sample_rpkms) tuples, in the order
    # they were returned from the pool. We want to sort them by identifier,
    # so that we know which RPKMs belong to which BAM file
    sample_rpkms.sort()
    
    # Now we can discard the identifiers
    sample_rpkms = [i[1] for i in sample_rpkms]
    
    # Each BAM file MUST contain the same headers
    if not all(len(rpkms) == len(lengths) for rpkms in sample_rpkms):
        raise ValueError('Not all BAM files contain the same amount of headers.')
    
    print('Outputs alright. Printing table.')
    
    with open(outfile, 'w') as filehandle:
        # Print header if asked
        if not is_matrix:
            print('#contig\tcontiglength', '\t'.join(paths), sep='\t', file=filehandle)
    
        # Print the actual output
        for fields in zip(references, lengths, *sample_rpkms):
            numbers = '\t'.join([str(round(i, 3)) for i in fields[2:]])
            
            if not is_matrix:
                print(fields[0], fields[1], sep='\t', end='\t', file=filehandle)
                
            print(numbers, file=filehandle)



if __name__ == '__main__':
    usage = "python rpkmtable.py [OPTION ...] OUTFILE BAMFILE [BAMFILE ...]"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage)
    
    # Get the maximal number of extra processes to spawn
    cpus = os.cpu_count()
    
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
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
        
    args = parser.parse_args()
    
    # Check number of cores
    if args.processors < 1:
        raise argsparse.ArgumentTypeError('Zero or negative processors provided.')
        
    # Check presence of output file
    if os.path.exists(args.outfile):
        raise FileExistsError('Output file: ' + args.outfile)
        
    # Check presence of all BAM files
    for bamfile in args.bamfiles:
        if not os.path.isfile(bamfile):
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
    
    starttime = time.time()
    print('Starting RPKM estimation.')
    
    # Run the stuff!
    main(args.bamfiles, args.minscore, args.outfile, args.matrix)
    
    elapsed = round(time.time() - starttime)
    print('Done in {} seconds.'.format(elapsed))

