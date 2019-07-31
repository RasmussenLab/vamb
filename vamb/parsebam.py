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

# We count each read independently, because BWA MEM often assigns mating reads
# to different contigs. If a read has their mate unmapped, we count it twice
# to compensate (one single read corresponds to two paired reads).

__doc__ = """Estimate RPKM (depths) from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = read_bamfiles(bampaths)
"""

import pysam as _pysam

import sys as _sys
import os as _os
import multiprocessing as _multiprocessing
import numpy as _np
import time as _time

DEFAULT_SUBPROCESSES = min(8, _os.cpu_count())

def mergecolumns(pathlist):
    """Merges multiple npz files with columns to a matrix.

    All paths must be npz arrays with the array saved as name 'arr_0',
    and with the same length.

    Input: pathlist: List of paths to find .npz files to merge
    Output: Matrix with one column per npz file
    """

    if len(pathlist) == 0:
        return _np.array([], dtype=_np.float32)

    for path in pathlist:
        if not _os.path.exists(path):
            raise FileNotFoundError(path)

    first = _np.load(pathlist[0])['arr_0']
    length = len(first)
    ncolumns = len(pathlist)

    result = _np.zeros((length, ncolumns), dtype=_np.float32)
    result[:,0] = first

    for columnno, path in enumerate(pathlist[1:]):
        column = _np.load(path)['arr_0']
        if len(column) != length:
            raise ValueError("Length of data at {} is not equal to that of {}".format(path, pathlist[0]))
        result[:,columnno + 1] = column

    return result

def _identity(segment):
    "Return the nucleotide identity of the given aligned segment."
    mismatches, matches = 0, 0
    for kind, number in segment.cigartuples:
        # 0, 7, 8, is match/mismatch, match, mismatch, respectively
        if kind in (0, 7, 8):
            matches += number
        # 1, 2 is insersion, deletion
        elif kind in (1, 2):
            mismatches += number
    matches -= segment.get_tag('NM')
    return matches / (matches+mismatches)

def _filter_segments(segmentiterator, minscore, minid):
    """Returns an iterator of AlignedSegment filtered for reads with low
    alignment score.
    """

    for alignedsegment in segmentiterator:
        # Skip if unaligned or suppl. aligment
        if alignedsegment.flag & 0x804 != 0:
            continue

        if minscore is not None and alignedsegment.get_tag('AS') < minscore:
            continue

        if minid is not None:
            if _identity(alignedsegment) < minid:
                continue

        yield alignedsegment

def _get_contig_rpkms(inpath, outpath, minscore, minlength, minid):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.

    Inputs:
        inpath: Path to BAM file
        outpath: Path to dump depths array to or None
        minscore [None]: Minimum alignment score (AS field) to consider
        minlength [2000]: Discard any references shorter than N bases
        minid [None]: Discard any reads with ID lower than this

    Outputs:
        path: Same as input path
        rpkms:
            If outpath is not None: None
            Else: A float32-array with RPKM for each contig in BAM header
        length: Length of rpkms array
    """

    bamfile = _pysam.AlignmentFile(inpath, "rb")
    contiglengths = bamfile.lengths
    readcounts = _np.zeros(len(contiglengths), dtype=_np.float32)

    # Initialize with first aligned read
    filtered_segments = _filter_segments(bamfile, minscore, minid)
    try:
        segment = next(filtered_segments)
        read_name = segment.query_name
        multimap = 1.0
        reference_ids = [segment.reference_id]
    except StopIteration:
        pass

    total_reads = 0
    # Now count up each read in the BAM file
    for segment in filtered_segments:
        # If we reach a new read_name, we tally up the previous read
        # towards all its references.
        if segment.query_name != read_name:
            read_name = segment.query_name
            total_reads += 1
            to_add = 1.0 / multimap
            for reference_id in reference_ids:
                readcounts[reference_id] += to_add
            reference_ids.clear()
            multimap = 0.0

        multimap += 1.0
        reference_ids.append(segment.reference_id)

    bamfile.close()

    # Add final read
    if total_reads != 0:
        to_add = 1.0 / multimap
        for reference_id in reference_ids:
            readcounts[reference_id] += to_add

    # Calcuate RPKM iself
    rpkms = _np.zeros(len(contiglengths), dtype=_np.float32)

    if total_reads > 0:
        millionmappedreads = total_reads / 1e6

        for i, (contiglength, nreads) in enumerate(zip(contiglengths, readcounts)):
            kilobases = contiglength / 1000
            rpkms[i] = nreads / (kilobases * millionmappedreads)

    # Now filter for small contigs
    lengthmask = _np.array(contiglengths, dtype=_np.int32) >= minlength
    rpkms = rpkms[lengthmask]

    # If dump to disk, array returned is None instead of rpkm array
    if outpath is not None:
        arrayresult = None
        _np.savez_compressed(outpath, rpkms)
    else:
        arrayresult = rpkms

    return inpath, arrayresult, len(rpkms)

def read_bamfiles(paths, dumpdirectory=None, minscore=None, minlength=100, minid=None,
                  subprocesses=DEFAULT_SUBPROCESSES, logfile=None):
    "Placeholder docstring - replaced after this func definition"

    # Define callback function depending on whether a logfile exists or not
    if logfile is not None:
        def _callback(result):
            path, rpkms, length = result
            print('\tProcessed', path, file=logfile)
            logfile.flush()

    else:
        def _callback(result):
            pass

    # Bam files must be unique.
    if len(paths) != len(set(paths)):
        raise ValueError('All paths to BAM files must be unique.')

    # Bam files must exist
    for path in paths:
        if not _os.path.isfile(path):
            raise FileNotFoundError(path)

    if dumpdirectory is not None:
        # Dumpdirectory cannot exist, but its parent must exist
        dumpdirectory = _os.path.abspath(dumpdirectory)
        if _os.path.exists(dumpdirectory):
            raise FileExistsError(dumpdirectory)

        parentdir = _os.path.dirname(_os.path.abspath(dumpdirectory))
        if not _os.path.isdir(parentdir):
            raise FileNotFoundError("Parent dir of " + dumpdirectory)

        # Create directory to dump in
        _os.mkdir(dumpdirectory)

    # Get references and lengths from first BAM file.
    # We need these to print them in the output.
    # Might as well do it before spawning all those processes.
    firstfile = _pysam.AlignmentFile(paths[0], "rb")
    ncontigs = sum(1 for length in firstfile.lengths if length >= minlength)

    # Probe to check that the "AS" aux field is present (BWA makes this)
    if minscore is not None:
        segments = [j for i, j in zip(range(25), firstfile)]
        if not all(segment.has_tag("AS") for segment in segments):
            raise ValueError("If minscore is set, 'AS' field must be present in BAM file.")

    firstfile.close()
    del firstfile

    if ncontigs == 0:
        raise ValueError('No headers in first bam file after length filtering')

    # Spawn independent processes to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=subprocesses) as pool:
        for pathnumber, path in enumerate(paths):
            if dumpdirectory is None:
                outpath = None
            else:
                outpath = _os.path.join(dumpdirectory, str(pathnumber) + '.npz')

            arguments = (path, outpath, minscore, minlength, minid)
            processresults.append(pool.apply_async(_get_contig_rpkms, arguments,
                                                   callback=_callback))

        all_done, any_fail = False, False
        while not (all_done or any_fail):
            _time.sleep(5)
            all_done = all(process.ready() and process.successful() for process in processresults)
            any_fail = any(process.ready() and not process.successful() for process in processresults)

            if all_done:
                pool.close() # exit gently
            if any_fail:
                pool.terminate() # exit less gently

        # Wait for all processes to be cleaned up
        pool.join()

    # Raise the error if one of them failed.
    for path, process in zip(paths, processresults):
        if process.ready() and not process.successful():
            print('\tERROR WHEN PROCESSING:', path, file=logfile)
            print('Vamb aborted due to error in subprocess. See stacktrace for source of exception.')
            logfile.flush()
            process.get()

    # Verify we didn't get errors or wrong lengths
    for processresult in processresults:
        path, rpkm, length = processresult.get()

        if length != ncontigs:
            raise ValueError('Expected {} headers in {}, got {}.'.format(
                             ncontigs, path, length))

    # If we did not dump to disk, load directly from process results to
    # one big matrix...
    if dumpdirectory is None:
        columnof = {p:i for i, p in enumerate(paths)}
        rpkms = _np.zeros((ncontigs, len(paths)), dtype=_np.float32)

        for processresult in processresults:
            path, rpkm, length = processresult.get()
            rpkms[:, columnof[path]] = rpkm

    # If we did, instead merge them from the disk
    else:
        dumppaths = [_os.path.join(dumpdirectory, str(i) + '.npz') for i in range(len(paths))]
        rpkms = mergecolumns(dumppaths)

    return rpkms

read_bamfiles.__doc__ = """Spawns processes to parse BAM files and get contig rpkms.

Input:
    path: List or tuple of paths to BAM files
    dumpdirectory: [None] Dir to create and dump per-sample depths NPZ files to
    minscore [None]: Minimum alignment score (AS field) to consider
    minlength [100]: Ignore any references shorter than N bases
    minid [None]: Discard any reads with nucleotide identity less than this
    subprocesses [{}]: Number of subprocesses to spawn
    logfile: [None] File to print progress to

Output: A (n_contigs x n_samples) Numpy array with RPKM
""".format(DEFAULT_SUBPROCESSES)
