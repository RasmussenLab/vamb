import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysam
from git_commit import get_git_commit


# Function to process a single BAM file
def process_bam(bam_path, max_insert_len):
    """
    This function parses a bam file, and identifies paired-end reads where each read mate maps to the end of the contig, within certain distance

    Args:
        bam_path (str): Path of the bam file containing the paired end reads from a sample mapping to all contigs.
        max_insert_len (in): Max distance between the end of the contig and the read mapping position for it to be considered.

    Returns:
        local_circular_contigs (set): set of contigs with paired-end reads mapping to contig ends.
        contig_reads_d (dict): dictionary containing the reads and read position for circularized contigs.
        allcontig_reads_d (dict): dictionary containing the reads and read position for all contigs.
    """
    local_circular_contigs = set()
    contig_reads_d = dict()
    allcontig_reads_d = dict()

    with pysam.AlignmentFile(bam_path, "rb") as bamfile:
        reference_lengths = {}

        for read in bamfile:
            if (
                read.is_unmapped
                or read.mate_is_unmapped
                or read.next_reference_name != read.reference_name
            ):
                continue
            is_start_mapping = (
                read.is_reverse and read.reference_start < max_insert_len
            ) and (
                not read.mate_is_reverse
                and read.next_reference_start
                > (read.reference_length - read.query_length - max_insert_len)
            )
            is_end_mapping = (
                not read.is_reverse
                and (
                    read.reference_start
                    > read.reference_length - read.query_length - max_insert_len
                )
            ) and (read.mate_is_reverse and read.next_reference_start < max_insert_len)
            if not (is_start_mapping or is_end_mapping):
                continue

            reference_name = read.reference_name
            if reference_name not in reference_lengths:
                reference_lengths[reference_name] = bamfile.get_reference_length(
                    reference_name
                )

            reference_length = reference_lengths[reference_name]

            read_start = read.reference_start + 1
            read_end = read.reference_end

            mate_start = read.next_reference_start + 1
            mate_end = mate_start + read.query_length - 1

            read_dist_5_prime = read_start
            read_dist_3_prime = reference_length - read_end

            mate_dist_5_prime = mate_start
            mate_dist_3_prime = reference_length - mate_end
            if (read_dist_5_prime + mate_dist_3_prime <= max_insert_len) or (
                mate_dist_5_prime + read_dist_3_prime <= max_insert_len
            ):
                local_circular_contigs.add(reference_name)
                if reference_name not in contig_reads_d.keys():
                    contig_reads_d[reference_name] = []
                contig_reads_d[reference_name].append(
                    [
                        read_start,
                        mate_start,
                        not read.is_reverse,
                        read.mate_is_reverse,
                    ]
                )
                if reference_name not in allcontig_reads_d.keys():
                    allcontig_reads_d[reference_name] = []
                allcontig_reads_d[reference_name].append(
                    [
                        read_start,
                        mate_start,
                        not read.is_reverse,
                        read.mate_is_reverse,
                    ]
                )

            else:
                if reference_name not in allcontig_reads_d.keys():
                    allcontig_reads_d[reference_name] = []
                allcontig_reads_d[reference_name].append(
                    [
                        read_start,
                        mate_start,
                        not read.is_reverse,
                        read.mate_is_reverse,
                    ]
                )
    return local_circular_contigs, contig_reads_d, allcontig_reads_d


# Function to process all BAM files in parallel
def process_all_bams_parallel(dir_bams, max_insert_len):
    """Parellelize the parsing of the bam files

    Args:
        dir_bams (str): Path that contains bam files from all samples.
        max_insert_len (int): Max distance between the end of the contig and the read mapping position for it to be considered.

    Returns:
        contig_reads_d: dictionary containing the reads and read position for circularized contigs.
        allcontig_reads_d: dictionary containing the reads and read position for all contigs.
    """
    bam_files = [
        os.path.join(dir_bams, f)
        for f in os.listdir(dir_bams)
        if "bam" in f and ".bai" not in f  # f.endswith("bam")
    ]
    contig_reads_d = dict()
    allcontig_reads_d = dict()

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_bam, bam, max_insert_len): bam for bam in bam_files
        }
        for future in as_completed(futures):
            bam = futures[future]
            try:
                (
                    local_circular_contigs,
                    contig_samplereads_d,
                    allcontig_samplereads_d,
                ) = future.result()

                # Update global sets with results from the parallel tasks
                circular_contigs.update(local_circular_contigs)

                # update contig_reads_d
                for c in contig_samplereads_d.keys():
                    if c not in contig_reads_d.keys():
                        contig_reads_d[c] = contig_samplereads_d[c]
                    else:
                        contig_reads_d[c] += contig_samplereads_d[c]

                # update allcontig_reads_d
                for c in allcontig_samplereads_d.keys():
                    if c not in allcontig_reads_d.keys():
                        allcontig_reads_d[c] = allcontig_samplereads_d[c]
                    else:
                        allcontig_reads_d[c] += allcontig_samplereads_d[c]

                print(f"{os.path.basename(bam)} completed")

            except Exception as e:
                print(f"Error processing {bam}: {e}")
                raise e
    return contig_reads_d, allcontig_reads_d


def save_circular_contig_clusters(circular_contigs, cl_f):
    with open(cl_f, "w") as f:
        f.write("%s\t%s\n" % ("clustername", "contigname"))
        for c in circular_contigs:
            f.write("%s_%s\t%s\n" % (c, "circ", c))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract circular contigs")

    # Add optional arguments with `--flags`
    parser.add_argument(
        "--dir_bams",
        required=True,
        type=str,
        help="Path to the directory that contains the bam files.",
    )

    parser.add_argument(
        "--max_insert",
        required=False,
        type=int,
        help="Max length of sequence between paired-end reads [50].",
        default=50,
    )

    parser.add_argument(
        "--outcls",
        type=str,
        required=True,
        help="Path were circular contigs will be dumped.",
    )

    parser.add_argument(
        "--outreads",
        type=str,
        help="Path were contig-read information will be stored for contigs circularizable.",
    )
    parser.add_argument(
        "--outallreads",
        type=str,
        help="Path were contig-read information will be stored for all contigs.",
    )

    # Parse the arguments
    args = parser.parse_args()

    ## Print git commit so we can debug
    commit_hash = get_git_commit(os.path.abspath(__file__))
    print(f"Git commit hash: {commit_hash}")

    max_insert_len = args.max_insert

    plas_recov = set()
    circular_contigs = set()  ## GLOBAL VARIABLE

    # Run the parallel processing function
    contig_reads_d, allcontig_reads_d = process_all_bams_parallel(
        args.dir_bams, max_insert_len
    )
    if args.outcls != None:
        save_circular_contig_clusters(circular_contigs, args.outcls)
    else:
        pass
    if args.outreads != None:
        with open(args.outreads, "w") as f:
            f.write(
                "%s\t%s\t%s\t%s\t%s\n"
                % ("contig", "start", "start_mate", "is_fw", "mate_is_rv")
            )
            for c in contig_reads_d.keys():
                for read_info in contig_reads_d[c]:
                    s, s_m, read_is_fw, mate_is_rv = read_info
                    f.write(
                        "%s\t%i\t%i\t%s\t%s\n"
                        % (c, s, s_m, str(read_is_fw), str(mate_is_rv))
                    )
    if args.outallreads != None:
        with open(args.outallreads, "w") as f:
            f.write(
                "%s\t%s\t%s\t%s\t%s\n"
                % ("contig", "start", "start_mate", "is_fw", "mate_is_rv")
            )
            for c in allcontig_reads_d.keys():
                for read_info in allcontig_reads_d[c]:
                    s, s_m, read_is_fw, mate_is_rv = read_info
                    f.write(
                        "%s\t%i\t%i\t%s\t%s\n"
                        % (c, s, s_m, str(read_is_fw), str(mate_is_rv))
                    )
