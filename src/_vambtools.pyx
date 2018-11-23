#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import array
from cpython cimport array

# Notes/assumptions for nucleotide counters:
# 1) Contigs ONLY consists of values 64, 67, 71, 84, NOTHING ELSE
# 2) They are at least 4 bp long

# Lookup in this array gives the index of the canonical tetranucleotide.
# E.g CCTA is the 92nd alphabetic 4mer, whose reverse complement, TAGG, is the 202nd.
# So the 92th and 202th value in this array is the same.
# Hence we can map 256 4mers to 136 normal OR reverse-complemented ones
cdef unsigned char[:] complementer_fourmer = bytearray([0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 11, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 23, 42, 43, 44, 7, 45, 46,
        47, 48, 49, 50, 51, 34, 52, 53, 54, 19, 55, 56, 57, 3, 58, 59,
        60, 57, 61, 62, 63, 44, 64, 65, 66, 30, 67, 68, 69, 14, 70, 71,
        72, 54, 73, 74, 75, 41, 76, 77, 78, 26, 79, 80, 66, 10, 81, 82,
        83, 51, 84, 85, 86, 37, 87, 88, 75, 22, 89, 90, 63, 6, 91, 92,
        93, 47, 94, 95, 83, 33, 96, 97, 72, 18, 98, 99, 60, 2, 100,
        101, 99, 56, 102, 103, 90, 43, 104, 105, 80, 29, 106, 107, 68,
        13, 108, 109, 97, 53, 110, 111, 88, 40, 112, 113, 77, 25, 114,
        105, 65, 9, 115, 116, 95, 50, 117, 118, 85, 36, 119, 111, 74,
        21, 120, 103, 62, 5, 121, 122, 92, 46, 123, 116, 82, 32, 124,
        109, 71, 17, 125, 101, 59, 1, 126, 125, 98, 55, 127, 120, 89,
        42, 128, 114, 79, 28, 129, 106, 67, 12, 130, 124, 96, 52, 131,
        119, 87, 39, 132, 112, 76, 24, 128, 104, 64, 8, 133, 123, 94,
        49, 134, 117, 84, 35, 131, 110, 73, 20, 127, 102, 61, 4, 135,
        121, 91, 45, 133, 115, 81, 31, 130, 108, 70, 16, 126, 100, 58, 0])

cpdef array.array zeros(typecode, int size):
    """Returns a zeroed-out array.array of size `size`"""

    cpdef array.array arr = array.array(typecode)
    array.resize(arr, size)
    array.zero(arr)

    return arr

cpdef int _overwrite_matrix(float[:,:] matrix, unsigned char[:] mask):
    """Given a float32 matrix and Uint8 mask, does the same as setting the first
    rows of matrix to matrix[mask], but in-place.
    This is only important to save on memory.
    """

    cdef int i, j, matrixindex
    cdef int length = matrix.shape[1]
    cdef int masklength = len(mask)

    for i in range(masklength):
        if mask[i] == 0:
            break

    if i == masklength:
        return masklength

    matrixindex = i

    for i in range(matrixindex, masklength):
        if mask[i] == 1:
            for j in range(length):
                matrix[matrixindex, j] = matrix[i, j]
            matrixindex += 1

    return matrixindex

cdef void c_kmercounts(unsigned char[:] bytesarray, int k, int[:] counts):
    """Count tetranucleotides of contig and put them in counts vector.

    The bytearray is expected to be np.uint8 of bytevalues of the contig.
    The counts is expected to be an array of 4^k 32-bit integers with value 0.
    """

    cdef int kmer = 0
    cdef int character
    cdef int charvalue
    cdef int i
    cdef int countdown = k - 1
    cdef int contiglength = len(bytesarray)
    cdef int mask = (1 << 2 * k - 2) - 1

    for i in range(contiglength):
        character = bytesarray[i]

        if character == 65: # A = 0b00 = 0
            charvalue = 0
        elif character == 67: # C = 0b01 = 1
            charvalue = 1
        elif character == 71: # G = 0b10 = 2
            charvalue = 2
        elif character == 84: # T = 0b11 = 3
            charvalue = 3
        else:
            kmer = 0
            countdown = k - 1
            continue

        kmer += charvalue

        # Countdown skips non-ACGT bases
        if countdown == 0:
            counts[kmer] += 1
            kmer &= mask # Remove leftmost base

        else:
            countdown -= 1

        kmer <<= 2 # Shift to prepare for next base

cpdef _kmercounts(bytearray sequence, int k):
    """Returns a 32-bit integer array containing the count of all kmers
    in the given bytearray.

    Only Kmers containing A, C, G, T (bytes 65, 67, 71, 84) are counted"""

    if k > 10 or k < 1:
        return ValueError('k must be between 1 and 10, inclusive.')

    counts = zeros('i', 4**k)

    cdef unsigned char[:] sequenceview = sequence
    cdef int[:] countview = counts

    c_kmercounts(sequenceview, k, countview)

    return counts

cdef void c_fourmer_freq(int[:] counts, float[:] result):
    """Puts kmercounts of k=4 in a nonredundant vector.

    The result is expected to be a 136 32-bit float vector
    The counts is expected to be an array of 256 32-bit integers
    """

    cdef int countsum = 0
    cdef int i

    for i in range(256):
        countsum += counts[i]

    if countsum == 0:
        return

    cdef float floatsum = <float>countsum

    for i in range(256):
        result[complementer_fourmer[i]] += counts[i] / floatsum

# Assining these arrays for each sequence takes about 6% longer time than
# having assigned them once in userspace. Worth it.
cpdef _fourmerfreq(bytearray sequence):
    """Returns float32 array of 136-length float32 representing the
    tetranucleotide (fourmer) frequencies of the DNA.
    Only fourmers containing A, C, G, T (bytes 65, 67, 71, 84) are counted"""

    counts = zeros('i', 256)
    frequencies = zeros('f', 136)

    cdef unsigned char[:] sequenceview = sequence
    cdef int[:] fourmercountview = counts
    cdef float[:] frequencyview = frequencies

    c_kmercounts(sequenceview, 4, fourmercountview)
    c_fourmer_freq(fourmercountview, frequencyview)

    return frequencies
