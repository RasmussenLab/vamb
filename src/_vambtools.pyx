#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import array
from cpython cimport array

# Notes/assumptions for nucleotide counters:
# 1) Contigs ONLY consists of values 64, 67, 71, 84, all others are skipped

cpdef array.array zeros(typecode, int size):
    """Returns a zeroed-out array.array of size `size`"""

    cpdef array.array arr = array.array(typecode)
    array.resize(arr, size)
    array.zero(arr)

    return arr

cpdef int _overwrite_matrix(float[:,::1] matrix, unsigned char[::1] mask):
    """Given a float32 matrix and Uint8 mask, does the same as setting the first
    rows of matrix to matrix[mask], but in-place.
    This is only important to save on memory.
    """

    cdef int i = 0
    cdef int j = 0
    cdef int matrixindex = 0
    cdef int length = matrix.shape[1]
    cdef int masklength = len(mask)

    # First skip to the first zero in the mask, since the matrix at smaller
    # indices than this should remain untouched.
    for i in range(masklength):
        if mask[i] == 0:
            break

    # If the mask is all true, don't touch array.
    if i == masklength:
        return masklength

    matrixindex = i

    for i in range(matrixindex, masklength):
        if mask[i] == 1:
            for j in range(length):
                matrix[matrixindex, j] = matrix[i, j]
            matrixindex += 1

    return matrixindex

cdef void c_kmercounts(unsigned char[::1] bytesarray, int k, int[::1] counts):
    """Count tetranucleotides of contig and put them in counts vector.

    The bytearray is expected to be np.uint8 of bytevalues of the contig.
    The counts is expected to be an array of 4^k 32-bit integers with value 0.
    """

    cdef unsigned int kmer = 0
    cdef int character
    cdef int charvalue
    cdef int i
    cdef int countdown = k-1
    cdef int contiglength = len(bytesarray)
    cdef int mask = (1 << (2 * k)) - 1
    cdef unsigned int* lut = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 4:
            countdown = k

        kmer = ((kmer << 2) | charvalue) & mask

        if countdown == 0:
            counts[kmer] += 1
        else:
            countdown -= 1

cpdef _kmercounts(bytearray sequence, int k):
    """Returns a 32-bit integer array containing the count of all kmers
    in the given bytearray.

    Only Kmers containing A, C, G, T (bytes 65, 67, 71, 84) are counted"""

    if k > 10 or k < 1:
        return ValueError('k must be between 1 and 10, inclusive.')

    counts = zeros('i', 4**k)

    cdef unsigned char[::1] sequenceview = sequence
    cdef int[::1] countview = counts

    c_kmercounts(sequenceview, k, countview)

    return counts
