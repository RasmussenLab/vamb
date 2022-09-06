__doc__ = """Create kernel for use in kmer frequencies.
Method copied from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2765972/

Principle:
There are 256 tetranucleotides, so a frequency distribution (tetranucleotide frequency, TNF)
is a length 256 vector. But the individual TNFs are not independent. For example, AAAT
must correlate highly with AATA. The TNFs are subject to at least 3 linear constrains:

1) The vector must sum to one. We simply shift the TNF down by 1/256 to make it sum to zero
for simplicity instead.
2) We cannot distinguish between a kmer and its reverse complement because the sequencede
strand is arbitrary. So we must count e.g. AGAT as one half of AGAT and one half ATCT.
So each kmer's frequency is the same as its reverse-complement.
3) Every time a kmer is observed, the next kmer must have three overlapping nucleotides.
E.g. every observation of AGAT is followed by GATA, GATC, GATG or GATT. Same for previous
kmer. in other words, sum(xABC) = sum(ABCx).
This is not true right at the ends of the sequences because the kmers stop eventually, but
that can be considered a measurement error, and we don't care about it.

We list these linear constrains and produce kernel L that works on tnf matrix T such that
TL = P, a smaller projected TNF space.

Notably, for constraint 2 to be true, we need to average the frequency between a kmer
and its reverse complement. We can do this with a matrix multiply with an averaging kernel
R. So:

P = (TR)L = T(RL) = TK

We thus calculate K = RL and save this for use in Vamb for projection.
"""

from os.path import abspath, dirname, join
import numpy as np
import itertools
from scipy.linalg import null_space


def reverse_complement(nuc):
    table = str.maketrans("ACGT", "TGCA")
    return nuc[::-1].translate(table)


def all_kmers(k):
    for i in itertools.product("ACGT", repeat=k):
        yield ("".join(i))


def create_projection_kernel():
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}
    linear_equations = list()

    # Constraint one: Frequencies sum to one (or in this scaled case, zero)
    linear_equations.append([1] * 256)

    # Constaint two: Frequencies are same as that of reverse complement
    for kmer in all_kmers(4):
        revcomp = reverse_complement(kmer)

        # Only look at canonical kmers - this makes no difference
        if kmer >= revcomp:
            continue

        line = [0] * 256
        line[indexof[kmer]] = 1
        line[indexof[revcomp]] = -1
        linear_equations.append(line)

    # Constraint three: sum(ABCx) = sum(xABC)
    for trimer in all_kmers(3):
        line = [0] * 256
        for suffix in "ACGT":
            line[indexof[trimer + suffix]] += 1
        for prefix in "ACGT":
            line[indexof[prefix + trimer]] += -1
        linear_equations.append(line)

    linear_equations = np.array(linear_equations)
    kernel = null_space(linear_equations).astype(np.float32)
    assert kernel.shape == (256, 103)
    return kernel


def create_rc_kernel():
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}
    rc_matrix = np.zeros((256, 256), dtype=np.float32)
    for col, kmer in enumerate(all_kmers(4)):
        revcomp = reverse_complement(kmer)
        rc_matrix[indexof[kmer], col] += 0.5
        rc_matrix[indexof[revcomp], col] += 0.5

    return rc_matrix


def create_dual_kernel():
    return np.dot(create_rc_kernel(), create_projection_kernel())


dual_kernel = create_dual_kernel()

# Prevent overwriting kernel when running tests
if __name__ == "__main__":
    path = join(dirname(dirname(abspath(__file__))), "vamb", "kernel.npz")
    np.savez_compressed(path, dual_kernel)
