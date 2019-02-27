import sys
import os
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb

fasta_path = os.path.join(parentdir, 'test', 'data', 'fasta.fna')

# Test it fails with non binary opened
file = open(fasta_path)
try:
    entries = vamb.vambtools.byte_iterfasta(file)
    next(entries)
except TypeError:
    pass
else:
    raise AssertionError('Should have failed when opening FASTA file in text mode')

# Open and read file
with open(fasta_path, 'rb') as file:
    contigs = list(vamb.vambtools.byte_iterfasta(file))

# Lengths are correct
assert [len(i) for i in contigs] == [100, 100, 150, 99, 0, 150]

# Correctly translates ambiguous nucleotides to Ns
contig3 = str(contigs[2].sequence)
for invalid in 'SWKMYRBDHV':
    assert contig3.count(invalid) == 0
assert contig3.count('N') == 11

# Correctly counts 4mers
contig3_fourmers_expected = """000000210000000100000102120001001010011000100112000100011
0010001000101000001000000100111100020200001012100000001102111011100010011000000
0101010000001001020111010010000111001000010010000010200001000100211110101100010
10000120010100010001000010011110100000100""".replace('\n', '')
contig3_fourmers_observed = contigs[2].kmercounts(4)
contig3_tnf_expected = """0001103301000001000101021300121202102201120111011001100001000010
002100011111100300001120000202221101210022010011022100022011130031100200""".replace('\n', '')
contig3_tnf_observed = contigs[2].fourmer_freq()

for i, j in zip(contig3_fourmers_expected, contig3_fourmers_observed):
    assert int(i) == j

for i, j in zip(contig3_tnf_expected, contig3_tnf_observed):
    assert int(i)/103 - 1e-8 < j < int(i)/103 + 1e-8 # float weirdness

assert all(i == 0 for i in contigs[3].kmercounts(4))
assert all(i == 0 for i in contigs[3].fourmer_freq())

# Correctly deals with lowercase
assert contigs[2].sequence == contigs[5].sequence

# Correctly fails at opening bad fasta file
badfasta_path = os.path.join(parentdir, 'test', 'data', 'badfasta.fna')
with open(badfasta_path, 'rb') as file:
    try:
        entries = list(vamb.vambtools.byte_iterfasta(file))
    except ValueError as error:
        assert error.args == ("Non-IUPAC DNA in line 7: 'P'",)
    else:
        raise AssertionError("Didn't fail at opening fad FASTA file")

# Reader works well
gzip_path = os.path.join(parentdir, 'test', 'data', 'fasta.fna.gz')

with vamb.vambtools.Reader(fasta_path, 'rb') as file:
    contigs2 = list(vamb.vambtools.byte_iterfasta(file))

assert len(contigs) == len(contigs2)
assert all(i.sequence == j.sequence for i,j in zip(contigs, contigs2))

with vamb.vambtools.Reader(gzip_path, 'rb') as file:
    contigs2 = list(vamb.vambtools.byte_iterfasta(file))

assert len(contigs) == len(contigs2)
assert all(i.sequence == j.sequence for i,j in zip(contigs, contigs2))

# Test read_contigs
for preallocate in True, False:
    with open(fasta_path, 'rb') as file:
        tnf, contignames, contiglengths = vamb.parsecontigs.read_contigs(file, preallocate=preallocate, minlength=100)

    assert len(tnf) == len([i for i in contigs if len(i) >= 100])
    assert all(i-1e-8 < j < i+1e-8 for i,j in zip(tnf[2], contig3_tnf_observed))

    assert contignames == ['Sequence1_100nt_no_special',
     'Sequence2 100nt whitespace in header',
     'Sequence3 150 nt, all ambiguous bases',
     'Sequence6 150 nt, same as seq4 but mixed case']

    assert np.all(contiglengths == np.array([len(i) for i in contigs if len(i) >= 100]))

bigpath = os.path.join(parentdir, 'test', 'data', 'bigfasta.fna.gz')
with vamb.vambtools.Reader(bigpath, 'rb') as f:
    tnf, _, __ = vamb.parsecontigs.read_contigs(f)

target_tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
assert np.all(abs(tnf - target_tnf) < 1e-8)
