import sys
import os
import pysam
import numpy as np
import shutil

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(parentdir)
import vamb

inpaths = [os.path.join(parentdir, 'vamb', 'test', 'data', i) for i in ('one.bam', 'two.bam', 'three.bam')]

file = pysam.AlignmentFile(inpaths[0])
records = list(file)
file.close()

# First some simple tests
assert [records[i].get_tag('AS') for i in range(10)] == [84, 93, 145, 151, 149, 151, 121, 151, 50, 130]

# Check get_all_references work
assert vamb.parsebam._get_all_references(records[0]) == ['s30_NODE_1779_length_10476_cov_694.871']
assert vamb.parsebam._get_all_references(records[8]) == ['s30_NODE_1779_length_10476_cov_694.871',
                                                         's101_NODE_2230_length_4552_cov_5.17673',
                                                         's178_NODE_2230_length_4552_cov_5.17673']
assert vamb.parsebam._get_all_references(records[-1]) == ['s179_NODE_95_length_77775_cov_33.3836']

# Check _filter_segments work
file = pysam.AlignmentFile(inpaths[0])
above = sum(1 for i in (vamb.parsebam._filter_segments(file, minscore=100)))
file.close()

assert above == 357

# Check _filter_segments raises an error on missing AS tag
file = pysam.AlignmentFile(os.path.join(parentdir, 'vamb', 'test', 'data', 'bad.bam'))
error_iterator = vamb.parsebam._filter_segments(file, minscore=100)

try:
    next(error_iterator)
except KeyError:
    pass
else:
    raise AssertionError("Should have raised KeyError")

# With minscore = 0, it shouldn't need to check AS tag
ok_iterator = vamb.parsebam._filter_segments(file, minscore=0)
records_minus_one = sum(1 for i in ok_iterator)
assert records_minus_one == 21

file.close()

# Check _get_contig_rpkms work
target_rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'vamb', 'test', 'data', 'target_rpkm.npz'))
p, arr, length  = vamb.parsebam._get_contig_rpkms(inpaths[0])

assert p == inpaths[0]
assert len(arr) == length
assert np.all(abs(arr - target_rpkm[:,0]) < 1e-8)

# Check _read_bamfiles work
inpaths = [os.path.join(parentdir, 'vamb', 'test', 'data', x + '.bam') for x in ('one', 'two', 'three')]
rpkm = vamb.parsebam.read_bamfiles(inpaths)
assert np.all(abs(rpkm - target_rpkm) < 1e-8)

rpkm = vamb.parsebam.read_bamfiles(inpaths, dumpdirectory='/tmp/dumpdirectory')
assert np.all(abs(rpkm - target_rpkm) < 1e-8)
shutil.rmtree('/tmp/dumpdirectory')
