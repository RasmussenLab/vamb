import sys
import os
import pysam
import numpy as np
import shutil

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parentdir)
import vamb

inpaths = [os.path.join(parentdir, 'test', 'data', i) for i in ('one.bam', 'two.bam', 'three.bam')]

file = pysam.AlignmentFile(inpaths[0])
records = list(file)
file.close()

# First some simple tests
assert [records[i].get_tag('AS') for i in range(10)] == [84, 93, 145, 151, 149, 151, 121, 151, 50, 130]

# Check _filter_segments work
file = pysam.AlignmentFile(inpaths[0])
above = sum(1 for i in (vamb.parsebam._filter_segments(file, minscore=100, minid=None)))
file.close()

assert above == 357

# Check _filter_segments raises an error on missing AS tag
file = pysam.AlignmentFile(os.path.join(parentdir, 'test', 'data', 'bad.bam'))
error_iterator = vamb.parsebam._filter_segments(file, minscore=100, minid=None)

try:
    next(error_iterator)
except KeyError:
    pass
else:
    raise AssertionError("Should have raised KeyError")

# With minscore = None, it shouldn't need to check AS tag
ok_iterator = vamb.parsebam._filter_segments(file, minscore=None, minid=None)
records_minus_one = sum(1 for i in ok_iterator)
assert records_minus_one == 21

file.close()

# Check _get_contig_rpkms work
# target_rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
# p, arr, length = vamb.parsebam._get_contig_rpkms(inpaths[0], outpath=None, minscore=50, minlength=100, minid=None)
#
# assert p == inpaths[0]
# assert len(arr) == length
# assert np.all(abs(arr - target_rpkm[:,0]) < 1e-8)
#
# # Check _read_bamfiles work
# inpaths = [os.path.join(parentdir, 'test', 'data', x + '.bam') for x in ('one', 'two', 'three')]
# rpkm = vamb.parsebam.read_bamfiles(inpaths, minscore=50, minlength=100, minid=None)
# assert np.all(abs(rpkm - target_rpkm) < 1e-8)
#
# rpkm = vamb.parsebam.read_bamfiles(inpaths, dumpdirectory='/tmp/dumpdirectory', minscore=50, minlength=100, minid=None)
# assert np.all(abs(rpkm - target_rpkm) < 1e-8)
# shutil.rmtree('/tmp/dumpdirectory')
