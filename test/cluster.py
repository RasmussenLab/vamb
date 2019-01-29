import sys
import os
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(parentdir)
import vamb

latent = vamb.vambtools.read_npz(os.path.join(parentdir, 'vamb', 'test', 'data', 'target_latent.npz'))

# Threshold works
extended = np.tile(latent, (15,1))
x = vamb.threshold.getthreshold(extended, vamb.cluster._pearson_distances,
                                samples=100, maxsize=500)
assert x == (0.10125, 1.0, 0.58)

# Check: Threshold fails with too few thresholds returned
try:
    x = vamb.threshold.getthreshold(latent, vamb.cluster._pearson_distances, samples=50, maxsize=1)
except vamb.threshold.TooLittleData as e:
    assert e.args == ('Less than 5 samples returned a threshold',)
else:
    raise AssertionError('Should have failed no samples returned threshold')

# Check: It fails with fewer sequences than nsamples
try:
    vamb.cluster.cluster(latent, nsamples=len(latent)+1)
except ValueError as e:
    assert e.args == ('Cannot estimate from less than nsamples contigs',)
else:
    raise AssertionError('Should have failed when nsamples > len(latent)')

# Check clustering in general
x = vamb.cluster.cluster(latent, threshold=0.1)
assert list(x) == [(20,
  {5, 6, 8, 9, 11, 13, 14, 15, 16, 18, 20, 23, 24, 26, 28, 29, 30, 34, 36, 41,
   43, 47, 48, 50, 55, 58, 62, 64, 65, 71}),
 (40, {40, 54, 60}),
 (44, {4, 10, 17, 27, 31, 42, 44, 57, 66, 68, 69, 70, 73}),
 (35, {7, 19, 21, 35, 37, 49, 51}),
 (32, {1, 2, 12, 22, 25, 32, 33, 45, 46, 56, 61, 63, 67}),
 (72, {38, 39, 52, 53, 72}),
 (59, {3, 59})]

x = vamb.cluster.cluster(extended, nsamples=100)
assert next(x) == (589, {1027, 516, 5, 1031, 520, 9, 1056, 545, 34, 1063, 552,
                         41, 1070, 559, 48, 1077, 566, 55, 589, 78, 593, 82,
                         618, 107, 625, 114, 632, 121, 639, 128, 662, 151, 666,
                         155, 691, 180, 698, 187, 705, 194, 712, 201, 735, 224,
                         739, 228, 764, 253, 771, 260, 778, 267, 785, 274, 808,
                         297, 812, 301, 837, 326, 844, 333, 851, 340, 858, 347,
                         881, 370, 885, 374, 910, 399, 917, 406, 924, 413, 931,
                         420, 954, 443, 958, 447, 983, 472, 990, 479, 997, 486,
                         1004, 493})

# Check for bad inputs
try:
    vamb.cluster.cluster(latent, threshold=0.1, labels=np.arange(10))
except ValueError:
    pass
else:
    raise AssertionError('Should have failed when passed too few labels')

badlabels = [str(i) for i in range(len(latent))]
badlabels[20] = '10'
try:
    vamb.cluster.cluster(latent, threshold=0.1, labels=badlabels)
except ValueError:
    pass
else:
    raise AssertionError('Should have failed when passed non-unique labels')
