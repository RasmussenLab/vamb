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
assert list(x) == [(22, {1, 2, 68, 71, 73, 43, 12, 22, 25, 61}),
(56, {32, 66, 4, 42, 10, 47, 15, 17, 51, 56, 57, 31}),
(55, {5, 6, 8, 9, 11, 13, 14, 16, 18, 20, 23, 24, 26, 28, 29, 30, 34, 36, 41, 44, 48, 50, 55, 58, 60, 62, 65, 70}),
(53, {35, 37, 7, 72, 45, 49, 19, 21, 53}),
(63, {64, 33, 3, 59, 46, 27, 63}),
(67, {67, 69, 38, 39, 52}),
(54, {40, 54})]

x = vamb.cluster.cluster(extended, nsamples=100)
assert next(x) == (207, {1, 2, 1031, 1034, 12, 526, 1040, 1043, 533, 22, 536, 25, 540, 543, 32,
       1058, 556, 45, 1072, 561, 1082, 1083, 572, 61, 574, 1086, 578, 67, 585,
       74, 75, 85, 89, 601, 95, 609, 98, 613, 105, 109, 118, 631, 637, 640, 134,
       651, 140, 654, 147, 158, 679, 168, 682, 171, 689, 178, 191, 709, 718, 207,
       720, 213, 732, 221, 231, 236, 752, 753, 247, 763, 769, 262, 264, 272, 275,
       787, 788, 791, 286, 293, 294, 300, 812, 304, 823, 317, 324, 839, 332, 845,
       337, 855, 348, 860, 867, 870, 359, 360, 875, 366, 882, 884, 886, 377, 379,
       898, 905, 908, 397, 410, 411, 414, 423, 426, 431, 433, 449, 451, 460, 463,
       978, 982, 470, 983, 480, 993, 483, 999, 1004, 492, 497, 499, 503, 505, 1021})

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
