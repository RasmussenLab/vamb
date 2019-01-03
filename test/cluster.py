import sys
import os
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parentdir)
import vamb

latent = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_latent.npz'))

# Threshold works
extended = np.tile(latent, (15,1))
x = vamb.threshold.getthreshold(extended, vamb.cluster._pearson_distances,
                                samples=100, maxsize=500)

assert x == (0.12375, 1.0, 0.59)

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
                   (55, {5, 6, 8, 9, 11, 13, 14, 16, 18, 20, 23, 24, 26, 28, 29, 30, 34, 36, 41, 44, 48, 50, 55, 58, 60, 62, 65, 70}), (53, {35, 37, 7, 72, 45, 49, 19, 21, 53}),
                   (63, {64, 33, 3, 59, 46, 27, 63}),
                   (67, {67, 69, 38, 39, 52}),
                   (40, {40, 54})]

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
