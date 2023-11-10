import unittest
import numpy as np
from hashlib import md5

import vamb


class TestClusterer(unittest.TestCase):
    # This seed has been set just so the unit tests runs faster.
    # How many iterations of the clustering depends on the input data
    rng = np.random.RandomState(5)
    data = rng.random((1024, 40)).astype(np.float32)
    lens = rng.randint(500, 1000, size=1024)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data.astype(np.float64), self.lens)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, self.lens, maxsteps=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, self.lens, windowsize=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, self.lens, minsuccesses=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(
                self.data, self.lens, minsuccesses=5, windowsize=4
            )

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(
                np.random.random((0, 40)), np.array([], dtype=int)
            )

    def test_basics(self):
        clstr = vamb.cluster.ClusterGenerator(self.data, self.lens)
        self.assertIs(clstr, iter(clstr))

        x = next(clstr)
        self.assertIsInstance(x, vamb.cluster.Cluster)

        clusters = list(clstr)
        clusters.append(x)

        # All members are clustered
        self.assertEqual(sum(map(lambda x: len(x.members), clusters)), len(self.data))

        # Elements of members are exactly the matrix row indices
        mems = set()
        for i in clusters:
            mems.update(i.members)
        self.assertEqual(mems, set(range(len(self.data))))

    def test_detruction(self):
        copy = self.data.copy()
        clstr = vamb.cluster.ClusterGenerator(self.data, self.lens)
        self.assertTrue(np.any(np.abs(self.data - clstr.matrix.numpy()) > 0.001))
        clstr = vamb.cluster.ClusterGenerator(copy, self.lens, destroy=True)
        self.assertTrue(np.all(np.abs(copy - clstr.matrix.numpy()) < 1e-6))
        self.assertTrue(np.any(np.abs(self.data - clstr.matrix.numpy()) > 0.001))

    @staticmethod
    def xor_rows_hash(matrix):
        m = np.frombuffer(matrix.copy().data, dtype=np.uint32)
        m.shape = matrix.shape
        v = m[0]
        for i in range(1, len(m)):
            v ^= m[i]
        return md5(v).digest().hex()

    def test_normalization(self):
        hash_before = md5(self.data.data.tobytes()).digest().hex()
        vamb.cluster.ClusterGenerator(self.data, self.lens)
        self.assertEqual(hash_before, md5(self.data.data.tobytes()).digest().hex())
        cp = self.data.copy()
        vamb.cluster.ClusterGenerator(cp, self.lens, destroy=True)
        hash_after = md5(cp.data.tobytes()).digest().hex()
        self.assertNotEqual(hash_before, hash_after)

        # Rows are permuted by the clusterer. We use xor to check the rows
        # are still essentially the same.
        before_xor = self.xor_rows_hash(cp)
        vamb.cluster.ClusterGenerator(cp, self.lens, destroy=True, normalized=True)
        self.assertEqual(before_xor, self.xor_rows_hash(cp))

    def test_cluster(self):
        x = next(vamb.cluster.ClusterGenerator(self.data, self.lens))
        self.assertIsInstance(x.members, np.ndarray)
