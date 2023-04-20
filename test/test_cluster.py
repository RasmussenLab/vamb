import unittest
import numpy as np
from hashlib import md5
import random
import string

import vamb


class TestClusterer(unittest.TestCase):
    data = np.random.random((1024, 40)).astype(np.float32)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data.astype(np.float64))

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, maxsteps=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, windowsize=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, minsuccesses=0)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(self.data, minsuccesses=5, windowsize=4)

        with self.assertRaises(ValueError):
            vamb.cluster.ClusterGenerator(np.random.random((0, 40)))

    # In the code, in __init__ of the cluster generator, the input matrix
    # is shuffled, and an index array is permuted to keep track of which
    # indices was which.
    # This depends on the implementation of shuffling and permute being
    # the same, which I test here.
    def test_shuffling(self):
        cp = self.data.copy()
        np.random.RandomState(0).shuffle(cp)
        indices = np.random.RandomState(0).permutation(len(cp))
        cplike = self.data[indices]
        self.assertTrue(np.all(cplike == cp))

    def test_basics(self):
        clstr = vamb.cluster.ClusterGenerator(self.data)
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
        clstr = vamb.cluster.ClusterGenerator(self.data)
        self.assertTrue(np.any(np.abs(self.data - clstr.matrix.numpy()) > 0.001))
        clstr = vamb.cluster.ClusterGenerator(copy, destroy=True)
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
        vamb.cluster.ClusterGenerator(self.data)
        self.assertEqual(hash_before, md5(self.data.data.tobytes()).digest().hex())
        cp = self.data.copy()
        vamb.cluster.ClusterGenerator(cp, destroy=True)
        hash_after = md5(cp.data.tobytes()).digest().hex()
        self.assertNotEqual(hash_before, hash_after)

        # Rows are permuted by the clusterer. We use xor to check the rows
        # are still essentially the same.
        before_xor = self.xor_rows_hash(cp)
        vamb.cluster.ClusterGenerator(cp, destroy=True, normalized=True)
        self.assertEqual(before_xor, self.xor_rows_hash(cp))

    def test_cluster(self):
        x = next(vamb.cluster.ClusterGenerator(self.data))
        self.assertIsInstance(x.members, np.ndarray)
        med, st = x.as_tuple()
        self.assertIsInstance(st, set)
        self.assertEqual(set(x.members), st)
        self.assertIn(med, st)


class TestPairs(unittest.TestCase):
    n_samples = 1024
    data = np.random.random((n_samples, 40)).astype(np.float32)

    @staticmethod
    def randstring(len):
        return "".join(random.choices(string.ascii_letters, k=len))

    def test_too_few_names(self):
        clstr = vamb.cluster.ClusterGenerator(self.data)
        nameset = {self.randstring(10) for i in range(len(self.data) - 1)}
        with self.assertRaises(ValueError):
            list(vamb.cluster.pairs(clstr, list(nameset)))

    def test_pairs(self):
        clstr = vamb.cluster.ClusterGenerator(self.data)
        nameset = {self.randstring(10) for i in range(len(self.data))}
        pairs = list(vamb.cluster.pairs(clstr, list(nameset)))

        medoid, members = pairs[0]
        self.assertIsInstance(medoid, str)
        self.assertIsInstance(members, set)
        self.assertTrue(all(len(i[1]) > 0 for i in pairs))
        self.assertTrue(sum(map(lambda i: len(i[1]), pairs)), len(nameset))
        allmembers = set()
        for medoid, mems in pairs:
            allmembers.update(mems)
        self.assertEqual(allmembers, nameset)
