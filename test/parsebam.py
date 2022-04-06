import os
import unittest
import numpy as np

import vamb


PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PARENTDIR, "test", "data", "bam")


class TestParseBam(unittest.TestCase):
    files = [os.path.join(DATADIR, i) for i in os.listdir(DATADIR)]
    lens = [
        2271, 3235, 3816, 2625, 2716,
        4035, 3001, 2583, 5962, 3774,
        2150, 2161, 2218, 2047, 5772,
        2633, 3400, 3502, 2103, 4308,
        3061, 2464, 4099, 2640, 2449
    ]

    def test_refhash(self):
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles(self.files, b"", None, 0.97, None, 1)

    def test_badfile(self):
        with self.assertRaises(FileNotFoundError):
            vamb.parsebam.read_bamfiles(["nofile"], None, None, 0.97, None, 1)

    def test_inconsistent_args(self):
        # Minlength, but no lengths given
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles(self.files, None, 2500, 0.97, None, 1)

        # Lengths, but no minlength given
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles(self.files, None, None, 0.97, np.array([50]), 1)

        # Too low minlength
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles(self.files, None, 50, 0.97, np.array([50]), 1)

        # Minid too high
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles(self.files, None, None, 1.01, None, 1)

    def test_wrong_lengths(self):
        with self.assertRaises(ValueError):
            vamb.parsebam.read_bamfiles([self.files[0]], None, 2500, 0.97, [3000, 1000], 1)


    @classmethod
    def setUpClass(cls):
        cls.arr = vamb.parsebam.read_bamfiles(
            cls.files,
            bytes.fromhex("129db373e4a0dd86cc3217aa9af7b1c2"),
            None,
            0.0,
            None,
            3
        )

    def test_parse(self):
        self.assertEqual(self.arr.shape, (25, 3))
        self.assertEqual(self.arr.dtype, np.float32)

    def test_minlength(self):
        length = 3502
        arr = vamb.parsebam.read_bamfiles(
            self.files, None, length, 0.0, self.lens, 3)
        mask = [i >= length for i in self.lens]
        self.assertTrue(np.all(self.arr[mask] == arr))

    def test_minid(self):
        arr = vamb.parsebam.read_bamfiles(
            self.files, None, None, 0.95, None, 3)
        self.assertTrue(np.any(arr < self.arr))
