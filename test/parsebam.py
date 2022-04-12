import os
import unittest
import io
import numpy as np

import vamb
from vamb.parsecontigs import CompositionMetaData


PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PARENTDIR, "test", "data", "bam")


class TestParseBam(unittest.TestCase):
    files = [os.path.join(DATADIR, i) for i in os.listdir(DATADIR)]

    @classmethod
    def setUpClass(cls):
        cls.lens = [
            2271, 3235, 3816, 2625, 2716,
            4035, 3001, 2583, 5962, 3774,
            2150, 2161, 2218, 2047, 5772,
            2633, 3400, 3502, 2103, 4308,
            3061, 2464, 4099, 2640, 2449
        ]

        cls.names = [
            "S27C175628",
            "S27C95602",
            "S27C25358",
            "S26C115410",
            "S4C529736",
            "S27C181335",
            "S4C222286",
            "S27C38468",
            "S11C13125",
            "S4C480978",
            "S27C255582",
            "S27C170328",
            "S7C221395",
            "S26C281881",
            "S12C228927",
            "S26C86604",
            "S27C93037",
            "S9C124493",
            "S27C236159",
            "S27C214882",
            "S7C273086",
            "S8C93079",
            "S12C85159",
            "S10C72456",
            "S27C19079",
        ]

        cls.comp_metadata = CompositionMetaData(
            np.array(cls.names, dtype=object), np.array(cls.lens), np.ones(len(cls.lens), dtype=bool), 2000
        )

        cls.abundance = vamb.parsebam.Abundance.from_files(cls.files, cls.comp_metadata, True, 0.0, 3)

    def test_refhash(self):
        m = self.comp_metadata
        cp = CompositionMetaData(m.identifiers, m.lengths, m.mask, m.minlength)
        cp.refhash = b"a" * 32 # write bad refhash
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(self.files, cp, True, 0.97, 1)

    def test_badfile(self):
        with self.assertRaises(FileNotFoundError):
            vamb.parsebam.Abundance.from_files(["noexist"], self.comp_metadata, True, 0.97, 1)

    # Minid too high
    def test_minid_off(self):
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(self.files, self.comp_metadata, True, 1.01, 1)

    def test_parse(self):
        self.assertEqual(self.abundance.matrix.shape, (25, 3))
        self.assertEqual(self.abundance.matrix.dtype, np.float32)

    def test_minid(self):
        abundance = vamb.parsebam.Abundance.from_files(self.files, self.comp_metadata, True, 0.95, 3)
        self.assertTrue(np.any(abundance.matrix < self.abundance.matrix))

    def test_save_load(self):
        buf = io.BytesIO()
        self.abundance.save(buf)
        buf.seek(0)

        # Bad refhash
        with self.assertRaises(ValueError):
            abundance2 = vamb.parsebam.Abundance.load(buf, b'a'*32)

        buf.seek(0)
        abundance2 = vamb.parsebam.Abundance.load(buf, self.abundance.refhash)
        self.assertTrue(np.all(abundance2.matrix == self.abundance.matrix))
        self.assertEqual(abundance2.refhash, self.abundance.refhash)