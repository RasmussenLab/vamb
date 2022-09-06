import os
import unittest
import io
import numpy as np

import vamb
import testtools
from vamb.parsecontigs import CompositionMetaData


class TestParseBam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp_metadata = CompositionMetaData(
            np.array(testtools.BAM_NAMES, dtype=object),
            np.array(testtools.BAM_SEQ_LENS),
            np.ones(len(testtools.BAM_SEQ_LENS), dtype=bool),
            2000,
        )

        cls.abundance = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, cls.comp_metadata, True, 0.0, 3
        )

    def test_refhash(self):
        m = self.comp_metadata
        cp = CompositionMetaData(m.identifiers, m.lengths, m.mask, m.minlength)
        cp.refhash = b"a" * 32  # write bad refhash
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(testtools.BAM_FILES, cp, True, 0.97, 1)

    def test_bad_metadata_mask(self):
        m = self.comp_metadata

        # If last element of mask is False, then the invariants of CompositionMetaData will
        # not hold after removing the last element of its mask, and that is NOT what we
        # are testing here.
        assert m.mask[-1]
        cp = CompositionMetaData(
            m.identifiers[:-1], m.lengths[:-1], m.mask[:-1], m.minlength
        )
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(testtools.BAM_FILES, cp, True, 0.97, 1)

    def test_badfile(self):
        with self.assertRaises(FileNotFoundError):
            vamb.parsebam.Abundance.from_files(
                ["noexist"], self.comp_metadata, True, 0.97, 1
            )

    # Minid too high
    def test_minid_off(self):
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(
                testtools.BAM_FILES, self.comp_metadata, True, 1.01, 1
            )

    def test_parse(self):
        self.assertEqual(self.abundance.matrix.shape, (25, 3))
        self.assertEqual(self.abundance.nseqs, 25)
        self.assertEqual(self.abundance.matrix.dtype, np.float32)

    def test_minid(self):
        abundance = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, self.comp_metadata, True, 0.95, 3
        )
        self.assertTrue(np.any(abundance.matrix < self.abundance.matrix))

    def test_save_load(self):
        buf = io.BytesIO()
        self.abundance.save(buf)
        buf.seek(0)

        # Bad refhash
        with self.assertRaises(ValueError):
            abundance2 = vamb.parsebam.Abundance.load(buf, b"a" * 32)

        buf.seek(0)
        abundance2 = vamb.parsebam.Abundance.load(buf, self.abundance.refhash)
        self.assertTrue(np.all(abundance2.matrix == self.abundance.matrix))
        self.assertTrue(np.all(abundance2.samplenames == self.abundance.samplenames))
        self.assertEqual(abundance2.refhash, self.abundance.refhash)
        self.assertEqual(abundance2.minid, self.abundance.minid)
