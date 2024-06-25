import unittest
import io
import numpy as np
import random
import tempfile

import vamb
import testtools
from vamb.parsecontigs import CompositionMetaData


class TestParseBam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        minlen = 3000
        mask = np.array(
            list(map(lambda x: x >= minlen, testtools.BAM_SEQ_LENS)), dtype=bool
        )
        cls.comp_metadata = CompositionMetaData(
            np.array(
                [i for (i, m) in zip(testtools.BAM_NAMES, mask) if m], dtype=object
            ),
            np.array([i for (i, m) in zip(testtools.BAM_SEQ_LENS, mask) if m]),
            mask,
            minlen,
        )

        cls.abundance = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, "/tmp/bam_tmpfile", cls.comp_metadata, True, 0.0, 2
        )

    def test_refhash(self):
        m = self.comp_metadata
        cp = CompositionMetaData(m.identifiers, m.lengths, m.mask, m.minlength)
        # Change the refnames slighty
        cp.identifiers = cp.identifiers.copy()
        cp.identifiers[3] = cp.identifiers[3] + "w"
        cp.refhash = vamb.vambtools.RefHasher.hash_refnames(cp.identifiers)
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(
                testtools.BAM_FILES, None, cp, True, 0.97, 4
            )

        ab2 = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, None, cp, False, 0.97, 4
        )
        self.assertEqual(self.abundance.refhash, ab2.refhash)

    def test_bad_metadata_mask(self):
        m = self.comp_metadata

        # If last element of mask is False, then the invariants of CompositionMetaData will
        # not hold after removing the last element of its mask, and that is NOT what we
        # are testing here.
        assert list(m.mask[-3:]) == [True, False, False]
        cp = CompositionMetaData(
            m.identifiers[:-1], m.lengths[:-1], m.mask[:-3], m.minlength
        )
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(
                testtools.BAM_FILES, None, cp, True, 0.97, 4
            )

    def test_badfile(self):
        with self.assertRaises(BaseException):
            vamb.parsebam.Abundance.from_files(
                ["noexist"], None, self.comp_metadata, True, 0.97, 1
            )

    # Minid too high
    def test_minid_off(self):
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_files(
                testtools.BAM_FILES, None, self.comp_metadata, True, 1.01, 4
            )

    def test_parse(self):
        nm = sum(self.comp_metadata.mask)
        self.assertEqual(nm, 12)

        self.assertEqual(self.abundance.matrix.shape, (nm, 3))
        self.assertEqual(self.abundance.nseqs, nm)
        self.assertEqual(self.abundance.matrix.dtype, np.float32)
        self.assertEqual(self.abundance.nsamples, 3)

    def test_minid(self):
        abundance = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, None, self.comp_metadata, True, 0.95, 3
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

    def test_parse_from_aemb(self):
        abundance = vamb.parsebam.Abundance.from_aemb(
            testtools.AEMB_FILES, self.comp_metadata
        )

        self.assertTrue(abundance.refhash == self.comp_metadata.refhash)
        self.assertTrue(
            (abundance.samplenames == [str(i) for i in testtools.AEMB_FILES]).all()
        )

        manual_matrix = []
        for file in testtools.AEMB_FILES:
            manual_matrix.append([])
            with open(file) as f:
                for line in f:
                    (_, ab) = line.split()
                    manual_matrix[-1].append(float(ab))
        manual_matrix = list(map(list, zip(*manual_matrix)))

        self.assertTrue((np.abs(manual_matrix - abundance.matrix) < 1e-5).all())

        # Good headers, but in other order
        names = list(self.comp_metadata.identifiers)
        random.shuffle(names)
        files = [self.make_aemb_file(names) for i in range(2)]
        a = vamb.parsebam.Abundance.from_aemb(
            [f.name for f in files], self.comp_metadata
        )
        self.assertIsInstance(a, vamb.parsebam.Abundance)

    def make_aemb_file(self, headers):
        file = tempfile.NamedTemporaryFile(mode="w+")
        for header in headers:
            n = random.random() * 10
            print(header, "\t", str(n), file=file)
        file.seek(0)
        return file

    def test_bad_aemb(self):
        # One header is wrong
        names = list(self.comp_metadata.identifiers)
        names[-4] = names[-4] + "a"
        files = [self.make_aemb_file(names) for i in range(2)]
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_aemb(
                [f.name for f in files], self.comp_metadata
            )

        # Too many headers
        names = list(self.comp_metadata.identifiers)
        names.append(names[0])
        files = [self.make_aemb_file(names) for i in range(2)]
        with self.assertRaises(ValueError):
            vamb.parsebam.Abundance.from_aemb(
                [f.name for f in files], self.comp_metadata
            )
