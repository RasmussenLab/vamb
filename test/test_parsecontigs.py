import io
import unittest
import random
import numpy as np

import testtools
from vamb.parsecontigs import Composition, CompositionMetaData


class TestReadContigs(unittest.TestCase):
    records = []
    large_io = io.BytesIO()
    io = io.BytesIO()

    @classmethod
    def setUpClass(cls):
        rng = random.Random()
        for i in range(random.randrange(1400, 1500)):
            cls.records.append(testtools.make_randseq(rng, 400, 600))

        for i in cls.records:
            cls.io.write(i.format().encode())
            cls.io.write(b"\n")

        for i in range(25_000):
            record = testtools.make_randseq(rng, 250, 300)
            cls.large_io.write(record.format().encode())
            cls.large_io.write(b"\n")

    def setUp(self):
        self.io.seek(0)
        self.large_io.seek(0)

    def test_only_ns(self):
        file = io.BytesIO()
        file.write(b">abc\n")
        file.write(b"N" * 2500)
        file.write(b"\n")
        file.seek(0)

        with self.assertRaises(ValueError):
            Composition.from_file(file, None)

    def test_unique_names(self):
        with self.assertRaises(ValueError):
            CompositionMetaData(
                np.array(["foo", "foo"], dtype=object),
                np.array([1000, 1000]),
                np.array([True, True], dtype=bool),
                1000,
            )

    def test_filter_minlength(self):
        minlen = 500
        composition = Composition.from_file(self.io, None, minlength=450)
        md = composition.metadata
        hash1 = md.refhash

        composition.filter_min_length(minlen)
        n_initial_seq = md.nseqs

        hash2 = md.refhash
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(len(md.identifiers), len(md.lengths))
        self.assertEqual(md.nseqs, md.mask.sum())
        self.assertLessEqual(minlen, composition.metadata.lengths.min(initial=minlen))
        self.assertEqual(len(md.mask), len(self.records))

        # NB: Here we filter metadata without filtering the composition.
        # That means from this point on, the metadata and comp is out of sync,
        # and comp is invalid.
        md.filter_min_length(minlen + 50)
        self.assertEqual(len(md.identifiers), len(md.lengths))
        self.assertEqual(md.nseqs, md.mask.sum())
        self.assertLessEqual(
            minlen, composition.metadata.lengths.min(initial=minlen + 50)
        )
        self.assertEqual(len(md.mask), len(self.records))
        self.assertLess(md.nseqs, n_initial_seq)

        hash3 = md.refhash
        md.filter_min_length(minlen - 50)
        self.assertEqual(hash3, md.refhash)

        md.filter_min_length(50000000000)
        self.assertEqual(md.nseqs, 0)
        self.assertFalse(np.any(md.mask))

    def test_minlength(self):
        with self.assertRaises(ValueError):
            Composition.from_file(self.io, None, minlength=3)

    def test_properties(self):
        composition = Composition.from_file(self.io, None, minlength=420)
        passed = list(filter(lambda x: len(x.sequence) >= 420, self.records))

        self.assertEqual(composition.nseqs, len(composition.metadata.identifiers))
        self.assertEqual(composition.nseqs, len(composition.metadata.lengths))

        self.assertTrue(composition.matrix.dtype, np.float32)
        self.assertEqual(composition.matrix.shape, (len(passed), 103))

        # Names
        self.assertEqual(
            list(composition.metadata.identifiers), [i.header for i in passed]
        )

        # Lengths
        self.assertTrue(np.issubdtype(composition.metadata.lengths.dtype, np.integer))
        self.assertEqual(
            [len(i.sequence) for i in passed], list(composition.metadata.lengths)
        )

    def test_save_load(self):
        buf = io.BytesIO()
        composition_1 = Composition.from_file(self.io, None)
        md1 = composition_1.metadata
        composition_1.save(buf)
        buf.seek(0)
        composition_2 = Composition.load(buf)
        md2 = composition_2.metadata

        self.assertTrue(np.all(composition_1.matrix == composition_2.matrix))
        self.assertTrue(np.all(md1.identifiers == md2.identifiers))
        self.assertTrue(np.all(md1.lengths == md2.lengths))
        self.assertTrue(np.all(md1.refhash == md2.refhash))
        self.assertTrue(np.all(md1.minlength == md2.minlength))

    def test_windows_newlines(self):
        rng = random.Random()
        buf1 = io.BytesIO()
        buf2 = io.BytesIO()
        for i in range(10):
            record = testtools.make_randseq(rng, 10, 20)
            buf1.write(b">" + record.header.encode())
            buf2.write(b">" + record.header.encode())
            buf1.write(b"\r\n")
            buf2.write(b"\n")
            buf1.write(record.sequence)
            buf2.write(record.sequence)
            buf1.write(b"\r\n")
            buf2.write(b"\n")

        buf1.seek(0)
        buf2.seek(0)
        comp1 = Composition.from_file(buf1, None)
        comp2 = Composition.from_file(buf2, None)

        self.assertEqual(comp1.metadata.refhash, comp2.metadata.refhash)
        self.assertTrue(np.all(comp1.matrix == comp2.matrix))
