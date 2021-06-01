import sys
import os
import unittest
import tempfile
import gzip
import bz2
import lzma
import numpy as np

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

DATADIR = os.path.join(PARENTDIR, "test", "data")

import vamb.vambtools

class TestReader(unittest.TestCase):
    def setUp(self):
        # Do not create files if they exist
        if hasattr(self, "files"):
            return None

        self.files = []
        with open(os.path.join(DATADIR, "fasta.fna"), "rb") as file:
            self.lines = list(file)

        for f in [gzip.compress, bz2.compress, lzma.compress]:
            file = tempfile.NamedTemporaryFile()
            file.write(f(b"".join(self.lines)))
            file.flush()
            self.files.append(file)

        return None

    def test_reader(self):
        for file in self.files:
            with vamb.vambtools.Reader(file.name, "rb") as reader:
                self.assertEqual(self.lines, list(reader))

            with vamb.vambtools.Reader(file.name, "rt") as reader:
                self.assertEqual([line.decode() for line in self.lines], list(reader))

class TestPushArray(unittest.TestCase):
    def test_append(self):
        arr = vamb.vambtools.PushArray(np.float32, start_capacity=3)
        values = [1.0, 1, -3, 55.3]
        for i in values:
            arr.append(i)
        
        self.assertTrue(np.all(arr.take() == np.array(values, dtype=np.float32)))

    def test_extend(self):
        arr = vamb.vambtools.PushArray(np.float32, start_capacity=1)
        values = [1.0, 1, -3, 55.3, 0, 53, -1.2]
        arr.extend(values)
        self.assertTrue(np.all(arr.take() == np.array(values, dtype=np.float32)))

        arr = vamb.vambtools.PushArray(np.float32, start_capacity=1)
        for i in range(1, 10000, 100):
            arr.extend(range(i, i+100))
        self.assertTrue(np.all(arr.take() == np.array(range(1, 10001), dtype=np.float32)))

    def test_clear(self):
        arr = vamb.vambtools.PushArray(np.float32, start_capacity=1)
        values = [1.0, 1, -3, 55.3, 0, 53, -1.2]
        arr.extend(values)
        self.assertTrue(np.all(arr.take() == np.array(values, dtype=np.float32)))
        arr.clear()
        self.assertTrue(len(arr.take()) == 0)

class TestFASTAEntry(unittest.TestCase):
    def test_init(self):
        # Begins with '>'
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry(">foo", bytearray(b"TAG"))

        # Begins with '#'
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry("#foo", bytearray(b"AAA"))

        # With whitespace
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry(" foo", bytearray(b"SWK"))

        # Contains tab
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry("contig\t", bytearray(b"NNN"))

        # Various
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry("foo", bytearray(b"ATGCpN"))

    def test_masking(self):
        self.assertEqual(
            vamb.vambtools.FastaEntry("foo", bytearray(b"TaGkmYnAC")).sequence,
            bytearray(b"TAGNNNNAC")
        )
        self.assertEqual(vamb.vambtools.FastaEntry("foo", bytearray()).sequence, bytearray())

    def test_various(self):
        # Length
        self.assertEqual(len(vamb.vambtools.FastaEntry("x", bytearray(b"TAGCA"))), 5)
        self.assertEqual(len(vamb.vambtools.FastaEntry("x", bytearray())), 0)
        self.assertEqual(len(vamb.vambtools.FastaEntry("x", bytearray(b"TGTAmnyAncC"))), 11)

        # Str
        self.assertEqual(str(vamb.vambtools.FastaEntry("x", bytearray(b"TAGCA"))), ">x\nTAGCA")
        self.assertEqual(str(vamb.vambtools.FastaEntry("yz", bytearray())), ">yz\n")
        self.assertEqual(str(vamb.vambtools.FastaEntry("1_2", bytearray(b"TGTAmnyAncC"))), ">1_2\nTGTANNNANCC")

    def test_kmercounts(self):
        seq = vamb.vambtools.FastaEntry("", bytearray(b"TTAyCAAnGAC"))

        with self.assertRaises(ValueError):
            seq.kmercounts(0)

        with self.assertRaises(ValueError):
            seq.kmercounts(13)

        self.assertTrue(np.all(seq.kmercounts(1) == np.array([4, 2, 1, 2])))

        self.assertTrue(np.all(seq.kmercounts(2) == np.array([
            1, 1, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 1,
        ])))

class TestFASTAReader(unittest.TestCase):
    def test_bad_files(self):
        # First non-comment line must be header
        with self.assertRaises(ValueError):
            data = b"#foo\n#bar\n  \n>foo\nTAG".splitlines()
            list(vamb.vambtools.byte_iterfasta(data))

        # String input
        with self.assertRaises(TypeError):
            data = ">abc\nTAG\na\nAC".splitlines()
            list(vamb.vambtools.byte_iterfasta(data))

    # Various correct formats
    def test_good_files(self):
        # Empty file
        data = b"".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data))
        self.assertEqual(0, len(records))

        # Only comments
        data = b"#bar\n#foo\n#".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data))
        self.assertEqual(0, len(records))

        # A few sequences
        data = b"#bar\n#foo\n>ab\nTA\nT\n \t\nA\nNN\n>foo\nCyAmmkg\n>bar\n".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data))
        self.assertEqual(3, len(records))
        self.assertEqual(records[0].sequence, bytearray(b"TATANN"))
        self.assertEqual(records[1].sequence, bytearray(b"CNANNNG"))
        self.assertEqual(records[2].sequence, bytearray(b""))

class TestInplaceMaskArray(unittest.TestCase):
    pass

class TestZscore(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
