import sys
import io
import os
import unittest
import tempfile
import gzip
import bz2
import lzma
import numpy as np
import torch

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

        for f in [gzip.compress, bz2.compress, lzma.compress, lambda x: x]:
            file = tempfile.NamedTemporaryFile()
            file.write(f(b"".join(self.lines)))
            file.flush()
            self.files.append(file)

        return None

    def test_reader(self):
        for file in self.files:
            with vamb.vambtools.Reader(file.name) as reader:
                self.assertEqual(self.lines, list(reader))

class TestPushArray(unittest.TestCase):
    def test_append(self):
        arr = vamb.vambtools.PushArray(np.float32, start_capacity=3)
        values = [1.0, 1, -3, 55.3]
        for i in values:
            arr.append(i)
        
        self.assertEqual(len(arr), 4)
        self.assertTrue(np.all(arr.take() == np.array(values, dtype=np.float32)))
        arr.clear()
        self.assertEqual(len(arr), 0)

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
        passing = [
            b">0",
            b">foo",
            b">FOO123_@@^",
            b">%%;;/=~~"
        ]
        failing = [
            b"", # empty
            b">", # empty
            b"foo", # no >
            b">=", # cannot start with =
            b">*", # cannot start with *
            b">\xff", # out of range
            b">))",
            b"> " # zero-length identifier
        ]

        for i in passing:
            x = vamb.vambtools.FastaEntry(i, bytearray(b"TAG"))
            self.assertTrue(i[1:].decode().startswith(x.header))

        for i in failing:
            with self.assertRaises(ValueError):
                vamb.vambtools.FastaEntry(i, bytearray(b"TAG"))

        # Various
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry(b">foo", bytearray(b"ATGCpN"))

    def test_masking(self):
        self.assertEqual(
            vamb.vambtools.FastaEntry(b">foo", bytearray(b"TaGkmYnAC")).sequence,
            bytearray(b"TAGNNNNAC")
        )
        self.assertEqual(vamb.vambtools.FastaEntry(b">foo", bytearray()).sequence, bytearray())

    def test_various(self):
        # Length
        self.assertEqual(len(vamb.vambtools.FastaEntry(b">x", bytearray(b"TAGCA"))), 5)
        self.assertEqual(len(vamb.vambtools.FastaEntry(b">x", bytearray())), 0)
        self.assertEqual(len(vamb.vambtools.FastaEntry(b">x", bytearray(b"TGTAmnyAncC"))), 11)

        # Str
        self.assertEqual(str(vamb.vambtools.FastaEntry(b">x", bytearray(b"TAGCA"))), ">x\nTAGCA")
        self.assertEqual(str(vamb.vambtools.FastaEntry(b">yz", bytearray())), ">yz\n")
        self.assertEqual(str(vamb.vambtools.FastaEntry(b">1_2", bytearray(b"TGTAmnyAncC"))), ">1_2\nTGTANNNANCC")

    def test_kmercounts(self):
        seq = vamb.vambtools.FastaEntry(b">X", bytearray(b"TTAyCAAnGAC"))

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

        # Comment inside seqs
        data = b"#bar\n>foo\nTAG\n##xxx\n>@@AA\nAAA".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data))
        self.assertEqual(records[0].sequence, bytearray(b"TAG"))

class TestZscore(unittest.TestCase):
    arr = np.array([[1, 2, 2.5], [2, 4, 3], [0.9, 3.1, 2.8]])
    zscored = np.array([
        [-1.44059316, -0.3865006 ,  0.14054567],
        [-0.3865006 ,  1.7216845 ,  0.66759195],
        [-1.54600241,  0.77300121,  0.45677344]]
    )

    def almost_similar(self, x, y):
        self.assertTrue(np.all(np.abs(x - y) < 1e-6))

    def test_simple(self):
        self.almost_similar(vamb.vambtools.zscore(self.arr), self.zscored)

    def test_axes(self):
       self.almost_similar(vamb.vambtools.zscore(self.arr, axis=0), np.array([
           [-0.60404045, -1.26346568, -1.29777137],
            [ 1.40942772,  1.18195176,  1.13554995],
            [-0.80538727,  0.08151391,  0.16222142]]
        ))

       self.almost_similar(vamb.vambtools.zscore(self.arr, axis=1), np.array([
            [-1.33630621,  0.26726124,  1.06904497],
            [-1.22474487,  1.22474487,  0.        ],
            [-1.40299112,  0.85548239,  0.54750873]]
       ))

    def test_axis_bounds(self):
        with self.assertRaises(np.AxisError):
            vamb.vambtools.zscore(self.arr, axis=-1)

        with self.assertRaises(np.AxisError):
            vamb.vambtools.zscore(self.arr, axis=2)

    def test_integer(self):
        with self.assertRaises(TypeError):
            vamb.vambtools.zscore(np.array([1,2,3]), inplace=True)

    def test_novar(self):
        arr = np.array([4, 4, 4])
        self.almost_similar(vamb.vambtools.zscore(arr), np.array([0, 0, 0]))

    def test_inplace(self):
        vamb.vambtools.zscore(self.arr, inplace=True)
        self.almost_similar(self.arr, self.zscored)

class TestInplaceMaskArray(unittest.TestCase):
    def almost_similar_np(self, x, y):
        self.assertTrue(np.all(np.abs(x - y) < 1e-6))

    def almost_similar_torch(self, x, y):
            self.assertTrue(torch.all(torch.abs(x - y) < 1e-6))

    def test_numpy(self):
        arr = np.random.random((10, 3)).astype(np.float32)
        mask = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]).astype(np.uint8)
        arr2 = arr[[bool(i) for i in mask]]
        vamb.vambtools.numpy_inplace_maskarray(arr, mask)
        self.almost_similar_np(arr, arr2)

        # arr is now too short after being masked
        with self.assertRaises(ValueError):
            vamb.vambtools.numpy_inplace_maskarray(arr, mask)


    def test_torch(self):
        arr = torch.rand(10, 3)
        mask = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        arr2 = arr[[bool(i) for i in mask]]
        vamb.vambtools.torch_inplace_maskarray(arr, mask)
        self.almost_similar_torch(arr, arr2)

        # arr is now too short after being masked
        with self.assertRaises(ValueError):
                vamb.vambtools.torch_inplace_maskarray(arr, mask)

class TestHashRefNames(unittest.TestCase):
    def test_refhash(self):
        names = ["foo", "9", "eleven", "a"]
        b1 = vamb.vambtools.hash_refnames(names)
        names[1] = names[1] + 'x'
        b2 = vamb.vambtools.hash_refnames(names)
        names[1] = names[1][:-1] + ' \t' # it strips whitespace off right end
        b3 = vamb.vambtools.hash_refnames(names)

        self.assertIsNone(vamb.vambtools.verify_refhash(names, b1))

        names = names[::-1]
        b4 = vamb.vambtools.hash_refnames(names)
        names.clear()
        b5 = vamb.vambtools.hash_refnames(names)

        hashes = [b1, b2, b3, b4, b5]
        self.assertTrue(b1 == b3)
        self.assertTrue(len(set(hashes)) == 4)
        self.assertTrue(len(set(map(len, hashes))) == 1)

        with self.assertRaises(ValueError):
            vamb.vambtools.verify_refhash(names, b1)

class TestBinSplit(unittest.TestCase):
    before = [
        ('bin1', ['s1-c1', 's1-c2', 's8-c1', 's9-c11']),
        ('bin2', ['s12-c0'])
    ]

    after = [
        ('s1-bin1', {'s1-c1', 's1-c2'}),
        ('s8-bin1', {'s8-c1'}),
        ('s9-bin1', {'s9-c11'}),
        ('s12-bin2', {'s12-c0'}),
    ]

    def test_split(self):
        self.assertEqual(list(vamb.vambtools.binsplit(self.before, '-')), self.after)

    def test_badsep(self):
        with self.assertRaises(KeyError):
            list(vamb.vambtools.binsplit(self.before, '2'))

    def test_badtype(self):
        with self.assertRaises(TypeError):
            list(vamb.vambtools.binsplit([(1, [2])], ''))

class TestWriteClusters(unittest.TestCase):
    test_clusters = [
        ('C1', set('abc')),
        ('C2', set('hkjlmn')),
        ('C3', set('x'))
    ]
    io = io.StringIO()

    def setUp(self):
        self.io.truncate(0)
        self.io.seek(0)

    def linesof(self, str):
        return list(filter(lambda x: not x.startswith('#'), str.splitlines()))

    def conforms(self, str, clusters):
        lines = self.linesof(str)
        self.assertEqual(
            len(lines),
            sum(len(v) for (k, v) in clusters)
        )
        allcontigs = set()
        printed = set()
        printed_names = set()
        read = set()
        read_names = set()

        for (k, v) in clusters:
            allcontigs.update(v)

        for line in lines:
            name, _, contig = line.partition('\t')
            printed.add(contig)
            printed_names.add(name)
        
        for (k, v) in vamb.vambtools.read_clusters(io.StringIO(str)).items():
            read.update(v)
            read_names.add(k)

        self.assertEqual(allcontigs, printed)
        self.assertEqual(allcontigs, read)
        self.assertEqual(read_names, printed_names)

    def test_not_writable(self):
        buf = io.BufferedReader(io.BytesIO(b''))
        with self.assertRaises(ValueError):
            vamb.vambtools.write_clusters(buf, self.test_clusters)

    def test_invalid_max_clusters(self):
        with self.assertRaises(ValueError):
            vamb.vambtools.write_clusters(self.io, self.test_clusters, max_clusters=0)

        with self.assertRaises(ValueError):
            vamb.vambtools.write_clusters(self.io, self.test_clusters, max_clusters=-11)

    def test_header_has_newline(self):
        with self.assertRaises(ValueError):
            vamb.vambtools.write_clusters(self.io, self.test_clusters, header="foo\n")

    def test_normal(self):
        vamb.vambtools.write_clusters(self.io, self.test_clusters, header="someheader")
        self.assertTrue(self.io.getvalue().startswith('# someheader'))
        self.conforms(self.io.getvalue(), self.test_clusters)

    def test_max_clusters(self):
        vamb.vambtools.write_clusters(self.io, self.test_clusters, max_clusters=2)
        lines = self.linesof(self.io.getvalue())
        self.assertEqual(len(lines), 9)
        self.conforms(self.io.getvalue(), self.test_clusters[:2])

    def test_min_size(self):
        vamb.vambtools.write_clusters(self.io, self.test_clusters, min_size=5)
        lines = self.linesof(self.io.getvalue())
        self.assertEqual(len(lines), 6)
        self.conforms(self.io.getvalue(), self.test_clusters[1:2])



if __name__ == "__main__":
    unittest.main()
