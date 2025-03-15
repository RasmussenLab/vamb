import io
import os
import unittest
import tempfile
import gzip
import bz2
import lzma
import random
import itertools
import numpy as np
import string
import torch
import pathlib
import shutil

import vamb
from vamb.vambtools import BinSplitter
import testtools

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PARENTDIR, "test", "data")


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
            arr.extend(range(i, i + 100))
        self.assertTrue(
            np.all(arr.take() == np.array(range(1, 10001), dtype=np.float32))
        )

    def test_clear(self):
        arr = vamb.vambtools.PushArray(np.float32, start_capacity=1)
        values = [1.0, 1, -3, 55.3, 0, 53, -1.2]
        arr.extend(values)
        self.assertTrue(np.all(arr.take() == np.array(values, dtype=np.float32)))
        arr.clear()
        self.assertTrue(len(arr.take()) == 0)


class TestFASTAEntry(unittest.TestCase):
    def test_init(self):
        passing = [b"0", b"foo", b"FOO123_@@^", b"%%;;/=~~"]
        failing = [
            b"",  # empty
            b">",  # empty
            b">foo",  # with >
            b"=",  # cannot start with =
            b"*",  # cannot start with *
            b"\xff",  # out of range
            b"))",
            b" ",  # zero-length identifier
        ]

        for i in passing:
            x = vamb.vambtools.FastaEntry(i, bytearray(b"TAG"))
            self.assertTrue(i.decode(), x.header)

        for i in failing:
            with self.assertRaises(ValueError):
                vamb.vambtools.FastaEntry(i, bytearray(b"TAG"))

        # Various
        with self.assertRaises(ValueError):
            vamb.vambtools.FastaEntry(b"foo", bytearray(b"ATGCpN"))

    def test_masking(self):
        self.assertEqual(
            vamb.vambtools.FastaEntry(b"foo", bytearray(b"TaGkmYnAC")).sequence,
            bytearray(b"TaGkmYnAC"),
        )
        self.assertEqual(
            vamb.vambtools.FastaEntry(b"foo", bytearray()).sequence, bytearray()
        )

    def test_various(self):
        # Length
        self.assertEqual(len(vamb.vambtools.FastaEntry(b"x", bytearray(b"TAGCA"))), 5)
        self.assertEqual(len(vamb.vambtools.FastaEntry(b"x", bytearray())), 0)
        self.assertEqual(
            len(vamb.vambtools.FastaEntry(b"x", bytearray(b"TGTAmnyAncC"))), 11
        )

        # format
        self.assertEqual(
            vamb.vambtools.FastaEntry(b"x", bytearray(b"TAGCA")).format(), ">x\nTAGCA"
        )
        self.assertEqual(
            vamb.vambtools.FastaEntry(b"yz", bytearray()).format(), ">yz\n"
        )
        self.assertEqual(
            vamb.vambtools.FastaEntry(b"1_2", bytearray(b"TGTAmnyAncC")).format(),
            ">1_2\nTGTAmnyAncC",
        )

    def test_random_kmercounts(self):
        indexof = {
            "".join(ncs): idx
            for (idx, ncs) in enumerate(itertools.product("ACGT", repeat=4))
        }
        seq = testtools.make_randseq(random.Random(), 900, 1100)
        sequence = seq.sequence.decode()
        manual_counts = np.zeros(256, dtype=int)
        for i in range(len(sequence) - 3):
            ind = indexof.get(sequence[i : i + 4].upper())
            if ind is not None:
                manual_counts[ind] += 1

        automatic = seq.kmercounts()
        self.assertTrue(np.all(manual_counts == automatic))

    def test_rename(self):
        seq = vamb.vambtools.FastaEntry(b"foo", bytearray(b"TaGkmYnAC"))

        # Does not error
        seq.rename(b"identifier\t desc")
        self.assertEqual(seq.identifier, "identifier")
        self.assertEqual(seq.description, "\t desc")

        seq.rename(b"newname")
        self.assertEqual(seq.identifier, "newname")
        self.assertEqual(seq.description, "")

        # Errors
        with self.assertRaises(ValueError):
            seq.rename(b"\tabc def")

        with self.assertRaises(ValueError):
            seq.rename(b"=123")

        with self.assertRaises(ValueError):
            seq.rename(b"")

        with self.assertRaises(ValueError):
            seq.rename(b"\xff")


class TestFASTAReader(unittest.TestCase):
    def test_bad_files(self):
        # First non-comment line must be header
        with self.assertRaises(ValueError):
            data = b"#foo\n#bar\n  \n>foo\nTAG".splitlines()
            list(vamb.vambtools.byte_iterfasta(data, None))

        # String input
        with self.assertRaises(TypeError):
            data = ">abc\nTAG\na\nAC".splitlines()
            list(vamb.vambtools.byte_iterfasta(data, None))  # type:ignore

    # Various correct formats
    def test_good_files(self):
        # Empty file
        data = b"".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data, None))
        self.assertEqual(0, len(records))

        # A few sequences
        data = b">ab\nTA\nT\n \t\nA\nNN\n>foo\nCyAmmkg\n>bar\n".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data, None))
        self.assertEqual(3, len(records))
        self.assertEqual(records[0].sequence, bytearray(b"TATANN"))
        self.assertEqual(records[1].sequence, bytearray(b"CyAmmkg"))
        self.assertEqual(records[2].sequence, bytearray(b""))

        # Comment inside seqs
        data = b">foo\nTAG\n>@@AA\nAAA".splitlines()
        records = list(vamb.vambtools.byte_iterfasta(data, None))
        self.assertEqual(records[0].sequence, bytearray(b"TAG"))


class TestZscore(unittest.TestCase):
    arr = np.array([[1, 2, 2.5], [2, 4, 3], [0.9, 3.1, 2.8]])
    zscored = np.array(
        [
            [-1.44059316, -0.3865006, 0.14054567],
            [-0.3865006, 1.7216845, 0.66759195],
            [-1.54600241, 0.77300121, 0.45677344],
        ]
    )

    def almost_similar(self, x, y):
        self.assertTrue(np.all(np.abs(x - y) < 1e-6))

    def test_simple(self):
        self.almost_similar(vamb.vambtools.zscore(self.arr), self.zscored)

    def test_axes(self):
        self.almost_similar(
            vamb.vambtools.zscore(self.arr, axis=0),
            np.array(
                [
                    [-0.60404045, -1.26346568, -1.29777137],
                    [1.40942772, 1.18195176, 1.13554995],
                    [-0.80538727, 0.08151391, 0.16222142],
                ]
            ),
        )

        self.almost_similar(
            vamb.vambtools.zscore(self.arr, axis=1),
            np.array(
                [
                    [-1.33630621, 0.26726124, 1.06904497],
                    [-1.22474487, 1.22474487, 0.0],
                    [-1.40299112, 0.85548239, 0.54750873],
                ]
            ),
        )

    def test_axis_bounds(self):
        with self.assertRaises(np.exceptions.AxisError):
            vamb.vambtools.zscore(self.arr, axis=-1)

        with self.assertRaises(np.exceptions.AxisError):
            vamb.vambtools.zscore(self.arr, axis=2)

    def test_integer(self):
        with self.assertRaises(TypeError):
            vamb.vambtools.zscore(np.array([1, 2, 3]), inplace=True)

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
        mask = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0]).astype(bool)
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
        b1 = vamb.vambtools.RefHasher.hash_refnames(names)

        # Test that hashing them all at once is the same as hashing them one at a time
        hasher = vamb.vambtools.RefHasher()
        hasher.add_refname(names[0])
        hasher.add_refname(names[1])
        for j in names[2:]:
            hasher.add_refname(j)
        b7 = hasher.digest()

        names[1] = names[1] + "x"
        b2 = vamb.vambtools.RefHasher.hash_refnames(names)
        names[1] = names[1][:-1] + " \t"  # it strips whitespace off right end
        b3 = vamb.vambtools.RefHasher.hash_refnames(names)
        names = names[::-1]
        b4 = vamb.vambtools.RefHasher.hash_refnames(names)

        names = (i + "   " for i in names[::-1])
        b5 = vamb.vambtools.RefHasher.hash_refnames(names)
        b6 = vamb.vambtools.RefHasher.hash_refnames(names)  # now empty generator

        self.assertNotEqual(b1, b2)
        self.assertEqual(b1, b3)
        self.assertNotEqual(b1, b4)
        self.assertEqual(b1, b5)
        self.assertNotEqual(b1, b6)
        self.assertEqual(b1, b7)


class TestBinSplit(unittest.TestCase):
    before = [("bin1", ["s1-c1", "s1-c2", "s8-c1", "s9-c11"]), ("bin2", ["s12-c0"])]

    after = [
        ("s1-bin1", {"s1-c1", "s1-c2"}),
        ("s8-bin1", {"s8-c1"}),
        ("s9-bin1", {"s9-c11"}),
        ("s12-bin2", {"s12-c0"}),
    ]

    def test_inert(self):
        self.assertEqual(
            BinSplitter("").splitter, BinSplitter.inert_splitter().splitter
        )

    def test_split(self):
        self.assertEqual(list(BinSplitter("-").binsplit(self.before)), self.after)

    def test_badsep(self):
        with self.assertRaises(KeyError):
            list(BinSplitter("2").binsplit(self.before))

    def test_badtype(self):
        with self.assertRaises(Exception):
            list(BinSplitter("x").binsplit([(1, [2])]))  # type:ignore

    def test_nosplit(self):
        self.assertEqual(
            list(BinSplitter("").binsplit(self.before)),
            [(k, set(s)) for (k, s) in self.before],
        )

    def test_initialize(self):
        # Nothing happens to an inert splitter
        b = BinSplitter.inert_splitter()
        s = b.splitter
        b.initialize([""])
        self.assertEqual(s, b.splitter)

        b = BinSplitter("X")
        with self.assertRaises(ValueError):
            b.initialize(["AXC", "S1C2"])

        b = BinSplitter(None)
        b.initialize(["S1C2", "ABC"])
        self.assertEqual(b.splitter, None)

        b = BinSplitter(None)
        b.initialize(["S1C2", "KMCPLK"])

        b = BinSplitter("XYZ")
        b.initialize(["ABXYZCD", "KLMXYZA"])


class TestConcatenateFasta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(19)
        ios = [io.BytesIO(), io.BytesIO()]
        for i in ios:
            for _ in range(4):
                fasta = testtools.make_randseq(rng, 10, 200)
                i.write(fasta.format().encode())
                i.write(b"\n")

        cls.file_1 = ios[0]
        cls.file_2 = ios[1]

    def setUp(self):
        self.file_1.seek(0)
        self.file_2.seek(0)

    def test_no_rename(self):
        out = io.StringIO()
        vamb.vambtools.concatenate_fasta_ios(
            out, [self.file_1, self.file_2], minlength=10, rename=False
        )
        out.seek(0)
        data = out.read()
        self.file_1.seek(0)
        self.file_2.seek(0)
        data_1 = self.file_1.read().decode()
        data_2 = self.file_2.read().decode()
        self.assertEqual(data, data_1 + data_2)

    def test_rename(self):
        out = io.StringIO()
        vamb.vambtools.concatenate_fasta_ios(
            out, [self.file_1, self.file_2], minlength=10, rename=True
        )
        out.seek(0)
        out_bytes = io.BytesIO(out.read().encode())
        records_out = list(vamb.vambtools.byte_iterfasta(out_bytes, None))
        self.file_1.seek(0)
        self.file_2.seek(0)
        records_1 = list(vamb.vambtools.byte_iterfasta(self.file_1, None))
        records_2 = list(vamb.vambtools.byte_iterfasta(self.file_2, None))
        records_in = records_1 + records_2
        self.assertEqual(
            [i.sequence for i in records_out], [i.sequence for i in records_in]
        )
        self.assertEqual(len(set(i.identifier for i in records_out)), len(records_out))

        # Check that sequences present in each file now starts with S1 and S2
        identifier_by_seq_hash = dict()
        for record in records_out:
            identifier_by_seq_hash[hash(bytes(record.sequence))] = record.identifier

        for prefix, records in [("S1", records_1), ("S2", records_2)]:
            for record in records:
                identifier = identifier_by_seq_hash[hash(bytes(record.sequence))]
                self.assertTrue(identifier.startswith(prefix))


class TestWriteClusters(unittest.TestCase):
    test_clusters = [("C1", set("abc")), ("C2", set("hkjlmn")), ("C3", set("x"))]
    io = io.StringIO()

    def setUp(self):
        self.io.truncate(0)
        self.io.seek(0)

    def linesof(self, string: str):
        return list(filter(lambda x: not x.startswith("#"), string.splitlines()))

    def conforms(self, str, clusters):
        lines = self.linesof(str)
        self.assertEqual(lines[0], vamb.vambtools.CLUSTERS_HEADER)
        self.assertEqual(len(lines) - 1, sum(len(v) for (_, v) in clusters))
        allcontigs = set()
        printed = set()
        printed_names = set()
        read = set()
        read_names = set()

        for k, v in clusters:
            allcontigs.update(v)

        for line in lines[1:]:
            name, _, contig = line.partition("\t")
            printed.add(contig)
            printed_names.add(name)

        for k, v in vamb.vambtools.read_clusters(io.StringIO(str)).items():
            read.update(v)
            read_names.add(k)

        self.assertEqual(allcontigs, printed)
        self.assertEqual(allcontigs, read)
        self.assertEqual(read_names, printed_names)

    def test_not_writable(self):
        buf = io.BufferedReader(io.BytesIO(b""))  # type:ignore
        with self.assertRaises(ValueError):
            vamb.vambtools.write_clusters(buf, self.test_clusters)  # type:ignore

    def test_normal(self):
        vamb.vambtools.write_clusters(self.io, self.test_clusters)
        self.conforms(self.io.getvalue(), self.test_clusters)

    def test_max_clusters(self):
        vamb.vambtools.write_clusters(self.io, self.test_clusters[:2])
        lines = self.linesof(self.io.getvalue())
        self.assertEqual(len(lines), 10)
        self.conforms(self.io.getvalue(), self.test_clusters[:2])


class TestWriteBins(unittest.TestCase):
    file = io.BytesIO()
    N_BINS = 10
    dir = pathlib.Path(
        os.path.join(
            tempfile.gettempdir(),
            "".join(random.choices(string.ascii_letters + string.digits, k=10)),
        )
    )

    @classmethod
    def setUpClass(cls):
        bins: dict[str, set[str]] = dict()
        seqs: dict[str, vamb.vambtools.FastaEntry] = dict()
        for _ in range(cls.N_BINS):
            binname = "".join(random.choices(string.ascii_letters, k=12))
            bins[binname] = set()
            for _ in range(random.randrange(3, 7)):
                seq = testtools.make_randseq(random.Random(), 100, 250)
                seqs[seq.identifier] = seq
                bins[binname].add(seq.identifier)
                cls.file.write(seq.format().encode())
                cls.file.write(b"\n")

        cls.bins = bins
        cls.seqs = seqs

    def setUp(self):
        self.file.seek(0)

    def tearDown(self):
        try:
            shutil.rmtree(self.dir)
        except FileNotFoundError:
            pass

    def test_bad_params(self):
        # Too many bins for maxbins
        with self.assertRaises(ValueError):
            vamb.vambtools.write_bins(
                self.dir, self.bins, self.file, maxbins=self.N_BINS - 1
            )

        # Parent does not exist
        with self.assertRaises(NotADirectoryError):
            vamb.vambtools.write_bins(
                pathlib.Path("svogew/foo"),
                self.bins,
                self.file,
                maxbins=self.N_BINS + 1,
            )

        # Target is an existing file
        with self.assertRaises(FileExistsError):
            with tempfile.NamedTemporaryFile() as file:
                vamb.vambtools.write_bins(
                    pathlib.Path(file.name),
                    self.bins,
                    self.file,
                    maxbins=self.N_BINS + 1,
                )

        # One contig missing from fasta dict
        with self.assertRaises(IndexError):
            bins = {k: v.copy() for k, v in self.bins.items()}
            next(iter(bins.values())).add("a_new_bin_which_does_not_exist")
            vamb.vambtools.write_bins(
                self.dir, bins, self.file, maxbins=self.N_BINS + 1
            )

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as dir:
            vamb.vambtools.write_bins(
                pathlib.Path(dir),
                self.bins,
                self.file,
                maxbins=self.N_BINS,
            )

            reconstructed_bins: dict[str, set[str]] = dict()
            for filename in os.listdir(dir):
                with open(os.path.join(dir, filename), "rb") as file:
                    entries = list(vamb.vambtools.byte_iterfasta(file, None))
                    binname = filename[:-4]
                    reconstructed_bins[binname] = set()
                    for entry in entries:
                        reconstructed_bins[binname].add(entry.identifier)

        # Same bins
        self.assertEqual(len(self.bins), len(reconstructed_bins))
        self.assertEqual(
            sum(map(len, self.bins.values())),
            sum(map(len, reconstructed_bins.values())),
        )
        self.assertEqual(self.bins, reconstructed_bins)
