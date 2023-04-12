from math import sqrt
import unittest
import io

from vamb.benchmark import Contig, Genome, Reference, Binning

REFERENCE_DICT = {
    "genomes": {
        "gA": {
            "subjA1": [
                100,
                {
                    "sA1c1": [1, 50],
                    "sA1c2": [15, 65],
                    "sA1c3": [20, 40],
                    "sA1c4": [70, 100],
                },
            ]
        },
        "gB": {
            "subjB1": [
                20,
                {
                    "sB1c1": [1, 20],
                },
            ],
            "subjB2": [
                60,
                {
                    "sB2c1": [1, 60],
                    "sB2c2": [10, 50],
                    "sB2c3": [15, 40],
                    "sB2c4": [20, 30],
                },
            ],
        },
        "gC": {
            "subjY1": [
                40,
                {
                    "sY1c1": [10, 20],
                    "sY1c2": [21, 40],
                },
            ],
            "subjY2": [
                100,
                {
                    "sY2c1": [1, 60],
                    "sY2c2": [5, 70],
                    "sY2c3": [50, 80],
                    "sY2c4": [60, 99],
                    "sY2c5": [99, 100],
                },
            ],
            "subjY3": [
                100,
                {
                    "sY3c1": [40, 50],
                    "sY3c2": [45, 55],
                },
            ],
        },
    },
    "taxmaps": [{"gA": "D", "gB": "D", "gC": "E"}, {"D": "F", "E": "F"}],
}

# C1: All of strainA
# C2: All of B2, but missing redundant contig c3
# C3: Y1
# C4: Y2 + Y3 + B1
# C5: The missing contig from C2
BINNING_STR = """C1\tsA1c1
C1\tsA1c2
C1\tsA1c3
C1\tsA1c4
C2\tsB2c1
C2\tsB2c2
C2\tsB2c4
C3\tsY1c1
C3\tsY1c2
C4\tsB1c1
C4\tsY2c1
C4\tsY2c3
C4\tsY2c4
C4\tsY2c5
C4\tsY3c1
C4\tsY3c2
C5\tsB2c3"""

# Test breadth of genomes
# Test breadth of bins
#


class TestContig(unittest.TestCase):
    def test_neg_len(self):
        with self.assertRaises(ValueError):
            Contig("x", "y", 5, 5)

        with self.assertRaises(ValueError):
            Contig("x", "y", 5, -10)

        with self.assertRaises(ValueError):
            Contig("x", "y", -1, 5)

    def test_works(self):
        c = Contig("x", "y", 9, 19)
        self.assertEqual(repr(c), 'Contig("x", "y", 9, 19)')
        c2 = Contig.subjectless("x", 10)
        self.assertEqual(c.name, c2.name)
        self.assertEqual(len(c), len(c2))

    def test_hash_eq(self):
        c1 = Contig("x", "y", 1, 2)
        c2 = Contig("y", "y", 1, 2)
        c3 = Contig("x", "other", 999, 5000)
        self.assertEqual(c1, c3)
        self.assertNotEqual(c1, c2)
        self.assertEqual({c1}, {c3})
        self.assertEqual(len({c1, c3}), 1)


class TestGenome(unittest.TestCase):
    def test_genome(self):
        g = Genome("gen")
        with self.assertRaises(ValueError):
            g.add("foo", 0)

        with self.assertRaises(ValueError):
            g.add("foo", -1)

        self.assertEqual(g.breadth, 0)
        g.add("bar", 10)
        self.assertEqual(g.breadth, 10)
        g.add("qux", 21)
        self.assertEqual(g.breadth, 31)
        self.assertEqual(repr(g), 'Genome("gen")')

        with self.assertRaises(ValueError):
            g.add("bar", 5)

        self.assertEqual(g.breadth, sum(g.sources.values()))

    def test_eq(self):
        g1 = Genome("x")
        g1.add("foo", 5)
        g1.add("bar", 10)
        g2 = Genome("x")
        g3 = Genome("different")
        self.assertEqual(g1, g2)
        self.assertNotEqual(g2, g3)
        self.assertEqual({g1}, {g2})
        self.assertEqual(len({g1, g2}), 1)


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = Reference.from_dict(REFERENCE_DICT)
        cls.binning = Binning.from_file(io.StringIO(BINNING_STR), cls.reference)

    def test_bin_basics(self):
        bins = {b.name: b for b in self.binning.bins}
        self.assertEqual(repr(bins["C1"]), 'Bin("C1")')

        self.assertEqual(bins["C1"].ncontigs, 4)
        self.assertEqual(bins["C2"].ncontigs, 3)
        self.assertEqual(bins["C3"].ncontigs, 2)
        self.assertEqual(bins["C4"].ncontigs, 7)
        self.assertEqual(bins["C5"].ncontigs, 1)

        self.assertEqual(bins["C1"].breadth, 96)
        self.assertEqual(bins["C2"].breadth, 60)
        self.assertEqual(bins["C3"].breadth, 31)
        self.assertEqual(bins["C4"].breadth, 136)
        self.assertEqual(bins["C5"].breadth, 26)

    def test_bin_confusion_matrix(self):
        bins = {b.name: b for b in self.binning.bins}
        genomes = {g.name: g for g in self.reference.genomes}
        self.assertEqual(bins["C1"].confusion_matrix(genomes["gA"]), (96, 0, 4))
        self.assertEqual(bins["C4"].confusion_matrix(genomes["gC"]), (116, 20, 124))
        self.assertEqual(
            bins["C2"].confusion_matrix(genomes["gA"]),
            (0, bins["C2"].breadth, genomes["gA"].breadth),
        )

    def test_bin_recall_precision(self):
        bins = {b.name: b for b in self.binning.bins}
        genomes = {g.name: g for g in self.reference.genomes}
        self.assertEqual(bins["C1"].recall_precision(genomes["gA"]), (0.96, 1.0))
        self.assertEqual(
            bins["C4"].recall_precision(genomes["gC"]), (116 / 240, 116 / 136)
        )
        self.assertEqual(bins["C2"].recall_precision(genomes["gA"]), (0.0, 0.0))

    def test_bin_fscore(self):
        bins = {b.name: b for b in self.binning.bins}
        genomes = {g.name: g for g in self.reference.genomes}
        pairs = [("C1", "gA"), ("C2", "gA"), ("C4", "gC")]

        for binname, gname in pairs:
            self.assertEqual(
                bins[binname].f1(genomes[gname]),
                bins[binname].fscore(1.0, genomes[gname]),
            )

        # Alternatively, these could be considered to be undefined
        self.assertEqual(bins["C1"].fscore(2.0, genomes["gB"]), 0.0)
        self.assertEqual(bins["C5"].fscore(0.5, genomes["gA"]), 0.0)

        self.assertAlmostEqual(
            bins["C4"].fscore(sqrt(2), genomes["gC"]), 0.5649350649350648
        )
        self.assertAlmostEqual(
            bins["C1"].fscore(0.4, genomes["gA"]), 0.9942857142857143
        )

    def test_ref_errors(self):
        # Can't add same contig twice
        genome = self.reference.genomeof[self.reference.contig_by_name["sA1c4"]]
        with self.assertRaises(ValueError):
            self.reference._add_contig(Contig("sA1c4", "subjA1", 10, 20), genome)

        # Can't add genome of same name
        with self.assertRaises(ValueError):
            self.reference._add_genome(Genome("gB"))

        # Can't add contig with unknown source
        with self.assertRaises(ValueError):
            self.reference._add_contig(
                Contig("newcontig", "newsubject", 10, 20), genome
            )

        # Can't add contig with unknown genome
        with self.assertRaises(ValueError):
            g = Genome("newgenome")
            g.add("subjA1", 99)
            self.reference._add_contig(Contig("newcontig", "subjA1", 10, 20), g)

        # Can't add contig longer than its subject
        with self.assertRaises(IndexError):
            ref = Reference()
            g = Genome("newgenome")
            g.add("subj", 5)
            ref._add_genome(g)
            ref._add_contig(Contig("x", "subj", 1, 6), g)

        with self.assertRaises(ValueError):
            g = Genome("newgenome")
            g.add("foo", 99)
            self.reference._add_contig(
                Contig("newcontig", "foo", 10, 20), Genome("newgenome")
            )

        # Can't add taxonomy to wrong clade
        with self.assertRaises(ValueError):
            self.reference._add_taxonomy(1, "D", "H")

        # Can't add taxonomy of a genome that does not exist
        with self.assertRaises(KeyError):
            self.reference._add_taxonomy(1, "X", "H")

    def test_ref_basics(self):
        self.assertEqual(self.reference.ngenomes, 3)
        self.assertEqual(self.reference.ncontigs, 18)
        self.assertEqual(self.reference.nranks, 3)
        self.assertEqual(
            repr(self.reference), "<Reference with 3 genomes, 18 contigs and 3 ranks>"
        )

    def test_ref_roundtrip(self):
        ref = self.reference
        buffer = io.StringIO()
        ref.save(buffer)
        buffer.seek(0)
        ref2 = Reference.from_file(buffer)

        self.assertEqual(ref.ncontigs, ref2.ncontigs)
        self.assertEqual(ref.ngenomes, ref2.ngenomes)
        self.assertEqual(ref.nranks, ref2.nranks)
        self.assertEqual(ref.genomes, ref2.genomes)
        self.assertEqual(ref.contig_by_name, ref2.contig_by_name)
        self.assertEqual(ref.taxmaps, ref2.taxmaps)

    def test_binning_basics(self):
        self.assertEqual(self.binning.nbins, 5)
        self.assertRegex(
            repr(self.binning), "<Binning with 5 bins and reference 0x[a-f0-9]+>"
        )

    def test_binning_badargs(self):
        ref = self.reference
        with self.assertRaises(ValueError):
            Binning.from_file(io.StringIO(BINNING_STR), ref, recalls=[])

        with self.assertRaises(ValueError):
            Binning.from_file(io.StringIO(BINNING_STR), ref, recalls=[-0.01, 0.5])

        with self.assertRaises(ValueError):
            Binning.from_file(io.StringIO(BINNING_STR), ref, recalls=[0.5, 0.5])

    def test_binning_disjoint(self):
        s = BINNING_STR + "\nC99\tsY2c1"
        buffer = io.StringIO(s)

        with self.assertRaises(ValueError):
            Binning.from_file(buffer, self.reference)

        buffer.seek(0)
        # Does not raise an error
        self.assertIsInstance(
            Binning.from_file(buffer, self.reference, disjoint=False), Binning
        )

    def test_binning_strain_counter(self):
        # This approach is simple and easy to verify to be correct, but very inefficient.
        # Strain-level counter (level = 0)
        for (min_recall, min_precision), n_obs in self.binning.counters[0].items():
            n_exp = 0
            for genome in self.reference.genomes:
                for bin in self.binning.bins:
                    recall, precision = bin.recall_precision(genome)
                    if recall >= min_recall and precision >= min_precision:
                        n_exp += 1
                        break

            self.assertEqual(n_exp, n_obs)

    def test_binning_clade_counter(self):
        counters: list[
            dict[tuple[float, float], int]
        ] = self.binning.counters  # type: ignore
        genomesof = [{"D": ["gA", "gB"], "E": ["gC"]}, {"F": ["gA", "gB", "gC"]}]

        genomes = {g.name: g for g in self.reference.genomes}
        for rank in (1, 2):
            for (min_recall, min_precision), n_obs in counters[rank].items():
                seen = {c: False for c in genomesof[rank - 1]}
                for clade, genomenames in genomesof[rank - 1].items():
                    for bin in self.binning.bins:
                        rec = 0.0
                        prec = 0.0
                        for genomename in genomenames:
                            recall, precision = bin.recall_precision(
                                genomes[genomename]
                            )
                            rec = max(rec, recall)
                            prec += precision
                        if rec >= min_recall and prec >= min_precision:
                            seen[clade] = True

                self.assertEqual(sum(seen.values()), n_obs)

    def test_filtering(self):
        def test_binnames(bin: Binning, names):
            self.assertEqual({b.name for b in bin.bins}, set(names))

        bin2 = Binning.from_file(
            io.StringIO(BINNING_STR), self.reference, mincontigs=0, minsize=0
        )
        test_binnames(bin2, ["C1", "C2", "C3", "C4", "C5"])

        bin2 = Binning.from_file(io.StringIO(BINNING_STR), self.reference, mincontigs=3)
        test_binnames(bin2, ["C1", "C2", "C4"])

        bin2 = Binning.from_file(io.StringIO(BINNING_STR), self.reference, minsize=50)
        test_binnames(bin2, ["C1", "C2", "C4"])

        bin2 = Binning.from_file(io.StringIO(BINNING_STR), self.reference, minsize=100)
        test_binnames(bin2, ["C4"])

        bin2 = Binning.from_file(
            io.StringIO(BINNING_STR), self.reference, minsize=50, mincontigs=4
        )
        test_binnames(bin2, ["C1", "C4"])

    def test_printmatrix(self):
        with self.assertRaises(IndexError):
            self.binning.print_matrix(3)

        buffer = io.StringIO()
        self.binning.print_matrix(0, buffer)
