from math import sqrt
import unittest
import random
import io

from vamb.benchmark import Contig, Genome, Reference, Binning, filter_clusters

REFERENCE = """contig1 ecoli   chrom1  55  90
cnt2    ecoli   chrom1  91  99
contig3 ecoli   chrom1  99  101
cnt11   ecoli   cnt11   1   40
cnt13   bsubtil chrom1  5   15
cnt15   bsubtil chrom1  27  35
cnt18   bsubtil chrom1  21  40
cnt19   bsubtil chrom2  2   11
cnt22   bsubtil chrom2  6   14
cnt10   bsubtil chrom2  5   12
cnt12   bsubtil chrom2  2   14"""

BINNING = """clus1   contig1
clus1   cnt2
clus1   cnt11
clus2   contig3
clus3   cnt13
clus3   cnt15
clus3   cnt18
clus3   cnt22
clus3   cnt10
clus4   cnt19
clus5   cnt12"""

TAXMAP = """ecoli\tbacterium
bsubtil\tbacterium"""

BREADTH_ECOLI = (101 - 55 + 1) + (40)
BREADTH_BSUBTIL = (11 + 40 - 21 + 1) + 13


def process_const(str, seed):
    lines = str.splitlines()
    random.Random(seed).shuffle(lines)
    return '\n'.join(['\t'.join(line.split()) for line in lines])


REFERENCE = process_const(REFERENCE, 0)
BINNING = process_const(BINNING, 1)


class TestContig(unittest.TestCase):
    def test_neg_len(self):
        with self.assertRaises(ValueError):
            Contig("x", "y", 5, 5)

        with self.assertRaises(ValueError):
            Contig("x", "y", 5, -10)

    def test_works(self):
        c = Contig("x", "y", 9, 19)
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
    contigs = [
        Contig.subjectless("a", 22),  # ref a: 22
        Contig("p", "k", 71, 89),
        Contig("w", "k", 32, 59),
        Contig("k", "m", 2, 15),  # ref m: 19
        Contig("q", "k", 13, 49),
        Contig("s", "m", 15, 21),
        Contig("v", "k", 11, 44),  # ref k: 66
    ]

    def test_basics(self):
        g = Genome("w")
        self.assertEqual(g.ncontigs, 0)
        self.assertEqual(g.name, "w")
        self.assertEqual(g.breadth, 0)

    def test_add(self):
        g = Genome("gen")
        for c in self.contigs:
            g.add(c)
        self.assertEqual(g.ncontigs, len(self.contigs))
        self.assertEqual(set(self.contigs), g.contigs)
        for c in self.contigs:
            g.discard(c)
        self.assertEqual(g.ncontigs, 0)
        g.add(self.contigs[0])
        self.assertEqual(g.ncontigs, 1)
        g.remove(self.contigs[0])
        self.assertEqual(g.ncontigs, 0)
        with self.assertRaises(KeyError):
            g.remove(self.contigs[0])

    def test_breadth(self):
        g = Genome("g")
        breadth = 22 + 19 + 66
        self.assertEqual(g.getbreadth(self.contigs), breadth)
        g.update_breadth()
        # Contigs not added yet
        self.assertEqual(g.breadth, 0)
        for c in self.contigs:
            g.add(c)
        g.update_breadth()
        self.assertEqual(g.breadth, breadth)
        g.discard(self.contigs[0])
        g.update_breadth()
        self.assertLess(g.breadth, breadth)

    def test_hash_eq(self):
        c1 = Contig("x", "y", 1, 2)
        c2 = Contig("y", "y", 1, 2)
        g1 = Genome("x")
        g1.add(c1)
        g1.add(c2)
        g2 = Genome("x")
        g3 = Genome("different")
        self.assertEqual(g1, g2)
        self.assertNotEqual(g2, g3)
        self.assertEqual({g1}, {g2})
        self.assertEqual(len({g1, g2}), 1)


class TestReference(unittest.TestCase):
    def test_breadth(self):
        ref = Reference.from_file(io.StringIO(REFERENCE))
        self.assertEqual(ref.breadth, BREADTH_BSUBTIL + BREADTH_ECOLI)
        self.assertEqual(ref.ngenomes, 2)
        self.assertEqual(ref.ncontigs, 11)

    def test_remove(self):
        ref = Reference.from_file(io.StringIO(REFERENCE))
        genome = [i for i in ref.genomes if i.name == "ecoli"][0]
        ref.discard(genome)
        self.assertEqual(ref.breadth, BREADTH_BSUBTIL)
        self.assertEqual(ref.ngenomes, 1)
        self.assertEqual(ref.ncontigs, 7)

        with self.assertRaises(KeyError):
            ref.remove(genome)

        ref.add(genome)
        self.assertEqual(ref.breadth, BREADTH_BSUBTIL + BREADTH_ECOLI)
        self.assertEqual(ref.ngenomes, 2)
        self.assertEqual(ref.ncontigs, 11)

    def test_unique_contignames(self):
        ref = Reference.from_file(io.StringIO(REFERENCE))
        with self.assertRaises(KeyError):
            genome = Genome("g")
            genome.add(Contig.subjectless("cnt12", 11))
            ref.add(genome)

    def test_unique_genomenames(self):
        g1, g2 = Genome("g"), Genome('g')
        with self.assertRaises(KeyError):
            Reference([g1, g2])

    def test_taxfile(self):
        with self.assertRaises(KeyError):
            badfile = "ecoli\tbact\nnoexist\tbacterium"
            ref = Reference.from_file(io.StringIO(REFERENCE))
            ref.load_tax_file(io.StringIO(badfile))

        ref = Reference.from_file(io.StringIO(REFERENCE))
        ref.load_tax_file(io.StringIO(TAXMAP))
        self.assertEqual(len(ref.taxmaps), 1)

        # Rank A mapped to both B and C
        badstr = "ecoli\ta\tb\nbsubtil\ta\tc"
        ref = Reference.from_file(io.StringIO(REFERENCE))
        with self.assertRaises(KeyError):
            ref.load_tax_file(io.StringIO(badstr))


class TestFilterContigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clusters = dict()
        cls.contigs_by_name = dict()
        for cluster, binname in map(str.split, BINNING.splitlines()):
            if cluster not in cls.clusters:
                cls.clusters[cluster] = set()
            cls.clusters[cluster].add(binname)
        for (contig, genome, subject, start, end) in map(str.split, REFERENCE.splitlines()):
            cls.contigs_by_name[contig] = Contig(
                contig, subject, int(start), int(end) + 1)

    def test_no_filter(self):
        self.assertEqual(filter_clusters(
            self.clusters, self.contigs_by_name, 0, 0), self.clusters)

    def test_minsize(self):
        size = 40
        filt = filter_clusters(self.clusters, self.contigs_by_name, size, 0)
        # We use Genome's breadth calcular to verify
        kept_binnames = set()
        for (binname, contigs) in self.clusters.items():
            g = Genome("")
            for contigname in contigs:
                g.add(self.contigs_by_name[contigname])
            g.update_breadth()
            if g.breadth >= size:
                kept_binnames.add(binname)

        self.assertEqual(set(filt), kept_binnames)

    def test_mincontigs(self):
        filt = filter_clusters(self.clusters, self.contigs_by_name, 0, 3)
        self.assertEqual(set(filt), {'clus1', 'clus3'})


class TestBininng(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = Reference.from_file(io.StringIO(REFERENCE))
        cls.bin = Binning.from_file(io.StringIO(BINNING), cls.ref)

    def test_bad_init(self):
        # Contig not found in reference
        binstr = BINNING + "\nbinx\tcontignoexist"
        with self.assertRaises(KeyError):
            Binning.from_file(io.StringIO(binstr), self.ref)

        # This errors in a different code path
        with self.assertRaises(KeyError):
            Binning.from_file(io.StringIO(binstr), self.ref, minsize=1)
        Binning.from_file(io.StringIO(binstr), self.ref, checkpresence=False)

        # Contig in multiple bins
        binstr = BINNING + "\nbinx\tcontig3"
        with self.assertRaises(KeyError):
            Binning.from_file(io.StringIO(binstr), self.ref)
        Binning.from_file(io.StringIO(binstr), self.ref, disjoint=False)

    def test_basics(self):
        self.assertEqual(self.bin.nbins, 5)
        self.assertEqual(self.bin.ncontigs, 11)

    def test_statistics(self):
        g1, g2 = sorted(self.bin.reference.genomes, key=lambda x: x.name)
        tp, tn, fp, fn = self.bin.confusion_matrix(g1, 'clus3')
        # Manually calculated
        self.assertEqual(tp, 41)
        self.assertEqual(tn, g2.breadth)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 3)

        # Alternative formulae
        mcc = (tp * tn - fp * fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        self.assertAlmostEqual(self.bin.mcc(g1, "clus3"), mcc)

        f1 = tp / (tp + 0.5 * (fp + fn))
        self.assertAlmostEqual(self.bin.f1(g1, "clus3"), f1)

        self.assertEqual([[2, 2, 2, 2, 2, 2, 2, 1, 0]], self.bin.summary())

        tp, tn, fp, fn = self.bin.confusion_matrix(g1, 'clus1')
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 2)

    def test_filtering(self):
        bin = Binning.from_file(io.StringIO(BINNING), self.ref, minsize=40)
        self.assertEqual(bin.nbins, 2)
        self.assertEqual(bin.ncontigs, 8)

        bin = Binning.from_file(io.StringIO(BINNING), self.ref, minsize=12)
        self.assertEqual(bin.nbins, 3)
        self.assertEqual(bin.ncontigs, 9)

        bin = Binning.from_file(io.StringIO(BINNING), self.ref, mincontigs=2)
        self.assertEqual(bin.nbins, 2)
        self.assertEqual(bin.ncontigs, 8)

        bin = Binning.from_file(io.StringIO(BINNING), self.ref, mincontigs=4)
        self.assertEqual(bin.nbins, 1)
        self.assertEqual(bin.ncontigs, 5)

    def test_with_taxmaps(self):
        ref = Reference.from_file(io.StringIO(REFERENCE))
        ref.load_tax_file(io.StringIO(TAXMAP))
        bin = Binning.from_file(io.StringIO(BINNING), ref)
        self.assertEqual(bin.summary(), [[2, 2, 2, 2, 2, 2, 2, 1, 0], [
                         1, 1, 1, 1, 1, 1, 1, 1, 0]])

    def test_print_matrix(self):
        buf = io.StringIO()
        self.bin.print_matrix(0, buf)
        buf.getvalue()

        with self.assertRaises(IndexError):
            self.bin.print_matrix(1)
