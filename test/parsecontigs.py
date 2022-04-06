import io
import unittest
import random
import string
import numpy as np

import vamb


class TestReadContigs(unittest.TestCase):
    records = []
    io = io.BytesIO()
    chars = string.ascii_letters + string.digits
    symbols = np.array(bytearray(b"acgtACGTnNywsdbK"))
    weights = np.array([0.12] * 8 + [0.005] * 8)

    def setUp(self) -> None:
        self.io.seek(0)
        self.io.truncate(0)
        self.records.clear()
        for i in range(random.randrange(1400, 1500)):
            name = (
                b">"
                + "".join(
                    random.choices(self.chars, k=random.randrange(1, 10))
                ).encode()
                + b" "
                + "".join(
                    random.choices(self.chars, k=random.randrange(1, 10))
                ).encode()
            )
            seq = bytearray(
                np.random.choice(
                    self.symbols, p=self.weights, size=random.randrange(400, 600)
                )
            )
            self.records.append(vamb.vambtools.FastaEntry(name, seq))

        for i in self.records:
            self.io.write(i.format().encode())
            self.io.write(b"\n")

        self.io.seek(0)

    def test_minlength(self):
        with self.assertRaises(ValueError):
            vamb.parsecontigs.read_contigs(self.io, minlength=3)

    def test_properties(self):
        tnfs, names, lengths = vamb.parsecontigs.read_contigs(self.io, minlength=410)
        passed = list(filter(lambda x: len(x.sequence) >= 410, self.records))

        self.assertEqual(len(tnfs), len(names))
        self.assertEqual(len(lengths), len(names))

        # TNFSs
        self.assertTrue(tnfs.dtype, np.float32)
        self.assertEqual(tnfs.shape, (len(passed), 103))

        # Names
        self.assertEqual(names, [i.header for i in passed])

        # Lengths
        self.assertTrue(np.issubdtype(lengths.dtype, np.integer))
        self.assertEqual([len(i.sequence) for i in passed], list(lengths))
