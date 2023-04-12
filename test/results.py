import unittest
import io
import random
import numpy as np
import torch
from hashlib import sha256

import vamb
import testtools


class TestCompositionResult(unittest.TestCase):
    io = io.BytesIO()

    @classmethod
    def setUpClass(cls):
        rng = random.Random(15)
        for i in range(4):
            seq = testtools.make_randseq(rng, 400, 600)
            cls.io.write(seq.format().encode())
            cls.io.write(b"\n")

    def setUp(self):
        self.io.seek(0)

    def test_result(self):
        comp = vamb.parsecontigs.Composition.from_file(self.io)
        self.assertEqual(
            sha256(comp.matrix.data.tobytes()).digest().hex(),
            "9e9a2d7b021654e874894722bdd6cd3eda18bed03fabd32a9440e806a8ab1bd1",
        )


class TestAbundanceResult(unittest.TestCase):
    def test_result(self):
        comp_metadata = vamb.parsecontigs.CompositionMetaData(
            np.array(testtools.BAM_NAMES, dtype=object),
            np.array(testtools.BAM_SEQ_LENS),
            np.ones(len(testtools.BAM_SEQ_LENS), dtype=bool),
            2000,
        )

        abundance = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, "/tmp/tmpbam", comp_metadata, True, 0.9, 2
        )
        self.assertEqual(
            sha256(abundance.matrix.data.tobytes()).digest().hex(),
            "c346abb53b62423fe95ed4b2eb5988d77141b2d7a5c58c03fdf09abc6476df78",
        )
        abundance2 = vamb.parsebam.Abundance.from_files(
            testtools.BAM_FILES, None, comp_metadata, True, 0.9, 4
        )
        self.assertTrue(np.all(np.abs(abundance.matrix - abundance2.matrix) < 1e-5))


class TestEncodingResult(unittest.TestCase):
    torch.manual_seed(0)

    def test_result(self):
        rng = np.random.RandomState(15)
        tnfs = rng.random((200, 103)).astype(np.float32)
        rpkm = rng.random((200, 6)).astype(np.float32)
        lens = rng.randint(2000, 5000, 200)

        self.assertEqual(
            sha256(lens.data.tobytes()).digest().hex(),
            "68894f01cc435a5f032a655faecddd817cd35a71397129296a11f8c40bd29fcb",
        )

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        np.random.seed(0)
        random.seed(0)
        vae = vamb.encode.VAE(6)
        dl, mask = vamb.encode.make_dataloader(rpkm, tnfs, lens, batchsize=16)
        vae.trainmodel(dl, nepochs=3, batchsteps=[1, 2])
        latent = vae.encode(dl)

        self.assertEqual(
            sha256(latent.data.tobytes()).digest().hex(),
            "0148ec0767e88c756615340d6fd0b31ca07aa6b4b172a1874fb7de7179acb57d",
        )

        self.assertEqual(
            sha256(torch.rand(10).numpy().tobytes()).digest().hex(),
            "c417b9722e14e854fbe79cc5c797cc6653360c1e6536064205ca0c073f41eaf6",
        )


class TestClusterResult(unittest.TestCase):
    def test_result(self):
        rng = np.random.RandomState(15)
        latent = rng.random((1000, 3)).astype(np.float32) - 0.5
        self.assertEqual(
            sha256(latent.tobytes()).digest().hex(),
            "630a98a4b44c3754a3f423e915847f44767bb69fb13ea5901dc512428aee9811",
        )

        hash = sha256()

        # Use this to check that the clustering used in this test produces
        # a reasonable cluster size, and that it doesn't just pass because
        # it always clusters everything in 1-point clusters.
        # Uncomment when updating this test.
        # lens = list()
        for cluster in vamb.cluster.ClusterGenerator(latent):
            medoid, points = cluster.as_tuple()
            # Set hashing may differ from run to run, so turn into sorted arrays
            arr = np.array(list(points))
            arr.sort()
            # lens.append(arr)
            hash.update(medoid.to_bytes(4, "big"))
            hash.update(arr.data)

        # self.assertGreater(len(list(map(lambda x: len(lens) > 1))), 3)
        self.assertEqual(
            hash.digest().hex(),
            "2b3caf674ff1d1906a831219e0953b2d9f1b78ecefec709b70c672280af49aee",
        )
