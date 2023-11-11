import unittest
import numpy as np
import torch
import tempfile
import vamb


class TestDataLoader(unittest.TestCase):
    tnfs = np.random.random((111, 103)).astype(np.float32)
    rpkm = np.random.random((111, 14)).astype(np.float32)
    lens = np.random.randint(2000, 5000, size=111)

    def nearly_same(self, A, B):
        self.assertTrue(np.all(np.abs(A - B) < 1e-5))

    def not_nearly_same(self, A, B):
        self.assertTrue(np.any(np.abs(A - B) > 1e-4))

    def test_bad_args(self):
        # Bad rpkm
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader([[1, 2, 3]], self.tnfs, self.lens, batchsize=32)

        # bad tnfs
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(self.rpkm, [[1, 2, 3]], self.lens, batchsize=32)

        # Bad batchsize
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=0)

        # Differing lengths
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(
                np.random.random((len(self.rpkm) - 1)).astype(np.float32),
                self.tnfs,
                self.lens,
                batchsize=32,
            )

        # Bad dtype
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(
                self.rpkm.astype(np.float64), self.tnfs, self.lens, batchsize=32
            )

    def test_destroy(self):
        copy_rpkm = self.rpkm.copy()
        copy_tnfs = self.tnfs.copy()

        _ = vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=32)
        self.nearly_same(self.rpkm, copy_rpkm)
        self.nearly_same(self.tnfs, copy_tnfs)

        _ = vamb.encode.make_dataloader(
            copy_rpkm, copy_tnfs, self.lens, batchsize=32, destroy=True
        )
        self.not_nearly_same(self.rpkm, copy_rpkm)
        self.not_nearly_same(self.tnfs, copy_tnfs)

    def test_normalized(self):
        copy_rpkm = self.rpkm.copy()
        copy_tnfs = self.tnfs.copy()

        _ = vamb.encode.make_dataloader(
            copy_rpkm, copy_tnfs, self.lens, batchsize=32, destroy=True
        )

        # TNFS: Mean of zero, std of one
        self.nearly_same(np.mean(copy_tnfs, axis=0), np.zeros(copy_tnfs.shape[1]))
        self.nearly_same(np.std(copy_tnfs, axis=0), np.ones(copy_tnfs.shape[1]))

        # RPKM: Sum to 1, all zero or above
        # print(copy_rpkm)
        self.nearly_same(np.sum(copy_rpkm, axis=1), np.ones(copy_rpkm.shape[0]))
        self.assertTrue(np.all(copy_rpkm >= 0.0))

    def test_single_sample(self):
        single_rpkm = self.rpkm[:, [0]]
        copy_single = single_rpkm.copy()
        dl = vamb.encode.make_dataloader(
            single_rpkm, self.tnfs.copy(), self.lens, batchsize=32, destroy=True
        )
        # When destroying a single sample, RPKM is set to 1.0
        self.assertAlmostEqual(np.abs(np.mean(single_rpkm)), 1.0)
        self.assertLess(abs(np.std(single_rpkm)), 1e-6)

        # ... and the abundance are the same abundances as before,
        # except normalized and scaled. We test that they are ordered
        # in the same order
        self.assertTrue(
            (
                torch.argsort(dl.dataset.tensors[2], dim=0)
                == torch.argsort(torch.from_numpy(copy_single), dim=0)
            )
            .all()
            .item()
        )

    def test_iter(self):
        bs = 32
        dl = vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=bs)

        # Check right element type
        for M in next(iter(dl)):
            self.assertEqual(M.dtype, torch.float32)
            self.assertEqual(M.shape[0], bs)

        # Check it iterates the right order (rpkm, tnfs)
        rpkm, tnfs, abundance, weights = next(iter(dl))
        self.nearly_same(np.sum(rpkm.numpy(), axis=1), np.ones(bs))

    def test_randomized(self):
        dl = vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=64)
        rpkm, tnfs, abundances, weights = next(iter(dl))

        # Test that first batch is not just the first 64 elements.
        # Could happen, but vanishingly unlikely.
        self.assertTrue(np.any(np.abs(tnfs.numpy() - self.tnfs[:64]) > 1e-3))


class TestVAE(unittest.TestCase):
    tnfs = np.random.random((111, 103)).astype(np.float32)
    rpkm = np.random.random((111, 14)).astype(np.float32)
    lens = np.random.randint(2000, 5000, size=111)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            vamb.encode.VAE(-1)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, nlatent=0)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, nhiddens=[128, 0])

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, alpha=0.0)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, alpha=1.0)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, beta=0.0)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, dropout=1.0)

        with self.assertRaises(ValueError):
            vamb.encode.VAE(5, dropout=-0.001)

    def test_loss_falls(self):
        vae = vamb.encode.VAE(self.rpkm.shape[1])
        rpkm_copy = self.rpkm.copy()
        tnfs_copy = self.tnfs.copy()
        dl = vamb.encode.make_dataloader(
            rpkm_copy, tnfs_copy, self.lens, batchsize=16, destroy=True
        )
        (di, ti, ai, we) = next(iter(dl))
        do, to, ao, mu = vae(di, ti, ai)
        start_loss = vae.calc_loss(di, do, ti, to, ao, ai, mu, we)[0].data.item()

        with tempfile.TemporaryFile() as file:
            # Loss drops with training
            vae.trainmodel(dl, nepochs=3, batchsteps=[1, 2], modelfile=file)
            do, to, ao, mu = vae(di, ti, ai)
            end_loss = vae.calc_loss(di, do, ti, to, ao, ai, mu, we)[0].data.item()
            self.assertLess(end_loss, start_loss)

            # Also test save/load
            before_encoding = vae.encode(dl)
            file.flush()
            file.seek(0)
            vae_2 = vamb.encode.VAE.load(file)

        after_encoding = vae_2.encode(dl)
        self.assertTrue(np.all(np.abs(before_encoding - after_encoding) < 1e-6))

    def test_encoding(self):
        nlatent = 15
        vae = vamb.encode.VAE(self.rpkm.shape[1], nlatent=nlatent)
        dl = vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=32)
        encoding = vae.encode(dl)
        self.assertEqual(encoding.dtype, np.float32)
        self.assertEqual(encoding.shape, (len(self.rpkm), nlatent))
