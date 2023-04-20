import unittest
import numpy as np
import torch
import tempfile
import io

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
            vamb.encode.make_dataloader(
                [[1, 2, 3]], self.tnfs, np.ndarray([2000]), batchsize=32
            )

        # bad tnfs
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(
                self.rpkm, [[1, 2, 3]], np.ndarray([2000]), batchsize=32
            )

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

        # Batchsize is too large
        with self.assertRaises(ValueError):
            vamb.encode.make_dataloader(self.rpkm, self.tnfs, self.lens, batchsize=256)

    def test_destroy(self):
        copy_rpkm = self.rpkm.copy()
        copy_tnfs = self.tnfs.copy()

        (dl, mask) = vamb.encode.make_dataloader(
            self.rpkm, self.tnfs, self.lens, batchsize=32
        )
        self.nearly_same(self.rpkm, copy_rpkm)
        self.nearly_same(self.tnfs, copy_tnfs)

        (dl, mask) = vamb.encode.make_dataloader(
            copy_rpkm, copy_tnfs, self.lens, batchsize=32, destroy=True
        )
        self.not_nearly_same(self.rpkm, copy_rpkm)
        self.not_nearly_same(self.tnfs, copy_tnfs)

    def test_normalized(self):
        copy_rpkm = self.rpkm.copy()
        copy_tnfs = self.tnfs.copy()

        (dl, mask) = vamb.encode.make_dataloader(
            copy_rpkm, copy_tnfs, self.lens, batchsize=32, destroy=True
        )

        # TNFS: Mean of zero, std of one
        self.nearly_same(np.mean(copy_tnfs, axis=0), np.zeros(copy_tnfs.shape[1]))
        self.nearly_same(np.std(copy_tnfs, axis=0), np.ones(copy_tnfs.shape[1]))

        # RPKM: Sum to 1, all zero or above
        # print(copy_rpkm)
        self.nearly_same(np.sum(copy_rpkm, axis=1), np.ones(copy_rpkm.shape[0]))
        self.assertTrue(np.all(copy_rpkm >= 0.0))

    def test_mask(self):
        copy_rpkm = self.rpkm.copy()
        copy_tnfs = self.tnfs.copy()
        mask = np.ones(len(copy_rpkm)).astype(bool)

        for bad_tnf in [0, 4, 9]:
            copy_tnfs[bad_tnf, :] = 0
            mask[bad_tnf] = False

        for bad_rpkm in [1, 4, 11, 19]:
            copy_rpkm[bad_rpkm, :] = 0
            mask[bad_rpkm] = False

        (dl, mask2) = vamb.encode.make_dataloader(
            copy_rpkm, copy_tnfs, self.lens, batchsize=32
        )

        self.assertTrue(np.all(mask == mask2))

    def test_single_sample(self):
        single_rpkm = self.rpkm[:, [0]]
        (dl, mask) = vamb.encode.make_dataloader(
            single_rpkm, self.tnfs.copy(), self.lens, batchsize=32, destroy=True
        )
        self.assertLess(np.abs(np.mean(single_rpkm)), 1e-6)
        self.assertLess(abs(np.std(single_rpkm) - 1), 1e-6)

    def test_iter(self):
        bs = 32
        (dl, mask) = vamb.encode.make_dataloader(
            self.rpkm, self.tnfs, self.lens, batchsize=bs
        )

        # Check right element type
        for M in next(iter(dl)):
            self.assertEqual(M.dtype, torch.float32)
            self.assertEqual(M.shape[0], bs)

        # Check it iterates the right order (rpkm, tnfs)
        rpkm, tnfs, weights = next(iter(dl))
        self.nearly_same(np.sum(rpkm.numpy(), axis=1), np.ones(bs))

    def test_randomized(self):
        (dl, mask) = vamb.encode.make_dataloader(
            self.rpkm, self.tnfs, self.lens, batchsize=64
        )
        rpkm, tnfs, weights = next(iter(dl))

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
        dl, mask = vamb.encode.make_dataloader(
            rpkm_copy, tnfs_copy, self.lens, batchsize=16, destroy=True
        )
        di = torch.Tensor(rpkm_copy)
        ti = torch.Tensor(tnfs_copy)
        we = dl.dataset.tensors[2]
        do, to, mu, lsigma = vae(di, ti)
        start_loss = vae.calc_loss(di, do, ti, to, mu, lsigma, we)[0].data.item()
        iobuffer = io.StringIO()

        with tempfile.TemporaryFile() as file:
            # Loss drops with training
            vae.trainmodel(
                dl, nepochs=3, batchsteps=[1, 2], logfile=iobuffer, modelfile=file
            )
            do, to, mu, lsigma = vae(di, ti)
            end_loss = vae.calc_loss(di, do, ti, to, mu, lsigma, we)[0].data.item()
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
        dl, mask = vamb.encode.make_dataloader(
            self.rpkm, self.tnfs, self.lens, batchsize=32
        )
        encoding = vae.encode(dl)
        self.assertEqual(encoding.dtype, np.float32)
        self.assertEqual(encoding.shape, (len(self.rpkm), nlatent))
