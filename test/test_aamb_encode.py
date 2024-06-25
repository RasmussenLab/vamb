import unittest
import numpy as np
import random
import vamb


class TestAAE(unittest.TestCase):
    tnfs = np.random.random((111, 103)).astype(np.float32)
    rpkm = np.random.random((111, 14)).astype(np.float32)
    lens = np.random.randint(2000, 5000, size=111)
    contignames = ["".join(random.choices("abcdefghijklmnopqrstu", k=10)) for _ in lens]
    nlatent_l = 32
    default_args = (14, 256, nlatent_l, 25, 0.5, 0.5, 0.15, False, 0)
    default_temperature = 0.16

    # Construction
    def test_bad_args(self):
        default_args = self.default_args

        # Test the default args work
        aae = vamb.aamb_encode.AAE(*default_args)
        self.assertIsInstance(aae, vamb.aamb_encode.AAE)

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(0, *default_args[1:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:1], 0, *default_args[2:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:2], 0, *default_args[3:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:3], 0, *default_args[4:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:5], float("nan"), *default_args[6:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:5], -0.0001, *default_args[6:])

        with self.assertRaises(ValueError):
            vamb.aamb_encode.AAE(*default_args[:6], float("nan"), *default_args[7:])

    def test_loss_falls(self):
        aae = vamb.aamb_encode.AAE(*self.default_args)
        rpkm_copy = self.rpkm.copy()
        tnfs_copy = self.tnfs.copy()
        dl = vamb.encode.make_dataloader(
            rpkm_copy, tnfs_copy, self.lens, batchsize=16, destroy=True
        )
        (di, ti, ai, we) = next(iter(dl))
        mu, do, to, _, _, _, _ = aae(di, ti)
        start_loss = aae.calc_loss(di, do, ti, to)[0].data.item()

        # Loss drops with training
        aae.trainmodel(
            dl,
            nepochs=3,
            batchsteps=[1, 2],
            T=self.default_temperature,
            modelfile=None,
        )
        mu, do, to, _, _, _, _ = aae(di, ti)
        end_loss = aae.calc_loss(di, do, ti, to)[0].data.item()
        self.assertLess(end_loss, start_loss)

    def test_encode(self):
        aae = vamb.aamb_encode.AAE(*self.default_args)
        dl = vamb.encode.make_dataloader(
            self.rpkm.copy(), self.tnfs.copy(), self.lens, batchsize=16, destroy=True
        )
        (_, encoding) = aae.get_latents(self.contignames, dl)
        self.assertIsInstance(encoding, np.ndarray)
        self.assertEqual(encoding.dtype, np.float32)
        self.assertEqual(encoding.shape, (len(self.rpkm), self.nlatent_l))
