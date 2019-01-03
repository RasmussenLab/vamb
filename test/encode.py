import sys
import os
import numpy as np
import torch

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb

# Test making the dataloader
tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf)

assert np.all(mask == np.array([False, False, False, False,  True,  True, False,  True, False,
        True,  True, False,  True,  True, False,  True,  True, False,
       False, False, False, False,  True,  True,  True, False, False,
       False,  True, False,  True, False, False,  True,  True,  True,
       False,  True,  True,  True, False, False,  True, False,  True,
        True, False, False,  True,  True,  True,  True,  True, False,
       False,  True, False, False, False,  True,  True,  True, False,
       False, False, False, False, False,  True, False, False,  True,
        True,  True,  True,  True,  True,  True,  True, False,  True,
       False,  True,  True,  True, False, False, False, False, False,
       False,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True, False, False,  True,
        True,  True, False,  True,  True,  True, False,  True, False,
       False, False, False, False, False, False, False, False, False,
        True,  True, False,  True,  True, False, False,  True, False,
        True, False,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False]))

assert len(dataloader) == sum(mask) // dataloader.batch_size

# Dataloader fails with too large batchsize
try:
    dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, batchsize=128)
except ValueError as e:
    assert e.args == ('Fewer sequences left after filtering than the batch size.',)
else:
    raise AssertionError('Should have raised ArgumentError when instantiating dataloader')

try:
    dataloader, mask = vamb.encode.make_dataloader(rpkm.flatten(), tnf, batchsize=128)
except ValueError as e:
    assert e.args == ('Lengths of RPKM and TNF must be the same',)
else:
    raise AssertionError('Should have raised ArgumentError when instantiating dataloader')

# Normalization
assert not np.all(np.abs(np.mean(tnf, axis=0)) < 1e-4) # not normalized
assert not np.all(np.abs(np.sum(rpkm, axis=1) - 1) < 1e-5) # not normalized

assert np.all(np.abs(np.mean(dataloader.dataset.tensors[1].numpy(), axis=0)) < 1e-4) # normalized
assert np.all(np.abs(np.sum(dataloader.dataset.tensors[0].numpy(), axis=1) - 1) < 1e-5) # normalized

dataloader2, mask2 = vamb.encode.make_dataloader(rpkm, tnf, destroy=True)

assert np.all(mask == mask2)
assert np.all(np.mean(tnf, axis=0) < 1e-4) # normalized
assert np.all(np.abs(np.sum(rpkm, axis=1) - 1) < 1e-5) # normalized

# Can instantiate the VAE
vae = vamb.encode.VAE(nsamples=3)

# Training model works in general
tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, batchsize=16)
vae.trainmodel(dataloader, batchsteps=[5, 10], nepochs=15)
vae.trainmodel(dataloader, batchsteps=None, nepochs=15)

# Training model fails with weird batch steps
try:
    vae.trainmodel(dataloader, batchsteps=[5, 10, 15, 20], nepochs=25)
except ValueError as e:
    assert e.args == ('Last batch size exceeds dataset length',)
else:
    raise AssertionError('Should have raised ArgumentError when having too high batch size')

try:
    vae.trainmodel(dataloader, batchsteps=[5, 10], nepochs=10)
except ValueError as e:
    assert e.args == ('Max batchsteps must not equal or exceed nepochs',)
else:
    raise AssertionError('Should have raised ArgumentError when having too high batchsteps')

# Loading saved VAE and encoding
modelpath = os.path.join(parentdir, 'test', 'data', 'model.pt')
vae = vamb.encode.VAE.load(modelpath)

target_latent = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_latent.npz'))

latent = vae.encode(dataloader)

assert np.all(np.abs(latent - target_latent) < 1e-4)

# Encoding also withs with a minibatch of one
inputs = dataloader.dataset.tensors
inputs = inputs[0][:65], inputs[1][:65]
ds = torch.utils.data.dataset.TensorDataset(inputs[0], inputs[1])
new_dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=64, shuffle=False, num_workers=1,)

latent = vae.encode(new_dataloader)
assert np.all(np.abs(latent - target_latent[:65]) < 1e-4)
