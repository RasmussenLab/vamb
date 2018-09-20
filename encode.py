__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader = vae.trainmodel(depths, tnf) # Train VAE on Numpy arrays
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 40)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import sys as _sys
import os as _os
import numpy as _np
import torch as _torch

from math import log as _log

from torch import nn as _nn
from torch import optim as _optim
from torch.nn import functional as _F
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import vamb.vambtools as _vambtools

if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')

def make_dataloader(rpkm, tnf, batchsize=128, cuda=False):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with "zero" contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x 136)
        batchsize: Size of minibatches for dataloader
        cuda: Pagelock memory of dataloader (use when using CUDA)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, _np.ndarray) or not isinstance(tnf, _np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf):
        raise ValueError('Lengths of RPKM and TNF must be the same')

    if tnf.shape[1] != 136:
        raise ValueError('TNF must be 136 long along axis 1')

    depthssum = rpkm.sum(axis=1)
    tnfsum = tnf.sum(axis=1)
    mask = depthssum != 0
    mask &= tnfsum != 0

    # Remove zero-observations Unfortunately, a copy is necessary.
    # It doesn't matter much, as cluster.py will create a copy anyway when.
    # removing observations during clustering. This cannot easily prevented.
    rpkm_copy = rpkm[mask]
    tnf_copy = tnf[mask]
    depthssum = depthssum[mask]

    # Normalize arrays and create the Tensors
    rpkm_copy /= depthssum.reshape((-1, 1))
    _vambtools.zscore(tnf_copy, axis=0, inplace=True)
    depthstensor = _torch.from_numpy(rpkm_copy)
    tnftensor = _torch.from_numpy(tnf_copy)

    # Create dataloader
    dataset = _TensorDataset(depthstensor, tnftensor)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize,
                             shuffle=True, num_workers=1, pin_memory=cuda)

    return dataloader, mask

class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [[325, 325]]
        nlatent: Number of neurons in the latent layer [40]
        mseratio: (α) Approximate TNF/(CE+TNF) ratio in loss. [0.05]
        capacity: (β) Inverse of KL-divergence weight term in loss [200]
        cuda: Use CUDA (GPU accelerated training) [False]

    Useful methods:
    VAE.trainmodel(rpkm, tnf, nepochs, batchsize, lrate, decay, logfile, modelfile)
        Trains the model, returning None

    VAE.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    """

    def __init__(self, nsamples, nhiddens=[325, 325], nlatent=40, mseratio=0.05,
                 capacity=200, cuda=False):
        if any(i < 1 for i in nhiddens):
            raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))

        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))

        if capacity <= 0:
            raise ValueError('capacity must be > 0')

        if not (0 < mseratio < 1):
            raise ValueError('mseratio must be 0 < mseratio < 1')

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.mseratio = mseratio
        self.capacity = capacity
        self.nhiddens = nhiddens
        self.nlatent = nlatent

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers (do nothing if only 1 hidden layer)
        for nin, nout in zip([self.nsamples + 136] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = _nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nsamples + 136)

        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.relu(encoderlayer(tensor)))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu, logsigma):
        epsilon = _torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        latent = mu + epsilon * _torch.exp(logsigma/2)

        return latent

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.relu(decoderlayer(tensor)))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = _F.softmax(reconstruction.narrow(1, 0, self.nsamples), dim=1)
        tnf_out = reconstruction.narrow(1, self.nsamples, 136)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        ce = - _torch.sum((depths_out.log() * depths_in))
        mse = _torch.sum((tnf_out - tnf_in).pow(2))
        kld = -0.5 * _torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())

        # mseratio and capacity is α and β in our paper, respectively
        ce_weight = (1 - self.mseratio) / _log(self.nsamples)
        mse_weight = self.mseratio / 136
        kld_weight = 1 / (self.nsamples * self.capacity)
        loss = ce * ce_weight + mse * mse_weight + kld * kld_weight

        return loss, ce, mse, kld

    def trainepoch(self, data_loader, epoch, optimizer, decay, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_mseloss = 0
        epoch_celoss = 0
        learning_rate = optimizer.param_groups[0]['lr']

        for depths_in, tnf_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

            loss, ce, mse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_mseloss += mse.data.item()
            epoch_celoss += ce.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.7f}\tCE: {:.7f}\tMSE: {:.7f}\tKLD: {:.7f}\tLR: {:.4e}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_mseloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  learning_rate,
                  ), file=logfile)

            logfile.flush()

        optimizer.param_groups[0]['lr'] = learning_rate * decay

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array = data_loader.dataset.tensors
        length = len(depths_array)

        latent = _torch.zeros((length, self.nlatent), dtype=_torch.float32)
        assert not latent.is_cuda # Should be on CPU, not GPU

        row = 0
        with _torch.no_grad():
            for depths, tnf in new_data_loader:
                depths.requires_grad = True
                tnf.requires_grad = True

                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                out_depths, out_tnf, mu, logsigma = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        latent_array = latent.detach().numpy()

        return latent_array

    def save(self, filehandle):
        state = {'nsamples': self.nsamples,
                 'mseratio': self.mseratio,
                 'capacity': self.capacity,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  encode.trainvae.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = _torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary['nsamples']
        mseratio = dictionary['mseratio']
        capacity = dictionary['capacity']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        vae = cls(nsamples, nhiddens, nlatent, mseratio, capacity, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(self, dataloader, nepochs=200, batchsize=128, lrate=1e-3,
                 decay=0.99, logfile=None, modelfile=None):

        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [200]
            batchsize: Mini-batch size for training [128]
            lrate: Starting learning rate for the optimizer [0.001]
            decay: Learning rate multiplier per epoch [0.99]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Outputs: The DataLoader object used to train the VAE
        """

        if lrate < 0:
            raise ValueError('Learning rate cannot be negative')

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if not (0 < decay <= 1):
            raise ValueError('Decay must be in interval ]0:1]')

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _optim.Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tCapacity:', self.capacity, file=logfile)
            print('\tMSE ratio:', self.mseratio, file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tBatch size:', batchsize, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tDecay:', decay, file=logfile)
            print('\tN contigs:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            self.trainepoch(dataloader, epoch, optimizer, decay, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
