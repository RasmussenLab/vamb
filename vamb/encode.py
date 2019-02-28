__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = make_dataloader(depths, tnf)
>>> vae.trainmodel(dataloader)
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

def make_dataloader(rpkm, tnf, batchsize=64, destroy=False, cuda=False):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x 136)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

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

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if mask.sum() < batchsize:
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        if not (rpkm.dtype == tnf.dtype == _np.float32):
            raise ValueError('Arrays must be of data type np.float32 if destroy is True')

        rpkm = _vambtools.inplace_maskarray(rpkm, mask)
        tnf = _vambtools.inplace_maskarray(tnf, mask)
    else:
        rpkm = rpkm[mask].astype(_np.float32, copy=False)
        tnf = tnf[mask].astype(_np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _vambtools.zscore(rpkm, axis=0, inplace=True)

    # Normalize arrays and create the Tensors
    _vambtools.zscore(tnf, axis=0, inplace=True)
    depthstensor = _torch.from_numpy(rpkm)
    tnftensor = _torch.from_numpy(tnf)

    # Create dataloader
    dataset = _TensorDataset(depthstensor, tnftensor)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=1, pin_memory=cuda)

    return dataloader, mask

class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [[325, 325]]
        nlatent: Number of neurons in the latent layer [40]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = 0.05]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [None = 0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(self, nsamples, nhiddens=[325, 325], nlatent=40, alpha=None,
                 beta=200, dropout=None, cuda=False):
        if any(i < 1 for i in nhiddens):
            raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))

        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0')

        # If only 1 sample, we weigh alpha and dropout differently
        if alpha is None:
            alpha = 0.05 if nsamples > 1 else 0.99

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if beta <= 0:
            raise ValueError('beta must be > 0')

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1')

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1')

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

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
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
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
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, 136)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _F.softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            ce = - (depths_out.log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / 136
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + kld * kld_weight

        return loss, ce, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

            loss, ce, sse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf in new_data_loader:
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
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
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
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = _torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: Double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps = list()
        else:
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')


        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _optim.Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            print('\tBatchsteps:', ', '.join(map(str, batchsteps)), file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
