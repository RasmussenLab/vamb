
__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae, dataloader = trainvae(depths, tnf) # Make & train VAE on Numpy arrays
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
from torch.autograd import Variable as _Variable
from torch.nn import functional as _F
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import vamb.vambtools as _vambtools



if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')



def _dataloader_from_arrays(depthsarray, tnfarray, cuda, batchsize):
    # Normalized depths
    depthssum = depthsarray.sum(axis=1).reshape((-1, 1))
    depthssum[depthssum == 0] = 1
    depthstensor = _torch.Tensor(depthsarray / depthssum)
    
    # Normalized TNF
    tnftensor = _torch.Tensor(_vambtools.zscore(tnfarray, axis=0))
    
    dataset = _TensorDataset(depthstensor, tnftensor)
    loader = _DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True,
                        num_workers=1, pin_memory=cuda)
    
    return loader



class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.
    
    init with:
        nsamples: Length of dimension 1 of depths
        ntnf: Length of dimension 1 of tnf
        hiddens: List with number of hidden neurons per layer
        latent: Number of neurons in hidden layer
        cuda: Boolean, use CUDA or not (GPU accelerated training)
        
    Useful methods:
    encode(self, data_loader):
        encodes the data in the data loader and returns the encoded matrix
    """
    
    def __init__(self, nsamples, ntnf, hiddens, latent, cuda, cefactor, msefactor, dropout):
        super(VAE, self).__init__()
      
        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = ntnf
        self.nfeatures = nsamples + ntnf
        self.nhiddens = hiddens
        self.nlatent = latent
        self.cefactor = cefactor
        self.msefactor = msefactor
        self.dropout = dropout
      
        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()
        
        # Add all other hidden layers (do nothing if only 1 hidden layer)
        for nin, nout in zip([self.nfeatures] + self.nhiddens, self.nhiddens):
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
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nfeatures)
      
        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)
   
    def _encode(self, tensor):
        tensors = list()
        
        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = self.dropoutlayer(encodernorm(self.relu(encoderlayer(tensor))))
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
        
        epsilon = _Variable(epsilon)
      
        latent = mu + epsilon * _torch.exp(logsigma/2)
      
        return latent
    
    def _decode(self, tensor):
        tensors = list()
        
        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = self.dropoutlayer(decodernorm(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)
            
        reconstruction = self.outputlayer(tensor)
        
        # Decompose reconstruction to depths and tnf signal
        depths_out = _F.softmax(reconstruction.narrow(1, 0, self.nsamples), dim=1)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        
        return depths_out, tnf_out
    
    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)
        
        return depths_out, tnf_out, mu, logsigma
    
    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        ce = - _torch.mean((depths_out.log() * depths_in))
        mse = _torch.mean((tnf_out - tnf_in).pow(2))
        kld = -0.5 * _torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        loss = self.cefactor * ce + self.msefactor * mse + kld
        
        return loss, ce, mse, kld
    
    def trainmodel(self, data_loader, epoch, optimizer, verbose, outfile=_sys.stdout):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_mseloss = 0
        epoch_celoss = 0

        for depths_in, tnf_in in data_loader:
            depths = _Variable(depths_in)
            tnf = _Variable(tnf_in)

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

        if verbose:
            print('\tEpoch: {}\tLoss: {:.2f}\tCE: {:.5f}\tMSE: {:.5f}\tKLD: {:.5f}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_mseloss / len(data_loader),
                  epoch_kldloss / len(data_loader)
                  ), file=outfile)
            
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
        
        depths, tnf = data_loader.dataset.tensors
        length = len(depths)

        latent = _torch.zeros((length, self.nlatent), dtype=_torch.float32)

        row = 0
        for batch, (depths, tnf) in enumerate(new_data_loader):
            depths = _Variable(depths)
            tnf = _Variable(tnf)

            # Move input to GPU if requested
            if self.usecuda:
                depths = depths.cuda()
                tnf = tnf.cuda()

            # Evaluate
            out_depths, out_tnf, mu, logsigma = self(depths, tnf)
            latent[row: row + len(mu)] = mu

            row += len(mu)

        assert row == length

        # Move latent representation back to CPU for output
        if self.usecuda:
            latent = latent.cpu()

        latent_array = latent.detach().numpy()

        return latent_array
    
    def save(self, filehandle):
        state = {'cefactor': self.cefactor,
                 'msefactor': self.msefactor,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'nsamples': self.nsamples,
                 'state': self.state_dict(),
                }
        
        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.
        
        Inputs:
            path: Path to model file as created by functions VAE.save,
                  encode.trainvae or torch.save(vae.state_dict(), file).
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]
               
        Output: VAE with weights and parameters matching the saved network.
        """
            
        dictionary = _torch.load(path)
        
        cefactor = dictionary['cefactor']
        msefactor = dictionary['msefactor']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        nsamples = dictionary['nsamples']
        state = dictionary['state']

        vae = cls(nsamples, 136, nhiddens, nlatent, cuda, cefactor, msefactor, dropout)
        vae.load_state_dict(state)
        
        if cuda:
            vae.cuda()
            
        if evaluate:
            vae.eval()
        
        return vae



def trainvae(depths, tnf, nhiddens=[325, 325], nlatent=100, nepochs=400,
             batchsize=256, cuda=False, errorsum=3000, mseratio=0.25, dropout=0.05,
             lrate=1e-4, verbose=False, logfile=_sys.stdout, modelfile=None):
    
    """Create an latent encoding iterator from depths array and tnf array.
    First trains the VAE and then returns an iterator yielding the encoding
    in chunks of `batchsize`.
    
    Inputs:
        depths: An (n_contigs x n_samples) z-normalized Numpy matrix of depths
        tnf: An (n_contigs x 136) z-normalized Numpy matrix of tnf
        nhiddens: List of n_neurons in the hidden layers of VAE [325, 325]
        nlatent: Number of n_neurons in the latent layer [100]
        nepochs: Train for this many epochs before encoding [400]
        batchsize: Mini-batch size for training [256]
        cuda: Use CUDA (GPU acceleration) [False]
        errorsum: How much latent layer can deviate from prior [3000]
        mseratio: Balances error from TNF versus depths in loss [0.25]
        dropout: Dropout of hidden layers [0.05]
        lrate: Learning rate for the optimizer [1e-4]
        verbose: Print loss and other measures to stdout each epoch [False]
        logfile: Print loss to this file if verbose is True
        modelfile: Save the models weights in this file if not None
        
    Outputs:
        model: The trained VAE
        data_loader A DataLoader instance to feed the VAE for evaluation
        """
    
    # Check all the args here
    if any(i < 1 for i in nhiddens):
        raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))
        
    if nlatent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))
        
    if nepochs < 1:
        raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))
        
    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))
        
    if lrate < 0:
        raise ValueError('Learning rate cannot be negative')
        
    if errorsum <= 0:
        raise ValueError('errorsum must be > 0')
        
    if not (0 < dropout < 1):
        raise ValueError('dropout must be 0 < dropout < 1')
        
    if not (0 < mseratio < 1):
        raise ValueError('mseratio must be 0 < mseratio < 1')
    
    data_loader = _dataloader_from_arrays(depths, tnf, cuda, batchsize)
    
    # Get number of features
    nsamples, ntnfs = [tensor.shape[1] for tensor in data_loader.dataset.tensors]
    
    # Get CE and MSE factor in loss calculation:
    # A completely uninformed network will guess TNF randomly from N(0, 1)
    # (because of how it's normalized), and depths to be 1/n_samples.
    # This will give an expected TNF error of 2 and an expected depth of
    # log(n_samples) / n_samples.
    expected_ce = _log(nsamples) / nsamples
    expected_mse = 2
    cefactor = errorsum * (1 - mseratio) / expected_ce
    msefactor = errorsum * mseratio / expected_mse
        
    # Instantiate the VAE
    model = VAE(nsamples, ntnfs, nhiddens, nlatent, cuda, cefactor, msefactor, dropout)
    
    if cuda:
        model.cuda()
        
    optimizer = _optim.Adam(model.parameters(), lr=lrate)
    
    if verbose:
        print('\tErrorsum:', errorsum, file=logfile)
        print('\tMSE ratio:', mseratio, file=logfile)
        print('\tCUDA:', cuda, file=logfile)
        print('\tN latent:', nlatent, file=logfile)
        print('\tN hidden:', ', '.join(map(str, nhiddens)), file=logfile)
        print('\tN contigs:', depths.shape[0], file=logfile)
        print('\tN samples:', depths.shape[1], file=logfile)
        print('\tN epochs:', nepochs, file=logfile)
        print('\tBatch size:', batchsize, file=logfile)
        print('\tDropout:', dropout, end='\n\n', file=logfile)
   
    # Train
    for epoch in range(nepochs):
        model.trainmodel(data_loader, epoch, optimizer, verbose, logfile)

    # Save weights - Lord forgive me, for I have sinned when catching all exceptions
    if modelfile is not None:
        try:
            model.save(modelfile)
        except:
            pass

        
    return model, data_loader

