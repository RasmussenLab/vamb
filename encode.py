
__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae, dataloader = trainvae(depths, tnf) # Make & train VAE on Numpy arrays
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 40)

Loss is KL-divergence of latent layer + MSE of tnf reconstruction + BCE of
depths reconstruction.
"""



__cmd_doc__ = """VAE"""



import sys
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    import argparse

import torch
from torch import nn, optim 
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset



def _dataloader_from_arrays(depthsarray, tnfarray, cuda, batchsize):
    depthstensor = torch.Tensor(depthsarray)
    tnftensor = torch.Tensor(tnfarray)
    
    dataset = TensorDataset(depthstensor, tnftensor)
    loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False,
                        num_workers=1, pin_memory=cuda)
    
    return loader



class VAE(nn.Module):
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
    
    def __init__(self, nsamples, ntnf, hiddens, latent, cuda):
        super(VAE, self).__init__()
      
        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = ntnf
        self.nfeatures = nsamples + ntnf
        self.nhiddens = hiddens
        self.nlatent = latent
      
        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()
        
        # Add all other hidden layers (do nothing if only 1 hidden layer)
        for nin, nout in zip([self.nfeatures] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))
      
        # Latent layers
        self.mu = nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = nn.Linear(self.nhiddens[-1], self.nlatent)
            
        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))
      
        # Reconstruction (output) layer
        self.outputlayer = nn.Linear(self.nhiddens[0], self.nfeatures)
      
        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
   
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
        epsilon = torch.randn(mu.size(0), mu.size(1))
        
        if self.usecuda:
            epsilon = epsilon.cuda()
        
        epsilon = Variable(epsilon)
      
        latent = mu + epsilon * torch.exp(logsigma/2)
      
        return latent
    
    def _decode(self, tensor):
        tensors = list()
        
        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = self.relu(decodernorm(decoderlayer(tensor)))
            tensors.append(tensor)
            
        reconstruction = self.outputlayer(tensor)
        
        # Decompose reconstruction to depths and tnf signal
        depths_out = F.softmax(reconstruction.narrow(1, 0, self.nsamples), dim=1)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        
        return depths_out, tnf_out
    
    def forward(self, depths, tnf):
        tensor = torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)
        
        return depths_out, tnf_out, mu, logsigma
    
    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma, tnfweight):
        criterion = nn.MSELoss() 

        mse = tnfweight * criterion(tnf_out, tnf_in)
        bce = F.binary_cross_entropy(depths_out, depths_in, size_average=False) 
        kld = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())

        loss = bce + kld + mse
        return loss, bce, mse, kld
    
    def trainmodel(self, data_loader, epoch, optimizer, tnfweight, verbose):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_bceloss = 0
        epoch_mseloss = 0

        for depths_in, tnf_in in data_loader:
            depths = Variable(depths_in)
            tnf = Variable(tnf_in)

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

            loss, bce, mse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma, tnfweight)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_bceloss += bce.data.item()
            epoch_mseloss += mse.data.item()

        if verbose:
            print('Epoch: {}\tLoss: {:.4f}\tBCE: {:.4f}\tMSE: {:.5f}\tKLD: {:.4f}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader.dataset),
                  epoch_bceloss / len(data_loader.dataset),
                  epoch_kldloss / len(data_loader.dataset),
                  epoch_mseloss / len(data_loader.dataset)
                  ))
            
    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE
        
        Input: data_loader: As generated by train_vae
        
        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """
        
        self.eval()

        mu_arrays = list()

        for batch, (depths, tnf) in enumerate(data_loader):
            depths = Variable(depths)
            tnf = Variable(tnf)

            # Move input to GPU if requested
            if self.usecuda:
                depths = depths.cuda()
                tnf = tnf.cuda()

            # Evaluate
            out_depths, out_tnf, mu, logsigma = self(depths, tnf)

            # Move latent representation back to CPU for output
            if self.usecuda:
                mu = mu.cpu()

            mu_array = mu.data.numpy()
            mu_arrays.append(mu_array)

        if len(mu_arrays) == 0:
            return np.array([], dtype=np.float32)

        length = sum(len(i) for i in mu_arrays)
        width = mu_arrays[0].shape[1]
        
        result = np.empty((length, width), dtype=np.float32)

        i = 0
        for array in mu_arrays:
            result[i:i+len(array)] = array
            i += len(array)

        return result



def trainvae(depths, tnf, nhiddens=[325, 325], latent=40, nepochs=300,
            batchsize=100, cuda=False, tnfweight=1, lrate=1e-4, verbose=False):
    
    """Create an latent encoding iterator from depths array and tnf array.
    First trains the VAE and then returns an iterator yielding the encoding
    in chunks of `batchsize`.
    
    Inputs:
        depths: An (n_contigs x n_samples) z-normalized Numpy matrix of depths
        tnf: An (n_contigs x 136) z-normalized Numpy matrix of tnf
        nhiddens: List of n_neurons in the hidden layers of VAE [325, 325]
        latent: Number of n_neurons in the latent layer [40]
        nepochs: Train for this many epochs before encoding [300]
        batchsize: Mini-batch size for training [100]
        cuda: Use CUDA (GPU acceleration) [False]
        tnfweight: Relative weight of TNF error when computing loss [1.0]
        lrate: Learning rate for the optimizer [1e-4]
        verbose: Print loss and other measures to stdout each epoch [False]
        
    Outputs:
        model: The trained VAE
        data_loader A DataLoader instance to feed the VAE for evaluation
        """
    
    # Check all the args here
    if any(i < 1 for i in nhiddens):
        raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))
        
    if latent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))
        
    if nepochs < 1:
        raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))
        
    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))
        
    if tnfweight < 0:
        raise ValueError('TNF weight cannot be negative')
        
    if lrate < 0:
        raise ValueError('Learning rate cannot be negative')
    
    data_loader = _dataloader_from_arrays(depths, tnf, cuda, batchsize)
    
    # Get number of features
    nsamples, ntnfs = [tensor.shape[1] for tensor in data_loader.dataset.tensors]
        
    # Instantiate the VAE
    model = VAE(nsamples, ntnfs, nhiddens, latent, cuda)
    
    if cuda:
        model.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=lrate)
   
    # Train
    for epoch in range(nepochs):
        model.trainmodel(data_loader, epoch, optimizer, tnfweight, verbose)

    return model, data_loader



if __name__ == '__main__':
    usage = "python encode.py DEPTHSFILE TNFFILE [OPTIONS ...]"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage)
    
    # Positional arguments
    parser.add_argument('depthsfile', help='path to depths/RPKM file')
    parser.add_argument('tnffile', help='path to TNF file')
    parser.add_argument('outfile', help='path to put latent representation')
    
    # Optional arguments
    parser.add_argument('-n', dest='nhiddens', type=int, nargs='+',
                        default=[325, 325], help='hidden neurons [325 325]')
    parser.add_argument('-l', dest='latent', type=int,
                        default=40, help='latent neurons [40]')
    parser.add_argument('-e', dest='nepochs', type=int,
                        default=300, help='epochs [300]')
    parser.add_argument('-b', dest='batchsize', type=int,
                        default=100, help='batch size [100]') 
    parser.add_argument('-t', dest='tnfweight', type=float,
                        default=1.0, help='TNF weight [1.0]')
    parser.add_argument('-r', dest='lrate', type=float,
                        default=0.0001, help='learning rate [1e-4]')
    parser.add_argument('--cuda', help='Use GPU [False]', action='store_true')
    
    # If no arguments, print help
    if len(_sys.argv) == 1:
        parser.print_help()
        sys.exit()
        
    args = parser.parse_args()
    
    if any(i < 1 for i in args.nhiddens):
        raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))
        
    if args.latent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(args.latent))
        
    if args.nepochs < 1:
        raise ValueError('Minimum 1 epoch, not {}'.format(args.nepochs))
        
    if args.batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(args.batchsize))
        
    if args.tnfweight < 0:
        raise ValueError('TNF weight cannot be negative')
        
    if args.lrate < 0:
        raise ValueError('Learning rate cannot be negative')
        
    if not os.path.isfile(args.depthsfile):
        raise FileNotFoundError(args.depthsfile)
        
    if not os.path.isfile(args.tnffile):
        raise FileNotFoundError(args.tnffile)
        
    if os.path.exists(args.outfile):
        raise FileExistsError(args.outfile)
    
    directory = os.path.dirname(args.outfile)
    if directory and not os.path.isdir(directory):
        raise NotADirectoryError(directory)

