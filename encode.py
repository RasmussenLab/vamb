
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



__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""



import sys as _sys
import os as _os
import numpy as _np

import torch as _torch
from torch import nn as _nn
from torch import optim as _optim 
from torch.autograd import Variable as _Variable
from torch.nn import functional as _F
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset

if __package__ is None or __package__ == '':
    import vambtools as _vambtools
    
else:
    import vamb.vambtools as _vambtools
    
if __name__ == '__main__':
    import argparse



if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')



def _dataloader_from_arrays(depthsarray, tnfarray, cuda, batchsize):
    depthstensor = _torch.Tensor(depthsarray)
    tnftensor = _torch.Tensor(tnfarray)
    
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
        
        epsilon = _Variable(epsilon)
      
        latent = mu + epsilon * _torch.exp(logsigma/2)
      
        return latent
    
    def _decode(self, tensor):
        tensors = list()
        
        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = self.relu(decodernorm(decoderlayer(tensor)))
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
        bce = _F.binary_cross_entropy(depths_out, depths_in, size_average=True)
        mse = _torch.mean((tnf_out - tnf_in).pow(2))
        kld = -0.5 * _torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        loss = 1000 * ce + kld + mse
        
        return loss, bce, ce, mse, kld
    
    def trainmodel(self, data_loader, epoch, optimizer, verbose, outfile=_sys.stdout):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_bceloss = 0
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

            loss, bce, ce, mse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_bceloss += bce.data.item()
            epoch_mseloss += mse.data.item()
            epoch_celoss += ce.data.item()

        if verbose:
            print('Epoch: {}\tLoss: {:.4f}\tBCE: {:.5f}\tCE: {:.7f}\tMSE: {:.5f}\tKLD: {:.5f}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_bceloss / len(data_loader),
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
         _torch.save(self.state_dict(), filehandle)

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

        ntnf = 136
        nsamples = dictionary['outputlayer.bias'].shape[0] - ntnf

        nhiddens = list()
        for k, v in dictionary.items():
            if k.startswith('encoderlayers') and k.endswith('.bias'):
                nhiddens.append(v.shape[0])

        nlatent = dictionary['mu.bias'].shape[0]

        vae = cls(nsamples, ntnf, nhiddens, nlatent, cuda)
        
        if cuda:
            vae.cuda()
            
        if evaluate:
            vae.eval()
        
        vae.load_state_dict(dictionary)

        return vae



def trainvae(depths, tnf, nhiddens=[325, 325], nlatent=40, nepochs=200,
            batchsize=100, cuda=False, lrate=1e-4, verbose=False,
            logfile=_sys.stdout, modelfile=None):
    
    """Create an latent encoding iterator from depths array and tnf array.
    First trains the VAE and then returns an iterator yielding the encoding
    in chunks of `batchsize`.
    
    Inputs:
        depths: An (n_contigs x n_samples) z-normalized Numpy matrix of depths
        tnf: An (n_contigs x 136) z-normalized Numpy matrix of tnf
        nhiddens: List of n_neurons in the hidden layers of VAE [325, 325]
        nlatent: Number of n_neurons in the latent layer [40]
        nepochs: Train for this many epochs before encoding [200]
        batchsize: Mini-batch size for training [100]
        cuda: Use CUDA (GPU acceleration) [False]
        lrate: Learning rate for the optimizer [1e-4]
        verbose: Print loss and other measures to stdout each epoch [False]
        logfile: Print loss to this file is verbose is True
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
    
    data_loader = _dataloader_from_arrays(depths, tnf, cuda, batchsize)
    
    # Get number of features
    nsamples, ntnfs = [tensor.shape[1] for tensor in data_loader.dataset.tensors]
        
    # Instantiate the VAE
    model = VAE(nsamples, ntnfs, nhiddens, nlatent, cuda)
    
    if cuda:
        model.cuda()
        
    optimizer = _optim.Adam(model.parameters(), lr=lrate)
   
    # Train
    for epoch in range(nepochs):
        model.trainmodel(data_loader, epoch, optimizer, verbose, logfile)

    # Save weights - Lord forgive me, for I have sinned when catching all exceptions
    if modelfile is not None:
        try:
            _torch.save(model.state_dict(), modelfile)
        except:
            pass

        
    return model, data_loader



if __name__ == '__main__':
    
    ################ Create parser ############################
    
    usage = "python encode.py DEPTHSFILE TNFFILE [OPTIONS ...]"
    parser = argparse.ArgumentParser(
        description=__cmd_doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=usage)
    
    # Positional arguments
    parser.add_argument('depthsfile', help='path to depths/RPKM file')
    parser.add_argument('tnffile', help='path to TNF file')
    parser.add_argument('latentfile', help='path to put latent representation')
    
    # Optional arguments
    parser.add_argument('-m', dest='modelfile', help='path to put model weights [None]')
    parser.add_argument('-n', dest='nhiddens', type=int, nargs='+',
                        default=[325, 325], help='hidden neurons [325 325]')
    parser.add_argument('-l', dest='nlatent', type=int,
                        default=40, help='latent neurons [40]')
    parser.add_argument('-e', dest='nepochs', type=int,
                        default=300, help='epochs [300]')
    parser.add_argument('-b', dest='batchsize', type=int,
                        default=100, help='batch size [100]') 
    parser.add_argument('-r', dest='lrate', type=float,
                        default=0.0001, help='learning rate [1e-4]')
    parser.add_argument('--cuda', help='Use GPU [False]', action='store_true')
    
    # If no arguments, print help
    if len(_sys.argv) == 1:
        parser.print_help()
        _sys.exit()
        
    args = parser.parse_args()
    
    ################# Check inputs ###################
    
    if args.cuda and not _torch.cuda.is_available():
        raise ModuleNotFoundError('Cuda is not available for PyTorch')
    
    if any(i < 1 for i in args.nhiddens):
        raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))
        
    if args.nlatent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(args.latent))
        
    if args.nepochs < 1:
        raise ValueError('Minimum 1 epoch, not {}'.format(args.nepochs))
        
    if args.batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(args.batchsize))
        
    if args.lrate < 0:
        raise ValueError('Learning rate cannot be negative')
    
    for inputfile in (args.depthsfile, args.tnffile):
        if not _os.path.isfile(inputfile):
            raise FileNotFoundError(inputfile)
            
    for outputfile in (args.outfile, args.modelfile):
        if _os.path.exists(outputfile):
            raise FileExistsError(outputfile)
        
        directory = _os.path.dirname(outputfile)
        if directory and not _os.path.isdir(directory):
            raise NotADirectoryError(directory)
        
    ############## Run program #######################
    
    depths = _vambtools.read_tsv(args.depthsfile)
    tnf = _vambtools.read_tsv(args.tnffile)
    
    vae, data_loader = trainvae(depths, tnf, nhiddens=args.nhiddens, latent=args.latent,
                                nepochs=args.nepochs, batchsize=args.batchsize,
                                cuda=args.cuda, lrate=args.lrate, verbose=True,
                               modelfile=args.modelfile)
    
    latent = vae.encode(data_loader)
    
    _vambtools.write_tsv(args.outfile, latent)

