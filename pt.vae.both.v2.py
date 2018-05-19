#!/usr/bin/env python3

from __future__ import print_function
import argparse
import os
import gzip
import linecache
import numpy as np
import pandas as pd
import torch
from torch import nn, optim 
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision import transforms, datasets

class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_path, tnf_path):
        # stuff
        self.data = pd.read_csv(data_path, sep="\t", header=None)
        self.tnf = pd.read_csv(tnf_path, sep="\t", header=None)
         
    def __getitem__(self, index):
        d = torch.from_numpy(np.asarray(self.data.iloc[index,], dtype=np.float32))
        t = torch.from_numpy(np.asarray(self.tnf.iloc[index,], dtype=np.float32))
        return (d,t)

    def __len__(self):
        return len(self.data.index)

class CustomDatasetFromCSV_line(Dataset):
   def __init__(self, data_path, tnf_path):
      self.data_path = data_path
      self.tnf_path = tnf_path
      self.total_lines = 0
      with open(data_path, "r") as fh:
         for line in fh:
            self.total_lines = self.total_lines + 1
   
   def __getitem__(self, index):
      dline = linecache.getline(self.data_path, index+1)
      tline = linecache.getline(self.tnf_path, index+1)
      d = torch.from_numpy(np.asarray(dline.rstrip().split('\t'), dtype="float32"))
      t = torch.from_numpy(np.asarray(tline.rstrip().split('\t'), dtype="float32"))
      return (d,t)
   
   def __len__(self):
      return self.total_lines

def load_data(args, training=True):
   '''Data loader from files'''
   
   # handle multiple inputfiles
   # if files are gzipped: read in through pandas
   # if files are text: read in through linecache
   datasets = []
   for i in range(len(args.i)):
      if args.i[0].endswith('.gz'):
         dat = CustomDatasetFromCSV(args.i[i], args.tnf[i])
      else:
         dat = CustomDatasetFromCSV_line(args.i[i], args.tnf[i])
      datasets.append(dat)
   custom_data_from_csv = torch.utils.data.ConcatDataset(datasets)
   
   # DataLoader instances will load tensors directly into GPU memory
   if args.cuda:
      kwargs = {'num_workers': 4, 'pin_memory': True}
   else:
      kwargs = {'num_workers': 1, 'pin_memory': False}
   if training:
      data_loader = torch.utils.data.DataLoader(dataset=custom_data_from_csv, batch_size=args.batch, shuffle=False, **kwargs)
   else:
      data_loader = torch.utils.data.DataLoader(dataset=custom_data_from_csv, batch_size=args.batch, shuffle=False, **kwargs)
   return data_loader

def load_data_mem(args, training=True):
   '''Data loader directly to memory'''
   
   # read in data as pandas and convert to tensors
   data_tensor = torch.Tensor(np.asarray(pd.read_csv(args.i[0], sep='\t', header=None)))
   tnf_tensor = torch.Tensor(np.asarray(pd.read_csv(args.tnf[0], sep='\t', header=None)))
   dataset = TensorDataset(data_tensor, tnf_tensor)
   
   # set if loading to memory for gpu
   if args.cuda:
      kwargs = {'num_workers': 1, 'pin_memory': True}
   else:
      kwargs = {'num_workers': 1, 'pin_memory': False}
   if training:
      data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False, **kwargs)
   else:
      data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False, **kwargs)
   return data_loader

def to_var(x):
    if args.cuda:
        x = x.cuda()
    return Variable(x)

def get_features(file):
   '''Determine number of features by reading in first 10 rows'''
   
   if file.endswith(".gz"):
      fh = gzip.open(file, "rt")
   else:
      fh = open(file, "r")
   
   df = pd.read_csv(file, sep="\t", header=None, nrows=10)
   return(df.shape[1])

class VAE(nn.Module):
   def __init__(self, n_features, dp_rate):
      super(VAE, self).__init__()
      
      # init
      self.n_features = n_features
      self.total_features = n_features[0]+n_features[1]
      self.dp_rate = dp_rate 
      self.n_hidden = list(map(int, args.hidden))
      self.n_latent = int(args.latent)
      
      # initialize list for holding layers
      self.encoder_list = nn.ModuleList()
      self.encoder_bn_list = nn.ModuleList()
      self.decoder_list = nn.ModuleList()
      self.decoder_bn_list = nn.ModuleList()
      
      # encoder layers
      for i in range(len(self.n_hidden)):
         if i == 0:
            self.encoder_list.append(nn.Linear(self.total_features, self.n_hidden[i]))
         else:
            self.encoder_list.append(nn.Linear(self.n_hidden[i-1], self.n_hidden[i]))
         self.encoder_bn_list.append(nn.BatchNorm1d(self.n_hidden[i]))
      
      # latent layer
      self.zmu = nn.Linear(self.n_hidden[-1], self.n_latent)
      self.zlsig = nn.Linear(self.n_hidden[-1], self.n_latent)
            
      # decoder layers
      self.n_hidden_reverse = self.n_hidden[::-1]
      for i in range(len(self.n_hidden_reverse)):
         if i == 0:
            self.decoder_list.append(nn.Linear(self.n_latent, self.n_hidden_reverse[i]))
         else:
            self.decoder_list.append(nn.Linear(self.n_hidden_reverse[i-1], self.n_hidden_reverse[i]))
         self.decoder_bn_list.append(nn.BatchNorm1d(self.n_hidden_reverse[i]))
      
      # output layer
      self.out = nn.Linear(self.n_hidden[0], self.total_features)
      
      # activation functions
      self.relu = nn.LeakyReLU()
      self.softplus = nn.Softplus()
      self.dropout = nn.Dropout(p=self.dp_rate)
   
   def encode(self, x):
      # hidden representation
      h_in = list()
      for i in range(len(self.n_hidden)):
         if i == 0:
            h_in.append(self.dropout(self.encoder_bn_list[i](self.relu(self.encoder_list[i](x)))))
         else:
            h_in.append(self.dropout(self.encoder_bn_list[i](self.relu(self.encoder_list[i](h_in[i-1])))))
      
      # latent
      z_mu = self.zmu(h_in[-1])
      z_lsig = self.softplus(self.zlsig(h_in[-1]))
      return z_mu, z_lsig
   
   # sample with gaussian noise
   def reparameterize(self, mu_, logsig):
      eps = to_var(torch.randn(mu_.size(0), mu_.size(1)))
      z = mu_ + eps * torch.exp(logsig/2)
      return z
   
   def decode(self, z):
      # reconstruct x from latent
      h_out = list()
      for i in range(len(self.n_hidden)):
         if i == 0:
            h_out.append(self.dropout(self.relu(self.decoder_bn_list[i](self.decoder_list[i](z)))))
         else:
            h_out.append(self.dropout(self.relu(self.decoder_bn_list[i](self.decoder_list[i](h_out[i-1])))))
      y_both = self.out(h_out[-1])
      
      # decompose into data and tnf signals
      y_data = F.softmax(y_both.narrow(1, 0, self.n_features[0]), dim=1)
      y_tnf = y_both.narrow(1, self.n_features[0], self.n_features[1])
      return y_data, y_tnf
   
   # forward pass 
   def forward(self, x_data, x_tnf):
      x = torch.cat((x_data,x_tnf), 1)
      mu, lsig = self.encode(x)
      z = self.reparameterize(mu, lsig)
      y_data,y_tnf = self.decode(z)
      return y_data, y_tnf, mu, lsig


def loss_function(y, x, y_tnf, x_tnf, mu, lsig, beta, mse):
   #BCE = F.binary_cross_entropy(y, x, size_average=False)
   #KLD = -0.5 * torch.sum(1 + lsig - mu.pow(2) - lsig.exp())
   criterion = nn.MSELoss() 
   MSE = mse * criterion(y_tnf, x_tnf)
   BCE = F.binary_cross_entropy(y,x, size_average=False) 
   KLD = -0.5 * torch.mean(1 + lsig - mu.pow(2) - lsig.exp())
   loss = BCE + beta*KLD + MSE
   return BCE, KLD, loss, MSE


def train(epoch, model, data_loader, optimizer, beta, mse):
    model.train()
    train_loss = 0
    KLD_loss = 0
    BCE_loss = 0
    MSE_loss = 0
    for batch_idx, (data,tnf) in enumerate(data_loader):
        data = Variable(data)
        tnf = Variable(tnf)
        if args.cuda:
            data = data.cuda()
            tnf = tnf.cuda()
        optimizer.zero_grad()
        y_data, y_tnf, mu, logvar = model(data, tnf)
        BCE, KLD, loss, MSE = loss_function(y_data, data, y_tnf, tnf, mu, logvar, beta, mse)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        KLD_loss += KLD.data[0]
        BCE_loss += BCE.data[0]
        MSE_loss += MSE.data[0]
        if batch_idx % 50 == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tBCE: {:.4f}\tKLD: {:.4f}\tBeta: {:.2f}'.format(
            epoch, batch_idx * len(data), len(data_loader.dataset),
            100. * batch_idx / len(data_loader),
            loss.data[0] / len(data),
            BCE.data[0] / len(data),
            KLD.data[0] / len(data),
            beta))

    print('====> Epoch: {}\tLoss: {:.4f}\tBCE: {:.4f}\tKLD: {:.4f}\tMSE: {:.5f}\tBeta: {:.2f}'.format(
          epoch, train_loss / len(data_loader.dataset),
          BCE_loss / len(data_loader.dataset),
          KLD_loss / len(data_loader.dataset),
          MSE_loss / len(data_loader.dataset),
          beta))

def evaluate(model, data_loader, beta, mse, o):
   # write out
   f_out = '%s/latent.representation.tab' % o
   if os.path.exists(f_out): 
      os.remove(f_out)
   fh_out = open(f_out, 'ab')
   
   # evaluate
   model.eval()
   for batch_idx,(data,tnf) in enumerate(data_loader):
      data = Variable(data)
      tnf = Variable(tnf)
      if args.cuda:
         data = data.cuda()
         tnf = tnf.cuda()
      y_data, y_tnf, mu, logvar = model(data, tnf)
      BCE, KLD, loss, MSE = loss_function(y_data, data, y_tnf, tnf, mu, logvar, beta, mse)
      if batch_idx % 50 == 0:
         print('Eval [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tBCE: {:.4f}\tKLD: {:.4f}\tBeta: {:.2f}'.format(
            batch_idx * len(data), len(data_loader.dataset),
            100. * batch_idx / len(data_loader),
            loss.data[0] / len(data),
            BCE.data[0] / len(data),
            KLD.data[0] / len(data),
            beta))
      
      if args.cuda:
         mu = mu.cpu()
      np.savetxt(fh_out, mu.data.numpy(), fmt="%.3f", delimiter="\t")
   
   fh_out.close()

def main(args):
   '''Main program'''
   
   # get number of features
   n_features = []
   n_features.append(get_features(args.i[0]))
   n_features.append(get_features(args.tnf[0]))
   
   # load data for training
   if args.to_memory:
      data_loader = load_data_mem(args)
   else:
      data_loader = load_data(args)
   
   # define model and optimizer
   model = VAE(n_features, args.dp_rate)
   if args.cuda:
      model.cuda()
   optimizer = optim.Adam(model.parameters(), lr=args.lrate)
   
   # set beta
   if args.warmup == 0:
      beta = 1
      to_add = 0
   else:
      to_add = 1/args.warmup
      beta = 0
   
   # train
   for epoch in range(1, args.nepochs + 1):
      train(epoch, model, data_loader, optimizer, beta, args.mse)
      if beta >= 1:
         beta = 1
      else:
         beta = beta + to_add
   
   # evaluate
   # currently data does not need to be reloaded because input-data is already shuffled #
   # load data for eval
   #if args.to_memory:
   #   data_loader_eval = load_data_mem(args, training=False)
   #else:
   #   data_loader_eval = load_data(args, training=False)
   #evaluate(model, data_loader_eval, beta, args.o)
   evaluate(model, data_loader, beta, args.mse, args.o)

if __name__ == '__main__':
   # create the parser
   parser = argparse.ArgumentParser(prog='pt.vae.both.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), usage='%(prog)s [options]', description='Variational Autoencoder using PyTorch on contig abundance and TNF')
   
   parser.add_argument('--i', help='input depth data', default=None, nargs="+")
   parser.add_argument('--tnf', help='input tnf', default=None, nargs="+")
   parser.add_argument('--hidden', help='number of hidden neurons (list)', type=int, nargs="+", required=True)
   parser.add_argument('--latent', help='number of latent neurons', required=True)
   parser.add_argument('--batch', help='batch size [100]', type=int, default=100)
   parser.add_argument('--nepochs', help='number of epochs [10]', type=int, default=10)
   parser.add_argument('--warmup', help='use warmup for this number of epochs [0]', type=int, default=0)
   parser.add_argument('--lrate', help='learning rate [1e-4]', type=float, default=1e-4)
   parser.add_argument('--dp_rate', help='dropout rate [0.1]', type=float, default=0.1)
   parser.add_argument('--mse', help='scaling factor for MSE loss [1]', type=float, default=1)
   parser.add_argument('--to-memory', help='load all data to memory (False)', default=False, action='store_true')
   parser.add_argument('--cuda', help='use cuda (False)', default=False, action='store_true')
   parser.add_argument('--o', help='output folder [vae]', default='vae')
   
   args = parser.parse_args()
   #args = parser.parse_args('--i t_depth.gz --tnf t_tnf.gz --batch 100 --hidden 250 200 --latent 150 --nepochs 2 --warmup 20 --o pt_vae_test_both_test --mse 10'.split())
   
   # make output dir
   if not os.path.exists(args.o):
      os.makedirs(args.o)
   
   # load data, create graph and layers, run calculations and report
   main(args)
   


