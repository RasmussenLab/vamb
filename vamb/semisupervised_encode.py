"""Semisupervised multimodal VAEs for metagenomics binning, this files contains the implementation of the VAEVAE for MMSEQ predictions"""


__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
import torch as _torch
import torch.nn.functional as F
_torch.manual_seed(0)
from functools import partial, partialmethod

from math import log as _log

from torch import nn as _nn
from torch.optim import Adam as _Adam
from torch.nn.functional import softmax as _softmax
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.sampler import SubsetRandomSampler as _SubsetRandomSampler
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import vamb.vambtools as _vambtools
import vamb.encode as _encode

if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')


def collate_fn_labels(num_categories, batch):
    return [F.one_hot(_torch.as_tensor(batch), num_classes=max(num_categories, 105)).squeeze(1).float()]


def collate_fn_concat(num_categories, batch):
        a = _torch.stack([i[0] for i in batch])
        b = _torch.stack([i[1] for i in batch])
        c = _torch.stack([i[2] for i in batch])
        d = [i[3] for i in batch]
        return a, b, c, F.one_hot(_torch.as_tensor(d), num_classes=max(num_categories, 105)).squeeze(1).float()


def collate_fn_semisupervised(num_categories, batch):
        a = _torch.stack([i[0] for i in batch])
        b = _torch.stack([i[1] for i in batch])
        c = _torch.stack([i[2] for i in batch])
        d = [i[3] for i in batch]
        e = _torch.stack([i[4] for i in batch])
        f = _torch.stack([i[5] for i in batch])
        g = _torch.stack([i[6] for i in batch])
        h = [i[7] for i in batch]
        return a, b, c, F.one_hot(_torch.as_tensor(d), num_classes=max(num_categories, 105)).squeeze(1).float(), \
               e, f, g, F.one_hot(_torch.as_tensor(h), num_classes=max(num_categories, 105)).squeeze(1).float()


def kld_gauss(p_mu, p_logstd, q_mu, q_logstd):
    p = _torch.distributions.normal.Normal(p_mu, p_logstd.exp())
    q = _torch.distributions.normal.Normal(q_mu, q_logstd.exp())
    loss = _torch.distributions.kl_divergence(p, q)
    return loss.mean()


def _make_dataset(rpkm, tnf, lengths, batchsize=256, destroy=False, cuda=False):
    n_workers = 4 if cuda else 1
    dataloader, mask = _encode.make_dataloader(rpkm, tnf, lengths, batchsize, destroy, cuda)
    depthstensor, tnftensor, weightstensor = dataloader.dataset.tensors
    return depthstensor, tnftensor, weightstensor, batchsize, n_workers, cuda, mask


def make_dataloader_concat(rpkm, tnf, lengths, labels, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, weightstensor, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels_int = _np.unique(labels, return_inverse=True)[1]
    dataset = _TensorDataset(depthstensor, tnftensor, weightstensor, _torch.from_numpy(labels_int))
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda, collate_fn=partial(collate_fn_concat, len(set(labels_int))))
    return dataloader, mask


def make_dataloader_labels(rpkm, tnf, lengths, labels, batchsize=256, destroy=False, cuda=False):
    _, _, _, batchsize, n_workers, cuda, mask = _make_dataset(rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda)
    labels_int = _np.unique(labels, return_inverse=True)[1]
    dataset = _TensorDataset(_torch.from_numpy(labels_int))
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda, collate_fn=partial(collate_fn_labels, len(set(labels_int))))

    return dataloader, mask


def permute_indices(n_current, n_total):
    x = _np.arange(n_current)
    to_add = int(n_total / n_current)
    return _np.concatenate([_np.random.permutation(x)] + [_np.random.permutation(x) for _ in range(to_add)])[:n_total]


def make_dataloader_semisupervised(dataloader_joint, dataloader_vamb, dataloader_labels, shapes, batchsize=256, destroy=False, cuda=False):
    n_labels = shapes[-1]
    n_total = len(dataloader_vamb.dataset)
    indices_unsup_vamb = permute_indices(len(dataloader_vamb.dataset), n_total)
    indices_unsup_labels = permute_indices(len(dataloader_labels.dataset), n_total)
    indices_sup = permute_indices(len(dataloader_joint.dataset), n_total)
    dataset_all = _TensorDataset(
        dataloader_vamb.dataset.tensors[0][indices_unsup_vamb], 
        dataloader_vamb.dataset.tensors[1][indices_unsup_vamb], 
        dataloader_vamb.dataset.tensors[2][indices_unsup_vamb], 
        dataloader_labels.dataset.tensors[0][indices_unsup_labels], 
        dataloader_joint.dataset.tensors[0][indices_sup], 
        dataloader_joint.dataset.tensors[1][indices_sup], 
        dataloader_joint.dataset.tensors[2][indices_sup], 
        dataloader_joint.dataset.tensors[3][indices_sup], 
    )
    dataloader_all = _DataLoader(dataset=dataset_all, batch_size=batchsize, drop_last=True,
                        shuffle=True, num_workers=dataloader_joint.num_workers, pin_memory=cuda, collate_fn=partial(collate_fn_semisupervised, n_labels))
    return dataloader_all


class VAELabels(_encode.VAE):
    """Variational autoencoder that encodes only the labels, subclass of VAE.
    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(self, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAELabels, self).__init__(nlabels - 103, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nlabels = nlabels

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)
        labels_out = reconstruction.narrow(1, 0, self.nlabels)
        return labels_out

    def forward(self, labels):
        mu, logsigma = self._encode(labels)
        latent = self.reparameterize(mu, logsigma)
        labels_out = self._decode(latent)
        return labels_out, mu, logsigma

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels*ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for labels_in in data_loader:
            labels_in = labels_in[0]
            labels_in.requires_grad = True

            if self.usecuda:
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            labels_out, mu, logsigma = self(labels_in)

            loss, ce_labels, kld, correct_labels = self.calc_loss(labels_in, labels_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_celabelsloss += ce_labels.data.item()
            epoch_correct_labels+= correct_labels.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celabelsloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  epoch_correct_labels / (len(data_loader)*256),
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
                                      num_workers=0,
                                      pin_memory=data_loader.pin_memory, 
                                      collate_fn=data_loader.collate_fn)

        labels_array = data_loader.dataset.tensors
        length = len(labels_array[0])

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for labels in new_data_loader:
                labels = labels[0]
                # Move input to GPU if requested
                if self.usecuda:
                    labels = labels.cuda()

                # Evaluate
                out_labels, mu, logsigma = self(labels)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent


class VAEConcat(_encode.VAE):
    """Variational autoencoder that uses labels as concatented input, subclass of VAE.
    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(VAEConcat, self).__init__(nsamples + nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nsamples = nsamples
        self.nlabels = nlabels

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        labels_out = reconstruction.narrow(1, self.nsamples + self.ntnf, self.nlabels)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)
            labels_out = _softmax(labels_out, dim=1)

        return depths_out, tnf_out, labels_out

    def forward(self, depths, tnf, labels):
        tensor = _torch.cat((depths, tnf, labels), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out, labels_out = self._decode(latent)

        return depths_out, tnf_out, labels_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, mu, logsigma, weights):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = (ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight)*weights

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce, sse, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in, weights, labels_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            labels_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                weights = weights.cuda()
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, labels_out, mu, logsigma = self(depths_in, tnf_in, labels_in)

            loss, ce, sse, ce_labels, kld, correct_labels = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, labels_in, labels_out, mu, logsigma, weights)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()
            epoch_celabelsloss += ce_labels.data.item()
            epoch_correct_labels+= correct_labels.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_celabelsloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  epoch_correct_labels / (len(data_loader)*256),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        self.eval()
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
                                      pin_memory=data_loader.pin_memory,
                                      collate_fn=data_loader.collate_fn)

        depths_array, _, _, _ = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, weights, labels in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    labels = labels.cuda()

                # Evaluate
                _, _, _, mu, _ = self(depths, tnf, labels)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent


class VAEVAE(object):
    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        self.VAEVamb = _encode.VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEJoint = VAEConcat(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def calc_loss_joint(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, 
                            mu_sup, logsigma_sup, 
                            mu_vamb_unsup, logsigma_vamb_unsup,
                            mu_labels_unsup, logsigma_labels_unsup,
                            weights,
                            ):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.VAEVamb.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.VAEVamb.alpha) / _log(self.VAEVamb.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.VAEVamb.alpha

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1. #TODO: figure out

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)
        sse_weight = self.VAEVamb.alpha / self.VAEVamb.ntnf
        kld_weight = 1 / (self.VAEVamb.nlatent * self.VAEVamb.beta)
        reconstruction_loss = ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight

        kld_vamb = kld_gauss(mu_sup, logsigma_sup, mu_vamb_unsup, logsigma_vamb_unsup)
        kld_labels = kld_gauss(mu_sup, logsigma_sup, mu_labels_unsup, logsigma_labels_unsup)
        kld = kld_vamb + kld_labels
        kld_loss = kld * kld_weight

        loss = (reconstruction_loss + kld_loss) * weights

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss.mean(), ce.mean(), sse.mean(), ce_labels, kld_vamb.mean(), kld_labels.mean(), _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        metrics = [
            'loss_vamb', 'ce_vamb', 'sse_vamb', 'kld_vamb', 
            'loss_labels', 'ce_labels_labels', 'kld_labels', 'correct_labels_labels',
            'loss_joint', 'ce_joint', 'sse_joint', 'ce_labels_joint', 'kld_vamb_joint', 'kld_labels_joint', 'correct_labels_joint',
            'loss',
        ]
        metrics_dict = {k: 0 for k in metrics}
        tensors_dict = {k: None for k in metrics}

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in_sup, tnf_in_sup, weights_in_sup, labels_in_sup, depths_in_unsup, tnf_in_unsup, weights_in_unsup, labels_in_unsup in data_loader:
            depths_in_sup.requires_grad = True
            tnf_in_sup.requires_grad = True
            labels_in_sup.requires_grad = True
            depths_in_unsup.requires_grad = True
            tnf_in_unsup.requires_grad = True
            labels_in_unsup.requires_grad = True

            if self.VAEVamb.usecuda:
                depths_in_sup = depths_in_sup.cuda()
                tnf_in_sup = tnf_in_sup.cuda()
                weights_in_sup = weights_in_sup.cuda()
                labels_in_sup = labels_in_sup.cuda()
                depths_in_unsup = depths_in_unsup.cuda()
                tnf_in_unsup = tnf_in_unsup.cuda()
                weights_in_unsup = weights_in_unsup.cuda()
                labels_in_unsup = labels_in_unsup.cuda()

            optimizer.zero_grad()

            _, _, _, mu_sup, logsigma_sup = self.VAEJoint(depths_in_sup, tnf_in_sup, labels_in_sup) # use the two-modality latent space

            depths_out_sup, tnf_out_sup = self.VAEVamb._decode(self.VAEVamb.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders
            labels_out_sup = self.VAELabels._decode(self.VAELabels.reparameterize(mu_sup, logsigma_sup)) # use the one-modality decoders

            depths_out_unsup, tnf_out_unsup,  mu_vamb_unsup, logsigma_vamb_unsup = self.VAEVamb(depths_in_unsup, tnf_in_unsup)
            labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup = self.VAELabels(labels_in_unsup)

            tensors_dict['loss_vamb'], tensors_dict['ce_vamb'], tensors_dict['sse_vamb'], tensors_dict['kld_vamb'] = \
                self.VAEVamb.calc_loss(depths_in_unsup, depths_out_unsup, tnf_in_unsup, tnf_out_unsup, mu_vamb_unsup, logsigma_vamb_unsup, weights_in_unsup)
            tensors_dict['loss_labels'], tensors_dict['ce_labels_labels'], tensors_dict['kld_labels'], tensors_dict['correct_labels_labels'] = \
                self.VAELabels.calc_loss(labels_in_unsup, labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup)
            
            tensors_dict['loss_joint'], tensors_dict['ce_joint'], tensors_dict['sse_joint'], tensors_dict['ce_labels_joint'], \
                tensors_dict['kld_vamb_joint'], tensors_dict['kld_labels_joint'], tensors_dict['correct_labels_joint'] = self.calc_loss_joint(
                    depths_in_sup, depths_out_sup, tnf_in_sup, tnf_out_sup, labels_in_sup, labels_out_sup, 
                    mu_sup, logsigma_sup, 
                    mu_vamb_unsup, logsigma_vamb_unsup,
                    mu_labels_unsup, logsigma_labels_unsup,
                    weights_in_sup,
                )

            tensors_dict['loss'] = tensors_dict['loss_joint'] + tensors_dict['loss_vamb'] + tensors_dict['loss_labels']

            tensors_dict['loss'].backward()
            optimizer.step()

            for k, v in tensors_dict.items():
                metrics_dict[k] += v.data.item()

        metrics_dict['correct_labels_joint'] /= 256
        metrics_dict['correct_labels_labels'] /= 256
        if logfile is not None:
            print(', '.join([k + f' {v/len(data_loader):.6f}' for k, v in metrics_dict.items()]), file=logfile)
            logfile.flush()

        return data_loader

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.
        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]
        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(list(self.VAEVamb.parameters()) + list(self.VAELabels.parameters()) + list(self.VAEJoint.parameters()), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.VAEVamb.usecuda, file=logfile)
            print('\tAlpha:', self.VAEVamb.alpha, file=logfile)
            print('\tBeta:', self.VAEVamb.beta, file=logfile)
            print('\tDropout:', self.VAEVamb.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.VAEVamb.nhiddens)), file=logfile)
            print('\tN latent:', self.VAEVamb.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load
        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.VAEVamb.nsamples,
                 'nlabels': self.VAELabels.nlabels,
                 'alpha': self.VAEVamb.alpha,
                 'beta': self.VAEVamb.beta,
                 'dropout': self.VAEVamb.dropout,
                 'nhiddens': self.VAEVamb.nhiddens,
                 'nlatent': self.VAEVamb.nlatent,
                 'state_VAEVamb': self.VAEVamb.state_dict(),
                 'state_VAELabels': self.VAELabels.state_dict(),
                 'state_VAEJoint': self.VAEJoint.state_dict(),
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
        nlabels = dictionary['nlabels']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']

        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.VAEVamb.load_state_dict(dictionary['state_VAEVamb'])
        vae.VAELabels.load_state_dict(dictionary['state_VAELabels'])
        vae.VAEJoint.load_state_dict(dictionary['state_VAEJoint'])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.cuda()
            vae.VAEJoint.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae.VAEJoint.eval()

        return vae
    

def product_of_gaussians(p_mu, p_std, q_mu, q_std):
    var1, var2 = p_std**2, q_std**2
    mu = (p_mu*var2 + q_mu*var1) / (var1 + var2)
    std = _torch.sqrt(var2*var1 / (var1 + var2))
    return mu, std


def make_PoE_Vamb(expert1, expert2):
    class PoE_Vamb(_encode.VAE):
        def forward_poe_vamb(self, depths, tnf, expert1=None, expert2=None):
            """
            A partial method to patch on a VAE to make it PoE_VAE
            """
            tensor = _torch.cat((depths, tnf), 1)
            mu1, logsigma1 = expert1._encode(tensor)
            mu2, logsigma2 = expert2._encode(tensor)

            mu, sigma = product_of_gaussians(mu1, logsigma1.exp(), mu2, logsigma2.exp())
            logsigma = sigma.log()

            latent = expert1.reparameterize(mu, logsigma)

            depths_out, tnf_out = self._decode(latent) # decoder weights are in THIS class
            return depths_out, tnf_out, mu, logsigma
        
        forward = partialmethod(forward_poe_vamb, expert1=expert1, expert2=expert2)
    return PoE_Vamb


def make_PoE_Labels(expert1, expert2):
    class PoE_Labels(VAELabels):
        def forward_poe_labels(self, labels, expert1=None, expert2=None):
            """
            A partial method to patch on a VAE to make it PoE_VAE
            """
            mu1, logsigma1 = expert1._encode(labels)
            mu2, logsigma2 = expert2._encode(labels)

            mu, sigma = product_of_gaussians(mu1, logsigma1.exp(), mu2, logsigma2.exp())
            logsigma = sigma.log()

            latent = expert1.reparameterize(mu, logsigma)

            labels_out = self._decode(latent) # decoder weights are in THIS class
            return labels_out, mu, logsigma
        
        forward = partialmethod(forward_poe_labels, expert1=expert1, expert2=expert2)

    return PoE_Labels


def make_PoE_Joint(expert1, expert2):
    class PoE_Joint(VAEConcat):
        def forward_poe_joint(self, depths, tnf, labels, expert1=None, expert2=None):
            """
            A partial method to patch on a VAE to make it PoE_VAE
            """
            tensor = _torch.cat((depths, tnf), 1)
            mu1, logsigma1 = expert1._encode(tensor)
            mu2, logsigma2 = expert2._encode(labels)

            mu, sigma = product_of_gaussians(mu1, logsigma1.exp(), mu2, logsigma2.exp())
            logsigma = sigma.log()

            latent = expert1.reparameterize(mu, logsigma)

            depths_out, tnf_out = expert1._decode(latent)
            labels_out = expert2._decode(latent)

            return depths_out, tnf_out, labels_out, mu, logsigma
        
        forward = partialmethod(forward_poe_joint, expert1=expert1, expert2=expert2)

    return PoE_Joint


class SVAE(VAEVAE):
    def __init__(self, nsamples, nlabels, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        self._VAEVamb = _encode.VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self._VAELabels = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self._VAEVamb_joint = _encode.VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self._VAELabels_joint = VAELabels(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

        self.VAEVamb = make_PoE_Vamb(self._VAEVamb, self._VAEVamb_joint)(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = make_PoE_Labels(self._VAELabels, self._VAELabels_joint)(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAEJoint = make_PoE_Joint(self._VAEVamb_joint, self._VAELabels_joint)(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load
        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.VAEVamb.nsamples,
                 'nlabels': self.VAELabels.nlabels,
                 'alpha': self.VAEVamb.alpha,
                 'beta': self.VAEVamb.beta,
                 'dropout': self.VAEVamb.dropout,
                 'nhiddens': self.VAEVamb.nhiddens,
                 'nlatent': self.VAEVamb.nlatent,
                 'state_VAEVamb': self._VAEVamb.state_dict(),
                 'state_VAELabels': self._VAELabels.state_dict(),
                 'state_VAEVamb_joint': self._VAEVamb_joint.state_dict(),
                 'state_VAELabels_joint': self._VAELabels_joint.state_dict(),
                 'state_VAEVamb_decoder': self.VAEVamb.state_dict(),
                 'state_VAELabels_decoder': self.VAELabels.state_dict(),
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
        nlabels = dictionary['nlabels']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']

        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae._VAEVamb.load_state_dict(dictionary['state_VAEVamb'])
        vae._VAELabels.load_state_dict(dictionary['state_VAELabels'])
        vae._VAEVamb_joint.load_state_dict(dictionary['state_VAEVamb_joint'])
        vae._VAELabels_joint.load_state_dict(dictionary['state_VAELabels_joint'])
        vae.VAEVamb.load_state_dict(dictionary['state_VAEVamb_decoder'])
        vae.VAELabels.load_state_dict(dictionary['state_VAELabels_decoder'])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.cuda()
            vae._VAEVamb_joint.cuda()
            vae._VAELabels_joint.cuda()
            vae.VAEVamb.cuda()
            vae.VAELabels.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae._VAEVamb_joint.eval()
            vae._VAELabels_joint.eval()
            vae.VAEVamb.eval()
            vae.VAELabels.eval()

        return vae
