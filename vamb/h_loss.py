"""Hierarchical loss for the labels suggested in https://arxiv.org/abs/2210.10929"""


__cmd_doc__ = """Hierarchical loss for the labels"""

import torch as _torch
from torch import nn as _nn
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader as _DataLoader
_torch.manual_seed(0)

import networkx as nx

import numpy as _np
from functools import partial

from math import log as _log

import vamb.semisupervised_encode as _semisupervised_encode
import vamb.encode as _encode

if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')


def get_level(x):
    if x.startswith('d_'):
        return 1
    if x.startswith('p_'):
        return 2
    if x.startswith('c_'):
        return 3
    if x.startswith('o_'):
        return 4
    if x.startswith('f_'):
        return 5
    if x.startswith('g_'):
        return 6
    if x.startswith('s_'):
        return 7


def make_graph(taxs):
    table_indices = []
    table_true = []
    table_walkdown = []
    table_parent = []
    G = nx.DiGraph()
    G_op = nx.Graph()
    root = 'Domain'
    G.add_edge(root, 'd_Archaea')
    G_op.add_edge(root, 'd_Archaea')
    G.add_edge(root, 'd_Bacteria')
    G_op.add_edge(root, 'd_Bacteria')
    for i, row in enumerate(taxs):
        try:
            r = row.split(';')
        except AttributeError:
            print(row)
        if len(r) == 1:
            continue
        for j in range(1, len(r)):
            G.add_edge(r[j-1], r[j])
            G_op.add_edge(r[j-1], r[j])
        if (i+1) % 1000 == 0:
            print('Processed', i)
    edges = nx.bfs_edges(G, root)
    nodes = [root] + [v for u, v in edges]
    ind_nodes = {v: i for i, v in enumerate(nodes)}
    for i, n in enumerate(nodes):
        res = {}
        res_true = {}
        table_walkdown.append([ind_nodes[s] for s in G.successors(n)])
        if n == root:
            table_indices.append(res)
            table_true.append(res_true)
            table_parent.append(None)
            continue
        path = nx.shortest_path(G_op, n, root)
        table_parent.append(ind_nodes[list(G.predecessors(n))[0]])
        for p in path[:-1]:
            parent = list(G.predecessors(p))[0]
            siblings = list(G.successors(parent))
            res[get_level(p)] = [ind_nodes[s] for s in siblings]
            res_true[get_level(p)] = [j for j, v in enumerate(res[get_level(p)]) if v == ind_nodes[p]][0]
        table_indices.append(res)
        table_true.append(res_true)
    return nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent


def walk_down(logit, table_walkdown):
    res = []
    i = 0
    while table_walkdown[i]:
        max_ind_rel = logit[table_walkdown[i]].argmax().item()
        max_ind_abs = table_walkdown[i][max_ind_rel]
        res.append(max_ind_abs)
        i = max_ind_abs
    return res


def walk_up(target, table_parent):
    i = target
    res = [i]
    while table_parent[i]:
        i = table_parent[i]
        res.append(i)
    return res


def collate_fn_labels_hloss(num_categories, table_parent, labels): # target in form of indices
    target = [walk_up(l, table_parent) for l in labels]
    batch = _torch.zeros(len(target), max(num_categories, 105))
    for i, t in enumerate(target):
        batch[i, t] = 1
    return [batch.float()]


def collate_fn_concat_hloss(num_categories, table_parent, batch):
        a = _torch.stack([i[0] for i in batch])
        b = _torch.stack([i[1] for i in batch])
        c = _torch.stack([i[2] for i in batch])
        d = [i[3] for i in batch]
        return a, b, c, collate_fn_labels_hloss(num_categories, table_parent, d)[0]


def collate_fn_semisupervised_hloss(num_categories, table_parent, batch):
        a = _torch.stack([i[0] for i in batch])
        b = _torch.stack([i[1] for i in batch])
        c = _torch.stack([i[2] for i in batch])
        d = [i[3] for i in batch]
        e = _torch.stack([i[4] for i in batch])
        f = _torch.stack([i[5] for i in batch])
        g = _torch.stack([i[6] for i in batch])
        h = [i[7] for i in batch]
        return a, b, c, collate_fn_labels_hloss(num_categories, table_parent, d)[0], \
               e, f, g, collate_fn_labels_hloss(num_categories, table_parent, h)[0]


def make_dataloader_labels_hloss(rpkm, tnf, lengths, labels, N, table_parent, batchsize=256, destroy=False, cuda=False):
    _, _, _, batchsize, n_workers, cuda, mask = _semisupervised_encode._make_dataset(rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda)
    dataset = _TensorDataset(_torch.Tensor(labels).int())
    dataloader =_DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda, collate_fn=partial(collate_fn_labels_hloss, N, table_parent))

    return dataloader, mask


def make_dataloader_concat_hloss(rpkm, tnf, lengths, labels, N, table_parent, batchsize=256, destroy=False, cuda=False):
    depthstensor, tnftensor, weightstensor, batchsize, n_workers, cuda, mask = _semisupervised_encode._make_dataset(rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda)
    dataset = _TensorDataset(depthstensor, tnftensor, weightstensor, _torch.Tensor(labels).int())
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda, collate_fn=partial(collate_fn_concat_hloss, N, table_parent))
    return dataloader, mask


def permute_indices(n_current, n_total):
    x = _np.arange(n_current)
    to_add = int(n_total / n_current)
    return _np.concatenate([_np.random.permutation(x)] + [_np.random.permutation(x) for _ in range(to_add)])[:n_total]


def make_dataloader_semisupervised_hloss(dataloader_joint, dataloader_vamb, dataloader_labels, N, table_parent, shapes, batchsize=256, destroy=False, cuda=False):
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
                        shuffle=False, num_workers=dataloader_joint.num_workers, pin_memory=cuda, collate_fn=partial(collate_fn_semisupervised_hloss, N, table_parent))
    return dataloader_all


class VAELabelsHLoss(_semisupervised_encode.VAELabels):
    """Variational autoencoder that encodes only the labels, subclass of VAE.
    Uses hierarchical loss to utilize the taxonomic tree.
    """
    def __init__(self, nlabels, table_indices, table_true, table_walkdown, nodes, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False, logfile=None):
        super(VAELabelsHLoss, self).__init__(nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nlabels = nlabels
        self.table_indices = table_indices
        self.table_true = table_true
        self.table_walkdown = table_walkdown
        self.nodes = nodes
        self.logfile = logfile

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        coefs = {i: i for i in range(1, 8)}
        celoss = _nn.CrossEntropyLoss()
        ce_labels = 0
        for i, l in enumerate(labels_in):
            t = walk_down(l, self.table_walkdown)[-1]
            for level, inds in self.table_indices[t].items():
                true_level = _torch.tensor(self.table_true[t][level])
                if self.usecuda:
                    true_level = true_level.cuda()
                ce_labels += coefs[level]*celoss(labels_out[i, inds], true_level)
        ce_labels /= labels_out.size(0)

        ce_labels_weight = 1. #TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels*ce_labels_weight + kld * kld_weight

        i = 0
        if self.logfile is not None:
            print('Correct path (sample 0 from current batch): ' + ', '.join([self.nodes[i] for i in walk_down(labels_in[0], self.table_walkdown)]), file=self.logfile)
            print('Predicted path (sample 0 from current batch): ' + ', '.join([self.nodes[i] for i in walk_down(labels_out[0], self.table_walkdown)]), file=self.logfile)
            self.logfile.flush()

        return loss, ce_labels, kld, ce_labels


class VAEConcatHLoss(_semisupervised_encode.VAEConcat):
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

    def __init__(self, nsamples, nlabels, table_indices, table_true, table_walkdown, nodes, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False, logfile=None):
        super(VAEConcatHLoss, self).__init__(nsamples, nlabels, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.nsamples = nsamples
        self.nlabels = nlabels
        self.table_indices = table_indices
        self.table_true = table_true
        self.table_walkdown = table_walkdown
        self.nodes = nodes
        self.logfile = logfile

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, labels_in, labels_out, mu, logsigma, weights):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        coefs = {i: i for i in range(1, 8)}
        celoss = _nn.CrossEntropyLoss()
        ce_labels = 0
        for i, l in enumerate(labels_in):
            t = walk_down(l, self.table_walkdown)[-1]
            for level, inds in self.table_indices[t].items():
                true_level = _torch.tensor(self.table_true[t][level])
                if self.usecuda:
                    true_level = true_level.cuda()
                ce_labels += coefs[level]*celoss(labels_out[i, inds], true_level)
        ce_labels /= labels_out.size(0)

        ce_labels_weight = 1. #TODO: figure out
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = (ce * ce_weight + sse * sse_weight + ce_labels*ce_labels_weight + kld * kld_weight)*weights

        i = 0
        if self.logfile is not None:
            print('Correct path (sample 0 from current batch): ' + ', '.join([self.nodes[i] for i in walk_down(labels_in[0], self.table_walkdown)]), file=self.logfile)
            print('Predicted path (sample 0 from current batch): ' + ', '.join([self.nodes[i] for i in walk_down(labels_out[0], self.table_walkdown)]), file=self.logfile)
            self.logfile.flush()

        return loss, ce, sse, ce_labels, kld, ce_labels
    

class VAEVAEHLoss(_semisupervised_encode.VAEVAE):
    def __init__(self, nsamples, nlabels, table_indices, table_true, table_walkdown, nodes, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False, logfile=None):
        N_l = max(nlabels, 105)
        self.VAEVamb = _encode.VAE(nsamples, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda)
        self.VAELabels = VAELabelsHLoss(N_l, table_indices, table_true, table_walkdown, nodes, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda, logfile=logfile)
        self.VAEJoint = VAEConcatHLoss(nsamples, N_l, table_indices, table_true, table_walkdown, nodes, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha,
                 beta=beta, dropout=dropout, cuda=cuda, logfile=logfile)
