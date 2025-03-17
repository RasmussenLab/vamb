"""Hierarchical loss for the labels suggested in https://arxiv.org/abs/2210.10929"""

__cmd_doc__ = """Hierarchical loss for the labels"""


import networkx as nx
import numpy as _np
from collections import namedtuple
from functools import partial
from math import log as _log
from pathlib import Path
from typing import Optional, IO, Union, Iterable, Sequence
import dadaptation
from loguru import logger

import torch as _torch
from torch import nn as _nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset as _TensorDataset
from torch.utils.data import DataLoader as _DataLoader
from torch import Tensor

import vamb.semisupervised_encode as _semisupervised_encode
import vamb.encode as _encode
import vamb.hloss_misc as _hloss
from vamb.taxonomy import ContigTaxonomy


def make_graph(
    taxes: Sequence[Optional[ContigTaxonomy]],
) -> tuple[list[str], dict[str, int], list[int]]:
    G = nx.DiGraph()
    root = "root"
    for contig_taxonomy in taxes:
        if contig_taxonomy is None:
            continue
        if len(contig_taxonomy.ranks) == 0:
            continue
        G.add_edge(root, contig_taxonomy.ranks[0])
        for parent, child in zip(contig_taxonomy.ranks, contig_taxonomy.ranks[1:]):
            G.add_edge(parent, child)
    edges = nx.bfs_edges(G, root)
    # Nodes are unique here, because even though a taxon may appear multiple times
    # in contig taxonomies, its edges will only be added once to the graph
    nodes: list[str] = [root] + [v for _, v in edges]
    ind_nodes = {v: i for i, v in enumerate(nodes)}
    # We have checked during construction of the Taxonomy that node names are unique
    # in that they can't have distinct parents, so this should hold
    assert len(ind_nodes) == len(nodes)
    table_parent: list[int] = []
    for child_node, n in enumerate(nodes):
        if n == root:
            table_parent.append(-1)
        else:
            parent = only(G.predecessors(n))
            parent_index = ind_nodes[parent]
            assert parent_index < ind_nodes[n]
            table_parent.append(parent_index)
    return nodes, ind_nodes, table_parent


def only(itr):
    itr = iter(itr)
    y = next(itr)
    try:
        next(itr)
    except StopIteration:
        return y
    raise ValueError("More than one element in iterator")


def collate_fn_labels_hloss(
    num_categories: int, table_parent, labels
):  # target in form of indices
    return _semisupervised_encode.collate_fn_labels(num_categories, labels)


def collate_fn_concat_hloss(num_categories: int, table_parent, batch):
    a = _torch.stack([i[0] for i in batch])
    b = _torch.stack([i[1] for i in batch])
    c = _torch.stack([i[2] for i in batch])
    d = _torch.stack([i[3] for i in batch])
    e = [i[4] for i in batch]
    return a, b, c, d, collate_fn_labels_hloss(num_categories, table_parent, e)[0]


def collate_fn_semisupervised_hloss(num_categories: int, table_parent, batch):
    a = _torch.stack([i[0] for i in batch])
    b = _torch.stack([i[1] for i in batch])
    c = _torch.stack([i[2] for i in batch])
    d = _torch.stack([i[3] for i in batch])
    e = [i[4] for i in batch]
    f = _torch.stack([i[5] for i in batch])
    g = _torch.stack([i[6] for i in batch])
    h = _torch.stack([i[7] for i in batch])
    m = _torch.stack([i[8] for i in batch])
    n = [i[9] for i in batch]
    return (
        a,
        b,
        c,
        d,
        collate_fn_labels_hloss(num_categories, table_parent, e)[0],
        f,
        g,
        h,
        m,
        collate_fn_labels_hloss(num_categories, table_parent, n)[0],
    )


def make_dataloader_labels_hloss(
    rpkm,
    tnf,
    lengths,
    labels,
    N,
    table_parent,
    batchsize=256,
    destroy=False,
    cuda=False,
):
    _, _, _, _, batchsize, n_workers, cuda = _semisupervised_encode._make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    dataset = _TensorDataset(_torch.Tensor(labels).long())
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=dataset.tensors[0].shape[0] > batchsize,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_labels_hloss, N, table_parent),
    )

    return dataloader


def make_dataloader_concat_hloss(
    rpkm,
    tnf,
    lengths,
    labels,
    N: int,
    table_parent,
    no_filter: bool = True,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
):
    (
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        weightstensor,
        batchsize,
        n_workers,
        cuda,
    ) = _semisupervised_encode._make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    labelstensor = _torch.Tensor(labels).long()
    dataset = _TensorDataset(
        depthstensor, tnftensor, total_abundance_tensor, weightstensor, labelstensor
    )
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=len(depthstensor) > batchsize,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_concat_hloss, N, table_parent),
    )
    return dataloader


def permute_indices(n_current: int, n_total: int, seed: int):
    rng = _np.random.default_rng(seed)
    x = _np.arange(n_current)
    to_add = n_total // n_current
    to_concatenate = [rng.permutation(x)]
    for _ in range(to_add):
        b = rng.permutation(x)
        to_concatenate.append(b)
    return _np.concatenate(to_concatenate)[:n_total]


def make_dataloader_semisupervised_hloss(
    dataloader_joint,
    dataloader_vamb,
    dataloader_labels,
    N,
    table_parent,
    shapes,
    seed: int,
    batchsize=256,
    cuda=False,
):
    n_total = len(dataloader_vamb.dataset)
    indices_unsup_vamb = permute_indices(len(dataloader_vamb.dataset), n_total, seed)
    indices_unsup_labels = permute_indices(
        len(dataloader_labels.dataset), n_total, seed
    )
    indices_sup = permute_indices(len(dataloader_joint.dataset), n_total, seed)
    dataset_all = _TensorDataset(
        dataloader_vamb.dataset.tensors[0][indices_unsup_vamb],
        dataloader_vamb.dataset.tensors[1][indices_unsup_vamb],
        dataloader_vamb.dataset.tensors[2][indices_unsup_vamb],
        dataloader_vamb.dataset.tensors[3][indices_unsup_vamb],
        dataloader_labels.dataset.tensors[0][indices_unsup_labels],
        dataloader_joint.dataset.tensors[0][indices_sup],
        dataloader_joint.dataset.tensors[1][indices_sup],
        dataloader_joint.dataset.tensors[2][indices_sup],
        dataloader_joint.dataset.tensors[3][indices_sup],
        dataloader_joint.dataset.tensors[4][indices_sup],
    )
    dataloader_all = _DataLoader(
        dataset=dataset_all,
        batch_size=batchsize,
        drop_last=len(indices_unsup_vamb) > batchsize,
        shuffle=False,
        num_workers=dataloader_joint.num_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_semisupervised_hloss, N, table_parent),
    )
    return dataloader_all


LabelMap = namedtuple("LabelMap", ["to_node", "to_target"])

HierLoss = namedtuple(
    "HierLoss", ["name", "loss_fn", "pred_helper", "pred_fn", "n_labels"]
)

DEFAULT_HIER_LOSS = "flat_softmax"


def init_hier_loss(name, tree):
    CONFIG_LOSSES = dict(
        flat_softmax=HierLoss(
            name="flat_softmax",
            loss_fn=_hloss.FlatSoftmaxNLL(tree),
            pred_helper=_hloss.SumLeafDescendants(tree, strict=False),
            pred_fn=lambda sum_fn, theta: sum_fn(F.softmax(theta, dim=-1), dim=-1),
            n_labels=tree.leaf_mask().nonzero()[0].shape[0],
        ),
        cond_softmax=HierLoss(
            name="cond_softmax",
            loss_fn=_hloss.HierSoftmaxCrossEntropy(tree),
            pred_helper=_hloss.HierLogSoftmax(tree),
            pred_fn=lambda log_softmax_fn, theta: log_softmax_fn(theta).exp(),
            n_labels=tree.num_nodes() - 1,
        ),
        soft_margin=HierLoss(
            name="soft_margin",
            loss_fn=_hloss.MarginLoss(
                tree,
                with_leaf_targets=False,
                hardness="soft",
                margin="incorrect",
                tau=0.01,
            ),
            pred_helper=_hloss.SumDescendants(tree, strict=False),
            pred_fn=lambda sum_fn, theta: sum_fn(F.softmax(theta, dim=-1), dim=-1),
            n_labels=tree.num_nodes(),
        ),
    )
    if name not in CONFIG_LOSSES:
        raise AttributeError(f"Hierarchical loss {name} not found")
    return CONFIG_LOSSES[name]


class VAELabelsHLoss(_semisupervised_encode.VAELabels):
    """Variational autoencoder that encodes only the labels.
    Subclass of one-hot VAELabels class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nlabels: int,
        nodes,
        table_parent,
        nhiddens=None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        hier_loss=DEFAULT_HIER_LOSS,
        cuda: bool = False,
    ):
        super(VAELabelsHLoss, self).__init__(
            nlabels,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.nodes = nodes
        self.table_parent = table_parent
        self.tree = _hloss.Hierarchy(table_parent)
        self.hierloss = init_hier_loss(hier_loss, self.tree)
        self.nlabels = self.hierloss.n_labels
        self.loss_fn = self.hierloss.loss_fn
        self.pred_helper = self.hierloss.pred_helper
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(self.hierloss.pred_fn, self.pred_helper)
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hloss.FindLCA(self.tree)
        node_subset = _hloss.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hloss.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        ce_labels = self.loss_fn(labels_out, labels_in)

        ce_labels_weight = 1.0  # TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels * ce_labels_weight + kld * kld_weight
        return loss, ce_labels, kld, _torch.tensor(0)

    def trainmodel(
        self,
        dataloader: _DataLoader[tuple[Tensor]],
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 300],
        modelfile: Union[None, str, Path, IO[bytes]] = None,
    ):
        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        nlabels = dataloader.dataset.tensors[0].shape  # type: ignore
        # optimizer = _Adam(self.parameters(), lr=lrate)
        optimizer = dadaptation.DAdaptAdam(self.parameters(), lr=1, decouple=True)

        logger.info("\tNetwork properties:")
        logger.info(f"\t    CUDA: {self.usecuda}")
        logger.info(f"\t    Alpha: {self.alpha}")
        logger.info(f"\t    Beta: {self.beta}")
        logger.info(f"\t    Dropout: {self.dropout}")
        logger.info(f"\t    N hidden: {', '.join(map(str, self.nhiddens))}")
        logger.info(f"\t    N latent: {self.nlatent}")
        logger.info("\tTraining properties:")
        logger.info(f"\t    N epochs: {nepochs}")
        logger.info(f"\t    Starting batch size: {dataloader.batch_size}")
        batchsteps_string = (
            ", ".join(map(str, sorted(batchsteps_set))) if batchsteps_set else "None"
        )
        logger.info(f"\t    Batchsteps: {batchsteps_string}")
        logger.info(f"\t    Learning rate: {lrate}")
        logger.info(f"\t    N labels: {nlabels}")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader, epoch, optimizer, sorted(batchsteps_set)
            )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None


class VAEConcatHLoss(_semisupervised_encode.VAEConcat):
    """Variational autoencoder that encodes the concatenated input of VAMB and labels.
    Subclass of one-hot VAEConcat class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nlabels: int,
        nodes,
        table_parent,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha=None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        hier_loss=DEFAULT_HIER_LOSS,
        cuda: bool = False,
    ):
        super(VAEConcatHLoss, self).__init__(
            nsamples,
            nlabels,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.nsamples = nsamples
        self.nodes = nodes
        self.table_parent = table_parent
        self.tree = _hloss.Hierarchy(table_parent)
        self.hierloss = init_hier_loss(hier_loss, self.tree)
        self.nlabels = self.hierloss.n_labels
        self.loss_fn = self.hierloss.loss_fn
        self.pred_helper = self.hierloss.pred_helper
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(self.hierloss.pred_fn, self.pred_helper)
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hloss.FindLCA(self.tree)
        node_subset = _hloss.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hloss.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

    def calc_loss(
        self,
        depths_in,
        depths_out,
        tnf_in,
        tnf_out,
        abundance_in,
        abundance_out,
        labels_in,
        labels_out,
        mu,
        logsigma,
        weights,
    ):
        ab_sse = (abundance_out - abundance_in).pow(2).sum(dim=1)
        # Add 1e-9 to depths_out to avoid numerical instability.
        ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)
        kld = 0.5 * (mu.pow(2)).sum(dim=1)

        # Avoid having the denominator be zero
        if self.nsamples == 1:
            ce_weight = 0.0
        else:
            ce_weight = ((1 - self.alpha) * (self.nsamples - 1)) / (
                self.nsamples * _log(self.nsamples)
            )

        ab_sse_weight = (1 - self.alpha) * (1 / self.nsamples)
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)

        ce_labels = self.loss_fn(labels_out, labels_in)
        ce_labels_weight = 1.0

        loss = (
            ab_sse * ab_sse_weight
            + ce * ce_weight
            + sse * sse_weight
            + ce_labels * ce_labels_weight
            + kld * kld_weight
        ) * weights
        return loss, ce, sse, ce_labels, kld, _torch.tensor(0)


def kld_gauss(p_mu, p_logstd, q_mu, q_logstd):
    loss = (
        q_logstd
        - p_logstd
        + (p_logstd.exp().pow(2) + (p_mu - q_mu).pow(2)) / (2 * q_logstd.exp().pow(2))
        - 0.5
    )
    return loss.mean()


class VAEVAEHLoss(_semisupervised_encode.VAEVAE):
    """Bi-modal variational autoencoder that uses TNFs, abundances and labels.
    It contains three individual encoders (VAMB, labels and concatenated VAMB+labels)
    and two decoders (VAMB and labels).
    Subclass of one-hot VAEVAE class but the labels are structured differently.
    Uses hierarchical loss to utilize the taxonomic tree.
    Instantiate with:
        nlabels: Number of labels

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        - Hyperparameters:
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nlabels: int,
        nodes,
        table_parent,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        hier_loss=DEFAULT_HIER_LOSS,
        cuda: bool = False,
    ):
        self.usecuda = cuda
        N_l = max(nlabels, 105)
        self.VAEVamb = _encode.VAE(
            nsamples,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.VAELabels = VAELabelsHLoss(
            N_l,
            nodes,
            table_parent,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            hier_loss=hier_loss,
            cuda=cuda,
        )
        self.VAEJoint = VAEConcatHLoss(
            nsamples,
            N_l,
            nodes,
            table_parent,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            hier_loss=hier_loss,
            cuda=cuda,
        )

    @classmethod
    def load(cls, path, nodes, table_parent, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.
        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]
        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = _torch.load(
            path, map_location=lambda storage, loc: storage, weights_only=False
        )

        nsamples = dictionary["nsamples"]
        nlabels = dictionary["nlabels"]
        alpha = dictionary["alpha"]
        beta = dictionary["beta"]
        dropout = dictionary["dropout"]
        nhiddens = dictionary["nhiddens"]
        nlatent = dictionary["nlatent"]

        vae = cls(
            nsamples,
            nlabels,
            nodes,
            table_parent,
            nhiddens,
            nlatent,
            alpha,
            beta,
            dropout,
            cuda=cuda,
        )
        vae.VAEVamb.load_state_dict(dictionary["state_VAEVamb"])
        vae.VAELabels.load_state_dict(dictionary["state_VAELabels"])
        vae.VAEJoint.load_state_dict(dictionary["state_VAEJoint"])

        if cuda:
            vae.VAEVamb.cuda()
            vae.VAELabels.cuda()
            vae.VAEJoint.cuda()

        if evaluate:
            vae.VAEVamb.eval()
            vae.VAELabels.eval()
            vae.VAEJoint.eval()

        return vae

    def calc_loss_joint(
        self,
        depths_in,
        depths_out,
        tnf_in,
        tnf_out,
        abundance_in,
        abundance_out,
        labels_in,
        labels_out,
        mu_sup,
        logsigma_sup,
        mu_vamb_unsup,
        logsigma_vamb_unsup,
        mu_labels_unsup,
        logsigma_labels_unsup,
        weights,
    ):
        ab_sse = (abundance_out - abundance_in).pow(2).sum(dim=1)
        # Add 1e-9 to depths_out to avoid numerical instability.
        ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)

        # Avoid having the denominator be zero
        if self.VAEVamb.nsamples == 1:
            ce_weight = 0.0
        else:
            ce_weight = ((1 - self.VAEVamb.alpha) * (self.VAEVamb.nsamples - 1)) / (
                self.VAEVamb.nsamples * _log(self.VAEVamb.nsamples)
            )

        ab_sse_weight = (1 - self.VAEVamb.alpha) * (1 / self.VAEVamb.nsamples)
        sse_weight = self.VAEVamb.alpha / self.VAEVamb.ntnf

        ce_labels = self.VAEJoint.loss_fn(labels_out, labels_in)
        ce_labels_weight = 1.0  # TODO: figure out

        weighed_ab = ab_sse * ab_sse_weight
        weighed_ce = ce * ce_weight
        weighed_sse = sse * sse_weight
        weighed_labels = ce_labels * ce_labels_weight

        reconstruction_loss = weighed_ce + weighed_ab + weighed_sse + weighed_labels

        kld_vamb = kld_gauss(mu_sup, logsigma_sup, mu_vamb_unsup, logsigma_vamb_unsup)
        kld_labels = kld_gauss(
            mu_sup, logsigma_sup, mu_labels_unsup, logsigma_labels_unsup
        )
        kld = kld_vamb + kld_labels
        kld_weight = 1 / (self.VAEVamb.nlatent * self.VAEVamb.beta)
        kld_loss = kld * kld_weight

        loss = (reconstruction_loss + kld_loss) * weights
        return (
            loss.mean(),
            ce.mean(),
            sse.mean(),
            ce_labels.mean(),
            kld_vamb.mean(),
            kld_labels.mean(),
            _torch.tensor(0),
        )


class VAMB2Label(_nn.Module):
    """Label predictor, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix

        - The taxonomy graph information:
        nodes: list of node indices in the order of the traversal
        table_parent: a parent for each node in the order of the traversal

        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nlabels: int,
        nodes,
        table_parent,
        nhiddens: Optional[list[int]] = None,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        hier_loss=DEFAULT_HIER_LOSS,
        cuda: bool = False,
    ):
        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if beta <= 0:
            raise ValueError(f"beta must be > 0, not {beta}")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be 0 < alpha < 1, not {alpha}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")

        super(VAMB2Label, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip(
            [self.nsamples + self.ntnf + 1] + self.nhiddens, self.nhiddens
        ):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        self.tree = _hloss.Hierarchy(table_parent)
        self.n_tree_nodes = nlabels

        self.nodes = nodes
        self.table_parent = table_parent
        self.hierloss = init_hier_loss(hier_loss, self.tree)
        self.nlabels = self.hierloss.n_labels
        self.loss_fn = self.hierloss.loss_fn
        self.pred_helper = self.hierloss.pred_helper
        if self.usecuda:
            self.loss_fn = self.loss_fn.cuda()
            self.pred_helper = self.pred_helper.cuda()
        self.pred_fn = partial(self.hierloss.pred_fn, self.pred_helper)
        self.specificity = -self.tree.num_leaf_descendants()
        self.not_trivial = self.tree.num_children() != 1
        self.find_lca = _hloss.FindLCA(self.tree)
        node_subset = _hloss.find_subset_index(self.nodes, self.nodes)
        project_to_subtree = _hloss.find_projection(self.tree, node_subset)
        label_to_node = self.tree.parents()
        label_to_subtree_node = project_to_subtree[label_to_node]
        self.eval_label_map = LabelMap(
            to_node=label_to_subtree_node,
            to_target=label_to_subtree_node,
        )

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nlabels)
        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _predict(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        tensors: list[_torch.Tensor] = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)
        labels_out = reconstruction.narrow(1, 0, self.nlabels)
        return labels_out

    def forward(self, depths, tnf, abundances, weights):
        tensor = _torch.cat((depths, tnf, abundances), 1)
        labels_out = self._predict(tensor)
        return labels_out

    def calc_loss(self, labels_in, labels_out):
        ce_labels = self.loss_fn(labels_out, labels_in)
        return ce_labels, _torch.tensor(0)

    def predict(self, data_loader) -> Iterable[tuple[_np.ndarray, _np.ndarray]]:
        self.eval()

        new_data_loader = _encode.set_batchsize(
            data_loader,
            data_loader.batch_size,
            data_loader.dataset.tensors[0].shape[0],
            encode=True,
        )

        with _torch.no_grad():
            for depths, tnf, abundances, weights in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    abundances = abundances.cuda()
                    weights = weights.cuda()

                # Evaluate
                labels = self(depths, tnf, abundances, weights)
                with _torch.no_grad():
                    prob = self.pred_fn(labels)
                    if self.usecuda:
                        prob = prob.cpu()
                    pred = _hloss.argmax_with_confidence(
                        self.specificity, prob.numpy(), 0.5, self.not_trivial
                    )
                yield prob.numpy(), pred

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
            "nsamples": self.nsamples,
            "alpha": self.alpha,
            "beta": self.beta,
            "dropout": self.dropout,
            "nhiddens": self.nhiddens,
            "nlabels": self.nlabels,
            "state": self.state_dict(),
        }

        _torch.save(state, filehandle)

    @classmethod
    def load(
        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
    ):
        raise NotImplementedError

    def trainepoch(
        self,
        data_loader,
        epoch: int,
        optimizer,
        batchsteps: list[int],
    ):
        self.train()

        epoch_celoss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            new_batchsize = data_loader.batch_size * 2
            data_loader = _DataLoader(
                dataset=data_loader.dataset,
                batch_size=new_batchsize,
                shuffle=True,
                drop_last=data_loader.dataset.tensors[0].shape[0] > new_batchsize,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
            )

        for depths_in, tnf_in, abundances_in, weights, labels_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            abundances_in.requires_grad = True
            labels_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                abundances_in = abundances_in.cuda()
                weights = weights.cuda()
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            labels_out = self(depths_in, tnf_in, abundances_in, weights)

            loss, correct_labels = self.calc_loss(labels_in, labels_out)

            loss.mean().backward()
            optimizer.step()

            epoch_celoss += loss.mean().data.item()
            epoch_correct_labels += correct_labels.data.item()

        logger.info(
            f"\tEpoch: {epoch + 1}\tCE: {epoch_celoss / len(data_loader):.7f}\tBatchsize: {data_loader.batch_size}"
        )
        self.eval()
        return data_loader

    def trainmodel(
        self,
        dataloader: _DataLoader[tuple[Tensor, ...]],
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 300],
        modelfile: Union[None, str, Path, IO[bytes]] = None,
    ):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        nlabels = dataloader.dataset.tensors[0].shape  # type: ignore
        # optimizer = _Adam(self.parameters(), lr=lrate)
        optimizer = dadaptation.DAdaptAdam(self.parameters(), lr=1, decouple=True)

        logger.info("\tNetwork properties:")
        logger.info(f"\t    CUDA: {self.usecuda}")
        logger.info(f"\t    Hierarchical loss: {self.hierloss.name}")
        logger.info(f"\t    Alpha: {self.alpha}")
        logger.info(f"\t    Beta: {self.beta}")
        logger.info(f"\t    Dropout: {self.dropout}")
        logger.info(f"\t    N hidden: {', '.join(map(str, self.nhiddens))}")
        logger.info("\tTraining properties:")
        logger.info(f"\t    N epochs: {nepochs}")
        logger.info(f"\t    Starting batch size: {dataloader.batch_size}")
        batchsteps_string = (
            ", ".join(map(str, sorted(batchsteps_set))) if batchsteps_set else "None"
        )
        logger.info(f"\t    Batchsteps: {batchsteps_string}")
        logger.info(f"\t    Learning rate: {lrate}")
        logger.info(f"\t    N labels: {nlabels}")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader, epoch, optimizer, sorted(batchsteps_set)
            )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
