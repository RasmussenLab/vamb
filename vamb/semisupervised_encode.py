"""Semisupervised multimodal VAEs for metagenomics binning, this files contains the implementation of the VAEVAE for MMSEQ predictions"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
from functools import partial
from math import log as _log
from typing import Optional
from loguru import logger

import torch as _torch
import torch.nn.functional as F
from torch import nn as _nn
from torch.optim import Adam as _Adam
from torch.nn.functional import softmax as _softmax
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import vamb.encode as _encode
from vamb.vambtools import mask_lower_bits

if _torch.__version__ < "0.4":
    raise ImportError("PyTorch version must be 0.4 or newer")


def collate_fn_labels(num_categories: int, batch):
    return [
        F.one_hot(_torch.as_tensor(batch), num_classes=max(num_categories, 105))
        .squeeze(1)
        .float()
    ]


def collate_fn_concat(num_categories: int, batch):
    a = _torch.stack([i[0] for i in batch])
    b = _torch.stack([i[1] for i in batch])
    c = _torch.stack([i[2] for i in batch])
    d = _torch.stack([i[3] for i in batch])
    e = [i[4] for i in batch]
    return (
        a,
        b,
        c,
        d,
        F.one_hot(_torch.as_tensor(e), num_classes=max(num_categories, 105))
        .squeeze(1)
        .float(),
    )


def collate_fn_semisupervised(num_categories: int, batch):
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
        F.one_hot(_torch.as_tensor(e), num_classes=max(num_categories, 105))
        .squeeze(1)
        .float(),
        f,
        g,
        h,
        m,
        F.one_hot(_torch.as_tensor(n), num_classes=max(num_categories, 105))
        .squeeze(1)
        .float(),
    )


def kld_gauss(p_mu, p_logstd, q_mu, q_logstd):
    loss = (
        q_logstd
        - p_logstd
        + (p_logstd.exp().pow(2) + (p_mu - q_mu).pow(2)) / (2 * q_logstd.exp().pow(2))
        - 0.5
    )
    return loss.mean()


def _make_dataset(
    rpkm, tnf, lengths, batchsize: int = 256, destroy: bool = False, cuda: bool = False
):
    n_workers = 4 if cuda else 1
    dataloader = _encode.make_dataloader(rpkm, tnf, lengths, batchsize, destroy, cuda)
    (
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        weightstensor,
    ) = dataloader.dataset.tensors  # type:ignore
    return (
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        weightstensor,
        batchsize,
        n_workers,
        cuda,
    )


def make_dataloader_concat(
    rpkm,
    tnf,
    lengths,
    labels,
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
    ) = _make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    labels_int = _np.unique(labels, return_inverse=True)[1]
    dataset = _TensorDataset(
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        weightstensor,
        _torch.from_numpy(labels_int),
    )
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=len(depthstensor) > batchsize,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_concat, len(set(labels_int))),
    )
    return dataloader


def make_dataloader_labels(
    rpkm,
    tnf,
    lengths,
    labels,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
):
    _, _, _, _, batchsize, n_workers, cuda = _make_dataset(
        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
    )
    labels_int = _np.unique(labels, return_inverse=True)[1]
    dataset = _TensorDataset(_torch.from_numpy(labels_int))
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=len(rpkm) > batchsize,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
        collate_fn=partial(collate_fn_labels, len(set(labels_int))),
    )

    return dataloader


def permute_indices(n_current: int, n_total: int, seed: int):
    rng = _np.random.default_rng(seed)
    x = _np.arange(n_current)
    to_add = int(n_total / n_current)
    to_concatenate = [rng.permutation(x)]
    for _ in range(to_add):
        b = rng.permutation(x)
        to_concatenate.append(b)
    return _np.concatenate(to_concatenate)[:n_total]


class VAELabels(_encode.VAE):
    """Variational autoencoder that encodes only the onehot labels, subclass of VAE.
    Instantiate with:
        nlabels: Number of labels
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
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
    ):
        super(VAELabels, self).__init__(
            nlabels - 104,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
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
        mu = self._encode(labels)
        logsigma = _torch.zeros(mu.size())
        if self.usecuda:
            logsigma = logsigma.cuda()
        latent = self.reparameterize(mu)
        labels_out = self._decode(latent)
        return labels_out, mu, logsigma

    def calc_loss(self, labels_in, labels_out, mu, logsigma):
        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1.0  # TODO: figure out
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce_labels * ce_labels_weight + kld * kld_weight

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return loss, ce_labels, kld, _torch.sum(labels_out_indices == labels_in_indices)

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            new_batch_size = data_loader.batch_size * 2  # type: ignore
            drop_last = data_loader.dataset.tensors[0].shape[0] > new_batch_size
            data_loader = _DataLoader(
                dataset=data_loader.dataset,
                batch_size=new_batch_size,
                shuffle=True,
                drop_last=drop_last,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
            )

        for labels_in in data_loader:
            labels_in = labels_in[0]
            labels_in.requires_grad = True

            if self.usecuda:
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            labels_out, mu, logsigma = self(labels_in)

            loss, ce_labels, kld, correct_labels = self.calc_loss(
                labels_in, labels_out, mu, logsigma
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_celabelsloss += ce_labels.data.item()
            epoch_correct_labels += correct_labels.data.item()

        logger.info(
            "\tEpoch: {}\tLoss: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}".format(
                epoch + 1,
                epoch_loss / len(data_loader),
                epoch_celabelsloss / len(data_loader),
                epoch_kldloss / len(data_loader),
                epoch_correct_labels / (len(data_loader) * 256),
                data_loader.batch_size,
            ),
        )

        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE
        Input: data_loader: As generated by train_vae
        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()
        new_data_loader = _DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=data_loader.pin_memory,
            collate_fn=data_loader.collate_fn,
        )

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

                latent[row : row + len(mu)] = mu
                row += len(mu)

        assert row == length
        mask_lower_bits(latent, 12)
        return latent

    def trainmodel(
        self,
        dataloader,
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: list[int] = [25, 75, 150, 300],
        modelfile=None,
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
        optimizer = _Adam(self.parameters(), lr=lrate)

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


class VAEConcat(_encode.VAE):
    """Variational autoencoder that uses TNFs, abundances and labels as concatented input, subclass of VAE.
    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nlabels: Number of labels
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
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
    ):
        super(VAEConcat, self).__init__(
            nsamples + nlabels,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.nsamples = nsamples
        self.nlabels = nlabels

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths, tnf and abundance signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        abundance_out = reconstruction.narrow(1, self.nsamples + self.ntnf, 1)
        labels_out = reconstruction.narrow(
            1, self.nsamples + self.ntnf + 1, self.nlabels
        )

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)
            # labels_out = _softmax(labels_out, dim=1)

        return depths_out, tnf_out, abundance_out, labels_out

    def forward(self, depths, tnf, abundance, labels):
        tensor = _torch.cat((depths, tnf, abundance, labels), 1)
        mu = self._encode(tensor)
        logsigma = _torch.zeros(mu.size())
        if self.usecuda:
            logsigma = logsigma.cuda()
        latent = self.reparameterize(mu)
        depths_out, tnf_out, abundance_out, labels_out = self._decode(latent)

        return depths_out, tnf_out, abundance_out, labels_out, mu, logsigma

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

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
        ce_labels_weight = 1.0  # TODO: figure out

        weighed_ab = ab_sse * ab_sse_weight
        weighed_ce = ce * ce_weight
        weighed_sse = sse * sse_weight
        weighed_kld = kld * kld_weight
        weighed_labels = ce_labels * ce_labels_weight

        reconstruction_loss = weighed_ce + weighed_ab + weighed_sse + weighed_labels
        loss = (reconstruction_loss + weighed_kld) * weights

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return (
            loss,
            ce,
            sse,
            ce_labels,
            kld,
            _torch.sum(labels_out_indices == labels_in_indices),
        )

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0
        epoch_celabelsloss = 0
        epoch_correct_labels = 0

        if epoch in batchsteps:
            new_batch_size = data_loader.batch_size * 2  # type:ignore
            data_loader = _DataLoader(
                dataset=data_loader.dataset,
                batch_size=new_batch_size,  # type:ignore
                shuffle=True,
                drop_last=data_loader.dataset.tensors[0].shape[0] > new_batch_size,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
            )

        for depths_in, tnf_in, abundance_in, weights, labels_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            abundance_in.requires_grad = True
            labels_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                abundance_in = abundance_in.cuda()
                weights = weights.cuda()
                labels_in = labels_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, abundance_out, labels_out, mu, logsigma = self(
                depths_in, tnf_in, abundance_in, labels_in
            )

            loss, ce, sse, ce_labels, kld, correct_labels = self.calc_loss(
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
            )

            loss.mean().backward()
            optimizer.step()

            epoch_loss += loss.mean().data.item()
            epoch_kldloss += kld.mean().data.item()
            epoch_sseloss += sse.mean().data.item()
            epoch_celoss += ce.mean().data.item()
            epoch_celabelsloss += ce_labels.mean().data.item()
            epoch_correct_labels += correct_labels.data.item()

        logger.info(
            "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tCE_labels: {:.7f}\tKLD: {:.4f}\taccuracy: {:.4f}\tBatchsize: {}".format(
                epoch + 1,
                epoch_loss / len(data_loader),
                epoch_celoss / len(data_loader),
                epoch_sseloss / len(data_loader),
                epoch_celabelsloss / len(data_loader),
                epoch_kldloss / len(data_loader),
                epoch_correct_labels / len(data_loader),
                data_loader.batch_size,
            ),
        )
        self.eval()
        return data_loader

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE
        Input: data_loader: As generated by train_vae
        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()
        new_data_loader = _DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=data_loader.pin_memory,
            collate_fn=data_loader.collate_fn,
        )

        depths_array, _, _, _, _ = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, ab, weights, labels in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    ab = ab.cuda()
                    labels = labels.cuda()

                # Evaluate
                _, _, _, _, mu, _ = self(depths, tnf, ab, labels)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row : row + len(mu)] = mu
                row += len(mu)

        assert row == length
        mask_lower_bits(latent, 12)
        return latent


class VAEVAE(object):
    """Bi-modal variational autoencoder that uses TNFs, abundances and one-hot labels.
    It contains three individual encoders (VAMB, one-hot labels and concatenated VAMB+labels)
    and two decoders (VAMB and labels).
    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nlabels: Number of labels
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
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
    ):
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
        self.VAELabels = VAELabels(
            N_l,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )
        self.VAEJoint = VAEConcat(
            nsamples,
            N_l,
            nhiddens=nhiddens,
            nlatent=nlatent,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            cuda=cuda,
        )

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
            ce_weight = ((1 - self.alpha) * (self.nsamples - 1)) / (
                self.nsamples * _log(self.nsamples)
            )

        ab_sse_weight = (1 - self.alpha) * (1 / self.nsamples)
        sse_weight = self.alpha / self.ntnf

        _, labels_in_indices = labels_in.max(dim=1)
        ce_labels = _nn.CrossEntropyLoss()(labels_out, labels_in_indices)
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
        kld_weight = 1 / (self.nlatent * self.beta)
        kld_loss = kld * kld_weight

        loss = (reconstruction_loss + kld_loss) * weights

        _, labels_out_indices = labels_out.max(dim=1)
        _, labels_in_indices = labels_in.max(dim=1)
        return (
            loss.mean(),
            ce.mean(),
            sse.mean(),
            ce_labels,
            kld_vamb.mean(),
            kld_labels.mean(),
            _torch.sum(labels_out_indices == labels_in_indices),
        )

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps):
        metrics = [
            "loss_vamb",
            "ab_vamb",
            "ce_vamb",
            "sse_vamb",
            "kld_vamb",
            "loss_labels",
            "ce_labels_labels",
            "kld_labels",
            "correct_labels_labels",
            "loss_joint",
            "ce_joint",
            "sse_joint",
            "ce_labels_joint",
            "kld_vamb_joint",
            "kld_labels_joint",
            "correct_labels_joint",
            "loss",
        ]
        metrics_dict: dict[str, _torch.Tensor] = {k: 0 for k in metrics}
        tensors_dict: dict[str, _torch.Tensor] = {k: None for k in metrics}

        if epoch in batchsteps:
            new_batch_size = data_loader.batch_size * 2
            data_loader = _DataLoader(
                dataset=data_loader.dataset,
                batch_size=new_batch_size,
                shuffle=True,
                drop_last=data_loader.dataset.tensors[0].shape[0] > new_batch_size,
                num_workers=data_loader.num_workers,
                pin_memory=data_loader.pin_memory,
                collate_fn=data_loader.collate_fn,
            )

        for (
            depths_in_sup,
            tnf_in_sup,
            abundance_in_sup,
            weights_in_sup,
            labels_in_sup,
            depths_in_unsup,
            tnf_in_unsup,
            abundance_in_unsup,
            weights_in_unsup,
            labels_in_unsup,
        ) in data_loader:
            depths_in_sup.requires_grad = True
            tnf_in_sup.requires_grad = True
            abundance_in_sup.requires_grad = True
            labels_in_sup.requires_grad = True
            depths_in_unsup.requires_grad = True
            tnf_in_unsup.requires_grad = True
            abundance_in_unsup.requires_grad = True
            labels_in_unsup.requires_grad = True

            if self.VAEVamb.usecuda:
                depths_in_sup = depths_in_sup.cuda()
                tnf_in_sup = tnf_in_sup.cuda()
                abundance_in_sup = abundance_in_sup.cuda()
                weights_in_sup = weights_in_sup.cuda()
                labels_in_sup = labels_in_sup.cuda()
                depths_in_unsup = depths_in_unsup.cuda()
                tnf_in_unsup = tnf_in_unsup.cuda()
                abundance_in_unsup = abundance_in_unsup.cuda()
                weights_in_unsup = weights_in_unsup.cuda()
                labels_in_unsup = labels_in_unsup.cuda()

            optimizer.zero_grad()

            _, _, _, _, mu_sup, logsigma_sup = self.VAEJoint(
                depths_in_sup, tnf_in_sup, abundance_in_sup, labels_in_sup
            )  # use the two-modality latent space

            depths_out_sup, tnf_out_sup, abundance_out_sup = self.VAEVamb._decode(
                self.VAEVamb.reparameterize(mu_sup)
            )  # use the one-modality decoders
            labels_out_sup = self.VAELabels._decode(
                self.VAELabels.reparameterize(mu_sup)
            )  # use the one-modality decoders

            (
                depths_out_unsup,
                tnf_out_unsup,
                abundance_out_unsup,
                mu_vamb_unsup,
            ) = self.VAEVamb(depths_in_unsup, tnf_in_unsup, abundance_in_unsup)
            logsigma_vamb_unsup = _torch.zeros(mu_vamb_unsup.size())
            if self.usecuda:
                logsigma_vamb_unsup = logsigma_vamb_unsup.cuda()
            _, _, _, mu_vamb_sup_s = self.VAEVamb(
                depths_in_sup, tnf_in_sup, abundance_in_sup
            )
            logsigma_vamb_sup_s = _torch.zeros(mu_vamb_sup_s.size())
            if self.usecuda:
                logsigma_vamb_sup_s = logsigma_vamb_sup_s.cuda()
            labels_out_unsup, mu_labels_unsup, logsigma_labels_unsup = self.VAELabels(
                labels_in_unsup
            )
            _, mu_labels_sup_s, logsigma_labels_sup_s = self.VAELabels(labels_in_sup)

            (
                tensors_dict["loss_vamb"],
                tensors_dict["ab_vamb"],
                tensors_dict["ce_vamb"],
                tensors_dict["sse_vamb"],
                tensors_dict["kld_vamb"],
            ) = self.VAEVamb.calc_loss(
                depths_in_unsup,
                depths_out_unsup,
                tnf_in_unsup,
                tnf_out_unsup,
                abundance_in_unsup,
                abundance_out_unsup,
                mu_vamb_unsup,
                weights_in_unsup,
            )

            (
                tensors_dict["loss_labels"],
                tensors_dict["ce_labels_labels"],
                tensors_dict["kld_labels"],
                tensors_dict["correct_labels_labels"],
            ) = self.VAELabels.calc_loss(
                labels_in_unsup,
                labels_out_unsup,
                mu_labels_unsup,
                logsigma_labels_unsup,
            )

            losses_joint = self.calc_loss_joint(
                depths_in_sup,
                depths_out_sup,
                tnf_in_sup,
                tnf_out_sup,
                abundance_in_sup,
                abundance_out_sup,
                labels_in_sup,
                labels_out_sup,
                mu_sup,
                logsigma_sup,
                mu_vamb_sup_s,
                logsigma_vamb_sup_s,
                mu_labels_sup_s,
                logsigma_labels_sup_s,
                weights_in_sup,
            )

            (
                tensors_dict["loss_joint"],
                tensors_dict["ce_joint"],
                tensors_dict["sse_joint"],
                tensors_dict["ce_labels_joint"],
                tensors_dict["kld_vamb_joint"],
                tensors_dict["kld_labels_joint"],
                tensors_dict["correct_labels_joint"],
            ) = losses_joint

            tensors_dict["loss"] = (
                tensors_dict["loss_joint"]
                + tensors_dict["loss_vamb"]
                + tensors_dict["loss_labels"]
            )

            tensors_dict["loss"].backward()
            optimizer.step()

            for k, v in tensors_dict.items():
                metrics_dict[k] += v.data.item()

        metrics_dict["correct_labels_joint"] /= data_loader.batch_size
        metrics_dict["correct_labels_labels"] /= data_loader.batch_size
        logger.info(
            f"\t\tEpoch: {epoch}  "
            + "  ".join(
                [k + f": {v / len(data_loader):.5e}" for k, v in metrics_dict.items()]
            )
        )

        return data_loader

    def trainmodel(
        self,
        dataloader,
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: list[int] = [25, 75, 150, 300],
        modelfile=None,
    ):
        """Train the autoencoder from depths array, tnf array and one-hot labels array.
        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            modelfile: Save models to this file if not None [None]
        Output: None
        """

        if lrate < 0:
            raise ValueError("Learning rate must be positive, not {}".format(lrate))

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {}".format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
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
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(
            list(self.VAEVamb.parameters())
            + list(self.VAELabels.parameters())
            + list(self.VAEJoint.parameters()),
            lr=lrate,
        )

        logger.info("\tNetwork properties:")
        logger.info(f"\t    CUDA: {self.VAEVamb.usecuda}")
        logger.info(f"\t    Alpha: {self.VAEVamb.alpha}")
        logger.info(f"\t    Beta: {self.VAEVamb.beta}")
        logger.info(f"\t    Dropout: {self.VAEVamb.dropout}")
        logger.info(f"\t    N hidden: {', '.join(map(str, self.VAEVamb.nhiddens))}")
        logger.info(f"\t    N latent: {self.VAEVamb.nlatent}")
        logger.info("\tTraining properties:")
        logger.info(f"\t    N epochs: {nepochs}")
        logger.info(f"\t    Starting batch size: {dataloader.batch_size}")
        batchsteps_string = (
            ", ".join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
        )
        logger.info(f"\t    Batchsteps: {batchsteps_string}")
        logger.info(f"\t    Learning rate: {lrate}")
        logger.info(f"\t    N sequences: {ncontigs}")
        logger.info(f"\t    N samples: {nsamples}")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set)

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
        state = {
            "nsamples": self.VAEVamb.nsamples,
            "nlabels": self.VAELabels.nlabels,
            "alpha": self.VAEVamb.alpha,
            "beta": self.VAEVamb.beta,
            "dropout": self.VAEVamb.dropout,
            "nhiddens": self.VAEVamb.nhiddens,
            "nlatent": self.VAEVamb.nlatent,
            "state_VAEVamb": self.VAEVamb.state_dict(),
            "state_VAELabels": self.VAELabels.state_dict(),
            "state_VAEJoint": self.VAEJoint.state_dict(),
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

        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
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
