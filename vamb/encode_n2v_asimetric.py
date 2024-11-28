from typing import Optional, IO, Union
from pathlib import Path
import vamb
import vamb.vambtools as _vambtools
#from vamb.cluster import _normalize, _calc_distances
from torch.utils.data.dataset import TensorDataset as _TensorDataset
from torch.utils.data import DataLoader as _DataLoader
from torch.nn.functional import softmax as _softmax
from torch.optim import Adam as _Adam
from torch import Tensor
from torch import nn as _nn
from math import log as _log
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from loguru import logger

import sys


__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = make_dataloader(depths, tnf, lengths)
>>> vae.trainmodel(dataloader)
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 32)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
import torch as _torch


def set_batchsize(
    data_loader: _DataLoader, batch_size: int, encode=False
) -> _DataLoader:
    """Effectively copy the data loader, but with a different batch size.

    The `encode` option is used to copy the dataloader to use before encoding.
    This will not drop the last minibatch whose size may not be the requested
    batch size, and will also not shuffle the data.
    """
    return _DataLoader(
        dataset=data_loader.dataset,
        batch_size=batch_size,
        shuffle=not encode,
        drop_last=not encode,
        num_workers=1 if encode else data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )


def make_dataloader_n2v(
    rpkm: _np.ndarray,
    tnf: _np.ndarray,
    neighs_mask: _np.ndarray,
    neighs: _np.ndarray,
    lengths: _np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
) -> tuple[_DataLoader[tuple[Tensor, Tensor, Tensor, Tensor,Tensor,Tensor]], _np.ndarray]:
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        neighs_mask: boolean vector of len == N_contigs, where True if contig has neighs withing distance on embedding space and if contig had neighbours in the assembly-alignment graph.
        neighs: list of lists of len == N_contigs where each element indicates the neighbouring contig indices
        lenghts: contig lenghts
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if (
        not isinstance(rpkm, _np.ndarray)
        or not isinstance(tnf, _np.ndarray)
        or not isinstance(neighs_mask, _np.ndarray)
        or not isinstance(neighs, _np.ndarray)
    ):
        raise ValueError(
            "TNF, RPKM, neighs, and neighs mask must be Numpy arrays"
        )

    if batchsize < 1:
        raise ValueError(f"Batch size must be minimum 1, not {batchsize}")

    if (
        len(rpkm) != len(tnf)
        or len(tnf) != len(lengths)
        or len(neighs_mask) != len(lengths)
        or len(neighs) != len(lengths)
    ):
        raise ValueError(
            "Lengths of TNF, RPKM, neighs, neighs mask, and lengths arrays must be the same"
        )

    if not (
        rpkm.dtype
        == tnf.dtype
        == _np.float32
    ):
        raise ValueError(
            "TNF, and RPKM  must be Numpy arrays of dtype float32"
        )

    ### Copy arrays and mask them ###
    # Copy if not destroy - this way we can have all following operations in-place
    # for simplicity
    if not destroy:
        rpkm = rpkm.copy()
        tnf = tnf.copy()
        neighs_mask = neighs_mask.copy()
        neighs = neighs.copy()

    # Normalize samples to have same depth
    sample_depths_sum = rpkm.sum(axis=0)
    if _np.any(sample_depths_sum == 0):
        raise ValueError(
            "One or more samples have zero depth in all sequences, so cannot be depth normalized"
        )
    rpkm *= 1_000_000 / sample_depths_sum

    zero_tnf = tnf.sum(axis=1) == 0
    smallest_index = _np.argmax(zero_tnf)
    if zero_tnf[smallest_index]:
        raise ValueError(
            f"TNF row at index {smallest_index} is all zeros. "
            + "This implies that the sequence contained no 4-mers of A, C, G, T or U, "
            + "making this sequence uninformative. This is probably a mistake. "
            + "Verify that the sequence contains usable information (e.g. is not all N's)"
        )

    total_abundance = rpkm.sum(axis=1)


    ## for the contigs that are present only in one sample, define the object of abundances 
    # across samples (n_contigs x n_samples) z-score normalizing per sample only for contigs that satisfy (BOTH):
    # a) abundance only in one sample
    # b) abundance in the specific sample

    rpkm_unz=_np.zeros_like(rpkm)    

    
    rpkm_unz_mask = _np.zeros(rpkm.shape[0],dtype=bool)
    if rpkm.shape[1] <= 10:
        singlesample_contig_idxs = _np.where(_np.count_nonzero(rpkm, axis=1) == 1)[0]    
        rpkm_unz_mask[singlesample_contig_idxs]=True


        for i in range(rpkm.shape[1]):
            mask_nonzeroab_sample_i = rpkm[:,i] > 0.0 
            mask_single_ab_and_nonzeroab_si = rpkm_unz_mask & mask_nonzeroab_sample_i    
            if _np.sum((mask_single_ab_and_nonzeroab_si)) == 0:
                continue
            
            
            mean_ab_s_i = _np.mean(rpkm[mask_single_ab_and_nonzeroab_si,i])
            std_ab_s_i = _np.std(rpkm[mask_single_ab_and_nonzeroab_si,i])
            rpkm_unz[mask_single_ab_and_nonzeroab_si,i] = (rpkm[mask_single_ab_and_nonzeroab_si,i]-mean_ab_s_i)/std_ab_s_i
            
            #log_vals = _np.log(rpkm[mask_single_ab_and_nonzeroab_si,i])    
            #rpkm_unz[mask_single_ab_and_nonzeroab_si,i] = (log_vals-_np.mean(log_vals))/_np.std(log_vals)

        
        #rpkm_unz=_np.log(rpkm_unz.clip(min=0.001))
        #logger.info("Adding max(log(rpkm_unz),0.001) as input")
        logger.info("Adding rpkm z-score vector as input for contigs only present in one sample %i/%i" %(_np.sum(rpkm_unz_mask),len(rpkm_unz_mask)))
    

    else:
        singlesample_contig_idxs = []
            
    # Normalize rpkm to sum to 1
    n_samples = rpkm.shape[1]
    zero_total_abundance = total_abundance == 0
    rpkm[zero_total_abundance] = 1 / n_samples
    nonzero_total_abundance = total_abundance.copy()
    nonzero_total_abundance[zero_total_abundance] = 1.0
    rpkm /= nonzero_total_abundance.reshape((-1, 1))
    
    ## for the contigs that are only present in one sample, remove the sample correltaion signal ###### NOVELTY#########
    singlesample_contig_idxs = _np.where(_np.count_nonzero(rpkm, axis=1) == 1)[0]
    rpkm[singlesample_contig_idxs,:]=0
    logger.info("Removing rpkm signal for contigs present only in one sample")


    # Normalize TNF and total abundance to make SSE loss work better
    total_abundance = _np.log(total_abundance.clip(min=0.001))
    _vambtools.zscore(total_abundance, inplace=True)
    _vambtools.zscore(tnf, axis=0, inplace=True)
    total_abundance.shape = (len(total_abundance), 1)

    # Create weights
    lengths = (lengths).astype(_np.float32)
    weights = _np.log(lengths).astype(_np.float32) - 5.0
    weights[weights < 2.0] = 2.0
    weights *= len(weights) / weights.sum()
    weights.shape = (len(weights), 1)
    indices = _np.arange(tnf.shape[0])

    ### Create final tensors and dataloader ###
    depthstensor = _torch.from_numpy(rpkm)  # this is a no-copy operation
    depthstensor_unz = _torch.from_numpy(rpkm_unz)  # this is a no-copy operation
    depthstensor_unz_mask = _torch.from_numpy(rpkm_unz_mask)  # this is a no-copy operation
    tnftensor = _torch.from_numpy(tnf)
    total_abundance_tensor = _torch.from_numpy(total_abundance)
    weightstensor = _torch.from_numpy(weights)
    neighs_masktensor = _torch.from_numpy(neighs_mask)
    indicestensor = _torch.from_numpy(indices)

    n_workers = 4 if cuda else 1
    dataset = _TensorDataset(
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        neighs_masktensor,
        weightstensor,
        indicestensor,
        depthstensor_unz,
        depthstensor_unz_mask,
    )

    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=False,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return dataloader, neighs


class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        ncontigs: Number of contigs present in the catalog
        neighs_object: Array of list containing the indexes of the neighbouring contigs
        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        gamma: Weight for the contrastive loss term [1.0]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
        seed: Seed set from where to sample random numbers

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        ncontigs: int,
        neighs_object: _np.ndarray,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        gamma: float = 0.1,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
        seed: int = 0,
        margin: float = 0.01,
    ):
        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

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

        _torch.manual_seed(seed)
        self.rng = _torch.Generator()
        self.rng.manual_seed(seed)
        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ncontigs = ncontigs

        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.margin = margin
        


        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip(
            [self.nsamples + self.ntnf + 1 + self.nsamples] + self.nhiddens,
            self.nhiddens,
        ):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(
            self.nhiddens[0], self.nsamples + self.ntnf + 1 + self.nsamples  # + self.n_embedding
        )

        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        # Object where I will store the mu s , so we update after training
        
        self.mu_container = _torch.randn(
            self.ncontigs, self.nlatent, dtype=_torch.float32
        )
        
        # Container where I know the neighbours of each contig in embedding space
        self.neighs = neighs_object
        if self.usecuda:
            self.mu_container = self.mu_container.cuda()

    def _encode(self, tensor: Tensor) -> Tensor:
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)

        # Note: We ought to also compute logsigma here, but we had a bug in the original
        # implementation of Vamb where logsigma was fixed to zero, so we just remove it.

        return mu

    # sample with gaussian noise
    def reparameterize(self, mu: Tensor) -> Tensor:
        epsilon = _torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        latent = mu + epsilon

        return latent

    def _decode(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        abundance_out = reconstruction.narrow(1, self.nsamples + self.ntnf, 1)
        abundance_long_out = reconstruction.narrow(1, self.nsamples + self.ntnf + 1, self.nsamples)
        
        # emb_out = reconstruction.narrow(1, self.ntnf + self.nsamples, self.n_embedding)
        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)
        
        return depths_out, tnf_out, abundance_out ,abundance_long_out

    def forward(
        self,
        depths: Tensor,
        tnf: Tensor,
        abundance: Tensor,
        abundance_long: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensor = _torch.cat((depths, tnf, abundance,abundance_long), 1)
        mu = self._encode(tensor)
        latent = self.reparameterize(mu)
        depths_out, tnf_out, abundance_out,abundance_long_out = self._decode(latent)

        return depths_out, tnf_out, abundance_out,abundance_long_out, mu

    def calc_loss(
        self,
        depths_in: Tensor,
        depths_out: Tensor,
        tnf_in: Tensor,
        tnf_out: Tensor,
        abundance_in: Tensor,
        abundance_out: Tensor,
        mu: Tensor,
        weights: Tensor,
        preds_idxs: Tensor,
        neighs_mask: Tensor,
        abundance_long_in: Tensor,
        abundance_long_out: Tensor,
        abundance_long_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor,Tensor]:
        
        # If multiple samples, use cross entropy, else use SSE for abundance
        ab_sse = (abundance_out - abundance_in).pow(2).sum(dim=1)
        ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)
        sse_ab_long = (abundance_long_out - abundance_long_in).pow(2).sum(dim=1)*abundance_long_mask 
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
        ab_long_sse_weight = self.alpha / (self.nsamples)
        kld_weight = 1 / (self.nlatent * self.beta)
        
        weighed_ab = ab_sse * ab_sse_weight
        weighed_ce = ce * ce_weight
        weighed_sse = sse * sse_weight
        weighed_ab_long_sse = sse_ab_long * ab_long_sse_weight
        weighed_kld = kld * kld_weight
        loss_emb = self.cosinesimilarity_loss(
            mu,
            preds_idxs,
            neighs_mask,
        )

        loss_emb_cat = _torch.cat(
            (
                loss_emb.unsqueeze(0) - self.margin,
                _torch.zeros_like(loss_emb).unsqueeze(0),
            ),
            dim=0,
        )

        reconstruction_loss = (
            weighed_ce
            + weighed_ab
            + weighed_sse
            + weighed_ab_long_sse
        )
        loss = (reconstruction_loss + weighed_kld) * weights + _torch.max(
            loss_emb_cat, dim=0
        )[0] * self.gamma

        loss_emb_pop = loss_emb[neighs_mask]
        
        return (
            loss.mean(),
            weighed_ab.mean(),
            weighed_ce.mean(),
            weighed_sse.mean(),
            (weighed_ab_long_sse[abundance_long_mask]).mean(),
            weighed_kld.mean(),
            loss_emb_pop.mean(), 

        )

    def cosinesimilarity_loss(self, mu, idxs_preds, neighs_mask):
        avg_cosine_distances = _torch.empty(len(mu),device=mu.device)
        for i,(mu_i, idx_pred, neighs_mask_i) in enumerate(zip(mu, idxs_preds, neighs_mask)):
            if not neighs_mask_i:
            
                # If it has no neighbors or neighs_mask_i is False
                cosine_distances = _torch.tensor(0.0)  # A single value of 1.0
                avg_cosine_distance = cosine_distances

            else:
                # Select the neighbors
                mus_neighs = self.mu_container[self.neighs[idx_pred]].to(mu.device)
                
                # Compute distances
                cosine_distances = 2 * self.calc_cosine_distances(mus_neighs, mu_i )
                avg_cosine_distance = cosine_distances.mean()

            # Append to the result lists
            avg_cosine_distances[i] = avg_cosine_distance

        return avg_cosine_distances

    def calc_cosine_distances(self,matrix, vector):
        "Return vector of cosine distances from rows of normalized matrix to given row."
        # normalize vector 
        vector = vector/(vector.norm() * (2**0.5))
        
        dists = 0.5 - matrix.matmul(vector)
        
        return dists

    def trainepoch(
        self,
        data_loader: _DataLoader,
        epoch: int,
        optimizer,
        batchsteps: list[int],
    ) -> _DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        self.train()

        epoch_loss = 0.0
        epoch_kldloss = 0.0
        epoch_sseloss = 0.0
        epoch_celoss = 0.0
        epoch_embloss_pop = 0.0
        epoch_absseloss = 0.0
        epoch_ablongsseloss = 0.0

        if epoch in batchsteps:
            data_loader = set_batchsize(data_loader, data_loader.batch_size * 2)

        for (
            depths_in,
            tnf_in,
            abundance_in,
            neighs_mask,
            weights,
            preds_idxs,
            abundance_long_in,
            abundance_long_mask,
        ) in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True
            
            preds_idxs = preds_idxs.long()
            neighs_mask = neighs_mask.bool()
            abundance_long_mask = abundance_long_mask.bool()
            
            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()
                abundance_in = abundance_in.cuda()
                weights = weights.cuda()
                neighs_mask = neighs_mask.cuda()
                preds_idxs = preds_idxs.cuda()
                abundance_long_in = abundance_long_in.cuda()
                abundance_long_mask = abundance_long_mask.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, abundance_out,abundance_long_out, mu = self(
                depths_in, tnf_in, abundance_in,abundance_long_in
            )
            
            self.mu_container[preds_idxs] = vamb.cluster._normalize(mu.detach())
            (
                loss,
                ab_sse,
                ce,
                sse,
                ab_long_sse,
                kld,
                loss_emb_pop,
            ) = self.calc_loss(
                depths_in,
                depths_out,
                tnf_in,
                tnf_out,
                abundance_in,
                abundance_out,
                mu,
                weights,
                preds_idxs,
                neighs_mask,
                abundance_long_in,
                abundance_long_out,
                abundance_long_mask,
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()
            epoch_embloss_pop += loss_emb_pop.item() 
            epoch_absseloss += ab_sse.data.item()
            epoch_ablongsseloss += ab_long_sse.data.item()

        logger.info(
            "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.6f}\tAB:{:.4e}\tABlong:{:.4e}\tSSE: {:.6f}\tembloss_pop: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                epoch + 1,
                epoch_loss / len(data_loader),
                epoch_celoss / len(data_loader),
                epoch_absseloss / len(data_loader),
                epoch_ablongsseloss / len(data_loader),
                epoch_sseloss / len(data_loader),
                epoch_embloss_pop / len(data_loader),
                epoch_kldloss / len(data_loader),
                data_loader.batch_size,
            )
        )
        
        self.eval()
        return data_loader

    def encode(self, data_loader) -> _np.ndarray:
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = set_batchsize(
            data_loader, data_loader.batch_size, encode=True
        )

        depths_array, _, _, _, _, _,_,_ = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, ab, _, _, _,ab_long,_ in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()
                    ab = ab.cuda()
                    ab_long = ab_long.cuda()

                # Evaluate'
                _, _, _,_,mu = self(depths, tnf, ab,ab_long)
                

                if self.usecuda:
                    mu = mu.cpu()

                latent[row : row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
            "nsamples": self.nsamples,
            "ncontigs": self.ncontigs,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "dropout": self.dropout,
            "nhiddens": self.nhiddens,
            "nlatent": self.nlatent,
            "margin": self.margin,
            "state": self.state_dict(),
        }

        _torch.save(state, filehandle)


    @classmethod
    def load(
        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
    ):
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

        nsamples = dictionary["nsamples"]
        alpha = dictionary["alpha"]
        beta = dictionary["beta"]
        dropout = dictionary["dropout"]
        nhiddens = dictionary["nhiddens"]
        nlatent = dictionary["nlatent"]
        state = dictionary["state"]

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(
        self,
        dataloader: _DataLoader[
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ],
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
            logfile: Print status updates to this file if not None [None]
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
            last_batchsize = dataloader.batch_size * 2 ** len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:  # type: ignore
                raise ValueError(
                    f"Last batch size of {last_batchsize} exceeds dataset length "
                    f"of {len(dataloader.dataset)}. "  # type: ignore
                    "This means you have too few contigs left after filtering to train. "
                    "It is not adviced to run Vamb with fewer than 10,000 sequences "
                    "after filtering. "
                    "Please check the Vamb log file to see where the sequences were "
                    "filtered away, and verify BAM files has sensible content."
                )
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape  # type: ignore
        optimizer = _Adam(self.parameters(), lr=lrate)

        logger.info(f"\tNetwork properties:")
        logger.info(f"\tCUDA: {self.usecuda}")
        logger.info(f"\tAlpha: {self.alpha}")
        logger.info(f"\tBeta: {self.beta}")
        logger.info(f"\tGamma: {self.gamma}")
        logger.info(f"\tMargin: {self.margin}")
        logger.info(f"\tDropout: {self.dropout}")
        logger.info(f"\tN hidden: {', '.join(map(str, self.nhiddens))}")
        logger.info(f"\tN latent: {self.nlatent}\n")
        logger.info(f"\tTraining properties:")
        logger.info(f"\tN epochs: {nepochs}")
        logger.info(f"\tStarting batch size: {dataloader.batch_size}")
        batchsteps_string = (
            ", ".join(map(str, sorted(batchsteps_set))) if batchsteps_set else "None"
        )
        logger.info(f"\tBatchsteps: {batchsteps_string}")
        logger.info(f"\tLearning rate: {lrate}")
        logger.info(f"\tN sequences: {ncontigs}")
        logger.info(f"\tN samples: {nsamples}\n\n")
        

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader,
                epoch,
                optimizer,
                sorted(batchsteps_set),
            )
        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
