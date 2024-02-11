"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE"""


import numpy as np
from math import log, isfinite
import time
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset as _TensorDataset

from torch.utils.data import DataLoader as _DataLoader

from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import torch

from vamb.encode import set_batchsize
import vamb.vambtools as _vambtools

from collections.abc import Sequence
from typing import Optional, IO, Union
from numpy.typing import NDArray
from loguru import logger
from math import log as _log
import random





def make_dataloader_n2v(
    rpkm: np.ndarray,
    tnf: np.ndarray,
    embeddings: np.ndarray,
    embeddings_mask: np.ndarray,
    neighs: np.ndarray,
    lengths: np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
) -> tuple[_DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]], np.ndarray]:
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        embeddings: embeddings matrix (N_contigs x embedding_size)
        embeddings_mask: boolean vector of len == N_contigs, where True if
        contig has neighs withing distance on embedding space and if contig had neighbours in the graph.
        neighs: list of lists of len == N_contigs where each element indicates the neighbouring contig indices.
        lenghts: contig lenghts
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if (
        not isinstance(rpkm, np.ndarray)
        or not isinstance(tnf, np.ndarray)
        or not isinstance(embeddings, np.ndarray)
        or not isinstance(embeddings_mask, np.ndarray)
        # or not isinstance(neighs, np.ndarray)
    ):
        raise ValueError(
            "TNF, RPKM, Embedding, Embedding mask, and neighs must be Numpy arrays"
        )

    if batchsize < 1:
        raise ValueError(f"Batch size must be minimum 1, not {batchsize}")

    if (
        len(rpkm) != len(tnf)
        or len(tnf) != len(lengths)
        or len(embeddings) != len(lengths)
        or len(embeddings_mask) != len(lengths)
        or len(neighs) != len(lengths)
    ):
        raise ValueError(
            "Lengths of TNF, RPKM, Embedding, Embedding mask,neighs, and lengths arrays must be the same"
        )

    if not (
        rpkm.dtype
        == tnf.dtype
        == embeddings.dtype
        == embeddings_mask.dtype
        == np.float32
    ):
        raise ValueError(
            "TNF, RPKM, Embedding, and Embedding mask must be Numpy arrays of dtype float32"
        )

    ### Copy arrays and mask them ###
    # Copy if not destroy - this way we can have all following operations in-place
    # for simplicity
    if not destroy:
        rpkm = rpkm.copy()
        tnf = tnf.copy()
        embeddings = embeddings.copy()
        embeddings_mask = embeddings_mask.copy()
        # neighs = neighs.copy()

    # Normalize samples to have same depth
    sample_depths_sum = rpkm.sum(axis=0)
    if np.any(sample_depths_sum == 0):
        raise ValueError(
            "One or more samples have zero depth in all sequences, so cannot be depth normalized"
        )
    rpkm *= 1_000_000 / sample_depths_sum

    zero_tnf = tnf.sum(axis=1) == 0
    smallest_index = np.argmax(zero_tnf)
    if zero_tnf[smallest_index]:
        raise ValueError(
            f"TNF row at index {smallest_index} is all zeros. "
            + "This implies that the sequence contained no 4-mers of A, C, G, T or U, "
            + "making this sequence uninformative. This is probably a mistake. "
            + "Verify that the sequence contains usable information (e.g. is not all N's)"
        )

    total_abundance = rpkm.sum(axis=1)

    # Normalize rpkm to sum to 1
    n_samples = rpkm.shape[1]
    zero_total_abundance = total_abundance == 0
    rpkm[zero_total_abundance] = 1 / n_samples
    nonzero_total_abundance = total_abundance.copy()
    nonzero_total_abundance[zero_total_abundance] = 1.0
    rpkm /= nonzero_total_abundance.reshape((-1, 1))

    # Normalize TNF and total abundance to make SSE loss work better
    total_abundance = np.log(total_abundance.clip(min=0.001))
    _vambtools.zscore(total_abundance, inplace=True)
    _vambtools.zscore(tnf, axis=0, inplace=True)
    total_abundance.shape = (len(total_abundance), 1)

    # Create weights
    lengths = (lengths).astype(np.float32)
    weights = np.log(lengths).astype(np.float32) - 5.0
    weights[weights < 2.0] = 2.0
    weights *= len(weights) / weights.sum()
    weights.shape = (len(weights), 1)
    indices = np.arange(tnf.shape[0])

    ### Create final tensors and dataloader ###
    depthstensor = torch.from_numpy(rpkm)  # this is a no-copy operation
    tnftensor = torch.from_numpy(tnf)
    total_abundance_tensor = torch.from_numpy(total_abundance)
    weightstensor = torch.from_numpy(weights)
    embeddingstensor = torch.from_numpy(embeddings)
    embeddings_masktensor = torch.from_numpy(embeddings_mask)
    indicestensor = torch.from_numpy(indices)

    n_workers = 4 if cuda else 1
    dataset = _TensorDataset(
        depthstensor,
        tnftensor,
        total_abundance_tensor,
        embeddingstensor,
        embeddings_masktensor,
        weightstensor,
        indicestensor,
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




class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, pred, target, emb_mask):
        # Calculating the cosine similarity between the predicted and target vectors
        # print(pred.shape, target.shape)
        cos_sim = self.cosine_similarity(pred, target)
        # print(cos_sim.shape)

        # Since we want to minimize the loss, we'll subtract the cosine similarity from 1
        # Another option is to return the negative of cosine similarity to make it a minimization problem
        loss_ = 1 - cos_sim  # .mean()
        # print(loss.unsqueeze(1))
        # print(emb_mask)
        loss_unpop = loss_ * (1 - emb_mask)
        loss = loss_ * emb_mask

        # print(loss.shape)
        # print(loss.unsqueeze(1).shape)
        # print("\n")
        return loss.unsqueeze(1), loss_unpop.unsqueeze(1)

class OneHotTransform(nn.Module):
    def __init__(self, input_size):
        super(OneHotTransform, self).__init__()
        self.input_size = input_size
        self.output_size = self.input_size

    def forward(self, x):
        _, max_indices = torch.max(x, dim=1)
        one_hot = torch.zeros(x.size(0), self.output_size, device=x.device)
        one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
        return one_hot



def random_one_hot_vector(size):
    index = random.randint(0, size - 1)
    vector = torch.zeros(size)
    vector[index] = 1
    return vector

def generate_random_one_hot_vectors_tensor(num_vectors, size):
    return torch.stack([random_one_hot_vector(size) for _ in range(num_vectors)])




############################################################################# MODEL ###########################################################
class AAE_ASY(nn.Module):
    def __init__(
        self,
        nsamples: int,
        ncontigs:int,
        n_embeddings:int,
        neighs_object: np.ndarray,
        neighbourhoods_object: dict[int, set[int]],
        nhiddens: int,
        nlatent_l: int,
        nlatent_y: int,
        sl: float,
        slr: float,
        alpha: Optional[float],
        gamma:float,
        delta:float,
        _cuda: bool,
        seed: int,
        embeds_loss: str = "cos",

    ):
        for variable, name in [
            (nsamples, "nsamples"),
            (n_embeddings, "n_embeddings"),
            (nhiddens, "nhiddens"),
            (nlatent_l, "nlatents_l"),
            (nlatent_y, "nlatents_y"),
        ]:
            if variable < 1:
                raise ValueError(f"{name} must be at least 1, not {variable}")

        real_variables = [(sl, "sl"), (slr, "slr")]
        if alpha is not None:
            real_variables.append((alpha, "alpha"))
        for variable, name in real_variables:
            if not isfinite(variable) or not (0.0 <= variable <= 1.0):
                raise ValueError(
                    f"{name} must be in the interval [0.0, 1.0], not {variable}"
                )

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        super(AAE_ASY, self).__init__()
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        self.ntnf = 103
        self.nsamples = nsamples
        self.ncontigs = ncontigs
        self.n_embeddings= n_embeddings
        self.embeds_loss = embeds_loss
        input_len = self.ntnf + self.nsamples + self.n_embeddings +1 
        self.h_n = nhiddens
        self.ld = nlatent_l
        self.y_len = nlatent_y
        self.input_len = int(input_len)
        self.sl = sl
        self.slr = slr
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.usecuda = _cuda
        self.rng = np.random.Generator(np.random.PCG64(seed))


        # Container where I know the neighbours of each contig in embedding space
        self.neighs = neighs_object
        
        # dictionary i:{idx_a,idx_c,} , where the set of indices indicates contains all contigs that belong to the same neighbourhoods
        # ensuring the keys are sequential and wo gaps , i.e. 0,1,2,3,4,..., so last key == len(keys)-1
        self.neighbourhoods = { i:idxs for i,idxs in enumerate(neighbourhoods_object.values()) if len(idxs) > 100}
        self.neighbourhoods = { i:idxs for i,idxs in enumerate(self.neighbourhoods.items())}
        self.n_hoods = len(self.neighbourhoods.keys())
        print("# hoods when min len hoods is 100", self.n_hoods)
        assert (self.n_hoods -1)  == np.max(list(self.neighbourhoods.keys()))
        
        # get the hood if I give you the c_idx
        self.idx_hood_d = { idx:i for i,idxs in self.neighbourhoods.items() for idx in idxs }
        # get the hood in onehot if I give you the c_idx
        #self.idx_OHhood_d = { idx: torch.eye(self.n_hoods)[i] for i,idxs in self.neighbourhoods.items() for idx in idxs }



        # Object where I will store the y s , so we update after training
        self.y_container = generate_random_one_hot_vectors_tensor(self.ncontigs, self.y_len)


        self.cosine_loss = CosineSimilarityLoss()


        self.one_hot_transform = OneHotTransform(self.y_len)
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
        )
        # latent layers
        self.mu = nn.Linear(self.h_n, self.ld)
        self.logvar = nn.Linear(self.h_n, self.ld)
        self.y_vector = nn.Linear(self.h_n, self.y_len)
        

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(int(self.ld + self.y_len), self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.input_len),
        )

        # discriminator z
        self.discriminator_z = nn.Sequential(
            nn.Linear(self.ld, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), 1),
            nn.Sigmoid(),
        )

        # discriminator_z hood, can you guess which neighbourhood it belongs to?
        self.discriminator_hood = nn.Sequential(
            nn.Linear(self.ld, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), self.n_hoods),
            nn.Softmax(dim=1),
        )

        # # discriminator_z Y, can you guess which Y_class it belongs to?
        # self.discriminator_z_y = nn.Sequential(
        #     nn.Linear(self.ld, self.h_n),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.h_n, int(self.h_n / 2)),
        #     nn.LeakyReLU(),
        #     nn.Linear(int(self.h_n / 2), self.y_len),
        #     nn.Softmax(),
        # )


        # discriminator Y
        self.discriminator_y = nn.Sequential(
            nn.Linear(self.y_len, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), 1),
            nn.Sigmoid(),
        )

        if _cuda:
            self.cuda()

    ## Reparametrisation trick
    def _reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor

        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(self.rng.normal(0, 1, (mu.size(0), self.ld))))

        if self.usecuda:
            sampled_z = sampled_z.cuda()
        z = sampled_z * std + mu

        return z

    ## Encoder
    def _encode(self, depths, tnfs,embs,ab):
        _input = torch.cat((depths, tnfs,embs,ab), 1)
        #print(_input.shape)
        x = self.encoder(_input)
        mu = self.mu(x)
        logvar = self.logvar(x)
        _y = self.y_vector(x)
        y = F.softmax(_y, dim=1)

        return mu, logvar, y

    def y_length(self):
        return self.y_len

    def z_length(self):
        return self.ld

    def samples_num(self):
        return self.nsamples

    # masking
    def _y_argmax(self,y):
        _, indices = torch.topk(y, 1, dim=1)
        mask = torch.zeros_like(y)
        mask.scatter_(1, indices, 1)
        return mask

    ## Decoder
    def _decode(self, z, y):
        z_y = torch.cat((z, y), 1)

        reconstruction = self.decoder(z_y)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        emb_out = reconstruction.narrow(1, self.ntnf + self.nsamples, self.n_embeddings)
        abundance_out = reconstruction.narrow(
            1, self.nsamples + self.ntnf + self.n_embeddings, 1
        )
        # If multiple samples, apply softmax
        if self.nsamples > 1:            
            depths_out = F.softmax(depths_out, dim=1)
        
        return depths_out, tnf_out, emb_out, abundance_out


    ## Discriminator Z space (continuous latent space defined by mu and sigma layers)
    def _discriminator_z(self, z):
        return self.discriminator_z(z)

    ## Discriminator Z space for hoods (continuous latent space defined by mu and sigma layers)
    def _discriminator_hood(self, z):
        return self.discriminator_hood(z)


    ## Discriminator Y space (categorical latent space defined by Y layer)

    def _discriminator_y(self, y):
        return self.discriminator_y(y)

    def calc_loss(
        self, 
        depths_in, 
        depths_out, 
        tnf_in, 
        tnf_out,
        emb_in,
        emb_out,
        emb_mask,
        abundance_in,
        abundance_out,
        weights,
        ys,
        idxs_preds,
        ):

        ab_sse = (abundance_out - abundance_in).pow(2).sum(dim=1)
        # Add 1e-9 to depths_out to avoid numerical instability.
        ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)

        # Avoid having the denominator be zero
        if self.nsamples == 1:
            ce_weight = 0.0
        else:
            ce_weight = ((1 - self.alpha) * (self.nsamples - 1)) / (
                self.nsamples * _log(self.nsamples)
            )

        contrastive_loss_y = self.y_contrastive_loss(ys, idxs_preds, emb_mask)

        if self.embeds_loss == "cos":
            loss_emb, loss_emb_unpop = self.cosine_loss(emb_out, emb_in, emb_mask)
        elif self.embeds_loss == "sse":
            loss_emb = (emb_out - emb_in).pow(2).sum(dim=1) * emb_mask
            loss_emb_unpop = (emb_out - emb_in).pow(2).sum(dim=1) * (1 - emb_mask)

        else:
            raise ValueError("Embedding loss not defined, (either cos or sse)")


        ab_sse_weight = (1 - self.alpha) * (1 / self.nsamples)
        sse_weight = self.alpha / self.ntnf
        weighed_ab = ab_sse * ab_sse_weight
        weighed_ce = ce * ce_weight
        weighed_sse = sse * sse_weight
        weighed_emb = loss_emb * self.gamma
        weighed_contrastive = contrastive_loss_y * self.delta


        reconstruction_loss = weighed_ce + weighed_ab + weighed_sse + weighed_emb + weighed_contrastive
        
        loss = reconstruction_loss * weights
        
        
        #return loss, ce, sse
        return (
            loss.mean(),
            weighed_ab.mean(),
            weighed_ce.mean(),
            weighed_sse.mean(),
            contrastive_loss_y.mean(),
            (loss_emb.sum(dim=0) / torch.sum(emb_mask)) * self.gamma,
            (loss_emb_unpop.sum(dim=0) / torch.sum((1 - emb_mask))) * self.gamma,
        )

    def cat_CE_similarity(self,y_i,ys_neighs,exhaustive=False):
        #print(one_hot_vectors.shape)
        
        total_loss = 0.0
        
        if exhaustive:
            
            y_and_ys_neighs = torch.cat((y_i.unsqueeze(0),ys_neighs),0)
            num_samples = len(y_and_ys_neighs)
            
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    loss = F.binary_cross_entropy(y_and_ys_neighs[i], y_and_ys_neighs[j])
                    total_loss += loss / len(y_and_ys_neighs[i])  # Normalize the loss

        else:
            for i in range(len(ys_neighs)):
                loss = F.binary_cross_entropy(y_i, ys_neighs[i])
                total_loss += loss 

        return total_loss/len(ys_neighs)


    def y_contrastive_loss(self, y, idxs_preds, emb_mask):
        
        cat_CEs = []

        for y_i, idx_pred, emb_mask_i in zip(y, idxs_preds, emb_mask):
            if len(self.neighs[idx_pred]) == 0 or not emb_mask_i:
                # If it has no neighbors or emb_mask_i is False
                cat_CE = torch.tensor(0.0)  # A single value of 1.0

            else:
                # Select the neighbors
                
                ys_neighs = self.y_container[self.neighs[idx_pred]]
                
                #print(y_and_ys_neighs.shape,y_i.shape,ys_neighs.shape)
                # Compute cat_CE distances
                cat_CE = self.cat_CE_similarity(
                    y_i,
                    ys_neighs
                    )
                

            # Append to the result lists
            cat_CEs.append(cat_CE)
            

        return torch.stack(cat_CEs)  


    def forward(self, depths_in, tnfs_in,emb_in, abundance_in,warmup=False):
        mu, logvar, y_latent = self._encode(depths_in, tnfs_in,emb_in,abundance_in)
        z_latent = self._reparameterization(mu, logvar)
        
        #print(torch.argmax(y_latent,dim=1),y_latent)
        
        y_latent_one_hot = self.one_hot_transform(y_latent)
        #print(torch.argmax(y_latent_one_hot,dim=1),y_latent_one_hot)
        #y_latent_one_hot = y_latent # self._y_argmax(y_latent)
        if warmup == True:
            depths_out, tnfs_out, emb_out, abundance_out = self._decode(torch.zeros_like(z_latent), y_latent_one_hot)
        else:
            depths_out, tnfs_out, emb_out, abundance_out = self._decode(z_latent, y_latent_one_hot)
        
        #d_z_latent = self._discriminator_z(z_latent)
        #d_y_latent = self._discriminator_y(y_latent)

        return mu, depths_out, tnfs_out,emb_out,abundance_out, z_latent, y_latent,y_latent_one_hot#, d_z_latent, d_y_latent

    def Y_neighs_accuracy(self):
        accuracies = 0
        contigs_with_neighs = 0
        labels_used = set()
        for neigh_idxs in self.neighs:
            if len(neigh_idxs) == 0:
                continue
            labels = []
            for neigh_idx in neigh_idxs:
                labels.append(torch.argmax(self.y_container[neigh_idx]).item())
                
            unique_labels = set(labels)
            labels_used = unique_labels.union(labels_used)
            max_count = 0
            for label in unique_labels:
                count = labels.count(label)
                if count > max_count:
                    max_count = count
        
            accuracies +=  max_count / len(labels)
            contigs_with_neighs += 1 
        
        return accuracies/contigs_with_neighs,len(labels_used)

    def Y_neighbourhoods_accuracy(self):
        accuracies = 0
            
        for hood_idxs in self.neighbourhoods.values():
            labels = []
            for hood_idx in hood_idxs:
                labels.append(torch.argmax(self.y_container[hood_idx]).item())
                
            unique_labels = set(labels)
            max_count = 0
            for label in unique_labels:
                count = labels.count(label)
                if count > max_count:
                    max_count = count
        
            accuracies +=  max_count / len(labels)
            
        
        return accuracies/len(self.neighbourhoods.keys())

    # ----------
    #  Training
    # ----------

    def trainmodel(
        self,
        data_loader,
        nepochs: int,
        batchsteps: list[int],
        T,
        lr: float,
        modelfile: Union[None, str, IO[bytes]] = None,
    ):
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        batchsteps_set = set(batchsteps)
        ncontigs, _ = data_loader.dataset.tensors[0].shape

        # Initialize generator and discriminator
        logger.info("\tNetwork properties:")
        logger.info(f"\tCUDA: {self.usecuda}")
        logger.info(f"\tAlpha: {self.alpha}")
        logger.info(f"\tGamma: {self.gamma}")
        logger.info(f"\tDelta: {self.delta}")
        logger.info(f"\tY length: {self.y_len}")
        logger.info(f"\tZ length: {self.ld}")
        logger.info("\n\tTraining properties:")
        logger.info(f"\tN epochs: {nepochs}")
        logger.info(f"\tStarting batch size: {data_loader.batch_size}")
        batchsteps_string = (
            ", ".join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
        )
        logger.info(f"\tBatchsteps: {batchsteps_string}")
        logger.info(f"\tN sequences: {ncontigs}")
        logger.info(f"\tN samples: {self.nsamples}")
        logger.info(f"\tEmbedding size:{ self.n_embeddings}")

        # we need to separate the paramters due to the adversarial training

        disc_z_params = []
        disc_hood_params = []
        disc_y_params = []
        enc_params = []
        dec_params = []
        for name, param in self.named_parameters():
            if "hood" in name:
                disc_hood_params.append(param)
            elif "discriminator_z" in name :
                disc_z_params.append(param)

            elif "discriminator_y" in name:
                disc_y_params.append(param)
            elif "encoder" in name:
                enc_params.append(param)
            else:
                dec_params.append(param)
        
        #print(len(disc_hood_params),len(disc_z_params),len(enc_params))
        initial_params_hood = {name: param.clone().detach() for name, param in self.named_parameters() if "hood" in name}
        initial_params_z = {name: param.clone().detach() for name, param in self.named_parameters() if "hood" not in name if "discriminator_z" in name} 
        initial_params_encoder = {name: param.clone().detach() for name, param in self.named_parameters() if "encoder" in name} 
        #print(initial_params_hood.keys(),initial_params_z.keys(),initial_params_encoder.keys())
        
        # Define adversarial loss for the discriminators
        adversarial_loss = torch.nn.BCELoss()
        adversarial_hoods_loss = torch.nn.CrossEntropyLoss()
        if self.usecuda:
            adversarial_loss.cuda()
            adversarial_hoods_loss.cuda()

        #### Optimizers
        optimizer_E = torch.optim.Adam(enc_params, lr=lr)
        optimizer_D = torch.optim.Adam(dec_params, lr=lr)

        optimizer_D_z = torch.optim.Adam(disc_z_params, lr=lr)
        optimizer_D_z_hood = torch.optim.Adam(disc_hood_params, lr=lr)
        optimizer_D_y = torch.optim.Adam(disc_y_params, lr=lr)

        for epoch_i in range(nepochs):
            if epoch_i in batchsteps:
                data_loader = set_batchsize(
                    data_loader, data_loader.batch_size * 2  # type:ignore
                )

            (
                ED_loss_e,
                D_z_loss_e,
                D_z_hood_loss_e,
                D_y_loss_e,
                V_loss_e,
                CE_e,
                SSE_e,
            ) = (0, 0, 0, 0, 0, 0,0)
            (epoch_loss, epoch_rec_and_contr_loss, epoch_absseloss, epoch_celoss, epoch_sseloss, epoch_contrastive_loss, epoch_embloss_pop, epoch_d_z_loss,epoch_d_z_hood_loss,epoch_d_z_adv_loss,epoch_d_z_adv_hood_loss, epoch_d_y_loss) = (0, 0, 0, 0, 0, 0, 0,0, 0,0,0,0)
            
            # everything used 
            
            for depths_in, tnfs_in,abundance_in, emb_in,emb_mask, weighs,idx_preds in data_loader:
                
                nrows, _ = depths_in.shape

                # Adversarial ground truths

                labels_prior = Variable(
                    Tensor(nrows, 1).fill_(1.0), requires_grad=False
                )
                labels_latent = Variable(
                    Tensor(nrows, 1).fill_(0.0), requires_grad=False
                )

                labels_hood = Variable(
                    Tensor(nrows, self.n_hoods).fill_(0.0), requires_grad=False
                )
                for i,idx_pred in enumerate(idx_preds):
                    if emb_mask[i]:
                        labels_hood[i,self.idx_hood_d[idx_pred.item()]] = 1.0
                
                
                
                # Sample noise as discriminator Z,Y ground truth

                if self.usecuda:
                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
                    z_prior.cuda()
                    ohc = RelaxedOneHotCategorical(
                        torch.tensor([T], device="cuda"),
                        torch.ones([nrows, self.y_len], device="cuda"),
                    )
                    y_prior = ohc.sample()
                    y_prior = y_prior.cuda()

                else:
                    z_prior = Variable(Tensor(self.rng.normal(0, 1, (nrows, self.ld))))
                    ohc = RelaxedOneHotCategorical(T, torch.ones([nrows, self.y_len]))
                    y_prior = ohc.sample()

                del ohc

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_E.zero_grad()
                optimizer_D.zero_grad()

                # Forward pass
                (
                    _,
                    depths_out,
                    tnfs_out,
                    emb_out,
                    abundance_out,
                    z_latent,
                    _,
                    y_latent_one_hot,
                    #d_z_latent,
                    #d_y_latent,
                ) = self(depths_in, tnfs_in,emb_in,abundance_in,warmup= True if epoch_i < 0 else False )           

                # Here, if epoch > 1, I need to update the self.mu_obj, maybe even if it is not epoch > 1
                self.y_container[idx_preds] = self._y_argmax(y_latent_one_hot.detach()) 
                
                rec_and_contr_loss,ab_sse, ce, sse,contrastive_loss_y,loss_emb_pop,_ = self.calc_loss(
                    depths_in, depths_out, tnfs_in, tnfs_out,emb_in,emb_out,emb_mask,abundance_in,abundance_out,weighs,y_latent_one_hot,idx_preds
                )

                g_loss_adv_z = adversarial_loss(
                    self._discriminator_z(z_latent), labels_prior
                )
                # g_loss_adv_y = adversarial_loss(
                #     self._discriminator_y(y_latent), labels_prior
                # )

                g_loss_adv_z_hood = adversarial_hoods_loss(
                    self._discriminator_hood(z_latent[emb_mask.bool()]), labels_hood[emb_mask.bool()]
                    
                )


                ed_loss = (
                    (1 - self.sl) * rec_and_contr_loss
                    + (self.sl * self.slr) * g_loss_adv_z
                    #+ (self.sl * (1 - self.slr)) * g_loss_adv_z_hood
                    #+ (self.sl * (1 - self.slr)) * g_loss_adv_y
                )

                ed_loss.backward()
                optimizer_E.step()
                optimizer_D.step()

                # ----------------------
                #  Train Discriminator z
                # ----------------------

                optimizer_D_z.zero_grad()
                mu, logvar = self._encode(depths_in, tnfs_in,emb_in,abundance_in)[:2]
                z_latent = self._reparameterization(mu, logvar)

                d_z_loss_prior = adversarial_loss(
                    self._discriminator_z(z_prior), labels_prior
                )
                d_z_loss_latent = adversarial_loss(
                    self._discriminator_z(z_latent), labels_latent
                )
                d_z_loss = 0.5 * (d_z_loss_prior + d_z_loss_latent)

                d_z_loss.backward()
                optimizer_D_z.step()
                

                # ----------------------
                #  Train Discriminator z_hood
                # ----------------------

                optimizer_D_z_hood.zero_grad()
                mu, logvar = self._encode(depths_in, tnfs_in,emb_in,abundance_in)[:2]
                z_latent = self._reparameterization(mu, logvar)

                d_z_hood_loss = adversarial_hoods_loss(
                   #self._discriminator_hood(z_latent[emb_mask.bool()]), labels_hood[emb_mask.bool()]
                   self._discriminator_hood(z_latent), labels_hood
                )
                
                
                d_z_hood_loss.backward()
                
                optimizer_D_z_hood.step()
                print(d_z_hood_loss.item())
                # parameters_changed = False
                # for name, param in self.named_parameters():
                #     if "hood" not in name:
                #         continue
                #     if not torch.allclose(param, initial_params_hood[name]):
                #         print(name, param[:20],initial_params_hood[name])
                #         parameters_changed = True
                #         break

                # if parameters_changed:
                #     print("Parameters changed during training.")
                # else:
                #     print("Parameters did not change during training.")
                # ----------------------
                #  Train Discriminator y
                # ----------------------

                # optimizer_D_y.zero_grad()
                # y_latent = self._encode(depths_in, tnfs_in,emb_in,abundance_in)[2]
                # d_y_loss_prior = adversarial_loss(
                #     self._discriminator_y(y_prior), labels_prior
                # )
                # d_y_loss_latent = adversarial_loss(
                #     self._discriminator_y(y_latent), labels_latent
                # )
                # d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

                # d_y_loss.backward()
                # optimizer_D_y.step()

                # ED_loss_e += float(ed_loss.item())
                # V_loss_e += float(rec_and_contr_loss.item())
                # D_z_loss_e += float(d_z_loss.item())
                # #D_y_loss_e += float(d_y_loss.item())
                # CE_e += float(ce.item())
                # SSE_e += float(sse.item())

                epoch_loss += ed_loss.data.item()
                
                epoch_rec_and_contr_loss += float(rec_and_contr_loss.item())
                
                epoch_absseloss += ab_sse.data.item()
                epoch_celoss += ce.data.item()
                epoch_sseloss += sse.data.item()
                epoch_contrastive_loss += contrastive_loss_y.item()
                epoch_embloss_pop += loss_emb_pop.item()  # .data.item()
                epoch_d_z_loss += float(d_z_loss.item())
                epoch_d_z_hood_loss += float(d_z_hood_loss.item())
                epoch_d_z_adv_loss += float(g_loss_adv_z.item())
                epoch_d_z_adv_hood_loss += float(g_loss_adv_z_hood.item())
                #epoch_d_y_loss += float(d_y_loss.item())
                

            accuracy_neighs,ys_used_n = self.Y_neighs_accuracy()
            accuracy_hoods = self.Y_neighbourhoods_accuracy()
            
            
            logger.info(
                #"\tEp: {}\tLoss: {:.6f}\tRec: {:.6f}\tCE: {:.7f}\tAB:{:.5e}\tSSE: {:.6f}\tembloss_pop: {:.6f}\ty_contr: {:.6f}\tDz: {:.4f}\tDy: {:.4f}\tBatchsize: {}".format(
                "\tEp: {}\tLoss: {:.2f}\tRec: {:.2f}\tCE: {:.2f}\tAB:{:.2e}\tSSE: {:.2f}\tembloss_pop: {:.3f}\ty_contr: {:.3f}\tDz: {:.2f}\tDz_adv: {:.2f}\tDz_hood: {:.6f}\tDz_hood_adv: {:.6f}\tAcc_y_neighs: {:.2f}\tAcc_y_hoods: {:.2f}\tYs_used: {}\tBs: {}".format(
                    epoch_i + 1,
                    epoch_loss / len(data_loader),
                    epoch_rec_and_contr_loss / len(data_loader),
                    epoch_celoss / len(data_loader),
                    epoch_absseloss / len(data_loader),
                    epoch_sseloss / len(data_loader),
                    epoch_embloss_pop / len(data_loader),
                    epoch_contrastive_loss / len(data_loader),
                    epoch_d_z_loss/ len(data_loader),
                    epoch_d_z_adv_loss/ len(data_loader),
                    epoch_d_z_hood_loss/ len(data_loader),
                    epoch_d_z_adv_hood_loss / len(data_loader),
                    #epoch_d_y_loss/ len(data_loader),
                    accuracy_neighs,                    
                    accuracy_hoods,                    
                    ys_used_n,
                    data_loader.batch_size,
                )
            )

            # save model
            if modelfile is not None:
                try:
                    checkpoint = {
                        "state": self.state_dict(),
                        "optimizer_E": optimizer_E.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                        "optimizer_D_z": optimizer_D_z.state_dict(),
                        "optimizer_D_z_hood": optimizer_D_z_hood.state_dict(),
                        #"optimizer_D_y": optimizer_D_y.state_dict(),
                        "nsamples": self.num_samples,
                        "alpha": self.alpha,
                        "gamma": self.gamma,
                        "delta": self.delta,
                        "nhiddens": self.h_n,
                        "nlatent_l": self.ld,
                        "nlatent_y": self.y_len,
                        "sl": self.sl,
                        "slr": self.slr,
                        "temp": self.T,
                    }
                    torch.save(checkpoint, modelfile)

                except:
                    pass

        return None

    ########### funciton that retrieves the clusters from Y latents
    def get_latents(
        self, contignames: Sequence[str], data_loader
    ) -> tuple[dict[str, set[str]], NDArray[np.float32]]:
        """Retrieve the categorical latent representation (y) and the contiouous latents (l) of the inputs

        Inputs:
            dataloader
            contignames
            last_epoch


        Output:
            y_clusters_dict ({clust_id : [contigs]})
            l_latents array"""
        self.eval()

        new_data_loader = set_batchsize(data_loader, 256, encode=True)
        
        depths_array, _, _, _, _, _,_ = new_data_loader.dataset.tensors
        
        length = len(depths_array)
        latent = np.empty((length, self.ld), dtype=np.float32)
        index_contigname = 0
        row = 0
        clust_y_dict: dict[str, set[str]] = dict()
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        with torch.no_grad():
            #for depths, tnfs, emb, ab, _, _ in new_data_loader:
            for depths, tnfs,ab, emb,_, _,_ in new_data_loader:
                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()

                mu, _, _, _,_,_,_, y = self(depths, tnfs,emb,ab)

                if self.usecuda:
                    Ys = y.cpu().detach().numpy()
                    mu = mu.cpu().detach().numpy()
                    latent[row : row + len(mu)] = mu
                    row += len(mu)
                else:
                    Ys = y.detach().numpy()
                    mu = mu.detach().numpy()
                    latent[row : row + len(mu)] = mu
                    row += len(mu)
                del y

                for _y in Ys:
                    contig_name = contignames[index_contigname]
                    contig_cluster = np.argmax(_y) + 1
                    contig_cluster_str = str(contig_cluster)

                    if contig_cluster_str not in clust_y_dict:
                        clust_y_dict[contig_cluster_str] = set()

                    clust_y_dict[contig_cluster_str].add(contig_name)

                    index_contigname += 1
                del Ys

            return clust_y_dict, latent
