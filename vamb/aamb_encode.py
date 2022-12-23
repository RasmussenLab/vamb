"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE"""


import numpy as np
from math import log
import time
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader as _DataLoader

from typing import Optional

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

############################################################################# MODEL ###########################################################
class AAE(nn.Module):
    def __init__(
        self,
        nsamples: int,
        nhiddens: int,
        nlatent_l: int,
        nlatent_y,
        sl: float,
        slr: float,
        alpha: Optional[float],
        _cuda: bool,
    ):
        if nsamples is None:
            raise ValueError(
                f"Number of samples  should be provided to define the encoder input layer as well as the categorical latent dimension, not {nsamples}"
            )

        super(AAE, self).__init__()
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        self.nsamples = nsamples
        input_len = 103 + self.nsamples
        self.h_n = nhiddens
        self.ld = nlatent_l
        self.y_len = nlatent_y
        self.input_len = int(input_len)
        self.sl = sl
        self.slr = slr
        self.alpha = alpha
        self.usecuda = _cuda

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
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), self.ld))))

        if self.usecuda:
            sampled_z = sampled_z.cuda()
        z = sampled_z * std + mu

        return z

    ## Encoder
    def _encode(self, depths, tnfs):
        _input = torch.cat((depths, tnfs), 1)
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

    ## Decoder
    def _decode(self, z, y):
        z_y = torch.cat((z, y), 1)

        reconstruction = self.decoder(z_y)

        _depths_out, tnf_out = (
            reconstruction[:, : self.nsamples],
            reconstruction[:, self.nsamples :],
        )

        depths_out = F.softmax(_depths_out, dim=1)

        return depths_out, tnf_out

    ## Discriminator Z space (continuous latent space defined by mu and sigma layers)
    def _discriminator_z(self, z):
        return self.discriminator_z(z)

    ## Discriminator Y space (categorical latent space defined by Y layer)

    def _discriminator_y(self, y):
        return self.discriminator_y(y)

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        sse_weight = self.alpha / (tnf_in.shape[1] * 2)
        loss = ce * ce_weight + sse * sse_weight
        return loss, ce, sse

    def forward(self, depths_in, tnfs_in, z_prior, y_prior):
        mu, logvar, y_latent = self._encode(depths_in, tnfs_in)
        z_latent = self._reparameterization(mu, logvar)
        depths_out, tnfs_out = self._decode(z_latent, y_latent)
        d_z_latent = self._discriminator_z(z_latent)
        d_y_latent = self._discriminator_y(y_latent)

        return mu, depths_out, tnfs_out, z_latent, y_latent, d_z_latent, d_y_latent

    # ----------
    #  Training
    # ----------

    def trainmodel(
        self, data_loader, nepochs, batchsteps, T, lr, logfile=None, modelfile=None
    ):

        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        batchsteps_set = set(batchsteps)
        ncontigs, _ = data_loader.dataset.tensors[0].shape

        # Initialize generator and discriminator

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tY length:", self.y_len, file=logfile)
            print("\tZ length:", self.ld, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            print("\tStarting batch size:", data_loader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        # we need to separate the paramters due to the adversarial training

        disc_z_params = []
        disc_y_params = []
        enc_params = []
        dec_params = []
        for name, param in self.named_parameters():
            if "discriminator_z" in name:
                disc_z_params.append(param)
            elif "discriminator_y" in name:
                disc_y_params.append(param)
            elif "encoder" in name:
                enc_params.append(param)
            else:
                dec_params.append(param)

        # Define adversarial loss for the discriminators
        adversarial_loss = torch.nn.BCELoss()
        if self.usecuda:
            adversarial_loss.cuda()

        #### Optimizers
        optimizer_E = torch.optim.Adam(enc_params, lr=lr)
        optimizer_D = torch.optim.Adam(dec_params, lr=lr)

        optimizer_D_z = torch.optim.Adam(disc_z_params, lr=lr)
        optimizer_D_y = torch.optim.Adam(disc_y_params, lr=lr)

        for epoch_i in range(nepochs):
            if epoch_i in batchsteps:
                data_loader = _DataLoader(
                    dataset=data_loader.dataset,
                    batch_size=data_loader.batch_size * 2,
                    shuffle=True,
                    drop_last=True,
                    num_workers=data_loader.num_workers,
                    pin_memory=data_loader.pin_memory,
                )

            (
                ED_loss_e,
                D_z_loss_e,
                D_y_loss_e,
                V_loss_e,
                CE_e,
                SSE_e,
            ) = (0, 0, 0, 0, 0, 0)

            total_batches_inthis_epoch = len(data_loader)
            time_epoch_0 = time.time()

            for depths_in, tnfs_in, _ in data_loader:  # weights currently unused here
                nrows, _ = depths_in.shape

                # Adversarial ground truths

                labels_prior = Variable(
                    Tensor(nrows, 1).fill_(1.0), requires_grad=False
                )
                labels_latent = Variable(
                    Tensor(nrows, 1).fill_(0.0), requires_grad=False
                )

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
                    z_prior = Variable(
                        Tensor(np.random.normal(0, 1, (nrows, self.ld)))
                    )
                    ohc = RelaxedOneHotCategorical(
                        T, torch.ones([nrows, self.y_len])
                    )
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
                    z_latent,
                    y_latent,
                    d_z_latent,
                    d_y_latent,
                ) = self(depths_in, tnfs_in, z_prior, y_prior)

                vae_loss, ce, sse = self.calc_loss(
                    depths_in, depths_out, tnfs_in, tnfs_out
                )
                g_loss_adv_z = adversarial_loss(
                    self._discriminator_z(z_latent), labels_prior
                )
                g_loss_adv_y = adversarial_loss(
                    self._discriminator_y(y_latent), labels_prior
                )

                ed_loss = (
                    (1 - self.sl) * vae_loss
                    + (self.sl * self.slr) * g_loss_adv_z
                    + (self.sl * (1 - self.slr)) * g_loss_adv_y
                )

                ed_loss.backward()
                optimizer_E.step()
                optimizer_D.step()

                # ----------------------
                #  Train Discriminator z
                # ----------------------

                optimizer_D_z.zero_grad()
                mu, logvar = self._encode(depths_in, tnfs_in)[:2]
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
                #  Train Discriminator y
                # ----------------------

                optimizer_D_y.zero_grad()
                y_latent = self._encode(depths_in, tnfs_in)[2]
                d_y_loss_prior = adversarial_loss(
                    self._discriminator_y(y_prior), labels_prior
                )
                d_y_loss_latent = adversarial_loss(
                    self._discriminator_y(y_latent), labels_latent
                )
                d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

                d_y_loss.backward()
                optimizer_D_y.step()

                ED_loss_e += float(ed_loss.item())
                V_loss_e += float(vae_loss.item())
                D_z_loss_e += float(d_z_loss.item())
                D_y_loss_e += float(d_y_loss.item())
                CE_e += float(ce.item())
                SSE_e += float(sse.item())

            time_epoch_1 = time.time()
            time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)

            if logfile is not None:
                print(
                    "\tEpoch: {}\t Loss Enc/Dec: {:.6f}\t Rec. loss: {:.4f}\t CE: {:.4f}\tSSE: {:.4f}\t Dz loss: {:.7f}\t Dy loss: {:.6f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                        epoch_i + 1,
                        ED_loss_e / total_batches_inthis_epoch,
                        V_loss_e / total_batches_inthis_epoch,
                        CE_e / total_batches_inthis_epoch,
                        SSE_e / total_batches_inthis_epoch,
                        D_z_loss_e / total_batches_inthis_epoch,
                        D_y_loss_e / total_batches_inthis_epoch,
                        data_loader.batch_size,
                        time_e,
                    ),
                    file=logfile,
                )
                logfile.flush()

            # save model
            if modelfile is not None:
                try:
                    checkpoint = {
                        "state": self.state_dict(),
                        "optimizer_ED": optimizer_ED.state_dict(),
                        "optimizer_D_z": optimizer_D_z.state_dict(),
                        "optimizer_D_y": optimizer_D_y.state_dict(),
                        "nsamples": self.num_samples,
                        "alpha": self.alpha,
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
    def get_latents(self, contignames, data_loader, last_epoch=True):
        """Retrieve the categorical latent representation (y) and the contiouous latents (l) of the inputs

        Inputs:
            dataloader
            contignames
            last_epoch


        Output:
            y_clusters_dict ({clust_id : [contigs]})
            l_latents array"""
        self.eval()

        new_data_loader = _DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=data_loader.pin_memory,
        )

        depths_array, _, _ = data_loader.dataset.tensors

        length = len(depths_array)
        latent = np.empty((length, self.ld), dtype=np.float32)
        index_contigname = 0
        row = 0
        clust_y_dict = dict()
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        with torch.no_grad():

            for depths_in, tnfs_in, _ in new_data_loader:
                nrows, _ = depths_in.shape
                I = torch.cat((depths_in, tnfs_in), dim=1)

                if self.usecuda:
                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
                    z_prior.cuda()
                    ohc = RelaxedOneHotCategorical(
                        torch.tensor([0.15], device="cuda"),
                        torch.ones([nrows, self.y_len], device="cuda"),
                    )
                    y_prior = ohc.sample()
                    y_prior = y_prior.cuda()

                else:
                    z_prior = Variable(
                        Tensor(np.random.normal(0, 1, (nrows, self.ld)))
                    )
                    ohc = RelaxedOneHotCategorical(
                        0.15, torch.ones([nrows, self.y_len])
                    )
                    y_prior = ohc.sample()

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()

                if last_epoch:
                    mu, _, _, _, y_sample = self(depths_in, tnfs_in, z_prior, y_prior)[
                        0:5
                    ]
                else:
                    y_sample = self(depths_in, tnfs_in, z_prior, y_prior)[4]

                if self.usecuda:
                    Ys = y_sample.cpu().detach().numpy()
                    if last_epoch:
                        mu = mu.cpu().detach().numpy()
                        latent[row : row + len(mu)] = mu
                        row += len(mu)
                else:
                    Ys = y_sample.detach().numpy()
                    if last_epoch:
                        mu = mu.detach().numpy()
                        latent[row : row + len(mu)] = mu
                        row += len(mu)
                del y_sample

                for _y in Ys:
                    contig_name = contignames[index_contigname]
                    contig_cluster = np.argmax(_y) + 1
                    contig_cluster_str = str(contig_cluster)

                    if contig_cluster_str not in clust_y_dict:
                        clust_y_dict[contig_cluster_str] = set()

                    clust_y_dict[contig_cluster_str].add(contig_name)

                    index_contigname += 1
                del Ys

            if last_epoch:
                return clust_y_dict, latent
            else:
                return clust_y_dict
