"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE"""


import numpy as np
from math import log
import itertools
import time 
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import torch
from .data_loader_aae import make_dataloader 
import random
from .calc_loss import calc_loss
from datetime import datetime

torch.manual_seed(0)


############################################################################# MODEL ###########################################################
## Encoder
class Encoder(nn.Module):
    def __init__(self, h_n, ld,y_len, nsamples=0, _cuda=False):
        super(Encoder, self).__init__()
        if nsamples == None:
            raise ValueError('Number of samples  should be provided to define the encoder input layer as well as the categorical latent dimension, not {}'.format(nsamples))
        
        self.nsamples= nsamples
        input_len = 103 + self.nsamples
        self.h_n = h_n
        self.ld = ld
        self.y_len = y_len
        self.input_len = int(input_len)
        self._cuda = _cuda 
        
        self.model = nn.Sequential(
            nn.Linear(self.input_len, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
           )
        
        self.mu = nn.Linear(self.h_n, self.ld)
        self.logvar = nn.Linear(self.h_n, self.ld) 
        self.y_vector = nn.Linear(self.h_n,self.y_len)
        
    # Reparametrization trick to enble training sampling from the Z latent space
    def reparameterization(self, mu, logvar):    
        
        Tensor = torch.cuda.FloatTensor if self._cuda is True else torch.FloatTensor
       
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), self.ld))))  
        
        if self._cuda:
            sampled_z = sampled_z.cuda()
        z = sampled_z * std + mu
        
        return z

    def forward(self, depths, tnfs):
        _input = torch.cat((depths,tnfs),1)
        x = self.model(_input)
        mu = self.mu(x)
        
        logvar = self.logvar(x)
        
        z = self.reparameterization(mu, logvar)
        _y = self.y_vector(x)
        
        y = F.softmax(_y, dim=1) 
        
        return z,y,mu
    
    def y_length(self):
        return self.y_len
    def z_length(self):
        return self.ld
    
    def samples_num(self):
        return self.nsamples

## Decoder
class Decoder(nn.Module):
    def __init__(self, h_n, ld, y_len, nsamples=0):
        super(Decoder, self).__init__()
        
        if nsamples == 0:
            raise ValueError('Number of samples should be provided to define the decoder output layer as well as the categorical latent dimension, not {}'.format(nsamples))
        self.nsamples= nsamples
        self.input_len = int(103 + nsamples)
        
        self.h_n = h_n
        self.ld = ld
        self.y_len = y_len
        
        self.model = nn.Sequential(
            nn.Linear(int(self.ld + self.y_len), self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.input_len), 
        )
    def forward(self, z, y):
        z_y = torch.cat((z,y),1)

        reconstruction = self.model(z_y)
        
        _depths_out, tnf_out = (reconstruction[:,:self.nsamples],
                reconstruction[:,self.nsamples:])
        
        depths_out = F.softmax(_depths_out, dim=1)
        
        return reconstruction, depths_out, tnf_out
    

## Discriminator Z space (continuous latent space defined by mu and sigma layers)
class Discriminator_z(nn.Module):
    def __init__(self, h_n , ld ):
        super(Discriminator_z, self).__init__()
        
        self.h_n = h_n
        self.ld = ld
       
        self.model = nn.Sequential(
            nn.Linear(self.ld, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n/2), 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

## Discriminator Z space (continuous latent space defined by mu and sigma layers)

class Discriminator_y(nn.Module):
    def __init__(self, h_n, y_len, nsamples=0):
        super(Discriminator_y, self).__init__()
        if nsamples == None:
            raise ValueError('Number of samples should be provided to define the decoder output layer, not {}'.format(nsamples))
        input_len = 103 + nsamples
        self.h_n = h_n
        self.y_len = y_len
        
        self.model = nn.Sequential(
            nn.Linear(self.y_len, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n/2), 1),
            nn.Sigmoid(),
        )

    def forward(self, y):
        validity = self.model(y)
        return validity

########### funciton that retrieves the clusters from Y latents

def get_latents_y(encoder,contignames ,Dataloader,Cuda,last_epoch=False):
    """ Retrieve the categorical latent representation (y) and the contiouous latents (l) of the inputs 
        
        Inputs: dataloader that provides the batches of tnfs and depths
        Output: y_clusters, y_clusters_dict ({clust_id : [contigs]}) , z_latents array and mask  """
        
    dataloader,mask = Dataloader
    index_i=0
    time0=time.time()
    latent_matrix=np.zeros((1,283),dtype='float32')
    clust_y_dict = dict()
    
    for i, (depths_in, tnfs_in) in enumerate(dataloader):
        if Cuda == True:
            depths_in = depths_in.cuda()
            tnfs_in = tnfs_in.cuda()
        encoder.eval()
        
        with torch.no_grad():
            if last_epoch is True:
                _, y_sample, mu = encoder(depths_in,tnfs_in)
           
            else:
                y_sample = encoder(depths_in,tnfs_in)[1]
         
        if Cuda:
            Ys = y_sample.cpu().detach().numpy()
            if last_epoch is True:
                latent_matrix = np.append(latent_matrix,mu.cpu().detach().numpy(),0)
        else:
            Ys = y_sample.detach().numpy()
            if last_epoch is True:
                latent_matrix = np.append(latent_matrix,mu.detach().numpy(),0)
        del y_sample
        
        for t in Ys:
            contig_name = contignames[index_i]
            contig_cluster = np.argmax(t)+1
            #contig_cluster_str = 'cluster_'+str(contig_cluster)
            contig_cluster_str = str(contig_cluster)
           
            if contig_cluster_str not in clust_y_dict.keys():
                clust_y_dict[contig_cluster_str] = set()

            #else:
            clust_y_dict[contig_cluster_str].add(contig_name)
            
            index_i += 1
        del Ys
        encoder.train()
    #print(clust_y_dict.items())
    if last_epoch is True:
        latent_array=latent_matrix[1:,:]
        return clust_y_dict ,latent_array, mask 
    
    time1=time.time()
    time_Y=np.round(time1-time0)
    
# ----------
#  Training
# ----------

def train_model( encoder, decoder, discr_z, discr_y, depths, tnfs, contignames, batchsize, n_epochs, batchsteps, _cuda, s_l = 0.00964 , s_l_r = 0.5, T = 0.15, save_checkpoint = 25, modelfile=None ,alpha=0.15, logfile=None):

    Tensor = torch.cuda.FloatTensor if _cuda else torch.FloatTensor    
    ld=encoder.z_length()
    y_len=encoder.y_length()
    num_samples=encoder.samples_num()
    ncontigs = tnfs.shape[0]
    batchsteps_set = set(batchsteps)

    ##### ADVERSARIAL LOSS (reconstruction loss loaded from calc_loss.py)
    # Use binary cross-entropy loss for D_Z and D_Y
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    
    if logfile is not None:
        print('\tNetwork properties:', file=logfile)
        print('\tCUDA:', _cuda, file=logfile)
        print('\tAlpha:', alpha, file=logfile)
        print('\tY length:', y_len, file=logfile)
        print('\tl length:', ld, file=logfile)
        print('\n\tTraining properties:', file=logfile)
        print('\tN epochs:', n_epochs, file=logfile)
        print('\tStarting batch size:', batchsize, file=logfile)
        batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
        print('\tBatchsteps:', batchsteps_string, file=logfile)
        print('\tN sequences:', ncontigs, file=logfile)
        print('\tN samples:', num_samples, file=logfile, end='\n\n')

        

    ## Use cuda if available
    if _cuda:
        print(type(encoder.cuda))
        encoder.cuda()
        decoder.cuda()
        discr_z.cuda()
        discr_y.cuda()
        adversarial_loss.cuda()

    #### Optimizers
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=1e-3 )
    optimizer_D = torch.optim.Adam(decoder.parameters(), lr=1e-3 )
    optimizer_D_z = torch.optim.Adam(discr_z.parameters(), lr=1e-3 )
    optimizer_D_y = torch.optim.Adam(discr_y.parameters(), lr=1e-3 )

    
    G_loss_epochs=[]
    D_loss_z_epochs=[]
    D_loss_y_epochs=[]
    V_loss_epochs=[]
    G_loss_adv_Z_epochs=[]
    G_loss_adv_Y_epochs=[]

    #print('Training model\n')
    
    dataloader_e= make_dataloader(depths,tnfs,batchsize,Shuffle=True,Cuda=_cuda)
    
    #epochs_num=len(dataloader_e[0])
    
    for epoch_i in range(n_epochs):
        if epoch_i in batchsteps:
            batchsize *=2
            dataloader_e= make_dataloader(depths,tnfs,batchsize,Shuffle=True,Cuda=_cuda)

            #epochs_num=len(dataloader_e[0])

        time_e0=time.time()
        
        G_loss_e, D_z_loss_e, D_y_loss_e, V_loss_e, G_loss_adv_Z_e, G_loss_adv_Y_e,CE_e, SSE_e, NC_S_list_e = 0, 0, 0, 0, 0, 0, 0, 0, []
       
        total_batches_inthis_epoch = len(dataloader_e[0])
        time_epoch_0=time.time()

        for i, (depths_in, tnfs_in) in enumerate(dataloader_e[0]): 
            
            time_0=time.time()
            I=torch.cat((depths_in,tnfs_in),dim=1)
            
            # Adversarial ground truths
            
            valid = Variable(Tensor(I.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(I.shape[0], 1).fill_(0.0), requires_grad=False)
            
            # Sample noise as discriminator Z ground truth
            
            if _cuda:
                z_true = torch.cuda.FloatTensor(I.shape[0], ld).normal_()
            
            else:
                z_true = Variable(Tensor(np.random.normal(0, 1, (I.shape[0], ld))))
            
            if _cuda:
                z_true.cuda()
            
            # Sample noise as discriminator ground truth
           
            if _cuda:
                ohc = RelaxedOneHotCategorical(torch.tensor([T],device='cuda'),torch.ones([I.shape[0],y_len],device='cuda'))
            else:
                ohc = RelaxedOneHotCategorical(T,torch.ones([I.shape[0],y_len]))
            
            y_true = ohc.sample()
            
            if _cuda:
                y_true = y_true.cuda()
            
            time_1=time.time()
            
            del ohc
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_E.zero_grad()
            optimizer_D.zero_grad()
            
            # get the output of the discriminators

            if _cuda == True:
                depths_in=depths_in.cuda()
                tnfs_in=tnfs_in.cuda()
            
            # get encodings, i.e. latents

            z,y,mu = encoder(depths_in,tnfs_in)
            # get reconstruciton

            decoded_z,depths_out,tnfs_out= decoder(z,y)
            
            # get discriminators ouptup given the above encodings

            d_z_fake = discr_z(z)
            d_y_fake = discr_y(y)

            
            # Loss measures generator's ability to reconstruct the input        
            vae_loss,ce,sse = calc_loss(depths_in, depths_out, tnfs_in, tnfs_out,num_samples)
            
            # second g_loss part measures generator ability to fool the discriminator
            
            g_loss_adv_z = adversarial_loss(d_z_fake, valid)
            g_loss_adv_z = adversarial_loss(discr_z(z), valid) # already done 3 lines above!! 
            g_loss_adv_y = adversarial_loss(discr_y(y), valid) 

            g_loss =  (1-s_l) * vae_loss + (s_l*s_l_r) * g_loss_adv_z + (s_l*(1-s_l_r))* adversarial_loss(d_y_fake, valid) # possible bug, might affect the gradient computation!! use the g_loss_adv_y instead of adverarial_loss again! 

            g_loss.backward()
            optimizer_E.step() # comment one of these and check the reconstruction loss, if it increasese it is learning 
            optimizer_D.step()
            time_2=time.time()
            
            # ----------------------
            #  Train Discriminator z
            # ----------------------
            
            optimizer_D_z.zero_grad()
            z_fake = encoder(depths_in,tnfs_in)[0]#  I already have this one above!

                
            # Measure discriminator's ability to classify real from generated samples
            # merge inputs and labels so I only call once and don't compute by 0.5 
            real_z_loss = adversarial_loss(discr_z(z_true), valid)
            fake_z_loss = adversarial_loss(discr_z(z_fake.detach()), fake)
            d_z_loss = 0.5 * (real_z_loss + fake_z_loss)
            
            d_z_loss.backward()
            optimizer_D_z.step()
            time_3=time.time()

            # ----------------------
            #  Train Discriminator y
            # ----------------------
            
            optimizer_D_y.zero_grad()
            y_fake = encoder(depths_in,tnfs_in)[1]

            # Measure discriminator's ability to classify real from generated samples
            real_y_loss = adversarial_loss(discr_y(y_true), valid)
            fake_y_loss = adversarial_loss(discr_y(y_fake.detach()), fake)
            d_y_loss = 0.5 * (real_y_loss + fake_y_loss)
            
            d_y_loss.backward()
            optimizer_D_y.step()
            
            time_4=time.time()
            
            time_sampling=round(time_1-time_0,4)
            time_G=round(time_1-time_0,4)
            time_Dz=round(time_2-time_1,4)
            time_Dy=round(time_3-time_2,4)  
            time_batch=round(time_4-time_0,4)

            G_loss_e+=(float(g_loss.item()))
            V_loss_e+=(float(vae_loss.item()))
            D_z_loss_e+=(float(d_z_loss.item()))
            D_y_loss_e+=(float(d_y_loss.item()))
            CE_e += (float(ce.item()))
            SSE_e += (float(sse.item()))

        time_epoch_1=time.time()
        time_e = np.round((time_epoch_1 -time_epoch_0)/60,3)

        if logfile is not None:
            print('\tEpoch: {}\t Loss Enc/Dec: {:.6f}\t Rec. loss: {:.4f}\t CE: {:.4f}\tSSE: {:.4f}\t Dl loss: {:.7f}\t Dy loss: {:.6f}\t Batchsize: {}\t Epoch time(min): {: .4}'.format(
                  epoch_i + 1,
                  G_loss_e / total_batches_inthis_epoch,
                  V_loss_e / total_batches_inthis_epoch,
                  CE_e / total_batches_inthis_epoch,
                  SSE_e / total_batches_inthis_epoch,
                  D_z_loss_e / total_batches_inthis_epoch,
                  D_y_loss_e / total_batches_inthis_epoch,
                  batchsize,
                  time_e,
                  ), file=logfile)
            logfile.flush()

        # Save Y clusters at epohc i 
        if (epoch_i+1)  == n_epochs  :
            
            batch_size_val = min(50000,depths.shape[0])     
            dataloader_val = make_dataloader(depths, tnfs, batch_size_val, Cuda=_cuda, Shuffle=False, Drop_last=False)
            path_clust = modelfile.replace("aae_model.pt","aae_y_clusters.tsv")
            clust_y_dict, latent,mask =  get_latents_y(encoder,contignames, Dataloader = dataloader_val , Cuda=_cuda ,last_epoch=True)

        #else:
            #path_clust = modelfile.replace("model.pt","clusters_y_along_training/e_"+str(epoch_i+1)+"_aamb_y_clusters.tsv")

            #get_latents_y(encoder,contignames ,Dataloader = dataloader_val, path_clu_Y=path_clust,Cuda=cuda)
       
        # compute clusetring metrics
        #NC_S,NC_Sp,NC_G = benchmark(path_clust+'_clusters.tsv',dataset,0.9,0.9) # set to 0.9 0.9 so we can compare it with ../thesis_aae/Logs/grid_2.2.0.1_RS_Complete_results 
        #NC_S_list_e.append(NC_S)
        #print('NC Strains=', NC_S ,' NC Spec=', NC_Sp, 'NC Gen=', NC_G)
        
        #with open (train_results_file, 'a') as file_obj:
        #    file_obj.write("\n"+str((NC_S))+' '+ str(NC_Sp)+ ' '+ str(NC_G)+' ' +str(round(np.mean(V_loss),3)) +' '+str(np.round(np.mean(D_z_loss),3))+' '+str(np.round(np.mean(D_y_loss),3)) +' ' +str(round(s_l,5)))
        
        # print epoch time to log file
        time_e1=time.time()
        time_e=round((time_e1-time_e0)/60,2)
        
        #os.system('echo -e ' + train_data + '_e_' + str(epoch_i+1) + ' '+ str(time_e)+ ' >> '+log_file)
        #with open (log_file, 'a') as file_obj:
        #    file_obj.write("\nEpoch "+str(epoch_i+1)+', '+ str(time_e) +" minutes")
        
        #V_loss_e = np.round(np.mean(V_loss[-i:]),3)
        #D_z_loss_e= np.round(np.mean(D_z_loss[-i:]),3)
        #D_y_loss_e = np.round(np.mean(D_y_loss[-i:]),4)
        #losses_epoch_string = str(V_loss_epoch) + '\t' + str(D_z_loss_epoch) + '\t' + str(D_y_loss_epoch)
        
        #with open (loss_file, 'a') as file_obj:
        #    file_obj.write("\n"+losses_epoch_string)
           
        
        #G_loss += G_loss_e 
        #D_z_loss += D_z_loss_e 
        #D_y_loss += D_y_loss_e 
        #V_loss += V_loss_e 
        #G_loss_adv_Z += G_loss_adv_Z_e 
        #G_loss_adv_Y += G_loss_adv_Y_e 
        #NC_S_list += NC_S_list_e

        
        #G_loss_epochs.append(np.mean(G_loss_e))
        #D_loss_z_epochs.append(np.mean(D_z_loss_e))
        #D_loss_y_epochs.append(np.mean(D_y_loss_e))
        #V_loss_epochs.append(np.mean(V_loss_e))
        #G_loss_adv_Z_epochs.append(np.mean(G_loss_adv_Z_e))
        #G_loss_adv_Y_epochs.append(np.mean(G_loss_adv_Y_e))


        if (1+epoch_i) % save_checkpoint == 0:
            if modelfile is not None:
                try:

                    checkpoint = {
                            'encoder' : encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'discriminator_z': discriminator_z.state_dict(),
                            'discriminator_y': discriminator_y.state_dict(),
                            'optimizer_E': optimizer_E.state_dict(),
                            'optimizer_D': optimizer_D.state_dict(),
                            'optimizer_D_z': optimizer_D_z.state_dict(),
                            'optimizer_D_y': optimizer_D_y.state_dict(),
                            'nsamples': num_samples,
                            'alpha' : alpha,
                            }
                    torch.save(checkpoint,modelfile)
                except:
                    pass
            
        if modelfile is not None:
            try: 
                checkpoint = {
                        'encoder' : encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'discriminator_z': discriminator_z.state_dict(),
                        'discriminator_y': discriminator_y.state_dict(),
                        'optimizer_E': optimizer_E.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'optimizer_D_z': optimizer_D_z.state_dict(),
                        'optimizer_D_y': optimizer_D_y.state_dict(),
                        'nsamples': num_samples,
                        'alpha' : alpha,
                        }
                torch.save(checkpoint,modelfile)
    
            except:
                pass

    

    return clust_y_dict, latent, mask 





