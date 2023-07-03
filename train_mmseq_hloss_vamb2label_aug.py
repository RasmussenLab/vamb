import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

exp_id = '_hloss_mmseq_predict_aug_overlap'

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = args['dataset']
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
BAM_PATH = f'{DEPTH_PATH}/bam_sorted'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/svekut/mmseq2/{DATASET}_taxonomy.tsv'
GT_PATH = f'gt_tax_{DATASET}.csv'

AUGMENTED_PATH = f'/home/projects/cpr_10006/projects/semi_vamb/data/augmented/{DATASET}'
rpkms_aug = np.load(f'{AUGMENTED_PATH}/new_ab_overlap.npz', allow_pickle=True)['matrix']
comp_aug = np.load(f'{AUGMENTED_PATH}/new_comp_overlap.npz', allow_pickle=True)
tnfs_aug, lengths_aug = comp_aug['matrix'], comp_aug['lengths']

with vamb.vambtools.Reader(PATH_CONTIGS) as contigfile:
    composition = vamb.parsecontigs.Composition.from_file(contigfile)

rpkms = vamb.vambtools.read_npz(ABUNDANCE_PATH)
tnfs, lengths = composition.matrix, composition.metadata.lengths
contignames = composition.metadata.identifiers

ori_contigs = np.array(list(set([c.split('_')[0] for c in comp_aug['identifiers'] if len(c.split('_')) > 1])))
ind_map = {c: i for i, c in enumerate(contignames)}
inds_ori = [ind_map[c] for c in ori_contigs]

rpkms_train = np.concatenate([rpkms_aug, rpkms[inds_ori]])
tnfs_train = np.concatenate([tnfs_aug, tnfs[inds_ori]])
lengths_train = np.concatenate([lengths_aug, lengths[inds_ori]])
contignames_train = np.concatenate([comp_aug['identifiers'], ori_contigs])

df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)

df_mmseq = df_mmseq[~(df_mmseq[2] == 'no rank')]

data = {
    0: [],
    2: [],
    8: [],
}
mmseq_map = dict(zip(df_mmseq[0], df_mmseq[8]))
mmseq_map_genus = dict(zip(df_mmseq[0], df_mmseq[2]))
added = set()
for c in contignames_train:
    c_map = c.split('_')[0]
    if c_map in mmseq_map:
        data[0].append(c)
        data[8].append(mmseq_map[c_map])
        data[2].append(mmseq_map_genus[c_map])
df_mmseq_aug = pd.DataFrame(data)
df_mmseq_aug.to_csv(f'mmseq_aug_{DATASET}.csv', index=None)

ind_map = {c: i for i, c in enumerate(contignames_train)}
indices_mmseq = [ind_map[c] for c in df_mmseq_aug[0]]

graph_column = df_mmseq_aug[8]
nodes, ind_nodes, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

model = vamb.h_loss.VAMB2Label(
     rpkms.shape[1], 
     len(nodes), 
     table_indices, 
     table_true, 
     table_walkdown, 
     nodes, 
     table_parent,
     cuda=CUDA,
)

with open(f'indices_mmseq_genus_{DATASET}{exp_id}.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms_train[indices_mmseq], tnfs_train[indices_mmseq], lengths_train[indices_mmseq], targets, len(nodes), table_parent)

shapes = (rpkms_train.shape[1], 103, 1, len(nodes))
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    model.trainmodel(
        dataloader_joint,
        nepochs=100,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[],
    )
    print('training')

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)

latent_vamb = model.predict(dataloader_vamb)
LATENT_PATH = f'latent_predict_vamb_{DATASET}{exp_id}.npy'
# print('Saving latent space: Vamb', latent_vamb.shape)
# np.save(LATENT_PATH, latent_vamb)

print('Saving the tree', len(nodes))
TREE_PATH = f'tree_predict_vamb_{DATASET}{exp_id}.npy'
np.save(TREE_PATH, np.array(nodes))
PARENT_PATH = f'parents_predict_vamb_{DATASET}{exp_id}.npy'
np.save(PARENT_PATH, np.array(table_parent))

print('Getting the predictions')
df_gt = pd.read_csv(GT_PATH)
predictions = []
for i in range(len(df_gt)):
    predictions.append(';'.join(np.array(nodes)[latent_vamb[i] > 0.5][1:]))
df_gt[f'predictions{exp_id}'] = predictions
df_gt.to_csv(GT_PATH, index=None)

print('Getting augmented predictions')
dataloader_train, mask_vamb = vamb.encode.make_dataloader(rpkms_train, tnfs_train, lengths_train)

latent_vamb = model.predict(dataloader_train)
LATENT_PATH = f'latent_predict_vamb_aug_{DATASET}{exp_id}.npy'
# print('Saving latent space: Vamb', latent_vamb.shape)
# np.save(LATENT_PATH, latent_vamb)

df_train = pd.DataFrame({'contigs': contignames_train})
predictions = []
for i in range(len(df_train)):
    predictions.append(';'.join(np.array(nodes)[latent_vamb[i] > 0.5][1:]))
df_train[f'predictions{exp_id}'] = predictions
df_train.to_csv(f'train_{DATASET}.csv', index=None)

print('Starting the VAE')

graph_column = df_train[f'predictions{exp_id}']
nodes, ind_nodes, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms_train, tnfs_train, lengths_train, targets, len(nodes), table_parent)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms_train, tnfs_train, lengths_train, targets, len(nodes), table_parent)

vae = vamb.h_loss.VAEVAEHLoss(
     rpkms.shape[1], 
     len(nodes), 
     table_indices, 
     table_true, 
     table_walkdown, 
     nodes, 
     table_parent,
     cuda=CUDA,
     logfile=sys.stdout,
)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
dataloader = vamb.h_loss.make_dataloader_semisupervised_hloss(dataloader_joint, dataloader_train, dataloader_labels, len(nodes), table_parent, shapes)
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    vae.trainmodel(
        dataloader,
        nepochs=N_EPOCHS,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[25, 75, 150],
    )
    print('training')

graph_column = df_gt[f'predictions{exp_id}']
classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)

latent_vamb = vae.VAEVamb.encode(dataloader_vamb)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_vamb_{DATASET}{exp_id}.npy'
print('Saving latent space: Vamb')
np.save(LATENT_PATH, latent_vamb)

latent_labels = vae.VAELabels.encode(dataloader_labels)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_labels_{DATASET}{exp_id}.npy'
print('Saving latent space: Labels')
np.save(LATENT_PATH, latent_labels)

latent_both = vae.VAEJoint.encode(dataloader_joint)
LATENT_PATH = f'latent_trained_lengths_mmseq_genus_both_{DATASET}{exp_id}.npy'
print('Saving latent space: Both')
np.save(LATENT_PATH, latent_both)

composition.metadata.filter_mask(mask_vamb)
names = composition.metadata.identifiers

with open(f'clusters_mmseq_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
    iterator = vamb.cluster.ClusterGenerator(latent_both)
    iterator = vamb.vambtools.binsplit(
        (
            (names[cluster.medoid], {names[m] for m in cluster.members})
            for cluster in iterator
        ),
        "C"
    )
    vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_vamb_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_vamb)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)

with open(f'clusters_mmseq_labels_{DATASET}_v4{exp_id}.tsv', 'w') as binfile:
        iterator = vamb.cluster.ClusterGenerator(latent_labels)
        iterator = vamb.vambtools.binsplit(
            (
                (names[cluster.medoid], {names[m] for m in cluster.members})
                for cluster in iterator
            ),
            "C"
        )
        vamb.vambtools.write_clusters(binfile, iterator)