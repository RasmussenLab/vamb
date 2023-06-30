import vamb
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='/Users/nmb127/Documents/vamb_data/data')
parser.add_argument("--nepoch", type=int, default=500)
parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)
parser.add_argument("--dataset", type=str, default='airways')
parser.add_argument("--supervision", type=float, default=1.)

args = vars(parser.parse_args())
print(args)

exp_id = '_hloss_mmseq_predict_replace_walktrap_no_components'

DATASET_MAP = {
    'airways': 'Airways',
    'gi': 'Gastrointestinal',
    'oral': 'Oral',
    'skin': 'Skin',
    'urog': 'Urogenital',
}

SUP = args['supervision']
CUDA = bool(args['cuda'])
DATASET = args['dataset']
DEPTH_PATH = f'/home/projects/cpr_10006/projects/vamb/data/datasets/cami2_{DATASET}'
PATH_CONTIGS = f'{DEPTH_PATH}/contigs_2kbp.fna.gz'
BAM_PATH = f'{DEPTH_PATH}/bam_sorted'
ABUNDANCE_PATH = f'{DEPTH_PATH}/depths.npz'
MODEL_PATH = f'model_semisupervised_mmseq_genus_{DATASET}{exp_id}.pt'
N_EPOCHS = args['nepoch']
MMSEQ_PATH = f'/home/projects/cpr_10006/people/paupie/vaevae/mmseq2_annotations/ptracker/{DATASET_MAP[DATASET]}_taxonomy.tsv'
GT_PATH = f'gt_tax_{DATASET}{exp_id}.csv'
COMPOSITION_PATH = f'/home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/{DATASET}/composition.npz'
ABUNDANCE_PATH = f'/home/projects/cpr_10006/people/paupie/vaevae/abundances_compositions/{DATASET}/abundance.npz'
WALKTRAP_PATH = f'/home/projects/cpr_10006/people/paupie/vaevae/graph/labels_walktrap/{DATASET_MAP[DATASET]}/walktrap.tsv'

rpkms = np.load(ABUNDANCE_PATH, mmap_mode='r')['matrix']
composition = np.load(COMPOSITION_PATH, mmap_mode='r', allow_pickle=True)
tnfs, lengths = composition['matrix'], composition['lengths']
contignames = composition['identifiers']

df_walktrap = pd.read_csv(WALKTRAP_PATH, delimiter='\t', header=None)
print('Len walktrap', len(df_walktrap))
walktrap_map = {c: i+1 for i, c in enumerate(set(df_walktrap[0]))}
df_walktrap[2] = df_walktrap[0].map(walktrap_map)
labels_assembly_map = {k: v for k, v in zip(df_walktrap[1], df_walktrap[2])}
labels_assembly = np.array([labels_assembly_map.get(c, 0) for c in contignames])
n_walktraps = len(set(labels_assembly_map.values())) + 1

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

ar = one_hot(labels_assembly, num_classes=n_walktraps)
# rpkms = np.concatenate([rpkms, ar], axis=1).astype(np.float32)

print('N contigs', len(contignames))
print('rpkms shape', rpkms.shape)

df_mmseq = pd.read_csv(MMSEQ_PATH, delimiter='\t', header=None)
df_mmseq = df_mmseq[~(df_mmseq[2] == 'no rank')]
df_mmseq_genus = df_mmseq

ind_map = {c: i for i, c in enumerate(contignames)}
indices_mmseq = [ind_map[c] for c in df_mmseq_genus[0]]

graph_column = df_mmseq_genus[8]
nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

model = vamb.h_loss.VAMB2Label(
     rpkms.shape[1], 
     len(nodes), 
     nodes, 
     table_parent,
     cuda=CUDA,
)

with open(f'indices_mmseq_genus_{DATASET}{exp_id}.pickle', 'wb') as handle:
    pickle.dump(indices_mmseq, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms[indices_mmseq], tnfs[indices_mmseq], lengths[indices_mmseq], targets, len(nodes), table_parent)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
with open(MODEL_PATH, 'wb') as modelfile:
    print('training')
    model.trainmodel(
        dataloader_joint,
        nepochs=100,
        modelfile=modelfile,
        logfile=sys.stdout,
        batchsteps=[25, 75],
    )
    print('training')

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
df_gt = pd.DataFrame({'contigs': contignames})
nodes_ar = np.array(nodes)
predictions = []
for i in range(len(df_gt)):
    predictions.append(';'.join(nodes_ar[latent_vamb[i] > 0.5][1:]))
df_gt[f'predictions{exp_id}'] = predictions

df_mmseq_sp = df_mmseq[(df_mmseq[2] == 'species')]
mmseq_map = {k: v for k, v in zip(df_mmseq_sp[0], df_mmseq_sp[8])}
counters = defaultdict(lambda: 0)
preds = []
for i, r in df_gt.iterrows():
    pred_line = r[f'predictions{exp_id}'].split(';')
    try:
        mmseq_line = mmseq_map.get(r['contigs'], '').split(';')
    except AttributeError:
        preds.append(';'.join(pred_line))
        continue
    if mmseq_line[0] != '':
        for i in range(len(mmseq_line)):
            if i < len(pred_line):
                pred_line[i] = mmseq_line[i]
    preds.append(';'.join(pred_line))
df_gt[f'predictions{exp_id}_replace'] = preds

df_gt.to_csv(GT_PATH, index=None)

print('Starting the VAE')

graph_column = df_gt[f'predictions{exp_id}_replace']
nodes, ind_nodes, table_indices, table_true, table_walkdown, table_parent = vamb.h_loss.make_graph(graph_column.unique())

classes_order = np.array(list(graph_column.str.split(';').str[-1]))
targets = [ind_nodes[i] for i in classes_order]

vae = vamb.h_loss.VAEVAEHLoss(
     rpkms.shape[1], 
     len(nodes), 
     nodes, 
     table_parent,
     cuda=CUDA,
     logfile=sys.stdout,
)

dataloader_vamb, mask_vamb = vamb.encode.make_dataloader(rpkms, tnfs, lengths)
dataloader_joint, mask = vamb.h_loss.make_dataloader_concat_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)
dataloader_labels, mask = vamb.h_loss.make_dataloader_labels_hloss(rpkms, tnfs, lengths, targets, len(nodes), table_parent)

shapes = (rpkms.shape[1], 103, 1, len(nodes))
dataloader = vamb.h_loss.make_dataloader_semisupervised_hloss(dataloader_joint, dataloader_vamb, dataloader_labels, len(nodes), table_parent, shapes)
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

names = contignames

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