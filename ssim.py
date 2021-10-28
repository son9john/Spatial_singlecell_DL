# %% codecell
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finding Structural Simiarity
Created on Sun Sep 12 23:47:57 2021

@author: Junho John Song
"""
import more_itertools as mit
import logging
from tqdm import tqdm

import hydra
import scanpy as sc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
from skimage.metrics import structural_similarity as ssim
import torch

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
    import os
    import hydra
    from omegaconf import OmegaConf as OC
    # Load config
    os.getcwd()
    # PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
    PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
    os.chdir(PROJECT_DIR)
    os.listdir()

    # Dummy class for debugging
    class Dummy():
        """ Dummy class for debugging """
        def __init__(self):
            pass
    log = Dummy()
    log.info=print

    # %%
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
    overrides = []
    overrides = ['save_x=True']
    overrides = ['criterion=bce_loss']
    overrides = ['criterion=bce_loss', 'train.epoch=0']
    cfg = hydra.compose(config_name='autoencoder', overrides=overrides)
    print(OC.to_yaml(cfg))

# %%
@hydra.main(config_path='conf', config_name='autoencoder')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load data
    data = hydra.utils.instantiate(cfg.data.dataset)
    x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).squeeze(1).numpy()

    # Test ssim
    n_genes = len(x_all)
    x_all.shape
    dist = np.zeros((n_genes,n_genes))

    # %%
    for (i, x1), (j, x2) in it.combinations(enumerate(x_all), 2):
        dist[i,j] = ssim(x1, x2)
    # Copy lower triangular
    dist[np.tril_indices(n_genes)] = dist.T[np.tril_indices(n_genes)]

    # %%
list(mit.pairwise(np.arange(10)))
import itertools as it


list(it.combinations(enumerate('abcde'), 2))
# %%

if __name__=='__main__':
    main()

# %% codecell
adata = sc.read_h5ad(cfg.path.data)
# adata = sc.read_h5ad('./data/smaller_dada.h5ad')
# %% codecell
adata.shape

# %% codecell
gene_names = adata.var.index.tolist()
row_col = adata.obs[['array_row', 'array_col']].values.astype(int)
df = pd.DataFrame(data=np.concatenate((row_col, adata.X), axis=1), columns=['row', 'col'] + gene_names)
df['row'] = df['row'].astype(int)
df['col'] = df['col'].astype(int)

# %% codecell
def min_max_scale(gm):
    not_nan_values = gm[~np.isnan(gm)]
    mx = not_nan_values.max()
    mn = not_nan_values.min()
    gm = gm - mn
    gm = gm / (mx - mn)
    return gm

def make_gene_map(gene_df, gene_name):
    gm = min_max_scale(gene_df.pivot('row', 'col', gene_name).values)
    gm[np.isnan(gm)] = 0
    return gm

row_len = row_col[:, 0].max() - row_col[:, 0].min() + 1
col_len = row_col[:, 1].max() - row_col[:, 1].min() + 1
gene_maps = np.zeros([len(gene_names), row_len, col_len], dtype=np.float32)
for i, name in enumerate(gene_names):
    gene_maps[i] = make_gene_map(df, name)
# %% codecell
data = gene_maps
data1 = gene_maps[gene_names.index('Calml4')]

plt.imshow(data1)
plt.show()
# %% codecell
def find_similar_genes(target_feat, all_features, top_k=10):
    dist = [-1 * ssim(target_feat, all_features[i]) for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(gene_maps[target_gene_idx], gene_maps)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
plt.plot(x, jaccard_mean)
plt.ylim([0, 1])
# %% codecell
np.mean(jaccard_auc_list)
# %% codecell
plt.plot(x, precision_mean)
plt.ylim([0, 1])
# %% codecell
np.mean(precision_auc_list)
# %% codecell
target_gene_name = 'Ttr'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(gene_maps[target_gene_idx], gene_maps)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% markdown
# ## Jaccard & Precision measurement
# %% codecell
def jaccard_precision_recall(gt, pred, threshold=0.5):
    assert gt.max() <= 1 and gt.min() >= 0
    assert pred.max() <= 1 and pred.min() >= 0
    assert threshold <= 1 and threshold >= 0
    assert gt.shape[0] == pred.shape[0]
    assert gt.shape[1] == pred.shape[1]

    gt = (gt >= threshold)
    pred = (pred >= threshold)

    inter = (gt & pred).sum()
    gt_area = gt.sum()
    pred_area = pred.sum()
    union = gt_area + pred_area - inter

    if union > 0:
        jaccard = inter / union
    else:
        jaccard = 0
    if pred_area > 0:
        precision = inter / pred_area
    else:
        precision = 0
    return jaccard, precision

def jaccard_precision_curve(gt, pred, threshold_num=1000):
    threshold_list = []
    jaccard_list = []
    precision_list = []
    for th in np.linspace(0, 1, threshold_num):
        threshold_list.append(th)
        j, p = jaccard_precision_recall(gt, pred, th)
        jaccard_list.append(j)
        precision_list.append(p)

    return threshold_list, jaccard_list, precision_list

# %% codecell

# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
# top10_genes = find_similar_genes(all_features[target_gene_idx], all_feats)
# %% codecell
import os
from tqdm import tqdm

import hydra
import numpy as np
from omegaconf import OmegaConf as OC
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

import dataset as DAT
import tools as T
import tools.modules
import tools.sklearn
# %% codecell
# Load config
os.getcwd()
PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
# PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)
os.listdir()

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='l_regression', overrides=overrides)
print(OC.to_yaml(cfg))
# %% codecell
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% codecell
l_model_features = np.zeros([len(gene_names), cfg.model.n_features], dtype=np.float32)
# model.eval()
# for i in range(len(gene_names)):
#     x = torch.tensor(np.expand_dims(gene_maps[i], [0, 1])).to(device)
#     with torch.no_grad():
#         l_model_features[i] = model(x)[0].cpu().numpy()
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)
gt = gene_maps[target_gene_idx]
jaccard_auc_list = []
precision_auc_list = []
threshold_num = 1000
jaccard_sum = np.zeros([threshold_num])
precision_sum = np.zeros([threshold_num])

# %% codecell

for i in top10_genes:
    pred = gene_maps[i]
    x, j, p = jaccard_precision_curve(gt, pred, threshold_num)
    jaccard_auc_list.append(metrics.auc(x, j))
    precision_auc_list.append(metrics.auc(x, p))

    jaccard_sum += j
    precision_sum += p

    plt.figure()
    plt.plot(x, j)
    plt.plot(x, p)
    plt.plot([0, 1], [1, 0])
    plt.show()

# %% codecell

jaccard_mean = jaccard_sum / len(top10_genes)
precision_mean = precision_sum /len(top10_genes)
plt.plot(x, jaccard_mean)
plt.ylim([0, 1])
np.mean(jaccard_auc_list)
plt.plot(x, precision_mean)
plt.ylim([0, 1])
np.mean(precision_auc_list)

# %% codecell
