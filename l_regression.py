# %% codecell
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression model to calculate bivariate L
Created on Sun Sep 12 21:20:14 2021

@author: Junho John Song
"""

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

'''
@John

1. importing
    (not 100% sure)
    neat way to import libraries:
    1. order the libraries into 3 groups.
    1) python native libraries (ex: os, random, tqdm, multiprocessing, signal, etc)
    2) official non-native python libraries (ex: pandas, numpy, torch)
    3) unofficial non-native python libraries (ex: personal libraries, or imports from this project)

    2. Order them alphabetically

2. official import for torch.optim is "import torch.optim as optim"

(3. "import torch.utils.data as D" is unofficial and it's just my style :P )
'''

# %%
# Load config
os.getcwd()
# PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)
os.listdir()

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='l_regression', overrides=overrides)
print(OC.to_yaml(cfg))

# %% codecell
# ## Load data
adata = sc.read_h5ad(cfg.path.data)

# Refine data
gene_maps, scalers = DAT.get_gene_map(adata)

# %% codecell
# Load bsc
df_bsc = pd.read_csv(cfg.path.bsc, sep='\t', header=None)
df_bsc.columns = ['gene1', 'gene2', 'a', 'b', 'c', 'd', 'L']
df_bsc = df_bsc[df_bsc['gene1'].isin(gene_names) & df_bsc['gene2'].isin(gene_names)]
df_bsc = df_bsc.reset_index(drop=True)

# %% markdown
# ## Make train data
# %% codecell
name2idx = {name:i for i, name in enumerate(gene_names)}
name2idx
pairs = np.zeros([len(df_bsc), 2], np.int32)
pairs[:, 0] = df_bsc['gene1'].apply(lambda x: name2idx[x])
pairs[:, 1] = df_bsc['gene2'].apply(lambda x: name2idx[x])
y = df['L'].values

# %% codecell
# train/validation/test split
assert cfg.split.train+cfg.split.val+cfg.split.test==1, f'ratio of train, val, test must sum to 1. received: [train: {cfg.split.train}][val: {cfg.split.val}][test: {cfg.split.test}][sum: {cfg.split.train+cfg.split.val+cfg.split.test}]'

train_i, val_i, test_i = T.sklearn.model_selection.train_val_test_split_i(y, val_size=cfg.split.val, test_size=cfg.split.test, random_state=cfg.random.seed)
pairs_train, pairs_val, pairs_test = pairs[train_i], pairs[val_i], pairs[test_i]
y_train, y_val, y_test = y[train_i], y[val_i], y[test_i]

# %% codecell
# Define torch dataset & model
class LDataset(D.Dataset):
    def __init__(self, gene_maps, pairs, y):
        self.gene_maps = np.expand_dims(gene_maps, 1)
        self.pairs = pairs
        self.y = y

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        x1 = self.gene_maps[pair[0]].astype(np.float32)
        x2 = self.gene_maps[pair[1]].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        return x1, x2, y

# %% codecell
class LModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(1), # Flattens out except for batch dimension
            nn.LazyLinear(512, bias=True), # you don't have to specify input nodes for LazyLinear. Great for CNN+FNN :)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, cfg.model.n_features, bias=True)
        )

    def forward(self, x):
        return self.layers(x)

# %% codecell
def _compute_loss(model, x1, x2, y):
    h1 = model(x1)
    h2 = model(x2)
    out = (h1 * h2).mean(1)
    loss = F.mse_loss(out, y)
    return loss

'''
@John
It's better for a model to forward a single input unless it's necessary.
What you wanted to do, which was to compute the dot product, is something what the "program" should do, not the model.
So keep the task of a model to its most simplest form.
'''

# %%
'''
@John since dataset & dataloader doesn't have to change every epoch, make them before the train loop

Q. why did you limit the num_workers=1?
'''
dataset_train = LDataset(gene_maps, pairs_train, y_train)
dataset_val = LDataset(gene_maps, pairs_val, y_val)
loader_train = D.DataLoader(dataset=dataset_train, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True) # better to shuffle when training. more stable.
loader_val = D.DataLoader(dataset=dataset_val, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)
# loader_train = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=1, drop_last=True) # better to shuffle when training. more stable.
# loader_train = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=1, drop_last=False)

# %%
loss_tracker_train = T.modules.ValueTracker()
loss_tracker_val = T.modules.ValueTracker()

# %%
model = LModel()
op = optim.Adam(model.parameters(), lr=cfg.train.lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(cfg.train.epoch):
    print(f'Epoch: {epoch}')

    # train
    model.train()
    loss_tracker_train.reset()
    for data in tqdm(loader_train):
        x1, x2, y = data
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        N = len(y) # to compute average loss

        loss = _compute_loss(model, x1, x2, y)
        op.zero_grad()
        loss.backward()
        op.step()

        loss_tracker_train.step(loss.item(), N) # item can only be called on single valued tensor. returns the number.

    print(f'[train][loss: {loss_tracker_val.avg}]')

    # validation
    model.eval()
    loss_tracker_val.reset()
    with torch.no_grad()
        for data in tqdm(loader_val):
            x1, x2, y = data
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            N = len(y)

            loss = _compute_loss(model, x1, x2, y)
            loss_tracker_val.step(loss.item(), N)
    print(f'[validation][loss: {loss_tracker_val.avg}]')

# %% codecell
l_model_features = np.zeros([len(gene_names), cfg.model.n_features], dtype=np.float32)
model.eval()
for i in range(len(gene_names)):
    x = torch.tensor(np.expand_dims(gene_maps[i], [0, 1]))
    with torch.no_grad():
        l_model_features[i] = model.extract_features(x)[0].numpy()

# %% markdown
# ## Find near ones
# %% codecell
def calc_l(dist_model, img1, img2):
    x1 = torch.tensor(np.expand_dims(img1, [0, 1]).astype(np.float32))
    x2 = torch.tensor(np.expand_dims(img2, [0, 1]).astype(np.float32))
    model.eval()
    with torch.no_grad():
        out = model(x1, x2,)[0].item()
    return out

def find_similar_genes(target_feat, all_features, top_k=10):
    dist = np.array([(target_feat * all_features[i]).mean() for i in range(all_features.shape[0])])
    top_k_idx = np.argsort(dist)[-top_k:]
    return top_k_idx, dist[top_k_idx]

def find_far_genes(target_feat, all_features, top_k=10):
    dist = np.array([(target_feat * all_features[i]).mean() for i in range(all_features.shape[0])])
    top_k_idx = np.argsort(dist)[:top_k]
    return top_k_idx, dist[top_k_idx]
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell
target_gene_name = 'Ttr'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell
target_gene_name = 'Zcchc12'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell

# %% codecell

# %% codecell

# %% markdown
# ## Fine far ones
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_far_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell

# %% codecell
for i in tqdm(range(100000000)):
    v = (l_model_features[0] * l_model_features[2]).mean()
# %% codecell

# %% codecell
