# %% codecell
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression model to calculate bivariate L
Created on Sun Sep 12 21:20:14 2021

@author: Junho John Song
"""


import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf as OC


# %%
# Load config
import os
os.getcwd()
PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)
os.listdir()

import hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='l_regression', overrides=overrides)
print(OC.to_yaml(cfg))

# %% codecell
# ## Load data
adda = sc.read_h5ad(cfg.path.data)
gene_names = adda.var.index.tolist()
row_col = adda.obs[['array_row', 'array_col']].values.astype(int)
df = pd.DataFrame(data=np.concatenate((row_col, adda.X), axis=1), columns=['row', 'col'] + gene_names)
df['row'] = df['row'].astype(int)
df['col'] = df['col'].astype(int)

df
adda

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
df = pd.read_csv(cfg.path.bsc, sep='\t', header=None)
df.columns = ['gene1', 'gene2', 'a', 'b', 'c', 'd', 'L']
df = df[df['gene1'].isin(gene_names)]
df = df[df['gene2'].isin(gene_names)]
df = df.reset_index(drop=True)
# %% markdown
# ## Make train data
# %% codecell
name2idx = {name:i for i, name in enumerate(gene_names)}
pairs = np.zeros([len(df), 2], np.int32)
pairs[:, 0] = df['gene1'].apply(lambda x: name2idx[x])
pairs[:, 1] = df['gene2'].apply(lambda x: name2idx[x])
y = df['L'].values
# %% codecell
rarr = np.arange(pairs.shape[0])
np.random.shuffle(rarr)
pairs = pairs[rarr]
y = y[rarr]

train_num = int(0.8 * pairs.shape[0])
val_num = int(0.1 * pairs.shape[0])
test_num = pairs.shape[0] - train_num - val_num

pairs_train = pairs[:train_num]
y_train = y[:train_num]
pairs_val = pairs[train_num:train_num + val_num]
y_val = y[train_num:train_num + val_num]
pairs_test = pairs[train_num + val_num:]
y_test = y[train_num + val_num:]
# %% codecell
class LDataset(Dataset):

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
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.lin = nn.Sequential(
            nn.Linear(4320, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128, bias=True)
        )

    def forward(self, x1, x2):
        h1 = self.cnn(x1)
        h1 = h1.view(h1.size(0), -1)
        h1 = self.lin(h1)
        h2 = self.cnn(x2)
        h2 = h2.view(h2.size(0), -1)
        h2 = self.lin(h2)
        o = (h1 * h2).mean(1)
        return o

    def extract_features(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        h = self.lin(h)
        return h
# %% codecell
def _compute_loss(model, x1, x2, y):
    out = model(x1, x2)
    loss = F.mse_loss(out, y)
    return loss


batch_size = 128
init_lr = 0.0001

model = LModel()
optim = torch.optim.Adam(model.parameters(), lr=init_lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for epoch in range(100):
    print(f'Epoch: {epoch}')
    for split in ['train', 'val']:

        if split == 'train':
            dataset = LDataset(gene_maps, pairs_train, y_train)
        elif split == 'val':
            dataset = LDataset(gene_maps, pairs_val, y_val)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=(split == 'train'))
        cnt = 0
        loss_sum = 0
        for step, batch_data in enumerate(tqdm(dataloader)):
            x1, x2, y = [x.to(device=device) for x in batch_data]

            if split == 'train':
                model.train()
                optim.zero_grad()
                loss = _compute_loss(model, x1, x2, y)
                loss.backward()
                optim.step()
            else:
                model.eval()
                with torch.no_grad():
                    loss = _compute_loss(model, x1, x2, y)

            loss_sum += loss.item()
            cnt += 1

        print(f'{split} loss: {loss_sum / cnt}')
# %% codecell
l_model_features = np.zeros([len(gene_names), 128], dtype=np.float32)
model.eval()
for i in range(len(gene_names)):
    x = torch.tensor(np.expand_dims(gene_maps[i], [0, 1]))
    with torch.no_grad():
        l_model_features[i] = model.extract_features(x)[0].numpy()
# %% codecell

# %% codecell

# %% codecell

# %% codecell

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
    top_k_idx = np.argsort(dist)[::-1][:top_k]
    return top_k_idx, dist[top_k_idx]

def find_far_genes(target_feat, all_features, top_k=10):
    dist = np.array([(target_feat * all_features[i]).mean() for i in range(all_features.shape[0])])
    top_k_idx = np.argsort(dist)[:top_k]
    return top_k_idx, dist[top_k_idx]
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell
target_gene_name = 'Ttr'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell
target_gene_name = 'Zcchc12'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes, l_values = find_similar_genes(l_model_features[target_gene_idx], l_model_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
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

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
print(l_values)
# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
for i in tqdm(range(100000000)):
    v = (l_model_features[0] * l_model_features[2]).mean()
# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
