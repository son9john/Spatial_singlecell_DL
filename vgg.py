# %% codecell
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression model to calculate bivariate L
Created on Sun Sep 12 20:22:32 2021

@author: Junho John Song
"""

import pickle
from tqdm import tqdm

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import torch
from PIL import Image
from torchvision import transforms
# %% codecell
adda = sc.read_h5ad('smaller_dada.h5ad')
# %% codecell
gene_names = adda.var.index.tolist()
row_col = adda.obs[['array_row', 'array_col']].values.astype(int)
df = pd.DataFrame(data=np.concatenate((row_col, adda.X), axis=1), columns=['row', 'col'] + gene_names)
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
# gene_maps[gene_names.index('St18')]

# %% codecell
gene_names
# %% codecell
sc.pl.spatial(adda, color=['Vxn'])
# %% codecell

# %% codecell
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
new_classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
model.classifier = new_classifier
model.eval()
# %% codecell
def extract_features(gene_map, model):
    input_image = Image.fromarray(np.uint8(gene_map * 255) , 'L').convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)

    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        out = model(input_batch)[0]
    return out.numpy()
# %% codecell
vgg_features = np.zeros([len(gene_names), 4096], dtype=np.float32)

for i in tqdm(range(len(gene_names))):
     vgg_features[i] = extract_features(gene_maps[i], model)
# %% codecell

# %% codecell
vgg_features[0]
# %% codecell
plt.matshow(gene_maps[1])
# %% codecell
vgg_features[1]
# %% codecell
def find_similar_genes(target_feat, all_features, top_k=10):
    dist = [distance.cosine(target_feat, all_features[i]) for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx

def find_similar_genes_euclidean(target_feat, all_features, top_k=10):
    dist = [np.square(target_feat - all_features[i]).sum() for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx

def find_similar_genes_dot_prod(target_feat, all_features, top_k=10):
    dist = [-1 * (target_feat * all_features[i]).sum() for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx
# %% codecell
target_gene_name = 'Zcchc12'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
target_gene_name = 'Ppp3ca'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
target_gene_name = 'Ttr'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes_euclidean(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
target_gene_name = 'Vxn'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes_dot_prod(vgg_features[target_gene_idx], vgg_features)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell

# %% codecell

# %% codecell

# %% codecell
gene_maps.shape
# %% codecell

# %% codecell
[1, 2, 3] [5, 2, 1] = 1*5 + 2*2 + 3*1
# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
