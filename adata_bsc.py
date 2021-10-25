# %% codecell
import os
import multiprocessing
import itertools as it
import tqdm

import scanpy as sc
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf as OC
%matplotlib inline

# %%
os.getcwd()
# PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)
os.listdir()

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='adata_bsc', overrides=overrides)
print(OC.to_yaml(cfg)) # print configuration in pretty format

# %% codecell
# Load data
'''
df.index: observation sites
df.columns: gene names + row,col index
df.values: expression level
'''
adata = sc.read_h5ad(cfg.path.data)

# %% codecell
def adjacency_vector(positions_in_tissue, position):
    row_i = int(position['array_row'])
    col_i = int(position['array_col'])

    pit_row1, pit_row2, pit_row3 = positions_in_tissue[positions_in_tissue['array_row'] == row_i-1], positions_in_tissue[positions_in_tissue['array_row'] == row_i], positions_in_tissue[positions_in_tissue['array_row'] == row_i+1]
    pit_row1 = pit_row1[(pit_row1['array_col'] == col_i-1) | (pit_row1['array_col'] == col_i+1)]
    pit_row2 = pit_row2[(pit_row2['array_col'] == col_i-2) | (pit_row2['array_col'] == col_i+2)]
    pit_row3 = pit_row3[(pit_row3['array_col'] == col_i-1) | (pit_row3['array_col'] == col_i+1)]
    pit_adjacent = pd.concat((pit_row1,pit_row2,pit_row3), axis=0)
    adjacent_i = np.sort(pit_adjacent.index.values) # sorting is not necessary though

    C_vector = np.zeros(len(positions_in_tissue), dtype=bool)
    C_vector[adjacent_i]=True
    return C_vector

# %%
def create_connectivity_matrix(adata):

    positions_in_tissue = adata.obs[['in_tissue', 'array_row', 'array_col']][adata.obs['in_tissue'] ==1]
    barcodes_in_tissue = positions_in_tissue.index
    nbarcodes_in_tissue = len(positions_in_tissue)
    positions_in_tissue = positions_in_tissue.reset_index().rename(columns={'index':'_id'})

    # accelerate using multiprocessing
    # def adjacency_vector(positions_in_tissue, position):
    #     row_i = int(position['array_row'])
    #     col_i = int(position['array_col'])
    #
    #     pit_row1, pit_row2, pit_row3 = positions_in_tissue[positions_in_tissue['array_row'] == row_i-1], positions_in_tissue[positions_in_tissue['array_row'] == row_i], positions_in_tissue[positions_in_tissue['array_row'] == row_i+1]
    #     pit_row1 = pit_row1[(pit_row1['array_col'] == col_i-1) | (pit_row1['array_col'] == col_i+1)]
    #     pit_row2 = pit_row2[(pit_row2['array_col'] == col_i-2) | (pit_row2['array_col'] == col_i+2)]
    #     pit_row3 = pit_row3[(pit_row3['array_col'] == col_i-1) | (pit_row3['array_col'] == col_i+1)]
    #     pit_adjacent = pd.concat((pit_row1,pit_row2,pit_row3), axis=0)
    #     adjacent_i = np.sort(pit_adjacent.index.values) # sorting is not necessary though
    #
    #     C_vector = np.zeros(len(positions_in_tissue), dtype=bool)
    #     C_vector[adjacent_i]=True
    #     return C_vector

    with multiprocessing.Pool(processes=16) as pool:
        C_list = pool.starmap(adjacency_vector, zip(it.repeat(positions_in_tissue, len(positions_in_tissue)), positions_in_tissue.iloc))
    C = np.stack(C_list, axis=0)

    # C = np.zeros([nbarcodes_in_tissue, nbarcodes_in_tissue])
    # for idx, barcode in enumerate(barcodes_in_tissue):
    #
    #     row_i  = int(positions_in_tissue[positions_in_tissue['_id'] == barcode ]['array_row'])
    #     col_i= int(positions_in_tissue[positions_in_tissue['_id'] == barcode ]['array_col'])
    #
    #     condition = ((positions_in_tissue['array_row'] == row_i-1 )&(positions_in_tissue['array_col'] == col_i-1))\
    #                 | ((positions_in_tissue['array_row'] == row_i-1 )&(positions_in_tissue['array_col'] == col_i+1))\
    #                 | ((positions_in_tissue['array_row'] == row_i )&(positions_in_tissue['array_col'] == col_i-2))\
    #                 | ((positions_in_tissue['array_row'] == row_i)&(positions_in_tissue['array_col'] == col_i+2))\
    #                 | ((positions_in_tissue['array_row'] == row_i+1 )&(positions_in_tissue['array_col'] == col_i-1))\
    #                 | (positions_in_tissue['array_row'] == row_i+1 )&(positions_in_tissue['array_col'] == col_i+1)
    #     tmp = positions_in_tissue[condition]
    #
    #     if len(tmp) > 0:
    #         for j in tmp.index:
    #             C[idx, j] = 1

    # Normalize connectivity matrix. when there's no connections, leave it zero
    row_sums = C.sum(1)
    row_sums[row_sums == 0] = 1 # entries which contain zeros, so just divide by 1
    W = C.astype(float) / row_sums.reshape(-1, 1)
    assert (~np.isnan(W)).all(), 'nan in W'
    assert (np.isclose(W.sum(1), 1) | (W.sum(1)==0)).all(), 'sum of W must equal 1 or 0 (none)' # due to floating point precision in W.sum(1)
    # pd.Series(W.sum(1)).value_counts()
    # np.isnan(W).flatten().any()
    # np.isnan(W).flatten().all()
    # C[C.sum(1)==0].sum()
    # W[C.sum(1)==0].sum()
    # W.sum(1)[W.sum(1)!=1]
    # [W.sum(1)!=1]
    # np.isnan(W).any()
    # np.where(np.isnan(W))
    # np.isnan(W)
    # pd.Series(row_sums).value_counts()
    # row_sums.min()

    # np.trace(W@W.T) == np.diagonal((np.dot(W, W).T)).sum()
    # np.trace(W@W.T) == np.diagonal((np.dot(W, W.T))).sum()
    # np.trace(W@W.T)
    # np.diagonal((np.dot(W, W).T)).sum()

    conn_info = dict()
    # conn_info['L_estimate_divR'] = np.diagonal((np.dot(W, W).T)).sum() / (nbarcodes_in_tissue - 1) # Error in mathematics! should be np.dot(W, W.T)
    conn_info['L_estimate_divR'] = np.trace(W@W.T) / (nbarcodes_in_tissue - 1)
    conn_info['barcodes_in_tissue'] = barcodes_in_tissue.tolist()
    conn_info['nbarcodes_in_tissue'] = nbarcodes_in_tissue
    conn_info['W'] = W
    conn_info['C'] = C

    return conn_info
# %% codecell
conn_info = create_connectivity_matrix(adata)
# pd.Series(conn_info['W'].sum(1)).value_counts()

# %% codecell
# def calculate_bsc(feature1, feature2, adata, conn_info):
def calculate_bsc(feature1, feature2, W):
    # Put them outside of the loop. memory&time cost
    # gene_names = adata.var.index.tolist()
    # row_col = adata.obs[['array_row', 'array_col']].values.astype(int)
    # df = pd.DataFrame(data=np.concatenate((row_col, adata.X), axis=1), columns=['row', 'col'] + gene_names)

    x_mean = np.mean(feature1)
    y_mean = np.mean(feature2)

    # x_smooth = np.dot(conn_info['W'], x_values)
    # y_smooth = np.dot(conn_info['W'], y_values)
    x_smooth = np.dot(W, x_values)
    y_smooth = np.dot(W, y_values)

    x_mean_sm = np.mean(x_smooth) # muX
    y_mean_sm = np.mean(y_smooth) # muY

    # Calculate Peason's r(X,Y), r(smooth), L_XX, L_YY, L_XY as in Lee S (2001)
    r = sum((x_values - x_mean) * (y_values - y_mean)) \
       / (np.sqrt(sum((x_values - x_mean) ** 2)) * np.sqrt(sum((y_values - y_mean) ** 2)))
    r_sm = sum((x_smooth - x_mean_sm) * (y_smooth - y_mean_sm)) \
          / (np.sqrt(sum((x_smooth - x_mean_sm) ** 2)) * np.sqrt(sum((y_smooth - y_mean_sm) ** 2)))

    L_XX = sum((x_smooth - x_mean) ** 2) / sum((x_values - x_mean) ** 2)
    L_YY = sum((y_smooth - y_mean) ** 2) / sum((y_values - y_mean) ** 2)
    L_XY = np.sqrt(L_XX) * np.sqrt(L_YY) * r_sm

    bsc = {
        'r': r,
        'r_sm': r_sm,
        'L_XX': L_XX,
        'L_YY': L_YY,
        'L_XY': L_XY
    }

    return bsc

def calculate_bsc_all(features, W):
    '''
    :param features: array of (N_barcodes, N_genes)
    :param W: array of (N_barcodes, N_barcodes)

    Note, ddof=1 using np.corrcoef
    '''
    mean = features.mean(axis=0)
    features_smooth = W @ features
    mean_smooth = features_smooth.mean(axis=0)

    d = features - mean # deviation
    d_squared_sum = (d**2).sum(0)
    rsd = np.sqrt(d_squared_sum) # rsd: root_standard_deviation
    d_smooth = features_smooth - mean_smooth
    d_smooth_squared_sum = (d_smooth**2).sum(0)
    rsd_smooth = np.sqrt(d_smooth_squared_sum)

    # Self L
    L = d_smooth_squared_sum / d_squared_sum

    # r, r_sm, L_XY
    # Pearson correlation? just use external library.
    r = np.corrcoef(features.T)
    r_sm = np.corrcoef(features_smooth.T)

    L_sqrt = np.sqrt(L)
    L_XY = (L_sqrt[...,None]@L_sqrt[None,...]) * r_sm

    bsc = {
    'r': r,
    'r_sm': r_sm,
    'L': L,
    'L_XY': L_XY,
    }
    return bsc

# # %%

# %% codecell
gene_names = adata.var.index.tolist()
row_col = adata.obs[['array_row', 'array_col']].values.astype(int)
df = pd.DataFrame(data=np.concatenate((row_col, adata.X), axis=1), columns=['row', 'col'] + gene_names)

import scipy
print(f'n_combinations: {scipy.special.comb(len(gene_names), 2)}')
features = df[gene_names].values

# %%
import tools as T
timer=T.Timer()

# %%
timer.start()
bsc_all = calculate_bsc_all(features, conn_info['W'])
timer.stop()
print(f'elapsed_time: {timer.elapsed_time}')

# %%
# CPU - takes long
bsc_list = []
for gene1, gene2 in tqdm.tqdm(it.combinations(gene_names, 2)):
    x_values = df[gene1].values
    y_values = df[gene2].values
    # y_values.shape
    # print(gene1, gene2)
    bsc = calculate_bsc(x_values, y_values, conn_info['W'])
    bsc['gene1'] = gene1
    bsc['gene2'] = gene2
    bsc_list.append(bsc)

# %%
# Consistency check.
# Little errors due to floating point precision & degree of freedom.
bsc_all['r'][0,1]
bsc_all['r_sm'][0,1]
bsc_all['L'][0]
bsc_all['L'][1]
bsc_all['L_XY'][0,1]
bsc_list[0]

bsc_all['r'][0,2]
bsc_all['r_sm'][0,2]
bsc_all['L'][0]
bsc_all['L'][2]
bsc_all['L_XY'][0,2]
bsc_list[1]
len(bsc_list)
# %%
# CPU with multiprocessing
# df
# len(gene_names)
# features = df[gene_names].values
# # features.shape
# with multiprocessing.Pool(processes=16) as pool:

# %%
# GPU


# %%
bsc = calculate_bsc('Ttr', 'Ecrg4', adata, conn_info)

# %% codecell

# %% codecell

# len(list(it.combinations(gene_names, 2)))
# %% codecell
import itertools
gene_names = adata.var.index.tolist()

# for gene1, gene2 in itertools.combinations(gene_names, 2):
for gene1, gene2 in it.combinations(gene_names, 2):
    print(gene1, gene2)
# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
# Checking validity of np.corrcoef,
# Checking validity of np.cov
X = np.random.rand(5, 3) # 3 variables, 5 observations
X.shape

(X-X.mean(axis=0)).T @ (X-X.mean(axis=0))/
m=X.mean(axis=0)
np.cov(X.T)
(X.T@X/len(X) - m[...,None]@m[None,...])*len(X)/(len(X)-1) # consistent

# taking the relevant value from the matrix returned by np.cov
print('Correlation: ' + str(np.cov(X,Y)[0,1]/(np.std(X,ddof=1)*np.std(Y,ddof=1))))
# Let's also use the builtin correlation function
print('Built-in Correlation: ' + str(np.corrcoef(X, Y)[0, 1]))

# Checking validity of np.corrcoef


# x_values=features[:,0]
# y_values=features[:,1]
# x_mean=x_values.mean()
# y_mean=y_values.mean()
# x_values
# y_values
# (x_values*y_values).mean() - x_values.mean()*y_values.mean()
# ((x_values-x_values.mean())*(y_values-y_values.mean())).mean()
# np.cov(x_values, y_values)[0,1]
#
# np.sqrt(((x_values-x_values.mean())**2).sum())
# ((x_values-x_values.mean())**2).sum()
# np.std(x_values)
# cov
#
# r1 = sum((x_values - x_mean) * (y_values - y_mean)) / (np.sqrt(sum((x_values - x_mean) ** 2)) * np.sqrt(sum((y_values - y_mean) ** 2)))
# r2 = ((x_values - x_mean) * (y_values - y_mean)).mean() / (np.std(x_values)*np.std(y_values))
# r3 = ((x_values - x_mean) * (y_values - y_mean)).mean() / (np.std(x_values, ddof=1)*np.std(y_values, ddof=1))
# d
# r
# features.T @ features
# features.shape
#
# cov = np.cov(features.T, ddof=1)
#
# r1
# r2
# r3
# cov[0,1]
# np.diag(cov)
# corrcoef[0,1]
# corrcoef
# cov
# np.isclose(cov, corrcoef).all()
#
# # %%

# # %%
#
#
# cov_diag = np.diag(cov)
# cov * np.sqrt(cov_diag[...,None]@cov_diag[None,...])
# corrcoef
# # np.arange(9).reshape(3,3) * np.arange(3)[None,...]
# # np.arange(9).reshape(3,3) * np.arange(3)[...,None]
#
# cov = np.cov(features.T)
# corrcoef = np.corrcoef(features.T)
# corrcoef = np.corrcoef(features, rowvar=False)
# corrcoef
# corrcoef.shape
# cov
# features.shape
# cov=np.cov(features.T)
# cov.shape
# cov
# (d.T @ d)
# (d_smooth.T @ d_smooth)
#
# rsd.shape
# d.shape
# d[:,0]*d[:,1]
#
# d_sum
# d_smooth_sum
#
#
# rsd.shape
# d.shape
# d[:,:2]
#
# rsd
# # features-mean
# # np.sqrt((d**2).sum(0))
# # (features-mean == features-mean[None,...]).all()
#
# features.shape
# mean.shape
# # mean = features.mean(axis=0).shape
# # W
# # features[:,0].shape
# # np.dot(W, features[:,0])
# # W @ features[:,0]
# # (W @ features)[:,0]
# # (W @ features)[:,0].shape
# # (np.dot(W, features[:,0]) == (W @ features)[:,0]).all()
# # v=np.dot(W, features[:,0]) == (W @ features)[:,0]
# # np.dot(W, features[:,0])[~v]
# # (W @ features)[:,0][~v]
# # np.dot(W, features[:,0])[~v][0]
# # (W @ features)[:,0][~v][0]
# # np.dot(W, features[:,0])[~v] == (W @ features)[:,0][~v]
# # np.isclose(np.dot(W, features[:,0]), (W @ features)[:,0]).all() # floating point precision error
