import os
import logging
import pprint

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import torchvision.transforms as transforms

import tools as T
import tools.sklearn

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
    import hydra
    from omegaconf import OmegaConf as OC

    PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
    os.chdir(PROJECT_DIR)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
    overrides = []
    cfg = hydra.compose(config_name='autoencoder', overrides=overrides)
    print(OC.to_yaml(cfg))

    # %%
    cfg = cfg.data.dataset.cfg

    # %%
    data = hydra.utils.instantiate(cfg.data.dataset)
    x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).squeeze(1).numpy()

    # %%
    data.keys()
    p_l = [T.load_pickle(path) for path in T.os.listdir(cfg.path.augmented, join=True)]
    for p in p_l:
        print(p.shape)
    p_l[0].shape

    import matplotlib.pyplot as plt
    for p in p_l:
        for i in range(5):
            p_img = p[i]
            fig, axes = plt.subplots(nrows=2)
            for p_img_, ax in zip(p_img, axes):
                ax.imshow(p_img_)
    i=0
    os.listdir(cfg.path.augmented)

# %%

# %%
# Refine data
'''
df.index: observation sites
df.columns: gene names + row,col index
df.values: expression level
'''
def get_gene_map(adata):
    gene_names = adata.var.index.tolist()
    df_gene = pd.DataFrame(data=adata.X, columns=gene_names, index=adata.obs.index)
    df_gene['row'] = adata.obs['array_row'].values
    df_gene['col'] = adata.obs['array_col'].values

    scalers, gene_maps = [], [] # store scalers just in case
    for gene_name in gene_names:
        gene_map, scaler = make_gene_map(df_gene, gene_name)
        gene_maps.append(gene_map), scalers.append(scaler)
    gene_maps = np.stack(gene_maps, axis=0)

    data = {
    'gene_names': gene_names,
    'df_gene': df_gene,
    'gene_maps': gene_maps,
    'scalers': scalers,
    }
    return data

# Make genemap
def make_gene_map(df, gene_name):
    mmscaler = MinMaxScaler()
    gm = df[[gene_name]+['row','col']].pivot('row','col').values
    gm = mmscaler.fit_transform(gm)
    gm = np.nan_to_num(gm)
    return gm, mmscaler

# %%
def get_autoencoder_data(cfg):
    adata = sc.read_h5ad(cfg.path)

    # Refine data
    data = get_gene_map(adata)
    gene_names, df_gene, gene_maps, scalers = data['gene_names'], data['df_gene'], data['gene_maps'], data['scalers']

    # train-test split
    train_i, test_i = T.sklearn.model_selection.train_test_split_i(gene_maps, test_size=cfg.test_size, random_state=cfg.seed)
    gene_maps_train, gene_maps_test = gene_maps[train_i], gene_maps[test_i]

    # Wrap in dataset
    dataset_train, dataset_test = GeneImageDataset(data=gene_maps_train, cfg=cfg.cfg_dataset), GeneImageDataset(data=gene_maps_test, cfg=cfg.cfg_dataset)

    # info
    sample = dataset_train[0]
    in_channels = sample.shape[0]
    input_shape = sample.shape

    info_basic = {
        'in_channels': in_channels,
        'input_shape': input_shape,
    }
    pp = pprint.PrettyPrinter()
    log.info('data.info:\n'+pp.pformat(info_basic)) # print basic information about data

    dataset_all = GeneImageDataset(data=gene_maps, cfg=cfg.cfg_dataset)
    dataset_augmented_d, gene_names_augmented_d = get_augmented_data(cfg)

    info = {
        **info_basic,
        'gene_names': gene_names,
        'df_gene': df_gene,
        'dataset_all': dataset_all,
        'dataset_augmented_d': dataset_augmented_d,
        'gene_names_augmented_d': gene_names_augmented_d,
        'adata': adata,
    }

    data = {
        'dataset': {
            'train': dataset_train,
            'test': dataset_test,
        },
        'split': {
            'train': train_i,
            'test': test_i,
        },
        'info': info,
    }
    return data

# %%
def get_augmented_data(cfg):
    data, gene_names_augmented_d = {}, {}
    for data_type, path in cfg.augmented.items():
        imgs = T.load_pickle(path) # shape: (100, 2, 64, 128), 2 stands for pairs
        imgs = imgs.reshape(-1, *imgs.shape[2:]) # adjacent data are pairs
        data[data_type] = imgs.astype(np.float32)
        gene_names_augmented_d[data_type] = np.char.add(np.repeat(np.arange(len(imgs)//2),2).astype(str), np.tile(['A','B'], len(imgs)//2))
    dataset_augmented_d = {data_type: GeneImageDataset(data=imgs, cfg=cfg.cfg_dataset) for data_type, imgs in data.items()}
    return dataset_augmented_d, gene_names_augmented_d

# %%
# Define dataset
class GeneImageDataset(D.Dataset):
    def __init__(self, data, cfg):
        print(cfg.width, cfg.height)
        self.transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Resize((cfg.height, cfg.width)),
            ]
        )
        self.data = data

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    pass
    # df_gene['row'].min()
    # df_gene[['St18']+['row','col']].pivot('row','col')
