# %% codecell
from glob import glob
import gzip
import os
from PIL import Image
import logging

import hydra
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf as OC
from omegaconf import DictConfig
import pandas as pd
import scanpy as sc
from scipy import misc
from scipy.spatial import distance
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import torchvision.transforms as transforms

PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)

import node as N
import utils as U
import eval as E

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
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
    # %%
    device, path = U.exp_setting(cfg)

    # Load data
    data = hydra.utils.instantiate(cfg.data.dataset)
    dataset_train, dataset_test = data['dataset']['train'], data['dataset']['test']

    # criterion
    criterion = hydra.utils.instantiate(cfg.criterion)
    log.info(f'criterion: {criterion}')

    # %%
    # load model
    model = hydra.utils.instantiate(cfg.model,  data['info'])
    log.info(model)

    kwargs = {'model': model, 'dataset': dataset_train, 'cv': None, 'cfg_train': cfg.train,
            'criterion': criterion, 'MODEL_DIR': path.MODEL, 'NODE_DIR': path.NODE.join('default'), 'name': 'server', 'verbose': True, 'amp': True}
    node = N.AENode(**kwargs)

    # %%
    node.model.to(device)
    node.step(no_val=True)
    # node.save(path.Node)

    # %%
    # Evaluate
    node.model.to(device)
    result = hydra.utils.instantiate(cfg.eval, model, data)
    score = hydra.utils.instantiate(cfg.scorer, result)
    E.save_score(score, path.RESULT)

    # %%
    with torch.no_grad():
        x_train = torch.stack([d for d in dataset_train], axis=0).to(device)
        z_train = node.model.encoder(x_train).flatten(1)
        x_hat_train = node.model(x_train)

        x_test = torch.stack([d for d in dataset_test], axis=0).to(device)
        z_test = node.model.encoder(x_test).flatten(1)
        x_hat_test = node.model(x_test)

        x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).to(device)
        z_all = node.model.encoder(x_all).flatten(1)
        x_hat_all = node.model(x_all)

    # %%
    import tools as T
    import tools.torch.plot

    for i in range(10):
        fig, ax = T.torch.plot.imshow(x_train[i])
        fig, ax = T.torch.plot.imshow(x_hat_train[i])

    # %%
    x_train, x_hat_train, z_train = x_train.cpu().numpy(), x_hat_train.cpu().numpy(), z_train.cpu().numpy()
    x_test, x_hat_test, z_test = x_test.cpu().numpy(), x_hat_test.cpu().numpy(), z_test.cpu().numpy()
    x_all, x_hat_all, z_all = x_all.cpu().numpy(), x_hat_all.cpu().numpy(), z_all.cpu().numpy()

    # %%
    # Plot dimension reductionality of latent variables
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from umap import UMAP

    n_clusters=5
    figsize=(10,10)

    gmm = BayesianGaussianMixture(n_components=n_clusters, random_state=cfg.random.seed)
    # gmm = GaussianMixture(n_components=n_clusters, random_state=cfg.random.seed)

    import pdb; pdb.set_trace()
    c_gmm = gmm.fit(z_train)

    z_d = {'train': z_train, 'test': z_test, 'all': z_all}
    reducers_d = {'pca': PCA(n_components=2), 'tsne': TSNE(n_components=2, n_jobs=10), 'umap': UMAP(n_components=2)}
    for z_type, z in z_d.items():
        log.info(f'z_type: {z_type}')
        path_z_distrib = path.RESULT.join('z_distrib', z_type)
        path_z_distrib.makedirs()

        c_gmm = gmm.predict(z)
        classes = np.unique(c_gmm).tolist()
        # if z_type=='test':
        #     import pdb; pdb.set_trace()

        for reducer_name, reducer in reducers_d.items():
            log.info(f'[Dimensionality reduction]: {reducer_name}')
            z_reduced = reducer.fit_transform(z)

            z_df = pd.DataFrame({'z1': z_reduced[:,0], 'z2': z_reduced[:,1]})
            z_df['cluster'] = c_gmm
            z_df = z_df.astype({'cluster':'category'})

            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=z_df, x='z1', y='z2', ax=ax)
            fig.savefig(path_z_distrib.join(f'all_{reducer_name}.png'))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=z_df, x='z1', y='z2', hue='cluster', ax=ax)
            fig.savefig(path_z_distrib.join(f'cluster_{reducer_name}.png'))
            plt.close(fig)

    # %%
    # Save original data per cluster
    if cfg.save_x:
        path_x = path.RESULT.join('x')
        split_d = {'train': data['split']['train'], 'test': data['split']['test']}

        gene_names = np.array(data['info']['gene_names'])
        c_gmm = gmm.predict(z_all)
        classes = np.unique(c_gmm).tolist()

        # Divide according to train/test set, just in case they have patterns
        for split_type, split_i in split_d.items():
            path_x_split = path_x.join(split_type)
            path_x_split.makedirs()

            x_all_subset, gene_names_subset, c_gmm_subset = x_all[split_i], gene_names[split_i], c_gmm[split_i]

            for c in classes:
                path_x_c = path_x_split.join(str(c))
                path_x_c.makedirs()

                c_i = np.where(c_gmm_subset==c)[0]
                x_c, gene_names_subset_c = x_all_subset[c_i], gene_names_subset[c_i]
                x_c = x_c.squeeze(1)
                for x_img, gene_name in zip(x_c, gene_names_subset_c):
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.imshow(x_img)
                    fig.savefig(path_x_c.join(f'{gene_name}.png'))
                    plt.close(fig)

    # %%
    # Distance sort
    target_gene_list = []
    target_gene
    test_i=data['split']['test']

    # %%

if __name__=='__main__':
    main()

self.model.encoder(x).shape
z=self.model.encoder(x)
x_hat=self.model.decoder(z)
x_hat.shape
# %%
plt.figure(figsize=(20, 4))
print("Test Images")
for i, x_img in enumerate(x_train.squeeze(1)[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_img, cmap='gray')

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i, x_img in enumerate(x_hat_train.squeeze(1)[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_img, cmap='gray')
plt.show()

# %%
# all_feats=z_train
all_feats=z_all

# %% codecell
def find_similar_genes_consine(target_feat, all_features, top_k=10):
    dist = [distance.cosine(target_feat, all_features[i]) for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx

def find_similar_genes_euclidean(target_feat, all_features, top_k=10):
    np.linalg.norm()
    dist = [np.square(target_feat - all_features[i]).sum() for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx

def find_similar_genes_dot_prod(target_feat, all_features, top_k=10):
    # dist = [-1 * (target_feat * all_features[i]).sum() for i in range(all_features.shape[0])]
    dist = [(target_feat * all_features[i]).sum() for i in range(all_features.shape[0])]
    top_k_idx = np.argsort(dist)[1:top_k+1]
    return top_k_idx

# %% codecell
target_gene_name = 'Ttr'
gene_names = adata.var.index.tolist()
target_gene_idx = gene_names.index(target_gene_name)
target_gene_idx

import scanpy as sc

adata = sc.read_h5ad(cfg.path.data)
print(OC.to_yaml(cfg))
top10_genes = find_similar_genes_cosine(all_feats[target_gene_idx], all_feats)
top10_genes = find_similar_genes_euclidean(all_feats[target_gene_idx], all_feats)
top10_genes = find_similar_genes_dot_prod(all_feats[target_gene_idx], all_feats)

sc.pl.spatial(adata, color=[target_gene_name] + [gene_names[i] for i in top10_genes])


# %% codecell
for idx in top10_genes:
    plt.imshow(gene_maps[idx])
    plt.show()
# %% codecell

# %% codecell
