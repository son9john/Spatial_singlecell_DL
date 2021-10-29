# %% codecell
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finding Structural Simiarity
Created on Sun Sep 12 23:47:57 2021

@author: Junho John Song
"""
import itertools as it
import logging
from tqdm import tqdm
import multiprocessing
import pprint

import more_itertools as mit
import hydra
import scanpy as sc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
from skimage.metrics import structural_similarity as ssim
import torch

import os
# PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
# os.chdir(PROJECT_DIR)

import tools as T
import eval as E
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

    # %%
    # os.chdir('outputs')

    # %%

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

def get_ssim(x_all):
    n_genes = len(x_all)
    dist = np.zeros((n_genes,n_genes))

    # %%
    # upper triangular (excluding diagonal)
    with multiprocessing.Pool() as pool:
        dist_upper = pool.starmap(ssim, it.combinations(x_all, 2))
        # dist_diag = pool.starmap(ssim, zip(x_all, x_all)) # diagnoal is always 1

    # Put scores
    for (i, j), ssim_score in zip(it.combinations(range(len(x_all)), 2), dist_upper):
        dist[i,j] = ssim_score

    # copy to lower triangular
    dist[np.tril_indices(n_genes)] = dist.T[np.tril_indices(n_genes)]
    # dist[np.diag_indices(n_genes)] = dist_diag
    dist[np.diag_indices(n_genes)] = 1
    return dist

# %%
@hydra.main(config_path='conf', config_name='autoencoder')
def main(cfg: DictConfig) -> None:
    # %%
    print(OmegaConf.to_yaml(cfg))

    # path: current directory
    path = T.Path()
    path.RESULT = 'result'
    path.makedirs()

    # Load data
    data = hydra.utils.instantiate(cfg.data.dataset)
    x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).squeeze(1).numpy()

    if cfg.augmented:
        dataset_d = data['info']['dataset_augmented_d']
        gene_names_augmented_d = data['info']['gene_names_augmented_d']

        for dataset_type in dataset_d.keys():
            log.info(f'dataset_type: {dataset_type}')
            dataset, gene_names = dataset_d[dataset_type], gene_names_augmented_d[dataset_type]
            x_augmented = torch.stack([d for d in dataset], axis=0).squeeze(1).numpy()

            path.RESULT.AUGMENT = dataset_type
            path.RESULT.AUGMENT.makedirs()
            data_type = {
            'info': {'gene_names': gene_names}
            }

            # dist = get_ssim(x_augmented.squeeze(1))
            dist = get_ssim(x_augmented)
            dist = -dist

            scores = E.score_jaccard_precision_plot(cfg=cfg.scorer_augment.cfg, x_all=x_augmented, data=data_type, path=path.RESULT.AUGMENT, dist=dist)
            log.info(f'score: {scores}')
            T.save_pickle(scores, path.RESULT.AUGMENT.join('result_augment.p'))

    # Test ssim
    dist = get_ssim(x_all)

    '''
    For Junho
    '''
    if False:
        # %%
        # tidy up ssim results
        gene_names = data['info']['gene_names']
        ssim_gene = pd.DataFrame(dist, index=gene_names, columns=gene_names)

        # If you want to change gene targets
        cfg.scorer.cfg.target_gene = 'Vxn'
        cfg.scorer.cfg.target_gene = ['Vxn', 'Ttr']

    # Invert distance
    dist = -dist

    # %%
    # Evaluate
    scores = E.score_jaccard_precision_plot(cfg=cfg.scorer.cfg, x_all=x_all, data=data, path=path.RESULT, dist=dist)

    cfg.scorer.cfg.target_gene
    pp = pprint.PrettyPrinter()
    log.info('data.info:\n'+pp.pformat(scores))
    T.write(str(pp.pformat(scores)), 'result.txt')

# %%
if __name__=='__main__':
    main()
