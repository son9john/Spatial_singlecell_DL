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

# PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
# PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
# os.chdir(PROJECT_DIR)

import node as N
import utils as U
import eval as E
import tools as T

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
    # Load config
    os.getcwd()
    PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
    # PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
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
    overrides = ['criterion=bce_loss', 'train.epoch=0']
    # overrides = ['criterion=bce_loss', 'train.epoch=100']
    # overrides = ['criterion=bce_loss', 'train.epoch=200', 'encoder=cnn_bn', 'decoder=upcnn_bn', 'channel_list=c256_6', 'scorer.cfg.plot=True', 'scorer.cfg.save_x=True']
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

    # kwargs = {'model': model, 'dataset': dataset_train, 'cv': None, 'cfg_train': cfg.train,
    #         'criterion': criterion, 'MODEL_DIR': path.MODEL, 'NODE_DIR': path.NODE.join('default'), 'name': 'server', 'verbose': True, 'amp': True}
    kwargs = {'model': model, 'dataset': dataset_train, 'cv': None, 'cfg_train': cfg.train,
            'criterion': criterion, 'MODEL_DIR': path.MODEL, 'NODE_DIR': path.NODE.join('default'), 'name': 'server', 'verbose': False, 'amp': True}
    node = N.AENode(**kwargs)

    # %%
    node.model.to(device)
    node.step(no_val=True)

    # %%
    # Evaluate
    # %%
    if cfg.augmented:
        node.model.to(device)
        result_d = hydra.utils.instantiate(cfg.eval_augment, model, data)
        node.model.cpu()
        score = hydra.utils.instantiate(cfg.scorer_augment, result_d=result_d, data=data, path=path.RESULT.join('augmented'))
        log.info(f'score: {score}')
        T.save_pickle(score, path.RESULT.join('result_augment.p'))

    # %%
    node.model.to(device)
    result = hydra.utils.instantiate(cfg.eval, model, data)
    node.model.cpu()
    score = hydra.utils.instantiate(cfg.scorer, result=result, data=data, path=path.RESULT)
    log.info(f'score: {score}')
    E.save_score(score, path.RESULT)

    # %%

    # %%

if __name__=='__main__':
    main()
