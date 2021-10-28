import os
import logging

import multiprocessing
import torch

import tools as T
import tools.torch
import tools.random
from omegaconf import OmegaConf as OC

log = logging.getLogger(__name__)

# %%
def exp_path(path=None, makedirs=True):
    path = '.' if path is None else path
    exp_path = T.Path(path)
    exp_path.MODEL = 'model' # Save model while training
    exp_path.RESULT = 'result' # Training results
    exp_path.NODE = 'node' # Save node info & model. Package to be communicated
    if makedirs:
        exp_path.makedirs()
    return exp_path

# def seed(random_seed, strict=False):
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     if strict:
#         torch.cuda.manual_seed(random_seed)
#         torch.cuda.manual_seed_all(random_seed)
#         random.seed(random_seed)
#         log.info('[seed] Strict random seed')
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

def exp_setting(cfg, path=None):
    # Print current experiment info
    log.info(OC.to_yaml(cfg))
    log.info(os.getcwd())

    # Set GPU for current experiment
    device = T.torch.multiprocessing_device(cfg.gpu_id)
    # device = multiprocessing_device(cfg.gpu_id)
    log.info(device)

    T.random.seed(cfg.random.seed, strict=cfg.random.strict)
    path = exp_path(path)
    return device, path
