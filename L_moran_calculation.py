
PROJECT_DIR = '/home/jaesungyoo/spatial_gene'

# %%
import os
import itertools as it

os.chdir(PROJECT_DIR)

import hydra
import pandas as pd
from omegaconf import OmegaConf as OC

# %%
# Load config
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='L_moran_calculation', overrides=overrides)
print(OC.to_yaml(cfg))

# %%
# Load data
positions=pd.read_csv(cfg.path.positions, header=None, index_col=0)
barcodes=pd.read_csv(cfg.path.barcodes, sep='\t', header=None)
features=pd.read_csv(cfg.path.features, sep='\t', header=None)
read_cnts=pd.read_csv(cfg.path.matrix, sep=' ')

# %%
# Refine data

# positions
columns = ['in_tissue', 'row', 'col', 'prow', 'pcol']
positions.rename(columns={c_old:c_new for c_old, c_new in zip(positions.columns, columns)}, inplace=True)

# read_cnts
columns=['feature', 'barcode', 'count']
read_cnts.rename(columns={c_old:c_new for c_old, c_new in zip(read_cnts.columns, columns)}, inplace=True)
read_cnts.drop(0, inplace=True)

dtypes_d = {'feature': int, 'barcode': int, 'count': int}
read_cnts = read_cnts.astype(dtypes_d)

# %%
# Data summary
print(f'[feature - read_cnt]')
print(f'[nunique: {read_cnts.feature.nunique()}]')
print(f"[min: {read_cnts['feature'].min()}][max: {read_cnts['feature'].max()}]")
print(f'[feature - feature]')
print(f'[{len(features)}]')

print(f'[barcode - read_cnt]')
print(f'[nunique: {read_cnts.barcode.nunique()}]')
print(f"[min: {read_cnts['barcode'].min()}][max: {read_cnts['barcode'].max()}]")
print(f'[barcode - barcode]')
print(f'[{len(barcodes)}]')

# %%
positions
features
barcodes
read_cnts

# %%
positions[positions['in_tissue']==1]
barcodes
features
