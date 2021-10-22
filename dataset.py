
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    return gene_maps, scalers

# %%
# Make genemap
def make_gene_map(df, gene_name):
    mmscaler = MinMaxScaler()
    gm = df[[gene_name]+['row','col']].pivot('row','col').values
    gm = mmscaler.fit_transform(gm)
    gm = np.nan_to_num(gm)
    return gm, mmscaler
