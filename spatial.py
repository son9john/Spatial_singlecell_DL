
#%%
# %pip install -U numpy
# !pip intall numpy --upgrade  
import anndata as an
import matplotlib as mpl
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanorama
import scanpy as sc
import seaborn as sns
import scanpy as sc
import squidpy as sq

#%%




#%%

#sc.logging.print_versions() # gives errror!!
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


#%%
adata_anterior = sc.datasets.visium_sge(
    sample_id="V1_Mouse_Brain_Sagittal_Anterior"
)
adata_posterior = sc.datasets.visium_sge(
    sample_id="V1_Mouse_Brain_Sagittal_Posterior"
)

#%%



#%%
adata_anterior.var_names_make_unique()
adata_posterior.var_names_make_unique()


#%%
# merge into one dataset
adata = adata_anterior.concatenate(
    adata_posterior,
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]
)

adata = adata_anterior

#%%
adata.uns['spatial']['V1_Mouse_Brain_Sagittal_Anterior']['images']['hires']


#%%
# sc.pl.spatial(adata, color=)


# sc.pl.spatial(adata[adata.obs,:], color = ["Ttr", "Hpca"])


#%%
# add info on mitochondrial and hemoglobin genes to the objects.
adata.var['mt'] = adata.var_names.str.startswith('mt-') 
adata.var['hb'] = adata.var_names.str.contains(("^Hb.*-"))

sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','hb'], percent_top=None, log1p=False, inplace=True)

#%%
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_hb'],
             jitter=0.4, rotation= 45)



#%%
# need to plot the two sections separately and specify the library_id

# for library in ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]:
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb'])


#%%
# need to plot the two sections separately and specify the library_id

sc.pl.spatial(adata, color = ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb'])


#%%

keep = (adata.obs['pct_counts_hb'] < 20) & (adata.obs['pct_counts_mt'] < 25) & (adata.obs['n_genes_by_counts'] > 1000)
print(sum(keep))

adata = adata[keep,:]


#%%
sc.pl.highest_expr_genes(adata, n_top=20)



# %%
# %%

#%%

mito_genes = adata.var_names.str.startswith('mt-')
hb_genes = adata.var_names.str.contains('^Hb.*-')

remove = np.add(mito_genes, hb_genes)
remove[adata.var_names == "Bc1"] = True
keep = np.invert(remove)
print(sum(remove))

adata = adata[:,keep]

print(adata.n_obs, adata.n_vars)


#%%
adara = adata

#%%

adata = adara

#%%
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

# take 1500 variable genes per batch and then use the union of them.
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1500, inplace=True)

# subset for variable genes
adata.raw = adata

adata = adata[:,adata.var.highly_variable > 0]

# scale data
sc.pp.scale(adata)

adata.shape
#%%

def con_mat(sdata):
    positions_in_tissue = sdata.obs[sdata.obs.columns[:3]] 

    return 0





#%%
#%%
#%%
#%%

#%%
# for i in genes:
#     sc.pl.spatial(adata, color=i, save='genes'+str(i)+'.png')
import os
os.makedirs('data/results/', exist_ok=True)

save_file = 'data/results/large_dada2.h5ad'
adata.write_h5ad(save_file)

#%%

# for library in ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]:
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["Ttr", "Hpca"])
sc.pl.spatial(adata[adata.obs,:], color = ["Ttr", "Hpca"])


#%%

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")


#%%

sc.pl.umap(
    adata, color=["clusters"], palette=sc.pl.palettes.default_20
)


#%%

clusters_colors = dict(
    zip([str(i) for i in range(len(adata.obs.clusters.cat.categories))], adata.uns["clusters_colors"])
)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
# two images are going to be presented in a row, but we only have 1 image.

for i in range(1):
# enumerate(
#     ["V1_Mouse_Brain_Sagittal_Anterior"])
    ad = adata
    sc.pl.spatial(
        ad,
        img_key="hires",
        color="clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist()
        ],
        legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()

#%%

adata.obs.index.to_list()
adata.obs.columns.to_list()

#%%
batches = ["V1_Mouse_Brain_Sagittal_Anterior"]
adatas = {}
for batch in batches:
    adatas[batch] = adata[adata.obs['V1_Mouse_Brain_Sagittal_Anterior'] == batch,]

adatas 

#%%
# scanorama only works with more than 1 imgs.
import scanorama

#convert to list of AnnData objects
adatas = list(adatas.values())

# run scanorama.integrate
scanorama.integrate_scanpy(adatas, dimred = 50)


# Get all the integrated matrices.
scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

# make into one matrix.
all_s = np.concatenate(scanorama_int)
print(all_s.shape)

# add to the AnnData object
adata.obsm["Scanorama"] = all_s


#%%


#%%
sc.pl.spatial(adata, color=["Itpka"])


#%%
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")


#%%
sc.pl.umap(
    adata, color=["clusters"], palette=sc.pl.palettes.default_20
)



#%%
clusters_colors = dict(
    zip([str(i) for i in range(len(adata.obs.clusters.cat.categories))], adata.uns["clusters_colors"])
)


fig, axs = plt.subplots(1, 2, figsize=(15, 10))

for i  in range(1):
    ad = adata
    sc.pl.spatial(
        ad,
        img_key="hires",
        color="clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist()
        ],
        legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()


#%%
# run t-test 
sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon")
# plot as heatmap for cluster5 genes
sc.pl.rank_genes_groups_heatmap(adata, groups="5", n_genes=10, groupby="clusters")


#%%
# plot onto spatial location
top_genes = sc.get.rank_genes_groups_df(adata, group='5',log2fc_min=0)['names'][:3]


# for library in ["V1_Mouse_Brain_Sagittal_Anterior"]:
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = top_genes)
sc.pl.spatial(adata, color = top_genes)
#%%
# !pip install --upgrade pip
# !pip install -U numpy
# !pip install SpatialDE
# !pip install spatialde
import SpatialDE

#First, we convert normalized counts and coordinates to pandas dataframe, needed for inputs to spatialDE.
counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
results = SpatialDE.run(coord, counts)

#We concatenate the results with the DataFrame of annotations of variables: `adata.var`.
results.index = results["g"]
adata.var = pd.concat([adata.var, results.loc[adata.var.index.values, :]], axis=1)

#Then we can inspect significant genes that varies in space and visualize them with `sc.pl.spatial` function.
results.sort_values("qval").head(10)


#%%











#%%

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%
## Moran
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from numpy.random import default_rng

#%%




#%%

#%%
# sc.logging.print_header()
# print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
# img = sq.datasets.visium_hne_image()
# adata = sq.datasets.visium_hne_adata()
img = sq.im.ImageContainer("/home/single_cell/mousebrain/spatial/tissue_hires_image.png")
img.show()


#%%
import matplotlib.image as mp_image
_img = mp_image.imread("/home/single_cell/mousebrain/spatial/tissue_hires_image.png")
_img.shape
#%%

img = adata.uns['spatial']['V1_Mouse_Brain_Sagittal_Anterior']['images']['hires']

#%%
img = sq.im.ImageContainer(img)
img.show()

#%%

# calculate features for different scales (higher value means more context)

for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata,
        img,
        features="summary",
        key_added=feature_name,
        n_jobs=1,
        scale=scale

    )


# combine features in one dataframe
adata.obsm["features"] = pd.concat(
    [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f], axis="columns"
)
# make sure that we have no duplicated feature names in the combined table
adata.obsm["features"].columns = ad.utils.make_index_unique(adata.obsm["features"].columns)


#%%
# img3 = sq.datasets.visium_hne_image()
# adata3 = sq.datasets.visium_hne_adata()

#%%

adata_backup2 = adata
genes = adata[:, adata.var.highly_variable].var_names.values


# for i in genes:
#     sc.pl.spatial(adata, color=i, save='genes'+str(i)+'.png')


#%%
import pickle

open_file = open('genes_list.txt', 'wb')
pickle.dump(genes, open_file)
open_file.close()


#%%
adata
#%%
sc.pl.spatial(adata, library_id = "V1_Mouse_Brain_Sagittal_Anterior", color="clusters")


#%%
adata.uns['spatial']['V1_Mouse_Brain_Sagittal_Anterior']['images']['hires']

plt.imshow(adata.uns['spatial']['V1_Mouse_Brain_Sagittal_Anterior']['images']['hires'])
plt.show()

#%%
{'spatial': {'V1_Mouse_Brain_Sagittal_Anterior': adata.uns['spatial']['V1_Mouse_Brain_Sagittal_Anterior']}}


#%%
library = 'V1_Mouse_Brain_Sagittal_Anterior'

# adata_backup = adata
adata = adata[adata.obs.library_id == library,:]
sc.pl.spatial(adata, library_id=library, color = "clusters")

#%%
# calculate features for different scales (higher value means more context)
for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata, 
        img, 
        library_id=library,
        features="summary",
        key_added=feature_name,
        n_jobs=1,
        scale=scale,
    )


# combine features in one dataframe
adata.obsm["features"] = pd.concat(
    [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f], axis="columns"
)
# make sure that we have no duplicated feature names in the combined table
adata.obsm["features"].columns = ad.utils.make_index_unique(adata.obsm["features"].columns)


#%%
adata_backup2 = adata
genes = adata[:, adata.var.highly_variable].var_names.values

# genes = adata.var_names.values

sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="clusters")
sq.pl.nhood_enrichment(adata, cluster_key="clusters")

#%%
import inspect
print(inspect.getsource(sq.gr.spatial_autocorr))

#%%
sq.gr.spatial_autocorr(
    adata,
    genes=genes,
)

sq.gr.spatial_autocorr(
    adata,
    genes=genes,
    mode='geary'
)


#%%
adata.uns["moranI"].head(10)
#%%
adata.uns["gearyC"].head(10)

#%%
sc.pl.spatial(adata, color=["Gm48086", "Plp1", "Itpka", "clusters"])

#%%
moranI = adata.uns["moranI"]


#%%
# moranI
# moranI[moranI['pval_norm']==0]
moranI.loc['Vxn']

#%%

moranI.iloc[:,[0]]

#%%


#%%



#%%


#%%



#%%

