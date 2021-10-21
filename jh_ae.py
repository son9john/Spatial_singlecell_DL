# %% codecell
from glob import glob
import gzip
import os
from PIL import Image

import hydra
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf as OC
import pandas as pd
import scanpy as sc
from scipy import misc
from scipy.spatial import distance
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ZeroPadding2D
# %%
# Load config
os.getcwd()
# PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
os.chdir(PROJECT_DIR)
os.listdir()

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
overrides = []
cfg = hydra.compose(config_name='l_regression', overrides=overrides)
print(OC.to_yaml(cfg))

# %% codecell
# ## Load data
adata = sc.read_h5ad(cfg.path.data)

# Refine data
gene_maps, scalers = DAT.get_gene_map(adata)


gene_names = adda.var.index.tolist()
row_col = adda.obs[['array_row', 'array_col']].values.astype(int)
df = pd.DataFrame(data=np.concatenate((row_col, adda.X), axis=1), columns=['row', 'col'] + gene_names)
df['row'] = df['row'].astype(int)
df['col'] = df['col'].astype(int)

RESIZE_WIDTH = 128
RESIZE_HEIGHT = 64

def resize_gene_maps(gene_map):
    img = Image.fromarray(np.uint8(gene_map * 255) , 'L')
    img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
    img = np.asarray(img) / 255
    return img

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
    gm = gm[1:-1]
    gm = np.concatenate((np.zeros([RESIZE_HEIGHT, RESIZE_WIDTH - gm.shape[1]]), gm), axis=1)
    return gm

# row_len = row_col[:, 0].max() - row_col[:, 0].min() + 1
# col_len = row_col[:, 1].max() - row_col[:, 1].min() + 1
gene_maps = np.zeros([len(gene_names), RESIZE_HEIGHT, RESIZE_WIDTH], dtype=np.float32)
for i, name in enumerate(gene_names):
    gene_maps[i] = make_gene_map(df, name)
# %% codecell
gene_maps.max()
# %% codecell

# %% codecell
data = gene_maps
data1 = gene_maps[gene_names.index('Vxn')]

plt.imshow(data1)
plt.show()
# %% codecell
images = gene_maps
images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')

images_arr = np.expand_dims(images_arr, -1)

# %% codecell
images_arr.shape
# %% codecell
train_X, valid_X, train_ground, valid_ground = train_test_split(images_arr,
                                                             images_arr,
                                                             test_size=0.2,
                                                             random_state=13)
# %% codecell
train_X.shape
# %% codecell
batch_size = 128
epochs = 100

inChannel = 1
x, y = train_X.shape[1], train_X.shape[2]
input_img = Input(shape = (x, y, inChannel))
# %% codecell
def autoencoder(cropped):
    #encoder
    h = Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))(cropped)
    h = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)

    # decoder
    h = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    return h
# %% codecell
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.0003))
# %% codecell
autoencoder.summary()
# %% codecell
autoencoder_train = autoencoder.fit(train_X,
                                    train_ground,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(valid_X, valid_ground))
# %% codecell
# %%
pred = autoencoder.predict(valid_X)


# %%

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(valid_ground[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')
plt.show()
# %% codecell

# %% codecell
intermediate_layer_model = Model(inputs=autoencoder.input,
                                       outputs=autoencoder.get_layer('conv2d_65').output)
# %% codecell
x = np.expand_dims(gene_maps[1], [0, -1])
print(x.shape)
feat = intermediate_layer_model.predict(x)
# %% codecell
plt.imshow(x[0, :, :, 0])
plt.show()
# %% codecell
for i in range(32):
    plt.imshow(feat[0, :, :, i])
    plt.show()
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
x = np.expand_dims(gene_maps, -1)
all_feats = intermediate_layer_model.predict(x)
# %% codecell
all_feats = all_feats.reshape(all_feats.shape[0], -1)
# %% codecell
def find_similar_genes_consine(target_feat, all_features, top_k=10):
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
target_gene_name = 'Ttr'
target_gene_idx = gene_names.index(target_gene_name)
top10_genes = find_similar_genes(all_feats[target_gene_idx], all_feats)

sc.pl.spatial(adda, color=[target_gene_name] + [gene_names[i] for i in top10_genes])
# %% codecell
for idx in top10_genes:
    plt.imshow(gene_maps[idx])
    plt.show()
# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
