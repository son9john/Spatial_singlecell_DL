# %% codecell
inChannel = 1
x=dataset_train[0]
x, y = x.shape[1], x.shape[2]
input_img = tf.keras.layers.Input(shape = (x, y, inChannel))

# %% codecell
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))

        self.ct1 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct2 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct3 = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct3 = layers.Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides=(2, 2))

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)

        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        return x

# %%
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
autoencoder = tf.keras.Model(input_img, autoencoder(input_img))
autoencoder = AutoEncoder()
autoencoder.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate=0.0003))
# %% codecell
autoencoder.summary()
# %% codecell
train_X=[d.numpy() for d in dataset_train]
train_X = np.stack(train_X, axis=0)
test_X=[d.numpy() for d in dataset_test]
test_X = np.stack(test_X, axis=0)

# %%
autoencoder_train = autoencoder.fit(train_X,
                                    train_X,
                                    batch_size=cfg.train.batch_size,
                                    epochs=cfg.train.epoch,
                                    verbose=1,
                                    validation_data=(test_X, test_X))
# %%
autoencoder.summary()

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
x_train, x_hat_train, z_train = x_train.cpu().numpy(), x_hat_train.cpu().numpy(), z_train.cpu().numpy()
x_test, x_hat_test, z_test = x_test.cpu().numpy(), x_hat_test.cpu().numpy(), z_test.cpu().numpy()
x_all, x_hat_all, z_all = x_all.cpu().numpy(), x_hat_all.cpu().numpy(), z_all.cpu().numpy()

# %%

    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from umap import UMAP

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
