from copy import deepcopy as dcopy
import itertools as it
import logging
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
import torch
import torch.utils.data as D

import tools as T
import tools.torch

# %%
log = logging.getLogger(__name__)

# %%
def reproducible_worker_dict():
    '''Generate separate random number generators for workers,
    so that the global random state is not consumed,
    thereby ensuring reproducibility'''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {'worker_init_fn': seed_worker, 'generator': g}

def eval_autoencoder(model, data, test_batch_size=128):
    train_i, test_i = data['split']['train'], data['split']['test']

    x_all, z_all, x_hat_all = test_autoencoder(model, data['info']['dataset_all'], test_batch_size)
    # x_all, z_all, x_hat_all = x_all.cpu().numpy(), z_all.cpu().numpy(), x_hat_all.cpu().numpy()

    x_train, z_train, x_hat_train = x_all[train_i], z_all[train_i], x_hat_all[train_i]
    x_test, z_test, x_hat_test = x_all[test_i], z_all[test_i], x_hat_all[test_i]

    result = {
    'all': {'x': x_all, 'z': z_all, 'x_hat': x_hat_all},
    'train': {'x': x_train, 'z': z_train, 'x_hat': x_hat_train},
    'test': {'x': x_test, 'z': z_test, 'x_hat': x_hat_test},
    }
    return result

def eval_autoencoder_augment(model, data, test_batch_size=128):
    dataset_d = data['info']['dataset_augmented_d']

    # Test
    result_d = {}
    for dataset_type, dataset in dataset_d.items():
        x_all, z_all, x_hat_all = test_autoencoder(model, dataset, test_batch_size)
        result_d[dataset_type] = {
            'all':{
            'x': x_all,
            'z': z_all,
            'x_hat': x_hat_all,
            },
        }

    return result_d

def test_autoencoder(model, dataset, test_batch_size=128):
    device = T.torch.get_device(model)

    model.eval()
    with torch.no_grad():
        # x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0)

        kwargs_dataloader = reproducible_worker_dict()
        loader = D.DataLoader(dataset, batch_size=test_batch_size, shuffle=False, drop_last=False, **kwargs_dataloader)
        x_list, z_list, x_hat_list = [], [], []
        for x in loader:
            x = x.to(device)
            z = model.encoder(x).flatten(1).cpu()
            x_hat = torch.sigmoid(model(x)).cpu()

            x_list.append(x), z_list.append(z), x_hat_list.append(x_hat)

        x_all, z_all, x_hat_all = torch.cat(x_list, dim=0).cpu().numpy().squeeze(1), torch.cat(z_list, dim=0).cpu().numpy(), torch.cat(x_hat_list, dim=0).cpu().numpy().squeeze(1)
    model.train()
    return x_all, z_all, x_hat_all

# %%
def score_autoencoder(cfg, result, data, path):
    log.info('[score_autoencoder]')
    path = dcopy(path)

    # 1. Save sample examples
    log.info('1. Saving sample examples')
    n_samples=5
    figsize_base=4
    figsize=(figsize_base*n_samples, figsize_base)

    x_train, x_hat_train = result['train']['x'], result['train']['x_hat']

    path.samples ='samples'
    path.samples.makedirs()

    fig, axes = plt.subplots(ncols=n_samples, figsize=figsize)
    # for i, (x_img, ax) in enumerate(zip(x_train.squeeze(1)[:n_samples], axes)):
    for i, (x_img, ax) in enumerate(zip(x_train[:n_samples], axes)):
        ax.imshow(x_img, cmap='gray')
    fig.savefig(path.samples.join('original.png'))
    plt.close(fig)

    fig, axes = plt.subplots(ncols=n_samples, figsize=figsize)
    # for i, (x_img, ax) in enumerate(zip(x_hat_train.squeeze(1)[:n_samples], axes)):
    for i, (x_img, ax) in enumerate(zip(x_hat_train[:n_samples], axes)):
        ax.imshow(x_img, cmap='gray')
    fig.savefig(path.samples.join('reconstructed.png'))
    plt.close(fig)

    # 2. Sort & visualize plots, Jaccard & Precision
    log.info('2. Sort & visualize plots, Jaccard & Precision')
    score = jaccard_precision_per_distance(cfg, result, data, path)
    log.info(f'score: {score}')

    # Reconstruction loss for test set (BCE)

    # 3. Other plots
    if cfg.plot:
        log.info('3. (optional) plot distribution of latent variables')
        # Dimension reduction figures
        c_gmm = plot_distribution(cfg, result, data, path)

        # original figures (clusters)
        if cfg.save_x:
            plot_clusters(result, data, path, c_gmm)

    return score

def score_autoencoder_augmented(cfg, result_d, data, path):
    log.info('[score_autoencoder_augmented]')
    path.makedirs()

    dataset_d = data['info']['dataset_augmented_d']
    gene_names_augmented_d = data['info']['gene_names_augmented_d']

    # Test
    # results_d = {}
    # for dataset_type, dataset in dataset_d.items():
    #     x_all, z_all, x_hat_all = test_autoencoder(model, dataset)
    #     results_d[dataset_type] = {
    #     'x': x_all,
    #     'z': z_all,
    #     'x_hat': x_hat_all,
    #     }

    # evaluate
    score_d = {}
    # for dataset_type, result in result_d.items():
    for dataset_type in result_d.keys():
        log.info(f'dataset_type: {dataset_type}')
        result, gene_names = result_d[dataset_type], gene_names_augmented_d[dataset_type]
        # result['x'] = result['x'].squeeze(1)
        result_type = {'all': {}}

        path_dataset_type = path.join(dataset_type)
        data_type = {
        'info': {'gene_names': gene_names}
        }

        score = jaccard_precision_per_distance(cfg, result, data_type, path_dataset_type)
        score_d[dataset_type] = score

    return score_d

# %%
def jaccard_precision_per_distance(cfg, result, data, path):
    path.makedirs()
    z_all = result['all']['z']
    x_all = result['all']['x']

    path.dist = 'distance'

    # distance measure
    score = {}
    distance_d = {'euclidian': dist_euclidian, 'cosine': dist_cosine}

    for distance_name, distance_f in distance_d.items():
        log.info(f'[distance: {distance_name}]')
        path.dist.type = distance_name
        path.dist.type.makedirs()

        dist = distance_f(z_all) # distance matrix for all pairs
        score[distance_name] = score_jaccard_precision_plot(cfg, x_all, data, path.dist.type, dist)

    return score

def score_jaccard_precision_plot(cfg, x_all, data, path, dist):
    figsize = (6,4)
    nrows, ncols = 3, 4
    top_n = nrows*ncols
    figsize_all = (figsize[0]*nrows, figsize[1]*ncols)
    info = {
    'figsize': figsize,
    'nrows': nrows,
    'ncols': ncols,
    'top_n': top_n,
    'figsize_all': figsize_all,
    }

    # x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).numpy()
    # x_all = result['all']['x']
    score = {}

    if not hasattr(cfg, 'target_gene'):
        pass
    elif type(cfg.target_gene) == str:
        score[cfg.target_gene] = jaccard_precision_plot_target_gene(cfg, x_all, data, path.join(cfg.target_gene), dist, cfg.target_gene, info)
    elif hasattr(cfg.target_gene, '__len__'):
        for target_gene in cfg.target_gene:
            score[target_gene] = jaccard_precision_plot_target_gene(cfg, x_all, data, path.join(target_gene), dist, target_gene, info)
    else:
        raise Exception()

    # Top 10 area under jaccard/precision curve for all genes
    with multiprocessing.Pool(processes=4) as pool:
        result_list = pool.starmap(mean_top_score, zip(range(len(x_all)), it.repeat(x_all), it.repeat(dist), it.repeat(top_n)))
    mean_aujc_list, mean_aupc_list = [r[0] for r in result_list], [r[1] for r in result_list]

    score['all'] = {}
    score['all']['aujc'] = np.mean(mean_aujc_list)
    score['all']['aupc'] = np.mean(mean_aupc_list)
    return score

def jaccard_precision_plot_target_gene(cfg, x_all, data, path, dist, target_gene, info):
    log.info(f'target_gene: {target_gene}')
    path.makedirs()

    figsize = info['figsize']
    nrows, ncols = info['nrows'], info['ncols']
    top_n = info['top_n']
    figsize_all = info['figsize_all']

    score = {}

    gene_names = np.array(data['info']['gene_names'])

    assert target_gene in gene_names, f'target: {target_gene}, gene_names: {gene_names[:10]}'

    # x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).numpy()
    # x_all = result['all']['x']

    target_gene_i = np.where(gene_names==target_gene)[0].item()
    else_i = np.arange(len(gene_names))!=target_gene_i

    x_target, x_else = x_all[target_gene_i], x_all[else_i]
    gene_name_target, gene_names_else = gene_names[target_gene_i], gene_names[else_i]

    dist_v = dist[target_gene_i][else_i]
    dist_i = np.argsort(dist_v)

    # Plot & save
    if cfg.augment:
        plot_aug(x_order=x_target, path=path, savename='gene_target.png', nrows=1, ncols=1, figsize=figsize)
    else:
        adata = data['info']['adata']
        plot_sc(adata, gene_names_order=gene_name_target, path=path, savename='gene_target.png', nrows=1, ncols=1, figsize=figsize)
    # fig, ax = plt.subplots(figsize=figsize)
    # sc.pl.spatial(adata, color=gene_name_target, ax=ax)
    # fig.savefig(path.join('gene_target.png'))
    # plt.close(fig)

    # closest, farthest
    i_close = dist_i[:top_n]
    i_far = dist_i[-top_n:]
    order_d = {'close': i_close, 'far': i_far}

    for order, i_order in order_d.items():

        score[order] = {}

        gene_names_order, x_else_order = gene_names_else[i_order], x_else[i_order]

        # plot & save
        if cfg.augment:
            plot_aug(x_order=x_else_order, path=path, savename=f'gene_else_{order}.png', nrows=nrows, ncols=ncols, figsize=figsize_all)
        else:
            plot_sc(adata=adata, gene_names_order=gene_names_order, path=path, savename=f'gene_else_{order}.png', nrows=nrows, ncols=ncols, figsize=figsize_all)

        # Get Jaccard & Precision score
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize_all)
        jaccards_list, precisions_list = [], []
        for x_else_order_sample, ax in zip(x_else_order, axes.flatten()):
            thresholds, jaccards, precisions = jaccard_precision_curve(x_target, x_else_order_sample)
            jaccards_list.append(jaccards), precisions_list.append(precisions)

            # ax.plot([0,1], [1,0]) # line
            ax.plot(thresholds, jaccards, label='jaccard')
            ax.plot(thresholds, precisions, label='precision')
        ax.legend()
        fig.savefig(path.join(f'jp_curve_{order}.png'))
        plt.close(fig)

        jaccards_else, precisions_else = np.stack(jaccards_list, axis=0), np.stack(precisions_list, axis=0)

        # Mean over top n genes
        jaccards_mean, precisions_mean = jaccards_else.mean(axis=0), precisions_else.mean(axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        # ax.plot([0,1], [1,0]) # line
        ax.plot(thresholds, jaccards_mean, label='jaccard')
        ax.plot(thresholds, precisions_mean, label='precision')
        ax.legend()
        fig.savefig(path.join(f'jp_curve_all_{order}.png'))
        plt.close(fig)

        # metrics
        aujc_else = np.array([metrics.auc(thresholds, jaccard) for jaccard in jaccards_else])
        aupc_else = np.array([metrics.auc(thresholds, precision) for precision in precisions_else])

        aujc, aupc = aujc_else.mean(), aupc_else.mean()

        # Average score on top N genes
        score[order]['aujc'] = aujc
        score[order]['aupc'] = aupc
    return score

def mean_top_score(target_gene_i, x_all, dist, top_n):
    else_i = np.arange(len(x_all))!=target_gene_i
    x_target, x_else = x_all[target_gene_i], x_all[else_i]

    dist_v = dist[target_gene_i][else_i]
    dist_i = np.argsort(dist_v)
    i_close = dist_i[:top_n]
    # x_else_close = x_else[i_close].squeeze(1)
    x_else_close = x_else[i_close]

    jaccards_list, precisions_list = [], []
    for x_else_close_sample in x_else_close:
        thresholds, jaccards, precisions = jaccard_precision_curve(x_target, x_else_close_sample, n_threshold=500)
        jaccards_list.append(jaccards), precisions_list.append(precisions)
    jaccards_else, precisions_else = np.stack(jaccards_list, axis=0), np.stack(precisions_list, axis=0)

    aujc_mean = np.array([metrics.auc(thresholds, jaccard) for jaccard in jaccards_else]).mean()
    aupc_mean = np.array([metrics.auc(thresholds, precision) for precision in precisions_else]).mean()
    return aujc_mean, aupc_mean

def plot_sc(adata, gene_names_order, path, savename, **kwargs):
    nrows, ncols, figsize = kwargs['nrows'], kwargs['ncols'], kwargs['figsize']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if hasattr(axes, '__len__'):
        for gene_name, ax in zip(gene_names_order, axes.flatten()):
            sc.pl.spatial(adata, color=gene_name, ax=ax)
    else:
        sc.pl.spatial(adata, color=gene_names_order, ax=axes)
    fig.savefig(path.join(savename))
    plt.close(fig)

def plot_aug(x_order, path, savename, **kwargs):
    nrows, ncols, figsize = kwargs['nrows'], kwargs['ncols'], kwargs['figsize']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if hasattr(axes, '__len__'):
        for x, ax in zip(x_order, axes.flatten()):
            ax.imshow(x)
    else:
        axes.imshow(x_order)
    fig.savefig(path.join(savename))
    plt.close(fig)

def plot_distribution(cfg, result, data, path):
    # Plot dimension reduction plots of latent variables
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from umap import UMAP

    n_clusters=5
    figsize=(10,10)
    main_reducer = 'umap'
    reducers_d = {'pca': PCA(n_components=2), 'tsne': TSNE(n_components=2, n_jobs=10), 'umap': UMAP(n_components=2)}
    reducer_main = reducers_d[main_reducer]

    z_train = result['train']['z']
    gmm = BayesianGaussianMixture(n_components=n_clusters, random_state=cfg.random_seed)
    # gmm = GaussianMixture(n_components=n_clusters, random_state=cfg.random.seed)

    z_all = result['all']['z']
    z_reduced = reducer_main.fit_transform(z_all)
    c_gmm = gmm.fit_predict(z_reduced)

    slice_d = {'train': data['split']['train'], 'test': data['split']['test'], 'all': slice(None)}
    z_d = {'train': result['train']['z'], 'test': result['test']['z'], 'all': result['all']['z']}
    for (z_type, z), slice_i in zip(z_d.items(), slice_d.values()):
        log.info(f'z_type: {z_type}')
        path_z_distrib = path.join('z_distrib', z_type)
        path_z_distrib.makedirs()

        c_gmm_type = c_gmm[slice_i]
        # c_gmm = gmm.predict(z)
        # if z_type=='test':
        #     import pdb; pdb.set_trace()

        for reducer_name, reducer in reducers_d.items():
            log.info(f'[Dimensionality reduction]: {reducer_name}')
            z_reduced = reducer.fit_transform(z)

            z_df = pd.DataFrame({'z1': z_reduced[:,0], 'z2': z_reduced[:,1]})
            z_df['cluster'] = c_gmm_type
            z_df = z_df.astype({'cluster':'category'})

            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=z_df, x='z1', y='z2', ax=ax)
            fig.savefig(path_z_distrib.join(f'all_{reducer_name}.png'))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=z_df, x='z1', y='z2', hue='cluster', ax=ax)
            fig.savefig(path_z_distrib.join(f'cluster_{reducer_name}.png'))
            plt.close(fig)
    return c_gmm

def plot_clusters(result, data, path, c_gmm):
    log.info('Plotting x given clusters')
    figsize=(10,10)

    path_x = path.join('x')
    split_d = {'train': data['split']['train'], 'test': data['split']['test']}
    # x_all, z_all = result['all']['x'], result['all']['z']
    x_all = result['all']['x']

    gene_names = np.array(data['info']['gene_names'])
    # c_gmm = c_gmm.predict(z_all)
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
            # x_c = x_c.squeeze(1)
            x_c = x_c
            for x_img, gene_name in zip(x_c, gene_names_subset_c):
                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(x_img)
                fig.savefig(path_x_c.join(f'{gene_name}.png'))
                plt.close(fig)

# %%
def jaccard_precision_recall(x1, x2, threshold=0.5):
    x1 = (x1 >= threshold)
    x2 = (x2 >= threshold)

    union = (x1 | x2).sum()
    inter = (x1 & x2).sum()
    pred_area = x2.sum()

    jaccard = inter/union if union>0 else 0
    precision = inter/pred_area if pred_area>0 else 0

    return jaccard, precision

def jaccard_precision_curve(x1, x2, n_threshold=1000):
    thresholds = np.linspace(0, 1, n_threshold)
    jaccard_list, precision_list = [], []
    for t in thresholds:
        j, p = jaccard_precision_recall(x1, x2, t)
        jaccard_list.append(j), precision_list.append(p)
    jaccards, precisions = np.array(jaccard_list), np.array(precision_list)

    return thresholds, jaccards, precisions

def dist_cosine(m):
    '''
    v: vector
    m: matrix
    '''
    norm = np.linalg.norm(m, axis=1, ord=2)
    dist = 1 - m@m.T / (norm[...,None] @ norm[None,...])
    # dist = 1- m@v / (np.linalg.norm(m, axis=1, ord=2) * np.linalg.norm(v, ord=2))

    return dist

def dist_euclidian(m):
    '''
    v: vector
    m: matrix (N, f)

    optimize euclidian distance with matrix multiplication
    (a-b).T@(a-b) = a.T@a + b.T@b - 2a.T@b
    '''
    n = m.shape[0]
    s=(m**2).sum(axis=1) # although np.isclose(np.diag(AB),s)==True, use s since s is more precise (floating point precision)
    A=np.tile(s[...,None], (1,n))
    AB = m@m.T
    d=A+A.T - 2*AB
    d[d<0] = 0 # floating point precision error
    dist = np.sqrt(d)

    # dist = np.linalg.norm((m-v), axis=1, ord=2)
    return dist

# %%
def save_score(score, path):
    # Save score
    T.save_pickle(score, os.path.join(path, 'result.p'))
