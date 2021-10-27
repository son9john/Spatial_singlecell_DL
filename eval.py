import numpy as np
from sklearn import metrics
import torch

import tools as T
import tools.torch

# %%
def eval_autoencoder(model, data):
    device = T.torch.get_device(model)
    train_i, test_i = data['split']['train'], data['split']['test']
    with torch.no_grad():
        x_all = torch.stack([d for d in data['info']['dataset_all']], axis=0).to(device)
        z_all = model.encoder(x_all).flatten(1)
        x_hat_all = model(x_all)

        x_all, z_all, x_hat_all = x_all.cpu().numpy(), z_all.cpu().numpy(), x_hat_all.cpu().numpy()

        x_train, z_train, x_hat_train = x_all[train_i], z_all[train_i], x_hat_all[train_i]
        x_test, z_test, x_hat_test = x_all[test_i], z_all[test_i], x_hat_all[test_i]

    result = {
    'all': {'x': x_all, 'z': z_all, 'x_hat': x_hat_all},
    'train': {'x': x_train, 'z': z_train, 'x_hat': x_hat_train},
    'test': {'x': x_test, 'z': z_test, 'x_hat': x_hat_test},
    }
    return result

# %%
def score_autoencoder(result, cfg):
    # 1. Sort

    # 2.

    # Figures
    
    return score

# %%

# %%
def jaccard_precision_recall(gt, pred, threshold=0.5):
    assert gt.max() <= 1 and gt.min() >= 0
    assert pred.max() <= 1 and pred.min() >= 0
    assert threshold <= 1 and threshold >= 0
    assert gt.shape[0] == pred.shape[0]
    assert gt.shape[1] == pred.shape[1]

    gt = (gt >= threshold)
    pred = (pred >= threshold)

    inter = (gt & pred).sum()
    gt_area = gt.sum()
    pred_area = pred.sum()
    union = gt_area + pred_area - inter

    if union > 0:
        jaccard = inter / union
    else:
        jaccard = 0
    if pred_area > 0:
        precision = inter / pred_area
    else:
        precision = 0

    return jaccard, precision

def jaccard_precision_curve(gt, pred, threshold_num=1000):
    threshold_list = []
    jaccard_list = []
    precision_list = []
    for th in np.linspace(0, 1, threshold_num):
        threshold_list.append(th)
        j, p = jaccard_precision_recall(gt, pred, th)
        jaccard_list.append(j)
        precision_list.append(p)

    return threshold_list, jaccard_list, precision_list

def find_similar_i(x, i, f):
    a

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
