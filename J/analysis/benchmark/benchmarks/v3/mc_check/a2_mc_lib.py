"""Own re-implementation of the A2 cross-donor control (aligned and rotated) from the protocol text,
with Monte Carlo sign-flip p-values next to the normal-approximation p-values. Reads data only."""
import os
import sys, time, numpy as np, pandas as pd
from scipy import sparse, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import h5min
ROOT = os.environ.get('JBENCH_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
PAIRS = [("151507","151673"),("151508","151674"),("151509","151669"),("151510","151670"),("151671","151675"),("151672","151676")]
LAY = {"L1","L2","L3","L4","L5","L6","WM"}
layers = pd.read_csv(f'{ROOT}/data/dlpfc/layers.tsv', sep='\t', header=None, names=['barcode','section','layer'], dtype=str)
def zcols(m):
    sd = m.std(axis=0); sd[sd == 0] = np.nan
    return (m - m.mean(axis=0)) / sd
def hexgraph(r, c):
    idx = {(a, b): k for k, (a, b) in enumerate(zip(r.tolist(), c.tolist()))}
    ii, jj = [], []
    for (a, b), k in idx.items():
        for da, db in ((0, 2), (1, -1), (1, 1)):
            m = idx.get((a + da, b + db))
            if m is not None: ii.append(k); jj.append(m)
    n = len(r)
    return sparse.csr_matrix((np.ones(2*len(ii)), (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
_cache = {}
def load(sec):
    if sec in _cache: return _cache[sec]
    h = h5min.read(f'{ROOT}/data/dlpfc/{sec}/counts.h5', ['matrix/shape','matrix/data','matrix/indices','matrix/indptr','matrix/barcodes'])
    shape = tuple(int(v) for v in h['matrix/shape'])
    X = sparse.csc_matrix((h['matrix/data'], h['matrix/indices'], h['matrix/indptr']), shape=shape).T.tocsr()
    bc = np.array([b.decode() for b in h['matrix/barcodes']])
    pos = pd.read_csv(f'{ROOT}/data/dlpfc/{sec}/positions.csv', header=None, index_col=0,
                      names=['barcode','in_tissue','array_row','array_col','pxl_row','pxl_col']).reindex(bc)
    lab = layers[layers.section == sec].set_index('barcode')['layer'].reindex(bc).values
    umi = np.asarray(X.sum(axis=1)).ravel()
    keep = (pos.in_tissue.values == 1) & np.isin(lab, list(LAY)) & (umi > 0)
    X = X[keep].astype(float).tocsr(); pos = pos[keep]
    lib = np.asarray(X.sum(axis=1)).ravel()
    N = sparse.diags(1e4/lib) @ X; N.data = np.log1p(N.data); N = N.tocsc()
    detect = np.asarray((X > 0).mean(axis=0)).ravel(); cand = np.where(detect >= 0.10)[0]
    x = N[:, cand].toarray(); z = zcols(x)
    g = hexgraph(pos.array_row.values, pos.array_col.values)
    d = np.asarray(g.sum(1)).ravel(); W = sparse.diags(np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)) @ g
    mi = np.einsum('ik,ik->k', z, W @ z) / W.sum()
    top = cand[np.argsort(-mi)[:50]]
    _cache[sec] = (N[:, top].toarray(), pos.array_row.values.astype(int), pos.array_col.values.astype(int))
    return _cache[sec]
