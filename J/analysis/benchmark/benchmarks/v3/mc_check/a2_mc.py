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
B = 4000
rows, perpair = [], []
for sa, sb in PAIRS:
    for mode in ('rotated', 'aligned'):
        t0 = time.time()
        xa, ra, ca = load(sa); xb, rb, cb = load(sb)
        if mode == 'rotated': rb, cb = 77 - rb, 127 - cb
        ka, kb = pd.Index(list(zip(ra, ca))), pd.Index(list(zip(rb, cb)))
        com = ka.intersection(kb); pa, pb = ka.get_indexer(com), kb.get_indexer(com)
        rr, cc = np.array([c[0] for c in com]), np.array([c[1] for c in com])
        g = hexgraph(rr, cc); n = g.shape[0]
        deg = np.asarray(g.sum(1)).ravel()
        mu, U = np.linalg.eigh((sparse.diags(deg) - g).toarray()); mu = np.clip(mu, 0, None)
        ndeg = int((np.diff(mu) <= 1e-8*mu.max()).sum())
        zx, zy = zcols(xa[pa]), zcols(xb[pb])
        zx, zy = zx[:, np.isfinite(zx).all(0)], zy[:, np.isfinite(zy).all(0)]
        xh, yh = U.T @ zx, U.T @ zy
        hJ = mu * (1 - mu/(2*deg.max()))**2
        rng = np.random.default_rng([2026, int(sa), int(sb), mode == 'rotated'])
        E = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(B, n))
        res = {}
        for name, h in (('J1', hJ), ('pearson', np.ones(n))):
            pn, pm, eff = [], [], []
            for i in range(zx.shape[1]):
                A = (h * xh[:, i])[:, None] * yh                      # n x Ky, per-mode contributions
                obs = A.sum(0); v = (A*A).sum(0)
                pn.append(2*stats.norm.sf(np.abs(obs)/np.sqrt(v)))
                nul = E @ A.astype(np.float32)
                pm.append((1 + (np.abs(nul) >= np.abs(obs) - 1e-9).sum(0)) / (B + 1))
                eff.append(v**2 / (A**4).sum(0))
            res[name] = (np.concatenate(pn), np.concatenate(pm), np.concatenate(eff))
        ex, ey = (hJ[:, None]*xh*xh).sum(0), (hJ[:, None]*yh*yh).sum(0)
        jstat = (-2*((xh*hJ[:, None]).T @ yh) / (ex[:, None] + ey[None, :])).ravel()
        unit = f'{sa}x{sb}'
        for k in range(len(jstat)):
            perpair.append((unit, mode, k, jstat[k], res['J1'][0][k], res['J1'][1][k], res['J1'][2][k],
                            res['pearson'][0][k], res['pearson'][1][k], res['pearson'][2][k]))
        rows.append(dict(unit=unit, mode=mode, n=n, pairs=len(jstat), repeated_eig=ndeg,
                         J_norm=100*(res['J1'][0] < .05).mean(), J_mc=100*(res['J1'][1] < .05).mean(), J_eff=np.median(res['J1'][2]),
                         P_norm=100*(res['pearson'][0] < .05).mean(), P_mc=100*(res['pearson'][1] < .05).mean(), P_eff=np.median(res['pearson'][2])))
        print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv('a2_mc_summary.csv', index=False)
pd.DataFrame(perpair, columns=['unit','mode','pair','J1_stat','J_p_norm','J_p_mc','J_eff','P_p_norm','P_p_mc','P_eff']).to_csv('a2_mc_pairs.csv.gz', index=False)
