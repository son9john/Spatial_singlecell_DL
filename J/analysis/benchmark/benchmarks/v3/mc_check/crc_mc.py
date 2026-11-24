"""Colorectal section: normal-approximation vs Monte Carlo sign-flip p-values for Pearson r and J1 (own code)."""
import os
import sys, gzip, time, numpy as np, pandas as pd
from scipy import sparse, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import h5min
from a2_helpers import zcols, hexgraph
ROOT = os.environ.get('JBENCH_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
h = h5min.read(f'{ROOT}/data/crc/filtered_feature_bc_matrix.h5', ['matrix/shape','matrix/data','matrix/indices','matrix/indptr','matrix/barcodes','matrix/features/name'])
shape = tuple(int(v) for v in h['matrix/shape'])
X = sparse.csc_matrix((h['matrix/data'], h['matrix/indices'], h['matrix/indptr']), shape=shape).T.tocsr()
bc = np.array([b.decode() for b in h['matrix/barcodes']]); genes = np.array([b.decode() for b in h['matrix/features/name']])
pos = pd.read_csv(gzip.open(f'{ROOT}/data/crc/tissue_positions.csv.gz', 'rt')); pos = pos.set_index(pos.columns[0]).reindex(bc)
keep = pos.in_tissue.values == 1; X = X[keep].astype(float).tocsr(); pos = pos[keep]
lib = np.asarray(X.sum(1)).ravel(); N = sparse.diags(1e4/lib) @ X; N.data = np.log1p(N.data); N = N.tocsc()
detect = np.asarray((X > 0).mean(0)).ravel()
cand = np.where((detect >= 0.10) & ~pd.Series(genes).str.upper().str.startswith('MT-').values)[0]
g = hexgraph(pos.array_row.values.astype(int), pos.array_col.values.astype(int)); n = g.shape[0]
deg = np.asarray(g.sum(1)).ravel(); W = sparse.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)) @ g
z = zcols(N[:, cand].toarray()); mi = np.einsum('ik,ik->k', z, W @ z) / W.sum(); sel = np.argsort(-mi)[:300]
ref = pd.read_csv(f'{ROOT}/results/v2/crc_genes_top300.csv')
print('spots', n, 'edges', g.nnz//2, 'same top-300 genes as archived:', list(genes[cand][sel]) == list(ref.gene), flush=True)
Z = z[:, sel]
mu, U = np.linalg.eigh((sparse.diags(deg) - g).toarray()); mu = np.clip(mu, 0, None)
xh = U.T @ Z; hJ = mu * (1 - mu/(2*deg.max()))**2
B = 4000; rng = np.random.default_rng(7); E = rng.choice(np.array([-1., 1.], dtype=np.float32), size=(B, n))
iu, ju = np.triu_indices(300, 1); out = {}
for name, hh in (('pearson', np.ones(n)), ('J1', hJ)):
    PN = np.full((300, 300), np.nan); PM = PN.copy(); EF = PN.copy(); t0 = time.time()
    for i in range(299):
        A = (hh * xh[:, i])[:, None] * xh[:, i+1:]
        obs = A.sum(0); v = (A*A).sum(0)
        PN[i, i+1:] = 2*stats.norm.sf(np.abs(obs)/np.sqrt(v))
        nul = E @ A.astype(np.float32)
        PM[i, i+1:] = (1 + (np.abs(nul) >= np.abs(obs) - 1e-9).sum(0)) / (B + 1)
        EF[i, i+1:] = v**2 / (A**4).sum(0)
    out[name] = (PN[iu, ju], PM[iu, ju], EF[iu, ju]); print(name, f'{time.time()-t0:.0f}s', flush=True)
def bh(p):
    m = len(p); o = np.argsort(p); r = p[o]*m/np.arange(1, m+1); r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(m); q[o] = np.clip(r, 0, 1); return q
rows = []
for name, (pn, pm, ef) in out.items():
    rows.append(dict(stat=name, eff_modes_median=np.median(ef), max_abs_z=stats.norm.isf(pn.min()/2),
                     norm_p05=100*(pn < .05).mean(), norm_q05=100*(bh(pn) < .05).mean(),
                     mc_p05=100*(pm < .05).mean(), mc_q05=100*(bh(pm) < .05).mean(), mc_min_p=pm.min()))
print(pd.DataFrame(rows).round(3).to_string())
pd.DataFrame(rows).to_csv('crc_mc_summary.csv', index=False)
np.savez_compressed('crc_mc_p.npz', **{f'{k}_{s}': v for k, t in out.items() for s, v in zip(('norm', 'mc', 'eff'), t)})
