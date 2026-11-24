"""Colorectal section: Lee's L under the spectral sign-flip null, normal approximation vs Monte Carlo (own code)."""
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
names = genes[cand][sel]; Z = z[:, sel]
mu, U = np.linalg.eigh((sparse.diags(deg) - g).toarray())
xh = U.T @ Z
norm = float((np.asarray(W.sum(1)).ravel()**2).sum())
Mh = U.T @ ((W.T @ W) @ U) / norm; Mh = 0.5*(Mh + Mh.T)
obs = xh.T @ Mh @ xh
var = (xh*xh).T @ (Mh*Mh) @ (xh*xh)
pn = 2*stats.norm.sf(np.abs(obs)/np.sqrt(var))
B = int(sys.argv[1]); rng = np.random.default_rng(11)
Mh32, xh32 = Mh.astype(np.float32), xh.astype(np.float32); cnt = np.zeros((300, 300)); t0 = time.time(); aobs = np.abs(obs) - 1e-7
for b in range(B):
    e = rng.choice(np.array([-1., 1.], dtype=np.float32), size=(2, n))
    S = (e[0][:, None]*xh32).T @ (Mh32 @ (e[1][:, None]*xh32))
    cnt += np.abs(S) >= aobs
    if b == 20: print(f'est. {(time.time()-t0)/21*B:.0f}s', flush=True)
pm = (1 + cnt)/(B + 1)
iu, ju = np.triu_indices(300, 1)
def bh(p):
    m = len(p); o = np.argsort(p); r = p[o]*m/np.arange(1, m+1); r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty(m); q[o] = np.clip(r, 0, 1); return q
pn_, pm_ = pn[iu, ju], pm[iu, ju]
print(dict(max_abs_z=float(np.abs(obs/np.sqrt(var))[iu, ju].max()), min_norm_p=float(pn_.min()),
           norm_p05=100*(pn_ < .05).mean(), norm_q05=100*(bh(pn_) < .05).mean(),
           mc_p05=100*(pm_ < .05).mean(), mc_q05=100*(bh(pm_) < .05).mean(), B=B))
arch = pd.read_csv(f'{ROOT}/results/v3/crc_pairs_top300.csv.gz')
print(arch.columns.tolist()[:12])
key = {(a, b): k for k, (a, b) in enumerate(zip(names[iu], names[ju]))}
gx, gy = [c for c in arch.columns if 'gene' in c.lower()][:2]
idx = np.array([key.get((a, b), key.get((b, a), -1)) for a, b in zip(arch[gx], arch[gy])])
print('matched', (idx >= 0).mean(), 'max |lee normal p - archived|', np.abs(pn_[idx] - arch.lee_p_spectral.values).max(),
      'median', np.median(np.abs(pn_[idx] - arch.lee_p_spectral.values)))
np.savez_compressed('crc_lee_mc_p.npz', norm=pn_, mc=pm_)
pd.DataFrame([dict(stat='lee', norm_p05=100*(pn_ < .05).mean(), norm_q05=100*(bh(pn_) < .05).mean(),
              mc_p05=100*(pm_ < .05).mean(), mc_q05=100*(bh(pm_) < .05).mean(), B=B)]).to_csv('crc_lee_mc_summary.csv', index=False)
