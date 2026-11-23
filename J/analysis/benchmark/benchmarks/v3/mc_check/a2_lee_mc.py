"""A2 cross-donor control: Lee's L under the spectral null, normal approximation vs Monte Carlo (own code)."""
import os
import sys, time, numpy as np, pandas as pd
from scipy import sparse, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_mc_lib as L
B = 1000; rows = []
for sa, sb in L.PAIRS:
    for mode in ('rotated', 'aligned'):
        t0 = time.time()
        xa, ra, ca = L.load(sa); xb, rb, cb = L.load(sb)
        if mode == 'rotated': rb, cb = 77 - rb, 127 - cb
        ka, kb = pd.Index(list(zip(ra, ca))), pd.Index(list(zip(rb, cb)))
        com = ka.intersection(kb); pa, pb = ka.get_indexer(com), kb.get_indexer(com)
        rr, cc = np.array([c[0] for c in com]), np.array([c[1] for c in com])
        g = L.hexgraph(rr, cc); n = g.shape[0]; deg = np.asarray(g.sum(1)).ravel()
        mu, U = np.linalg.eigh((sparse.diags(deg) - g).toarray())
        W = sparse.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)) @ g
        norm = float((np.asarray(W.sum(1)).ravel()**2).sum())
        M = U.T @ ((W.T @ W) @ U) / norm; M = 0.5*(M + M.T)
        zx, zy = L.zcols(xa[pa]), L.zcols(xb[pb]); zx, zy = zx[:, np.isfinite(zx).all(0)], zy[:, np.isfinite(zy).all(0)]
        xh, yh = U.T @ zx, U.T @ zy
        obs = xh.T @ M @ yh; var = (xh*xh).T @ (M*M) @ (yh*yh); pn = 2*stats.norm.sf(np.abs(obs)/np.sqrt(var))
        rng = np.random.default_rng([31, int(sa), int(sb), mode == 'rotated'])
        M32, x32, y32 = M.astype(np.float32), xh.astype(np.float32), yh.astype(np.float32); cnt = np.zeros_like(obs); a = np.abs(obs) - 1e-7
        for b in range(B):
            e = rng.choice(np.array([-1., 1.], dtype=np.float32), size=(2, n))
            cnt += np.abs((e[0][:, None]*x32).T @ (M32 @ (e[1][:, None]*y32))) >= a
        pm = (1 + cnt)/(B + 1)
        rows.append(dict(unit=f'{sa}x{sb}', mode=mode, pairs=pn.size, lee_norm=100*(pn < .05).mean(), lee_mc=100*(pm < .05).mean()))
        print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
d = pd.DataFrame(rows); d.to_csv('a2_lee_mc_summary.csv', index=False)
print(d.groupby('mode')[['lee_norm', 'lee_mc']].mean().round(2))
