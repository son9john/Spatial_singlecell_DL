import numpy as np
from scipy import sparse
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
