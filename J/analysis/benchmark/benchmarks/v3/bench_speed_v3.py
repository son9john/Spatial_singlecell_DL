"""Speed of the optimized J kernels (juxtaposed 0.3) against an optimized Lee's L on the same
hardware: all pairs of 4,000 genes, DLPFC 151673 (3,611 spots). Run once per thread count."""
import os
import sys
import time

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, jx  # noqa: E402
from datasets import load_dlpfc, lognorm  # noqa: E402


def best(f, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    c, g, m = load_dlpfc("151673")
    x = lognorm(c)
    rows = []
    graph = jx.visium_graph(m.array_row.values, m.array_col.values)
    for ng in (1000, 4000):
        top = np.argsort(-np.asarray(x.mean(0)).ravel())[:ng]
        X = x[:, top].toarray()
        Z = jx.standardize(X)
        lap, _ = jx.laplacian(graph)
        S = jx.smoother(graph)
        W = jx._core.row_standardize(graph)

        def j_gemm_v02():
            zs = S @ jx.standardize(X)
            gg = zs.T @ (lap @ zs)
            e = np.diag(gg)
            return np.clip(-2 * gg / (e[:, None] + e[None, :]), -1, 1)

        tasks = [
            ("J scale 1, general product (v0.2)", j_gemm_v02),
            ("J scale 1 (v0.3)", lambda: jx.juxtaposition_matrix(X, graph, scale=1)),
            ("J scale 0 (v0.3)", lambda: jx.juxtaposition_matrix(X, graph, scale=0)),
            ("J scale 1 (v0.3), float32", lambda: jx.juxtaposition_matrix(X, graph, scale=1, dtype=np.float32)),
            ("Lee's L", lambda: jx.lee_l_matrix(X, graph)),
            ("bivariate Moran's I", lambda: jx.moran_bv_matrix(X, graph)),
            ("Pearson r", lambda: jx.pearson_matrix(X)),
        ]
        for name, f in tasks:
            rows.append(dict(threads=THREADS, genes=ng, task=name, seconds=best(f)))
            print(rows[-1], flush=True)
        if ng == 4000:
            t0 = time.perf_counter()
            null = jx.SpectralNull(graph)
            eig = time.perf_counter() - t0
            rows.append(dict(threads=THREADS, genes=0, task="eigendecomposition (once per section)", seconds=eig))
            for name, dt in (("J + spectral p-values (v0.3), float64", None),
                             ("J + spectral p-values (v0.3), float32", np.float32)):
                rows.append(dict(threads=THREADS, genes=ng, task=name,
                                 seconds=best(lambda: jx.juxtaposition(X, graph, scale=1, null=null, dtype=dt), reps=2)))
                print(rows[-1], flush=True)
    out = RESULTS / "speed_v3.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[old.threads != THREADS]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(out, index=False)


if __name__ == "__main__":
    main()
