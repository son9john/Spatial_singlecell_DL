"""Export the cross-donor null pairs (aligned and 180-degree rotated) for the official-package runs.

Writes one folder per unit under results/v3/official_inputs/: expression.csv (spots x 100 genes:
x genes from section A prefixed 'X_', y genes from section B prefixed 'Y_'), coords.csv
(array_row, array_col, and pixel-scale coordinates at 137 px per 100 um spot spacing),
graph.mtx (binary first-ring Visium graph) and pairs.csv (the 2,500 cross pairs).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmwrite

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # benchmarks/v3
from bench_calibration import A2_PAIRS, top_moran_genes  # noqa: E402
from common import RESULTS, hex_xy, jx, zcols  # noqa: E402
from datasets import load_dlpfc, lognorm  # noqa: E402

PX = 137.0
OUT = RESULTS / "official_inputs"


def export(sa, sb, mode):
    ca, _, ma = load_dlpfc(sa)
    cb, _, mb = load_dlpfc(sb)
    ia, na, _ = top_moran_genes(sa)
    ib, nb, _ = top_moran_genes(sb)
    rb, cbcol = mb.array_row.values, mb.array_col.values
    if mode == "rotated":
        rb, cbcol = 77 - rb, 127 - cbcol
    ka = pd.Index(list(zip(ma.array_row, ma.array_col)))
    kb = pd.Index(list(zip(rb, cbcol)))
    com = ka.intersection(kb)
    pa, pb = ka.get_indexer(com), kb.get_indexer(com)
    xa = lognorm(ca)[:, ia].toarray()[pa]
    xb = lognorm(cb)[:, ib].toarray()[pb]
    okx, oky = np.isfinite(zcols(xa)).all(0), np.isfinite(zcols(xb)).all(0)
    xa, xb, na, nb = xa[:, okx], xb[:, oky], na[okx], nb[oky]
    rr = np.array([c[0] for c in com])
    cc = np.array([c[1] for c in com])
    xy = hex_xy(rr, cc) * PX
    folder = OUT / f"{sa}x{sb}_{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    cols = [f"X_{g}" for g in na] + [f"Y_{g}" for g in nb]
    pd.DataFrame(np.column_stack([xa, xb]), columns=cols).to_csv(folder / "expression.csv", index=False)
    pd.DataFrame(dict(array_row=rr, array_col=cc, x_px=xy[:, 0], y_px=xy[:, 1])).to_csv(
        folder / "coords.csv", index=False)
    mmwrite(folder / "graph.mtx", jx.visium_graph(rr, cc))
    ix, iy = np.meshgrid(np.arange(len(na)), np.arange(len(nb)), indexing="ij")
    pd.DataFrame(dict(pair=np.arange(ix.size), gene_x=[f"X_{g}" for g in na[ix.ravel()]],
                      gene_y=[f"Y_{g}" for g in nb[iy.ravel()]])).to_csv(folder / "pairs.csv", index=False)
    print(folder.name, len(rr), "spots", flush=True)


if __name__ == "__main__":
    for mode in ("aligned", "rotated"):
        for a, b in A2_PAIRS:
            export(a, b, mode)
