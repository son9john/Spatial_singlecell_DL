"""Calibration of the official Python packages on the exported cross-donor null pairs.

Run on a machine with the packages installed, e.g.
    python -m pip install liana==2.1.1 spatialdm==0.3.1 hotspotsc scanpy
    python benchmarks/v3/official/run_official_python.py results/v3/official_inputs
Each package is called with its tutorial defaults (see PROTOCOL_v3_amendment.md). Failures of
one package are recorded and do not stop the others. Output: official_python_pvalues.csv.gz.
NOTE: written without access to these packages; check the printed versions and any errors.
"""
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, stats

PX = 137.0


def load(folder):
    expr = pd.read_csv(folder / "expression.csv")
    coords = pd.read_csv(folder / "coords.csv")
    pairs = pd.read_csv(folder / "pairs.csv")
    return expr, coords, pairs


def make_adata(expr, coords):
    from anndata import AnnData
    ad = AnnData(expr.to_numpy(dtype=float), var=pd.DataFrame(index=expr.columns))
    ad.obs_names = [f"s{k}" for k in range(ad.n_obs)]
    ad.obsm["spatial"] = coords[["x_px", "y_px"]].to_numpy()
    return ad


def scipy_tests(expr, pairs):
    out = {}
    x = expr[pairs.gene_x].to_numpy()
    y = expr[pairs.gene_y].to_numpy()
    out["pearson_t"] = [stats.pearsonr(x[:, k], y[:, k])[1] for k in range(x.shape[1])]
    out["spearman_t"] = [stats.spearmanr(x[:, k], y[:, k])[1] for k in range(x.shape[1])]
    return out


def spatialdm_tests(expr, coords, pairs):
    import spatialdm as sdm
    ad = make_adata(expr, coords)
    sdm.weight_matrix(ad, l=1.2 * PX, cutoff=0.2, single_cell=False)
    names = [f"{a}^{b}" for a, b in zip(pairs.gene_x, pairs.gene_y)]
    ad.uns["ligand"] = pd.DataFrame(list(pairs.gene_x), index=names)
    ad.uns["receptor"] = pd.DataFrame(list(pairs.gene_y), index=names)
    ad.uns["geneInter"] = pd.DataFrame({"annotation": "Secreted Signaling"}, index=names)
    ad.uns.update(mean="algebra", single_cell=False)
    ad.raw = ad
    sdm.spatialdm_global(ad, n_perm=1000, specified_ind=names, method="both", nproc=1)
    res = ad.uns["global_res"].loc[names]
    zcol = [c for c in res.columns if "z" in c.lower() and "p" in c.lower()]
    pcol = [c for c in res.columns if "perm" in c.lower()]
    return {"spatialdm_z": res[zcol[0]].to_numpy(), "spatialdm_perm": res[pcol[0]].to_numpy()}


def liana_tests(expr, coords, pairs):
    import liana as li
    ad = make_adata(expr, coords)
    nb = getattr(li, "pp", li.ut) if hasattr(li, "pp") and hasattr(li.pp, "spatial_neighbors") else li.ut
    nb.spatial_neighbors(ad, bandwidth=200, cutoff=0.1, kernel="gaussian", set_diag=True)
    inter = list(zip(pairs.gene_x, pairs.gene_y))
    res = li.mt.bivariate(ad, interactions=inter, local_name=None, global_name="morans", n_perms=100,
                          mask_negatives=False, nz_prop=0, use_raw=False, complex_sep=None)
    var = res.var if hasattr(res, "var") else res
    pcol = [c for c in var.columns if "morans" in c and "pval" in c][0]
    names = [f"{a}^{b}" for a, b in inter]
    return {"liana_perm": var.loc[names, pcol].to_numpy()}


def hotspot_tests(expr, coords, pairs):
    import hotspot
    ad = make_adata(expr, coords)
    z = (ad.X - ad.X.mean(0)) / ad.X.std(0)
    ad.layers["z"] = z
    hs = hotspot.Hotspot(ad, layer_key="z", model="none", latent_obsm_key="spatial")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
    zmat = hs.compute_local_correlations(list(expr.columns), jobs=1)
    zz = np.array([zmat.loc[a, b] for a, b in zip(pairs.gene_x, pairs.gene_y)])
    return {"hotspot_z": stats.norm.sf(zz)}


def main(root):
    root = Path(root)
    frames, errors = [], []
    versions = {}
    for pkg in ("liana", "spatialdm", "hotspotsc", "anndata", "scipy", "numpy"):
        try:
            versions[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "not installed"
    print(versions, flush=True)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        expr, coords, pairs = load(folder)
        out = pd.DataFrame(dict(unit=folder.name, pair=pairs.pair, gene_x=pairs.gene_x, gene_y=pairs.gene_y))
        for fn in (scipy_tests, spatialdm_tests, liana_tests, hotspot_tests):
            try:
                for k, v in fn(expr, coords, pairs).items() if fn is not scipy_tests else fn(expr, pairs).items():
                    out[k] = v
            except Exception as err:  # noqa: BLE001
                errors.append(dict(unit=folder.name, step=fn.__name__, error=repr(err)))
                traceback.print_exc()
        frames.append(out)
        print(folder.name, "done", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(root.parent / "official_python_pvalues.csv.gz", index=False)
    pd.DataFrame(errors).to_csv(root.parent / "official_python_errors.csv", index=False)
    pd.Series(versions).to_csv(root.parent / "official_python_versions.csv")
    tests = [c for c in res.columns if c.endswith(("_t", "_z", "_perm"))]
    res["mode"] = res.unit.str.split("_").str[-1]
    print((res[tests] < 0.05).groupby(res["mode"]).mean().round(3).T)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/v3/official_inputs")
