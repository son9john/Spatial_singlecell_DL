"""Loaders for the public datasets used in the benchmark (cached as .npz)."""
from pathlib import Path
import gzip
import os
import numpy as np
import pandas as pd
from scipy import sparse, io

try:  # prefer h5py when available; fall back to the ctypes reader used in the offline build
    import h5py  # noqa: F401

    def read_10x_h5(path, gene_expression_only=True):
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            shape = tuple(int(v) for v in m["shape"][:])
            mat = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
            ftype = m["features/feature_type"].asstr()[:] if "feature_type" in m["features"] else None
            names = m["features/name"].asstr()[:]
            ids = m["features/id"].asstr()[:]
            barcodes = m["barcodes"].asstr()[:]
        keep = np.ones(shape[0], bool) if (ftype is None or not gene_expression_only) else (ftype == "Gene Expression")
        return mat[keep].T.tocsr(), np.asarray(barcodes), np.asarray(names[keep]), np.asarray(ids[keep])
except ImportError:  # pragma: no cover
    from h5lite import read_10x_h5

# J-analysis repository root (override with JBENCH_ROOT)
ROOT = Path(os.environ.get("JBENCH_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

SECTIONS = tuple(map(str, (*range(151507, 151511), *range(151669, 151677))))
DONOR = {s: ("Br5292" if s.startswith("15150") or s == "151510" else
             "Br5595" if s in ("151669", "151670", "151671", "151672") else "Br8100")
         for s in SECTIONS}
LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")


def load_dlpfc(section):
    """Spots in tissue with a layer label: counts (spots x genes, CSR int), metadata."""
    cache = CACHE / f"dlpfc_{section}.npz"
    meta_cache = CACHE / f"dlpfc_{section}_meta.csv"
    if cache.exists() and meta_cache.exists():
        counts = sparse.load_npz(cache).tocsr()
        meta = pd.read_csv(meta_cache, index_col=0)
        genes = np.load(CACHE / f"dlpfc_{section}_genes.npy", allow_pickle=True)
        return counts, genes, meta
    counts, barcodes, genes, _ = read_10x_h5(DATA / "dlpfc" / section / "counts.h5")
    pos = pd.read_csv(DATA / "dlpfc" / section / "positions.csv", header=None, index_col=0,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"])
    layers = pd.read_csv(DATA / "dlpfc" / "layers.tsv", sep="\t", header=None,
                         names=["barcode", "section", "layer"], dtype=str)
    lab = layers[layers.section == section].set_index("barcode")["layer"]
    meta = pos.reindex(barcodes)
    meta["layer"] = lab.reindex(barcodes).values
    meta["umi"] = np.asarray(counts.sum(axis=1)).ravel()
    keep = (meta["in_tissue"].values == 1) & meta["layer"].isin(LAYER_ORDER).values & (meta["umi"].values > 0)
    counts = counts[keep].astype(np.int32).tocsr()
    meta = meta[keep].copy()
    meta["section"] = section
    meta["donor"] = DONOR[section]
    sparse.save_npz(cache, counts)
    meta.to_csv(meta_cache)
    np.save(CACHE / f"dlpfc_{section}_genes.npy", genes, allow_pickle=True)
    return counts, genes, meta


def lognorm(counts, scale=1e4):
    """log1p(scale * count / library size), dense float64 for the selected columns."""
    lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    norm = sparse.diags(scale / lib) @ counts.astype(float)
    norm.data = np.log1p(norm.data)
    return norm.tocsr()


def load_crc():
    counts, barcodes, genes, _ = read_10x_h5(DATA / "crc" / "filtered_feature_bc_matrix.h5")
    with gzip.open(DATA / "crc" / "tissue_positions.csv.gz", "rt") as fh:
        pos = pd.read_csv(fh)
    pos = pos.set_index(pos.columns[0])
    meta = pos.reindex(barcodes)
    keep = meta["in_tissue"].values == 1
    return counts[keep].tocsr(), genes, meta[keep].copy()


def load_seqfish():
    up = DATA / "seqfish"
    cache = CACHE / "seqfish.npz"
    genes = np.array([g.strip() for g in open(up / "seqfish_genes.txt") if g.strip()])
    meta = pd.read_csv(up / "seqfish_z2_meta.csv.gz")
    if cache.exists():
        expr = sparse.load_npz(cache).tocsr()
    else:
        with gzip.open(up / "seqfish_z2_exprs.mtx.gz", "rb") as fh:
            m = io.mmread(fh)  # genes x cells
        expr = sparse.csr_matrix(m.T)
        sparse.save_npz(cache, expr)
    return expr, genes, meta
