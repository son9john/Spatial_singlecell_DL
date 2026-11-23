"""Readers for the juxtaposed app: every input format becomes one common dataset (a dict).

Dataset keys
    counts    scipy.sparse.csr_matrix, spots x genes, raw counts (or None when only normalized values exist)
    X         scipy.sparse.csr_matrix, spots x genes, log-normalized values (filled by normalize())
    raw       values of unknown scale before normalize() decides (tables)
    genes     list of gene names; spots: list of spot or cell IDs
    coords    float array (n, 2): plotting coordinates x, y (image pixels when an image is present)
    array     int array (n, 2) of lattice row, col (Visium, Visium HD, bins) or None
    lattice   "hex" | "square" | "none"
    image     uint8 array (height, width, 3) or None
    meta      pandas DataFrame of annotations (one row per spot)
    platform, unit, notes
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

import juxtaposed as jx


# ---- uploads ----------------------------------------------------------------------------------

def expand_uploads(files: list[dict]) -> dict[str, str]:
    """{file name: path}; .tar.gz/.tgz/.tar/.zip archives are unpacked."""
    out: dict[str, str] = {}
    for f in files:
        name, path = f["name"], f["datapath"]
        low = name.lower()
        if low.endswith((".tar.gz", ".tgz", ".tar")):
            d = tempfile.mkdtemp(prefix="untar_")
            with tarfile.open(path) as t:
                t.extractall(d)
            out.update(_walk(d))
        elif low.endswith(".zip"):
            d = tempfile.mkdtemp(prefix="unzip_")
            with zipfile.ZipFile(path) as z:
                z.extractall(d)
            out.update(_walk(d))
        else:
            out[name] = path
    return {k: v for k, v in out.items() if not k.startswith("._") and "__MACOSX" not in v}


def _walk(d):
    found = {}
    for root, _, names in os.walk(d):
        for n in names:
            found[n] = os.path.join(root, n)
    return found


def pick(files: dict, patterns) -> str | None:
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        for name, path in files.items():
            if re.search(p, name, flags=re.IGNORECASE):
                return path
    return None


# ---- count matrices ---------------------------------------------------------------------------

def _strings(ds) -> list[str]:
    v = ds[()]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(v).ravel()]


def read_10x_h5(path: str, gene_expression_only: bool = True):
    """10x HDF5 count matrix (Cell Ranger / Space Ranger / Xenium). Returns (spots x genes csr, barcodes, genes)."""
    import h5py

    with h5py.File(path, "r") as f:
        if "matrix" in f:
            g = f["matrix"]
            if "features" in g:
                names = _strings(g["features/name"])
                types = _strings(g["features/feature_type"]) if "feature_type" in g["features"] else ["Gene Expression"] * len(names)
            else:
                names = _strings(g["gene_names"])
                types = ["Gene Expression"] * len(names)
        else:
            key = next((k for k in f.keys() if isinstance(f[k], h5py.Group) and "data" in f[k]), None)
            if key is None:
                raise ValueError(f"{os.path.basename(path)} is not a 10x count matrix")
            g = f[key]
            names = _strings(g["gene_names"])
            types = ["Gene Expression"] * len(names)
        shape = tuple(int(x) for x in g["shape"][()])
        m = sparse.csc_matrix((g["data"][()].astype(np.float64), g["indices"][()], g["indptr"][()]), shape=shape)
        barcodes = _strings(g["barcodes"])
    types = np.asarray(types)
    keep = types == "Gene Expression" if gene_expression_only and (types == "Gene Expression").any() else np.ones(len(types), bool)
    m = m[keep, :].T.tocsr()
    return m, barcodes, _unique([n for n, k in zip(names, keep) if k])


def read_mex(files: dict):
    mtx = pick(files, r"^matrix\.mtx(\.gz)?$")
    bc = pick(files, r"^barcodes\.tsv(\.gz)?$")
    ft = pick(files, [r"^features\.tsv(\.gz)?$", r"^genes\.tsv(\.gz)?$"])
    if not (mtx and bc and ft):
        return None
    m = spio.mmread(mtx).tocsc()
    barcodes = pd.read_csv(bc, header=None, sep="\t")[0].astype(str).tolist()
    feat = pd.read_csv(ft, header=None, sep="\t", quoting=3)
    genes = (feat[1] if feat.shape[1] >= 2 else feat[0]).astype(str).to_numpy()
    if feat.shape[1] >= 3 and (feat[2] == "Gene Expression").any():
        keep = (feat[2] == "Gene Expression").to_numpy()
        m = m[keep, :]
        genes = genes[keep]
    return m.T.tocsr(), barcodes, _unique(list(genes))


def read_jx10x(path: str, gene_expression_only: bool = True):
    """10x count matrix in the compact binary written by the R app's browser reader (also used for the example)."""
    raw = open(path, "rb").read()
    if raw[:6] != b"JX10X1":
        raise ValueError("Not a converted 10x matrix")
    hl = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    h = json.loads(raw[12:12 + hl].decode("utf-8"))
    off = 12 + hl
    off += (4 - off % 4) % 4
    indptr = np.frombuffer(raw, dtype="<i4", count=h["n_indptr"], offset=off)
    off += 4 * h["n_indptr"]
    indices = np.frombuffer(raw, dtype="<i4", count=h["nnz"], offset=off)
    off += 4 * h["nnz"]
    data = (np.frombuffer(raw, dtype="<i4", count=h["nnz"], offset=off).astype(np.float64) if h["data_type"] == "int"
            else np.frombuffer(raw, dtype="<f8", count=h["nnz"], offset=off).copy())
    m = sparse.csc_matrix((data, indices, indptr), shape=tuple(h["shape"]))
    types = np.asarray(h.get("types") or ["Gene Expression"] * h["shape"][0])
    keep = types == "Gene Expression" if gene_expression_only and (types == "Gene Expression").any() else np.ones(len(types), bool)
    names = [n for n, k in zip(h["names"], keep) if k]
    return m[keep, :].T.tocsr(), [str(b) for b in h["barcodes"]], _unique(names)


def read_counts(files: dict, prefer=("filtered_feature_bc_matrix", "cell_feature_matrix")):
    h5 = pick(files, [p + r".*\.h5$" for p in prefer] + [r"\.h5$"])
    if h5:
        return read_10x_h5(h5)
    bin_ = pick(files, r"\.jx10x$")
    if bin_:
        return read_jx10x(bin_)
    return read_mex(files)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---- 10x Visium / Visium HD (Space Ranger) ----------------------------------------------------

POS_COLS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def read_positions(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        pos = pq.read_table(path).to_pandas()
    else:
        with open(path) as fh:
            first = fh.readline()
        header = bool(re.search("barcode|in_tissue", first))
        pos = pd.read_csv(path, header=0 if header else None)
        if not header:
            pos.columns = POS_COLS[: pos.shape[1]]
    if "barcode" not in pos.columns:
        pos = pos.rename(columns={pos.columns[0]: "barcode"})
    pos["barcode"] = pos["barcode"].astype(str)
    return pos


def read_image(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def read_spaceranger(files: dict) -> dict:
    got = read_counts(files)
    if got is None:
        raise ValueError("No count matrix found. Add filtered_feature_bc_matrix.h5, or matrix.mtx.gz with barcodes.tsv.gz and features.tsv.gz.")
    m, barcodes, genes = got
    pos_path = pick(files, [r"^tissue_positions\.parquet$", r"^tissue_positions\.csv$", r"^tissue_positions_list\.csv$", "tissue_positions"])
    if not pos_path:
        raise ValueError("No tissue_positions file found. Add the files of the spatial folder, or the spatial.tar.gz archive.")
    pos = read_positions(pos_path)
    if "in_tissue" in pos:
        pos = pos[pos["in_tissue"] == 1]
    index = {b: i for i, b in enumerate(barcodes)}
    pos = pos[pos["barcode"].isin(index)]
    if not len(pos):
        raise ValueError("The barcodes of the count matrix and of the tissue positions do not match.")
    m = m[[index[b] for b in pos["barcode"]], :]
    sf_path = pick(files, r"^scalefactors_json\.json$")
    sf = json.load(open(sf_path)) if sf_path else {}
    img_path = pick(files, [r"^tissue_lowres_image\.png$", r"^tissue_hires_image\.png$"])
    image, scale = None, 1.0
    if img_path:
        s = sf.get("tissue_lowres_scalef") if "lowres" in img_path else sf.get("tissue_hires_scalef")
        if s:
            image, scale = read_image(img_path), float(s)
    first = pos["barcode"].iloc[0]
    hd = "bin_size_um" in sf or bool(re.match(r"^s_\d+um_", first))
    bin_um = sf.get("bin_size_um")
    if bin_um is None and hd:
        mm = re.match(r"^s_0*(\d+)um_", first)
        bin_um = int(mm.group(1)) if mm else None
    return dict(
        counts=m, X=None, raw=None, genes=genes, spots=pos["barcode"].tolist(),
        coords=np.c_[pos["pxl_col_in_fullres"].to_numpy(float) * scale, pos["pxl_row_in_fullres"].to_numpy(float) * scale],
        array=np.c_[pos["array_row"].to_numpy(int), pos["array_col"].to_numpy(int)],
        lattice="square" if hd else "hex", image=image, meta=pd.DataFrame(index=pos["barcode"].to_numpy()),
        platform=f"Visium HD ({bin_um} µm bins)" if hd else "Visium", unit="pixels", notes=[],
    )


# ---- Xenium -----------------------------------------------------------------------------------

def read_table_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()
    sep = "\t" if re.search(r"\.(tsv|txt)(\.gz)?$", path, flags=re.IGNORECASE) else ","
    return pd.read_csv(path, sep=sep)


def read_xenium(files: dict) -> dict:
    got = read_counts(files, prefer=("cell_feature_matrix",))
    if got is None:
        raise ValueError("No count matrix found. Add cell_feature_matrix.h5 (or the cell_feature_matrix MEX files).")
    m, barcodes, genes = got
    cp = pick(files, [r"^cells\.parquet$", r"^cells\.csv\.gz$", r"^cells\.csv$"])
    if not cp:
        raise ValueError("No cells.csv.gz or cells.parquet found (it holds the cell centroids).")
    ce = read_table_any(cp)
    ids = ce["cell_id"].astype(str) if "cell_id" in ce else ce.iloc[:, 0].astype(str)
    index = {b: i for i, b in enumerate(barcodes)}
    keep = ids.isin(index).to_numpy()
    if not keep.any():
        raise ValueError("The cell IDs of the count matrix and of the cells file do not match.")
    ce, ids = ce[keep], ids[keep]
    return dict(
        counts=m[[index[b] for b in ids], :], X=None, raw=None, genes=genes, spots=ids.tolist(),
        coords=np.c_[ce["x_centroid"].to_numpy(float), ce["y_centroid"].to_numpy(float)],
        array=None, lattice="none", image=None, meta=pd.DataFrame(index=ids.to_numpy()),
        platform="Xenium (cells)", unit="µm", notes=[],
    )


# ---- any platform: tables ---------------------------------------------------------------------

COORD_PATTERNS = [("x", "y"), ("x_centroid", "y_centroid"), ("centerx_global_px", "centery_global_px"),
                  ("center_x", "center_y"), ("imagecol", "imagerow"), ("pxl_col_in_fullres", "pxl_row_in_fullres"),
                  ("xcoord", "ycoord"), ("x_location", "y_location"), ("spatial_x", "spatial_y"), ("coord_x", "coord_y")]
NON_GENE = re.compile(r"^(negprb|neg_|negcontrol|systemcontrol|blank|unassigned|falsecode|deprecated)", re.IGNORECASE)


def find_coords(df: pd.DataFrame):
    low = {c.lower(): c for c in df.columns}
    for a, b in COORD_PATTERNS:
        if a in low and b in low:
            return low[a], low[b]
    return None


def read_tables(files: dict) -> dict:
    tabs = {k: v for k, v in files.items() if re.search(r"\.(csv|tsv|txt)(\.gz)?$", k, flags=re.IGNORECASE)}
    if not tabs:
        raise ValueError("No CSV/TSV tables found.")
    dfs = [read_table_any(p) for p in tabs.values()]
    has_xy = [find_coords(d) is not None for d in dfs]
    if not any(has_xy):
        raise ValueError("No coordinate columns found (for example x and y, or x_centroid and y_centroid).")
    ci = has_xy.index(True)
    co = dfs[ci]
    xy = find_coords(co)
    if len(dfs) == 1:
        skip = {"fov", "cell_id", "cell", "in_tissue", "array_row", "array_col"}
        genes = [c for c in co.columns if c not in xy and pd.api.types.is_numeric_dtype(co[c])
                 and not NON_GENE.match(str(c)) and str(c).lower() not in skip]
        if len(genes) < 2:
            raise ValueError("The table needs at least two numeric gene columns besides the coordinates.")
        ids = co.iloc[:, 0].astype(str).tolist() if co.iloc[:, 0].dtype == object else [f"spot{i}" for i in range(len(co))]
        return dict(counts=None, X=None, raw=sparse.csr_matrix(co[genes].to_numpy(float)), genes=[str(g) for g in genes],
                    spots=_unique(ids), coords=co[list(xy)].to_numpy(float), array=None, lattice="none", image=None,
                    meta=pd.DataFrame(index=_unique(ids)), platform="Table", unit="", notes=[])
    ex = dfs[[i for i in range(len(dfs)) if i != ci][0]]

    def key_of(d):
        low = {c.lower(): c for c in d.columns}
        if "fov" in low and "cell_id" in low:
            return (d[low["fov"]].astype(str) + "_" + d[low["cell_id"]].astype(str)).to_numpy()
        return d.iloc[:, 0].astype(str).to_numpy()

    ck, ek = key_of(co), key_of(ex)
    if np.isin(ek, ck).mean() > 0.5:
        drop = {"fov", "cell_id", "cell", "barcode"}
        genes = [c for j, c in enumerate(ex.columns) if j > 0 and str(c).lower() not in drop
                 and pd.api.types.is_numeric_dtype(ex[c]) and not NON_GENE.match(str(c))]
        mat = pd.DataFrame(ex[genes].to_numpy(float), index=ek, columns=[str(g) for g in genes])
    elif np.isin(np.asarray(ex.columns[1:], dtype=str), ck).mean() > 0.5:
        mat = ex.set_index(ex.columns[0]).T
        mat.columns = [str(c) for c in mat.columns]
        mat = mat.loc[:, [c for c in mat.columns if not NON_GENE.match(c)]].astype(float)
        mat.index = mat.index.astype(str)
    else:
        raise ValueError("Could not match the expression table to the coordinate table by cell or spot IDs.")
    shared = [k for k in mat.index if k in set(ck)]
    pos = pd.Series(np.arange(len(ck)), index=ck)
    co = co.iloc[pos[shared].to_numpy()]
    mat = mat.loc[shared]
    return dict(counts=None, X=None, raw=sparse.csr_matrix(mat.to_numpy()), genes=list(mat.columns), spots=shared,
                coords=co[list(xy)].to_numpy(float), array=None, lattice="none", image=None,
                meta=pd.DataFrame(index=shared), platform="Tables", unit="", notes=[])


# ---- AnnData (.h5ad) ---------------------------------------------------------------------------

def _h5_matrix(node):
    import h5py

    if isinstance(node, h5py.Dataset):
        return sparse.csr_matrix(np.asarray(node[()], dtype=np.float64))
    enc = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", ""))
    enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    shape = tuple(int(x) for x in node.attrs.get("shape", node.attrs.get("h5sparse_shape")))
    data = node["data"][()].astype(np.float64)
    if "csc" in enc:
        return sparse.csc_matrix((data, node["indices"][()], node["indptr"][()]), shape=shape).tocsr()
    return sparse.csr_matrix((data, node["indices"][()], node["indptr"][()]), shape=shape)


def _h5_index(group) -> list[str]:
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    if key in group:
        return _strings(group[key])
    if "index" in group:
        return _strings(group["index"])
    return []


def _h5_column(group, name):
    import h5py

    node = group[name]
    if isinstance(node, h5py.Group):  # categorical (anndata >= 0.8)
        cats = _strings(node["categories"])
        codes = node["codes"][()]
        return pd.Categorical.from_codes(np.where(codes < 0, -1, codes), categories=cats)
    vals = node[()]
    if "categories" in node.attrs:  # anndata 0.7: categories stored as a reference
        ref = node.attrs["categories"]
        cats = _strings(group.file[ref])
        return pd.Categorical.from_codes(vals, categories=cats)
    if vals.dtype.kind in "SO":
        return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in vals])
    return vals


def read_h5ad(path: str) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        obs_names = _h5_index(f["obs"])
        var_names = _h5_index(f["var"])
        X = _h5_matrix(f["X"])
        counts = None
        notes = []
        for key in ("counts", "raw_counts", "count"):
            if "layers" in f and key in f["layers"]:
                counts = _h5_matrix(f["layers"][key])
                notes.append(f"layers['{key}'] used as counts.")
                break
        if counts is None and "raw" in f and "X" in f["raw"]:
            raw_var = _h5_index(f["raw/var"])
            if raw_var == var_names:
                counts = _h5_matrix(f["raw/X"])
                notes.append("raw.X used as counts.")
        if "obsm" not in f or not any(k in f["obsm"] for k in ("spatial", "X_spatial", "spatial_coords")):
            raise ValueError("This AnnData file has no spatial coordinates (obsm['spatial']).")
        sk = next(k for k in ("spatial", "X_spatial", "spatial_coords") if k in f["obsm"])
        xy = np.asarray(f["obsm"][sk][()], dtype=float)[:, :2]
        meta = {}
        order = f["obs"].attrs.get("column-order", [])
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in np.asarray(order).ravel()] or \
               [k for k in f["obs"].keys() if not k.startswith("_")]
        for c in cols:
            if c in f["obs"]:
                try:
                    meta[c] = _h5_column(f["obs"], c)
                except Exception:  # noqa: BLE001 - skip unusual columns
                    pass
        image, scale = None, 1.0
        if "uns" in f and "spatial" in f["uns"]:
            sp = f["uns/spatial"]
            lib = next(iter(sp.keys()), None)
            if lib is not None and "images" in sp[lib]:
                imgs = sp[lib]["images"]
                key = "lowres" if "lowres" in imgs else ("hires" if "hires" in imgs else None)
                sf = sp[lib]["scalefactors"] if "scalefactors" in sp[lib] else {}
                skey = f"tissue_{key}_scalef"
                if key and skey in sf:
                    img = np.asarray(imgs[key][()])
                    if img.dtype != np.uint8:
                        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
                    image, scale = img[..., :3], float(np.asarray(sf[skey][()]))
    meta = pd.DataFrame(meta, index=obs_names)
    arr = None
    lattice = "none"
    if {"array_row", "array_col"} <= set(meta.columns):
        arr = np.c_[np.asarray(meta["array_row"], int), np.asarray(meta["array_col"], int)]
        lattice = "hex"
    if looks_like_counts(X) and counts is None:
        counts, X = X, None
    elif counts is not None and X is not None and (X.data < 0).any():
        X = None  # scaled values: recompute from counts
    return dict(counts=counts, X=X, raw=None, genes=_unique(var_names), spots=obs_names, coords=xy * scale,
                array=arr, lattice=lattice, image=image, meta=annotation_columns(meta),
                platform="AnnData (.h5ad)", unit="pixels" if image is not None else "", notes=notes)


# ---- annotations, normalization, lattice, binning ---------------------------------------------

def annotation_columns(md: pd.DataFrame) -> pd.DataFrame:
    keep = {}
    for c in md.columns:
        if re.match(r"^(nCount|nFeature|percent|nUMI|nGene|n_counts|n_genes|total_counts|log1p|pct_)", str(c)):
            continue
        v = md[c]
        k = pd.Series(v).nunique(dropna=True)
        is_int = pd.api.types.is_integer_dtype(v) or (pd.api.types.is_float_dtype(v) and np.all(np.mod(v[~pd.isna(v)], 1) == 0))
        if (isinstance(v.dtype, pd.CategoricalDtype) or v.dtype == object or pd.api.types.is_bool_dtype(v)
                or (is_int and k <= 40)) and 2 <= k <= 60:
            keep[c] = pd.Categorical(v.astype(str))
    return pd.DataFrame(keep, index=md.index)


def looks_like_counts(m) -> bool:
    x = m.data if sparse.issparse(m) else np.asarray(m).ravel()
    x = x[:200_000]
    return x.size > 0 and bool(np.all(np.abs(x - np.round(x)) < 1e-8)) and float(x.max()) > 20


def normalize(d: dict) -> dict:
    if d.get("X") is not None:
        d.setdefault("normalization", "values as provided (already normalized)")
        return d
    if d.get("counts") is None and d.get("raw") is not None:
        if looks_like_counts(d["raw"]):
            d["counts"] = d.pop("raw")
        else:
            d["X"] = d.pop("raw").tocsr()
            d["normalization"] = "values as uploaded"
            return d
    c = d["counts"].tocsr().astype(np.float64)
    cs = np.asarray(c.sum(axis=1)).ravel()
    cs[cs == 0] = 1
    X = sparse.diags(1e4 / cs) @ c
    X = X.tocsr()
    X.data = np.log1p(X.data)
    d["X"] = X
    d["normalization"] = "log1p(counts per 10,000)"
    return d


def detect_lattice(xy: np.ndarray):
    from scipy.spatial import cKDTree

    n = len(xy)
    if n < 20:
        return "none", 6, None
    rng = np.random.default_rng(0)
    s = rng.choice(n, 3000, replace=False) if n > 3000 else np.arange(n)
    d, _ = cKDTree(xy).query(xy[s], k=min(9, n))
    d1 = float(np.median(d[:, 1]))
    cnt = float(np.median((d[:, 1:] <= 1.15 * d1).sum(axis=1)))
    if cnt >= 5.5:
        return "hex", 6, d1
    if 3.5 <= cnt <= 4.5:
        return "square", 4, d1
    return "none", 6, d1


def square_graph(r, c):
    key = {(a, b): i for i, (a, b) in enumerate(zip(r, c))}
    ii, jj = [], []
    for dr, dc in ((0, 1), (1, 0)):
        for i, (a, b) in enumerate(zip(r, c)):
            j = key.get((a + dr, b + dc))
            if j is not None:
                ii.append(i)
                jj.append(j)
    n = len(r)
    a = sparse.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    a = (a + a.T).tocsr()
    a.data[:] = 1.0
    return a


def build_graph(d: dict, neighbors: int | None = None):
    arr = d.get("array")
    if arr is not None and len({(a, b) for a, b in arr}) == len(arr):
        if d["lattice"] == "hex":
            return jx.visium_graph(arr[:, 0], arr[:, 1]), "Visium hexagonal lattice (6 neighbours)"
        if d["lattice"] == "square":
            return square_graph(arr[:, 0], arr[:, 1]), "square lattice (4 neighbours)"
    kind, k, d1 = detect_lattice(d["coords"])
    k = neighbors or k
    if kind in ("hex", "square"):
        name = "hexagonal" if kind == "hex" else "square"
        return jx.spatial_graph(d["coords"], neighbors=k, radius=1.2 * d1), f"{name} lattice detected from coordinates ({k} neighbours)"
    return jx.spatial_graph(d["coords"], neighbors=k), f"{k} nearest neighbours"


def bin_dataset(d: dict, size: float | None = None, merge: int | None = None) -> dict:
    """Sum counts (or average values) of the spots or cells in each square bin."""
    if merge is not None and d.get("array") is not None:
        br, bc = d["array"][:, 0] // merge, d["array"][:, 1] // merge
    else:
        x, y = d["coords"][:, 0], d["coords"][:, 1]
        br, bc = np.floor((y - y.min()) / size).astype(int), np.floor((x - x.min()) / size).astype(int)
    keys = pd.Series(list(zip(br, bc)))
    codes, uniq = pd.factorize(keys)
    nb = len(uniq)
    A = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(nb, len(codes)))
    n_in = np.asarray(A.sum(axis=1)).ravel()
    out = dict(d)
    if d.get("counts") is not None:
        out["counts"] = (A @ d["counts"]).tocsr()
        out["X"] = None
        out["raw"] = None
    else:
        src = d["X"] if d.get("X") is not None else d["raw"]
        out["raw"] = (sparse.diags(1 / n_in) @ (A @ src)).tocsr()
        out["X"] = None
    out["coords"] = np.c_[(A @ d["coords"][:, 0]) / n_in, (A @ d["coords"][:, 1]) / n_in]
    out["array"] = np.array([list(u) for u in uniq], dtype=int)
    out["lattice"] = "square"
    out["spots"] = [f"bin_{a}_{b}" for a, b in uniq]
    meta = {}
    for c in d["meta"].columns:
        tab = pd.crosstab(codes, np.asarray(d["meta"][c].astype(str)))
        meta[c] = pd.Categorical(tab.idxmax(axis=1).reindex(range(nb)).to_numpy())
    out["meta"] = pd.DataFrame(meta, index=out["spots"])
    out["bin_label"] = (f"{merge} x {merge} lattice positions per bin" if merge is not None
                        else f"square bins of {size:.3g} {d.get('unit', '')}".strip())
    return out


def auto_bin_size(d: dict, target: int) -> float:
    x, y = d["coords"][:, 0], d["coords"][:, 1]
    s = np.sqrt(np.ptp(x) * np.ptp(y) / target)
    for _ in range(6):
        nb = len(set(zip(np.floor((y - y.min()) / s).astype(int), np.floor((x - x.min()) / s).astype(int))))
        if nb <= target:
            break
        s *= np.sqrt(nb / target) * 1.02
    return float(s)


def auto_lattice_factor(d: dict, target: int) -> int:
    f = 1
    while True:
        nb = len(set(zip(d["array"][:, 0] // f, d["array"][:, 1] // f)))
        if nb <= target or f > 64:
            return f
        f += 1


# ---- gene panels ------------------------------------------------------------------------------

def gene_pool(X, genes, min_frac=0.05, max_genes=3000):
    det = np.asarray((X > 0).mean(axis=0)).ravel()
    ok = np.flatnonzero(det >= min_frac)
    if not len(ok):
        ok = np.argsort(-det)[: min(len(genes), 200)]
    sub = X[:, ok]
    m = np.asarray(sub.mean(axis=0)).ravel()
    v = np.asarray(sub.multiply(sub).mean(axis=0)).ravel() - m ** 2
    return ok[np.argsort(-v)[:max_genes]]


def morans_i(X, graph, idx):
    rs = np.asarray(graph.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    W = sparse.diags(1 / rs) @ graph
    out = np.full(len(idx), np.nan)
    for start in range(0, len(idx), 500):
        cols = idx[start:start + 500]
        Z = X[:, cols].toarray()
        Z -= Z.mean(axis=0)
        num = (Z * (W @ Z)).sum(axis=0)
        den = (Z ** 2).sum(axis=0)
        out[start:start + 500] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def select_panel(d: dict, graph, method="svg", n=100, custom=()):
    genes = d["genes"]
    upper = {g.upper(): i for i, g in enumerate(genes)}
    wanted = [upper[c.upper()] for c in custom if c.upper() in upper]
    missing = [c for c in custom if c.upper() not in upper]
    if method == "custom":
        return list(dict.fromkeys(wanted)), missing
    pool = gene_pool(d["X"], genes)
    if method == "svg":
        score = morans_i(d["X"], graph, pool)
    else:
        sub = d["X"][:, pool]
        m = np.asarray(sub.mean(axis=0)).ravel()
        score = np.asarray(sub.multiply(sub).mean(axis=0)).ravel() - m ** 2
    order = pool[np.argsort(-np.nan_to_num(score, nan=-np.inf))][:n]
    return list(dict.fromkeys(wanted + list(order))), missing
