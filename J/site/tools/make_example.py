"""Build the one-click example for the apps from a public 10x Genomics Visium dataset.

Mouse Brain Serial Section 1 (Sagittal-Anterior), Space Ranger 1.1.0, 10x Genomics, CC BY 4.0.
The count matrix is reduced to the most variable genes (plus anatomical markers) and stored in the
compact ".jx10x" binary that both apps read quickly in the browser.

Usage: python make_example.py <output dir> [--genes 2000]
"""
import argparse
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request

import h5py
import numpy as np
from scipy import sparse

SAMPLE = "V1_Mouse_Brain_Sagittal_Anterior"
BASE = f"https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/{SAMPLE}/{SAMPLE}_"
MARKERS = ["Hpca", "Ttr", "Mbp", "Plp1", "Nrgn", "Calb2", "Pcp4", "Gfap", "Snap25", "Cck", "Penk", "Ppp1r1b",
           "Rgs14", "Prox1", "C1ql2", "Slc17a7", "Gad1", "Mobp", "Enpp2", "Prkcd"]


def strings(ds):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]]


def write_jx10x(path, m, barcodes, names, source):
    """m: csc genes x barcodes with integer counts. Same layout as the browser converter (h5convert.js)."""
    m = m.tocsc()
    m.sort_indices()
    header = {"shape": [int(m.shape[0]), int(m.shape[1])], "nnz": int(m.nnz), "n_indptr": int(len(m.indptr)),
              "data_type": "int", "barcodes": list(barcodes), "names": list(names), "ids": list(names),
              "types": ["Gene Expression"] * m.shape[0], "source": source, "file": os.path.basename(path)}
    h = json.dumps(header).encode("utf-8")
    pad = (4 - ((8 + 4 + len(h)) % 4)) % 4
    with open(path, "wb") as f:
        f.write(b"JX10X1\n\0")
        f.write(np.uint32(len(h)).astype("<u4").tobytes())
        f.write(h)
        f.write(b"\0" * pad)
        f.write(m.indptr.astype("<i4").tobytes())
        f.write(m.indices.astype("<i4").tobytes())
        f.write(np.rint(m.data).astype("<i4").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--genes", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    h5_path = os.path.join(tmp, "matrix.h5")
    tar_path = os.path.join(tmp, "spatial.tar.gz")
    urllib.request.urlretrieve(BASE + "filtered_feature_bc_matrix.h5", h5_path)
    urllib.request.urlretrieve(BASE + "spatial.tar.gz", tar_path)

    with h5py.File(h5_path, "r") as f:
        g = f["matrix"]
        m = sparse.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g["shape"][()]))
        names = np.asarray(strings(g["features/name"]))
        types = np.asarray(strings(g["features/feature_type"]))
        barcodes = strings(g["barcodes"])
    keep = types == "Gene Expression"
    m, names = m[keep, :], names[keep]

    # choose genes: detected in >= 2% of spots, most variable after log-normalization, plus markers
    mt = m.T.tocsr().astype(float)
    cs = np.asarray(mt.sum(axis=1)).ravel()
    norm = sparse.diags(1e4 / np.maximum(cs, 1)) @ mt
    norm.data = np.log1p(norm.data)
    det = np.asarray((mt > 0).mean(axis=0)).ravel()
    mean = np.asarray(norm.mean(axis=0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(axis=0)).ravel() - mean ** 2
    var[det < 0.02] = -1
    chosen = list(np.argsort(-var)[: a.genes])
    for g_ in MARKERS:
        hit = np.flatnonzero(names == g_)
        if len(hit) and hit[0] not in chosen:
            chosen.append(int(hit[0]))
    chosen = sorted(set(chosen))
    write_jx10x(os.path.join(a.out, "filtered_feature_bc_matrix.jx10x"), m[chosen, :], barcodes, names[chosen],
                f"10x Genomics {SAMPLE} (CC BY 4.0), {len(chosen)} genes")

    with tarfile.open(tar_path) as t:
        t.extractall(tmp)
    sp = os.path.join(tmp, "spatial")
    for fn in ("tissue_lowres_image.png", "scalefactors_json.json", "tissue_positions_list.csv"):
        shutil.copy(os.path.join(sp, fn), os.path.join(a.out, fn))
    with open(os.path.join(a.out, "README.txt"), "w") as f:
        f.write("Mouse Brain Serial Section 1 (Sagittal-Anterior), Visium, Space Ranger 1.1.0.\n"
                "Source: 10x Genomics, https://www.10xgenomics.com/datasets (CC BY 4.0).\n"
                f"Reduced to {len(chosen)} genes (most variable plus anatomical markers) for the juxtaposed apps.\n")
    print("example written:", a.out, m.shape[1], "spots,", len(chosen), "genes")


if __name__ == "__main__":
    main()
