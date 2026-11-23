"""Recompute the existing local reference scores on the same 916 annotated pairs."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from experiment import OLD, V5, ExperimentGraph


def local_references(project,out):
    sys.path[:0] = [str(project/"revisions"/V5/"packages/python/src"),
                    str(project/"revisions"/OLD),str(project/"benchmarks/v3")]
    import juxtaposed as jx
    import seed_weighted_j
    from datasets import load_dlpfc, load_seqfish, lognorm
    from common import hex_xy
    from bench_jaccard import LocalMaps
    from bench_gold_accuracy import seq_gold
    from gold import GoldStandards

    chosen = pd.read_csv(out/"observed_edges.csv.gz",dtype={"unit":str}).query("method=='J1'")
    seq, seq_genes, seq_meta = load_seqfish()
    results = []
    v5 = project/"revisions"/V5/"results"
    keep = {"J_local_s1_ridge1":("J1 reference check",1),
            "J_local_lazy4_ridge1":("Local J lazy4",1),
            "Q_signed_cross_energy_s1":("Signed gradient Q",1),
            "unsigned_gradient_s1":("Absolute gradient",1),
            "joint_gradient_energy_s1":("Joint energy",1),
            "spatialdm_local_moran":("SpatialDM local",1),
            "liana_local_pearson":("LIANA+ local Pearson", -1),
            "liana_local_cosine":("LIANA+ local cosine",1)}
    for (design,unit),rows in chosen.groupby(["design","unit"]):
        orig = pd.read_csv(next(v5.glob(f"{design}_{unit}_pairs_*.csv.gz")))
        if design=="B1":
            count,genes,meta = load_dlpfc(unit)
            expr = lognorm(count)
            graph = jx.visium_graph(meta.array_row,meta.array_col)
            gold = GoldStandards(hex_xy(meta.array_row,meta.array_col),graph)
            label = meta.layer.to_numpy()
            type_x,type_y = "layer_x","layer_y"
        else:
            mask = seq_meta.embryo.eq(unit).to_numpy()
            genes,expr,meta = seq_genes,seq[mask],seq_meta.loc[mask]
            coords = meta[["x_global_affine","y_global_affine"]].to_numpy()
            graph = jx.spatial_graph(coords,neighbors=6)
            gold,_ = seq_gold(coords,graph)
            label = meta.celltype_mapped_refined.to_numpy()
            type_x,type_y = "type_x","type_y"
        merged = rows.merge(orig[["gene_x","gene_y",type_x,type_y]],on=["gene_x","gene_y"],validate="one_to_one")
        assert len(merged)==len(rows)
        use = np.flatnonzero(np.isin(genes,np.union1d(merged.gene_x,merged.gene_y)))
        idx = pd.Index(genes[use])
        m = LocalMaps(graph,gold.W_liana,gold.W_sdm)
        raw = expr[:,use].toarray().astype(float)
        pa, pb = idx.get_indexer(merged.gene_x), idx.get_indexer(merged.gene_y)
        scores = m.maps(raw,pa,pb)
        parity = ExperimentGraph(graph,raw[:, :2],jx,seed_weighted_j,phase_only=True)
        d = parity.t.gradient(parity.t.heat @ jx.standardize(raw))
        k0 = 0
        np.testing.assert_allclose(scores["J_local_s1_ridge1"][:,k0],
                                   parity.local(d[:,pa[k0]],d[:,pb[k0]]),atol=1e-12,rtol=0)
        for k,row in enumerate(merged.itertuples()):
            a,b = getattr(row,type_x),getattr(row,type_y)
            actual = ((label[m.i]==a)&(label[m.j]==b))|((label[m.i]==b)&(label[m.j]==a))
            for key,(method,sign) in keep.items():
                ap = average_precision_score(actual,sign*scores[key][:,k])
                results.append(dict(design=design,unit=unit,donor=row.donor,gene_x=row.gene_x,gene_y=row.gene_y,
                                    method=method,ap=ap))
        print(f"{design} {unit} {len(merged)} matched local comparisons",flush=True)
    pd.DataFrame(results).to_csv(out/"matched_local_references.csv.gz",index=False)


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    local_references(args.project.resolve(),args.out)
