"""Audit the DLPFC targeted validation and match the rotated-control parameters."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def reference_genes(pilot, genes):
    targets = ("L6","WM")
    gains = np.zeros((len(targets),len(genes)))
    levels = gains.copy()
    schedule = [s for s in pilot.SECTIONS if pilot.DONOR[s]!="Br5292"]
    for section in schedule:
        counts,sects_genes,meta = pilot.load_dlpfc(section)
        np.testing.assert_array_equal(genes,sects_genes)
        x = pilot.lognorm(counts)
        means = {}
        for layer in pilot.LAYER_ORDER:
            match = meta.layer.eq(layer).to_numpy()
            if match.sum()>=30:
                means[layer] = np.asarray(x[match].mean(axis=0)).ravel()
        for i,layer in enumerate(targets):
            levels[i] += means[layer]
            others = np.maximum.reduce([v for k,v in means.items() if k!=layer])
            gains[i] += means[layer]-others
    mt = np.char.startswith(np.char.upper(genes.astype(str)), "MT-")
    picks = []
    for i in range(len(targets)):
        cand = np.flatnonzero((gains[i]>0)&(levels[i]/len(schedule)>=.1)&~mt)
        picks.append(cand[np.argsort(-gains[i,cand])[:3]])
    return np.array(picks), mt


def control(project, out):
    sys.path.insert(0,str(project/"experiments"))
    import seed_j_pilot as pilot
    import seed_j_real_validation as original
    import seed_j_remedies as remedies
    out.mkdir(parents=True,exist_ok=True)
    a, genes, ma = pilot.load_dlpfc("151507")
    b, gb, mb = pilot.load_dlpfc("151673")
    np.testing.assert_array_equal(genes,gb)
    xa, xb = pilot.lognorm(a), pilot.lognorm(b)
    ref, mt = reference_genes(pilot, genes)
    assert len(set(ref.ravel()))==6
    graph = pilot.build_visium_graph(ma.array_row.values,ma.array_col.values)
    ref_maps = pilot.smooth_signal(original.standardize(xa[:,ref.ravel()].toarray()),graph["S"],4)
    accum = np.zeros(len(ma))
    for x in ref_maps.T[:3]:
        for y in ref_maps.T[3:]:
            v,*_ = pilot.spot_local_j(x,y,graph,ridge=1.)
            accum += np.maximum(v,0.)
    raw, _ = pilot.find_first_elbow_seeds(accum/9)
    filt, coh, *_ = remedies.filter_clustered_seeds(raw,graph,min_cluster_size=5,two_hop=True)
    seeds = filt if len(filt) else raw
    assert len(seeds)==57 and .65<coh<.67
    a_keys = {(int(r),int(c)):i for i,(r,c) in enumerate(zip(ma.array_row,ma.array_col))}
    b_keys = {(77-int(r),127-int(c)):i for i,(r,c) in enumerate(zip(mb.array_row,mb.array_col))}
    keys = sorted(a_keys.keys()&b_keys.keys())
    ia,ib = (np.array([m[key] for key in keys]) for m in (a_keys,b_keys))
    shared = pilot.build_visium_graph(ma.array_row.values[ia],ma.array_col.values[ia])
    seed_idx = np.flatnonzero(np.isin(ia,seeds))
    distance = cKDTree(shared["coords"][seed_idx]).query(shared["midpoints"])[0]
    mu,u = eigh(shared["L"].toarray())
    adjacency = original.spatial_weights(len(ia),shared["edges"])
    rankings = []
    for c,x,rows in [(a,xa,ia),(b,xb,ib)]:
        eligible = np.flatnonzero((np.asarray((c[rows]>0).mean(axis=0)).ravel()>=.1)&~mt)
        z = original.standardize(x[rows][:,eligible].toarray())
        rankings.append(eligible[np.argsort(-original.moran_i_vec(z,adjacency))])
    results, panels = [],[]
    for scale,bandwidth,drop in [(1,3,False),(4,4,False),(4,4,True)]:
        choose = [(v[~np.isin(v,ref.ravel())][:50] if drop else v[:50]) for v in rankings]
        za,zb = [original.standardize(x[idx][:,ix].toarray()) for x,idx,ix in zip([xa,xb],[ia,ib],choose)]
        fourier_a, fourier_b = (u.T@za)**2,(u.T@zb)**2
        left,right = [pilot.smooth_signal(v,shared["S"],scale) for v in (za,zb)]
        transfer = (1-mu/(2*shared["deg"].max()))**scale
        h = mu*transfer**2
        base_cross = left.T@(shared["L"]@right)
        base_variance = fourier_a.T@((h*h)[:,None]*fourier_b)
        mask = .05+.95*np.exp(-.5*(distance/bandwidth)**2)
        weighted = remedies.build_weighted_laplacian(len(ia),shared["edges"],mask)
        transformed = (u.T@(weighted@u))*np.outer(transfer,transfer)
        mapped_cross = left.T@(weighted@right)
        mapped_variance = fourier_a.T@((transformed*transformed)@fourier_b)
        p_std = 2*norm.sf(np.abs(base_cross)/np.sqrt(np.maximum(base_variance,1e-18)))
        p_fixed = 2*norm.sf(np.abs(mapped_cross)/np.sqrt(np.maximum(mapped_variance,1e-18)))
        if scale==1:
            np.testing.assert_allclose([(p_std<.05).mean(),(p_fixed<.05).mean()],[.0524,.058],atol=1e-12)
        for method,p in [("Standard J",p_std),("Map-Fixed J",p_fixed)]:
            results.append(dict(scale=scale,bandwidth=bandwidth,excluded_discovery_genes=drop,method=method,
                                n=len(p.ravel()),called=int((p<.05).sum()),rejection_rate=(p<.05).mean(),
                                n_shared_spots=len(ia),n_retained_seeds=len(seed_idx),
                                n_discovery_genes_in_panel=sum(len(np.intersect1d(v,ref)) for v in choose)))
        rows = pd.DataFrame(dict(gene_a=np.repeat(genes[choose[0]],50),gene_b=np.tile(genes[choose[1]],50),
                                 scale=scale,bandwidth=bandwidth,excluded_discovery_genes=drop,
                                 p_standard=p_std.ravel(),p_fixed=p_fixed.ravel()))
        panels.append(rows)
    pd.DataFrame(results).to_csv(out/"rotated_control_parameter_check.csv",index=False)
    pd.concat(panels).to_csv(out/"rotated_control_pvalues.csv.gz",index=False)
    target = pd.read_csv(project/"experiments/seed_j_real_validation_results/val2_dlpfc151507_l6wm_pairs.csv")
    label = target.pair_type.eq("L6_vs_WM")
    rows = []
    for tag,score in [("std","J1_std"),("mapfixed","J1_mapfixed")]:
        rows.append(dict(method=tag,target_pairs=label.sum(),total_pairs=len(label),
                         median_j=target.loc[label,score].median(),
                         median_z=target.loc[label,"z_"+tag].median(),
                         auroc_j=roc_auc_score(label,target[score]),
                         n_called=((target.loc[label,"q_"+tag]<.05)&(target.loc[label,"z_"+tag]>0)).sum()))
    pd.DataFrame(rows).to_csv(out/"verified_target_counts.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    control(a.project.resolve(),a.out)
