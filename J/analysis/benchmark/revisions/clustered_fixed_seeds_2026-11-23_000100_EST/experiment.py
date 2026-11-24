"""Seed clustering, fixed panel, and diffusion candidates for spatial boundary scoring."""
from argparse import ArgumentParser
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply, splu
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score

RNG_SEED = 20261123
B2_SEED = 20261122
VERSIONS = ("elbow_J", "cluster_J", "fixed_signed_J", "fixed_positive_J")
FIXED = ("fixed_signed_J", "fixed_positive_J")
FILTERS = ("wavelet_J", "resolvent_J", "anisotropic_J")
BASE = ("J1", "J4", "Pearson", "Lee_L", "Regional_J")
METHODS = BASE + VERSIONS + FILTERS
LOCAL = ("J1", *VERSIONS, *FILTERS)
OLD = "seed_weighting_2026-11-22_234926_EST"
V5 = "v5_2026-11-22_104348_EST"


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def opposition(x, y, w=1.):
    return ratio(-2*np.sum(w*x*y, axis=0), np.sum(w*(x*x+y*y), axis=0))


def smooth(values, operator, steps):
    for _ in range(steps):
        values = operator @ values
    return values


class ExperimentGraph:
    def __init__(self, graph, panel, jx, seed_module, phase_only=False):
        self.graph = graph
        self.jx = jx
        self.seed = seed_module
        self.t = seed_module.Tissue(graph, jx)
        self.i, self.j, self.w = self.t.i, self.t.j, self.t.w
        self.n = graph.shape[0]
        self.lap, degrees = jx.laplacian(graph)
        self.dt = 1/(2*degrees.max())
        self.phase_only = phase_only
        if phase_only:
            self.phase = (-2j*np.pi*self.dt)*self.lap
            self.phase_trace = self.phase.diagonal().sum()
            self.counts = dict.fromkeys(FIXED,0)
            return
        self.resolvent = splu((sparse.eye(self.n)+4*self.dt*self.lap).tocsc())
        raw_panel = jx.standardize(panel)
        dv = self.t.gradient(raw_panel)
        rms = np.sqrt(np.mean(dv*dv, axis=1))
        conductance = 1/(1+(rms/np.median(rms[rms>0]))**2)
        conductance *= self.w
        adapted = sparse.csr_matrix((np.tile(conductance, 2), (np.r_[self.i,self.j], np.r_[self.j,self.i])),
                                    shape=graph.shape)
        self.anisotropic = jx.smoother(adapted)
        self.fixed_weights, self.counts = self.build_fixed(self.t.heat @ raw_panel)

    def local_from_terms(self, cross, energy):
        cx, ex = self.t.incidence @ cross, self.t.incidence @ energy
        if cross.ndim == 1:
            ridge = self.t.edge_mass * energy.sum()/self.w.sum()
        else:
            ridge = self.t.edge_mass[:, None] * energy.sum(axis=0)/self.w.sum()
        return ratio(cx[self.i]+cx[self.j], ex[self.i]+ex[self.j]+ridge)

    def local(self, dx, dy):
        cross = -2*self.w*dx*dy
        energy = self.w*(dx*dx+dy*dy)
        return self.local_from_terms(cross, energy)

    def node_mean(self, score):
        return ratio(self.t.incidence @ score, self.t.degree)

    def clustered(self, candidates):
        if not len(candidates):
            return candidates
        _, components = connected_components(self.graph[candidates][:,candidates], directed=False)
        return candidates[np.bincount(components)[components] >= 3]

    def weights(self, candidates):
        node_weights = self.t.weights(candidates)
        return (node_weights[self.i]+node_weights[self.j])/2

    def build_fixed(self, panel):
        d = self.t.gradient(panel)
        aa, bb = np.triu_indices(panel.shape[1], 1)
        signed = np.zeros(len(self.w))
        positive = signed.copy()
        for start in range(0,len(aa),32):
            dx, dy = d[:,aa[start:start+32]], d[:,bb[start:start+32]]
            block = self.local_from_terms(-2*self.w[:,None]*dx*dy, self.w[:,None]*(dx*dx+dy*dy))
            signed += block.sum(axis=1)
            positive += np.maximum(block, 0).sum(axis=1)
        windows, counts = {}, {}
        for name, field in zip(FIXED, [signed, positive]):
            seeds = self.clustered(self.seed.first_elbow(self.node_mean(field/len(aa))))
            windows[name] = self.weights(seeds)
            counts[name] = len(seeds)
        return windows, counts

    def prepare(self, raw):
        z = self.jx.standardize(raw)
        z1 = self.t.heat @ z
        if self.phase_only:
            phases = expm_multiply(self.phase,z,traceA=self.phase_trace)-z
            return {"J1":self.t.gradient(z1)}, phases
        z4 = smooth(z1,self.t.heat,3)
        z16 = smooth(z4,self.t.heat,12)
        za = smooth(z,self.anisotropic,4)
        zr = self.resolvent.solve(z)
        grads = {name: self.t.gradient(x) for name, x in
                 zip(("J1","J4","resolvent_J","anisotropic_J"),(z1,z4,zr,za))}
        return grads, z1-z16

    def score(self, prepared, a, b, make_maps=False):
        g, band = prepared
        dx, dy = g["J1"][:,a], g["J1"][:,b]
        if self.phase_only:
            x,y = band[:,a],band[:,b]
            signed = -2*np.real(x.conj()*y)
            den = np.sum(np.abs(x)**2+np.abs(y)**2)
            base = {"J1":float(opposition(dx,dy,self.w)), "phase_J":float(ratio(signed.sum(),den))}
            if not make_maps:
                return base,{}
            return base,{},{"J1":self.local(dx,dy),"phase_J":(signed[self.i]+signed[self.j])/2}
        loc = self.local(dx, dy)
        all_seeds = self.seed.first_elbow(self.node_mean(loc))
        cluster = self.clustered(all_seeds)
        weights = {"elbow_J": self.weights(all_seeds), "cluster_J": self.weights(cluster), **self.fixed_weights}
        base = {key: float(opposition(d[:,a],d[:,b],self.w)) for key,d in g.items()}
        base.update({k:float(opposition(dx,dy,self.w*w)) for k,w in weights.items()})
        base["wavelet_J"] = float(opposition(band[:,a],band[:,b]))
        counts = {"elbow_J":len(all_seeds), "cluster_J":len(cluster), **self.counts}
        if not make_maps:
            return base, counts
        local_maps = {"J1":loc, **{k:loc+(w-1)*np.maximum(loc,0) for k,w in weights.items()}}
        local_maps.update({key:self.local(g[key][:,a],g[key][:,b]) for key in ("resolvent_J","anisotropic_J")})
        cross = -2*band[:,a]*band[:,b]
        local_maps["wavelet_J"] = (cross[self.i]+cross[self.j])/2
        return base, counts, local_maps


def auc_rows(frame, keys, extras=(), phase_only=False):
    out = []
    for name, group in frame.groupby(keys):
        for neg in ("same","apart"):
            subset = group[group.pair_class.isin(["touch",neg])]
            label = subset.pair_class.eq("touch")
            if not label.any() or label.all():
                continue
            for method in (("J1","phase_J") if phase_only else METHODS+extras):
                score = subset[method].to_numpy()
                if neg=="same" and method in ("Pearson","Lee_L","Regional_J",*extras):
                    score = -score
                out.append(dict(zip(keys,name), task=f"touch_vs_{neg}", method=method,
                                auroc=roc_auc_score(label,score), ap=average_precision_score(label,score),
                                n_positive=label.sum(), n_negative=(~label).sum()))
    return pd.DataFrame(out)


def spatial_grid(n):
    one = sparse.diags([np.ones(n-1),np.ones(n-1)],[-1,1],shape=(n,n))
    xy = np.meshgrid(np.arange(n),np.arange(n))
    return sparse.kronsum(one,one,format="csr"), xy[0].ravel(),xy[1].ravel()


def two_fields(signal, heat, rng):
    noise = rng.normal(size=signal.shape)
    return signal+.6*noise+.6*(heat@(heat@rng.normal(size=signal.shape)))


def simulation(out, jx, seed_module, weak_panel=False, phase_only=False):
    graph, xx, yy = spatial_grid(30)
    signal_rng = np.random.default_rng(RNG_SEED)
    panel_rng = np.random.default_rng([RNG_SEED,1])
    heat = jx.smoother(graph)
    rows, maps = [], []
    # Twelve shared geometries and eight independent replicates spread over them.
    geometries = [(int(signal_rng.integers(10,20)),int(signal_rng.integers(8,22)),int(signal_rng.integers(3,7)))
                  for _ in range(12)]
    for geometry in ("broad","focal"):
        for gid,(split,center,width) in enumerate(geometries):
            height = np.exp(-.5*((yy-center)/width)**2) if geometry=="focal" else np.ones_like(yy)
            left, right = (xx<split)*height,(xx>=split)*height
            panel_signal = np.column_stack([left]*16+[right]*16)
            panel_noise = two_fields(np.zeros_like(panel_signal),heat,panel_rng)
            strong_model = ExperimentGraph(graph,panel_signal+panel_noise,jx,seed_module,phase_only)
            truth = strong_model.t.interface((xx<split)&(height>=.65),(xx>=split)&(height>=.65))
            order = panel_rng.permutation(len(truth))
            for amp in (.1,.25,.5,1.):
                model = ExperimentGraph(graph,amp*panel_signal+panel_noise,jx,seed_module,phase_only) if weak_panel else strong_model
                for rep in range(8):
                    left_ref = amp*left
                    right_ref = [amp*right,amp*height*(xx>=split+4),amp*left]
                    for pair_class,b in zip(("touch","apart","same"),right_ref):
                        expression = two_fields(np.column_stack([left_ref,b]),heat,signal_rng)
                        scores, seeds, local = model.score(model.prepare(expression),0,1,make_maps=True)
                        scores.update(Pearson=np.corrcoef(expression.T)[0,1],
                                      Lee_L=jx.lee_l_matrix(expression,graph)[0,1],
                                      Regional_J=jx.regional_juxtaposition_matrix(expression,graph)[0,1])
                        context = dict(geometry=geometry,amp=amp,geometry_id=gid,replicate=rep,pair_class=pair_class)
                        rows.append(dict(**context,**scores,**{f"n_{k}":v for k,v in seeds.items()}))
                        if pair_class=="touch":
                            for method,(ap,iou) in seed_module.localization(truth,local,order).items():
                                maps.append(dict(**context,method=method,ap=ap,iou=iou,prevalence=truth.mean()))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out/"simulation_pairs.csv.gz",index=False)
    auc_rows(pairs,["geometry","amp"],phase_only=phase_only).to_csv(out/"simulation_ranking.csv",index=False)
    pd.DataFrame(maps).to_csv(out/"simulation_edges.csv.gz",index=False)


def code_seed(v):
    return zlib.crc32(str(v).encode())


def b2(out, project, jx, seed_module, data, phase_only=False):
    original = pd.read_csv(project/"results/v3/gold_b2_semisynthetic.csv",dtype={"section":str})
    all_rows, all_local = [], []
    for sec in data.SECTIONS:
        t0 = time.monotonic()
        counts,_,meta = data.load_dlpfc(sec)
        g = jx.visium_graph(meta.array_row,meta.array_col)
        heat = jx.smoother(g)
        xy = np.column_stack((meta.array_col/2,meta.array_row*np.sqrt(3)/2))
        dist = np.sqrt(3)*cdist(xy,xy)/3
        cov = (1+dist)*np.exp(-dist)
        cov.flat[::len(xy)+1] += 1e-6
        chol = cholesky(cov,lower=True,check_finite=False,overwrite_a=True)
        del cov, dist
        lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        test_rng = np.random.default_rng([B2_SEED,int(sec)])
        panel_rng = np.random.default_rng([RNG_SEED,int(sec)])
        layers = meta.layer.to_numpy()
        present = [l for l in data.LAYER_ORDER if (layers==l).sum()>=30]
        layout = {l:heat@(layers==l).astype(float) for l in present}
        markers = np.repeat(present,10)
        lab_index = np.array([data.LAYER_ORDER.index(x) for x in markers])
        a,b = np.triu_indices(len(markers),1)
        dist_cls = np.abs(lab_index[a]-lab_index[b])
        pair_class = np.where(dist_cls==0,"same",np.where(dist_cls==1,"touch","apart"))

        def generate(mark, strength, rng):
            z = np.column_stack([layout[l] for l in mark])
            noise = .3*(chol@rng.normal(size=z.shape))
            mu = (lib/lib.mean())[:,None]*np.exp(strength*z+noise)
            mu *= 2/mu.mean(axis=0)
            return np.log1p(1e4*rng.poisson(mu)/lib[:,None])

        # Draw independent discovery genes from the same weak beta and geometry.
        panel_mark = np.resize(np.asarray(present),32)
        panel = generate(panel_mark,.5,panel_rng)
        model = ExperimentGraph(g,panel,jx,seed_module,phase_only)
        truths = {(l,k):model.t.interface(layers==l,layers==k) for l in present for k in present}
        order = panel_rng.permutation(len(model.i))
        for rep in range(5):
            expr = generate(markers,.5,test_rng)
            prep = model.prepare(expr)
            base = {"Pearson":np.corrcoef(expr.T), "Lee_L":jx.lee_l_matrix(expr,g),
                    "Regional_J":jx.regional_juxtaposition_matrix(expr,g)}
            records = []
            for k,(i,j) in enumerate(zip(a,b)):
                local = pair_class[k]=="touch" and i%10==0 and j%10==0
                score, seeds, *edge_scores = model.score(prep,i,j,make_maps=local)
                meta_row = dict(section=sec,donor=data.DONOR[sec],replicate=rep,
                                pair_class=pair_class[k],marker_x=int(i),marker_y=int(j))
                records.append(dict(**meta_row,**score,**{key:value[i,j] for key,value in base.items()},
                                    **{f"n_{key}":value for key,value in seeds.items()}))
                if local:
                    for method,(ap,iou) in seed_module.localization(truths[(markers[i],markers[j])],edge_scores[0],order).items():
                        all_local.append(dict(**meta_row,method=method,ap=ap,iou=iou))
            frame = pd.DataFrame(records)
            test = auc_rows(frame,["section","donor","replicate"],phase_only=phase_only)
            r = original.query("section==@sec and beta==.5 and rep==@rep and method=='J1'").iloc[0]
            for label,col in [("touch_vs_same","auc_touch_vs_same"),("touch_vs_apart","auc_touch_vs_apart")]:
                np.testing.assert_allclose(test.loc[(test.task==label)&(test.method=="J1"),"auroc"],r[col],atol=1e-12,rtol=0)
            all_rows.append(test.assign(fixed_signed_seeds=model.counts["fixed_signed_J"],
                                        fixed_positive_seeds=model.counts["fixed_positive_J"]))
        print(f"B2 beta 0.5 {sec} {time.monotonic()-t0:.0f}s",flush=True)
    pd.concat(all_rows).to_csv(out/"b2_ranking.csv",index=False)
    pd.DataFrame(all_local).to_csv(out/"b2_edges.csv",index=False)


def observed(out, project, jx, seed_module, data, phase_only=False):
    bench = project/"revisions"/V5/"results"
    seq, seqgenes, seqmeta = data.load_seqfish()
    all_rows,all_local = [],[]
    for f in sorted(bench.glob("*_pairs_2026-11-22_104504_EST.csv.gz")):
        design, unit, *_ = f.name.split("_")
        if design not in ("B1","C1"):
            continue
        rng = np.random.default_rng([RNG_SEED,code_seed(unit)])
        direct = pd.read_csv(f,dtype={"unit":str})
        if design=="B1":
            counts,genes,meta = data.load_dlpfc(unit)
            expr = data.lognorm(counts)
            graph = jx.visium_graph(meta.array_row,meta.array_col)
            label = meta.layer.to_numpy()
            columns = ["layer_x","layer_y"]
        else:
            use = seqmeta.embryo.eq(unit).to_numpy()
            genes,expr,meta = seqgenes,seq[use],seqmeta.loc[use]
            graph = jx.spatial_graph(meta[["x_global_affine","y_global_affine"]].to_numpy(),neighbors=6)
            label = meta.celltype_mapped_refined.to_numpy()
            columns = ["type_x","type_y"]
        names = np.union1d(direct.gene_x,direct.gene_y)
        tested_cols = np.flatnonzero(np.isin(genes,names))
        tested = pd.Index(genes[tested_cols])
        assert len(tested)==len(names) and tested.is_unique
        excluded = np.isin(genes,names)
        mean = np.asarray(expr.mean(axis=0)).ravel()
        var = np.asarray(expr.power(2).mean(axis=0)).ravel()-mean*mean
        eligible = np.flatnonzero((~excluded)&(var>1e-12))
        highest = eligible[np.argsort(-var[eligible],kind="stable")[:min(128,len(eligible))]]
        panel_cols = rng.choice(highest,32,replace=False)
        assert not excluded[panel_cols].any()
        model = ExperimentGraph(graph,expr[:,panel_cols].toarray(),jx,seed_module,phase_only)
        pd.DataFrame({"design":design,"unit":unit,"gene":genes[panel_cols]}).to_csv(out/f"discovery_{design}_{unit}.csv",index=False)
        obs = model.prepare(expr[:,tested_cols].toarray())
        ga,gb = tested.get_indexer(direct.gene_x),tested.get_indexer(direct.gene_y)
        candidates = direct[direct.pair_class.eq("touch")].groupby(columns,group_keys=False)
        local_ids = set(candidates.apply(lambda g:g.sample(n=min(len(g),8),random_state=B2_SEED),
                                         include_groups=False).index)
        edge_order = rng.permutation(len(model.i))
        masks = {x:label==x for x in np.unique(label)}
        records = []
        for k,(a,b) in enumerate(zip(ga,gb)):
            row = direct.iloc[k]
            score,seeds,*maps = model.score(obs,a,b,make_maps=k in local_ids)
            context = dict(design=design,unit=unit,donor=row.tissue,gene_x=row.gene_x,gene_y=row.gene_y,pair_class=row.pair_class)
            records.append(dict(**context,**score,Pearson=row.pearson,Lee_L=row.lee_l,Regional_J=row.J_region,
                                Hotspot=row.hotspot_Z,Giotto=row.giotto_spatial_cor,
                                **{f"n_{key}":value for key,value in seeds.items()}))
            if k in local_ids:
                mask = model.t.interface(*(masks[row[c]] for c in columns))
                for method,(ap,iou) in seed_module.localization(mask,maps[0],edge_order).items():
                    all_local.append(dict(**context,method=method,ap=ap,iou=iou))
        frame = pd.DataFrame(records)
        np.testing.assert_allclose(frame["J1"],direct.J1,atol=1e-10,rtol=0)
        if not phase_only:
            np.testing.assert_allclose(frame["J4"],direct.J4,atol=1e-10,rtol=0)
        all_rows.append(frame)
        print(f"{design} {unit} {len(direct)} pairs; fixed seeds {model.counts}",flush=True)
    frame = pd.concat(all_rows)
    frame.to_csv(out/"observed_pairs.csv.gz",index=False)
    auc_rows(frame,["design","unit","donor"],("Hotspot","Giotto"),phase_only).to_csv(out/"observed_ranking.csv",index=False)
    pd.DataFrame(all_local).to_csv(out/"observed_edges.csv.gz",index=False)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("mode",choices=["simulation","simulation-weak-panel","b2","observed","null"])
    p.add_argument("--phase-only",action="store_true")
    args = p.parse_args()
    project = args.project.resolve()
    sys.path[:0] = [str(project/"revisions"/V5/"packages/python/src"),
                    str(project/"revisions"/OLD),str(project/"benchmarks/v3")]
    import juxtaposed as jx
    import seed_weighted_j as seed_module
    import datasets
    args.out.mkdir(exist_ok=True,parents=True)
    t0 = time.monotonic()
    if args.mode.startswith("simulation"):
        simulation(args.out,jx,seed_module,weak_panel=args.mode.endswith("weak-panel"),phase_only=args.phase_only)
    elif args.mode=="b2":
        b2(args.out,project,jx,seed_module,datasets,phase_only=args.phase_only)
    elif args.mode=="observed":
        observed(args.out,project,jx,seed_module,datasets,phase_only=args.phase_only)
    else:
        from calibration import calibration
        calibration(args.out,jx,seed_module)
    print(f"{args.mode}: {time.monotonic()-t0:.1f}s",flush=True)


if __name__=="__main__":
    main()
