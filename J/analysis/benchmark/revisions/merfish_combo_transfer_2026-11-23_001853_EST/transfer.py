"""Frozen DLPFC models and animal-held-out MERFISH transfer of J4 + regional J."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

V5 = "v5_2026-11-22_104348_EST"
CO = "Regional J"
J = "Boundary J4"
GROUP = "section"
NA = ("Distant", "Distant or same")


def features(frame, co=CO, boundary=J):
    r = frame.groupby(GROUP)[[co,boundary]].rank(pct=True)
    return np.column_stack([r[co],r[boundary],r[co]*r[boundary]])


def fit(frame, co=CO, boundary=J):
    return LogisticRegression(max_iter=1000).fit(features(frame,co,boundary),frame.contact)


def metrics(frame, method, score):
    y = frame.contact.to_numpy(dtype=bool)
    if not y.any() or y.all():
        raise ValueError("Both classes are required in every evaluated section")
    top = max(1,int(np.ceil(.2*len(y))))
    selected = np.lexsort((frame.gene_y,frame.gene_x,-score))[:top]
    shared = np.count_nonzero(y[selected])
    return dict(method=method, n_positive=int(y.sum()), n_negative=int((~y).sum()),
                auroc=roc_auc_score(y,score),ap=average_precision_score(y,score),
                jaccard_20=shared/(top+y.sum()-shared))


def select(frame, task):
    return frame[frame.category.isin(["Touch","Distant"]+(["Same"] if task=="Distant or same" else []))].copy()


def source(project):
    src = project/"revisions"/V5/"results"
    files = sorted(src.glob("B1_*_pairs_*.csv.gz"))
    train = pd.concat([pd.read_csv(f,dtype={"unit":str}) for f in files],ignore_index=True)
    assert train.unit.nunique()==12 and train.tissue.nunique()==3
    train = train.rename(columns={"unit":"section","tissue":"animal","J_region":CO,"J4":J,
                                  "J1":"Boundary J1", "hotspot_Z":"Hotspot Z"})
    train["category"] = train.pair_class.map(dict(touch="Touch",same="Same",apart="Distant"))
    train["contact"] = train.category.eq("Touch")
    return train


class Section:
    def __init__(self, section, animal, path, jx):
        self.section,self.animal,self.jx = section,animal,jx
        with np.load(path) as z:
            self.x,self.xy,self.labels,self.genes = (z[k] for k in ["x","xy","region","genes"])
        self.graph = jx.spatial_graph(self.xy,neighbors=6)


    def pairs(self, panel, hotspot_weights, row_normalize, domain_scores):
        # Only panel membership comes from training expression. Test labels are used for truth.
        regs,n = np.unique(self.labels,return_counts=True)
        regs = [r for r,k in zip(regs,n) if k>=100 and r in panel and len(panel[r])]
        idx = np.concatenate([panel[r] for r in regs])
        structure = np.repeat(regs,[len(panel[r]) for r in regs])
        names = self.genes[idx]
        assert len(set(names))==len(names)
        x = self.x[:,idx]
        adj = sparse.triu(self.graph,k=1,format="coo")
        region = pd.Index(regs).get_indexer(self.labels)
        valid = (region[adj.row]>=0)&(region[adj.col]>=0)
        ct = np.zeros((len(regs),len(regs)),int)
        np.add.at(ct,(region[adj.row[valid]],region[adj.col[valid]]),1)
        ct = ct + ct.T
        i,j = np.triu_indices(len(idx),1)
        s = pd.Index(regs).get_indexer(structure)
        crossing = ct[s[i],s[j]]
        same = s[i]==s[j]
        keep = same | (crossing==0) | (crossing>=10)
        i,j,crossing,same = i[keep],j[keep],crossing[keep],same[keep]
        category = np.where(same,"Same",np.where(crossing>=10,"Touch","Distant"))
        pair = pd.DataFrame(dict(section=self.section,animal=self.animal,gene_x=names[i],gene_y=names[j],
                                 structure_x=structure[i],structure_y=structure[j],cross_edges=crossing,
                                 category=category,contact=category=="Touch"))
        mat = domain_scores(x,self.graph)
        mat[J] = self.jx.juxtaposition_matrix(x,self.graph,scale=4)
        mat["Boundary J1"] = self.jx.juxtaposition_matrix(x,self.graph,scale=1)
        mat["Pearson"] = self.jx.pearson_matrix(x)
        mat["Lee L"] = self.jx.lee_l_matrix(x,self.graph)
        mat["Giotto network"] = np.corrcoef(row_normalize(self.graph) @ x,rowvar=False)
        z = self.jx.standardize(x)
        hz = hotspot_weights(self.xy)@z
        norms = np.sqrt(np.sum(hz*hz,axis=0))
        mat["Hotspot Z"] = (z.T @ hz)/np.maximum.outer(norms,norms)
        for k,v in mat.items():
            pair[k] = v[i,j]
            assert np.isfinite(pair[k]).all()
        return pair


def main(project,cache,output):
    v5 = project/"revisions"/V5
    sys.path[:0] = [str(v5/"packages/python/src"),str(v5/"code"),str(project/"benchmarks/v3")]
    import juxtaposed as jx
    from allen_transfer import markers
    from domain_matrix import domain_scores
    from gold import hotspot_weights,row_normalize

    out = output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    original = pd.read_csv(v5/"results/allen_transfer_pairs.csv.gz")
    design = original[["section","animal"]].drop_duplicates()
    assert len(design)==4 and design.animal.nunique()==2
    sections = {r.section:Section(r.section,r.animal,cache/(r.section+".npz"),jx)
                for r in design.itertuples()}
    source_frames = source(project)
    models, src_metrics, coefficient_rows = {},[],[]
    for task in NA:
        cortical = select(source_frames,task)
        models[task] = {}
        for label,co,boundary in [("Frozen DLPFC J4 + regional J",CO,J),
                                  ("Frozen DLPFC J1 + Hotspot","Hotspot Z","Boundary J1")]:
            model = fit(cortical,co,boundary)
            models[task][label] = (model,co,boundary)
            coefficient_rows.append(dict(task=task,method=label,training="DLPFC",intercept=model.intercept_.item(),
                                         coef_co=model.coef_[0,0],coef_boundary=model.coef_[0,1],coef_product=model.coef_[0,2]))
            for animal in sorted(cortical.animal.unique()):
                tr, te = cortical[cortical.animal.ne(animal)],cortical[cortical.animal.eq(animal)]
                clf = fit(tr,co,boundary)
                for name,g in te.groupby(GROUP):
                    src_metrics.append(dict(section=name,animal=animal,task=task,
                                            **metrics(g,label,clf.decision_function(features(g,co,boundary))),
                                            co_auroc=roc_auc_score(g.contact,g[co])))
    pd.DataFrame(src_metrics).to_csv(out/"source_cross_validation.csv",index=False)

    rs, predictions = [], []
    for test in sorted(design.animal.unique()):
        training = [s for s in sections.values() if s.animal!=test]
        testing = [s for s in sections.values() if s.animal==test]
        assert all(np.array_equal(s.genes,training[0].genes) for s in [*training,*testing])
        panel = markers(np.concatenate([s.x for s in training]),np.concatenate([s.labels for s in training]))
        calc = lambda s:s.pairs(panel,hotspot_weights,row_normalize,domain_scores)
        training_pairs = pd.concat([calc(s) for s in training],ignore_index=True)
        tests = {s.section:calc(s) for s in testing}
        for name,frame in tests.items():
            archive = original[original.section.eq(name)].reset_index(drop=True)
            core = select(frame,"Distant").reset_index(drop=True)
            for col in ["gene_x","gene_y","structure_x","structure_y","cross_edges"]:
                np.testing.assert_array_equal(core[col],archive[col])
            for col,old in [(CO,CO),("Boundary J1","Opposition J1"),("Matched Pearson","Matched Pearson")]:
                np.testing.assert_allclose(core[col],archive[old],rtol=0,atol=1e-10)
            print(name,frame.category.value_counts().to_dict(),flush=True)
        for task in NA:
            tp = select(training_pairs,task)
            mouse_model = fit(tp)
            coef=mouse_model.coef_[0]
            coefficient_rows.append(dict(task=task, method="Other mouse J4 + regional J",training=f"mouse {3-test}",
                                         intercept=mouse_model.intercept_.item(),coef_co=coef[0],
                                         coef_boundary=coef[1],coef_product=coef[2]))
            for name, frame in tests.items():
                g = select(frame,task)
                f = features(g)
                outputs = {col:g[col].to_numpy() for col in [
                    CO,J,"Matched Pearson","Boundary J1","Lee L","Hotspot Z","Pearson","Giotto network"
                ]}
                for score,(model,co,boundary) in models[task].items():
                    outputs[score] = model.decision_function(features(g,co,boundary))
                outputs["Other mouse J4 + regional J"] = mouse_model.decision_function(f)
                outputs["J4 × regional J rank product"] = f[:,2]
                for method,value in outputs.items():
                    rs.append(dict(section=name,animal=test,task=task,**metrics(g,method,value)))
                score_frame = g[["section","animal","gene_x","gene_y","structure_x","structure_y",
                                 "cross_edges","category","contact"]].assign(task=task)
                predictions.append(score_frame.assign(**outputs))
    pd.DataFrame(rs).to_csv(out/"merfish_metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"merfish_predictions.csv.gz",index=False)
    pd.DataFrame(coefficient_rows).to_csv(out/"model_coefficients.csv",index=False)


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("cache",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    main(a.project.resolve(),a.cache,a.out)
