"""One-step J combined with each co-localization score on the four MERFISH transfer sections (v5 allen_transfer
pairs; touching = at least 10 connecting edges, apart = none). Logistic model on within-section percentile ranks,
trained on the other mouse.     python merfish_combine.py <analysis root>   -> merfish_combine_J1.csv"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

root = Path(sys.argv[1])
d = pd.read_csv(root / "revisions/v5_2026-11-22_104348_EST/results/allen_transfer_pairs.csv.gz")
d = d[(d.cross_edges == 0) | (d.cross_edges >= 10)].copy()
d["y"] = (d.cross_edges >= 10).astype(int)
pr = lambda s: s.rank(pct=True)  # noqa: E731
rows = []
for co in ["Regional J", "Hotspot Z", "Lee L", "Matched Pearson", "Giotto network"]:
    F = pd.concat([pd.DataFrame({"c": pr(g[co]), "j": pr(g["Opposition J1"])}, index=g.index).assign(cj=lambda f: f.c * f.j)
                   for _, g in d.groupby("section")]).loc[d.index]
    for test in sorted(d.animal.unique()):
        tr, te = (d.animal != test).to_numpy(), (d.animal == test).to_numpy()
        p = LogisticRegression(max_iter=1000).fit(F[tr], d.y[tr]).predict_proba(F[te])[:, 1]
        for sec in d[te].section.unique():
            k = (d[te].section == sec).to_numpy()
            yy = d[te].y.to_numpy()[k]
            rows.append(dict(colocalization=co, section=sec, combined=roc_auc_score(yy, p[k]),
                             alone=roc_auc_score(yy, d[te][co].to_numpy()[k])))
pd.DataFrame(rows).to_csv("merfish_combine_J1.csv", index=False)
