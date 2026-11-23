"""J combined with each co-localization score: logistic model on within-section percentile ranks (score, J, product),
trained on the other donors (cortex) or embryos (seqFISH), evaluated per held-out section. Same design as
benchmarks/v3/bench_complementarity.py; J1 + Hotspot on cortex reproduces its 0.727.

    python combine.py   -> combine.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("JBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "benchmarks" / "v3"))
from datasets import DONOR  # noqa: E402

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "pairs_physics.csv.gz", dtype={"unit": str}, low_memory=False)
df["group"] = [DONOR.get(u, u) for u in df.unit]
pr = lambda s: s.rank(pct=True)  # noqa: E731
rows = []
for design in ("B1", "C1"):
    base = df[df.design == design]
    for task, negs in (("touch_apart_or_same", ["apart", "same"]), ("touch_apart", ["apart"])):
        d = base[base.pair_class.isin(["touch", *negs])].copy()
        d["y"] = (d.pair_class == "touch").astype(int)
        for co in ["hotspot_Z", "lee_l", "J_region", "flory", "sliced_W"]:
            for jj in ["J1", "J4"]:
                F = pd.concat([pd.DataFrame({"c": pr(g[co]), "j": pr(g[jj])}, index=g.index).assign(cj=lambda f: f.c * f.j)
                               for _, g in d.groupby("unit")]).loc[d.index]
                for test in sorted(d.group.unique()):
                    tr, te = (d.group != test).to_numpy(), (d.group == test).to_numpy()
                    p = LogisticRegression(max_iter=1000).fit(F[tr], d.y[tr]).predict_proba(F[te])[:, 1]
                    for unit in d[te].unit.unique():
                        k = (d[te].unit == unit).to_numpy()
                        yy = d[te].y.to_numpy()[k]
                        rows.append(dict(design=design, task=task, colocalization=co, J=jj, unit=unit,
                                         combined=roc_auc_score(yy, p[k]), alone=roc_auc_score(yy, d[te][co].to_numpy()[k])))
res = pd.DataFrame(rows)
res.to_csv(OUT / "combine.csv", index=False)
s = res.groupby(["design", "task", "colocalization", "J"]).apply(
    lambda g: pd.Series({"combined": g.combined.mean(), "alone": g.alone.mean(),
                         "improved": f"{(g.combined > g.alone).sum()}/{len(g)}"}))
print(s.round(3).to_string())
