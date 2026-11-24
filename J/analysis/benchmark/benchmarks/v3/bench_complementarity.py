"""Exploratory (post hoc, requested): does J add information to a co-localization statistic?

Touching-but-distinct domains should co-localize at the neighbourhood scale (high L) and change in
opposite directions across shared edges (high J). Two combinations are tested, both trained or fixed
without the test unit: (1) logistic regression on within-unit percentile ranks of L, J and their
product, fitted leave-one-donor-out (DLPFC) or leave-one-embryo-out (seqFISH); (2) the
parameter-free product of the two percentile ranks.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RESULTS
from datasets import DONOR


def evaluate(p, unit, group, pos, mask, co, dataset, task):
    rows = []
    X = lambda d: np.column_stack([d[co + "_r"], d["J1_r"], d[co + "_r"] * d["J1_r"]])  # noqa: E731
    for g_out in sorted(p[group].unique()):
        tr, te = p[(p[group] != g_out) & mask], p[(p[group] == g_out) & mask]
        m = LogisticRegression(C=1.0).fit(X(tr), tr[pos])
        for u, g in te.groupby(unit):
            if g[pos].nunique() < 2:
                continue
            rows.append(dict(dataset=dataset, task=task, colocalization=co, unit=u,
                             auc_combined_lr=roc_auc_score(g[pos], m.predict_proba(X(g))[:, 1]),
                             auc_rank_product=roc_auc_score(g[pos], g[co + "_r"] * g["J1_r"]),
                             auc_colocalization=roc_auc_score(g[pos], g[co]),
                             auc_J=roc_auc_score(g[pos], g["J1"])))
    return rows


def main():
    rows = []
    p = pd.read_csv(RESULTS / "gold_b1_pairs_K10.csv.gz", dtype={"section": str})
    p["donor"] = p.section.map(DONOR)
    for c in ("J1", "lee_l", "hotspot_Z", "spatialdm_R", "liana_morans_R", "pearson"):
        p[c + "_r"] = p.groupby("section")[c].rank(pct=True)
    p["touch"] = (p.layer_distance == 1).astype(int)
    for co in ("lee_l", "hotspot_Z", "spatialdm_R", "liana_morans_R", "pearson"):
        rows += evaluate(p, "section", "donor", "touch", p.layer_distance != 0, co, "DLPFC", "touch_vs_apart")
        rows += evaluate(p, "section", "donor", "touch", np.ones(len(p), bool), co, "DLPFC", "touch_vs_apart_or_same")
    s = pd.read_csv(RESULTS / "seqfish_c_pairs.csv.gz")
    s["lee_l"] = -s["neg_lee"]
    s["pearson"] = -s["neg_pearson"]
    for c in ("J1", "lee_l", "pearson"):
        s[c + "_r"] = s.groupby("embryo")[c].rank(pct=True)
    s = s[s.pair_class.isin(["contact", "apart", "same"])].copy()
    s["touch"] = (s.pair_class == "contact").astype(int)
    for co in ("lee_l", "pearson"):
        rows += evaluate(s, "embryo", "embryo", "touch", s.pair_class != "same", co, "seqFISH", "touch_vs_apart")
        rows += evaluate(s, "embryo", "embryo", "touch", np.ones(len(s), bool), co, "seqFISH", "touch_vs_apart_or_same")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "complementarity_auroc.csv", index=False)
    summ = res.groupby(["dataset", "task", "colocalization"]).agg(
        n_units=("unit", "size"), colocalization_alone=("auc_colocalization", "mean"), J=("auc_J", "mean"),
        combined_lr=("auc_combined_lr", "mean"), rank_product=("auc_rank_product", "mean"),
        lr_wins=("auc_combined_lr", lambda v: 0))
    wins = res.assign(w=res.auc_combined_lr > res.auc_colocalization,
                      w2=res.auc_rank_product > res.auc_colocalization).groupby(
        ["dataset", "task", "colocalization"])[["w", "w2"]].sum()
    summ["lr_wins"] = wins["w"]
    summ["product_wins"] = wins["w2"]
    summ.to_csv(RESULTS / "complementarity_summary.csv")  # index: dataset, task, colocalization
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
