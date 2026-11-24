"""Summarize paired analyses, keep biological units visible, and draw a reading guide."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

COLORS = {"J1":"#627888","cluster_J":"#D08842","fixed_signed_J":"#007975",
          "fixed_positive_J":"#255A73","anisotropic_J":"#AD5379","resolvent_J":"#6E57A0",
          "wavelet_J":"#587D51","J4":"#385C7D","Hotspot archive":"#74834E",
          "phase_J":"#77409A","Regional_J":"#26363B","Lee_L":"#AC6A55",
          "SpatialDM local":"#8E8F52","LIANA+ local cosine":"#A9695B"}
LABELS = {"J1":"J₁","cluster_J":"Clustered elbow","fixed_signed_J":"Fixed signed panel",
          "fixed_positive_J":"Fixed positive panel","anisotropic_J":"Anisotropic J",
          "resolvent_J":"Resolvent J","wavelet_J":"Wavelet contrast","J4":"J₄","Hotspot archive":"Hotspot (archive)",
          "phase_J":"Unitary phase","Regional_J":"Regional J","Lee_L":"Lee’s L",
          "SpatialDM local":"SpatialDM local","LIANA+ local cosine":"LIANA+ cosine"}


def load(root,archive):
    d = root/"results"

    def include_phase(filename):
        main = pd.read_csv(d/filename)
        phase = pd.read_csv(root/"phase"/filename)
        keys = sorted(k for k in set(main.columns)&set(phase.columns) if not k.startswith("fixed_"))
        left = main.loc[main.method=="J1",keys].reset_index(drop=True)
        right = phase.loc[phase.method=="J1",keys].reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right)
        return pd.concat([main,phase[phase.method=="phase_J"]],ignore_index=True)

    clean = include_phase("simulation_edges.csv.gz")
    weak_main = pd.read_csv(root/"weak_panel/simulation_edges.csv.gz")
    # The wave filter uses only the fixed geometry and evaluated map.
    weak = pd.concat([weak_main,clean[clean.method=="phase_J"]],ignore_index=True)
    np.testing.assert_array_equal(clean.loc[clean.method=="J1","ap"],weak.loc[weak.method=="J1","ap"])
    np.testing.assert_array_equal(clean.loc[clean.method=="J1","iou"],weak.loc[weak.method=="J1","iou"])
    ranked1 = pd.read_csv(d/"simulation_ranking.csv")
    ranked2 = pd.read_csv(root/"weak_panel/simulation_ranking.csv")
    np.testing.assert_array_equal(ranked1.loc[ranked1.method=="J1","auroc"],ranked2.loc[ranked2.method=="J1","auroc"])
    sim = pd.concat([clean.assign(panel="amplitude 1"),weak.assign(panel="amplitude matched")])
    b = include_phase("b2_ranking.csv")
    hot = pd.read_csv(archive,dtype={"section":str}).query("beta==.5 and method=='hotspot_Z'")
    assert b.section.nunique()==12 and len(hot)==60
    ref = pd.concat([hot.assign(task="touch_vs_same",auroc=1-hot.auc_touch_vs_same),
                     hot.assign(task="touch_vs_apart",auroc=hot.auc_touch_vs_apart)])
    baselines = b.groupby(["task","method"],as_index=False).auroc.mean()
    baselines = pd.concat([baselines,ref.groupby("task",as_index=False).auroc.mean().assign(method="Hotspot archive")])
    o = include_phase("observed_edges.csv.gz")
    refs = pd.read_csv(d/"matched_local_references.csv.gz")
    assert refs.loc[refs.method=="J1 reference check"].shape[0] == 916
    o = pd.concat([o,refs],ignore_index=True)
    obs = o.groupby(["design","unit","donor","method"],as_index=False)[["ap","iou"]].mean()
    null = pd.read_csv(d/"known_null.csv.gz")
    null = null.assign(rejected=null.p.le(.05))
    assert len(null.family.unique())==3 and null.replicate.nunique()==240
    return sim, baselines, obs, null, include_phase("observed_ranking.csv"), include_phase("b2_edges.csv")


def intervals(null):
    n = null.groupby(["method","inference"],as_index=False).agg(
        n=("p","size"), mean_score=("score","mean"), n_rejected=("rejected","sum"))
    n["rejection_rate"] = r = n.n_rejected/n.n
    z = norm.ppf(.975)
    midpoint = (r+z*z/(2*n.n))/(1+z*z/n.n)
    radius = z*np.sqrt((r*(1-r)+z*z/(4*n.n))/n.n)/(1+z*z/n.n)
    return n.assign(low=midpoint-radius,high=midpoint+radius)


def plot(root, sim, b2, obs, null):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,ax = plt.subplots(2,2,figsize=(11.5,7.5),layout="constrained")
    a = ax[0,0]
    focus = sim.query("geometry=='focal' and amp<=.25").groupby(["panel","amp","method"]).ap.mean()
    methods = ["J1","cluster_J","fixed_signed_J","fixed_positive_J","anisotropic_J"]
    for k,m in enumerate(methods):
        cells = [focus.loc[p,x,m] for p,x in [
            ("amplitude 1",.1),("amplitude 1",.25),("amplitude matched",.1),("amplitude matched",.25)]]
        a.scatter(np.arange(4)+(k-2)*.09, cells,s=32,color=COLORS[m],label=LABELS[m],zorder=4)
    a.set(xticks=range(4),xticklabels=["amp 0.10\nstrong panel","amp 0.25\nstrong panel",
                                      "amp 0.10\nweak panel","amp 0.25\nweak panel"],ylabel="Edge average precision",ylim=(0,.033))
    a.set_title("A   Focal interfaces in simulated maps",loc="left",weight="bold")
    a.legend(ncol=2,loc="upper left",fontsize=7,frameon=False)

    a = ax[0,1]
    plot_methods = ["J1","wavelet_J","phase_J","anisotropic_J","J4","Lee_L","Regional_J","Hotspot archive"]
    for k,m in enumerate(plot_methods):
        cells = [b2.query("task==@t and method==@m").auroc.item() for t in ("touch_vs_apart","touch_vs_same")]
        a.scatter(np.arange(2)+(k-3.5)*.064,cells,s=34,color=COLORS[m],label=LABELS[m],zorder=4)
    a.set(xticks=[0,1],xticklabels=["Touch vs distant","Touch vs same"],ylabel="Mean AUROC",ylim=(.45,.96),xlim=(-.4,1.4))
    a.set_title("B   Weak markers on cortex layouts (β = 0.5)",loc="left",weight="bold")
    a.legend(ncol=2,loc="upper left",fontsize=7,frameon=False)

    a = ax[1,0]
    scores = obs.groupby(["design","method"]).ap.mean()
    for k,m in enumerate(["J1","cluster_J","fixed_signed_J","anisotropic_J","phase_J","SpatialDM local","LIANA+ local cosine"]):
        a.bar(np.arange(2)+(k-3)*.115,[scores.loc[d,m] for d in ("B1","C1")],
              width=.108,color=COLORS[m],label=LABELS[m])
    a.set(xticks=[0,1],xticklabels=["DLPFC","seqFISH"],ylabel="Mean edge average precision",ylim=(0,.073))
    a.set_title("C   Localization in annotated tissues",loc="left",weight="bold")
    a.legend(ncol=2,loc="upper left",fontsize=7,frameon=False)

    a = ax[1,1]
    controls = [("J1","normal_numerator","J₁, normal"),
                ("fixed_signed_J","conditional_MC","Fixed panel, Monte Carlo"),
                ("cluster_J","freeze_same_pair_MC","Clustered, reuse chosen seeds"),
                ("cluster_J","reselected_MC","Clustered, reselect on each draw")]
    for k,(m,inf,label) in enumerate(controls):
        r = null.query("method==@m and inference==@inf").iloc[0]
        c=COLORS[m]
        a.errorbar(100*r.rejection_rate,k,xerr=[[100*(r.rejection_rate-r.low)],[100*(r.high-r.rejection_rate)]],
                   fmt="o",ms=5,color=c,lw=1.1,capsize=2)
    a.axvline(5,color="#8A9AA5",ls="--")
    a.set(yticks=range(len(controls)),yticklabels=[v[2] for v in controls],xlim=(0,17),ylim=(3.6,-.6),
          xlabel="Independent pairs called significant (%)")
    a.set_title("D   Known-null rejection, nominal 5%",loc="left",weight="bold")
    for a in ax.flat:
        a.set_axisbelow(True)
        a.grid(axis="y",color="#E4EAED")
    fig.suptitle("Spatial seed filtering and alternative boundary scores",fontsize=15,weight="bold")
    for ext in ("png","pdf"):
        fig.savefig(root/f"benchmark.{ext}",dpi=215,facecolor="white")
    plt.close(fig)


def report(root,archive):
    sim,b2,obs,null,observed,b2_edges = load(root,archive)
    cal = intervals(null)
    cal.to_csv(root/"null_summary.csv",index=False)
    b2.to_csv(root/"b2_summary.csv",index=False)
    obs.groupby(["design","method"],as_index=False)[["ap","iou"]].mean().to_csv(root/"observed_edges_summary.csv",index=False)
    b2_edges.groupby(["section","method"])[["ap","iou"]].mean().groupby("method").mean().to_csv(root/"b2_edges_summary.csv")
    obs.groupby(["design","donor","method"],as_index=False)[["ap","iou"]].mean().to_csv(root/"observed_edges_by_donor.csv",index=False)
    observed.groupby(["design","task","method"],as_index=False)[["auroc","ap"]].mean().to_csv(root/"observed_pair_summary.csv",index=False)
    sim.groupby(["panel","geometry","amp","method"],as_index=False)[["ap","iou"]].mean().to_csv(root/"simulation_edge_summary.csv",index=False)
    d = sim.groupby(["panel","geometry","amp","geometry_id","method"])[["ap","iou"]].mean()
    sample = np.random.default_rng(20261123)
    rows = []
    for (panel,geometry,amp), ds in d.reset_index().groupby(["panel","geometry","amp"]):
        for metric in ["ap","iou"]:
            vs = ds.pivot(index="geometry_id",columns="method",values=metric)
            assert len(vs)==12
            idx = sample.integers(0,len(vs),size=(12000,len(vs)))
            for m in vs.columns:
                if m=="J1":
                    continue
                delta = (vs[m]-vs.J1).to_numpy()
                low,high = np.quantile(delta[idx].mean(axis=1),[.025,.975])
                rows.append(dict(panel=panel,geometry=geometry,amp=amp,metric=metric,method=m,delta=delta.mean(),low=low,high=high))
    pd.DataFrame(rows).to_csv(root/"paired_simulation_intervals.csv",index=False)
    plot(root,sim,b2,obs,cal)


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("archive",type=Path)
    args = p.parse_args()
    report(args.root,args.archive)
