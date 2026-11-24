"""Plot held-out per-section changes in AUROC and export the companion summaries."""
from argparse import ArgumentParser
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

METHODS = ("Regional J","Frozen DLPFC J4 + regional J","Other mouse J4 + regional J")
COLORS = ("#677B88","#BC693B","#08786F")
METRICS = ("auroc","ap","jaccard_20")


def main(root):
    r = root/"results"
    d = pd.read_csv(r/"merfish_metrics.csv")
    means = d.groupby(["task","method"],as_index=False)[list(METRICS)].mean()
    means.to_csv(r/"merfish_mean_metrics.csv",index=False)
    baseline = d[d.method.eq(METHODS[0])].set_index(["task","section"])
    comparison = d[d.method.isin(METHODS[1:])].copy()
    for m in METRICS:
        comparison["delta_"+m] = comparison[m]-baseline.loc[
            pd.MultiIndex.from_frame(comparison[["task","section"]]),m].to_numpy()
    comparison.to_csv(r/"merfish_paired_deltas.csv",index=False)
    comparison.groupby(["task","animal","method"],as_index=False)[["delta_"+m for m in METRICS]].mean().to_csv(
        r/"merfish_animal_deltas.csv",index=False)

    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    f, ax = plt.subplots(1,2,figsize=(9.1,4.05), layout="constrained",sharey=True)
    short_titles = ["A   Contacting vs separated","B   Contacting vs separated or same"]
    for a, title, task in zip(ax,short_titles,["Distant","Distant or same"]):
        x = d[d.task.eq(task)].pivot(index=["animal","section"],columns="method",values="auroc")
        for j,((animal,section),row) in enumerate(x.iterrows()):
            vec = row[list(METHODS)].to_numpy()
            offset=(j-1.5)*.038
            a.plot(np.arange(3)+offset,vec,lw=.75,c="#96A3AB",alpha=.66)
            for k,color in enumerate(COLORS):
                a.scatter(k+offset,vec[k],s=24,facecolor="white" if animal==1 else color,
                          edgecolor=color,linewidth=.9,zorder=3)
        for k,(method,color) in enumerate(zip(METHODS,COLORS)):
            mean = x[method].mean()
            a.scatter(k+.14,mean,marker="D",s=44,color=color,zorder=4,edgecolor="white",linewidth=.5)
            a.text(k+.23,mean,f"{mean:.3f}",va="center",color=color,fontsize=8)
        a.axhline(.5,ls="--",lw=.8,color="#8898A2")
        a.set_axisbelow(True)
        a.grid(axis="y",color="#EBEFF2")
        a.set(xticks=np.arange(3),xticklabels=["Regional J","J₄ + regional J\nDLPFC coefficients",
                                              "J₄ + regional J\nOther mouse coefficients"],
              xlim=(-.25,2.60),ylim=(.46,.63))
        a.set_title(title,loc="left",weight="bold")
    ax[0].set_ylabel("AUROC")
    f.legend(handles=[Line2D([],[],marker="o",ls="",mfc="white",mec=COLORS[0],label="Mouse 1"),
                      Line2D([],[],marker="o",ls="",color=COLORS[0],label="Mouse 2"),
                      Line2D([],[],marker="D",ls="",color=COLORS[0],label="Section mean")],
             loc="outside lower center",ncol=3,frameon=False)
    f.suptitle("Transfer to four Allen MERFISH sections",fontsize=13,weight="bold")
    for ext in ["png","pdf"]:
        f.savefig(root/f"merfish_transfer.{ext}",dpi=240,facecolor="white")
    plt.close(f)


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    main(p.parse_args().root)
