"""Reproduce observed gene maps and additive edge contributions on the benchmark graphs.

Run: python tissue_plates.py --repo /path/to/J-analysis --out output --stamp YYYY-MM-DD_HHMMSS_EST
Independent selection: first specimen ID; most cross-label edges among positive class pairs
with training markers; rank-1 external marker in each class. Per specimen, choose the
square (width one quarter of the smaller tissue span) containing the most target-crossing
edge midpoints among candidate centers at those midpoints. Ties retain stable ordering.
No expression or association score is used to select the field of view.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy.spatial import cKDTree

PALETTE = ("#00BA38", "#619CFF")
NEG, POS = "#619CFF", "#F8766D"
NEUTRAL, INK = "#d4d8dc", "#18242b"
MAP = LinearSegmentedColormap.from_list("signed", ["#3A3A98", NEG, "#F7F7F7", POS, "#832424"])
SCALES = (1, 2, 4, 8, 16)


def sample_maps(repo):
    sys.path.insert(0, str(repo / "benchmarks/v3"))
    from bench_seqfish import markers_loeo
    from common import jx, hex_xy
    from datasets import load_dlpfc, load_seqfish, LAYER_ORDER, SECTIONS, lognorm, DONOR

    sample = sorted(SECTIONS)[0]
    counts, genes, meta = load_dlpfc(sample)
    xy = hex_xy(meta.array_row, meta.array_col)
    graph = jx.visium_graph(meta.array_row.values, meta.array_col.values)
    labels = meta.layer.to_numpy()
    coo = sparse.triu(graph, k=1).tocoo()
    marker = pd.read_csv(repo / "results/v2/dlpfc_b1_markers.csv")
    marker = marker[(marker.K == 10) & (marker["rank"] == 1) & (marker.test_donor == DONOR[sample])]
    available = dict(zip(marker.layer, marker.gene))
    eligible = {}
    for a, b in zip(LAYER_ORDER[:-1], LAYER_ORDER[1:]):
        if a in available and b in available:
            eligible[a, b] = cross(labels, a, b, coo.row, coo.col).sum()
    types = sorted(eligible, key=lambda k: (-eligible[k], k))[0]
    name = [available[t] for t in types]
    cols = [np.flatnonzero(genes == g).item() for g in name]
    out = [dict(dataset="dlpfc", sample=sample, groups=types, genes=name,
                x=lognorm(counts)[:, cols].toarray(), xy=xy, graph=graph,
                ids=meta.index.to_numpy(), labels=labels, contact_z=np.nan,
                title="Human prefrontal cortex", source=f"Visium section {sample}  ·  donor {DONOR[sample]}",
                expr_label="log-normalized counts", cell="spots", scale_label="1 mm", scale=10.)]

    expr, genes, meta = load_seqfish()
    sample = sorted(meta.embryo.unique())[0]
    m = (meta.embryo == sample).to_numpy()
    sub = meta.loc[m]
    xy = sub[["x_global_affine", "y_global_affine"]].to_numpy()
    graph = jx.spatial_graph(xy, neighbors=6)
    labels = sub.celltype_mapped_refined.to_numpy()
    markers = markers_loeo(expr, genes, meta, sample, sorted(meta.celltype_mapped_refined.unique()))
    contact = pd.read_csv(repo / "results/v2/seqfish_contact_z.csv")
    contact = contact[(contact.embryo == sample) & (contact.z >= 3)]
    contact = contact[contact.type_a.map(lambda t: len(markers[t]) > 0) &
                      contact.type_b.map(lambda t: len(markers[t]) > 0)]
    picked = contact.sort_values(["edges", "type_a", "type_b"], ascending=[False, True, True]).iloc[0]
    types = (picked.type_a, picked.type_b)
    cols = [int(markers[t][0]) for t in types]
    out.append(dict(dataset="seqfish", sample=sample, groups=types, genes=genes[cols],
                    x=expr[m][:, cols].toarray(), xy=xy, graph=graph, ids=sub.uniqueID.to_numpy(),
                    labels=labels, contact_z=picked.z, title="Mouse embryo",
                    source="seqFISH embryo 1  ·  optical plane z = 2", cell="cells",
                    expr_label="normalized expression", scale_label="1 affine unit", scale=1.))
    return out


def cross_donor_example(repo, data, out, stamp):
    """Select an illustrative pair for the held-out first section using only other donors."""
    from common import jx
    from datasets import load_dlpfc, lognorm, DONOR, SECTIONS, LAYER_ORDER
    section = data["sample"]
    marker = pd.read_csv(repo / "results/v2/dlpfc_b1_markers.csv")
    marker = marker[(marker.K == 10) & (marker.test_donor == DONOR[section])]
    candidate = marker.gene.to_numpy()
    other = [s for s in SECTIONS if DONOR[s] != DONOR[section]]
    scores = {}
    boundary_scores = {}
    for s in other:
        xx, gg, mm = load_dlpfc(s)
        select = [np.flatnonzero(gg==g)[0] for g in candidate]
        a = jx.visium_graph(mm.array_row, mm.array_col)
        z = jx.standardize(lognorm(xx)[:,select].toarray())
        sm = jx.smoother(a)
        sp = sparse.diags(np.ravel(a.sum(axis=1))) + a
        arr = []
        for step in range(1, max(SCALES)+1):
            z = sm @ z
            if step == 1:
                lap = sparse.diags(np.ravel(a.sum(axis=1))) - a
                gram = z.T @ (lap @ z)
                den = gram.diagonal()[:,None] + gram.diagonal()[None,:]
                boundary_scores[s] = np.divide(-2*gram, den, out=np.full_like(den,np.nan),where=den>0)
            if step in SCALES:
                gram = z.T @ (sp @ z)
                den = gram.diagonal()[:,None] + gram.diagonal()[None,:]
                arr.append(np.divide(2*gram, den, out=np.full_like(den,np.nan), where=den>0))
        scores[s] = np.mean(arr,axis=0)
    # Equal donor weighting; all scores must be defined in every training section.
    means = np.mean([np.mean([scores[s] for s in other if DONOR[s]==d],axis=0)
                     for d in sorted(set(DONOR[s] for s in other))],axis=0)
    boundary_mean = np.mean([np.mean([boundary_scores[s] for s in other if DONOR[s]==d],axis=0)
                     for d in sorted(set(DONOR[s] for s in other))],axis=0)
    ranks = np.array([LAYER_ORDER.index(layer) for layer in marker.layer])
    i,j = np.where((ranks[:,None]+1)==ranks[None,:])
    ranks_df = pd.DataFrame(dict(gene_x=candidate[i],gene_y=candidate[j],
                                 label_x=marker.layer.to_numpy()[i],label_y=marker.layer.to_numpy()[j],
                                 external_donor_regional_mean=means[i,j],
                                 external_donor_boundary_mean=boundary_mean[i,j]))
    ranks_df.to_csv(out/("dlpfc_training_ranked_pairs_"+stamp+".csv"),index=False)
    xx, gg, _ = load_dlpfc(section)
    examples=[]
    for channel in ("boundary","regional"):
        score_col=f"external_donor_{channel}_mean"
        best=ranks_df.sort_values([score_col,"gene_x","gene_y"],ascending=[False,True,True]).iloc[0]
        ii = [np.flatnonzero(gg==g)[0] for g in [best.gene_x,best.gene_y]]
        examples.append({**data, "dataset":"dlpfc_training_"+channel, "x":lognorm(xx)[:,ii].toarray(),
                "genes":[best.gene_x,best.gene_y], "groups":[best.label_x,best.label_y],
                "title":"Human prefrontal cortex", "source":f"Visium section {section}  ·  pair nominated by {channel} score in other donors",
                "training_mean":best[score_col], "candidate_count":len(ranks_df)})
    return examples


def cross(labels, a, b, u, v):
    return ((labels[u] == a) & (labels[v] == b)) | ((labels[u] == b) & (labels[v] == a))


def channel_scores(x, graph):
    from common import jx
    coo = sparse.triu(graph, k=1).tocoo()
    sm = jx.smoother(graph)
    u, v, w = coo.row, coo.col, coo.data
    z = jx.standardize(x)
    codomain = np.zeros(len(u))
    for step in range(1, max(SCALES) + 1):
        z = sm @ z
        if step == 1:
            diff = z[u] - z[v]
            edge_energy = w * (diff**2).sum(axis=1)
            energy_fraction = edge_energy / edge_energy.sum()
            change = -2 * w * diff[:, 0] * diff[:, 1] / edge_energy.sum()
        if step in SCALES:
            plus = z[u] + z[v]
            codomain += 2 * w * plus[:, 0] * plus[:, 1] / (w @ (plus**2).sum(axis=1))
    codomain /= len(SCALES)
    ref = jx.juxtaposition_matrix(x, graph)[0, 1]
    np.testing.assert_allclose(change.sum(), ref, rtol=1e-9, atol=1e-12)
    return u, v, change, codomain, energy_fraction


def get_window(xy, u, v, selected):
    mids = (xy[u[selected]] + xy[v[selected]]) / 2
    half_width = np.ptp(xy, axis=0).min() / 8
    n = cKDTree(mids).query_ball_point(mids, half_width, p=np.inf, return_length=True)
    # Resolve ties by x and then y for a graph ordering independent of sparse storage.
    tied = mids[n == n.max()]
    center = tied[np.lexsort((tied[:, 1], tied[:, 0]))[0]]
    return center - half_width, center + half_width


def short_label(t):
    return {"Haematoendothelial progenitors":"Haematoendothelial progenitors",
            "L1":"Layer 1", "L2":"Layer 2"}.get(t, t)


def draw(d, out, stamp):
    x, xy, labels = d["x"], d["xy"], d["labels"]
    u, v, change, codomain, energy_fraction = channel_scores(x, d["graph"])
    a, b = d["groups"]
    crossing = cross(labels, a, b, u, v)
    cross_energy = energy_fraction[crossing].sum()
    cross_j = change[crossing].sum() / cross_energy if cross_energy else np.nan
    lo, hi = get_window(xy, u, v, crossing)
    mid = .5 * (xy[u] + xy[v])
    inside = np.all((mid >= lo) & (mid <= hi), axis=1)
    full = (xy.min(axis=0), xy.max(axis=0))
    plot_scale = []
    edge_scale = []
    fig = plt.figure(figsize=(11.5, 8.25), facecolor="white")
    panel_x = (.044, .373, .701)
    pwidth = .255
    # Wider-than-tall tissue samples and upright embryos occupy the same coordinate frame in each row.
    def text(xx, yy, s, size=10, **kw):
        fig.text(xx, yy, s, fontsize=size, color=kw.pop("color", INK), **kw)

    def header(j, y, number, name):
        text(panel_x[j], y, number, 13, fontweight="bold")
        text(panel_x[j]+.025, y, name, 13, fontweight="bold")

    def axis(j, lower):
        bot, height = (.148, .232) if lower else (.558, .244)
        ax = fig.add_axes([panel_x[j], bot, pwidth+.018, height])
        ax.set_aspect("equal")
        ax.axis("off")
        low, high = (lo, hi) if lower else full
        delta = high - low
        pad = delta * (.012 if lower else .03)
        ax.set_xlim(low[0]-pad[0], high[0]+pad[0])
        ax.set_ylim(high[1]+pad[1], low[1]-pad[1])
        return ax

    def scatter(ax, vals, colors, size):
        ax.scatter(*xy[vals].T, c=colors, s=size, marker="o", linewidths=0, rasterized=True)
    
    def legend_bar(j, lower, cmap, endpoints, caption):
        bx = panel_x[j] + .015
        by = .117 if lower else .515
        width=.225
        ca = fig.add_axes([bx, by, width, .008])
        ca.imshow(np.linspace(0,1,256)[None,:], cmap=cmap, aspect="auto", extent=[0,1,0,1], interpolation="nearest")
        ca.axis("off")
        text(bx, by-.021, endpoints[0],7.7,ha="left")
        if lower:
            text(bx+width/2, by-.021,"0",8,ha="center")
        text(bx+width, by-.021, endpoints[1],7.7,ha="right")
        text(bx+width/2, by-.044, caption, 8.3,ha="center",color="#495b64")
    
    text(.044,.949,d["title"],24,fontweight="bold")
    text(.044,.906,d["source"] + f"  ·  {len(xy):,} {d['cell']}  ·  {len(u):,} graph edges",11)
    fig.add_artist(plt.Line2D([.044,.967],[.886,.886],transform=fig.transFigure, color="#c1cbd0",lw=.8))
    header(0,.843,"A","Anatomical labels")
    header(1,.843,"B",d["genes"][0]+" expression")
    header(2,.843,"C",d["genes"][1]+" expression")
    ax=axis(0,False)
    ms=2.75 if d["dataset"].startswith("dlpfc") else .7
    scatter(ax,np.ones(len(xy),bool),NEUTRAL,ms)
    for g,c in zip((a,b), PALETTE):
        scatter(ax,labels==g,c,ms*1.2)
    ax.add_patch(Rectangle(lo,*(hi-lo),ec=INK,lw=1,fill=False,zorder=4))
    text(panel_x[0],.516,"●  "+short_label(a),9,color=PALETTE[0],fontweight="bold")
    text(panel_x[0],.49,"●  "+short_label(b),9,color=PALETTE[1],fontweight="bold")
    for k in range(2):
        ax = axis(k+1,False)
        vals=x[:,k]; vmax=np.quantile(vals[vals>0],.95)
        plot_scale.append(float(vmax))
        light, dark=(("#E5F5EA","#00752A") if k==0 else ("#E8F0FF","#1F4E9E"))
        cm=LinearSegmentedColormap.from_list(f"expr{k}",[light,PALETTE[k],dark])
        scatter(ax,np.ones(len(xy),bool),cm(np.clip(vals/vmax,0,1)),ms*1.05)
        ax.add_patch(Rectangle(lo,*(hi-lo),ec=INK,lw=.7,fill=False,zorder=4))
        legend_bar(k+1,False,cm,("0",f"{vmax:.2f}+"),d["expr_label"])
    fig.add_artist(plt.Line2D([.044,.967],[.458,.458],transform=fig.transFigure, color="#c1cbd0",lw=.8))
    header(0,.417,"D","Labeled neighbors")
    header(1,.417,"E","Opposing changes")
    header(2,.417,"F","Shared neighborhoods")
    ax=axis(0,True)
    node_inside=np.all((xy>=lo)&(xy<=hi),axis=1)
    sel=crossing&inside
    seg=np.stack([xy[u],xy[v]],axis=1)
    ax.add_collection(LineCollection(seg[sel],colors="#30434e",linewidths=.4,alpha=.7))
    scatter(ax, node_inside, NEUTRAL, 8)
    for g,c in zip((a,b), PALETTE):
        scatter(ax,(labels==g)&node_inside,c,14)
    text(panel_x[0],.117,f"{int(sel.sum())} / {int(crossing.sum())} cross-label edges in view",9.3)
    length=d['scale']*(.5 if d['dataset']=='seqfish' else .5)
    width=float(hi[0]-lo[0])
    s0=lo + np.array([.03,.95])*width
    ax.plot([s0[0],s0[0]+length],[s0[1],s0[1]],color=INK,lw=2.2,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=4.9,foreground='white')])
    text(panel_x[0],.09,"Scale bar  " + ("0.5 mm" if d["dataset"].startswith("dlpfc") else "0.5 affine units"),8.4,
         color="#495b64")
    text(panel_x[0],.061,f"J₁ on all {int(crossing.sum())} cross-label edges = {cross_j:+.3f}",9.0,fontweight="bold")
    for j,(component,meaning) in enumerate(((change,"J₁"),(codomain,"J (shared neighborhood)")),1):
        ax=axis(j,True)
        # Cap color symmetrically at the specimen-wide 98th percentile of |edge contribution|.
        # Plot every intersecting edge, with stronger values on top; only the colors saturate.
        cut = float(np.quantile(np.abs(component), .98))
        edge_scale.append(cut)
        ids=np.flatnonzero(np.all(seg.max(axis=1)>=lo,axis=1) & np.all(seg.min(axis=1)<=hi,axis=1))
        ids=ids[np.argsort(np.abs(component[ids]),kind="stable")]
        vals=component[ids]
        norm=Normalize(-cut,cut,clip=True)
        ax.add_collection(LineCollection(seg[ids],colors=MAP(norm(vals)),linewidths=.95, rasterized=True))
        left, right = (("parallel", "opposing") if j==1 else ("discordant", "joint high / low"))
        legend_bar(j,True,MAP,(f"−{cut*1e4:.2f}  {left}",f"{right}  +{cut*1e4:.2f}"),"edge contribution  × 10⁴")
    text(.044,.019,"Box in A–C marks the same field shown in D–F.  ",9,color="#40515c")
    text(.52,.019,f"Whole specimen     J₁ = {change.sum():+.3f}       J (regional) = {codomain.sum():+.3f}",10,
         fontweight="bold")
    root = out / (d["dataset"]+"_observed_tissue_"+stamp)
    fig.savefig(str(root)+".pdf",dpi=300, metadata={"Title":d['title']+" | Observed gene-pair expression and spatial contributions"})
    fig.savefig(str(root)+".png",dpi=270)
    plt.close(fig)
    nodes=pd.DataFrame({"node_index":np.arange(len(xy)), "observation_id":d["ids"], "x":xy[:,0], "y":xy[:,1],
                        "annotation":labels, d["genes"][0]:x[:,0],d["genes"][1]:x[:,1], "inside_zoom":node_inside})
    edges=pd.DataFrame({"node_u":u,"node_v":v,"crosses_target_labels":crossing,
                        "midpoint_inside_zoom":inside,"joint_gradient_energy_fraction":energy_fraction,"J1_contribution":change,"shared_J_contribution":codomain})
    nodes.to_csv(str(root)+"_nodes.csv.gz",index=False)
    edges.to_csv(str(root)+"_edges.csv.gz",index=False)
    summary=dict(training_mean=d.get("training_mean",np.nan),training_pair_count=d.get("candidate_count",np.nan),dataset=d["dataset"],sample=d["sample"],label_x=a,label_y=b,gene_x=d["genes"][0],gene_y=d["genes"][1],
                 n_nodes=len(x),n_edges=len(u),crossing_edges=int(crossing.sum()),crossing_in_window=int(sel.sum()),
                 crossing_joint_energy_fraction=cross_energy, J1_across_labels=cross_j, contact_z=d['contact_z'],zoom_left=lo[0],zoom_right=hi[0],zoom_min_y=lo[1],zoom_max_y=hi[1],
                 J1=float(change.sum()),J1_positive=float(change[change>0].sum()),J1_negative=float(change[change<0].sum()),
                 shared_J=float(codomain.sum()),shared_J_positive=float(codomain[codomain>0].sum()),
                 shared_J_negative=float(codomain[codomain<0].sum()),color_x_cap=plot_scale[0],color_y_cap=plot_scale[1],
                 edge_J1_color_cap=edge_scale[0],edge_shared_color_cap=edge_scale[1])
    return summary


if __name__=="__main__":
    from matplotlib import patheffects
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--stamp",required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none",
                         "text.color":INK,"axes.edgecolor":INK})
    data=sample_maps(args.repo.resolve())
    data.extend(cross_donor_example(args.repo.resolve(),data[0],args.out,args.stamp))
    summary=pd.DataFrame([draw(d,args.out,args.stamp) for d in data])
    summary.to_csv(args.out/("observed_map_summary_"+args.stamp+".csv"),index=False)
    print(summary.to_string(index=False))
