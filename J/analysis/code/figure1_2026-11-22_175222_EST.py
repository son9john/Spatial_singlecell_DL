"""Fig 1 of manuscript v13: why a weak cross L leaves the boundary question open, and what J reads instead.

A  Controlled cycle: identical self L and |cross L| for touching and separated domains; boundary J differs.
B  Synthetic tissue with three domains (one marker each, plus a second marker of domain A).
C  Lee's L and boundary J for touching, separated and same-domain marker pairs.
D  Weight each statistic places on graph frequency (Laplacian spectrum of DLPFC section 151673).

Usage: python figure1_<stamp>.py <v12 results dir> <v13 results dir> <output dir>
The juxtaposed package (0.3 or later) must be importable for panels B and C.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import juxtaposed as jx  # noqa: E402

ST12 = "2026-11-22_135104_EST"
ST = "2026-11-22_175222_EST"
INK, MUTED, GRID = "#183644", "#506874", "#E6EAED"
# default R (ggplot2) RGB palette: domains A, B, C = red, green, blue; J red, Lee's L and Pearson as in Figs 2-7
DOM_A, DOM_B, DOM_C = "#F8766D", "#00BA38", "#619CFF"
JCOL, LEE, PEAR = "#F8766D", "#C77CFF", "#00BE67"


def synthetic():
    rng = np.random.default_rng(0)
    xy = np.column_stack(np.unravel_index(np.arange(1600), (40, 40))).astype(float)
    col = xy[:, 1]
    a, b, c = col < 13, (col >= 13) & (col < 27), col >= 27
    noise = lambda: 0.3 * rng.normal(size=1600)  # noqa: E731
    expr = np.column_stack([a + noise(), b + noise(), c + noise(), a + noise()])
    g = jx.spatial_graph(xy, neighbors=4)
    J = jx.juxtaposition_matrix(expr, g, scale=2)
    L = jx.lee_l_matrix(expr, g)
    domain = np.where(a, 0, np.where(b, 1, 2))
    return xy, domain, L, J


def draw(res12, res13, out):
    contrast = pd.read_csv(res12 / f"cycle_example_{ST12}.csv").set_index("condition")
    spec = pd.read_csv(res13 / f"spectrum_151673_{ST}.csv").iloc[0]
    xy, domain, L, J = synthetic()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(8.1, 6.3), facecolor="white")

    # A: controlled cycle (same drawing as v12 Fig 1A)
    fig.text(.04, .962, "A   Identical marker strengths, different interfaces", color=INK, weight="bold", size=12)
    fig.text(.04, .926, "Two eight-site domains on a 256-site cycle. Only the second domain moves.", color=MUTED, size=8.7)
    fig.text(.05, .872, "Domain arrangement", size=8.8, color=INK, weight="bold")
    for i, k in enumerate(["Self L, each", "Cross L", "Boundary J₁"]):
        fig.text([.61, .74, .87][i], .872, k, weight="bold", size=8.7, color=INK, ha="center")
    for i, key in enumerate(["touch", "apart"]):
        y = .812 - i * .072
        ax = fig.add_axes([.162, y - .017, .375, .048])
        ax.set_xlim(42, 94)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.plot([42, 94], [.5, .5], color="#c5d4d5", lw=2, zorder=1)
        ax.barh(.5, 8, left=48, height=.62, color=DOM_A, zorder=2)
        ax.barh(.5, 8, left=56 if key == "touch" else 80, height=.62, color=DOM_B, zorder=2)
        fig.text(.05, y, "Touching" if key == "touch" else "Separated", size=9.1, color=INK)
        row = contrast.loc[key]
        for cx, s in zip([.61, .74, .87], [f"{row.Lxx:.3f}", f"{row.Lxy:+.3f}".replace("-", "−"), f"{max(row.J1, 0):.3f}"]):
            fig.text(cx, y, s, ha="center", size=9.7, color=INK, weight="bold" if cx == .87 else "normal")
    fig.add_artist(Line2D([.04, .96], [.692, .692], transform=fig.transFigure, color="#d0dadd", lw=.8))

    # B: synthetic tissue
    fig.text(.04, .645, "B   Three domains", color=INK, weight="bold", size=11)
    fig.text(.04, .612, "40 × 40 tissue, one marker per domain", color=MUTED, size=8.3)
    ax = fig.add_axes([.05, .15, .24, .38])
    from matplotlib.colors import ListedColormap
    grid = domain.reshape(40, 40)
    ax.imshow(grid, cmap=ListedColormap([DOM_A, DOM_B, DOM_C]), interpolation="nearest", vmin=0, vmax=2)
    ax.axis("off")
    for k, (name, cx) in enumerate([("A", 6), ("B", 19.5), ("C", 33)]):
        ax.text(cx, 19.5, name, ha="center", va="center", color="white", weight="bold", size=14)
    fig.text(.05, .082, "Genes A and A′ both mark domain A.\nNoise SD 0.3 on every spot.", color=MUTED, size=7.6,
             va="top")

    # C: Lee's L versus boundary J
    fig.text(.355, .645, "C   Only J separates touching", color=INK, weight="bold", size=11)
    fig.text(.355, .612, "Lee's L and boundary J₂ for three pairs", color=MUTED, size=8.3)
    ax = fig.add_axes([.385, .15, .30, .38])
    pairs = [("A–B\ntouching", 0, 1), ("A–C\nseparated", 0, 2), ("A–A′\nsame domain", 0, 3)]
    xpos = np.arange(3)
    lv = [L[i, j] for _, i, j in pairs]
    jv = [J[i, j] for _, i, j in pairs]
    w = .34
    ax.bar(xpos - w / 2, lv, width=w, color=LEE, edgecolor="white", linewidth=1, zorder=2)
    ax.bar(xpos + w / 2, jv, width=w, color=JCOL, edgecolor="white", linewidth=1, zorder=2)
    for k in range(3):
        for dx, v in [(-w / 2, lv[k]), (w / 2, jv[k])]:
            ax.text(k + dx, v + (.03 if v >= 0 else -.03), f"{v:+.2f}".replace("-", "−"), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.6, color=INK)
    ax.axhline(0, color="#8e9aa3", lw=.8, zorder=1)
    ax.set_xticks(xpos, [p[0] for p in pairs], fontsize=8.2, color=INK)
    ax.set_ylim(-.55, .9)
    ax.set_yticks([-.5, 0, .5])
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, lw=.6, zorder=0)
    ax.spines["left"].set_color("#B3BEC5")
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Patch(color=LEE, label="Lee's L"), Patch(color=JCOL, label="Boundary J₂")],
              frameon=False, loc="upper left", fontsize=8, handlelength=1.1, ncol=2, bbox_to_anchor=(0, 1.04))

    # D: frequency weighting
    fig.text(.725, .645, "D   Frequency weighting", color=INK, weight="bold", size=11)
    fig.text(.725, .612, "Laplacian spectrum, DLPFC 151673", color=MUTED, size=8.3)
    ax = fig.add_axes([.765, .15, .195, .38])
    mu_max, dmax, dbar = float(spec.mu_max), float(spec.d_max), 6.0
    mu = np.linspace(0, mu_max, 400)
    curves = [("Pearson r", np.ones_like(mu), PEAR, (0, (4, 2))),
              ("Lee's L", (1 - mu / dbar) ** 2, LEE, "-"),
              ("Boundary J₁", mu * (1 - mu / (2 * dmax)) ** 2, JCOL, "-")]
    for name, h, c, ls in curves:
        ax.plot(mu / mu_max, h / np.max(np.abs(h)), color=c, lw=1.9, ls=ls, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(-.03, 1.32)
    ax.set_xticks([0, 1], ["smooth", "sharp"], fontsize=8, color=INK)
    ax.set_yticks([0, .5, 1])
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.set_ylabel("Relative weight", color=MUTED, fontsize=8.4)
    ax.yaxis.grid(True, color=GRID, lw=.6, zorder=0)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#B3BEC5")
    ax.legend(frameon=False, loc="upper center", fontsize=7.6, handlelength=1.6, ncol=1,
              bbox_to_anchor=(.5, 1.03))
    fig.text(.765, .082, "J ignores the shared smooth trend\nand reads the gradients.", color=MUTED, size=7.6, va="top")

    out.mkdir(exist_ok=True, parents=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"figure1_concept_{ST}.{ext}", dpi=340)
    plt.close(fig)
    vals = pd.DataFrame({"pair": ["A-B touching", "A-C separated", "A-A' same domain"], "lee_L": lv, "boundary_J2": jv})
    vals.to_csv(Path(out).parent / "results" / f"figure1_synthetic_values_{ST}.csv", index=False)
    return vals


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("results_v12", type=Path)
    p.add_argument("results_v13", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(draw(a.results_v12, a.results_v13, a.output))
