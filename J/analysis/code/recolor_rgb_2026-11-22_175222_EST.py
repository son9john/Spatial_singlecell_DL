"""Re-draw every manuscript figure in the default R (ggplot2) RGB palette.

The figure scripts of v3, v4, v5, v7 and v12 are copied unchanged except for their colour constants,
then run on the same result tables. Colour roles used throughout manuscript v13:

  boundary J            #F8766D  (ggplot2 red)
  regional J            #00BA38  (ggplot2 green)
  pooled comparators    #619CFF  (ggplot2 blue)
  individual packages   scales::hue_pal()(8) in a fixed order, J first:
      J #F8766D, SpatialDM #CD9600, LIANA+ #7CAE00, Scanpy/Seurat #00BE67, Hotspot #00BFC4,
      MERINGUE #00A9FF, Lee's L #C77CFF, Giotto #FF61CC; baselines grey #7F7F7F
  signed maps           muted blue #3A3A98 - #619CFF - white - #F8766D - muted red #832424
  expression maps       light grey - #56B1F7 - #132B43 (ggplot2 default continuous blues)

Usage: python recolor_rgb_<stamp>.py <staged J-analysis> <staged manuscript dir> <local data> <local cache>
                                     <local python package src> <v13 dir>
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ST = "2026-11-22_175222_EST"
RED, GREEN, BLUE, GREY = "#F8766D", "#00BA38", "#619CFF", "#7F7F7F"
HUE8 = ("#F8766D", "#CD9600", "#7CAE00", "#00BE67", "#00BFC4", "#00A9FF", "#C77CFF", "#FF61CC")


def patch(src, dst, pairs):
    text = Path(src).read_text()
    for old, new in pairs:
        n = text.count(old)
        if n < 1:
            raise SystemExit(f"{Path(src).name}: pattern not found: {old[:70]!r}")
        text = text.replace(old, new)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(text)


def run(cmd, env):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def main(staged, staged_ms, data, cache, pkg_src, v13):
    staged, staged_ms, v13 = Path(staged), Path(staged_ms), Path(v13)
    mirror = v13 / "mirror" / "J-analysis"
    if mirror.exists():
        shutil.rmtree(mirror.parent)
    shutil.copytree(staged, mirror, ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "seqfish", "dlpfc", "crc"))
    for sub in ("data", "cache", "packages/python_v0.3"):
        if (mirror / sub).is_symlink() or (mirror / sub).is_file():
            (mirror / sub).unlink()
        elif (mirror / sub).exists():
            shutil.rmtree(mirror / sub)
    (mirror / "data").symlink_to(data)
    (mirror / "cache").symlink_to(cache)
    (mirror / "packages" / "python_v0.3").mkdir(parents=True)
    (mirror / "packages" / "python_v0.3" / "src").symlink_to(pkg_src)
    for d in ("figures/v2", "figures/v3", "figures/v4"):
        (mirror / d).mkdir(parents=True, exist_ok=True)
    rgb = v13 / "code" / "rgb"
    out = v13 / "figures"
    env = dict(os.environ, JBENCH_ROOT=str(mirror), MPLBACKEND="Agg")
    b3 = mirror / "benchmarks" / "v3"

    # shared palettes of the v3 figure modules (Figs 2, 3, 5, 6, 7)
    patch(b3 / "figures_v2.py", b3 / "figures_v2.py", [
        ('"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"',
         ", ".join(f'"{c}"' for c in HUE8)),
        ('"Lee\'s L": ORANGE', '"Lee\'s L": VIOLET'),
        ('"J (scale 0)": VIOLET', '"J (scale 0)": RED'),
        ('["#1c5cab", "#86b6ef", "#f0efec", "#ef9a92", "#b8302f"]', '["#3A3A98", "#619CFF", "#F7F7F7", "#F8766D", "#832424"]'),
        ('["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"]', '["#F2F2F2", "#9DCDF9", "#56B1F7", "#2F6E9E", "#132B43"]'),
    ])
    patch(b3 / "figures_v3.py", b3 / "figures_v3.py", [('"baseline": MUTED}', f'"baseline": "{GREY}"}}')])
    rgb.mkdir(parents=True, exist_ok=True)
    shutil.copy(b3 / "figures_v2.py", rgb / "figures_v2_rgb.py")
    shutil.copy(b3 / "figures_v3.py", rgb / "figures_v3_rgb.py")
    run([sys.executable, b3 / "figures_v2.py", "4"], env)          # Fig 3 (colorectal)
    code = ("import sys; sys.path.insert(0, %r); import figures_v3 as f; f.figure5(); f.figure6(); f.figure7()" % str(b3))
    run([sys.executable, "-c", code], env)                          # Figs 5, 6, 7
    for name in ("fig4_crc", "fig5_local_jaccard", "fig6_jaccard_repro_redundancy", "fig7_speed"):
        for ext in ("png", "pdf"):
            shutil.copy(mirror / "figures" / "v3" / f"{name}.{ext}", out / f"{name}_{ST}.{ext}")

    # Fig 2 (v7 calibration figure; imports the patched figures_v3 palette)
    f2 = rgb / f"figure2_{ST}.py"
    patch(mirror / "revisions/v7_2026-11-22_133437_EST/code/figure2_2026-11-22_133437_EST.py", f2, [("import argparse", "import argparse")])
    run([sys.executable, f2, mirror, out], env)

    # Fig 4 (v4 semi-synthetic figure)
    f4dir = v13 / "mirror" / "manuscript" / "source_v4"
    f4dir.mkdir(parents=True, exist_ok=True)
    patch(staged_ms / "source_v4" / "figure4.py", f4dir / "figure4.py", [
        ('BLUE, DARK, TEAL, GRID = "#145492", "#384b5b", "#22796f", "#e5eaf0"',
         f'BLUE, DARK, TEAL, GRID = "{RED}", "{GREY}", "{BLUE}", "#e5eaf0"')])
    shutil.copy(f4dir / "figure4.py", rgb / f"figure4_{ST}.py")
    code = ("import sys; sys.path.insert(0, %r); import figure4 as f; f.simulation_figure(f.matched_simulations(), %r)"
            % (str(f4dir), ST))
    run([sys.executable, "-c", code], env)
    for ext in ("png", "pdf"):
        shutil.copy(mirror / "figures" / "v4" / f"figure4_pair_accuracy_{ST}.{ext}", out)

    # Fig 8 and Fig 9 (v12)
    v12 = mirror / "revisions/v12_2026-11-22_135104_EST/code"
    f8 = rgb / f"figure8_{ST}.py"
    patch(v12 / "figure8_2026-11-22_135104_EST.py", f8, [("REGION = '#06605F'", f"REGION = '{GREEN}'"),
                                                          ("CONTROL = '#465C70'", f"CONTROL = '{BLUE}'")])
    run([sys.executable, f8, "--metrics", mirror / "revisions/v5_2026-11-22_104348_EST/results/metrics_by_unit_2026-11-22_104504_EST.csv",
         "--output", out], env)
    f9 = rgb / f"figure9_{ST}.py"
    patch(v12 / "figure9_2026-11-22_142751_EST.py", f9, [
        ('RUST, TEAL, SLATE = "#B05A2A", "#06605F", "#465C70"', f'RUST, TEAL, SLATE = "{RED}", "{GREEN}", "{BLUE}"')])
    (v13 / "results").mkdir(exist_ok=True)
    run([sys.executable, f9, mirror, out, "--source-output", v13 / "results"], env)
    src9 = v13 / "results" / "figure9_boundary_method_auroc_2026-11-22_142751_EST_source.csv"
    src9.replace(v13 / "results" / f"figure9_boundary_method_auroc_{ST}_source.csv")

    # Fig S1 (v5 observed accuracy with both J scores)
    s1 = rgb / f"figure_s1_{ST}.py"
    patch(mirror / "revisions/v5_2026-11-22_104348_EST/code/figure_s1.py", s1, [
        ("'#0b665d'", f"'{GREEN}'"), ("'#aa512f'", f"'{RED}'"), ("'#384f63'", f"'{BLUE}'"),
        # v13: anatomical proximity is a co-localization question; Fig S1 shows regional J and comparators only
        ("METHODS={'J1':'Boundary J₁','J_region':'Regional J',", "METHODS={'J_region':'Regional J',"),
        ("HIST={'J_unnorm':'Boundary J₀ (weighted)','Pearson':'Pearson',", "HIST={'Pearson':'Pearson',"),
        ("[(k,v,k!='J_unnorm') for k,v in historical]", "[(k,v,True) for k,v in historical]")])
    run([sys.executable, s1, mirror / "revisions/v5_2026-11-22_104348_EST/results", out], env)

    # Fig S2 (package-comparison overlap)
    pc = mirror / "benchmarks" / "package_comparison"
    patch(pc / "plot_overlap.py", pc / "plot_overlap.py", [
        ('"Regularized J", "#2066a6"', f'"Regularized J", "{RED}"'),
        ('"Gradient-difference magnitude", "#212529"', f'"Gradient-difference magnitude", "{GREY}"'),
        ('"LIANA+ weighted Pearson (negative)", "#bd570b"', f'"LIANA+ weighted Pearson (negative)", "{HUE8[2]}"'),
        ('"SpatialDM local Moran (positive)", "#765396"', f'"SpatialDM local Moran (positive)", "{HUE8[1]}"'),
        ('"LIANA+ weighted Jaccard (positive)", "#68727b"', f'"LIANA+ weighted Jaccard (positive)", "{HUE8[4]}"'),
        ('"Opposite-gradient weighted Jaccard", "#00795b"', f'"Opposite-gradient weighted Jaccard", "{HUE8[5]}"'),
        ('("j", "J, scale 1", "#2066a6")', f'("j", "J, scale 1", "{RED}")'),
        ('("lee_positive", "Lee\'s L, positive", "#bd570b")', f'("lee_positive", "Lee\'s L, positive", "{HUE8[6]}")'),
        # v13: the three local panels only (the global touching-vs-separated panel is a proximity task)
        ("fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))", "fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4))"),
        ("fig.subplots_adjust(top=.91, bottom=.17, right=.98, left=.105, hspace=.55, wspace=.38)",
         "fig.subplots_adjust(top=.88, bottom=.38, right=.985, left=.075, wspace=.42)"),
        ("handles, labels = axes[0, 0].get_legend_handles_labels()", "handles, labels = axes[0].get_legend_handles_labels()"),
        ('"B  Exact · annotated seqFISH"', '"B  Exact · seqFISH"'), ('"C  One-hop · annotated seqFISH"', '"C  One-hop · seqFISH"'),
    ])
    text = (pc / "plot_overlap.py").read_text()
    text = text[:text.index("        ax = axes[1, 1]\n")] + text[text.index("        handles, labels = axes[0]"):]
    (pc / "plot_overlap.py").write_text(text)
    shutil.copy(pc / "plot_overlap.py", rgb / f"plot_overlap_{ST}.py")
    run([sys.executable, pc / "plot_overlap.py"], env)
    shutil.copy(mirror / "figures" / "v2" / "figs1_jaccard_overlap.png", out / f"figs1_jaccard_overlap_{ST}.png")

    # Figs 10, 11, S3, S4 (v5 measured tissue plates)
    tp = rgb / f"tissue_plates_{ST}.py"
    patch(mirror / "revisions/v5_2026-11-22_104348_EST/code/tissue_plates_2026-11-22_104348_EST.py", tp, [
        ('PALETTE = ("#00735E", "#774095")', f'PALETTE = ("{GREEN}", "{BLUE}")'),
        ('NEG, POS = "#236ca0", "#bf4917"', f'NEG, POS = "{BLUE}", "{RED}"'),
        ('MAP = LinearSegmentedColormap.from_list("signed", [NEG, "#f5f3ef", POS])',
         'MAP = LinearSegmentedColormap.from_list("signed", ["#3A3A98", NEG, "#F7F7F7", POS, "#832424"])'),
        ('(("#e7eeeb","#005b4a") if k==0 else ("#eee9ef","#5e297d"))',
         '(("#E5F5EA","#00752A") if k==0 else ("#E8F0FF","#1F4E9E"))'),
    ])
    run([sys.executable, tp, "--repo", mirror, "--out", out, "--stamp", ST], env)

    # give every regenerated file the v13 stamp; tables written next to the figures go to results/
    import re
    for f in sorted(out.iterdir()):
        new = re.sub(r"2026-11-22_\d{4,6}_EST", ST, f.name)
        if ST not in new:
            new = f.stem + f"_{ST}" + f.suffix
        target = (v13 / "results" if f.suffix in (".csv", ".gz") else out) / new
        if target != f:
            f.rename(target)


if __name__ == "__main__":
    main(*sys.argv[1:7])
