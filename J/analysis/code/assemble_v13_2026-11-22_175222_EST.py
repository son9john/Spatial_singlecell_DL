"""Assemble the v13 Markdown source: new front matter, Results and Discussion (top_v13.md) + the v12 Methods,
references and supplement with v13 edits, then bold the best value in supplementary metric tables that had none.

Usage: python assemble_v13_<stamp>.py <v12 markdown> <top_v13.md> <output markdown> <time string, e.g. 18:40>
"""
import re
import sys
from pathlib import Path

ST = "2026-11-22_175222_EST"
OLD_FIG = {
    "figures/figure_s1_observed_pair_accuracy_2026-11-22_104348_EST.png": f"figures/figure_s1_observed_pair_accuracy_{ST}.png",
    "figures/v2/figs1_jaccard_overlap.png": f"figures/figs1_jaccard_overlap_{ST}.png",
    "figures/dlpfc_training_regional_observed_tissue_2026-11-22_104348_EST.png": f"figures/dlpfc_training_regional_observed_tissue_{ST}.png",
    "figures/dlpfc_observed_tissue_2026-11-22_104348_EST.png": f"figures/dlpfc_observed_tissue_{ST}.png",
}
ROW_TABLES = ("S2", "S3", "S14", "S15")      # rows are endpoints, columns are methods
COL_TABLES = ("S5", "S6", "S7", "S9", "S12")        # rows are methods, columns are settings
EXCLUDE_COLS = {"Pairs", "Reference interface"}


def sub1(text, old, new):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected one occurrence, found {n}: {old[:80]!r}")
    return text.replace(old, new)


def number(cell):
    c = cell.strip().replace("**", "").replace("−", "-").replace(",", "")
    try:
        return float(c)
    except ValueError:
        return None


def bold(cell):
    c = cell.strip()
    return f" {c} " if c.startswith("**") else f" **{c}** "


def bold_table(lines, by):
    """lines: Markdown table lines. by: 'row' or 'col'. Returns new lines."""
    header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
    body = [[c for c in ln.strip().strip("|").split("|")] for ln in lines[2:]]
    keep = [j for j, h in enumerate(header) if j > 0 and h not in EXCLUDE_COLS]
    if by == "row":
        for row in body:
            vals = {j: number(row[j]) for j in keep if j < len(row)}
            vals = {j: v for j, v in vals.items() if v is not None}
            if len(vals) >= 2:
                best = max(vals.values())
                for j, v in vals.items():
                    if abs(v - best) < 1e-12:
                        row[j] = bold(row[j])
    else:
        for j in keep:
            vals = {i: number(r[j]) for i, r in enumerate(body) if j < len(r)}
            vals = {i: v for i, v in vals.items() if v is not None}
            if len(vals) >= 2:
                best = max(vals.values())
                for i, v in vals.items():
                    if abs(v - best) < 1e-12:
                        body[i][j] = bold(body[i][j])
    return lines[:2] + ["|" + "|".join(r) + "|" for r in body]


def bold_section(text, table_id, by):
    head = f"## Supplementary Table {table_id}."
    start = text.index(head)
    nxt = text.find("\n## ", start + 1)
    end = nxt if nxt != -1 else len(text)
    sec = text[start:end].split("\n")
    out, i, first = [], 0, True
    while i < len(sec):
        if sec[i].startswith("|") and i + 1 < len(sec) and re.match(r"^\|[\s:|-]+\|$", sec[i + 1].strip()):
            j = i
            while j < len(sec) and sec[j].startswith("|"):
                j += 1
            if first:
                out += [f"Bold marks the highest value in each {'row' if by == 'row' else 'column'}.", ""]
                first = False
            out += bold_table(sec[i:j], by)
            i = j
        else:
            out.append(sec[i])
            i += 1
    return text[:start] + "\n".join(out) + text[end:]


def edit_table(text, table_id, drop_cols=(), drop_rows=()):
    """Drop named columns and rows (by first-cell prefix) from every table in a supplementary section."""
    head = f"## Supplementary Table {table_id}."
    start = text.index(head)
    nxt = text.find("\n## ", start + 1)
    end = nxt if nxt != -1 else len(text)
    sec = text[start:end].split("\n")
    out, i = [], 0
    while i < len(sec):
        if sec[i].startswith("|") and i + 1 < len(sec) and re.match(r"^\|[\s:|-]+\|$", sec[i + 1].strip()):
            j = i
            while j < len(sec) and sec[j].startswith("|"):
                j += 1
            rows = [ln.strip().strip("|").split("|") for ln in sec[i:j]]
            header = [h.strip() for h in rows[0]]
            keep = [k for k, h in enumerate(header) if h not in drop_cols]
            new = []
            for r_i, r in enumerate(rows):
                if r_i >= 2 and any(r[0].strip().startswith(d) for d in drop_rows):
                    continue
                new.append("|" + "|".join(r[k] for k in keep) + "|")
            out += new
            i = j
        else:
            out.append(sec[i])
            i += 1
    return text[:start] + "\n".join(out) + text[end:]


def drop_section(text, heading):
    start = text.index(heading)
    nxt = text.find("\n## ", start + 1)
    return text[:start] + text[nxt + 1:]


def drop_paragraph(text, prefix):
    start = text.index("\n" + prefix) + 1
    end = text.index("\n", start)
    return text[:start] + text[end + 2:] if text[end:end + 2] == "\n\n" else text[:start] + text[end + 1:]


def misuse_edits(rest):
    """v13: boundary J is reported only for its indicated uses; proximity endpoints use regional J."""
    rest = sub1(rest, "The amendment uses each statistic's natural sign, and both are reported.",
                "The amendment uses each statistic's natural sign. The archived endpoint values are listed in Table S16. "
                "Because anatomical proximity of measured markers is a co-localization question, the primary endpoint was "
                "not met by one-step boundary J (mean AUROC 0.451) and is evaluated with regional J in this manuscript.")
    rest = sub1(rest, " The three-term screen and its empirical validation are retrospective.", "")
    rest = drop_paragraph(rest, "Within each specimen we required both self terms to be at least the median")
    rest = drop_paragraph(rest, "We used the existing held-out marker assignments and evaluated AUROC")
    rest = sub1(rest, "## Supplementary Results. Graph identities, calibration and the original proximity endpoint",
                "## Supplementary Results. Graph identities and calibration")
    rest = drop_paragraph(rest, "We next asked whether marker pairs of touching domains can be told apart")
    rest = drop_paragraph(rest, "The seqFISH mouse embryos [20] gave the same picture.")
    rest = drop_paragraph(rest, "The lab’s 2021 bivariate spatial correlation (BSC) [5] reached mean AUROC")
    # Table S2: co-localization statistics only, proximity endpoints only
    rest = edit_table(rest, "S2", drop_cols=("J~1~", "J~0~", "Q"), drop_rows=("DLPFC adjacent vs same", "seqFISH enriched vs same type"))
    rest = sub1(rest, "## Supplementary Table S2. Observed-expression pair-level accuracy",
                "## Supplementary Table S2. Co-localization statistics on observed anatomical proximity")
    rest = sub1(rest, " No same-domain result in this table is emphasized as an overall winner.",
                " Regional J on the same endpoints is in Table S14; touching versus same-domain markers are in Table S15 and Fig 9.")
    rest = edit_table(rest, "S7", drop_rows=("J (scale 1)",))
    rest = edit_table(rest, "S14", drop_cols=("Boundary J",))
    rest = sub1(rest, "Columns show the primary methods and the two leading ecosystem comparators from Fig 8. Bold is reserved for J when its unrounded metric also equals the best among all computed methods for that endpoint, including single-scale sensitivity analyses.",
                "Columns show regional J, its matched-scale Pearson control and the two leading ecosystem comparators from Fig 8.")
    rest = sub1(rest, "Fig S1. Observed expression with both J scores and the comparator panel.** A shows cortical layer adjacency. B shows embryonic cell-type contact enrichment with markers selected in the other embryos. CrossExpression used its native single-neighbor graph. C shows the within-embryo contact-rate design and reproduces the historical sign comparison. Diamonds show mean natural-orientation AUROC. Small points show specimens for methods without reversed scores. Open points connected to filled diamonds in C mark the historical sign-reversed result.",
                "Fig S1. Regional J and the full comparator panel for anatomical proximity.** A shows cortical layer adjacency. B shows embryonic cell-type contact enrichment with markers selected in the other embryos. CrossExpression used its native single-neighbor graph. C shows the within-embryo contact-rate design; open points connected to filled diamonds mark the sign-reversed orientation used in an earlier analysis. Diamonds show mean natural-orientation AUROC. Small points show specimens for methods without reversed scores.")
    rest = sub1(rest, " (D) Global overlap for touching versus separated gene pairs shows the effect of the direction of Lee's L.", "")
    rest = sub1(rest, "(A–C) Local exact and one-hop IoU curves on simulated MERFISH counts and annotated seqFISH contacts.",
                "(A–C) Local exact and one-hop IoU curves on simulated MERFISH counts and annotated seqFISH cell-type contacts.")
    rest = sub1(rest, "\n\nBold marks J when it reaches the highest unrounded value in its row.", "")
    rest = drop_section(rest, "## Supplementary Table S16.")
    rest = drop_section(rest, "## Supplementary Table S17.")
    rest = sub1(rest, "## Supplementary Table S18. Original boundary benchmarking endpoints",
                "## Supplementary Table S16. Archived endpoints of the original benchmark protocol")
    rest = sub1(rest, "**Table S18.** Boundary-score endpoints.",
                "**Table S16.** Endpoints of the original benchmark protocol, kept as its record. Rows on anatomical proximity "
                "(DLPFC adjacent versus distant layers, seqFISH touching versus separated types) apply boundary J outside its "
                "indicated use; regional J is the score for those endpoints (Fig 8).")
    return rest


def main(v12_path, top_path, out_path, clock):
    v12 = Path(v12_path).read_text()
    lines = v12.split("\n")
    moved_cross = lines[126]
    moved_merfish = lines[128]
    moved_screen = lines[158]
    assert moved_cross.startswith("A published single-cell comparator, CrossExpression")
    assert moved_merfish.startswith("On four additional MERFISH sections")
    assert moved_screen.startswith("The 2021 lag terms help identify maps worth inspecting")
    rest = v12[v12.index("# Methods"):]

    # Methods
    rest = sub1(rest, "default or tutorial setting of its test (Table 2).", "default or tutorial setting of its test (Table 3).")
    rest = sub1(rest, "**Table 2.** Comparators and their default tests.", "**Table 3.** Comparators and their default tests.")
    rest = sub1(rest, "is developed in Supplementary Results and Fig S5.", "is developed in Supplementary Results and Fig 1D.")
    rest = sub1(rest, "The matched subset of the published colorectal counts is documented in PROVENANCE.md.",
                "The matched subset of the published colorectal counts is documented in PROVENANCE.md. For v13, every figure "
                "was redrawn from the archived result tables with only its colour constants changed to the default R "
                "(ggplot2) palette, and Fig 1B–D was recomputed with the package. The redrawn Fig 9 source table and the "
                "tissue-map summaries reproduced the archived values to within 9 × 10^−15^.")

    # Supplementary figures: new files; the v3 concept figure is now Fig 1B–D
    for old, new in OLD_FIG.items():
        rest = sub1(rest, f"]({old})", f"]({new})")
    concept = rest.index("![Fig S5. J reads shared transitions.")
    rest = rest[:concept] + rest[rest.index("\n", concept) + 1:]
    rest = sub1(rest, "which vanishes at zero frequency (Fig S5D).", "which vanishes at zero frequency (Fig 1D).")
    rest = sub1(rest, "whereas J separated them (Fig S5A–C).", "whereas J separated them (Fig 1B,C).")

    moved_merfish = moved_merfish.replace(", Hotspot 0.547 and boundary J 0.494.", " and Hotspot 0.547.")
    assert "boundary J 0.494" not in moved_merfish
    moved = f"""## Supplementary Results. Additional comparators and anatomical transfer

**CrossExpression.** {moved_cross}

**MERFISH transfer.** {moved_merfish}

"""
    rest = sub1(rest, "## Supplementary Table S16.", moved + "## Supplementary Table S16.")
    rest = misuse_edits(rest)

    for t in ROW_TABLES:
        rest = bold_section(rest, t, "row")
    for t in COL_TABLES:
        rest = bold_section(rest, t, "col")

    top = Path(top_path).read_text().replace("@@ST@@", ST).replace("@@TIME@@", clock)
    out = top.rstrip() + "\n\n" + rest
    for bad in ("Fig S5", "Table S17", "Table S18", "S18", "boundary J 0.494", "figures/v3/", "figures/v4/", "_2026-11-22_104348_EST.png", "Voyager"):
        if bad in out:
            raise SystemExit(f"left over: {bad}")
    Path(out_path).write_text(out)
    print("wrote", out_path, len(out))


if __name__ == "__main__":
    main(*sys.argv[1:5])
