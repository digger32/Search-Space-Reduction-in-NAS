#!/usr/bin/env python3
"""Figures for B6, read from <outdir> after aggregate.py.

  anytime_grid.pdf       anytime test accuracy, mean +- bootstrap 95% CI bands,
                         rows = datasets, columns = algorithms (print width)
  anytime_<dataset>.pdf  the same, one file per dataset
  cd_spaces.pdf          critical-difference diagram over spaces
                         (Friedman blocks = dataset x algorithm)
  effect_heatmap.pdf     mean paired delta (reduction - matched random) per
                         dataset x algorithm, stars for Holm-significant cells
  budget_alloc.pdf       fraction of the best trajectory inside each informed
                         region for FULL-space searches
  advantage_vs_budget.pdf paired advantage of each reduction over its
                         size-matched control vs budget, per algorithm

v3 (camera-ready, reviewer R2.4): every space, algorithm and dataset is
labelled exactly as in the manuscript (Table I names, abbreviations defined in
Section III-C) and plotted in the manuscript's order, not alphabetically.
Colours and line styles are fixed per space, so a space looks the same in
every figure. Statistics are unchanged.

Greyscale-legible, colourblind-safe (Okabe-Ito), vector output, fonts sized
at final print width (IEEE column 3.5 in, text width 7.16 in).
Usage: python3 make_figures.py <run_dir> [<figdir>]
       (write figures outside a frozen run directory with the 2nd argument)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# --- manuscript vocabulary and order ----------------------------------------
SPACE_ORDER = ["full", "no_none", "synflow50", "naswot50", "param50",
               "randM4096", "randM7813"]
SPACE_LABEL = {"full": "FULL", "no_none": "NO-NONE",
               "synflow50": "SYNFLOW-50", "naswot50": "NASWOT-50",
               "param50": "PARAM-50", "randM4096": "RAND-4096",
               "randM7813": "RAND-7813"}
ALGO_ORDER = ["rs", "re", "reinforce", "ls"]
ALGO_LABEL = {"rs": "RS", "re": "RE", "reinforce": "REINFORCE", "ls": "LS"}
DATASET_ORDER = ["cifar10-valid", "cifar100", "ImageNet16-120"]
REDUCTION_ORDER = ["no_none", "synflow50", "naswot50", "param50"]
N_ARCHS = 15625
REGION_SIZE = {"no_none": 4096, "synflow50": 7813, "naswot50": 7813,
               "param50": 7813}   # Table I; frozen by construction

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]
# seven distinct dash patterns, so every space stays identifiable in greyscale
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)),
              (0, (1, 1))]
SPACE_COLOUR = {s: OKABE_ITO[i] for i, s in enumerate(SPACE_ORDER)}
SPACE_STYLE = {s: LINESTYLES[i] for i, s in enumerate(SPACE_ORDER)}
N_BOOT = 2000


def ordered(present, order):
    """Items of `order` that are present, then any unexpected extras."""
    present = set(present)
    return [x for x in order if x in present] + sorted(present - set(order))


def load(outdir):
    rows = []
    for p in sorted(Path(outdir).glob("*__*__*__b*.json")):
        u = json.loads(p.read_text())
        for s in u["seeds"]:
            rows.append((u["dataset"], u["algo"], u["space"], s))
    return rows


def boot_ci(vals, rng):
    boots = [np.mean(rng.choice(vals, len(vals), replace=True))
             for _ in range(N_BOOT)]
    return np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def _curves(rows, ds, al, sp):
    curves = defaultdict(list)
    for d, a, s, rec in rows:
        if d == ds and a == al and s == sp:
            for ck, t in rec["anytime_test"].items():
                curves[int(ck)].append(t)
    return curves


def _draw_space(ax, curves, sp, rng, lw):
    cks = sorted(curves)
    mean = [np.mean(curves[c]) for c in cks]
    lohi = [boot_ci(np.array(curves[c]), rng) for c in cks]
    ax.plot(cks, mean, label=SPACE_LABEL.get(sp, sp),
            color=SPACE_COLOUR.get(sp, "#777777"),
            linestyle=SPACE_STYLE.get(sp, "-"), linewidth=lw)
    ax.fill_between(cks, [l for l, _ in lohi], [h for _, h in lohi],
                    color=SPACE_COLOUR.get(sp, "#777777"), alpha=0.15,
                    linewidth=0)


def fig_anytime(rows, figdir):
    rng = np.random.default_rng(1)
    datasets = ordered({r[0] for r in rows}, DATASET_ORDER)
    algos = ordered({r[1] for r in rows}, ALGO_ORDER)
    spaces = ordered({r[2] for r in rows}, SPACE_ORDER)

    for ds in datasets:
        fig, axes = plt.subplots(1, len(algos), figsize=(3.2 * len(algos), 2.8),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, al in zip(axes, algos):
            for sp in spaces:
                c = _curves(rows, ds, al, sp)
                if c:
                    _draw_space(ax, c, sp, rng, 1.4)
            ax.set_xscale("log")
            ax.set_title(ALGO_LABEL.get(al, al))
            ax.set_xlabel("queries")
        axes[0].set_ylabel("best test accuracy (%)")
        axes[-1].legend(fontsize=6, frameon=False)
        fig.suptitle(ds)
        fig.tight_layout()
        fig.savefig(figdir / f"anytime_{ds}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(len(datasets), len(algos),
                             figsize=(7.16, 1.5 * len(datasets)), sharex=True)
    axes = np.atleast_2d(axes)
    for r, ds in enumerate(datasets):
        for c_i, al in enumerate(algos):
            ax = axes[r, c_i]
            for sp in spaces:
                c = _curves(rows, ds, al, sp)
                if c:
                    _draw_space(ax, c, sp, rng, 1.0)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(ALGO_LABEL.get(al, al))
            if r == len(datasets) - 1:
                ax.set_xlabel("queries")
            if c_i == 0:
                ax.set_ylabel(f"{ds}\ntest acc. (%)", fontsize=7)
            ax.tick_params(labelsize=6.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figdir / "anytime_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cd(outdir, figdir):
    ph_path = Path(outdir) / "stats" / "posthoc.json"
    if not ph_path.exists():
        return
    ph = json.loads(ph_path.read_text())
    if "error" in ph:
        return
    import pandas as pd
    import scikit_posthocs as sp_ph
    names = [SPACE_LABEL.get(t, t) for t in ph["treatments"]]
    ranks = pd.Series({SPACE_LABEL.get(k, k): v
                       for k, v in ph["mean_ranks"].items()})
    pmat = pd.DataFrame(ph["p_matrix"], index=names, columns=names)
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    sp_ph.critical_difference_diagram(ranks, pmat, ax=ax,
                                      label_props={"fontsize": 7})
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "cd_spaces.pdf")
    plt.close(fig)


def fig_effect_heatmap(outdir, figdir):
    pj = Path(outdir) / "stats" / "paired.json"
    if not pj.exists():
        return
    paired = json.loads(pj.read_text())
    # rows in the order of Table III: dataset blocks, algorithms inside
    cells = []
    for ds in ordered({k.split("|")[0] for k in paired}, DATASET_ORDER):
        for al in ordered({k.split("|")[1] for k in paired
                           if k.startswith(ds + "|")}, ALGO_ORDER):
            if f"{ds}|{al}" in paired:
                cells.append(f"{ds}|{al}")
    reds = ordered({c["reduction"] for comps in paired.values() for c in comps},
                   REDUCTION_ORDER)
    if not cells or not reds:
        return
    mat = np.full((len(cells), len(reds)), np.nan)
    stars = np.zeros_like(mat, dtype=bool)
    for i, cell in enumerate(cells):
        for c in paired[cell]:
            if c["reference"] != "matched_random":
                continue
            j = reds.index(c["reduction"])
            mat[i, j] = c.get("mean_delta", c.get("median_delta"))
            stars[i, j] = c["significant_0.05"]
    row_labels = [f"{c.split('|')[0]} · {ALGO_LABEL.get(c.split('|')[1], c)}"
                  for c in cells]
    fig, ax = plt.subplots(figsize=(3.5, 0.55 + 0.19 * len(cells)))
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="PuOr", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(reds)), [SPACE_LABEL.get(r, r) for r in reds],
                  rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(cells)), row_labels, fontsize=6.5)
    for i in range(len(cells)):
        for j in range(len(reds)):
            if stars[i, j]:
                dark = abs(mat[i, j]) > 0.6 * vmax   # keep the star visible
                ax.text(j, i, "*", ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
    cb = fig.colorbar(im)
    cb.set_label("mean Δ (pp), reduction − control", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "effect_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_budget_alloc(rows, figdir):
    frac = defaultdict(lambda: defaultdict(list))
    for d, a, sp, rec in rows:
        if sp != "full":
            continue
        for token, f in rec.get("membership_frac_of_best_trajectory",
                                {}).items():
            frac[d][token].append(f)
    if not frac:
        return
    datasets = ordered(frac, DATASET_ORDER)
    tokens = ordered({t for d in frac.values() for t in d}, REDUCTION_ORDER)
    x = np.arange(len(tokens))
    width = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for k, ds in enumerate(datasets):
        vals = [np.mean(frac[ds].get(t, [np.nan])) for t in tokens]
        ax.bar(x + k * width, vals, width, label=ds,
               color=OKABE_ITO[k % 8], edgecolor="black", linewidth=0.4)
    # size share of each region (dashed), the reference the caption refers to
    for i, t in enumerate(tokens):
        if t in REGION_SIZE:
            share = REGION_SIZE[t] / N_ARCHS
            ax.hlines(share, x[i] - width / 2, x[i] + width * (len(datasets) - 0.5),
                      colors="black", linestyles="dashed", linewidth=0.9,
                      label="size share" if i == 0 else None, zorder=3)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2,
                  [SPACE_LABEL.get(t, t) for t in tokens], fontsize=6.3)
    ax.set_ylabel("best-trajectory fraction\ninside region", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, frameon=False, ncol=len(datasets) + 1,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(figdir / "budget_alloc.pdf")
    plt.close(fig)


MATCHED = {"no_none": "randM4096", "synflow50": "randM7813",
           "naswot50": "randM7813", "param50": "randM7813"}


def fig_advantage(rows, figdir):
    """Paired advantage of each informed reduction over its size-matched
    random control, per algorithm, pooled over datasets (equal seeds per
    dataset, so the pooled mean equals the dataset average of Table IV in the
    extended version). Bands: bootstrap 95% CI of that mean."""
    rng = np.random.default_rng(1)
    by = defaultdict(dict)  # (ds, algo, space) -> seed -> anytime_test
    for d, a, s, rec in rows:
        by[(d, a, s)][rec["seed"]] = rec["anytime_test"]
    datasets = ordered({k[0] for k in by}, DATASET_ORDER)
    algos = ordered({k[1] for k in by}, ALGO_ORDER)
    reds = [r for r in REDUCTION_ORDER if any(k[2] == r for k in by)]
    fig, axes = plt.subplots(1, len(algos), figsize=(7.16, 1.75), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, al in zip(axes, algos):
        for red in reds:
            ctrl = MATCHED[red]
            cks = None
            diffs_at = defaultdict(list)
            for ds in datasets:
                A, B = by.get((ds, al, red), {}), by.get((ds, al, ctrl), {})
                for s in A:
                    if s in B:
                        for ck, v in A[s].items():
                            diffs_at[int(ck)].append(v - B[s][ck])
            if not diffs_at:
                continue
            cks = sorted(diffs_at)
            mean = [np.mean(diffs_at[c]) for c in cks]
            lohi = [boot_ci(np.array(diffs_at[c]), rng) for c in cks]
            ax.plot(cks, mean, label=SPACE_LABEL.get(red, red),
                    color=SPACE_COLOUR.get(red), linestyle=SPACE_STYLE.get(red),
                    linewidth=1.1, marker="o", markersize=2.2)
            ax.fill_between(cks, [l for l, _ in lohi], [h for _, h in lohi],
                            color=SPACE_COLOUR.get(red), alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="grey", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_xscale("log")
        ax.set_title(ALGO_LABEL.get(al, al))
        ax.set_xlabel("queries")
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("paired advantage (pp)", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figdir / "advantage_vs_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = Path(sys.argv[1])
    figdir = Path(sys.argv[2]) if len(sys.argv) > 2 else outdir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    rows = load(outdir)
    if not rows:
        sys.exit(f"[figures] no unit outputs under {outdir}")
    fig_anytime(rows, figdir)
    fig_advantage(rows, figdir)
    fig_cd(outdir, figdir)
    fig_effect_heatmap(outdir, figdir)
    fig_budget_alloc(rows, figdir)
    print(f"[figures] written to {figdir}")


if __name__ == "__main__":
    main()
