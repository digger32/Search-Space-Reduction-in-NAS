#!/usr/bin/env python3
"""Figures for B6, read from <outdir> after aggregate.py.

  figs/anytime_<dataset>.pdf   anytime test accuracy, mean +- bootstrap 95% CI
                               bands over seeds, one panel per algorithm
  figs/cd_spaces.pdf           critical-difference diagram over spaces
                               (Friedman blocks = dataset x algo)
  figs/effect_heatmap.pdf      median paired delta (reduction - matched random)
                               per dataset x algo, stars for Holm-significant
  figs/budget_alloc.pdf        fraction of the best-trajectory inside each
                               informed region for FULL-space searches

Greyscale-legible, colourblind-safe (Okabe-Ito), vector output.
Usage: python3 make_figures.py <outdir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Print-scale typography: figures are sized at their FINAL printed width
# (IEEE \columnwidth = 3.5 in, \textwidth = 7.16 in), so fonts below render
# at true size in the PDF. Type 42 keeps text editable/searchable.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]
N_BOOT = 2000


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


def fig_anytime(rows, outdir, figdir):
    rng = np.random.default_rng(1)
    datasets = sorted({r[0] for r in rows})
    algos = sorted({r[1] for r in rows})
    spaces = sorted({r[2] for r in rows})
    for ds in datasets:
        fig, axes = plt.subplots(1, len(algos), figsize=(3.2 * len(algos), 2.8),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, al in zip(axes, algos):
            for i, sp in enumerate(spaces):
                curves = defaultdict(list)
                for d, a, s, rec in rows:
                    if d == ds and a == al and s == sp:
                        for ck, t in rec["anytime_test"].items():
                            curves[int(ck)].append(t)
                if not curves:
                    continue
                cks = sorted(curves)
                mean = [np.mean(curves[c]) for c in cks]
                lohi = [boot_ci(np.array(curves[c]), rng) for c in cks]
                ax.plot(cks, mean, label=sp, color=OKABE_ITO[i % 8],
                        linestyle=LINESTYLES[i % 7], linewidth=1.4)
                ax.fill_between(cks, [l for l, _ in lohi], [h for _, h in lohi],
                                color=OKABE_ITO[i % 8], alpha=0.15,
                                linewidth=0)
            ax.set_xscale("log")
            ax.set_title(al)
            ax.set_xlabel("queries")
        axes[0].set_ylabel("best test accuracy (%)")
        axes[-1].legend(fontsize=6, frameon=False)
        fig.suptitle(ds)
        fig.tight_layout()
        fig.savefig(figdir / f"anytime_{ds}.pdf")
        plt.close(fig)

    # combined grid at print width: rows = datasets, cols = algorithms
    fig, axes = plt.subplots(len(datasets), len(algos),
                             figsize=(7.16, 1.5 * len(datasets)),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, ds in enumerate(datasets):
        for c, al in enumerate(algos):
            ax = axes[r, c]
            for i, sp in enumerate(spaces):
                curves = defaultdict(list)
                for d, a, s, rec in rows:
                    if d == ds and a == al and s == sp:
                        for ck, tt in rec["anytime_test"].items():
                            curves[int(ck)].append(tt)
                if not curves:
                    continue
                cks = sorted(curves)
                mean = [np.mean(curves[ck]) for ck in cks]
                lohi = [boot_ci(np.array(curves[ck]), rng) for ck in cks]
                ax.plot(cks, mean, label=sp, color=OKABE_ITO[i % 8],
                        linestyle=LINESTYLES[i % 7], linewidth=1.0)
                ax.fill_between(cks, [l for l, _ in lohi],
                                [h for _, h in lohi],
                                color=OKABE_ITO[i % 8], alpha=0.15,
                                linewidth=0)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(al)
            if r == len(datasets) - 1:
                ax.set_xlabel("queries")
            if c == 0:
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
    ranks = pd.Series(ph["mean_ranks"])
    pmat = pd.DataFrame(ph["p_matrix"], index=ph["treatments"],
                        columns=ph["treatments"])
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
    cells = sorted(paired)
    reds = sorted({c["reduction"] for comps in paired.values() for c in comps})
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
    fig, ax = plt.subplots(figsize=(3.5, 0.55 + 0.19 * len(cells)))
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="PuOr", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(reds)), reds, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(cells)), cells, fontsize=6.5)
    for i in range(len(cells)):
        for j in range(len(reds)):
            if stars[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im)
    cb.set_label("mean Δ test acc (reduction − matched random)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "effect_heatmap.pdf")
    plt.close(fig)


def fig_budget_alloc(rows, outdir, figdir):
    frac = defaultdict(lambda: defaultdict(list))
    for d, a, sp, rec in rows:
        if sp != "full":
            continue
        for token, f in rec.get("membership_frac_of_best_trajectory",
                                {}).items():
            frac[d][token].append(f)
    if not frac:
        return
    datasets = sorted(frac)
    tokens = sorted({t for d in frac.values() for t in d})
    x = np.arange(len(tokens))
    width = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for k, ds in enumerate(datasets):
        vals = [np.mean(frac[ds].get(t, [np.nan])) for t in tokens]
        ax.bar(x + k * width, vals, width, label=ds,
               color=OKABE_ITO[k % 8], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2, tokens, fontsize=7)
    ax.set_ylabel("best-trajectory fraction\ninside region", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, frameon=False, ncol=len(datasets),
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(figdir / "budget_alloc.pdf")
    plt.close(fig)


def main():
    outdir = Path(sys.argv[1])
    # optional 2nd arg: write figures OUTSIDE the frozen run dir
    # (recommended after a gated final: python3 make_figures.py <run> figs_print)
    figdir = Path(sys.argv[2]) if len(sys.argv) > 2 else outdir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    rows = load(outdir)
    if not rows:
        sys.exit(f"[figures] no unit outputs under {outdir}")
    fig_anytime(rows, outdir, figdir)
    fig_cd(outdir, figdir)
    fig_effect_heatmap(outdir, figdir)
    fig_budget_alloc(rows, outdir, figdir)
    print(f"[figures] written to {figdir}")


if __name__ == "__main__":
    main()
