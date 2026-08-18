#!/usr/bin/env python3
"""Merge per-unit outputs, compute the statistics, emit the artifacts the gate
and the figures read.

Outputs under <outdir>:
  merged.csv            long table: dataset, algo, space, seed, checkpoint,
                        val, test (checkpoint 0 = final at full budget)
  stats/omnibus.json    Friedman over spaces; blocks = dataset x algo
  stats/posthoc.json    Nemenyi p-value matrix + mean ranks (CD diagram data)
  stats/paired.json     per (dataset, algo): each informed reduction vs its
                        size-matched random control and vs full —
                        Wilcoxon signed-rank paired by seed, Holm-corrected,
                        median paired delta with bootstrap 95% CI
  report.md             human-readable summary

Pairing map (size-matched controls):
  no_none   <-> randM4096
  synflow50 / naswot50 / param50 <-> randM7813

Usage: python3 aggregate.py <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

CONTROL = {"no_none": "randM4096", "synflow50": "randM7813",
           "naswot50": "randM7813", "param50": "randM7813"}
N_BOOT = 10_000


def load_units(outdir):
    rows = []
    for p in sorted(Path(outdir).glob("*__*__*__b*.json")):
        u = json.loads(p.read_text())
        for s in u["seeds"]:
            rows.append({"dataset": u["dataset"], "algo": u["algo"],
                         "space": u["space"], "seed": s["seed"],
                         "final_val": s["final_val"],
                         "final_test": s["final_test"],
                         "anytime_test": {int(k): v for k, v in
                                          s["anytime_test"].items()},
                         "membership": s.get(
                             "membership_frac_of_best_trajectory", {})})
    return rows


def write_merged_csv(rows, outdir):
    lines = ["dataset,algo,space,seed,checkpoint,val,test"]
    for r in rows:
        lines.append(f"{r['dataset']},{r['algo']},{r['space']},{r['seed']},"
                     f"final,{r['final_val']:.4f},{r['final_test']:.4f}")
        for ck, t in sorted(r["anytime_test"].items()):
            lines.append(f"{r['dataset']},{r['algo']},{r['space']},{r['seed']},"
                         f"{ck},,{t:.4f}")
    (Path(outdir) / "merged.csv").write_text("\n".join(lines) + "\n")


def series(rows, dataset, algo, space):
    xs = {r["seed"]: r["final_test"] for r in rows
          if r["dataset"] == dataset and r["algo"] == algo
          and r["space"] == space}
    return xs


def paired_effects(a, b, rng):
    """Effect summaries robust to heavy ties (both conditions often select an
    identical best architecture, so most paired deltas are exactly zero):
    mean delta with bootstrap CI, Hodges-Lehmann pseudo-median, sign counts."""
    d = a - b
    boots = np.array([np.mean(rng.choice(d, size=len(d), replace=True))
                      for _ in range(N_BOOT)])
    nz = d[d != 0]
    if len(nz):
        walsh = (nz[:, None] + nz[None, :]) / 2.0
        hl = float(np.median(walsh[np.triu_indices(len(nz))]))
    else:
        hl = 0.0
    return {"mean_delta": float(np.mean(d)),
            "mean_ci95": [float(np.quantile(boots, 0.025)),
                          float(np.quantile(boots, 0.975))],
            "median_delta": float(np.median(d)),
            "hl_delta": hl,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            "n_zero": int((d == 0).sum())}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(rows, outdir):
    datasets = sorted({r["dataset"] for r in rows})
    algos = sorted({r["algo"] for r in rows})
    rng = np.random.default_rng(0xB6)
    out = {}
    for ds in datasets:
        for al in algos:
            comps, pvals = [], []
            for red, ctrl in CONTROL.items():
                for ref_name, ref_space in (("matched_random", ctrl),
                                            ("full", "full")):
                    xs, ys = series(rows, ds, al, red), \
                        series(rows, ds, al, ref_space)
                    common = sorted(set(xs) & set(ys))
                    if len(common) < 10:
                        continue
                    a = np.array([xs[s] for s in common])
                    b = np.array([ys[s] for s in common])
                    if np.allclose(a, b):
                        stat, p = 0.0, 1.0
                    else:
                        # pratt: zero differences kept in the ranking — the
                        # conservative choice for tie-heavy paired data
                        stat, p = sps.wilcoxon(a, b, zero_method="pratt")
                    eff = paired_effects(a, b, rng)
                    direction = ("reduction_better" if eff["n_pos"] > eff["n_neg"]
                                 else "reduction_worse"
                                 if eff["n_neg"] > eff["n_pos"] else "tied")
                    comps.append({"reduction": red, "reference": ref_name,
                                  "n_seeds": len(common),
                                  "wilcoxon_stat": float(stat),
                                  "p_raw": float(p),
                                  "direction": direction, **eff})
                    pvals.append(p)
            if comps:
                for c, padj in zip(comps, holm(np.array(pvals))):
                    c["p_holm"] = float(padj)
                    c["significant_0.05"] = bool(padj < 0.05)
            out[f"{ds}|{al}"] = comps
    sdir = Path(outdir) / "stats"
    sdir.mkdir(exist_ok=True)
    (sdir / "paired.json").write_text(json.dumps(out, indent=2))
    return out


def friedman_nemenyi(rows, outdir):
    """Treatments = spaces; blocks = dataset x algo cells; value = mean
    final test accuracy over seeds within the cell."""
    spaces = sorted({r["space"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    algos = sorted({r["algo"] for r in rows})
    table, blocks = [], []
    for ds in datasets:
        for al in algos:
            vals = []
            for sp in spaces:
                xs = series(rows, ds, al, sp)
                if not xs:
                    vals = None
                    break
                vals.append(float(np.mean(list(xs.values()))))
            if vals is not None:
                table.append(vals)
                blocks.append(f"{ds}|{al}")
    table = np.array(table)
    sdir = Path(outdir) / "stats"
    sdir.mkdir(exist_ok=True)
    if table.shape[0] < 3 or table.shape[1] < 3:
        (sdir / "omnibus.json").write_text(json.dumps(
            {"error": "insufficient blocks/treatments for Friedman",
             "shape": list(table.shape)}, indent=2))
        (sdir / "posthoc.json").write_text(json.dumps({"error": "skipped"}))
        return
    stat, p = sps.friedmanchisquare(*[table[:, j]
                                      for j in range(table.shape[1])])
    ranks = table.shape[1] + 1 - sps.rankdata(table, axis=1)  # rank 1 = best
    mean_ranks = ranks.mean(axis=0)
    (sdir / "omnibus.json").write_text(json.dumps({
        "test": "friedman", "treatments": spaces, "blocks": blocks,
        "statistic": float(stat), "p_value": float(p),
        "mean_ranks": {sp: float(r) for sp, r in zip(spaces, mean_ranks)},
    }, indent=2))

    import scikit_posthocs as sp_ph
    ph = sp_ph.posthoc_nemenyi_friedman(table)
    ph.index = spaces
    ph.columns = spaces
    (sdir / "posthoc.json").write_text(json.dumps({
        "test": "nemenyi_friedman", "treatments": spaces,
        "p_matrix": ph.round(6).values.tolist(),
        "mean_ranks": {sp: float(r) for sp, r in zip(spaces, mean_ranks)},
        "n_blocks": int(table.shape[0]),
    }, indent=2))


def write_report(rows, paired, outdir):
    datasets = sorted({r["dataset"] for r in rows})
    algos = sorted({r["algo"] for r in rows})
    spaces = sorted({r["space"] for r in rows})
    lines = ["# B6 aggregate report", "",
             f"units loaded: {len(rows)} seed-records | datasets: "
             f"{datasets} | algos: {algos} | spaces: {spaces}", "",
             "## Mean final test accuracy (over seeds)", ""]
    header = "| dataset | algo | " + " | ".join(spaces) + " |"
    lines += [header, "|" + "---|" * (len(spaces) + 2)]
    for ds in datasets:
        for al in algos:
            cells = []
            for sp in spaces:
                xs = series(rows, ds, al, sp)
                cells.append(f"{np.mean(list(xs.values())):.2f}" if xs else "—")
            lines.append(f"| {ds} | {al} | " + " | ".join(cells) + " |")
    lines += ["", "## Reductions vs size-matched random "
                  "(Wilcoxon, Holm-corrected)", ""]
    for cell, comps in paired.items():
        for c in comps:
            if c["reference"] != "matched_random":
                continue
            sig = "SIG" if c["significant_0.05"] else "n.s."
            lines.append(f"- {cell} | {c['reduction']}: Δmean="
                         f"{c['mean_delta']:+.3f} "
                         f"CI[{c['mean_ci95'][0]:+.3f},"
                         f"{c['mean_ci95'][1]:+.3f}] HL={c['hl_delta']:+.3f} "
                         f"+/0/-={c['n_pos']}/{c['n_zero']}/{c['n_neg']} "
                         f"p_holm={c['p_holm']:.4f} [{sig} {c['direction']}]")
    (Path(outdir) / "report.md").write_text("\n".join(lines) + "\n")


def main():
    outdir = sys.argv[1]
    rows = load_units(outdir)
    if not rows:
        sys.exit(f"[aggregate] no unit outputs under {outdir}")
    write_merged_csv(rows, outdir)
    paired = paired_tests(rows, outdir)
    friedman_nemenyi(rows, outdir)
    write_report(rows, paired, outdir)
    print(f"[aggregate] {len(rows)} seed-records -> merged.csv, stats/, report.md")


if __name__ == "__main__":
    main()
