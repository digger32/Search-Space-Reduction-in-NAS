#!/usr/bin/env python3
"""
Aggregate the search-dynamics metrics for B6 (reviewer R1.4) and emit the
regularised-evolution dynamics table (Table V of the paper).

Two modes:
  reference only   python3 aggregate_dynamics.py --ref-dir runs/final_20260718_1728
                   -> columns computable from the gated run itself
  with replay      python3 aggregate_dynamics.py --ref-dir runs/final_20260718_1728 \
                        --replay-dir runs/replay_dynamics
                   -> adds population diversity and exploration breadth; this
                      mode is a GATE: it exits 2 without writing anything if a
                      single expected unit is missing or a single seed failed
                      verification against the reference run.

Outputs (in --out-dir, default dynamics_summary/):
  dynamics_summary.json    all metrics, per algorithm x space, dataset-averaged
  dynamics_report.md       human-readable version, incl. verification counts
  tab_re_dynamics.tex      drop-in LaTeX table for the manuscript
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]
SPACES = ["full", "no_none", "synflow50", "naswot50", "param50",
          "randM4096", "randM7813"]
LABEL = {"full": "FULL", "no_none": "NO-NONE", "synflow50": "SYNFLOW-50",
         "naswot50": "NASWOT-50", "param50": "PARAM-50",
         "randM4096": "RAND-4096", "randM7813": "RAND-7813"}
TEX = {k: r"\textsc{" + v.lower() + "}" for k, v in LABEL.items()}
CONVERGE_AT = "500"


def load_units(d):
    units = {}
    for p in sorted(Path(d).glob("*__*__*__b*.json")):
        u = json.loads(p.read_text())
        units[p.name] = u
    return units


def mean(xs):
    return sum(xs) / len(xs)


def reference_metrics(ref_units, algo):
    """Per space, dataset-averaged: rejected attempts per proposal, share of
    runs that already found their final incumbent by B=500, and the number of
    distinct final architectures over all seeds."""
    by = defaultdict(lambda: defaultdict(list))  # space -> ds -> seed records
    budget = None
    for u in ref_units.values():
        if u["algo"] != algo:
            continue
        budget = u["budget"]
        by[u["space"]][u["dataset"]].extend(u["seeds"])
    out = {}
    for sp in SPACES:
        if sp not in by:
            continue
        rej, conv, distinct = [], [], []
        for ds in DATASETS:
            recs = by[sp].get(ds, [])
            if not recs:
                continue
            rej.append(mean([r["n_proposals_rejected"] for r in recs]) / budget)
            conv.append(mean([1.0 if r["anytime_val"][CONVERGE_AT] ==
                              r["final_val"] else 0.0 for r in recs]))
            distinct.append(len({tuple(r["final_arch"]) for r in recs}))
        out[sp] = {"rejected_per_proposal": mean(rej),
                   f"found_final_by_{CONVERGE_AT}": mean(conv),
                   "distinct_final_archs": mean(distinct),
                   "n_datasets": len(rej)}
    return out


def check_replay(ref_units, rep_units, algos):
    expected = [n for n, u in ref_units.items() if u["algo"] in algos]
    missing = [n for n in expected if n not in rep_units]
    unverified = []
    for n in expected:
        r = rep_units.get(n)
        if r is None:
            continue
        ref_seeds = {s["seed"] for s in ref_units[n]["seeds"]}
        got = {s["seed"] for s in r["seeds"] if s["verified"]}
        if got != ref_seeds:
            bad = sorted(ref_seeds - got)
            mism = [s["mismatch"] for s in r["seeds"] if not s["verified"]][:1]
            unverified.append((n, bad[:5], mism))
    return expected, missing, unverified


def replay_metrics(rep_units, algos):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in rep_units.values():
        if u["algo"] not in algos:
            continue
        for s in u["seeds"]:
            for metric, series in s["dyn"].items():
                for c, v in series.items():
                    acc[(u["algo"], u["space"])][(metric, c)][u["dataset"]].append(v)
    out = defaultdict(dict)
    for (algo, sp), metrics in acc.items():
        for (metric, c), per_ds in metrics.items():
            out[algo].setdefault(sp, {}).setdefault(metric, {})[c] = mean(
                [mean(v) for v in per_ds.values()])
    return out


def tex_table(ref_m, rep_m):
    """Table V of the paper, in the IEEE conference template's table style:
    caption above, table footnote marked with a letter in a final row."""
    with_replay = rep_m is not None and "re" in rep_m
    ncol = 6 if with_replay else 4
    cols = "@{}lccc" + ("cc" if with_replay else "") + "@{}"
    h1 = r"Space & Rej. & Found & Distinct"
    h2 = r" & /prop. & by 500 & final"
    if with_replay:
        h1 += r" & Div. & Unique"
        h2 += r" & @1{,}000 & @1{,}000"
    cap = (r"\caption{Search dynamics of regularised evolution per space, "
           r"averaged over the three datasets (200 seeds each). Rej./prop.: "
           r"rejected out-of-space attempts per accepted proposal. Found by "
           r"500: share of runs whose final incumbent is found by 500 queries. "
           r"Distinct final: distinct final architectures over the 200 runs.")
    if with_replay:
        cap += (r" Div.: mean pairwise Hamming distance (0--6) in the "
                r"population. Unique: distinct architectures evaluated. Both "
                r"come from a replay verified bit for bit against the reported "
                r"runs.")
    cap += "}"
    lines = [r"\begin{table}[!tb]", cap, r"\label{tab:dynamics}", r"\centering",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{" + cols + "}", r"\toprule",
             h1 + r" \\", h2 + r" \\", r"\midrule"]
    for sp in SPACES:
        if sp not in ref_m:
            continue
        m = ref_m[sp]
        mark = r"$^{\mathrm{a}}$" if sp.startswith("randM") else ""
        row = (f"{TEX[sp]} & {m['rejected_per_proposal']:.2f} & "
               f"{m[f'found_final_by_{CONVERGE_AT}']:.2f} & "
               f"{m['distinct_final_archs']:.1f}{mark}")
        if with_replay:
            r = rep_m["re"][sp]
            row += (f" & {r['pop_hamming']['1000']:.2f} & "
                    f"{r['unique']['1000']:.0f}")
        lines.append(row + r" \\")
        if sp == "param50":
            lines.append(r"\midrule")
    lines += [r"\bottomrule",
              r"\multicolumn{" + str(ncol) + r"}{@{}p{0.97\columnwidth}@{}}{"
              r"$^{\mathrm{a}}$Random controls are resampled per seed, so this "
              r"count is not comparable with the fixed spaces.}",
              r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--replay-dir")
    ap.add_argument("--out-dir", default="dynamics_summary")
    a = ap.parse_args()

    ref_units = load_units(a.ref_dir)
    if not ref_units:
        sys.exit(f"[dynamics] no reference units in {a.ref_dir}")
    ref_m = reference_metrics(ref_units, "re")

    rep_m, verification = None, None
    if a.replay_dir:
        rep_units = load_units(a.replay_dir)
        algos = {u["algo"] for u in rep_units.values()} or {"re"}
        expected, missing, unverified = check_replay(ref_units, rep_units, algos)
        verification = {"algos": sorted(algos), "expected_units": len(expected),
                        "missing": len(missing), "unverified": len(unverified)}
        if missing or unverified:
            print(f"[dynamics] GATE FAIL: {len(missing)} missing, "
                  f"{len(unverified)} unit(s) with unverified seeds "
                  f"(of {len(expected)} expected).")
            for n in missing[:10]:
                print(f"  missing    {n}")
            for n, seeds, mism in unverified[:10]:
                print(f"  unverified {n} seeds={seeds} first mismatch={mism}")
            print("[dynamics] nothing written. Rerun replay_dynamics.py "
                  "(resume redoes only incomplete units).")
            sys.exit(2)
        rep_m = replay_metrics(rep_units, algos)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"reference_re": ref_m, "replay": rep_m,
               "verification": verification}
    (out / "dynamics_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "tab_re_dynamics.tex").write_text(tex_table(ref_m, rep_m))

    md = ["# Search dynamics (B6, reviewer R1.4)", ""]
    if verification:
        md += [f"Verification: {verification['expected_units']} units expected, "
               f"all present and every seed bit-identical to the reference "
               f"run (algos: {', '.join(verification['algos'])}).", ""]
    md += ["## Regularised evolution, dataset-averaged", "",
           "| Space | Rej./prop. | Found by 500 | Distinct final |"
           + (" Div@100 | Div@1000 | Unique@1000 |" if rep_m else ""),
           "|---|---|---|---|" + ("---|---|---|" if rep_m else "")]
    for sp in SPACES:
        if sp not in ref_m:
            continue
        m = ref_m[sp]
        row = (f"| {LABEL[sp]} | {m['rejected_per_proposal']:.2f} | "
               f"{m[f'found_final_by_{CONVERGE_AT}']:.2f} | "
               f"{m['distinct_final_archs']:.1f} |")
        if rep_m:
            r = rep_m["re"][sp]
            row += (f" {r['pop_hamming']['100']:.2f} | "
                    f"{r['pop_hamming']['1000']:.2f} | {r['unique']['1000']:.0f} |")
        md.append(row)
    if rep_m:
        for algo in sorted(rep_m):
            md += ["", f"## Exploration breadth, {algo}", "",
                   "| Space | Unique@100 | Unique@1000 | Revisit@1000 |",
                   "|---|---|---|---|"]
            for sp in SPACES:
                if sp in rep_m[algo]:
                    r = rep_m[algo][sp]
                    md.append(f"| {LABEL[sp]} | {r['unique']['100']:.0f} | "
                              f"{r['unique']['1000']:.0f} | "
                              f"{r['revisit']['1000']:.3f} |")
    (out / "dynamics_report.md").write_text("\n".join(md) + "\n")
    print(f"[dynamics] {'GATE PASS, ' if verification else ''}written to {out}")


if __name__ == "__main__":
    main()
