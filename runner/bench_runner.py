#!/usr/bin/env python3
"""
B6 job-based runner — Search-Space Reduction in NAS: Help or Reshuffle?

Unit = dataset x algorithm x space x seed-block. Each unit runs in its OWN
subprocess (resume by output file, per-unit hard timeout, atomic write,
manifest record). A seed-block of --block seeds keeps the file count sane
(16,800 searches -> ~672 unit files) while units stay cheap to redo.

Launch (house convention — tmux built in):

    tmux new -s b6
    python3 bench_runner.py --backend nats --outdir runs/full --jobs 24
    # detach: Ctrl-b d ; reattach: tmux attach -t b6

FINAL pass: fresh outdir + --no-resume (gate A1); single box, no sharding.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHECKPOINTS = [10, 25, 50, 100, 200, 500, 1000]


# --------------------------------------------------------------------------- #
# PER-UNIT WORK                                                               #
# --------------------------------------------------------------------------- #
def run_unit(dataset, algo, space, block_start, block, budget, backend_kind,
             bench_path, spaces_dir, out_path):
    from nb201 import make_backend, resolve_bench_sha
    from spaces import load_members, INFORMED
    from algorithms import ALGORITHMS

    zc_path = Path(spaces_dir) / f"zc_scores_{dataset}.json"
    backend = make_backend(backend_kind, bench_path, dataset,
                           zc_path=zc_path if zc_path.exists() else None)
    spaces_path = Path(spaces_dir) / f"spaces_{backend_kind}_{dataset}.json"

    # membership sets of the informed reductions, for budget-allocation stats
    informed_sets = {}
    if spaces_path.exists():
        for token in INFORMED:
            try:
                informed_sets[token] = set(load_members(spaces_path, token, 0))
            except RuntimeError:
                pass

    fn = ALGORITHMS[algo]
    per_seed = []
    for seed in range(block_start, block_start + block):
        random.seed((seed * 2_654_435_761) & 0xFFFFFFFF)  # nats is_random trials
        rng = random.Random(seed)                           # algorithm decisions
        members = load_members(spaces_path, space, seed)
        member_set = set(members)
        t0 = time.time()
        trace = fn(members, member_set, backend, budget, rng)

        cks = [c for c in CHECKPOINTS if c <= budget]
        anytime_val, anytime_test = {}, {}
        for c in cks:
            anytime_val[c] = trace["best_val"][c - 1]
            anytime_test[c] = backend.test_acc(tuple(trace["best_arch"][c - 1]))
        final_arch = tuple(trace["best_arch"][-1])
        visited = {tuple(a) for a in trace["best_arch"]}
        membership_frac = {t: (sum(1 for a in visited if a in s) / len(visited))
                           for t, s in informed_sets.items()}
        per_seed.append({
            "seed": seed,
            "final_val": trace["best_val"][-1],
            "final_test": backend.test_acc(final_arch),
            "final_arch": list(final_arch),
            "anytime_val": anytime_val,
            "anytime_test": anytime_test,
            "membership_frac_of_best_trajectory": membership_frac,
            "n_proposals_rejected": trace["n_proposals_rejected"],
            "wall_s": round(time.time() - t0, 2),
        })

    result = {
        "dataset": dataset, "algo": algo, "space": space,
        "block_start": block_start, "block": block, "budget": budget,
        "backend": backend_kind,
        "algo_hparams_frozen": True,
        "bench_sha256": (resolve_bench_sha(bench_path)
                         if backend_kind == "nats" else None),
        "seeds": per_seed,
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out_path)  # atomic


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def unit_id(d, a, sp, bs):
    return f"{d}__{a}__{sp}__b{bs}"


def unit_out_path(outdir, d, a, sp, bs):
    return outdir / f"{unit_id(d, a, sp, bs)}.json"


_manifest_lock = threading.Lock()


def append_manifest(outdir, record):
    with _manifest_lock, (outdir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_worker(args):
    outdir = Path(args.outdir)
    out_path = unit_out_path(outdir, args.dataset, args.algo, args.space,
                             args.block_start)
    run_unit(args.dataset, args.algo, args.space, args.block_start, args.block,
             args.budget, args.backend, args.bench_path, args.spaces_dir,
             out_path)


def launch_one(unit, args, outdir, run_started):
    d, a, sp, bs = unit
    uid = unit_id(d, a, sp, bs)
    out_path = unit_out_path(outdir, d, a, sp, bs)

    if out_path.exists() and not args.no_resume:
        append_manifest(outdir, {"unit": uid, "dataset": d, "algo": a,
                                 "space": sp, "block_start": bs,
                                 "status": "skip", "started": run_started,
                                 "no_resume": args.no_resume})
        print(f"[skip] {uid} (output exists)", flush=True)
        return "skip"
    if out_path.exists() and args.no_resume:
        out_path.unlink()

    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--dataset", d, "--algo", a, "--space", sp,
           "--block-start", str(bs), "--block", str(args.block),
           "--budget", str(args.budget), "--backend", args.backend,
           "--bench-path", args.bench_path, "--spaces-dir", args.spaces_dir,
           "--outdir", str(outdir)]
    t0 = time.time()
    status = "ok"
    try:
        subprocess.run(cmd, timeout=args.timeout_s, check=True,
                       stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        status = "timeout"
        print(f"[TIMEOUT] {uid} > {args.timeout_s}s — unit killed, batch continues",
              flush=True)
    except subprocess.CalledProcessError as e:
        status = f"fail(rc={e.returncode})"
        print(f"[FAIL] {uid} rc={e.returncode} — batch continues", flush=True)
    else:
        print(f"[ok] {uid} ({time.time()-t0:.1f}s)", flush=True)
    append_manifest(outdir, {"unit": uid, "dataset": d, "algo": a, "space": sp,
                             "block_start": bs, "status": status,
                             "started": run_started,
                             "finished": datetime.now(timezone.utc).isoformat(),
                             "wall_s": round(time.time() - t0, 1),
                             "no_resume": args.no_resume})
    return status


def run_orchestrator(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets.split(",")
    algos = args.algos.split(",")
    spaces = args.spaces.split(",")
    lo, hi = (int(x) for x in args.seeds.split("-"))
    block_starts = list(range(lo, hi + 1, args.block))

    run_started = datetime.now(timezone.utc).isoformat()
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": run_started, "no_resume": args.no_resume,
        "backend": args.backend, "budget": args.budget, "block": args.block,
        "axes": {"datasets": datasets, "algos": algos, "spaces": spaces,
                 "seeds": f"{lo}-{hi}"},
        "timeout_s": args.timeout_s, "jobs": args.jobs,
    }, indent=2))

    units = [(d, a, sp, bs) for d in datasets for a in algos
             for sp in spaces for bs in block_starts]
    print(f"[runner] {len(units)} units | outdir={outdir} | jobs={args.jobs} | "
          f"no_resume={args.no_resume} | timeout={args.timeout_s}s", flush=True)

    counts = {"ok": 0, "skip": 0, "other": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for status in pool.map(
                lambda u: launch_one(u, args, outdir, run_started), units):
            counts["ok" if status == "ok" else
                   "skip" if status == "skip" else "other"] += 1
    print(f"[runner] done | ok={counts['ok']} skip={counts['skip']} "
          f"fail/timeout={counts['other']}", flush=True)
    if counts["other"]:
        print("[runner] some units did not complete — inspect manifest before "
              "freezing.", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--datasets", default="cifar10-valid,cifar100,ImageNet16-120")
    ap.add_argument("--algos", default="rs,re,reinforce,ls")
    ap.add_argument("--spaces",
                    default="full,no_none,synflow50,naswot50,param50,"
                            "randM4096,randM7813")
    ap.add_argument("--seeds", default="0-199", help="inclusive range LO-HI")
    ap.add_argument("--block", type=int, default=25, help="seeds per unit")
    ap.add_argument("--budget", type=int, default=1000, help="queries per search")
    ap.add_argument("--backend", default="nats", choices=["nats", "mock"])
    ap.add_argument("--bench-path", dest="bench_path",
                    default=str(HERE.parent / "data" / "NATS-tss-v1_0-3ffb9-simple"))
    ap.add_argument("--spaces-dir", dest="spaces_dir",
                    default=str(HERE.parent / "data"))
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel unit subprocesses (CPU-bound; <= cores)")
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=1800)
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    # worker-only
    ap.add_argument("--dataset"); ap.add_argument("--algo")
    ap.add_argument("--space"); ap.add_argument("--block-start",
                                                dest="block_start", type=int)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if a.worker:
        run_worker(a)
    else:
        run_orchestrator(a)
