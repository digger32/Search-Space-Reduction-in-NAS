#!/usr/bin/env python3
"""
Deterministic, instrumented replay of the gated final run (B6, reviewer R1.4).

Re-runs units of the reference run through EXACTLY the code path of
bench_runner.run_unit: same global seeding of nats trial sampling, same
per-seed algorithm rng, same membership loading, same algorithm functions,
same backend. The only addition is a read-only observer around the backend
that logs every queried architecture; it draws no random numbers and changes
no return value. algorithms.py, spaces.py and nb201.py are NOT modified.

From the query log it derives, at every checkpoint c:
  unique       distinct architectures evaluated in the first c queries
  revisit      share of the first c queries spent on already-evaluated ones
  pop_hamming  (RE only) mean pairwise Hamming distance (0-6) of the
               population; in this RE implementation the population after c
               queries is exactly the last min(c, 10) queried architectures
  pop_unique   (RE only) distinct architectures in that population

Every seed is VERIFIED against the reference unit file: the number of
queries, final_arch, final_val, final_test, anytime_val at every checkpoint
and n_proposals_rejected must match exactly. A mismatch is recorded, never
silently accepted; aggregate_dynamics.py refuses to report if any seed of
any expected unit is missing or unverified.

Run on Server B, from runner/ (the reference run is runs/final_20260718_1728):

    tmux new -s b6replay
    source ../.venv/bin/activate
    # 1) smoke: 2 seeds of one unit, ~30 s; must print "SMOKE PASS"
    python3 replay_dynamics.py --ref-dir runs/final_20260718_1728 --smoke
    # 2) full replay of regularised evolution: 168 units, ~20 min at --jobs 24
    python3 replay_dynamics.py --ref-dir runs/final_20260718_1728 \
        --outdir runs/replay_dynamics --jobs 24
    # 3) aggregate + verification gate (exit 0 only if everything verified)
    python3 aggregate_dynamics.py --replay-dir runs/replay_dynamics \
        --ref-dir runs/final_20260718_1728

Optional, if time allows: --algos re,rs,reinforce,ls (672 units, ~1.5 h at
--jobs 24) adds exploration breadth for all four searchers.

Resume is ON by default (an existing verified unit file is skipped), so an
interrupted replay continues where it stopped. --no-resume forces a redo.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHECKPOINTS = [10, 25, 50, 100, 200, 500, 1000]
RE_POP_SIZE = 10          # frozen RE hyperparameter (algorithms.py)
UNIT_RE = re.compile(r"^(?P<ds>.+?)__(?P<algo>[a-z]+)__(?P<space>[A-Za-z0-9_]+)"
                     r"__b(?P<bs>\d+)\.json$")


# --------------------------------------------------------------------------- #
# Observer                                                                    #
# --------------------------------------------------------------------------- #
class ObservedBackend:
    """Pass-through wrapper that records every queried architecture.

    query_val forwards the call unchanged (same positional argument, same
    return value), so the global `random` stream consumed by nats trial
    sampling is identical to the original run. Every other attribute is
    delegated to the wrapped backend."""

    def __init__(self, backend):
        self._backend = backend
        self.log = []

    def query_val(self, arch, *args, **kwargs):
        self.log.append(tuple(arch))
        return self._backend.query_val(arch, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def dynamics_from_log(log, algo, checkpoints):
    out = {"unique": {}, "revisit": {}}
    if algo == "re":
        out["pop_hamming"], out["pop_unique"] = {}, {}
    for c in checkpoints:
        prefix = log[:c]
        n_unique = len(set(prefix))
        out["unique"][str(c)] = n_unique
        out["revisit"][str(c)] = 1.0 - n_unique / c
        if algo == "re":
            pop = log[max(0, c - RE_POP_SIZE):c]
            pairs = list(combinations(pop, 2))
            out["pop_hamming"][str(c)] = (sum(hamming(a, b) for a, b in pairs)
                                          / len(pairs)) if pairs else 0.0
            out["pop_unique"][str(c)] = len(set(pop))
    return out


# --------------------------------------------------------------------------- #
# Per-unit work (runs in its own subprocess)                                  #
# --------------------------------------------------------------------------- #
def replay_unit(ref_file, backend_kind, bench_path, spaces_dir, out_path,
                max_seeds=None):
    from nb201 import make_backend, resolve_bench_sha
    from spaces import load_members
    from algorithms import ALGORITHMS

    ref = json.loads(Path(ref_file).read_text())
    dataset, algo, space = ref["dataset"], ref["algo"], ref["space"]
    budget = int(ref["budget"])
    if ref.get("backend") != backend_kind:
        raise SystemExit(f"[replay] backend mismatch: reference ran on "
                         f"'{ref.get('backend')}', replay asked for "
                         f"'{backend_kind}'")
    if backend_kind == "nats":
        sha = resolve_bench_sha(bench_path)
        if sha is None or sha != ref.get("bench_sha256"):
            raise SystemExit(f"[replay] benchmark fingerprint mismatch: "
                             f"reference {ref.get('bench_sha256')} vs "
                             f"{sha} at {bench_path}")

    zc_path = Path(spaces_dir) / f"zc_scores_{dataset}.json"
    backend = make_backend(backend_kind, bench_path, dataset,
                           zc_path=zc_path if zc_path.exists() else None)
    spaces_path = Path(spaces_dir) / f"spaces_{backend_kind}_{dataset}.json"
    fn = ALGORITHMS[algo]
    cks = [c for c in CHECKPOINTS if c <= budget]

    ref_seeds = ref["seeds"][:max_seeds] if max_seeds else ref["seeds"]
    per_seed = []
    for rs in ref_seeds:
        seed = int(rs["seed"])
        # --- identical to bench_runner.run_unit ------------------------------
        random.seed((seed * 2_654_435_761) & 0xFFFFFFFF)  # nats is_random trials
        rng = random.Random(seed)                           # algorithm decisions
        members = load_members(spaces_path, space, seed)
        member_set = set(members)
        obs = ObservedBackend(backend)
        t0 = time.time()
        trace = fn(members, member_set, obs, budget, rng)
        # ---------------------------------------------------------------------
        final_arch = tuple(trace["best_arch"][-1])
        mismatch = []
        if len(obs.log) != budget:
            mismatch.append(f"n_queries {len(obs.log)} != budget {budget}")
        if list(final_arch) != list(rs["final_arch"]):
            mismatch.append("final_arch")
        if float(trace["best_val"][-1]) != float(rs["final_val"]):
            mismatch.append("final_val")
        if float(backend.test_acc(final_arch)) != float(rs["final_test"]):
            mismatch.append("final_test")
        for c in cks:
            if float(trace["best_val"][c - 1]) != float(rs["anytime_val"][str(c)]):
                mismatch.append(f"anytime_val@{c}")
        if int(trace["n_proposals_rejected"]) != int(rs["n_proposals_rejected"]):
            mismatch.append("n_proposals_rejected")
        per_seed.append({
            "seed": seed,
            "verified": not mismatch,
            "mismatch": mismatch,
            "dyn": dynamics_from_log(obs.log, algo, cks),
            "wall_s": round(time.time() - t0, 2),
        })

    result = {
        "dataset": dataset, "algo": algo, "space": space,
        "block_start": ref["block_start"], "block": ref["block"],
        "budget": budget, "backend": backend_kind,
        "ref_file": Path(ref_file).name,
        "n_seeds": len(per_seed),
        "n_verified": sum(1 for s in per_seed if s["verified"]),
        "seeds": per_seed,
    }
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out_path)  # atomic
    return result


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
_lock = threading.Lock()


def ref_units(ref_dir, algos):
    units = []
    for p in sorted(Path(ref_dir).glob("*__*__*__b*.json")):
        m = UNIT_RE.match(p.name)
        if m and m.group("algo") in algos:
            units.append(p)
    return units


def unit_is_complete(out_path):
    """A replay unit counts as done only if its file parses and every seed
    was verified; anything else is redone on resume."""
    try:
        r = json.loads(Path(out_path).read_text())
        return r["n_seeds"] == r["block"] and r["n_verified"] == r["n_seeds"]
    except Exception:
        return False


def launch(ref_path, args, outdir):
    out_path = outdir / ref_path.name
    if not args.no_resume and unit_is_complete(out_path):
        print(f"[skip] {ref_path.stem} (verified output exists)", flush=True)
        return "skip"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--ref-file", str(ref_path), "--out", str(out_path),
           "--backend", args.backend, "--bench-path", args.bench_path,
           "--spaces-dir", args.spaces_dir]
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, timeout=args.timeout_s,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True)
        if cp.returncode != 0:
            status = f"fail(rc={cp.returncode})"
            tail = (cp.stderr or "").strip().splitlines()[-3:]
            print(f"[FAIL] {ref_path.stem}: " + " | ".join(tail), flush=True)
        elif unit_is_complete(out_path):
            status = "ok"
            print(f"[ok] {ref_path.stem} ({time.time()-t0:.0f}s)", flush=True)
        else:
            status = "unverified"
            print(f"[UNVERIFIED] {ref_path.stem} — replay differs from the "
                  f"reference; see 'mismatch' in {out_path.name}", flush=True)
    except subprocess.TimeoutExpired:
        status = "timeout"
        print(f"[TIMEOUT] {ref_path.stem} > {args.timeout_s}s", flush=True)
    with _lock, (outdir / "replay_manifest.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "unit": ref_path.stem, "status": status,
            "wall_s": round(time.time() - t0, 1),
            "finished": datetime.now(timezone.utc).isoformat()}) + "\n")
    return status


def preflight(args):
    ref_dir = Path(args.ref_dir)
    if not ref_dir.is_dir():
        raise SystemExit(f"[replay] reference run not found: {ref_dir}")
    if args.backend == "nats" and not Path(args.bench_path).exists():
        raise SystemExit(f"[replay] benchmark not found: {args.bench_path}")
    for ds in ("cifar10-valid", "cifar100", "ImageNet16-120"):
        sp = Path(args.spaces_dir) / f"spaces_{args.backend}_{ds}.json"
        if not sp.exists():
            print(f"[replay] warning: {sp} missing (only needed if units of "
                  f"{ds} use informed spaces)", flush=True)


def run_smoke(args):
    preflight(args)
    algos = set(args.algos.split(","))
    units = ref_units(args.ref_dir, algos)
    if not units:
        raise SystemExit("[replay] no reference units for the requested algos")
    # an informed space exercises the frozen membership file as well
    pick = next((u for u in units if "__synflow50__" in u.name), units[0])
    outdir = Path(args.outdir) / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / pick.name
    t0 = time.time()
    r = replay_unit(pick, args.backend, args.bench_path, args.spaces_dir, out,
                    max_seeds=2)
    ok = r["n_verified"] == r["n_seeds"] == 2
    for s in r["seeds"]:
        print(f"  seed {s['seed']}: verified={s['verified']} "
              f"mismatch={s['mismatch']} unique@1000="
              f"{s['dyn']['unique'].get('1000')}", flush=True)
    print(f"[replay] SMOKE {'PASS' if ok else 'FAIL'} on {pick.stem} "
          f"({time.time()-t0:.0f}s)", flush=True)
    sys.exit(0 if ok else 1)


def run_orchestrator(args):
    preflight(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    algos = set(args.algos.split(","))
    units = ref_units(args.ref_dir, algos)
    if not units:
        raise SystemExit("[replay] no reference units for the requested algos")
    (outdir / "replay_meta.json").write_text(json.dumps({
        "ref_dir": str(Path(args.ref_dir).resolve()), "algos": sorted(algos),
        "n_units": len(units), "backend": args.backend,
        "started": datetime.now(timezone.utc).isoformat(),
        "jobs": args.jobs, "no_resume": args.no_resume}, indent=2))
    print(f"[replay] {len(units)} units | algos={sorted(algos)} | "
          f"jobs={args.jobs} | outdir={outdir}", flush=True)
    counts = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for st in pool.map(lambda u: launch(u, args, outdir), units):
            counts[st] = counts.get(st, 0) + 1
    print(f"[replay] done | {counts}", flush=True)
    bad = {k: v for k, v in counts.items() if k not in ("ok", "skip")}
    if bad:
        print("[replay] some units failed or did not verify; rerun the same "
              "command to retry only those (resume skips verified units).",
              flush=True)
        sys.exit(1)


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", default="runs/final_20260718_1728")
    ap.add_argument("--outdir", default="runs/replay_dynamics")
    ap.add_argument("--algos", default="re")
    ap.add_argument("--backend", default="nats", choices=["nats", "mock"])
    ap.add_argument("--bench-path", dest="bench_path",
                    default=str(HERE.parent / "data" / "NATS-tss-v1_0-3ffb9-simple"))
    ap.add_argument("--spaces-dir", dest="spaces_dir",
                    default=str(HERE.parent / "data"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=1800)
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # worker-only
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--ref-file"); ap.add_argument("--out")
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if a.worker:
        replay_unit(a.ref_file, a.backend, a.bench_path, a.spaces_dir, a.out)
    elif a.smoke:
        run_smoke(a)
    else:
        run_orchestrator(a)
