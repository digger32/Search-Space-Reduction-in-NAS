#!/usr/bin/env python3
"""
B6 review-proofing gate. Exits non-zero on any failure so it can block the
figure/freeze step: `python3 review_gate.py runs/final && python3 make_figures.py runs/final`

Assertions:
  A1  clean final run   run_meta has no_resume=true; manifest has no skips;
                        every record carries this run's run_started
  B1  external validity every declared comparative claim has >=1 run on each
                        of its independent datasets (all datasets present)
  E1  stats present     stats/omnibus.json + stats/posthoc.json exist
  S1  frozen spaces     spaces files exist and hashes match
                        data/spaces_hashes_frozen.json
  M1  single benchmark  every nats unit records the same bench_sha256
                        (no mixed benchmark files inside one run)

Usage: python3 review_gate.py <outdir> [--config gate_config.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[gate] PyYAML required: pip install pyyaml --break-system-packages")
    sys.exit(2)

HERE = Path(__file__).resolve().parent


def load_manifest(outdir):
    mf = outdir / "manifest.jsonl"
    if not mf.exists():
        return []
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]


def load_units(outdir):
    units = []
    for p in outdir.glob("*__*__*__b*.json"):
        try:
            units.append(json.loads(p.read_text()))
        except Exception:
            pass
    return units


def check_A1(outdir, manifest):
    meta_path = outdir / "run_meta.json"
    if not meta_path.exists():
        return False, "run_meta.json missing — cannot verify the final pass"
    meta = json.loads(meta_path.read_text())
    if not meta.get("no_resume", False):
        return False, "final pass ran WITHOUT --no-resume (resume was enabled)"
    if any(r.get("status") == "skip" for r in manifest):
        return False, "manifest shows skipped units in a no-resume pass"
    run_started = meta.get("run_started")
    stale = [r["unit"] for r in manifest if r.get("started") != run_started]
    if stale:
        return False, (f"{len(stale)} unit(s) carry a different run_started "
                       f"(carry-over): {stale[:3]}...")
    bad = [r["unit"] for r in manifest if r.get("status") != "ok"]
    if bad:
        return False, f"{len(bad)} unit(s) not ok in final pass: {bad[:3]}..."
    return True, "final pass clean: --no-resume, no skips, single run_started"


def check_B1(units, cfg):
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared in config — declare them"
    present = {u.get("dataset") for u in units}
    failures = []
    for c in claims:
        needed = set(c.get("independent_datasets", []))
        waived = c.get("waive", False)
        if waived:
            if not c.get("waiver_justification"):
                failures.append(f"claim '{c.get('id')}' waived without "
                                "waiver_justification")
            continue
        if not needed:
            failures.append(f"claim '{c.get('id')}' lists no independent_datasets")
        elif not needed <= present:
            failures.append(f"claim '{c.get('id')}' missing runs on "
                            f"{sorted(needed - present)}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(claims)} comparative claim(s) covered on their " \
                 "independent datasets"


def check_E1(outdir, cfg):
    art = cfg.get("stats_artifacts",
                  ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (outdir / a).exists()]
    if missing:
        return False, f"missing stats artifacts: {missing}"
    om = json.loads((outdir / "stats" / "omnibus.json").read_text())
    if "error" in om:
        return False, f"omnibus.json carries an error: {om['error']}"
    return True, f"stats artifacts present: {art}"


def check_S1(cfg):
    data_dir = Path(cfg.get("data_dir", HERE.parent / "data"))
    frozen_path = data_dir / "spaces_hashes_frozen.json"
    if not frozen_path.exists():
        return False, "spaces_hashes_frozen.json missing — run build_spaces first"
    frozen = json.loads(frozen_path.read_text())
    problems = []
    for key, hashes in frozen.items():
        backend, ds = key.split("/", 1)
        sp = data_dir / f"spaces_{backend}_{ds}.json"
        if not sp.exists():
            problems.append(f"{sp.name} missing")
            continue
        payload = json.loads(sp.read_text())
        if payload.get("hashes") != hashes:
            problems.append(f"{sp.name} hash drift")
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(frozen)} frozen space file(s) verified by hash"


def check_M1(units, cfg):
    if not units:
        return False, "no unit outputs found"
    backends = {u.get("backend") for u in units}
    if backends == {"mock"}:
        return True, "mock backend (bench hash not applicable)"
    shas = {u.get("bench_sha256") for u in units if u.get("backend") == "nats"}
    shas.discard(None)
    if len(shas) == 0:
        return False, "nats units recorded no bench_sha256"
    if len(shas) > 1:
        return False, f"MIXED benchmark files inside one run: {len(shas)} hashes"
    exp = cfg.get("bench_sha256")
    if exp and exp not in shas:
        return False, "bench_sha256 differs from the value pinned in gate config"
    return True, f"single benchmark file across all units ({list(shas)[0][:12]}…)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--config", default=str(HERE / "gate_config.yaml"))
    a = ap.parse_args()

    outdir = Path(a.outdir)
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        cfg_path = outdir / a.config
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    manifest = load_manifest(outdir)
    units = load_units(outdir)

    results = [
        ("A1 clean-final-run", *check_A1(outdir, manifest)),
        ("B1 external-validity", *check_B1(units, cfg)),
        ("E1 stats", *check_E1(outdir, cfg)),
        ("S1 frozen-spaces", *check_S1(cfg)),
        ("M1 single-benchmark", *check_M1(units, cfg)),
    ]

    print("=" * 64)
    print(f"REVIEW-PROOFING GATE  | outdir={outdir}")
    print("=" * 64)
    ok = True
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:22s} {msg}")
        ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
