#!/usr/bin/env python3
"""Gate unit test (Stage 0 requirement): build a synthetic CLEAN run and a
synthetic DIRTY run, prove review_gate.py exits 0 and 1 respectively.

Usage: python3 gate_unittest.py    (exit 0 iff both behave correctly)
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def synth_unit(dataset, algo, space, bs, backend="nats", sha="deadbeef" * 8):
    return {"dataset": dataset, "algo": algo, "space": space,
            "block_start": bs, "block": 2, "budget": 10, "backend": backend,
            "bench_sha256": sha if backend == "nats" else None,
            "seeds": [{"seed": bs + i, "final_val": 90.0, "final_test": 89.0,
                       "final_arch": [1] * 6, "anytime_val": {"10": 90.0},
                       "anytime_test": {"10": 89.0},
                       "membership_frac_of_best_trajectory": {},
                       "n_proposals_rejected": 0, "wall_s": 0.1}
                      for i in range(2)]}


def build_run(root, clean=True):
    outdir = root / ("clean" if clean else "dirty")
    outdir.mkdir(parents=True)
    started = "2026-07-17T00:00:00+00:00"
    meta = {"run_started": started, "no_resume": clean, "backend": "nats",
            "budget": 10, "block": 2,
            "axes": {"datasets": ["cifar10-valid", "cifar100",
                                  "ImageNet16-120"],
                     "algos": ["rs"], "spaces": ["full"], "seeds": "0-1"},
            "timeout_s": 60, "jobs": 1}
    (outdir / "run_meta.json").write_text(json.dumps(meta))
    manifest = []
    for ds in ["cifar10-valid", "cifar100", "ImageNet16-120"]:
        uid = f"{ds}__rs__full__b0"
        u = synth_unit(ds, "rs", "full", 0)
        (outdir / f"{uid}.json").write_text(json.dumps(u))
        manifest.append({"unit": uid, "dataset": ds, "algo": "rs",
                         "space": "full", "block_start": 0, "status": "ok",
                         "started": started, "finished": started,
                         "wall_s": 0.1, "no_resume": clean})
    if not clean:
        # dirty: resume enabled (meta), a skipped unit, and missing stats
        manifest.append({"unit": "cifar100__rs__full__b2", "dataset": "cifar100",
                         "algo": "rs", "space": "full", "block_start": 2,
                         "status": "skip", "started": started,
                         "no_resume": False})
    (outdir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    if clean:
        sdir = outdir / "stats"
        sdir.mkdir()
        (sdir / "omnibus.json").write_text(json.dumps(
            {"test": "friedman", "statistic": 1.0, "p_value": 0.5}))
        (sdir / "posthoc.json").write_text(json.dumps(
            {"test": "nemenyi_friedman"}))
    return outdir


def build_spaces_fixture(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"full_placeholder": "0" * 64}
    frozen = {}
    for ds in ["cifar10-valid", "cifar100", "ImageNet16-120"]:
        payload = {"dataset": ds, "n_archs": 15625,
                   "spaces": {"no_none": [[1] * 6]},
                   "hashes": {"no_none": "1" * 64}}
        (data_dir / f"spaces_nats_{ds}.json").write_text(json.dumps(payload))
        frozen[f"nats/{ds}"] = {"no_none": "1" * 64}
    (data_dir / "spaces_hashes_frozen.json").write_text(json.dumps(frozen))
    return hashes


def run_gate(outdir, cfg_path):
    p = subprocess.run([sys.executable, str(HERE / "review_gate.py"),
                        str(outdir), "--config", str(cfg_path)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    root = Path(tempfile.mkdtemp(prefix="b6gate_"))
    try:
        data_dir = root / "data"
        build_spaces_fixture(data_dir)
        cfg = {"stats_artifacts": ["stats/omnibus.json", "stats/posthoc.json"],
               "data_dir": str(data_dir), "bench_sha256": "",
               "comparative_claims": [
                   {"id": "reduction_no_significant_gain_over_matched_random",
                    "independent_datasets": ["cifar100", "ImageNet16-120"]}]}
        cfg_path = root / "gate_config.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg))

        clean = build_run(root, clean=True)
        rc_clean, out_clean = run_gate(clean, cfg_path)
        dirty = build_run(root, clean=False)
        rc_dirty, out_dirty = run_gate(dirty, cfg_path)

        ok = (rc_clean == 0) and (rc_dirty == 1)
        print(f"[gatetest] clean run exit={rc_clean} (want 0) | "
              f"dirty run exit={rc_dirty} (want 1)")
        if not ok:
            print("---- clean output ----\n" + out_clean)
            print("---- dirty output ----\n" + out_dirty)
            sys.exit(1)
        print("[gatetest] PASS — gate distinguishes clean from dirty.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
