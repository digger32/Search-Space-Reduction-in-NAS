#!/usr/bin/env python3
"""Freeze the informed search-space reductions per dataset (run before any
search; pipeline stage 'spaces'). Writes data/spaces_<backend>_<dataset>.json
and appends the hashes to data/spaces_hashes_frozen.json, which the gate
asserts against so no space silently changes between runs.

Usage: python3 build_spaces.py --backend nats|mock [--datasets ...]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nb201 import make_backend, DATASETS  # noqa: E402
from spaces import build_informed_spaces  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nats", choices=["nats", "mock"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--bench-path",
                    default=str(HERE.parent / "data" / "NATS-tss-v1_0-3ffb9-simple"))
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = data_dir / "spaces_hashes_frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}

    for ds in a.datasets.split(","):
        zc = data_dir / f"zc_scores_{ds}.json"
        backend = make_backend(a.backend, a.bench_path, ds,
                               zc_path=zc if zc.exists() else None)
        out = data_dir / f"spaces_{a.backend}_{ds}.json"
        hashes = build_informed_spaces(backend, ds, out)
        key = f"{a.backend}/{ds}"
        if key in frozen and frozen[key] != hashes:
            sys.exit(f"[spaces] HASH MISMATCH for {key}: frozen "
                     f"{frozen[key]} vs rebuilt {hashes}. Investigate before "
                     "overwriting — a reduction changed under your feet.")
        frozen[key] = hashes
        print(f"[spaces] {key}: " +
              ", ".join(f"{t}={h[:10]}" for t, h in hashes.items()))
    frozen_path.write_text(json.dumps(frozen, indent=2))
    print(f"[spaces] frozen hashes -> {frozen_path}")


if __name__ == "__main__":
    main()
