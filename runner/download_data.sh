#!/usr/bin/env bash
# B6 data acquisition — run ON SERVER B inside tmux:
#     tmux new -s b6data
#     cd ~/Documents/b6-build && source .venv/bin/activate && cd runner
#     bash download_data.sh          # Ctrl-b d to detach
#
# Sources are the OFFICIAL ones, verified 2026-07-17:
#   - NATS-tss simple tar: Drive id from the NATS-Bench README table
#     (github.com/D-X-Y/NATS-Bench)
#   - zc_nasbench201.json: gdown id from NASLib's zerocost-branch
#     scripts/bash_scripts/download_nbs_zero.sh (github.com/automl/NASLib)
set -euo pipefail
cd "$(dirname "$0")/../data"
pip show gdown >/dev/null 2>&1 || pip install gdown

# Item 1 — NATS-Bench topology-space benchmark (REQUIRED)
if [ ! -d NATS-tss-v1_0-3ffb9-simple ]; then
  [ -f NATS-tss-v1_0-3ffb9-simple.tar ] || \
    gdown 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU -O NATS-tss-v1_0-3ffb9-simple.tar
  tar xf NATS-tss-v1_0-3ffb9-simple.tar
fi
echo "[data] benchmark dir present"
[ -f NATS-tss-v1_0-3ffb9-simple.tar ] && \
  sha256sum NATS-tss-v1_0-3ffb9-simple.tar | tee BENCH_SHA256.txt
echo "[data] ^ pin this hash into runner/gate_config.yaml -> bench_sha256"

# Item 2 — NAS-Bench-Suite-Zero precomputed ZC scores (enables synflow50/naswot50)
[ -f zc_nasbench201.json ] || gdown 1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H
if [ -f zc_nasbench201.json ]; then
  echo "[data] converting suite-zero scores to per-dataset files"
  ( cd ../runner && python3 convert_zc.py ../data/zc_nasbench201.json \
      --bench-path ../data/NATS-tss-v1_0-3ffb9-simple --out-dir ../data )
else
  echo "[data] zc_nasbench201.json MISSING — synflow50/naswot50 unavailable; "
  echo "       grid degrades to no_none/param50/random controls (see EXP_PLAN)"
fi
echo "[data] done"
