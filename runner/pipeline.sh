#!/usr/bin/env bash
# B6 pipeline — one command per stage; chains run -> aggregate -> figures ->
# gate -> report. Launch each stage once under tmux and check when it ends:
#     tmux new -s b6
#     ./pipeline.sh <stage>        # then Ctrl-b d to detach
#
# Stages:
#   gatetest  unit-test the gate on synthetic clean/dirty runs   (Stage 0 req.)
#   spaces    freeze the informed reductions (per backend)
#   smoke     1 unit, tiny budget, full chain incl. gate          (<2 min)
#   micro     MICRO timing slice: one representative unit per algorithm at the
#             full budget on the real backend; read wall_s from the manifest
#             and extrapolate BEFORE trusting any arithmetic estimate
#   pilot     2 seeds x 2 units per axis value, real backend
#   full      complete grid, resume ENABLED
#   final     fresh outdir, --no-resume, gate blocks dirty numbers
#
# Backend override: BACKEND=mock ./pipeline.sh smoke   (default: nats)
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:?usage: pipeline.sh gatetest|spaces|smoke|micro|pilot|full|final}"
BACKEND="${BACKEND:-nats}"
JOBS="${JOBS:-24}"
SPACES="${SPACES:-}"
EXTRA=(); [ -n "$SPACES" ] && EXTRA=(--spaces "$SPACES")
PY=python3

run_chain () {  # gate_mode(block|info), outdir, then runner args...
  local GATE_MODE="$1" OUT="$2"; shift 2
  $PY bench_runner.py --backend "$BACKEND" --outdir "$OUT" "$@"
  $PY aggregate.py "$OUT"
  $PY make_figures.py "$OUT"
  if [ "$GATE_MODE" = block ]; then
    $PY review_gate.py "$OUT" --config gate_config.yaml || {
        echo "[pipeline] GATE FAILED for $OUT — numbers are NOT frozen"; exit 1; }
  else
    $PY review_gate.py "$OUT" --config gate_config.yaml || \
        echo "[pipeline] gate informational at stage '$STAGE' (expected to fail: not a --no-resume full-grid pass); it BLOCKS at stage 'final'"
  fi
  echo "[pipeline] $STAGE complete -> $OUT (merged.csv, stats/, figs/, report.md)"
}

case "$STAGE" in
  gatetest)
    $PY gate_unittest.py
    ;;
  spaces)
    $PY build_spaces.py --backend "$BACKEND"
    ;;
  smoke)
    # 1 dataset x 1 algo x 2 spaces x 1 block of 2 seeds, budget 30
    run_chain info "runs/smoke_$BACKEND" \
      --datasets cifar10-valid --algos rs --spaces full,randM4096 \
      --seeds 0-1 --block 2 --budget 30 --jobs 2 --timeout-s 300
    ;;
  micro)
    # one representative unit per algorithm, widest space, full budget,
    # 1 block of 2 seeds; extrapolate from manifest wall_s
    run_chain info "runs/micro_$BACKEND" \
      --datasets cifar10-valid --algos rs,re,reinforce,ls --spaces full \
      --seeds 0-1 --block 2 --budget 1000 --jobs 4 --timeout-s 1800
    echo "[pipeline] micro wall_s per unit (extrapolate BEFORE full):"
    $PY -c "
import json, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    r = json.loads(line)
    if r.get('status') == 'ok':
        print(f\"  {r['unit']}: {r['wall_s']}s\")
" "runs/micro_$BACKEND/manifest.jsonl"
    ;;
  pilot)
    run_chain info "runs/pilot_$BACKEND" \
      --datasets cifar10-valid,cifar100 --algos rs,re \
      --spaces full,no_none,randM4096 \
      --seeds 0-3 --block 2 --budget 200 --jobs 8 --timeout-s 900
    ;;
  full)
    run_chain info "runs/full_$BACKEND" \
      --seeds 0-199 --block 25 --budget 1000 --jobs "$JOBS" --timeout-s 1800 "${EXTRA[@]}"
    ;;
  final)
    OUT="runs/final_$(date +%Y%m%d_%H%M)"
    run_chain block "$OUT" \
      --seeds 0-199 --block 25 --budget 1000 --jobs "$JOBS" \
      --timeout-s 1800 --no-resume "${EXTRA[@]}"
    ;;
  *)
    echo "unknown stage: $STAGE"; exit 2;;
esac
