#!/usr/bin/env bash
# B6 camera-ready, reviewer R1.4: deterministic replay of the gated final run
# with search-dynamics logging. Three stages, stops at the first failure:
#   1) preflight + smoke (2 seeds of one unit, must print "SMOKE PASS")
#   2) replay of all units of the chosen algorithms (resume is on)
#   3) gate: aggregate_dynamics.py fails unless every seed matches the reference
#
# v2: runs WITHOUT tmux (nohup), writes a full log, activates the venv itself,
# takes the parallelism of the reference run (16 on Server B) instead of a
# fixed 24, works on old bash (no empty-array expansion under `set -u`), and
# refuses to be sourced.
#
# Recommended launch (no tmux needed; survives a dropped SSH session):
#   cd ~/Documents/b6-build/runner
#   nohup bash run_replay.sh > replay_console.log 2>&1 &
#   tail -f replay_console.log        # Ctrl-C closes the view, the run goes on

if [ -z "${BASH_VERSION:-}" ]; then
  echo "run_replay.sh: run it with bash (bash run_replay.sh), not sh" >&2
  exit 2
fi
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run_replay.sh: do not source this script; run: bash run_replay.sh" >&2
  return 2
fi
set -eo pipefail
cd "$(dirname "$0")"

REF=${REF:-runs/final_20260718_1728}
OUT=${OUT:-runs/replay_dynamics}
ALGOS=${ALGOS:-re}
BACKEND=${BACKEND:-nats}
VENV=${VENV:-../.venv}
SPACES_DIR=${SPACES_DIR:-}

# venv: activate it here if the caller did not
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
PY=${PY:-python3}

mkdir -p "$OUT"
LOG="$OUT/run_replay.log"
exec > >(tee -a "$LOG") 2>&1

# parallelism: same as the reference run unless JOBS is given
if [[ -z "${JOBS:-}" ]]; then
  JOBS=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/run_meta.json')).get('jobs', 8))" "$REF" 2>/dev/null || echo 8)
fi

echo "=== run_replay v2 | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "ref=$REF out=$OUT algos=$ALGOS jobs=$JOBS backend=$BACKEND log=$LOG"
echo "bash $BASH_VERSION | $($PY --version 2>&1) | venv=${VIRTUAL_ENV:-none} | cores=$(nproc 2>/dev/null || echo ?)"
if command -v free >/dev/null 2>&1; then free -g | sed -n '1,2p'; fi
if [[ "$BACKEND" == "nats" ]]; then
  $PY -c "import nats_bench" 2>/dev/null \
    || { echo "PREFLIGHT FAIL: nats_bench not importable (venv not active?)"; exit 3; }
fi
[[ -d "$REF" ]] || { echo "PREFLIGHT FAIL: reference run not found: $REF"; exit 3; }

SP_ARGS=""
if [[ -n "$SPACES_DIR" ]]; then SP_ARGS="--spaces-dir $SPACES_DIR"; fi

echo "--- stage 1/3: smoke"
# shellcheck disable=SC2086
$PY replay_dynamics.py --ref-dir "$REF" --outdir "$OUT" --algos "$ALGOS" \
    --backend "$BACKEND" $SP_ARGS --smoke
echo "--- stage 2/3: replay"
# shellcheck disable=SC2086
$PY replay_dynamics.py --ref-dir "$REF" --outdir "$OUT" --algos "$ALGOS" \
    --backend "$BACKEND" $SP_ARGS --jobs "$JOBS"
echo "--- stage 3/3: gate"
$PY aggregate_dynamics.py --ref-dir "$REF" --replay-dir "$OUT" \
    --out-dir "$OUT/summary"
echo "[run_replay] DONE. Send back the folder $OUT/summary/ (3 files)."
