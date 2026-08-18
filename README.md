# Search-Space Reduction in NAS: Does It Help or Just Reshuffle the Budget?

Anonymised reproduction package for the EDMML @ ICDM 2026 submission.
Contains the complete equal-budget, size-matched evaluation pipeline, the
frozen configuration of the gated final run, and the aggregated results
behind every number in the paper.

## What is here

- `runner/` — the full pipeline: `download_data.sh` (benchmark + zero-cost
  scores), `build_spaces.py` (freeze + hash space memberships),
  `bench_runner.py` (job-based unit runner: one unit = dataset x algorithm x
  space x seed-block of 25, resume by skipping existing outputs, per-unit
  timeout), `algorithms.py` (rs / regularised evolution / REINFORCE / local
  search, in-space closure with bounded rejection MAX_REJECT=100),
  `aggregate.py` (Wilcoxon-Pratt + Holm, bootstrap CIs with N_BOOT=10,000,
  Hodges-Lehmann, Friedman + Nemenyi), `make_figures.py` (print-scale,
  colourblind-safe figures; optional second argument writes figures outside
  a frozen run directory), `review_gate.py` + `gate_config.yaml` (the gate
  that blocks the final stage unless A1 clean no-resume, B1 external
  validity, E1 statistics, S1 frozen spaces and M1 single-benchmark checks
  pass), `gate_unittest.py`, `pipeline.sh`.
- `results/` — the gated final run (`GATE PASSED 5/5`): `merged.csv`
  (16,800 seed-records), `stats/` (paired.json / omnibus.json /
  posthoc.json), `report.md`, `manifest.jsonl` (per-unit wall-clock and
  no-resume flags for the A1 check), `run_meta.json`, `figs/`.
- `requirements.txt` — pinned environment (numpy 1.26.4, scipy 1.13.1,
  pandas 2.2.3, matplotlib 3.9.4, scikit-posthocs 0.14.0, nats_bench 1.8).

Every table and figure in the paper is derived from `results/`: Table II =
per-cell accuracies in `report.md`; Table III = `stats/paired.json`
(matched_random reference); Table IV and Fig. 3 = `anytime_test` in the unit
outputs aggregated in `merged.csv`; Fig. 2 = `stats/posthoc.json`; Figs. 4-5
= unit outputs and `stats/paired.json`.

## Minimal run

```bash
pip install -r requirements.txt
bash runner/download_data.sh          # NATS-tss benchmark (sha256-pinned) + suite-zero scores
python3 runner/build_spaces.py        # freeze memberships, write data/spaces_*.json + hashes
bash runner/pipeline.sh smoke         # end-to-end sanity on a tiny slice
bash runner/pipeline.sh final         # full grid, resume disabled, gate-blocked
```

The full grid is 672 units / 16,800 searches and completed in 33.5 CPU-hours
(median 179 s per unit) on a CPU node; no GPU is used.

## What is not included, and why

- The NATS-Bench benchmark file (~GBs) and the NAS-Bench-Suite-Zero score
  release are third-party assets; `download_data.sh` fetches them and the
  pipeline verifies the benchmark archive against its pinned SHA-256
  (prefix 580fd8f3) before any query is answered.
- `data/spaces_*.json` (frozen membership lists) are regenerated
  deterministically by `build_spaces.py` from the downloaded scores; their
  hashes are asserted by the gate (S1), so a regenerated copy is
  bit-verifiable against the run recorded here.

## Anonymity note

This repository is anonymised for triple-blind review: no author names,
emails, or institutional identifiers appear in code, comments, configuration
files, or this README.
