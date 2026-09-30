# Search-Space Reduction in Neural Architecture Search: Does It Help or Just Reshuffle the Budget?

Code and results for the paper by Sergei Kurashkin, Vladimir Nelyub, Aleksei
Borodulin and Vadim Tynchenko, EDMML workshop at IEEE ICDM 2026 (ICDM
Workshops proceedings). The repository contains the complete equal-budget,
size-matched evaluation pipeline, the frozen configuration of the gated final
run, and the per-run and aggregated results behind every number in the paper.

## What is here

- `runner/` — the full pipeline:
  - `download_data.sh` fetches the NATS-Bench topology benchmark and the
    NAS-Bench-Suite-Zero proxy scores; `build_spaces.py` freezes and hashes
    the space memberships.
  - `bench_runner.py` is the job-based unit runner (one unit = dataset x
    algorithm x space x block of 25 seeds; resume by existing output,
    per-unit timeout). `algorithms.py` holds random search, regularised
    evolution, REINFORCE and local search with in-space closure (bounded
    rejection, at most 100 attempts per proposal).
  - `aggregate.py` computes the statistics (Wilcoxon signed-rank with Pratt
    zeros, Holm within each dataset x algorithm family of eight tests,
    bootstrap CIs with 10,000 resamples, Hodges-Lehmann, Friedman + Nemenyi);
    `make_figures.py` draws the figures with the names used in the paper.
  - `review_gate.py` + `gate_config.yaml` block the final stage unless the
    clean no-resume, external-validity, statistics, frozen-space and
    single-benchmark checks pass.
  - `replay_dynamics.py`, `aggregate_dynamics.py`, `run_replay.sh` replay the
    regularised-evolution runs deterministically, verify every seed bit for
    bit against the final run, and compute the search-dynamics metrics of
    Table V (population diversity, distinct architectures evaluated,
    rejection rates, convergence, distinct end points).
- `results/` — the gated final run: per-unit JSON files (every seed with its
  anytime trace, final architecture, membership fractions and rejection
  count), `merged.csv`, `stats/` (paired.json, omnibus.json, posthoc.json),
  `report.md`, `manifest.jsonl`, `run_meta.json`, `figs/`.
- `results/replay_dynamics/` — the deterministic replay of all 4,200
  regularised-evolution searches: `summary/` (Table V, the report and all
  metrics), the replay manifest, meta data and log. Every seed reproduced the
  final run bit for bit.
- `paper/SearchSpaceReductionNAS_extended.pdf` — extended version of the
  paper; its Appendix A adds the absolute anytime curves, the full
  per-checkpoint table, the effect heatmap and the critical-difference
  diagram that the eight-page proceedings version omits.
- `requirements.txt` — the pinned environment.

## Where each table and figure comes from

| Paper item | Source |
|---|---|
| Table II (means and SDs) | per-unit results, or `report.md` for the means |
| Table III | `stats/paired.json` (reference `matched_random`) |
| Fig. 2, Table IV | `anytime_test` in the per-unit results / `merged.csv` |
| Fig. 3 | `membership_frac_of_best_trajectory` in the full-space units |
| Extended version, Appendix A | same sources; CD diagram from `stats/posthoc.json` |
| Table V | `results/replay_dynamics/summary/` (made by `runner/aggregate_dynamics.py` from the final run and its replay) |

## Reproduce

```bash
pip install -r requirements.txt
bash runner/download_data.sh          # benchmark (SHA-256 pinned) + proxy scores
python3 runner/build_spaces.py        # freeze memberships, write data/spaces_*.json
bash runner/pipeline.sh smoke         # end-to-end check on a small slice
bash runner/pipeline.sh final         # full grid, resume disabled, gate-blocked
bash runner/run_replay.sh             # Table V: deterministic replay + gate
```

The full grid is 672 units and 16,800 searches and took 33.5 CPU-hours
(median 179 s per unit) on a CPU node; no GPU is used. The replay of
regularised evolution (168 units) took about 25 minutes with 16 parallel jobs.
Recommended launch, no terminal multiplexer needed:
`cd runner && nohup bash run_replay.sh > replay_console.log 2>&1 &`.

## Not included

The NATS-Bench benchmark file and the NAS-Bench-Suite-Zero scores are
third-party data; `download_data.sh` fetches them, and the pipeline verifies
the benchmark archive against its pinned SHA-256 (prefix 580fd8f3) before
answering any query. The frozen membership files are rebuilt
deterministically by `build_spaces.py`, and the gate checks their hashes.

## Citation

```bibtex
@inproceedings{kurashkin2026searchspace,
  author    = {Kurashkin, Sergei and Nelyub, Vladimir and Borodulin, Aleksei and Tynchenko, Vadim},
  title     = {Search-Space Reduction in Neural Architecture Search: Does It Help or Just Reshuffle the Budget?},
  booktitle = {IEEE International Conference on Data Mining Workshops (ICDMW)},
  year      = {2026}
}
```

## Licence

MIT, see `LICENSE`.
