#!/usr/bin/env python3
"""Convert NAS-Bench-Suite-Zero's zc_nasbench201.json into the per-dataset
files spaces.py reads:

    data/zc_scores_<dataset>.json = {"synflow": {"<api_index>": score, ...},
                                     "naswot":  {...}}

Run AFTER the NATS-tss benchmark is downloaded (the arch-string -> api-index
mapping needs the api). The suite-zero schema is probed defensively: keys may
be arch strings or indices; per-proxy values may be floats or {"score": ...}
dicts; suite-zero's dataset token 'cifar10' maps to our 'cifar10-valid'.

Usage:
    python3 convert_zc.py ../data/zc_nasbench201.json \
        --bench-path ../data/NATS-tss-v1_0-3ffb9-simple --out-dir ../data
"""
import argparse
import json
import sys
from pathlib import Path

PROXIES = ("synflow", "naswot")  # naswot is sometimes stored as 'nwot'
ALIASES = {"naswot": ("naswot", "nwot"), "synflow": ("synflow",)}
DS_MAP = {"cifar10": "cifar10-valid", "cifar10-valid": "cifar10-valid",
          "cifar100": "cifar100", "ImageNet16-120": "ImageNet16-120"}

# NASLib op_indices encoding, ported VERBATIM from
# NASLib/zerocost naslib/search_spaces/nasbench201/conversions.py
# (verified against the repo on 2026-07-18): tuple position follows EDGE_LIST,
# int value indexes OP_NAMES_NB201.
OP_NAMES_NB201 = ["skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1",
                  "avg_pool_3x3"]
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def op_indices_to_str(op_indices):
    edge_op_dict = {edge: OP_NAMES_NB201[op]
                    for edge, op in zip(EDGE_LIST, op_indices)}
    op_edge_list = ["{}~{}".format(edge_op_dict[(i, j)], i - 1)
                    for i, j in sorted(edge_op_dict, key=lambda x: x[1])]
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def get_score(rec, proxy):
    for name in ALIASES[proxy]:
        if name in rec:
            v = rec[name]
            return float(v["score"]) if isinstance(v, dict) else float(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zc_json")
    ap.add_argument("--bench-path", required=True)
    ap.add_argument("--out-dir", default="../data")
    a = ap.parse_args()

    raw = json.loads(Path(a.zc_json).read_text())
    datasets = [d for d in raw if d in DS_MAP]
    if not datasets:
        sys.exit(f"[convert_zc] no known dataset keys in {list(raw)[:5]} — "
                 "inspect the file schema manually")

    from nats_bench import create
    api = create(a.bench_path, "tss", fast_mode=True, verbose=False)

    for ds in datasets:
        table = raw[ds]
        out = {p: {} for p in PROXIES}
        n_str = n_idx = n_tup = 0
        for key, rec in table.items():
            if key.startswith("("):  # NASLib op_indices tuple as str
                import ast
                idx = api.query_index_by_arch(
                    op_indices_to_str(ast.literal_eval(key)))
                n_tup += 1
            elif key.lstrip("-").isdigit():
                idx = int(key)
                n_idx += 1
            else:
                idx = api.query_index_by_arch(key)
                n_str += 1
            for p in PROXIES:
                s = get_score(rec, p)
                if s is not None:
                    out[p][str(idx)] = s
        missing = [p for p in PROXIES if not out[p]]
        if missing:
            print(f"[convert_zc] WARNING {ds}: proxies missing entirely: "
                  f"{missing} (available keys example: "
                  f"{list(next(iter(table.values())))[:8]})")
        covered = {p: len(out[p]) for p in PROXIES}
        out_path = Path(a.out_dir) / f"zc_scores_{DS_MAP[ds]}.json"
        out_path.write_text(json.dumps(out))
        print(f"[convert_zc] {ds} -> {out_path.name} | coverage {covered} | "
              f"keys: {n_tup} tuple, {n_str} arch-str, {n_idx} index")
        if any(c not in (0, 15625) for c in covered.values()):
            print(f"[convert_zc] NOTE {ds}: partial coverage — synflow50/"
                  "naswot50 would rank only covered archs; verify before "
                  "freezing spaces")


if __name__ == "__main__":
    main()
