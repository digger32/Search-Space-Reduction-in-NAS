"""Search-space definitions: the full NB201 topology space and its reductions.

Informed reductions are FROZEN per dataset by build_spaces (membership lists on
disk + sha256), so every algorithm and every seed searches literally the same
subspace and the gate can assert the hash. Size-matched random subspaces are
derived deterministically from (size, search seed) at run time — they are the
control that answers "help or reshuffle", and they must vary per seed to be a
proper control distribution.

Space tokens (the --spaces axis of the runner):
  full         all 15,625 architectures
  no_none      exclude any architecture containing the 'none' op  (4,096)
  synflow50    top 50% by the synflow zero-cost proxy             (7,813)
  naswot50     top 50% by the NASWOT zero-cost proxy              (7,813)
  param50      top 50% by parameter count                          (7,813)
  randM<size>  size-matched random subspace, e.g. randM4096, randM7813
               (sampled per search seed; paired control for the informed
                reduction of the same size)
"""
import hashlib
import json
import random as _random
from pathlib import Path

from nb201 import all_archs, N_ARCHS

INFORMED = ["no_none", "synflow50", "naswot50", "param50"]


def _hash_members(members):
    payload = ",".join(str(m) for m in sorted(members))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_informed_spaces(backend, dataset, out_path):
    """Construct and freeze the informed reductions for one dataset.
    Run ONCE per dataset per backend (pipeline 'spaces' step), before any
    search. Writes membership lists + hashes; reruns must reproduce the hash.
    """
    archs = all_archs()
    spaces = {}

    spaces["no_none"] = [a for a in archs if 0 not in a]

    half = N_ARCHS // 2 + 1  # 7813
    for proxy, token in (("synflow", "synflow50"), ("naswot", "naswot50")):
        try:
            scored = sorted(archs, key=lambda a: backend.zc_score(a, proxy),
                            reverse=True)
            spaces[token] = scored[:half]
        except (RuntimeError, KeyError):
            spaces[token] = None  # zc scores not available for this dataset

    by_params = sorted(archs, key=backend.params, reverse=True)
    spaces["param50"] = by_params[:half]

    payload = {"dataset": dataset, "n_archs": N_ARCHS, "spaces": {}, "hashes": {}}
    for token, members in spaces.items():
        if members is None:
            payload["spaces"][token] = None
            continue
        as_lists = [list(m) for m in members]
        payload["spaces"][token] = as_lists
        payload["hashes"][token] = _hash_members(members)
    Path(out_path).write_text(json.dumps(payload))
    return payload["hashes"]


def load_members(spaces_path, token, seed):
    """Return the membership list for `token` (list of tuples).
    randM<size> is derived from (size, seed); informed tokens come from the
    frozen file. `full` needs no file."""
    if token == "full":
        return all_archs()
    if token.startswith("randM"):
        size = int(token[len("randM"):])
        rng = _random.Random(0xB6 ^ (size * 1_000_003) ^ seed)
        return [tuple(a) for a in rng.sample(all_archs(), size)]
    payload = json.loads(Path(spaces_path).read_text())
    members = payload["spaces"].get(token)
    if members is None:
        raise RuntimeError(f"space '{token}' unavailable for "
                           f"{payload['dataset']} (zc scores missing?)")
    return [tuple(m) for m in members]


def frozen_hashes(spaces_path):
    return json.loads(Path(spaces_path).read_text())["hashes"]
