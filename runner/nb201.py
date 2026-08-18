"""NAS-Bench-201 (NATS-Bench topology space) backend.

Two implementations behind one interface:
  - NatsBackend : real tabular benchmark via nats_bench (file required)
  - MockBackend : deterministic synthetic benchmark for smoke / pilot / gate
                  unit tests, no download needed

Interface used by the search algorithms:
  backend.n_archs                     -> int (15625 for tss)
  backend.query_val(arch, seed_rng)   -> float  validation accuracy (noisy trial)
  backend.test_acc(arch)              -> float  mean test accuracy (final report)
  backend.params(arch)                -> float  parameter count (MB)
  backend.zc_score(arch, proxy)       -> float  precomputed zero-cost proxy score

An architecture is a 6-tuple of op indices (edges in fixed NB201 order).
API facts verified against nats_bench==1.8 source:
  create(path, 'tss', fast_mode=True, verbose=False)
  api.get_more_info(index, dataset, hp='200', is_random=True)
      -> dict with 'valid-accuracy', 'test-accuracy'
  api.get_cost_info(index, dataset, hp='200') -> dict with 'params'
  datasets: 'cifar10-valid', 'cifar100', 'ImageNet16-120'
"""
import hashlib
import os
import json
import math
from pathlib import Path

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
N_EDGES = 6
N_ARCHS = len(OPS) ** N_EDGES  # 15625

# nats_bench dataset tokens; 'cifar10-valid' is the proper train/val protocol
DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]


def arch_to_str(arch):
    """6-tuple of op ids -> NB201 arch string (format verified in nats_bench)."""
    o = [OPS[i] for i in arch]
    return (f"|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|")


def all_archs():
    """Enumerate all 6-tuples in a fixed, reproducible order."""
    archs = []
    for idx in range(N_ARCHS):
        t, rem = [], idx
        for _ in range(N_EDGES):
            t.append(rem % len(OPS))
            rem //= len(OPS)
        archs.append(tuple(t))
    return archs


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_bench_sha(bench_path):
    """Benchmark fingerprint for the gate's M1.
    fast_mode uses a DIRECTORY, which cannot be hashed directly; in that case
    read data/BENCH_SHA256.txt (written by download_data.sh from the tar)."""
    p = Path(bench_path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        for cand in (p.parent / "BENCH_SHA256.txt", p / "BENCH_SHA256.txt"):
            if cand.exists():
                tok = cand.read_text().split()
                if tok:
                    return tok[0]
    return None


class NatsBackend:
    """Real NATS-Bench topology-space backend (lazy import, file required).

    fast_mode caches every touched architecture in api.arch2infos_dict and
    never evicts; a full-grid unit (25 seeds x 1000 queries) touches most of
    the 15,625-arch space, so 24 parallel workers would hold ~24 copies of the
    benchmark in RAM and trip the OOM killer (observed on Server B, 64 GB).
    We bound the cache: past CACHE_ARCHS entries the oldest are dropped and
    transparently reloaded from disk on the next touch (insertion-order
    eviction; api internals verified against nats_bench==1.8:
    _prepare_info -> reload(archive_dir, index), OrderedDict + set).
    Override via env B6_CACHE_ARCHS."""

    CACHE_ARCHS = int(os.environ.get("B6_CACHE_ARCHS", "1000"))

    def __init__(self, bench_path, dataset, zc_path=None, hp="200"):
        from nats_bench import create  # heavy import stays inside the unit
        self.api = create(str(bench_path), "tss", fast_mode=True, verbose=False)
        self.dataset = dataset
        self.hp = hp
        self.n_archs = N_ARCHS
        self._idx = {}          # arch tuple -> api index (filled lazily)
        self._test_cache = {}
        self._param_cache = {}
        self._zc = None
        if zc_path and Path(zc_path).exists():
            self._zc = json.loads(Path(zc_path).read_text())

    def _evict(self):
        cache = self.api.arch2infos_dict
        while len(cache) > self.CACHE_ARCHS:
            idx, _ = cache.popitem(last=False)
            self.api.evaluated_indexes.discard(idx)

    def _index(self, arch):
        if arch not in self._idx:
            self._idx[arch] = self.api.query_index_by_arch(arch_to_str(arch))
        return self._idx[arch]

    def query_val(self, arch, rng=None):
        """One noisy training trial's validation accuracy (search feedback).
        nats_bench draws the trial via the GLOBAL `random` module when
        is_random=True; the worker seeds `random` per unit, so this is
        deterministic given the unit seed."""
        info = self.api.get_more_info(self._index(arch), self.dataset,
                                      hp=self.hp, is_random=True)
        self._evict()
        return float(info["valid-accuracy"])

    def test_acc(self, arch):
        """Mean test accuracy over trials (is_random=False) — reporting only."""
        if arch not in self._test_cache:
            info = self.api.get_more_info(self._index(arch), self.dataset,
                                          hp=self.hp, is_random=False)
            self._evict()
            self._test_cache[arch] = float(info["test-accuracy"])
        return self._test_cache[arch]

    def params(self, arch):
        if arch not in self._param_cache:
            info = self.api.get_cost_info(self._index(arch), self.dataset,
                                          hp=self.hp)
            self._evict()
            self._param_cache[arch] = float(info["params"])
        return self._param_cache[arch]

    def zc_score(self, arch, proxy):
        if self._zc is None:
            raise RuntimeError("zero-cost scores not loaded (data/zc_scores_*.json)")
        return float(self._zc[proxy][str(self._index(arch))])


class MockBackend:
    """Deterministic synthetic benchmark. Accuracy is a smooth function of the
    ops plus per-query noise, so search algorithms have a real (if easy)
    landscape and the full pipeline exercises identical code paths."""

    _GAIN = {0: 0.0, 1: 1.5, 2: 3.0, 3: 4.0, 4: 1.0}  # op id -> contribution

    def __init__(self, bench_path=None, dataset="cifar10-valid", zc_path=None,
                 hp="200"):
        self.dataset = dataset
        self.n_archs = N_ARCHS
        self._dshift = {"cifar10-valid": 66.0, "cifar100": 45.0,
                        "ImageNet16-120": 30.0}.get(dataset, 50.0)

    def _base(self, arch):
        return self._dshift + sum(self._GAIN[o] for o in arch)

    def query_val(self, arch, rng=None):
        import random
        return self._base(arch) + random.gauss(0.0, 0.5)

    def test_acc(self, arch):
        # deterministic pseudo-test with a small arch-dependent offset
        h = int(hashlib.sha256(str(arch).encode()).hexdigest(), 16)
        return self._base(arch) + ((h % 1000) / 1000.0 - 0.5)

    def params(self, arch):
        return 0.1 + 0.3 * sum(1 for o in arch if o in (2, 3))

    def zc_score(self, arch, proxy):
        # correlated-with-accuracy proxy plus arch-dependent distortion
        h = int(hashlib.sha256((proxy + str(arch)).encode()).hexdigest(), 16)
        return self._base(arch) + ((h % 2000) / 2000.0 - 0.5) * 6.0


def make_backend(kind, bench_path, dataset, zc_path=None, hp="200"):
    if kind == "nats":
        return NatsBackend(bench_path, dataset, zc_path=zc_path, hp=hp)
    if kind == "mock":
        return MockBackend(bench_path, dataset, zc_path=zc_path, hp=hp)
    raise ValueError(f"unknown backend kind: {kind}")
