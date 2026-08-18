"""Search algorithms over an explicit membership set, at a fixed query budget.

Every algorithm sees the same interface: a list of member architectures
(6-tuples), a backend, a budget B, and an rng. One query = one call to
backend.query_val (a noisy training trial). Every proposal consumes budget,
including re-proposals of already-seen architectures — re-evaluation under a
fresh trial is informative and matches the deployed cost model. Proposals are
constrained to the membership set by construction (mutation / neighbourhood /
sampling restricted to members), with a bounded rejection loop and a uniform
in-space fallback so no algorithm can stall.

Hyperparameters follow the NB201 / common-practice settings for this space and
are FROZEN across all spaces and datasets (recorded in the unit output), so no
per-space tuning can favour any condition:
  RE:        population 10, tournament sample 3, single-edge mutation
  REINFORCE: edge-factorised categorical policy, lr 0.01, EMA baseline 0.9
  LS:        first-improvement over the one-edge neighbourhood, random restarts

Each run returns the query trace:
  {"best_val": [...], "best_arch": [...], "n_proposals_rejected": int}
with one entry per query (running best), from which anytime curves and final
selections are computed downstream.
"""
import math

from nb201 import OPS, N_EDGES

MAX_REJECT = 100


def _record(trace, best_val, best_arch):
    trace["best_val"].append(best_val)
    trace["best_arch"].append(list(best_arch))


def _fallback(members, rng):
    return members[rng.randrange(len(members))]


def random_search(members, member_set, backend, budget, rng):
    trace = {"best_val": [], "best_arch": [], "n_proposals_rejected": 0}
    best_val, best_arch = -math.inf, None
    for _ in range(budget):
        arch = _fallback(members, rng)
        v = backend.query_val(arch)
        if v > best_val:
            best_val, best_arch = v, arch
        _record(trace, best_val, best_arch)
    return trace


def _mutate_in_space(arch, member_set, members, rng, counter):
    for _ in range(MAX_REJECT):
        edge = rng.randrange(N_EDGES)
        op = rng.randrange(len(OPS))
        cand = list(arch)
        cand[edge] = op
        cand = tuple(cand)
        if cand in member_set and cand != arch:
            return cand
        counter[0] += 1
    return _fallback(members, rng)


def regularized_evolution(members, member_set, backend, budget, rng,
                          pop_size=10, sample_size=3):
    trace = {"best_val": [], "best_arch": [], "n_proposals_rejected": 0}
    rejected = [0]
    population = []  # list of (arch, val), FIFO ageing
    best_val, best_arch = -math.inf, None
    for t in range(budget):
        if t < pop_size:
            arch = _fallback(members, rng)
        else:
            contestants = [population[rng.randrange(len(population))]
                           for _ in range(sample_size)]
            parent = max(contestants, key=lambda av: av[1])[0]
            arch = _mutate_in_space(parent, member_set, members, rng, rejected)
        v = backend.query_val(arch)
        population.append((arch, v))
        if len(population) > pop_size:
            population.pop(0)  # remove oldest (ageing)
        if v > best_val:
            best_val, best_arch = v, arch
        _record(trace, best_val, best_arch)
    trace["n_proposals_rejected"] = rejected[0]
    return trace


def reinforce(members, member_set, backend, budget, rng,
              lr=0.01, baseline_decay=0.9):
    trace = {"best_val": [], "best_arch": [], "n_proposals_rejected": 0}
    n_ops = len(OPS)
    logits = [[0.0] * n_ops for _ in range(N_EDGES)]
    baseline, rejected = None, 0

    def sample():
        arch = []
        for e in range(N_EDGES):
            mx = max(logits[e])
            ps = [math.exp(l - mx) for l in logits[e]]
            z = sum(ps)
            r, acc = rng.random() * z, 0.0
            pick = n_ops - 1
            for i, p in enumerate(ps):
                acc += p
                if r <= acc:
                    pick = i
                    break
            arch.append(pick)
        return tuple(arch)

    best_val, best_arch = -math.inf, None
    for _ in range(budget):
        arch = None
        for _try in range(MAX_REJECT):
            cand = sample()
            if cand in member_set:
                arch = cand
                break
            rejected += 1
        if arch is None:
            arch = _fallback(members, rng)
        v = backend.query_val(arch)
        baseline = v if baseline is None else \
            baseline_decay * baseline + (1 - baseline_decay) * v
        adv = v - baseline
        for e in range(N_EDGES):
            mx = max(logits[e])
            ps = [math.exp(l - mx) for l in logits[e]]
            z = sum(ps)
            for i in range(n_ops):
                grad = ((1.0 if i == arch[e] else 0.0) - ps[i] / z)
                logits[e][i] += lr * adv * grad
        if v > best_val:
            best_val, best_arch = v, arch
        _record(trace, best_val, best_arch)
    trace["n_proposals_rejected"] = rejected
    return trace


def local_search(members, member_set, backend, budget, rng):
    trace = {"best_val": [], "best_arch": [], "n_proposals_rejected": 0}
    best_val, best_arch = -math.inf, None
    queries = 0

    def neighbours(arch):
        out = []
        for e in range(N_EDGES):
            for op in range(len(OPS)):
                if op == arch[e]:
                    continue
                cand = list(arch)
                cand[e] = op
                cand = tuple(cand)
                if cand in member_set:
                    out.append(cand)
        rng.shuffle(out)
        return out

    while queries < budget:
        current = _fallback(members, rng)
        cur_val = backend.query_val(current)
        queries += 1
        if cur_val > best_val:
            best_val, best_arch = cur_val, current
        _record(trace, best_val, best_arch)
        improved = True
        while improved and queries < budget:
            improved = False
            for cand in neighbours(current):
                v = backend.query_val(cand)
                queries += 1
                if v > best_val:
                    best_val, best_arch = v, cand
                _record(trace, best_val, best_arch)
                if v > cur_val:
                    current, cur_val = cand, v
                    improved = True
                    break
                if queries >= budget:
                    break
    return trace


ALGORITHMS = {
    "rs": random_search,
    "re": regularized_evolution,
    "reinforce": reinforce,
    "ls": local_search,
}
