"""
Permutation-space TSP solver with cluster-flip updates.

Rationale
---------
The n^2-spin QUBO formulation of TSP confines single-spin Gibbs
dynamics to an exponentially small feasible manifold inside the
full 2^{n^2} configuration space; the main-benchmark results in
bench_tsp.py demonstrate the consequent performance ceiling (gap
~50%, feasibility rate <25%). This module implements an
alternative search strategy that stays within the feasibility
manifold by construction: the state is a permutation and each
update is a move between valid permutations. The acceptance rule
is the standard Metropolis-Hastings criterion driven by tour-
length difference, equivalent in form to the Metropolis update
used by the Ising solver elsewhere in this framework.

The relationship to the p-bit / sMTJ abstraction
-------------------------------------------------
In the QUBO representation, a 2-opt move (reversing tour segment
[i..j]) corresponds to simultaneously flipping 2(j-i+1) one-hot
spin variables; a swap move flips 4 variables. The sMTJ
hardware implementing this would require clustered simultaneous
flips across multiple p-bits — this is a non-trivial engineering
constraint that a pure single-spin p-bit network does not meet.
The permutation-space solver therefore serves as an **upper
bound** on what an enhanced sMTJ architecture (supporting cluster
flips or hierarchical group updates) could achieve, and is
reported alongside the QUBO lower bound to delimit the
algorithm-hardware design space.

Public API
----------
    PermutationSolverConfig : hyperparameters
    PermutationResult       : per-trial result (tour, length, trajectory)
    run_permutation_sa(D, config, rng) : single trial
    multistart_permutation(D, config, n_trials, ...) : parallel batch
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import List, Optional

import numpy as np

from isim import get_logger, schedule_beta, _next_rec, tts_at_confidence

log = get_logger("permtsp")


# ===========================================================================
# Config and result dataclasses
# ===========================================================================

@dataclass
class PermutationSolverConfig:
    """Hyperparameters for the permutation-space TSP solver.

    Fields
    ------
    schedule_shape : 'geometric' | 'linear' | 'inverse_log'
        Annealing schedule for beta.
    beta0, betaf : float
        Initial and final inverse temperatures (applied directly to
        tour-length differences, not to spin energies).
    n_sweeps : int
        Number of sweeps; each sweep performs n proposal-acceptance
        steps where n is the number of cities.
    move_mix : dict
        Probabilities of each move type per proposal. Must sum to 1.
        Supported keys: '2opt', 'swap', 'insert'.
    record : str, record_step : int
        Trajectory recording control, same semantics as IsingSolver.
    """
    schedule_shape: str = "geometric"
    beta0: float = 0.1
    betaf: float = 50.0
    n_sweeps: int = 5000
    move_mix: dict = field(default_factory=lambda: {
        "2opt": 0.85, "swap": 0.10, "insert": 0.05,
    })
    record: str = "geometric"
    record_step: int = 100

    def validate(self):
        if self.schedule_shape not in ("geometric", "linear", "inverse_log"):
            raise ValueError(f"bad schedule: {self.schedule_shape}")
        if not np.isclose(sum(self.move_mix.values()), 1.0):
            raise ValueError("move_mix probabilities must sum to 1")
        for k in self.move_mix:
            if k not in ("2opt", "swap", "insert"):
                raise ValueError(f"unknown move type: {k}")


@dataclass
class PermutationResult:
    """Outcome of one permutation-space SA trial."""
    tour: np.ndarray         # final tour as 0-indexed int array
    length: float            # final tour length (same as length at tour)
    length_best: float       # best length seen during the trajectory
    tour_best: np.ndarray    # tour corresponding to length_best
    seed: int
    wall_time: float
    sweeps: np.ndarray       # sweep indices at which length was recorded
    lengths: np.ndarray      # corresponding tour lengths
    accepted: int            # total accepted proposals
    proposed: int            # total proposals made
    config: PermutationSolverConfig = None

    def to_dict(self):
        return {
            "tour": [int(x) for x in self.tour],
            "tour_best": [int(x) for x in self.tour_best],
            "length": float(self.length),
            "length_best": float(self.length_best),
            "seed": int(self.seed),
            "wall_time": float(self.wall_time),
            "accepted": int(self.accepted),
            "proposed": int(self.proposed),
            "trajectory": {
                "sweeps": [int(x) for x in self.sweeps],
                "lengths": [float(x) for x in self.lengths],
            },
        }


# ===========================================================================
# Move proposals and delta-length computation
# ===========================================================================

def _propose_2opt(tour, D, rng):
    """Reverse tour[i..j]. Returns (i, j, dL). The move flips
    segment orientation; the two edges (a,b) and (c,d) are
    replaced by (a,c) and (b,d) where a=tour[i-1], b=tour[i],
    c=tour[j], d=tour[(j+1)%n]."""
    n = len(tour)
    # Draw i, j uniformly with i < j and at least one interior edge
    # affected (i.e., exclude the degenerate j==i and the full-cycle
    # reversal case where the move is an identity on cyclic tours).
    i = int(rng.integers(1, n - 1))
    j = int(rng.integers(i + 1, n))
    a = tour[i - 1]
    b = tour[i]
    c = tour[j]
    d = tour[(j + 1) % n]
    # Self-edge handling: a==c only if i==j (excluded above)
    dL = D[a, c] + D[b, d] - D[a, b] - D[c, d]
    return i, j, dL


def _apply_2opt(tour, i, j):
    tour[i:j + 1] = tour[i:j + 1][::-1]


def _propose_swap(tour, D, rng):
    """Swap positions i and j. Computes dL by looking at the 4 edges
    incident to positions i and j; special case when i, j are adjacent
    (one shared edge)."""
    n = len(tour)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    if i == j:
        return i, j, 0.0
    if i > j:
        i, j = j, i
    a = tour[(i - 1) % n]; b = tour[i]
    c = tour[j];           d = tour[(j + 1) % n]
    if j == i + 1:
        # Adjacent: only 2 edges change
        # Before: a-b-c-d, After: a-c-b-d
        dL = D[a, c] + D[b, d] - D[a, b] - D[c, d]
    elif (i == 0) and (j == n - 1):
        # Adjacent under cyclic wrap: positions 0 and n-1 share both edges
        # if swapped, tour becomes (b, ... interior ..., a) which is
        # just a rotation — dL = 0. Skip.
        return i, j, 0.0
    else:
        # Non-adjacent: 4 distinct edges affected
        bp = tour[i + 1]; ap = tour[j - 1]
        dL = (D[a, c] + D[c, bp] + D[ap, b] + D[b, d]
              - D[a, b] - D[b, bp] - D[ap, c] - D[c, d])
    return i, j, dL


def _apply_swap(tour, i, j):
    tour[i], tour[j] = tour[j], tour[i]


def _propose_insert(tour, D, rng):
    """Remove city at position i, reinsert at position j. Equivalent
    to a 3-edge exchange (3-opt restricted)."""
    n = len(tour)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    if i == j or (j == i + 1) or (i == 0 and j == n - 1):
        return i, j, 0.0
    # Compute dL by listing the 3 affected edges in old and new tours.
    # Build the proposed tour to get exact dL (cheap for small n).
    new_tour = np.concatenate([
        tour[:i], tour[i + 1:]
    ])
    x = tour[i]
    insert_at = j if j < i else j - 1
    new_tour = np.concatenate([
        new_tour[:insert_at], [x], new_tour[insert_at:]
    ])
    old_L = sum(D[tour[k], tour[(k + 1) % n]] for k in range(n))
    new_L = sum(D[new_tour[k], new_tour[(k + 1) % n]] for k in range(n))
    dL = new_L - old_L
    return i, j, dL


def _apply_insert(tour, i, j):
    x = tour[i]
    tour = np.delete(tour, i)
    insert_at = j if j < i else j - 1
    tour = np.insert(tour, insert_at, x)
    return tour


# ===========================================================================
# Single trial
# ===========================================================================

def run_permutation_sa(D: np.ndarray,
                       config: PermutationSolverConfig,
                       rng: np.random.Generator,
                       seed_hint: int = -1) -> PermutationResult:
    """One permutation-space SA trial. Metropolis acceptance on
    tour-length delta."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    T = config.n_sweeps

    # Initial random tour
    tour = rng.permutation(n).astype(np.int64)
    L = float(sum(D[tour[k], tour[(k + 1) % n]] for k in range(n)))
    L_best = L
    tour_best = tour.copy()

    # Move mix as cumulative probs
    mix = config.move_mix
    move_names = list(mix.keys())
    cum_probs = np.cumsum([mix[k] for k in move_names])

    sweeps = [0]
    lengths = [L]
    next_rec = _next_rec(0, config.record, config.record_step)
    n_accepted = 0
    n_proposed = 0

    t0 = time.perf_counter()
    for sw in range(T):
        beta = schedule_beta(config.schedule_shape, sw, T,
                             config.beta0, config.betaf)
        # One sweep = n proposals
        for _ in range(n):
            u = rng.random()
            move_idx = int(np.searchsorted(cum_probs, u))
            move = move_names[min(move_idx, len(move_names) - 1)]
            if move == "2opt":
                i, j, dL = _propose_2opt(tour, D, rng)
                if dL <= 0.0 or rng.random() < np.exp(-beta * dL):
                    _apply_2opt(tour, i, j)
                    L += dL
                    n_accepted += 1
            elif move == "swap":
                i, j, dL = _propose_swap(tour, D, rng)
                if dL == 0.0:
                    pass
                elif dL <= 0.0 or rng.random() < np.exp(-beta * dL):
                    _apply_swap(tour, i, j)
                    L += dL
                    n_accepted += 1
            else:  # insert
                i, j, dL = _propose_insert(tour, D, rng)
                if dL == 0.0:
                    pass
                elif dL <= 0.0 or rng.random() < np.exp(-beta * dL):
                    tour = _apply_insert(tour, i, j)
                    L += dL
                    n_accepted += 1
            n_proposed += 1
            if L < L_best:
                L_best = L
                tour_best = tour.copy()
        while sw + 1 >= next_rec:
            sweeps.append(sw + 1)
            lengths.append(L)
            next_rec = _next_rec(sw + 1, config.record, config.record_step)

    wall = time.perf_counter() - t0
    # Final length recomputation as a consistency check
    L_check = float(sum(D[tour[k], tour[(k + 1) % n]] for k in range(n)))
    if abs(L_check - L) > 1e-6 * (1.0 + abs(L_check)):
        L = L_check  # recover from any accumulated floating error
    return PermutationResult(
        tour=tour.copy(), length=L,
        length_best=L_best, tour_best=tour_best,
        seed=seed_hint, wall_time=wall,
        sweeps=np.array(sweeps), lengths=np.array(lengths),
        accepted=n_accepted, proposed=n_proposed,
        config=config,
    )


# ===========================================================================
# Parallel multistart
# ===========================================================================

def _worker(args):
    child_ss, D, config, entropy = args
    rng = np.random.default_rng(child_ss)
    return run_permutation_sa(D, config, rng, seed_hint=entropy)


def multistart_permutation(D: np.ndarray,
                           config: PermutationSolverConfig,
                           n_trials: int = 100,
                           master_seed: int = 2024,
                           n_jobs: int = 1,
                           progress: bool = True) -> List[PermutationResult]:
    """Run n_trials independent trials with seeds derived from a
    single master_seed via SeedSequence.spawn. Identical RNG
    protocol to isim.multistart for cross-method reproducibility."""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_trials)
    entropies = [int(c.entropy % (2 ** 63 - 1)) for c in children]
    args = [(children[i], D, config, entropies[i]) for i in range(n_trials)]

    log.info(f"multistart_permutation: n_trials={n_trials}, n_jobs={n_jobs}, "
             f"n_sweeps={config.n_sweeps}, "
             f"beta=({config.beta0}, {config.betaf}), "
             f"move_mix={config.move_mix}")

    use_tqdm = False
    if progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            pass

    results = []
    t_start = time.perf_counter()

    if n_jobs <= 1:
        iterator = args
        if use_tqdm:
            iterator = tqdm(args, ncols=80, unit="trial")
        for idx, a in enumerate(iterator):
            r = _worker(a)
            results.append(r)
            if not use_tqdm and progress \
                    and (idx + 1) % max(1, n_trials // 10) == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed * (n_trials - idx - 1) / (idx + 1)
                bests = np.array([r.length_best for r in results])
                log.info(f"  {idx+1}/{n_trials} L_best_min={bests.min():.1f} "
                         f"L_best_median={np.median(bests):.1f} "
                         f"elapsed={elapsed:.1f}s eta={eta:.1f}s")
    else:
        with Pool(n_jobs) as pool:
            if use_tqdm:
                from tqdm import tqdm
                for r in tqdm(pool.imap_unordered(_worker, args),
                              total=n_trials, ncols=80, unit="trial"):
                    results.append(r)
            else:
                chunksize = max(1, n_trials // (4 * n_jobs))
                for idx, r in enumerate(pool.imap_unordered(
                        _worker, args, chunksize=chunksize)):
                    results.append(r)
                    if progress and (idx + 1) % max(1, n_trials // 10) == 0:
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed * (n_trials - idx - 1) / (idx + 1)
                        bests = np.array([r.length_best for r in results])
                        log.info(f"  {idx+1}/{n_trials} "
                                 f"L_best_min={bests.min():.1f} "
                                 f"L_best_median={np.median(bests):.1f} "
                                 f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    total = time.perf_counter() - t_start
    bests = np.array([r.length_best for r in results])
    log.info(f"done: total_wall={total:.1f}s "
             f"per_trial_median="
             f"{np.median([r.wall_time for r in results]):.3f}s "
             f"L_best_min={bests.min():.1f} "
             f"L_best_median={np.median(bests):.1f}")
    return results


def summarize_permutation_runs(results: List[PermutationResult],
                                opt_len: float,
                                tol: float = 0.01) -> dict:
    lengths = np.array([r.length_best for r in results])
    times = np.array([r.wall_time for r in results])
    accepted = np.array([r.accepted for r in results])
    proposed = np.array([r.proposed for r in results])
    acc_rate = float(accepted.sum() / proposed.sum()) if proposed.sum() > 0 else 0.0
    best = float(lengths.min())
    median = float(np.median(lengths))
    gap_best = (best - opt_len) / opt_len
    gap_median = (median - opt_len) / opt_len
    success_mask = lengths <= opt_len * (1 + tol)
    p_s = float(success_mask.mean())
    t_med = float(np.median(times))
    tts = tts_at_confidence(t_med, p_s, 0.99)
    return {
        "n_trials": len(results),
        "opt_len": float(opt_len),
        "length_best": best,
        "length_median": median,
        "gap_best": gap_best,
        "gap_median": gap_median,
        "p_success": p_s,
        "tts_99_wall": tts,
        "time_median": t_med,
        "accept_rate": acc_rate,
    }
