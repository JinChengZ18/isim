"""
Ising solver simulation framework for sMTJ-based p-bit computing.

Performance design
------------------
Two execution paths are provided with identical semantics at the
statistical level; users pick by setting SolverConfig.update_mode:

  * 'async_numba' (default):   strict sequential single-spin updates,
                               JIT-compiled via numba. Requires numba
                               to be importable; raises ImportError at
                               configuration time otherwise. Expected
                               throughput ~50-100 million spin-updates
                               per second on a single CPU core.

  * 'block' (vectorized):      greedy-colored block-parallel Gibbs
                               updates, pure NumPy, no JIT dependency.
                               Expected throughput ~5-20 M updates/s.
                               Correct by construction: spins in the
                               same color class are mutually non-adjacent
                               so their simultaneous sampling coincides
                               with sequential Gibbs in distribution.

Both modes reproduce the Gibbs stationary distribution; the block mode
is preferred when numba is unavailable or when problems fit in cache
(small N, dense graphs).

Layer contract (Section 3.3.1)
------------------------------
    SpinBackend : given local field h_eff, current spin, and inverse
                  temperature, return an updated spin value in {-1, +1}.
                  Passing current spin allows Metropolis-Hastings to
                  compute delta-energy acceptance; ideal-Gibbs ignores
                  the current value.
    Problem     : symmetric sparse coupling matrix J (zero diagonal)
                  and external field vector h.
    IsingSolver : Gibbs/Metropolis update loop with annealing control.
    Metrics     : residual energy, success probability, time-to-solution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# ===========================================================================
# Logger
# ===========================================================================

def get_logger(name="isim"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


# ===========================================================================
# Numba-JIT inner loop (optional dependency)
# ===========================================================================

def _has_numba():
    try:
        import numba  # noqa
        return True
    except ImportError:
        return False


_NUMBA_COMPILED = None


def _compile_numba_kernels():
    """Build JIT-compiled async-sweep kernels the first time they are
    requested. Returns (gibbs_fn, metropolis_fn) or None if numba is
    not available."""
    global _NUMBA_COMPILED
    if _NUMBA_COMPILED is not None:
        return _NUMBA_COMPILED
    if not _has_numba():
        _NUMBA_COMPILED = (None, None)
        return _NUMBA_COMPILED

    from numba import njit

    @njit(cache=True, fastmath=True)
    def gibbs_async_sweep(s, indptr, indices, data, h, beta,
                          perm, rand_u, H_in):
        """One sweep: for each i in perm, compute h_eff, sample via Gibbs
        sigmoid, update s[i] and running energy H. Returns updated H."""
        H = H_in
        n = s.shape[0]
        for k in range(n):
            i = perm[k]
            start = indptr[i]
            end = indptr[i + 1]
            he = h[i]
            for kk in range(start, end):
                he += data[kk] * s[indices[kk]]
            # Sigmoid of 2*beta*he, numerically stable via tanh
            x = beta * he
            # p = 0.5 * (1 + tanh(x))
            # Compute tanh manually avoiding math import in nopython
            if x > 20.0:
                p = 1.0
            elif x < -20.0:
                p = 0.0
            else:
                e2 = np.exp(-2.0 * x)
                p = 1.0 / (1.0 + e2)
            new = 1 if rand_u[k] < p else -1
            if new != s[i]:
                H += -2.0 * new * he
                s[i] = new
        return H

    @njit(cache=True, fastmath=True)
    def metropolis_async_sweep(s, indptr, indices, data, h, beta,
                               perm, rand_u, H_in):
        """Sequential Metropolis-Hastings: propose flipping s[i], accept
        with probability min(1, exp(-beta * dE)) where
        dE = 2 * s[i] * h_eff (current spin)."""
        H = H_in
        n = s.shape[0]
        for k in range(n):
            i = perm[k]
            start = indptr[i]
            end = indptr[i + 1]
            he = h[i]
            for kk in range(start, end):
                he += data[kk] * s[indices[kk]]
            # Current spin is s[i]; flipping changes it to -s[i].
            # dE = 2 * s[i] * h_eff
            dE = 2.0 * s[i] * he
            accept = False
            if dE <= 0.0:
                accept = True
            else:
                x = -beta * dE
                if x > -20.0:
                    if rand_u[k] < np.exp(x):
                        accept = True
            if accept:
                s[i] = -s[i]
                H += dE  # H_new - H_old = +dE (we minimize H = -0.5 s J s - h s)
                          # Careful: dE here is the change in cost that the
                          # Metropolis test compares against; convention below.
        return H

    _NUMBA_COMPILED = (gibbs_async_sweep, metropolis_async_sweep)
    return _NUMBA_COMPILED


# ===========================================================================
# Spin backends
# ===========================================================================

def _sigmoid_np(x):
    """Numerically stable logistic sigmoid via tanh."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x)))


class SpinBackend:
    """Abstract spin backend. Subclasses implement the single-spin
    update and its batched form. The JIT-compiled solver bypasses this
    interface for performance; backends are used by the 'block' mode
    and by non-JIT fallbacks."""

    kind: str = "abstract"

    def sample(self, h_eff: float, s_cur: int, beta: float,
               rng: np.random.Generator) -> int:
        raise NotImplementedError

    def sample_batch(self, h_eff_arr: np.ndarray, s_cur_arr: np.ndarray,
                     beta: float, rng: np.random.Generator,
                     idx: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError


class IdealGibbsSpin(SpinBackend):
    """Exact Gibbs sampler. The update is conditional on h_eff only,
    independent of the current spin value (classical Glauber dynamics)."""

    kind = "ideal"

    def sample(self, h_eff, s_cur, beta, rng):
        p = _sigmoid_np(2.0 * beta * h_eff)
        return 1 if rng.random() < p else -1

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        p = _sigmoid_np(2.0 * beta * h_eff_arr)
        u = rng.random(size=p.shape)
        return np.where(u < p, 1, -1).astype(np.int8)


class SMTJSpin(SpinBackend):
    """sMTJ p-bit implementing the correspondence from Section 3.2.1:
        I_bias = 2 * beta * I_0 * h_eff
        P(+1)  = sigma(I_bias / I_0)
    The default here is an ideal-sigmoid device; subclass and override
    sample/sample_batch to plug in a Chapter-2 behavioral model."""

    kind = "smtj"

    def __init__(self, I_0: float = 1.0):
        self.I_0 = float(I_0)

    def sample(self, h_eff, s_cur, beta, rng):
        I_bias = 2.0 * beta * self.I_0 * h_eff
        p = _sigmoid_np(I_bias / self.I_0)
        return 1 if rng.random() < p else -1

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        I_bias = 2.0 * beta * self.I_0 * h_eff_arr
        p = _sigmoid_np(I_bias / self.I_0)
        u = rng.random(size=p.shape)
        return np.where(u < p, 1, -1).astype(np.int8)


class MetropolisSpin(SpinBackend):
    """Classical Metropolis-Hastings update: propose flipping the
    current spin and accept with probability min(1, exp(-beta * dE))
    where dE = 2 * s_cur * h_eff. Distinct from Gibbs dynamics."""

    kind = "metropolis"

    def sample(self, h_eff, s_cur, beta, rng):
        dE = 2.0 * s_cur * h_eff
        if dE <= 0.0:
            return -s_cur
        if rng.random() < np.exp(-beta * dE):
            return -s_cur
        return s_cur

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        dE = 2.0 * s_cur_arr * h_eff_arr
        u = rng.random(size=dE.shape)
        accept = (dE <= 0.0) | (u < np.exp(np.minimum(-beta * dE, 0.0)))
        return np.where(accept, -s_cur_arr, s_cur_arr).astype(np.int8)


def _rebuild_spin(spec):
    """Reconstruct a spin backend from a (kind, kwargs) pickle-safe spec,
    used by the parallel worker to avoid pickling device objects.
    Backends registered via register_spin_backend() are also resolvable."""
    kind, kwargs = spec
    if kind == "ideal":
        return IdealGibbsSpin()
    if kind == "smtj":
        return SMTJSpin(**kwargs)
    if kind == "metropolis":
        return MetropolisSpin()
    factory = _SPIN_REGISTRY.get(kind)
    if factory is None:
        raise ValueError(f"Unknown spin spec: {kind}")
    return factory(**kwargs)


# Registry for externally-supplied backends. The entry must be a
# zero-side-effect factory callable taking only keyword arguments,
# because parallel workers rebuild backends via _rebuild_spin.
_SPIN_REGISTRY = {}


def register_spin_backend(kind: str, factory):
    """Register a backend factory under the given kind string. After
    registration, spin_spec=(kind, kwargs) is resolvable in worker
    processes. Re-registration with the same kind silently overwrites
    the prior entry, which is the intended behaviour when a downstream
    module is reloaded.

    The factory must accept only keyword arguments and must be
    importable via fully-qualified name in worker processes -- in
    practice this means the factory must be defined at module top
    level in a module that the workers will re-import. A subclass
    of SpinBackend defined at top level satisfies this."""
    _SPIN_REGISTRY[str(kind)] = factory


# ===========================================================================
# Problem representation and preprocessing
# ===========================================================================

@dataclass
class Problem:
    """Ising problem H(s) = -(1/2) s^T J s - h^T s with J symmetric,
    J_ii = 0, stored in CSR format."""
    name: str
    n: int
    J: sp.csr_matrix
    h: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not sp.issparse(self.J):
            self.J = sp.csr_matrix(self.J)
        self.J = self.J.tocsr().astype(np.float64)
        self.h = np.asarray(self.h, dtype=np.float64).ravel()
        if self.h.size != self.n or self.J.shape != (self.n, self.n):
            raise ValueError(
                f"Shape mismatch: n={self.n}, h.size={self.h.size}, "
                f"J.shape={self.J.shape}")

    def energy(self, s) -> float:
        s = np.asarray(s, dtype=np.float64)
        return float(-0.5 * s @ (self.J @ s) - self.h @ s)

    def local_fields(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        return np.asarray(self.J @ s).ravel() + self.h


def preprocess(J, h, normalize: bool = True):
    """Symmetrize, clear diagonal, optionally normalize so max|J_ij|=1.
    See Section 3.3.2."""
    J = sp.csr_matrix(J).astype(np.float64)
    h = np.asarray(h, dtype=np.float64).copy()
    J = 0.5 * (J + J.T)
    d = J.diagonal()
    if np.any(d != 0):
        J = J - sp.diags(d)
    J = J.tocsr()
    J.eliminate_zeros()
    if normalize and J.nnz > 0:
        jmax = float(np.max(np.abs(J.data)))
        if jmax > 0:
            J = J / jmax
            h = h / jmax
    return J, h


# ===========================================================================
# Annealing schedules
# ===========================================================================

def schedule_beta(shape: str, t: int, T: int, beta0: float, betaf: float) -> float:
    if T <= 0:
        return betaf
    if shape == "linear":
        return beta0 + (betaf - beta0) * (t / T)
    if shape == "geometric":
        if beta0 <= 0:
            raise ValueError("geometric schedule requires beta0 > 0")
        gamma = (betaf / beta0) ** (1.0 / T)
        return beta0 * gamma ** t
    if shape == "inverse_log":
        c = (betaf - beta0) / np.log(1.0 + T)
        return beta0 + c * np.log(1.0 + t)
    raise ValueError(f"Unknown schedule shape: {shape}")


# ===========================================================================
# Solver configuration and result
# ===========================================================================

VALID_MODES = ("async_numba", "block", "async_python")


@dataclass
class SolverConfig:
    schedule_shape: str = "geometric"
    beta0: float = 0.1
    betaf: float = 10.0
    n_sweeps: int = 10000
    update_mode: str = "async_numba"   # see VALID_MODES
    dynamics: str = "gibbs"             # 'gibbs' | 'metropolis'
    record: str = "geometric"           # 'geometric' | 'linear' | 'none'
    record_step: int = 100
    energy_check_every: int = 2000

    def validate(self):
        if self.update_mode not in VALID_MODES:
            raise ValueError(
                f"update_mode must be one of {VALID_MODES}, got {self.update_mode!r}")
        if self.update_mode == "async_numba" and not _has_numba():
            raise ImportError(
                "update_mode='async_numba' requires the numba package. "
                "Install it with `pip install numba`, or pick "
                "update_mode='block' (fast pure-NumPy) or 'async_python' "
                "(slow reference).")
        if self.dynamics not in ("gibbs", "metropolis"):
            raise ValueError(f"dynamics must be 'gibbs' or 'metropolis'")


@dataclass
class RunResult:
    problem_name: str
    seed: int
    state_final: np.ndarray
    energy_final: float
    sweeps: np.ndarray
    energies: np.ndarray
    wall_time: float
    config: SolverConfig = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        """Serialize to a JSON-friendly dict (no numpy arrays)."""
        return {
            "problem_name": self.problem_name,
            "seed": int(self.seed),
            "energy_final": float(self.energy_final),
            "wall_time": float(self.wall_time),
            "n_sweeps_total": int(self.sweeps[-1]) if len(self.sweeps) else 0,
            "trajectory": {
                "sweeps": [int(x) for x in self.sweeps],
                "energies": [float(x) for x in self.energies],
            },
        }


# ===========================================================================
# IsingSolver
# ===========================================================================

class IsingSolver:
    def __init__(self, spin: SpinBackend, config: SolverConfig = None):
        self.spin = spin
        self.config = config or SolverConfig()
        self.config.validate()

    def run(self, problem: Problem, rng: np.random.Generator,
            state0: Optional[np.ndarray] = None,
            seed_hint: int = -1) -> RunResult:
        cfg = self.config
        n = problem.n
        if state0 is None:
            s = rng.choice([-1, 1], size=n).astype(np.int8)
        else:
            s = np.asarray(state0, dtype=np.int8).copy()

        H = problem.energy(s)
        sweeps = [0]
        energies = [H]
        next_rec = _next_rec(0, cfg.record, cfg.record_step)

        t0 = time.perf_counter()
        if cfg.update_mode == "async_numba":
            H = self._async_numba_loop(s, problem, rng, H, sweeps, energies, next_rec)
        elif cfg.update_mode == "block":
            H = self._block_loop(s, problem, rng, H, sweeps, energies, next_rec)
        elif cfg.update_mode == "async_python":
            H = self._async_python_loop(s, problem, rng, H, sweeps, energies, next_rec)
        else:
            raise ValueError(f"Unknown update_mode: {cfg.update_mode}")
        wall = time.perf_counter() - t0

        H_final = problem.energy(s)
        return RunResult(
            problem_name=problem.name,
            seed=seed_hint,
            state_final=s,
            energy_final=H_final,
            sweeps=np.array(sweeps),
            energies=np.array(energies),
            wall_time=wall,
            config=cfg,
        )

    # -- JIT-compiled sequential async update ------------------------------

    def _async_numba_loop(self, s, problem, rng, H, sweeps, energies, next_rec):
        cfg = self.config
        gibbs_fn, metropolis_fn = _compile_numba_kernels()
        if gibbs_fn is None:
            raise ImportError("numba not available")
        kernel = gibbs_fn if cfg.dynamics == "gibbs" else metropolis_fn

        n = problem.n
        T = cfg.n_sweeps
        indptr = problem.J.indptr.astype(np.int64)
        indices = problem.J.indices.astype(np.int64)
        data = problem.J.data.astype(np.float64)
        h = problem.h.astype(np.float64)
        s_arr = s.astype(np.int8)

        # Prepare all randomness in bulk to minimize per-sweep Python overhead.
        # Memory: 8*n per sweep for the uniforms; T sweeps would be too much.
        # Instead, draw per-sweep but keep Python work out of the inner loop.
        for sw in range(T):
            beta = schedule_beta(cfg.schedule_shape, sw, T, cfg.beta0, cfg.betaf)
            perm = rng.permutation(n).astype(np.int64)
            rand_u = rng.random(size=n)
            H = kernel(s_arr, indptr, indices, data, h, float(beta),
                       perm, rand_u, float(H))
            while sw + 1 >= next_rec:
                sweeps.append(sw + 1)
                energies.append(H)
                next_rec = _next_rec(sw + 1, cfg.record, cfg.record_step)
            if (sw + 1) % cfg.energy_check_every == 0:
                H_true = problem.energy(s_arr)
                if abs(H_true - H) > 1e-6 * (1.0 + abs(H_true)):
                    H = H_true
        # Copy state back
        s[:] = s_arr
        return H

    # -- Pure-Python async update (reference; slow; for debugging) --------

    def _async_python_loop(self, s, problem, rng, H, sweeps, energies, next_rec):
        cfg = self.config
        n = problem.n
        T = cfg.n_sweeps
        indptr, indices, data = (problem.J.indptr, problem.J.indices,
                                 problem.J.data)
        h = problem.h
        for sw in range(T):
            beta = schedule_beta(cfg.schedule_shape, sw, T, cfg.beta0, cfg.betaf)
            perm = rng.permutation(n)
            for i in perm:
                start, end = indptr[i], indptr[i + 1]
                h_eff = float(data[start:end] @ s[indices[start:end]]) + h[i]
                s_cur = int(s[i])
                s_new = self.spin.sample(h_eff, s_cur, beta, rng)
                if s_new != s_cur:
                    # Energy change: for Gibbs we use post-flip value;
                    # for Metropolis dE = 2 s_cur * h_eff and s_new = -s_cur
                    H += 2.0 * s_cur * h_eff
                    s[i] = s_new
            while sw + 1 >= next_rec:
                sweeps.append(sw + 1)
                energies.append(H)
                next_rec = _next_rec(sw + 1, cfg.record, cfg.record_step)
            if (sw + 1) % cfg.energy_check_every == 0:
                H_true = problem.energy(s)
                if abs(H_true - H) > 1e-6 * (1.0 + abs(H_true)):
                    H = H_true
        return H

    # -- Block-parallel vectorized update ---------------------------------

    def _block_loop(self, s, problem, rng, H, sweeps, energies, next_rec):
        cfg = self.config
        T = cfg.n_sweeps
        h = problem.h
        blocks, J_blocks = _get_coloring(problem)
        s_f = s.astype(np.float64)
        for sw in range(T):
            beta = schedule_beta(cfg.schedule_shape, sw, T, cfg.beta0, cfg.betaf)
            for b, Jb in zip(blocks, J_blocks):
                h_eff = np.asarray(Jb @ s_f).ravel() + h[b]
                s_cur = s_f[b]
                # idx=b lets device-aware backends apply per-spin
                # parameters (D2D dispersion) using the global
                # spin index of each entry in this colour class.
                # Backends that ignore idx (the ideal sigmoid,
                # Metropolis) accept the kwarg as a no-op.
                s_new = self.spin.sample_batch(h_eff, s_cur.astype(np.int8),
                                               beta, rng,
                                               idx=b).astype(np.float64)
                flipped = s_new != s_cur
                if np.any(flipped):
                    # dH = 2 * s_cur * h_eff on flipped sites
                    H += float(np.sum(2.0 * s_cur[flipped] * h_eff[flipped]))
                    s_f[b] = s_new
            while sw + 1 >= next_rec:
                sweeps.append(sw + 1)
                energies.append(H)
                next_rec = _next_rec(sw + 1, cfg.record, cfg.record_step)
            if (sw + 1) % cfg.energy_check_every == 0:
                H_true = problem.energy(s_f)
                if abs(H_true - H) > 1e-6 * (1.0 + abs(H_true)):
                    H = H_true
        s[:] = s_f.astype(np.int8)
        return H


def _next_rec(current: int, mode: str, step: int) -> float:
    if mode == "none":
        return float("inf")
    if mode == "linear":
        return current + step
    if current < 1:
        return 1
    order = 10 ** int(np.floor(np.log10(current)))
    m = current / order
    if m < 2:
        return int(2 * order)
    if m < 5:
        return int(5 * order)
    return int(10 * order)


# ===========================================================================
# Graph coloring for block-parallel mode
# ===========================================================================

def _get_coloring(problem: Problem) -> Tuple[List[np.ndarray], List[sp.csr_matrix]]:
    """Return (blocks, J_blocks) cached on the problem. J_blocks[k] is
    the CSR submatrix of rows in blocks[k], pre-extracted so the
    vectorized matmul inside the sweep loop has no overhead."""
    if "_coloring" not in problem.meta:
        blocks = _greedy_color(problem.J)
        J_blocks = [problem.J[b] for b in blocks]
        problem.meta["_coloring"] = (blocks, J_blocks)
    return problem.meta["_coloring"]


def _greedy_color(J):
    J = J.tocsr()
    n = J.shape[0]
    color = -np.ones(n, dtype=np.int32)
    for i in range(n):
        nbrs = J.indices[J.indptr[i]:J.indptr[i + 1]]
        used = set()
        for j in nbrs:
            if j != i and color[j] >= 0:
                used.add(int(color[j]))
        c = 0
        while c in used:
            c += 1
        color[i] = c
    k = int(color.max()) + 1
    return [np.where(color == c)[0] for c in range(k)]


# ===========================================================================
# Multi-start parallel runner with progress reporting
# ===========================================================================

def _trial_worker(args):
    child_ss, problem, solver_config, spin_spec, entropy = args
    rng = np.random.default_rng(child_ss)
    spin = _rebuild_spin(spin_spec)
    solver = IsingSolver(spin, solver_config)
    return solver.run(problem, rng, seed_hint=entropy)


def multistart(problem: Problem, solver_config: SolverConfig,
               spin_spec=("ideal", {}),
               n_trials: int = 100,
               master_seed: int = 2024,
               n_jobs: int = 1,
               progress: bool = True) -> List[RunResult]:
    """Run `n_trials` independent trials with derived seeds.

    Prints per-trial progress to stdout when `progress=True`. Uses
    tqdm if available, falls back to periodic log messages otherwise.
    """
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_trials)
    entropies = [int(c.entropy % (2**63 - 1)) for c in children]
    args = [(children[i], problem, solver_config, spin_spec, entropies[i])
            for i in range(n_trials)]

    log.info(f"multistart: problem={problem.name!r}, n_trials={n_trials}, "
             f"n_jobs={n_jobs}, mode={solver_config.update_mode}, "
             f"dynamics={solver_config.dynamics}, "
             f"sweeps={solver_config.n_sweeps}, "
             f"beta0={solver_config.beta0}, betaf={solver_config.betaf}")

    results = []
    t_start = time.perf_counter()

    use_tqdm = False
    if progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            pass

    if n_jobs <= 1:
        iterator = args
        if use_tqdm:
            iterator = tqdm(args, desc=f"  {problem.name}", ncols=80,
                            unit="trial")
        for idx, a in enumerate(iterator):
            r = _trial_worker(a)
            results.append(r)
            if not use_tqdm and progress and (idx + 1) % max(1, n_trials // 10) == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed * (n_trials - idx - 1) / (idx + 1)
                ef = np.array([r.energy_final for r in results])
                log.info(f"  {problem.name}: {idx+1}/{n_trials} "
                         f"min={ef.min():.4f} median={np.median(ef):.4f} "
                         f"elapsed={elapsed:.1f}s eta={eta:.1f}s")
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            if use_tqdm:
                from tqdm import tqdm
                for r in tqdm(pool.imap_unordered(_trial_worker, args),
                              total=n_trials, desc=f"  {problem.name}",
                              ncols=80, unit="trial"):
                    results.append(r)
            else:
                chunksize = max(1, n_trials // (4 * n_jobs))
                for idx, r in enumerate(pool.imap_unordered(
                        _trial_worker, args, chunksize=chunksize)):
                    results.append(r)
                    if progress and (idx + 1) % max(1, n_trials // 10) == 0:
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed * (n_trials - idx - 1) / (idx + 1)
                        ef = np.array([r.energy_final for r in results])
                        log.info(f"  {problem.name}: {idx+1}/{n_trials} "
                                 f"min={ef.min():.4f} "
                                 f"median={np.median(ef):.4f} "
                                 f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    total = time.perf_counter() - t_start
    ef = np.array([r.energy_final for r in results])
    log.info(f"multistart done: problem={problem.name!r} "
             f"total_wall={total:.1f}s "
             f"per_trial_median={np.median([r.wall_time for r in results]):.3f}s "
             f"E_min={ef.min():.4f} E_median={np.median(ef):.4f}")
    return results


# ===========================================================================
# Metrics (Section 3.4.1)
# ===========================================================================

def p_success(energies, target: float,
              atol: float = 1e-6, rtol: float = 0.0,
              sense: str = "min") -> float:
    """Fraction of trials reaching target within tolerance."""
    energies = np.asarray(energies, dtype=np.float64)
    tol = max(atol, rtol * abs(target))
    if sense == "min":
        return float(np.mean(energies <= target + tol))
    return float(np.mean(energies >= target - tol))


def tts_at_confidence(t_single: float, p_s: float,
                      confidence: float = 0.99) -> float:
    """TTS_{conf} = t_single * log(1 - conf) / log(1 - p_s)."""
    if p_s <= 0:
        return float("inf")
    if p_s >= 1:
        return float(t_single)
    return float(t_single * np.log(1.0 - confidence) / np.log(1.0 - p_s))


def residual_energy(energies, target: float) -> np.ndarray:
    energies = np.asarray(energies, dtype=np.float64)
    denom = abs(target) if target != 0 else 1.0
    return (energies - target) / denom


def cut_value_from_energy(H: float, edge_weight_sum: float) -> float:
    """For Max-Cut with J_ij = -w_ij/2 and h = 0, cut = W/2 - H."""
    return edge_weight_sum / 2.0 - H


def summarize_runs(results: List[RunResult], target: float,
                   sense: str = "min") -> dict:
    energies = np.array([r.energy_final for r in results])
    times = np.array([r.wall_time for r in results])
    sweeps = np.array([r.sweeps[-1] if len(r.sweeps) else 0 for r in results])
    ps = p_success(energies, target, sense=sense)
    t_med = float(np.median(times))
    tts = tts_at_confidence(t_med, ps, confidence=0.99)
    res = residual_energy(energies, target)
    return {
        "n_trials": len(results),
        "target": target,
        "p_success": ps,
        "tts_99_wall": tts,
        "tts_99_sweeps": tts_at_confidence(float(np.median(sweeps)), ps, 0.99),
        "time_median": t_med,
        "time_mean": float(times.mean()),
        "energy_min": float(energies.min()),
        "energy_median": float(np.median(energies)),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "residual_median": float(np.median(res)),
        "residual_min": float(res.min()),
    }


# ===========================================================================
# Results I/O: JSON (readable, portable)
# ===========================================================================

def save_results_json(results: List[RunResult], path, extras: dict = None):
    """Save a list of RunResult to a JSON file. Human-readable and
    directly inspectable in any text editor."""
    path = Path(path)
    payload = {
        "meta": extras or {},
        "runs": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_results_json(path) -> dict:
    """Inverse of save_results_json. Returns the raw payload dict."""
    with open(path, "r") as f:
        return json.load(f)


def save_final_states(results: List[RunResult], path):
    """Save only the final spin states as a compact text matrix,
    one row per trial (+1/-1 characters). Easy to diff in git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write("".join("1" if x > 0 else "0" for x in r.state_final))
            f.write("\n")
