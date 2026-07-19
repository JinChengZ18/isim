#!/usr/bin/env python3
"""RX-02 — parallel-update semantics vs the 表3.8 N_par=64 projection assumption.

Three update schedules on the shared G-set protocol (T = 10000 sweeps,
geometric beta 0.1 -> 10.0 indexed per sweep t in 0..T-1 exactly as
isim.schedule_beta does it, master_seed 2024, per-trial seeds via
np.random.SeedSequence(2024).spawn(N), per-trial random init in {-1,+1}^n):

  async     strict sequential single-spin Gibbs via isim.multistart with
            update_mode='async_numba' — the trusted path behind every
            Section 3.3 table. For G22 the RX-01 N=2000 baseline
            (results_rerun/results_compare_maxcut_N2000, 8/2000 hits) is
            REUSED for the summary row; a verification re-run checks that
            this driver's wiring reproduces it trial-for-trial.
  chromatic greedy colour classes of the instance graph, computed by the
            SAME algorithm isim's 'block' mode uses (isim._greedy_color is
            imported, not re-implemented). Per sweep the classes are visited
            in a per-trial-rng-permuted order and each class is sampled
            simultaneously from the field seen at class-visit time. Classes
            are independent sets, so this is correct Gibbs; the class-size
            distribution is the honest achievable parallel width.
  fixed64   the 表3.8 N_par=64 assumption made literal: ceil(n/64) fixed
            contiguous 64-spin blocks ignoring adjacency, permuted visit
            order per sweep, the whole block sampled simultaneously from the
            PRE-update field (parallel Glauber inside a block; breaks
            detailed balance whenever a block contains an edge).

Engine: a per-sweep numba kernel (default) plus a pure-NumPy reference path
sharing the identical RNG consumption order (init choice, then per sweep:
permutation(n_blocks) followed by random(n) uniforms consumed class-by-class
in visit order); --selftest-only cross-checks the two engines bitwise on
truncated runs. Multiprocessing over trials (Windows spawn: the __main__
guard below is mandatory).

Run:
  python eda/interface/run_parallel_semantics.py --selftest-only
  python eda/interface/run_parallel_semantics.py --probe-only
  python eda/interface/run_parallel_semantics.py [--jobs 16]

Outputs (eda/interface/):
  parallel_semantics_summary.csv    one row per (instance, schedule)
  parallel_semantics_summary.json   config header, seeds, Wilson + TTS-ratio
                                    CIs, G22 async-reuse note, runtimes
  results_parallel_semantics/       per-(instance, schedule) per-trial final
                                    energies (resumable cache; integrity)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isim import (SolverConfig, multistart, schedule_beta,   # noqa: E402
                  _greedy_color, get_logger)
from problems import load_gset                               # noqa: E402
from stats import wilson, tts_ratio_ci                       # noqa: E402

log = get_logger("rx02")

INSTANCES = ["G1", "G14", "G22"]
BKS = {"G1": 11624, "G14": 3064, "G22": 13359}
N_TRIALS = {"G1": 200, "G14": 200, "G22": 2000}
T_SWEEPS = 10000
BETA0, BETAF = 0.1, 10.0
SHAPE = "geometric"
MASTER_SEED = 2024
TOL = 1e-6
FIXED_WIDTH = 64
SCHEDULES = ["async", "chromatic", "fixed64"]

RX01_DIR = ROOT / "results_rerun" / "results_compare_maxcut_N2000"
ODIR = HERE / "results_parallel_semantics"
CSV_OUT = HERE / "parallel_semantics_summary.csv"
JSON_OUT = HERE / "parallel_semantics_summary.json"

# Canonical async G1 anchor (results_rerun/results_compare_maxcut, RX-01
# protocol): p_s = 0.685 at N = 200. The chromatic arm must land nearby.
G1_ASYNC_CANON = 0.685
G1_BAND = (0.60, 0.80)


# ===========================================================================
# Block constructions
# ===========================================================================

def chromatic_blocks(problem):
    """Greedy colour classes — identical colouring to isim block mode."""
    return [np.asarray(b, dtype=np.int64) for b in _greedy_color(problem.J)]


def fixed_blocks(n, width=FIXED_WIDTH):
    """ceil(n/width) contiguous blocks ignoring adjacency (表3.8 premise)."""
    return [np.arange(i, min(i + width, n), dtype=np.int64)
            for i in range(0, n, width)]


# ===========================================================================
# Trial payload + engines
# ===========================================================================

def build_payload(problem, blocks, engine, T=T_SWEEPS):
    J = problem.J.tocsr()
    payload = dict(
        engine=engine, n=problem.n, T=T, beta0=BETA0, betaf=BETAF,
        shape=SHAPE, J=J, h=problem.h.astype(np.float64),
        blocks=blocks,
        bidx=np.concatenate(blocks).astype(np.int64),
        bptr=np.cumsum([0] + [b.size for b in blocks]).astype(np.int64),
        max_block=int(max(b.size for b in blocks)),
        indptr=J.indptr.astype(np.int64),
        indices=J.indices.astype(np.int64),
        data=J.data.astype(np.float64),
    )
    if engine == "numpy":
        payload["J_blocks"] = [J[b] for b in blocks]
    return payload


_KERNEL = None


def _get_kernel():
    """Lazily JIT-compile the per-sweep block-update kernel.

    Semantics per class (in the caller-supplied visit order): pass 1 computes
    h_eff for every member from the CURRENT state (pre-update within the
    class); pass 2 samples all members simultaneously via
    p = 0.5*(1+tanh(beta*h_eff)) — the same stable sigmoid formula as
    isim._sigmoid_np(2*beta*h_eff). No fastmath: accumulation order matches
    scipy's CSR matvec so the NumPy reference path is bitwise reproducible.
    """
    global _KERNEL
    if _KERNEL is None:
        from numba import njit

        @njit(cache=True)
        def _sweep(s, indptr, indices, data, h, beta, bidx, bptr, order,
                   u_all, buf):
            off = 0
            for k in range(order.shape[0]):
                bi = order[k]
                start = bptr[bi]
                end = bptr[bi + 1]
                m = end - start
                for t in range(m):
                    i = bidx[start + t]
                    acc = 0.0
                    for kk in range(indptr[i], indptr[i + 1]):
                        acc += data[kk] * s[indices[kk]]
                    buf[t] = acc + h[i]
                for t in range(m):
                    i = bidx[start + t]
                    p = 0.5 * (1.0 + np.tanh(beta * buf[t]))
                    if u_all[off + t] < p:
                        s[i] = 1.0
                    else:
                        s[i] = -1.0
                off += m

        _KERNEL = _sweep
    return _KERNEL


def run_trial(child_ss, P, return_state=False):
    """One trial of the chromatic/fixed64 machinery.

    RNG consumption order (identical for both engines): init choice(n),
    then per sweep permutation(n_blocks) + random(n) uniforms consumed
    class-by-class in visit order.
    """
    rng = np.random.default_rng(child_ss)
    n, T = P["n"], P["T"]
    s = rng.choice([-1, 1], size=n).astype(np.float64)
    blocks = P["blocks"]
    nb = len(blocks)
    if P["engine"] == "numba":
        kern = _get_kernel()
        indptr, indices, data = P["indptr"], P["indices"], P["data"]
        h, bidx, bptr = P["h"], P["bidx"], P["bptr"]
        buf = np.empty(P["max_block"], dtype=np.float64)
        for sw in range(T):
            beta = schedule_beta(P["shape"], sw, T, P["beta0"], P["betaf"])
            order = rng.permutation(nb).astype(np.int64)
            u_all = rng.random(n)
            kern(s, indptr, indices, data, h, float(beta), bidx, bptr,
                 order, u_all, buf)
    else:
        J_blocks, h = P["J_blocks"], P["h"]
        for sw in range(T):
            beta = schedule_beta(P["shape"], sw, T, P["beta0"], P["betaf"])
            order = rng.permutation(nb)
            u_all = rng.random(n)
            off = 0
            for bi in order:
                b = blocks[bi]
                h_eff = np.asarray(J_blocks[bi] @ s).ravel() + h[b]
                p = 0.5 * (1.0 + np.tanh(beta * h_eff))
                s[b] = np.where(u_all[off:off + b.size] < p, 1.0, -1.0)
                off += b.size
    E = float(-0.5 * s @ (P["J"] @ s) - P["h"] @ s)
    if return_state:
        return E, s
    return E


# -- multiprocessing scaffolding (Windows spawn-safe) -----------------------

_W = {}


def _init_worker(payload):
    _W.clear()
    _W.update(payload)
    if payload["engine"] == "numba":
        _get_kernel()


def _pool_trial(args):
    idx, child_ss = args
    return idx, run_trial(child_ss, _W)


def run_custom(problem, schedule, n_trials, jobs, engine):
    blocks = (chromatic_blocks(problem) if schedule == "chromatic"
              else fixed_blocks(problem.n))
    payload = build_payload(problem, blocks, engine)
    children = np.random.SeedSequence(MASTER_SEED).spawn(n_trials)
    energies = np.empty(n_trials)
    log.info(f"{problem.name}/{schedule}: n_trials={n_trials} "
             f"n_blocks={len(blocks)} engine={engine} jobs={jobs}")
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_worker(payload)
        for i in range(n_trials):
            energies[i] = run_trial(children[i], payload)
            if (i + 1) % max(1, n_trials // 10) == 0:
                el = time.perf_counter() - t0
                log.info(f"  {i+1}/{n_trials} elapsed={el:.0f}s "
                         f"eta={el*(n_trials-i-1)/(i+1):.0f}s")
    else:
        from multiprocessing import Pool
        chunk = max(1, n_trials // (jobs * 20))
        with Pool(jobs, initializer=_init_worker,
                  initargs=(payload,)) as pool:
            done = 0
            for idx, E in pool.imap_unordered(
                    _pool_trial, list(enumerate(children)), chunksize=chunk):
                energies[idx] = E
                done += 1
                if done % max(1, n_trials // 10) == 0:
                    el = time.perf_counter() - t0
                    log.info(f"  {done}/{n_trials} elapsed={el:.0f}s "
                             f"eta={el*(n_trials-done)/done:.0f}s")
    wall = time.perf_counter() - t0
    sizes = [int(b.size) for b in blocks]
    block_stats = dict(n_classes=len(sizes),
                       mean_class_size=float(np.mean(sizes)),
                       max_class_size=int(max(sizes)),
                       min_class_size=int(min(sizes)),
                       class_sizes=sizes)
    return energies, wall, block_stats


def run_async(problem, n_trials, jobs):
    cfg = SolverConfig(schedule_shape=SHAPE, beta0=BETA0, betaf=BETAF,
                       n_sweeps=T_SWEEPS, update_mode="async_numba",
                       dynamics="gibbs")
    t0 = time.perf_counter()
    res = multistart(problem, cfg, ("ideal", {}), n_trials=n_trials,
                     master_seed=MASTER_SEED, n_jobs=jobs)
    wall = time.perf_counter() - t0
    return np.array([r.energy_final for r in res]), wall


def run_cached(problem, schedule, n_trials, jobs, engine, force=False):
    ODIR.mkdir(exist_ok=True)
    key = f"{problem.name}_{schedule}_N{n_trials}"
    fp = ODIR / f"{key}.json"
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("n_trials") == n_trials:
            log.info(f"[cache] {key} <- {fp.name} (wall {d['wall_s']:.0f}s)")
            return d
    if schedule == "async":
        energies, wall = run_async(problem, n_trials, jobs)
        block_stats = None
        eng = "isim_async_numba"
    else:
        energies, wall, block_stats = run_custom(problem, schedule, n_trials,
                                                 jobs, engine)
        eng = engine
    d = dict(instance=problem.name, schedule=schedule, n_trials=n_trials,
             engine=eng, T=T_SWEEPS, beta=[BETA0, BETAF], shape=SHAPE,
             master_seed=MASTER_SEED,
             seed_derivation=("np.random.SeedSequence(2024).spawn(n_trials)"
                              "[i] -> np.random.default_rng per trial"),
             wall_s=wall, block_stats=block_stats,
             energies=[float(e) for e in energies])
    fp.write_text(json.dumps(d))
    log.info(f"[saved] {fp.name} (wall {wall:.0f}s)")
    return d


# ===========================================================================
# Statistics / summary
# ===========================================================================

def summarize(energies, target, n_trials):
    e = np.asarray(energies, dtype=np.float64)
    hits = int(np.sum(e <= target + TOL))
    p = hits / n_trials
    lo, hi = wilson(hits, n_trials)
    if p <= 0:
        tts = float("inf")
    elif p >= 1:
        tts = float(T_SWEEPS)
    else:
        tts = float(T_SWEEPS * np.log(0.01) / np.log(1.0 - p))
    return dict(hits=hits, p_s=p, wilson_lo=lo, wilson_hi=hi,
                tts99_sweeps=tts, energy_min=float(e.min()),
                energy_median=float(np.median(e)))


# ===========================================================================
# Validation stages
# ===========================================================================

def selftest(problems):
    """Cross-check numba vs NumPy engines bitwise on truncated runs."""
    cases = [("G14", "chromatic"), ("G14", "fixed64"), ("G22", "fixed64")]
    report = []
    for name, sched in cases:
        prob, _ = problems[name]
        blocks = (chromatic_blocks(prob) if sched == "chromatic"
                  else fixed_blocks(prob.n))
        out = {}
        for engine in ("numba", "numpy"):
            P = build_payload(prob, blocks, engine, T=300)
            children = np.random.SeedSequence(MASTER_SEED).spawn(2)
            out[engine] = [run_trial(c, P, return_state=True)
                           for c in children]
        ok = all(a[0] == b[0] and np.array_equal(a[1], b[1])
                 for a, b in zip(out["numba"], out["numpy"]))
        report.append(dict(instance=name, schedule=sched, T=300, trials=2,
                           bitwise_equal=bool(ok),
                           energies_numba=[float(x[0]) for x in out["numba"]],
                           energies_numpy=[float(x[0]) for x in out["numpy"]]))
        log.info(f"selftest {name}/{sched}: bitwise_equal={ok} "
                 f"E={[float(x[0]) for x in out['numba']]}")
        if not ok:
            raise RuntimeError(
                f"selftest FAILED for {name}/{sched}: numba and numpy "
                f"engines disagree — do not run the grid until resolved")
    return report


def probe(problems):
    """Time 2 single-process numba trials per heavy case; extrapolate."""
    _get_kernel()
    est = {}
    for name, sched in [("G22", "chromatic"), ("G22", "fixed64"),
                        ("G1", "chromatic")]:
        prob, _ = problems[name]
        blocks = (chromatic_blocks(prob) if sched == "chromatic"
                  else fixed_blocks(prob.n))
        P = build_payload(prob, blocks, "numba")
        children = np.random.SeedSequence(999).spawn(2)
        t0 = time.perf_counter()
        for c in children:
            run_trial(c, P)
        per = (time.perf_counter() - t0) / 2
        est[f"{name}_{sched}"] = per
        log.info(f"probe {name}/{sched}: {per:.2f} s/trial "
                 f"(n_blocks={len(blocks)})")
    return est


# ===========================================================================
# Main
# ===========================================================================

def load_instances():
    gdir = ROOT / "gset"
    try:
        from fetch_data import ensure_gset
        ensure_gset(INSTANCES, gdir)
    except Exception as err:
        log.warning(f"auto-fetch warning: {err}")
    out = {}
    for name in INSTANCES:
        prob = load_gset(gdir / name)
        target = prob.meta["edge_sum"] / 2.0 - BKS[name]
        log.info(f"{name}: n={prob.n} edge_sum={prob.meta['edge_sum']:.0f} "
                 f"BKS={BKS[name]} target E={target:.1f}")
        out[name] = (prob, target)
    return out


def read_rx01_g22():
    """RX-01 N=2000 async-Gibbs baseline (reused for the G22 async row)."""
    with open(RX01_DIR / "compare_G22_smtj.json") as f:
        payload = json.load(f)
    energies = np.array([r["energy_final"] for r in payload["runs"]])
    return energies, payload["meta"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--engine", choices=["numba", "numpy"], default="numba")
    ap.add_argument("--g22-trials", type=int, default=N_TRIALS["G22"])
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--skip-g22-async-rerun", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="continue past the G1 chromatic validation gate")
    args = ap.parse_args(argv)

    t_total0 = time.perf_counter()
    problems = load_instances()
    n_g22 = args.g22_trials

    selftest_report = None
    if args.selftest_only:
        selftest(problems)
        return
    if args.probe_only:
        probe(problems)
        return
    if not args.skip_selftest:
        selftest_report = selftest(problems)

    probe_est = probe(problems)
    grid_est = (probe_est["G22_chromatic"] * n_g22
                + probe_est["G22_fixed64"] * n_g22
                + probe_est["G1_chromatic"] * 2 * N_TRIALS["G1"]
                + probe_est["G1_chromatic"] * 2 * N_TRIALS["G14"] * 0.35
                ) / max(1, args.jobs)
    log.info(f"grid estimate (custom schedules, jobs={args.jobs}): "
             f"~{grid_est/60:.1f} min + async arms")

    runs = {}

    # --- Stage 1: G1 async + chromatic, validation gate -------------------
    probG1, tgtG1 = problems["G1"]
    runs[("G1", "async")] = run_cached(probG1, "async", N_TRIALS["G1"],
                                       args.jobs, args.engine,
                                       args.force_recompute)
    runs[("G1", "chromatic")] = run_cached(probG1, "chromatic",
                                           N_TRIALS["G1"], args.jobs,
                                           args.engine, args.force_recompute)
    sA = summarize(runs[("G1", "async")]["energies"], tgtG1, N_TRIALS["G1"])
    sB = summarize(runs[("G1", "chromatic")]["energies"], tgtG1,
                   N_TRIALS["G1"])
    log.info(f"gate: G1 async p_s={sA['p_s']:.3f} (canon {G1_ASYNC_CANON}), "
             f"chromatic p_s={sB['p_s']:.3f} (band {G1_BAND})")
    if abs(sA["p_s"] - G1_ASYNC_CANON) > 1e-9:
        log.warning("G1 async did NOT exactly reproduce the canonical "
                    "0.685 — same seeds/kernel expected bitwise repro; "
                    "investigate before trusting the wiring")
    gate_ok = (G1_BAND[0] <= sB["p_s"] <= G1_BAND[1]
               and abs(sB["p_s"] - sA["p_s"]) < 0.12)
    if not gate_ok and not args.force:
        log.error(f"VALIDATION GATE FAILED: G1 chromatic p_s={sB['p_s']:.3f} "
                  f"vs async {sA['p_s']:.3f} — suspect sweep semantics "
                  f"(beta indexing t=0..T-1, class independence). Aborting "
                  f"before the big grid; rerun with --force to override.")
        sys.exit(1)

    # --- Stage 2: remaining cheap arms ------------------------------------
    probG14, tgtG14 = problems["G14"]
    runs[("G14", "async")] = run_cached(probG14, "async", N_TRIALS["G14"],
                                        args.jobs, args.engine,
                                        args.force_recompute)
    runs[("G14", "chromatic")] = run_cached(probG14, "chromatic",
                                            N_TRIALS["G14"], args.jobs,
                                            args.engine, args.force_recompute)
    runs[("G14", "fixed64")] = run_cached(probG14, "fixed64",
                                          N_TRIALS["G14"], args.jobs,
                                          args.engine, args.force_recompute)
    runs[("G1", "fixed64")] = run_cached(probG1, "fixed64", N_TRIALS["G1"],
                                         args.jobs, args.engine,
                                         args.force_recompute)

    # --- Stage 3: G22 ------------------------------------------------------
    probG22, tgtG22 = problems["G22"]
    rx01_energies, rx01_meta = read_rx01_g22()
    reuse_note = dict(
        source=str(RX01_DIR / "compare_G22_smtj.json"),
        note=("G22 async row REUSES the RX-01 N=2000 async_numba Gibbs "
              "baseline (8/2000 hits, p_s=0.004)"),
        rx01_n_trials=int(len(rx01_energies)),
        rx01_hits=int(np.sum(rx01_energies <= tgtG22 + TOL)),
        verification_rerun=None)
    if not args.skip_g22_async_rerun:
        d = run_cached(probG22, "async", n_g22, args.jobs, args.engine,
                       args.force_recompute)
        rerun_sorted = np.sort(np.asarray(d["energies"]))
        rx01_sorted = np.sort(rx01_energies[:n_g22])
        exact = bool(np.array_equal(rerun_sorted, rx01_sorted))
        reuse_note["verification_rerun"] = dict(
            n_trials=n_g22, exact_reproduction=exact,
            hits=int(np.sum(np.asarray(d["energies"]) <= tgtG22 + TOL)),
            max_abs_diff_sorted=float(np.max(np.abs(rerun_sorted
                                                    - rx01_sorted))))
        log.info(f"G22 async verification re-run: exact_reproduction="
                 f"{exact}")
        if not exact:
            log.warning("G22 async re-run did NOT reproduce RX-01 "
                        "trial-for-trial — wiring or environment drift; "
                        "REPORT this, do not silently proceed")
    runs[("G22", "async")] = dict(
        instance="G22", schedule="async", n_trials=int(len(rx01_energies)),
        engine="isim_async_numba (RX-01 reuse)", wall_s=None,
        block_stats=None, energies=[float(e) for e in rx01_energies])
    runs[("G22", "chromatic")] = run_cached(probG22, "chromatic", n_g22,
                                            args.jobs, args.engine,
                                            args.force_recompute)
    runs[("G22", "fixed64")] = run_cached(probG22, "fixed64", n_g22,
                                          args.jobs, args.engine,
                                          args.force_recompute)

    # --- Summaries + CIs ---------------------------------------------------
    targets = {k: v[1] for k, v in problems.items()}
    summaries = {}
    rows = []
    for inst in INSTANCES:
        for sched in SCHEDULES:
            d = runs[(inst, sched)]
            n_tr = d["n_trials"]
            s = summarize(d["energies"], targets[inst], n_tr)
            summaries[(inst, sched)] = (s, n_tr)
            bs = d.get("block_stats") or {}
            rows.append(dict(
                instance=inst, schedule=sched, n_trials=n_tr,
                hits=s["hits"], p_s=s["p_s"],
                wilson_lo=round(s["wilson_lo"], 6),
                wilson_hi=round(s["wilson_hi"], 6),
                tts99_sweeps=(round(s["tts99_sweeps"], 1)
                              if np.isfinite(s["tts99_sweeps"]) else "inf"),
                n_classes=bs.get("n_classes", ""),
                mean_class_size=(round(bs["mean_class_size"], 2)
                                 if bs else ""),
                max_class_size=bs.get("max_class_size", ""),
                energy_min=s["energy_min"],
                energy_median=s["energy_median"],
                source=("reused-RX01" if (inst, sched) == ("G22", "async")
                        else "this-run"),
            ))
            log.info(f"[{inst:>4s} {sched:>9s}] hits={s['hits']:>4d}/{n_tr} "
                     f"p_s={s['p_s']:.4f} "
                     f"CI=[{s['wilson_lo']:.4f},{s['wilson_hi']:.4f}] "
                     f"E_med={s['energy_median']:.1f}")

    ratio_cis = []
    for inst in INSTANCES:
        sa, na = summaries[(inst, "async")]
        for sched in ("chromatic", "fixed64"):
            sx, nx = summaries[(inst, sched)]
            r = tts_ratio_ci(sa["hits"], na, sx["hits"], nx,
                             n_sweeps=T_SWEEPS)
            ratio_cis.append(dict(
                instance=inst, comparison=f"{sched}_vs_async",
                ratio=r["ratio"], lo=r["lo"], hi=r["hi"],
                frac_undefined=r["frac_undefined"],
                note=(f"ratio = TTS99({sched})/TTS99(async) in sweeps; "
                      f">1 means {sched} slower than async")))
            log.info(f"ratio {inst} {sched}/async: {r['ratio']:.3f} "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}] "
                     f"undef={r['frac_undefined']:.3f}")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    header = dict(
        _label=("RX-02 parallel-update semantics: async vs chromatic "
                "(correct Gibbs, honest width) vs fixed64 (表3.8 N_par=64 "
                "assumption, parallel Glauber inside blocks)"),
        protocol=dict(T_sweeps=T_SWEEPS, schedule=SHAPE,
                      beta=[BETA0, BETAF],
                      beta_indexing="schedule_beta(shape, t, T, b0, bf), "
                                    "t = 0..T-1 (isim convention)",
                      master_seed=MASTER_SEED,
                      seed_derivation=("np.random.SeedSequence(2024)."
                                       "spawn(N)[i] per trial"),
                      init="rng.choice([-1,+1], size=n) per trial",
                      target="edge_sum/2 - BKS_cut, tol 1e-6",
                      bks=BKS, n_trials=dict(G1=N_TRIALS["G1"],
                                             G14=N_TRIALS["G14"],
                                             G22=n_g22)),
        engine=args.engine, jobs=args.jobs,
        versions=_versions(),
        instances={
            inst: dict(n=problems[inst][0].n,
                       edge_sum=float(problems[inst][0].meta["edge_sum"]),
                       target=float(targets[inst]),
                       chromatic_blocks=(runs[(inst, "chromatic")]
                                         ["block_stats"]),
                       fixed64_blocks=(runs[(inst, "fixed64")]
                                       ["block_stats"]))
            for inst in INSTANCES},
        g22_async_reuse=reuse_note,
        selftest=selftest_report,
        probe_s_per_trial=probe_est,
        validation_gate=dict(g1_async_ps=sA["p_s"],
                             g1_chromatic_ps=sB["p_s"],
                             canonical_band=list(G1_BAND), passed=gate_ok),
        tts_ratio_cis=ratio_cis,
        wall_s={f"{i}_{s}": runs[(i, s)]["wall_s"]
                for i in INSTANCES for s in SCHEDULES},
        total_wall_s=time.perf_counter() - t_total0,
    )
    JSON_OUT.write_text(json.dumps(header, indent=2))
    log.info(f"-> {CSV_OUT}")
    log.info(f"-> {JSON_OUT}")
    log.info(f"total wall: {(time.perf_counter()-t_total0)/60:.1f} min")


def _versions():
    import numba
    import scipy
    return dict(numpy=np.__version__, scipy=scipy.__version__,
                numba=numba.__version__,
                python=sys.version.split()[0])


if __name__ == "__main__":
    main()
