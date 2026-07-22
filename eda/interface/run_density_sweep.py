#!/usr/bin/env python3
"""RX-11 -- does the Gibbs-vs-Metropolis ordering depend on connectivity?

Section 3.3.1/3.6 assert that the single-spin update rule buys no algorithmic
speedup: for the same energy difference the two rules obey
min(1, e^-x) >= sigma(-x), so the Metropolis single-step transition
probability is pointwise at least the heat-bath one (Peskun ordering), and the
measured G-set ratios agree (G1 SA faster at 0.84x [0.75, 0.94]; G22
unresolved at 1.6x [0.5, 7.0]). That negative claim rests on real instances
whose connectivity, size and weight structure all move together, and only one
of which is statistically resolved. This driver puts it on a CONTROLLED axis:
one structural knob (mean degree) at fixed n, fixed weight law, fixed
annealing protocol.

Ladder: random d-regular graphs, n = 1000, mean degree in {6, 12, 24, 48, 96},
edge weights drawn uniformly from {-1, +1}. Max-Cut/spin-glass convention
identical to the G-set loader (J_ij = -w_ij/2, h = 0), so the beta schedule
carries the same physical meaning as in the G1/G22 runs. Fixed instance seed
per (degree, replicate): seed = SEED_BASE + degree + REP_STRIDE * rep, spawned
into a topology stream and a weight stream, recorded in the config JSON.
Replicate 0 is the pre-registered ladder; further replicates are independent
draws from the same ensemble and exist to separate a degree effect from a
single-instance accident.

Topology construction (no networkx dependency; the repo ships numpy/scipy/
numba only): start from the circulant graph C_n(1..d/2), which is d-regular
and simple by construction, then randomize with SWAP_FACTOR * m accepted-or-
rejected double edge swaps. Double edge swaps preserve every degree exactly,
so regularity is structural rather than hoped-for; it is asserted anyway,
together with simplicity, symmetry and connectivity. The triangle count is
recorded next to the random-d-regular expectation (d-1)^3/6 as the
randomization diagnostic: the circulant seed graph is heavily clustered, a
mixed one is not.

Targets: no BKS exists for these instances. The LONGRUN_BEST convention of
run_reset_mechanism.py is used -- an independent long reference run (T_REF
sweeps, N_REF trials PER DYNAMICS, seed REF_SEED), target = the best energy
over the pooled reference. Pooling both dynamics matters here: a target taken
from a Gibbs-only reference would be an energy level Gibbs is known to reach,
which is exactly the quantity under test. The target is NOT a certified ground
state and is labelled as such in every output.

Sweep budget: the G-set protocol (T = 10^4, geometric beta 0.1 -> 10) is the
default rung. It is kept unless a pilot at a SEPARATE master seed shows the
resulting hit rate outside the measurable band [0.05, 0.6]; the pre-registered
ladder {3e3, 1e4, 3e4, 1e5} is then walked in the indicated direction, the
selection statistic being the geometric mean of the two arms' pilot p_s (a
statistic symmetric in the two dynamics, so tuning cannot favour either). The
selected T is applied to BOTH arms of that degree and reported per degree
(csv column `rung` = selected). Any budget passed with --extra-T is run on the
same instance under the same target and reported alongside (rung =
robustness), which is what distinguishes an ordering from a budget artefact.

Main measurement: 500 trials per (degree, dynamics) arm, master_seed 2024,
per-trial seeds via np.random.SeedSequence(2024).spawn(500), update_mode
'async_numba' (both 'ideal' and 'metropolis' are JIT-safe, so the kernel
selected by SolverConfig.dynamics is the measured object and the spin_spec is
its block-mode twin, exactly as compare_baselines.py wires it). Every p_s
carries a Wilson 95% interval and every TTS ratio a parametric-bootstrap 95%
interval (eda/interface/stats.py, the RX-01 convention).

Mechanism variable: for each degree the empirical distribution of |h_eff|
under uniformly random configurations (median and variance over N_HEFF random
configurations x n spins) is recorded, so the result can be read against the
quantity the chapter names rather than against degree alone.

Run (Windows python; the __main__ guard is mandatory for n_jobs > 1). The
committed result files come from the second form; per-arm results are cached,
so re-running reproduces the csv byte-for-byte without re-solving:
  python eda/interface/run_density_sweep.py --selftest-only
  python eda/interface/run_density_sweep.py --jobs 24 --reps 2 --extra-T 30000

Outputs (eda/interface/):
  density_sweep_summary.csv      one row per (degree, replicate, rung, dynamics)
  density_sweep_config.json      instances, seeds, targets, T selection trace
  results_density_sweep/         per-arm per-trial final energies (cache)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isim import (Problem, SolverConfig, multistart, preprocess,   # noqa: E402
                  p_success, tts_at_confidence, get_logger)
from stats import wilson, tts_ratio_ci                             # noqa: E402

log = get_logger("rx11")

N_SPINS = 1000
DEGREES = [6, 12, 24, 48, 96]
SEED_BASE = 700                      # instance seed = SEED_BASE + degree
REP_STRIDE = 10000                   #   + REP_STRIDE * replicate index
SWAP_FACTOR = 10                     # double-edge-swap attempts per edge

SHAPE, BETA0, BETAF = "geometric", 0.1, 10.0
T_DEFAULT = 10000                    # G-set protocol rung
T_LADDER = [3000, 10000, 30000, 100000]
BAND = (0.05, 0.60)                  # measurable p_s band (pre-registered)

N_TRIALS = 500
MASTER_SEED = 2024
PILOT_TRIALS, PILOT_SEED = 100, 7777
T_REF, N_REF, REF_SEED = 100000, 100, 20260723
N_HEFF, HEFF_SEED = 1000, 20260724
TOL = 1e-6

DYNAMICS = [("gibbs", ("ideal", {})), ("metropolis", ("metropolis", {}))]

ODIR = HERE / "results_density_sweep"
CSV_OUT = HERE / "density_sweep_summary.csv"
JSON_OUT = HERE / "density_sweep_config.json"


# ===========================================================================
# Instance construction
# ===========================================================================

def _circulant_edges(n, d):
    """C_n(1..d/2): d-regular and simple for even d < n."""
    half = d // 2
    return [(i, (i + k) % n) for i in range(n) for k in range(1, half + 1)]


def random_regular_edges(n, d, rng, swap_factor=SWAP_FACTOR):
    """Random d-regular simple graph by double-edge-swap randomization of the
    circulant seed graph. Returns (edges, accept_rate). Swaps preserve every
    vertex degree exactly; rejected swaps are those that would create a
    self-loop or a parallel edge."""
    if d % 2 or d >= n or (n * d) % 2:
        raise ValueError(f"need even d < n with n*d even; got n={n}, d={d}")
    edges = _circulant_edges(n, d)
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if j in adj[i]:
            raise ValueError("circulant seed is not simple")
        adj[i].add(j)
        adj[j].add(i)
    m = len(edges)
    if m != n * d // 2:
        raise ValueError(f"seed edge count {m} != {n * d // 2}")
    n_att = swap_factor * m
    pairs = rng.integers(0, m, size=(n_att, 2))
    flips = rng.random(n_att) < 0.5
    acc = 0
    for t in range(n_att):
        ia, ib = int(pairs[t, 0]), int(pairs[t, 1])
        if ia == ib:
            continue
        a, b = edges[ia]
        c, e = edges[ib]
        if flips[t]:
            c, e = e, c
        if len({a, b, c, e}) < 4 or e in adj[a] or b in adj[c]:
            continue
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(e); adj[e].discard(c)
        adj[a].add(e); adj[e].add(a)
        adj[c].add(b); adj[b].add(c)
        edges[ia] = (a, e)
        edges[ib] = (c, b)
        acc += 1
    return edges, acc / n_att


def build_instance(n, d, seed):
    """Signed d-regular Max-Cut instance. Weight law: w_ij uniform on {-1,+1};
    coupling convention J_ij = -w_ij/2 and h = 0, identical to
    problems.load_gset, so |J| = 0.5 as in G1/G22 and the annealing endpoints
    keep their meaning."""
    ss = np.random.SeedSequence(seed)
    topo_ss, wt_ss = ss.spawn(2)
    edges, arate = random_regular_edges(n, d, np.random.default_rng(topo_ss))
    w = np.random.default_rng(wt_ss).choice([-1.0, 1.0], size=len(edges))
    ei = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    ej = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.concatenate([-0.5 * w, -0.5 * w])
    J = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    h = np.zeros(n, dtype=np.float64)
    J, h = preprocess(J, h, normalize=False)
    prob = Problem(name=f"RR_n{n}_d{d}_s{seed}", n=n, J=J, h=h)

    # structural verification (regularity is asserted, never assumed)
    deg = np.diff(J.indptr)
    if not np.all(deg == d):
        raise AssertionError(f"degree spread {deg.min()}..{deg.max()} != {d}")
    if J.diagonal().any():
        raise AssertionError("self-loop present")
    if abs(J - J.T).nnz:
        raise AssertionError("J not symmetric")
    A = sp.csr_matrix((np.ones(J.nnz), J.indices, J.indptr), shape=J.shape)
    ncomp = sp.csgraph.connected_components(A, directed=False)[0]
    if ncomp != 1:
        raise AssertionError(f"graph has {ncomp} components")
    tri = float((A @ A).multiply(A).sum() / 6.0)
    n_pos = int((w > 0).sum())
    prob.meta.update(kind="maxcut", edge_sum=float(w.sum()),
                     rr_params=dict(n=n, degree=d, seed=int(seed),
                                    swap_factor=SWAP_FACTOR))
    info = dict(instance=prob.name, n=n, degree=d, seed=int(seed),
                n_edges=len(edges), swap_accept_rate=round(arate, 4),
                triangles=tri, triangles_random_expect=(d - 1) ** 3 / 6.0,
                n_edges_positive=n_pos, n_edges_negative=len(edges) - n_pos,
                connected=True)
    return prob, info


def heff_stats(problem, n_cfg=N_HEFF, seed=HEFF_SEED):
    """Empirical |h_eff| distribution under uniformly random configurations
    (h = 0, so h_eff = J s). Pools n_cfg configurations x n spins."""
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(problem.n, n_cfg))
    he = np.asarray(problem.J @ S)
    a = np.abs(he)
    d = int(np.diff(problem.J.indptr)[0])
    return dict(n_configs=n_cfg, seed=seed,
                heff_abs_median=float(np.median(a)),
                heff_abs_var=float(a.var()),
                heff_abs_mean=float(a.mean()),
                heff_var=float(he.var()),
                heff_var_analytic=0.25 * d,
                frac_zero_field=float(np.mean(a < 1e-12)))


# ===========================================================================
# Runs
# ===========================================================================

RNG_NOTE = ("per-trial stream = np.random.default_rng(SeedSequence("
            "master_seed).spawn(n_trials)[i]); numpy's spawned children share "
            ".entropy and differ only in .spawn_key, so RunResult.seed_hint is "
            "a degenerate label and is deliberately NOT recorded -- the "
            "reproducible identifier is (master_seed, n_trials, trial index). "
            "n_distinct_energies guards against the collapsed-stream failure "
            "mode recorded in .agents/TRIAL_LOG_eda.md (2026-07-20 RX-05a).")


def run_arm(problem, dyn, spec, T, trials, seed, jobs):
    cfg = SolverConfig(schedule_shape=SHAPE, beta0=BETA0, betaf=BETAF,
                       n_sweeps=T, update_mode="async_numba", dynamics=dyn,
                       record="none")
    cfg.validate()
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=trials, master_seed=seed, n_jobs=jobs,
                     progress=False)
    return (np.array([r.energy_final for r in res]),
            time.perf_counter() - t0)


def _payload(head, e, wall):
    n_dist = int(np.unique(e).size)
    if len(e) >= 10 and n_dist == 1:
        raise AssertionError(
            f"all {len(e)} trials of {head['instance']}/{head['dynamics']} "
            f"returned the identical energy -- collapsed RNG streams")
    return dict(config=head, wall_s=round(float(wall), 2),
                per_trial_rng=RNG_NOTE, n_distinct_energies=n_dist,
                energies=[float(x) for x in e])


def cached_arm(problem, dyn, spec, T, trials, seed, jobs, tag, refresh):
    """Persist per-trial final energies; reuse only on an exact config match.
    A cache file written before the metadata note existed is rewritten from its
    own stored energies (no re-solve, no measured number changes)."""
    ODIR.mkdir(exist_ok=True)
    f = ODIR / f"{tag}_{problem.name}_{dyn}_T{T}_N{trials}_s{seed}.json"
    head = dict(instance=problem.name, n=problem.n,
                degree=int(np.diff(problem.J.indptr)[0]), dynamics=dyn,
                spin_spec=spec[0], n_sweeps=T, n_trials=trials,
                master_seed=seed, schedule=SHAPE, beta0=BETA0, betaf=BETAF,
                update_mode="async_numba")
    if f.exists() and not refresh:
        d = json.loads(f.read_text())
        if all(d["config"].get(k) == v for k, v in head.items()):
            log.info(f"  [cache] {f.name}")
            e = np.array(d["energies"], dtype=float)
            if d.get("per_trial_rng") != RNG_NOTE:
                f.write_text(json.dumps(_payload(head, e, d["wall_s"]),
                                        indent=1))
            return e, d["wall_s"]
    e, wall = run_arm(problem, dyn, spec, T, trials, seed, jobs)
    f.write_text(json.dumps(_payload(head, e, wall), indent=1))
    return e, wall


def select_T(problem, target, jobs, refresh):
    """Pre-registered pilot ladder at PILOT_SEED (disjoint from the reporting
    seed): keep T_DEFAULT if the geometric mean of the two arms' pilot p_s is
    inside BAND, otherwise walk the ladder up (too few hits) or down (too
    many). Selection uses a statistic symmetric in the two dynamics."""
    trace, idx = [], T_LADDER.index(T_DEFAULT)
    direction = None
    while True:
        T = T_LADDER[idx]
        ps = {}
        for dyn, spec in DYNAMICS:
            e, _ = cached_arm(problem, dyn, spec, T, PILOT_TRIALS, PILOT_SEED,
                              jobs, "pilot", refresh)
            ps[dyn] = p_success(e, target, atol=TOL)
        g = float(np.sqrt(max(ps["gibbs"], 1e-9) * max(ps["metropolis"], 1e-9)))
        trace.append(dict(T=T, ps_gibbs=ps["gibbs"], ps_metropolis=
                          ps["metropolis"], geomean=round(g, 4)))
        log.info(f"  pilot T={T}: p_s gibbs={ps['gibbs']:.3f} "
                 f"metropolis={ps['metropolis']:.3f} geomean={g:.3f}")
        if BAND[0] <= g <= BAND[1]:
            return T, trace
        step = 1 if g < BAND[0] else -1
        if direction is not None and step != direction:
            log.info("  pilot ladder oscillated; keeping current rung")
            return T, trace
        direction = step
        nxt = idx + step
        if not 0 <= nxt < len(T_LADDER):
            log.info("  pilot ladder exhausted; keeping the end rung")
            return T, trace
        idx = nxt


# ===========================================================================
# Driver
# ===========================================================================

def selftest():
    """Construction invariants on a small instance, plus the two claims the
    summary rests on: the swap randomizer destroys the circulant clustering,
    and the two dynamics are actually different kernels."""
    prob, info = build_instance(200, 6, SEED_BASE + 6)
    assert np.all(np.diff(prob.J.indptr) == 6)
    assert abs(prob.J.data).max() == 0.5
    seed_edges = _circulant_edges(200, 6)
    A0 = sp.csr_matrix((np.ones(2 * len(seed_edges)),
                        (np.array([e[0] for e in seed_edges] +
                                  [e[1] for e in seed_edges]),
                         np.array([e[1] for e in seed_edges] +
                                  [e[0] for e in seed_edges]))),
                       shape=(200, 200))
    tri0 = float((A0 @ A0).multiply(A0).sum() / 6.0)
    print(f"selftest: n=200 d=6 triangles {tri0:.0f} (circulant) -> "
          f"{info['triangles']:.0f} (randomized), random expectation "
          f"{info['triangles_random_expect']:.0f}")
    assert info["triangles"] < 0.25 * tri0
    e_g, _ = run_arm(prob, "gibbs", ("ideal", {}), 200, 12, 5, 1)
    e_m, _ = run_arm(prob, "metropolis", ("metropolis", {}), 200, 12, 5, 1)
    assert not np.allclose(e_g, e_m), "the two kernels returned identical runs"
    # spawned children share .entropy but not their streams: a live check that
    # the 12 trials of one arm are not one run repeated (RX-05a failure mode)
    assert np.unique(e_g).size > 1 and np.unique(e_m).size > 1
    hs = heff_stats(prob, n_cfg=100)
    print(f"selftest: |h_eff| median {hs['heff_abs_median']:.3f}, "
          f"Var(h_eff) {hs['heff_var']:.3f} vs analytic "
          f"{hs['heff_var_analytic']:.3f}")
    assert abs(hs["heff_var"] - hs["heff_var_analytic"]) < 0.1
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--degrees", type=int, nargs="+", default=DEGREES)
    ap.add_argument("--reps", type=int, default=1,
                    help="independent instances per degree (rep 0 = the "
                         "pre-registered ladder)")
    ap.add_argument("--extra-T", type=int, nargs="*", default=[],
                    help="additional sweep budgets, run on replicate 0 only "
                         "and labelled rung=robustness")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the per-arm cache and re-solve")
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()

    if args.selftest_only:
        selftest()
        return
    selftest()

    ODIR.mkdir(exist_ok=True)
    rows, meta = [], {}
    for d in args.degrees:
      for rep in range(args.reps):
        seed = SEED_BASE + d + REP_STRIDE * rep
        t0 = time.perf_counter()
        prob, info = build_instance(N_SPINS, d, seed)
        info["build_s"] = round(time.perf_counter() - t0, 1)
        info["rep"] = rep
        hs = heff_stats(prob)
        log.info(f"[degree {d} rep {rep}] {prob.name}: {info['n_edges']} "
                 f"edges ({info['n_edges_negative']} negative), triangles "
                 f"{info['triangles']:.0f} vs random "
                 f"{info['triangles_random_expect']:.0f}; |h_eff| median "
                 f"{hs['heff_abs_median']:.3f} var {hs['heff_abs_var']:.3f}")

        # --- target: pooled long reference, both dynamics, independent seed
        ref = {}
        for dyn, spec in DYNAMICS:
            e, _ = cached_arm(prob, dyn, spec, T_REF, N_REF, REF_SEED,
                              args.jobs, "ref", args.refresh)
            ref[dyn] = e
        pooled = np.concatenate([ref["gibbs"], ref["metropolis"]])
        target = float(pooled.min())
        tkind = (f"LONGRUN_BEST(T={T_REF},trials={N_REF}/dynamics,"
                 f"both dynamics,seed={REF_SEED})")
        ref_hits = {k: int((v <= target + TOL).sum()) for k, v in ref.items()}
        log.info(f"  reference best = {target:.1f} (reached by "
                 f"{ref_hits['gibbs']}/{N_REF} gibbs and "
                 f"{ref_hits['metropolis']}/{N_REF} metropolis reference "
                 f"trials) -- NOT a certified ground state")

        # --- sweep budget selection at a disjoint seed
        T_sel, trace = select_T(prob, target, args.jobs, args.refresh)
        log.info(f"  selected T = {T_sel} for both arms of degree {d}")
        budgets = [(T_sel, "selected")]
        if rep == 0:
            budgets += [(t, "robustness") for t in args.extra_T
                        if t != T_sel]

        # --- reporting runs
        cis = {}
        for T, rung in budgets:
            arm = {}
            for dyn, spec in DYNAMICS:
                e, wall = cached_arm(prob, dyn, spec, T, args.trials,
                                     MASTER_SEED, args.jobs, "main",
                                     args.refresh)
                hits = int((e <= target + TOL).sum())
                ps = hits / len(e)
                lo, hi = wilson(hits, len(e))
                arm[dyn] = dict(e=e, hits=hits, ps=ps, lo=lo, hi=hi,
                                wall=wall, tts=tts_at_confidence(T, ps))
                log.info(f"  [{rung[:4]} T={T} {dyn:>10s}] p_s={ps:.3f} "
                         f"[{lo:.3f},{hi:.3f}] ({hits}/{len(e)})  "
                         f"E_med={np.median(e):.1f}  ({wall:.0f}s)")
            ci = tts_ratio_ci(arm["gibbs"]["hits"], args.trials,
                              arm["metropolis"]["hits"], args.trials,
                              n_sweeps=T)
            cis[f"T={T}"] = dict(rung=rung, **ci)
            log.info(f"  [{rung[:4]} T={T}] TTS ratio SA/Gibbs = "
                     f"{ci['ratio']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] "
                     f"(>1 = Gibbs faster; undefined frac "
                     f"{ci['frac_undefined']})")

            for dyn, _ in DYNAMICS:
                a = arm[dyn]
                rows.append(dict(
                    degree=d, rep=rep, rung=rung, n=N_SPINS,
                    instance=prob.name, instance_seed=seed, dynamics=dyn,
                    n_sweeps=T, n_trials=args.trials,
                    master_seed=MASTER_SEED, target=target,
                    target_kind=tkind, hits=a["hits"],
                    p_success=round(a["ps"], 5),
                    wilson_lo=round(a["lo"], 5), wilson_hi=round(a["hi"], 5),
                    tts99_sweeps=a["tts"],
                    tts_ratio_sa_over_gibbs=(round(ci["ratio"], 5)
                                             if dyn == "metropolis" else ""),
                    ratio_lo=round(ci["lo"], 5) if dyn == "metropolis" else "",
                    ratio_hi=round(ci["hi"], 5) if dyn == "metropolis" else "",
                    frac_undefined=(ci["frac_undefined"]
                                    if dyn == "metropolis" else ""),
                    energy_min=float(a["e"].min()),
                    energy_median=float(np.median(a["e"])),
                    energy_mean=float(a["e"].mean()),
                    energy_std=float(a["e"].std()),
                    heff_abs_median=round(hs["heff_abs_median"], 5),
                    heff_abs_var=round(hs["heff_abs_var"], 5),
                    heff_var=round(hs["heff_var"], 5),
                    wall_s=round(a["wall"], 1)))

        meta[f"d{d}_rep{rep}"] = dict(
            instance=info, heff=hs, target=target, target_kind=tkind,
            ref_hits_at_best=ref_hits, T_selected=T_sel, T_pilot_trace=trace,
            tts_ratio_sa_over_gibbs=cis)

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    JSON_OUT.write_text(json.dumps(dict(
        _label=("RX-11 controlled connectivity ladder for the "
                "Gibbs-vs-Metropolis single-step-rule claim of Section "
                "3.3.1/3.6"),
        n=N_SPINS, degrees=args.degrees, reps=args.reps,
        extra_T=args.extra_T, weight_law="uniform {-1,+1}",
        coupling_convention="J_ij = -w_ij/2, h = 0 (as problems.load_gset)",
        schedule=SHAPE, beta=[BETA0, BETAF], update_mode="async_numba",
        n_trials=args.trials, master_seed=MASTER_SEED,
        seeds=dict(instance="SEED_BASE + degree + REP_STRIDE * rep",
                   instance_base=SEED_BASE, rep_stride=REP_STRIDE,
                   reference=REF_SEED, pilot=PILOT_SEED, heff=HEFF_SEED),
        reference=dict(T=T_REF, trials_per_dynamics=N_REF,
                       pooled_over_both_dynamics=True),
        T_selection=dict(default=T_DEFAULT, ladder=T_LADDER, band=list(BAND),
                         statistic="geometric mean of the two arms' pilot p_s",
                         pilot_trials=PILOT_TRIALS),
        stats=("Wilson 95% CI on every p_s; parametric-bootstrap 95% CI on "
               "the TTS ratio (stats.py, RX-01 convention). Ratio is "
               "TTS_metropolis / TTS_gibbs, so > 1 means Gibbs faster."),
        note=("targets are long-run best energies, not certified ground "
              "states; p_success is the fraction of trials reaching or "
              "beating that energy"),
        per_degree=meta), indent=2))
    log.info(f"-> {CSV_OUT}")
    log.info(f"-> {JSON_OUT}")


if __name__ == "__main__":
    main()
