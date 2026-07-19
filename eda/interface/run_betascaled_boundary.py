#!/usr/bin/env python3
"""RX-08 — adversarial boundary for the beta-scaled coarse-grid speedup.

Section 3.5.2 reports that the beta-scaled 3-bit DAC configuration SPEEDS UP
the 14-spin ER instance (0.16x TTS vs ideal) via small-field inflation (the
mid-rise grid has no zero code), and refuses to generalize. As written that
caveat is unfalsifiable: no landscape where the coarse grid HURTS has been
shown. This driver runs the identical backend on three landscapes where fine
small-field resolution is plausibly load-bearing:

  * G14      — sparse unit-weight G-set graph (n=800), near-degenerate
               shallow minima (G-set protocol, T=1e4, beta 0.1->10; p_s vs
               BKS=3064 is ~0 in this budget, so the verdict METRIC SWITCHES
               to the energy median / best-energy degradation vs the ideal
               arm, with bootstrap CIs; p_s+Wilson still reported);
  * M=65     — factoring semiprime with the deepest competing pseudo-product
               trap of the 3.3.2 set (factor protocol, T=2e4, beta 0.05->30,
               penalty C=M^2+1 via build_factoring_problem defaults).
               Success = DECODED factors (p_hat*q_hat == M with all z
               constraints satisfied), exactly compare_baselines.py factor
               mode; the energy target is the planted-solution energy
               E* (= QUBO 0 minus the dropped Ising constant), tol 0.5;
  * reg3_n16 — random 3-regular +/-1-weight Max-Cut instance, frustrated,
               2^16 enumerable for an exact target (3.4.2-style protocol,
               T=2000, beta 0.1->5).

Plus the ER14 canonical point (3.4.2 protocol) as a WIRING VALIDATION:
expected ideal p_s ~= 0.185 and beta-scaled 3-bit p_s ~= 0.73 (the 0.16x
row of results_circuit_ablation/circuit_ablation_summary.csv).

Arms per landscape: ideal | beta-scaled 3-bit | beta-scaled 6-bit control
(circuit_chain mode="beta_scaled", u_span = h_clip from the instance:
ceil(max_i(sum_j |J_ij| + |h_i|)), the run_circuit_ablation.py rule
generalized to h != 0), master seeds {2024, 2025, 4096}, 200 trials each,
block update mode (custom backends are block-only).

Run:  python eda/interface/run_betascaled_boundary.py [--jobs 10]
Writes betascaled_boundary_summary.csv + config json next to this script and
per-arm trial energies under results_betascaled_boundary/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from isim import (SolverConfig, multistart, p_success,        # noqa: E402
                  tts_at_confidence, preprocess, Problem)
from problems import (load_gset, build_factoring_problem,     # noqa: E402
                      decode_factors, random_er_maxcut)
from stats import wilson, tts_ratio_ci                        # noqa: E402
import circuit_backends                                       # noqa: F401,E402

SEEDS = [2024, 2025, 4096]
TRIALS = 200
BOOT_B = 10000
BOOT_SEED = 20260720


def h_clip_of(problem):
    """ceil of the exact upper bound on |h_eff|: max_i(sum_j|J_ij| + |h_i|).
    Reduces to run_circuit_ablation.py's ceil(max row-sum |J|) when h=0."""
    rowsum = np.asarray(np.abs(problem.J).sum(axis=1)).ravel()
    return float(np.ceil((rowsum + np.abs(problem.h)).max()))


def reg3_maxcut(n=16, seed=7):
    """Random 3-regular graph, +/-1 edge weights, Max-Cut mapping J=-w/2.
    Configuration-model pairing with rejection of self/multi-edges."""
    rng = np.random.default_rng(seed)
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if (pairs[:, 0] == pairs[:, 1]).any():
            continue
        key = {tuple(sorted(p)) for p in pairs.tolist()}
        if len(key) < len(pairs):
            continue
        break
    rows, cols, data = [], [], []
    for (i, j) in pairs:
        w = rng.choice([-1.0, 1.0])
        rows += [i, j]
        cols += [j, i]
        data += [-w / 2, -w / 2]
    J = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    h = np.zeros(n)
    J, h = preprocess(J, h, normalize=False)
    return Problem(name=f"reg3_n{n}_s{seed}", n=n, J=J, h=h)


def enumerate_min(problem):
    n = problem.n
    assert n <= 22
    Jd = problem.J.toarray()
    best = np.inf
    for start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - start)
        ints = np.arange(start, start + m, dtype=np.int64)
        s = (2 * ((ints[:, None] >> np.arange(n)) & 1) - 1).astype(float)
        e = -0.5 * np.einsum("ij,ij->i", s @ Jd, s) - s @ problem.h
        best = min(best, float(e.min()))
    return best


def m65_planted_energy(prob):
    """Ising energy of the planted solution p=5, q=13 (unique for bp=3,
    bq=4). QUBO ground energy is 0; the Ising form drops a constant, so
    energy-based success must target THIS value, not 0."""
    bp, bq = prob.meta["bp"], prob.meta["bq"]
    n_p, n_q = bp - 1, bq - 1
    x = np.zeros(prob.n, dtype=int)
    pbits = {2: 1}            # p = 1 + 4 = 5
    qbits = {2: 1, 3: 1}      # q = 1 + 4 + 8 = 13
    for i in range(1, bp):
        x[i - 1] = pbits.get(i, 0)
    for j in range(1, bq):
        x[n_p + j - 1] = qbits.get(j, 0)
    for i in range(1, bp):
        for j in range(1, bq):
            x[n_p + n_q + (i - 1) * n_q + (j - 1)] = x[i - 1] * x[n_p + j - 1]
    s_star = 2 * x - 1
    p_hat, q_hat, ok = decode_factors(s_star, prob)
    assert ok and p_hat * q_hat == 65, "planted-solution encoding broken"
    return prob.energy(s_star)


def boot_median_ci(x, conf=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(BOOT_B, len(x)))
    med = np.median(x[idx], axis=1)
    a = (1 - conf) / 2
    lo, hi = np.quantile(med, [a, 1 - a])
    return float(lo), float(hi)


def boot_dmedian_ci(x_arm, x_ideal, conf=0.95, seed=BOOT_SEED):
    """Bootstrap CI on median(arm) - median(ideal), unpaired resampling."""
    rng = np.random.default_rng(seed)
    xa = np.asarray(x_arm, dtype=np.float64)
    xi = np.asarray(x_ideal, dtype=np.float64)
    ma = np.median(xa[rng.integers(0, len(xa), size=(BOOT_B, len(xa)))], axis=1)
    mi = np.median(xi[rng.integers(0, len(xi), size=(BOOT_B, len(xi)))], axis=1)
    d = ma - mi
    a = (1 - conf) / 2
    lo, hi = np.quantile(d, [a, 1 - a])
    return float(lo), float(hi)


def landscapes():
    """(name, problem, target, tol, cfg, seeds, arms, success_kind, metric)"""
    er14 = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0, name="ER14_p0.3")
    t_er = enumerate_min(er14)
    g14 = load_gset(ROOT / "gset" / "G14")
    t_g14 = g14.meta["edge_sum"] / 2.0 - 3064
    m65 = build_factoring_problem(65)
    t_m65 = m65_planted_energy(m65)
    r3 = reg3_maxcut()
    t_r3 = enumerate_min(r3)
    cfg_er = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                          n_sweeps=2000, update_mode="block")
    cfg_g = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=10.0,
                         n_sweeps=10000, update_mode="block")
    cfg_m = SolverConfig(schedule_shape="geometric", beta0=0.05, betaf=30.0,
                         n_sweeps=20000, update_mode="block")
    return [
        ("ER14_wiring", er14, t_er, 1e-6, cfg_er, [2024],
         ["ideal", "beta3"], "energy", "p_s (wiring validation)"),
        ("G14", g14, t_g14, 1e-6, cfg_g, SEEDS,
         ["ideal", "beta3", "beta6"], "energy",
         "energy_median (p_s=0 regime at T=1e4)"),
        ("M65", m65, t_m65, 0.5, cfg_m, SEEDS,
         ["ideal", "beta3", "beta6"], "decoded", "p_s/TTS (decoded factors)"),
        (r3.name, r3, t_r3, 1e-6, cfg_r3(), SEEDS,
         ["ideal", "beta3", "beta6"], "energy", "p_s/TTS (exact E_min)"),
    ]


def cfg_r3():
    return SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                        n_sweeps=2000, update_mode="block")


def arm_spec(arm, h_clip):
    if arm == "ideal":
        return ("ideal", {})
    nbits = {"beta3": 3, "beta6": 6}[arm]
    return ("circuit_chain", dict(nbits=nbits, u_span=h_clip,
                                  mode="beta_scaled"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    odir = HERE / "results_betascaled_boundary"
    odir.mkdir(exist_ok=True)

    rows = []
    for name, prob, target, tol, cfg, seeds, arms, succ, metric in landscapes():
        h_clip = h_clip_of(prob)
        print(f"== {name}: n={prob.n} target={target:.6f} h_clip={h_clip} "
              f"T={cfg.n_sweeps} metric={metric}", flush=True)
        for seed in seeds:
            e_ideal, k_ideal = None, None
            for arm in arms:
                res = multistart(problem=prob, solver_config=cfg,
                                 spin_spec=arm_spec(arm, h_clip),
                                 n_trials=TRIALS, master_seed=seed,
                                 n_jobs=args.jobs, progress=False)
                e = np.array([r.energy_final for r in res])
                factors = None
                if succ == "decoded":
                    M = int(prob.meta["M"])
                    dec = [decode_factors(r.state_final, prob) for r in res]
                    k = sum(1 for p_h, q_h, ok in dec
                            if ok and p_h * q_h == M and p_h > 1 and q_h > 1)
                    factors = Counter(f"{p_h}x{q_h}" for p_h, q_h, ok in dec)
                    k_energy = int(round(p_success(e, target, atol=tol)
                                         * TRIALS))
                else:
                    k = int(round(p_success(e, target, atol=tol) * TRIALS))
                    k_energy = k
                ps = k / TRIALS
                lo, hi = wilson(k, TRIALS)
                tts = tts_at_confidence(float(cfg.n_sweeps), ps, 0.99)
                med_lo, med_hi = boot_median_ci(e)
                if arm == "ideal":
                    e_ideal, k_ideal = e, k
                    ratio, r_lo, r_hi = 1.0, float("nan"), float("nan")
                    dmed = d_lo = d_hi = 0.0
                else:
                    if k_ideal > 0 and k > 0:
                        rc = tts_ratio_ci(k_ideal, TRIALS, k, TRIALS,
                                          n_sweeps=cfg.n_sweeps)
                        ratio, r_lo, r_hi = rc["ratio"], rc["lo"], rc["hi"]
                    elif k_ideal > 0:
                        ratio, r_lo, r_hi = float("inf"), float("nan"), \
                            float("nan")
                    else:
                        ratio, r_lo, r_hi = float("nan"), float("nan"), \
                            float("nan")
                    dmed = float(np.median(e) - np.median(e_ideal))
                    d_lo, d_hi = boot_dmedian_ci(e, e_ideal)
                rows.append(dict(
                    landscape=name, n=prob.n, seed=seed, arm=arm,
                    h_clip=h_clip, n_trials=TRIALS, n_sweeps=cfg.n_sweeps,
                    success_metric=succ, metric_note=metric,
                    k=k, p_s=ps, wilson_lo=round(lo, 4),
                    wilson_hi=round(hi, 4), k_energy=k_energy,
                    tts99_sweeps=tts, tts_ratio_vs_ideal=ratio,
                    tts_ratio_lo=r_lo, tts_ratio_hi=r_hi,
                    energy_min=float(e.min()),
                    energy_median=float(np.median(e)),
                    emed_lo=round(med_lo, 4), emed_hi=round(med_hi, 4),
                    dmed_vs_ideal=dmed, dmed_lo=round(d_lo, 4),
                    dmed_hi=round(d_hi, 4), target=target))
                (odir / f"{name}_s{seed}_{arm}.json").write_text(json.dumps(
                    dict(landscape=name, seed=seed, arm=arm,
                         spec=repr(arm_spec(arm, h_clip)),
                         energies=[float(v) for v in e],
                         decoded_products=(dict(factors.most_common())
                                           if factors else None)),
                    indent=1))
                print(f"[{name} s{seed} {arm:6s}] k={k}/{TRIALS} "
                      f"p_s={ps:.3f} CI[{lo:.3f},{hi:.3f}] "
                      f"E_med={np.median(e):.3f} E_min={e.min():.3f} "
                      f"dmed={dmed:+.3f} ratio={ratio}", flush=True)

    with open(HERE / "betascaled_boundary_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "betascaled_boundary_config.json").write_text(json.dumps(dict(
        _label=("RX-08 adversarial boundary for the beta-scaled coarse "
                "grid; block mode; ideal-Gibbs comparator arm per seed; "
                "beta6 = bit-depth control; all rows MEASURED"),
        seeds=SEEDS, trials=TRIALS, boot_B=BOOT_B, boot_seed=BOOT_SEED,
        h_clip_rule="ceil(max_i(sum_j|J_ij| + |h_i|))",
        m65_success=("decoded factors (compare_baselines.py factor mode); "
                     "energy target = planted-solution Ising energy, "
                     "tol 0.5 (QUBO 0 minus dropped constant)"),
        landscapes=[dict(name=n, n_spins=p.n, target=t, tol=tl,
                         sweeps=c.n_sweeps, beta=[c.beta0, c.betaf],
                         seeds=s, arms=a, success=sk, metric=m)
                    for n, p, t, tl, c, s, a, sk, m in landscapes()]),
        indent=2))
    print("-> betascaled_boundary_summary.csv", flush=True)


if __name__ == "__main__":
    main()
