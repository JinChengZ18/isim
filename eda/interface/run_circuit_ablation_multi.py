#!/usr/bin/env python3
"""RX-04(a) — multi-seed / multi-size robustness of the circuit-ablation
rules (Section 3.5.2/3.5.3), generalizing the single-instance ER14 result.

The canonical run (run_circuit_ablation.py, er_seed=0, n=14) established:
span >= 6 V_T saturates the rail-clip penalty, >= 4 bit fixed-rail
quantization saturates the resolution penalty, sticky reset with model
residual rho = 0.28^k is mild for k >= 1. All three rules come from ONE
instance seed. This driver re-runs axes A (bits, fixed_u) / C (span) /
E (reset-k, model rho) on er_seed in {1,2,3,4} x n = 14 plus
er_seed in {0,1,2,3,4} x n = 20 (all still exactly enumerable) under the
identical Section 3.4.2 protocol: T = 2000 sweeps, geometric beta
0.1 -> 5.0, 200 trials, master_seed = 2024, block update mode.

The canonical er_seed=0/n=14 rows are MERGED from the committed
results_circuit_ablation/circuit_ablation_summary.csv (not re-run), so the
pooled table contains 5 seeds x n=14 and 5 seeds x n=20.

Output rows carry scope = per_seed (one row per instance x point, Wilson CI
on p_s, bootstrap CI on the TTS ratio vs the same-instance ideal baseline)
and scope = pooled (per (n, axis, value): hits summed over seeds -> pooled
Wilson CI; ratio_pooled from pooled hit rates with bootstrap CI;
ratio_geomean/min/max over the per-seed finite ratios).

Run:  python eda/interface/run_circuit_ablation_multi.py [--jobs 12]
Writes circuit_ablation_multi_summary.csv + circuit_ablation_multi_config
.json next to this script; per-arm energies cached (resumable) under
results_circuit_ablation_multi/.
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

from isim import (SolverConfig, multistart, p_success,        # noqa: E402
                  tts_at_confidence)
from problems import random_er_maxcut                         # noqa: E402
from stats import wilson, tts_ratio_ci                        # noqa: E402
import circuit_backends                                       # noqa: F401,E402

ER_P = 0.30
COMBOS = [(14, s) for s in (1, 2, 3, 4)] + [(20, s) for s in range(5)]
SWEEPS, BETA0, BETAF = 2000, 0.1, 5.0
TRIALS, SEED = 200, 2024
TOL = 1e-6

BITS_AXIS = [2, 3, 4, 5, 6, 8]           # A: fixed_u, span 4 (canonical)
SPAN_AXIS = [2.0, 3.0, 4.0, 6.0, 8.0]    # C: 6 bits (canonical)
RESET_AXIS = [1, 2, 3, 4, 5]             # E: model rho = 0.28^k (canonical)

CANONICAL_CSV = HERE / "results_circuit_ablation" / \
    "circuit_ablation_summary.csv"
CANONICAL_AXIS_MAP = {"baseline": ("baseline", "ideal"),
                      "dac_bits_fixed_u": ("bits", "bits"),
                      "dac_span_6b": ("span", "span"),
                      "reset_pulses": ("k", "k")}


def enumerate_ground(problem):
    n = problem.n
    assert n <= 22
    Jd = problem.J.toarray()
    best = np.inf
    for start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - start)
        ints = np.arange(start, start + m, dtype=np.int64)
        s = (2 * ((ints[:, None] >> np.arange(n)) & 1) - 1).astype(np.float64)
        e = -0.5 * np.einsum("ij,ij->i", s @ Jd, s) - s @ problem.h
        best = min(best, float(e.min()))
    return best


def run_grid():
    """(axis, value, spec); bits=6 (span 4, 6b) == span=4 (6b, span 4) is a
    single config -> run once under axis 'span', emit the bits row by copy."""
    runs = [("baseline", 0, ("ideal", {}))]
    for b in BITS_AXIS:
        if b == 6:
            continue                      # dedup: identical to span@4.0
        runs.append(("bits", b, ("circuit_chain",
                                 dict(nbits=b, u_span=4.0, mode="fixed_u"))))
    for sp in SPAN_AXIS:
        runs.append(("span", sp, ("circuit_chain",
                                  dict(nbits=6, u_span=sp, mode="fixed_u"))))
    for k in RESET_AXIS:
        runs.append(("k", k, ("circuit_chain",
                              dict(mode="none", n_reset=k))))
    return runs


def run_cached(odir, problem, axis, value, spec, jobs, force=False):
    key = f"{problem.name}_{axis}_{value:g}"
    fp = odir / f"{key}.json"
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("n_trials") == TRIALS:
            return d
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=SWEEPS, update_mode="block")
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=TRIALS, master_seed=SEED, n_jobs=jobs,
                     progress=False)
    d = dict(instance=problem.name, axis=axis, value=value, spec=repr(spec),
             n_trials=TRIALS, n_sweeps=SWEEPS, beta=[BETA0, BETAF],
             master_seed=SEED, update_mode="block",
             wall_s=time.perf_counter() - t0,
             energies=[float(r.energy_final) for r in res])
    fp.write_text(json.dumps(d))
    return d


def per_seed_row(n, er_seed, axis, value, k, k_base, target, extra=None):
    ps = k / TRIALS
    lo, hi = wilson(k, TRIALS)
    tts = tts_at_confidence(float(SWEEPS), ps, 0.99)
    if axis == "baseline":
        ratio, r_lo, r_hi = 1.0, float("nan"), float("nan")
    elif k_base > 0 and k > 0:
        rc = tts_ratio_ci(k_base, TRIALS, k, TRIALS, n_sweeps=SWEEPS)
        ratio, r_lo, r_hi = rc["ratio"], rc["lo"], rc["hi"]
    elif k_base > 0:
        ratio, r_lo, r_hi = float("inf"), float("nan"), float("nan")
    else:
        ratio, r_lo, r_hi = float("nan"), float("nan"), float("nan")
    row = dict(scope="per_seed", n=n, er_seed=er_seed, axis=axis, value=value,
               n_seeds=1, n_trials=TRIALS, hits=k, p_s=round(ps, 4),
               wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
               tts99_sweeps=(round(tts, 1) if np.isfinite(tts) else "inf"),
               tts_ratio_vs_ideal=ratio, ratio_lo=r_lo, ratio_hi=r_hi,
               ratio_geomean="", ratio_min="", ratio_max="",
               target=target, source="this-run")
    if extra:
        row.update(extra)
    return row


def pooled_row(n, axis, value, seed_rows):
    k_tot = sum(r["hits"] for r in seed_rows)
    n_tot = TRIALS * len(seed_rows)
    ps = k_tot / n_tot
    lo, hi = wilson(k_tot, n_tot)
    ratios = [r["tts_ratio_vs_ideal"] for r in seed_rows
              if isinstance(r["tts_ratio_vs_ideal"], float)
              and np.isfinite(r["tts_ratio_vs_ideal"])]
    return dict(scope="pooled", n=n, er_seed="all", axis=axis, value=value,
                n_seeds=len(seed_rows), n_trials=n_tot, hits=k_tot,
                p_s=round(ps, 4), wilson_lo=round(lo, 4),
                wilson_hi=round(hi, 4),
                tts99_sweeps=(round(tts_at_confidence(float(SWEEPS), ps,
                                                      0.99), 1)
                              if ps > 0 else "inf"),
                tts_ratio_vs_ideal="", ratio_lo="", ratio_hi="",
                ratio_geomean=(round(float(np.exp(np.mean(np.log(ratios)))),
                                     3) if ratios else ""),
                ratio_min=(round(min(ratios), 3) if ratios else ""),
                ratio_max=(round(max(ratios), 3) if ratios else ""),
                target="", source=f"pooled over {len(seed_rows)} seeds")


def pooled_ratio_ci(base_rows, point_rows):
    """Bootstrap CI on TTS(pooled point)/TTS(pooled baseline)."""
    k_b = sum(r["hits"] for r in base_rows)
    k_p = sum(r["hits"] for r in point_rows)
    n_b = TRIALS * len(base_rows)
    n_p = TRIALS * len(point_rows)
    if k_b > 0 and k_p > 0:
        rc = tts_ratio_ci(k_b, n_b, k_p, n_p, n_sweeps=SWEEPS)
        return rc["ratio"], rc["lo"], rc["hi"]
    if k_b > 0:
        return float("inf"), float("nan"), float("nan")
    return float("nan"), float("nan"), float("nan")


def read_canonical():
    """Canonical er_seed=0/n=14 rows -> (axis, value, hits) list."""
    out = []
    with open(CANONICAL_CSV, newline="") as f:
        for rec in csv.DictReader(f):
            m = CANONICAL_AXIS_MAP.get(rec["axis"])
            if m is None:
                continue                  # axes B/D/F are not in this grid
            axis = m[0]
            value = float(rec["value"])
            k = int(round(float(rec["p_success"])
                          * int(rec["n_trials"])))
            out.append((axis, value, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    odir = HERE / "results_circuit_ablation_multi"
    odir.mkdir(exist_ok=True)
    t_all0 = time.perf_counter()

    all_rows = []
    seed_rows_by_point = {}               # (n, axis, value) -> [row, ...]

    # -- canonical seed-0/n=14 merge ---------------------------------------
    canon_base_k = None
    canon = read_canonical()
    for axis, value, k in canon:
        if axis == "baseline":
            canon_base_k = k
    for axis, value, k in canon:
        row = per_seed_row(14, 0, axis, value, k, canon_base_k,
                           target=-6.877554338927609)
        row["source"] = "canonical(results_circuit_ablation)"
        all_rows.append(row)
        seed_rows_by_point.setdefault((14, axis, value), []).append(row)
    # canonical grid ran bits=6 and span=4.0 as separate identical-config
    # runs; both rows are kept as-is (they agree: p_s = 0.08 each).
    print(f"merged {len(canon)} canonical rows (er_seed=0, n=14)",
          flush=True)

    # -- fresh combos -------------------------------------------------------
    targets = {"ER14_p0.3_s0": -6.877554338927609}
    for n, er_seed in COMBOS:
        prob = random_er_maxcut(n=n, p=ER_P, sigma=1.0, seed=er_seed,
                                name=f"ER{n}_p{ER_P:g}_s{er_seed}")
        target = enumerate_ground(prob)
        targets[prob.name] = target
        print(f"== {prob.name}: deg_mean={prob.J.nnz/prob.n:.2f} "
              f"E_min={target:.6f}", flush=True)
        base_k = None
        span4_d = None
        for axis, value, spec in run_grid():
            d = run_cached(odir, prob, axis, value, spec, args.jobs,
                           args.force_recompute)
            e = np.asarray(d["energies"], dtype=np.float64)
            k = int(round(p_success(e, target, atol=TOL) * TRIALS))
            if axis == "baseline":
                base_k = k
            if axis == "span" and value == 4.0:
                span4_d = d
            row = per_seed_row(n, er_seed, axis, value, k, base_k, target)
            all_rows.append(row)
            seed_rows_by_point.setdefault((n, axis, float(value)),
                                          []).append(row)
            print(f"  [{axis:>8s} v={value:>4}] hits={k:>3d}/{TRIALS} "
                  f"p_s={k/TRIALS:.3f} ratio={row['tts_ratio_vs_ideal']}",
                  flush=True)
        # dedup row: bits=6 == span@4.0 (identical config and seed chain)
        e = np.asarray(span4_d["energies"], dtype=np.float64)
        k = int(round(p_success(e, target, atol=TOL) * TRIALS))
        row = per_seed_row(n, er_seed, "bits", 6, k, base_k, target)
        row["source"] = "dedup(span@4)"
        all_rows.append(row)
        seed_rows_by_point.setdefault((n, "bits", 6.0), []).append(row)

    # -- pooled rows --------------------------------------------------------
    pooled = []
    for n in (14, 20):
        base_rows = seed_rows_by_point.get((n, "baseline", 0.0), [])
        for (nn, axis, value), rows in sorted(
                seed_rows_by_point.items(),
                key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2]))):
            if nn != n:
                continue
            pr = pooled_row(n, axis, value, rows)
            if axis != "baseline" and base_rows:
                r, lo, hi = pooled_ratio_ci(base_rows, rows)
                pr["tts_ratio_vs_ideal"] = r
                pr["ratio_lo"] = lo
                pr["ratio_hi"] = hi
            pooled.append(pr)
    all_rows.extend(pooled)

    csv_path = HERE / "circuit_ablation_multi_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    (HERE / "circuit_ablation_multi_config.json").write_text(json.dumps(dict(
        _label=("RX-04(a) multi-seed/multi-size ER robustness of the "
                "circuit-ablation axes A/C/E; Section 3.4.2 protocol "
                "(T=2000, beta 0.1->5, N=200, seed 2024), block mode; "
                "er_seed=0/n=14 rows merged from the canonical committed "
                "CSV, all other rows MEASURED here; bits=6 rows "
                "deduplicated from the identical span@4 config"),
        er_p=ER_P, combos=[list(c) for c in COMBOS],
        canonical_merge=str(CANONICAL_CSV.relative_to(ROOT)),
        sweeps=SWEEPS, beta=[BETA0, BETAF], trials=TRIALS, master_seed=SEED,
        update_mode="block", bits_axis=BITS_AXIS, span_axis=SPAN_AXIS,
        reset_axis=RESET_AXIS, r_reset=circuit_backends.R_RESET,
        reset_model="rho=(1-0.72)^k independence model (canonical rule "
                    "under test; RX-05 LLG-informed chain is separate)",
        targets=targets,
        total_wall_s=time.perf_counter() - t_all0), indent=2))
    print(f"-> {csv_path}", flush=True)
    print(f"total wall: {(time.perf_counter()-t_all0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
