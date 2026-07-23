#!/usr/bin/env python3
"""RX-15(b) — does the permutation-space cluster solver keep p_s ~= 1 as the
TSP size grows past the n<=17 instances of §3.3.3?

Runs ONLY the permutation-space arm (the QUBO arm is n^2 spins and meaningless
at these sizes) on three larger TSPLIB instances at the §3.3.3 permutation
protocol, and reports p_s at the published optimum with Wilson 95% intervals.

Optima verified 2026-07-23 against the Heidelberg TSPLIB95 optimal-solutions
table: att48=10628 (ATT distance), eil51=426 (EUC_2D), kroA100=21282 (EUC_2D).
The loader (problems.load_tsplib) implements ATT and EUC_2D, so att48 uses the
pseudo-Euclidean metric its optimum is defined under.

Run:  python bench_perm_scale.py [--trials 100] [--jobs 12]
Writes results_rerun/results_perm_scale/perm_scale_summary.csv (+ config json).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from problems import _parse_tsplib
from perm_tsp import (PermutationSolverConfig, multistart_permutation,
                      summarize_permutation_runs)

HERE = Path(__file__).resolve().parent
OPT = {"att48": 10628, "eil51": 426, "kroA100": 21282,
       # small controls to reproduce the §3.3.3 p_s=1 claim
       "burma14": 3323, "ulysses16": 6859, "gr17": 2085}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+",
                    default=["burma14", "ulysses16", "gr17",
                             "att48", "eil51", "kroA100"])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=5000)
    ap.add_argument("--beta0", type=float, default=0.1)
    ap.add_argument("--betaf", type=float, default=50.0)
    ap.add_argument("--opt-tol", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tsplib-dir", default="./tsplib")
    args = ap.parse_args()

    odir = HERE / "results_rerun" / "results_perm_scale"
    odir.mkdir(parents=True, exist_ok=True)
    rows = []
    for inst in args.instances:
        name, D = _parse_tsplib(Path(args.tsplib_dir) / f"{inst}.tsp")
        n = D.shape[0]
        opt = OPT[inst]
        cfg = PermutationSolverConfig(
            n_sweeps=args.sweeps, beta0=args.beta0, betaf=args.betaf,
            schedule_shape="geometric")
        res = multistart_permutation(D, cfg, n_trials=args.trials,
                                     master_seed=args.seed, n_jobs=args.jobs,
                                     progress=False)
        summ = summarize_permutation_runs(res, opt, tol=args.opt_tol)
        best = summ["length_best"]
        k = summ.get("n_success",
                     round(summ["p_success"] * args.trials))
        lo, hi = wilson(k, args.trials)
        rows.append(dict(instance=inst, n=n, opt=opt,
                         best=round(float(best), 2),
                         gap_best_pct=round(100 * (best - opt) / opt, 4),
                         p_success=summ["p_success"],
                         wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
                         tts99_wall=summ.get("tts_99_wall"),
                         gap_median_pct=round(100 * summ["gap_median"], 4),
                         n_trials=args.trials, n_sweeps=args.sweeps))
        print(f"{inst:9s} n={n:3d} opt={opt:6d} best={best:9.1f} "
              f"gap={rows[-1]['gap_best_pct']:.3f}% p_s={summ['p_success']:.3f} "
              f"[{lo:.3f},{hi:.3f}]", flush=True)

    with open(odir / "perm_scale_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (odir / "perm_scale_config.json").write_text(json.dumps(dict(
        _label=("RX-15(b): permutation-space cluster solver at scale; "
                "optima verified 2026-07-23 vs Heidelberg TSPLIB95 table"),
        protocol=dict(sweeps=args.sweeps, beta=[args.beta0, args.betaf],
                      trials=args.trials, opt_tol=args.opt_tol,
                      master_seed=args.seed, schedule="geometric"),
        optima=OPT), indent=2))
    print(f"-> {odir / 'perm_scale_summary.csv'}")


if __name__ == "__main__":
    main()
