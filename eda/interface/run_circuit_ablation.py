#!/usr/bin/env python3
"""W4 — feed the measured write-chain constraints back into the solver.

Same protocol as the Section 3.4.2 device ablation (bench_device_ablation.py):
fixed 14-spin ER Max-Cut instance (p_edge = 0.30, er_seed = 0), exact ground
state by enumeration, T = 2000 sweeps, geometric beta 0.1 -> 5.0, 200 trials,
block update mode, master seed 2024. Every run reports p_success and the
TTS_99 (sweeps) ratio against the ideal-Gibbs baseline.

Axes scanned (backend: eda/interface/circuit_backends.py):
  A. DAC bit width, fixed-rail (u-domain) quantization, span +/-4 (ideal grid)
  B. the MEASURED 6-bit grid from update_chain_summary.json (validates A@6)
  C. rail span (clip window) at 6 bits, ideal grid
  D. DAC bit width, beta-scaled rails (h-domain grid, h_clip from instance)
  E. reset pulse count k (sticky-AP residual rho = 0.28^k), no quantization
  F. combined realistic point: measured 6-bit grid + k = 3 resets

Run (Windows or WSL, pure Python):
  python eda/interface/run_circuit_ablation.py [--jobs 10]
Writes circuit_ablation_summary.csv + circuit_ablation_config.json into
eda/interface/results_circuit_ablation/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from isim import SolverConfig, multistart, summarize_runs  # noqa: E402
from problems import random_er_maxcut                      # noqa: E402
import circuit_backends                                    # noqa: F401,E402

TB = ROOT / "eda" / "testbenches"

ER_N, ER_P, ER_SEED = 14, 0.30, 0
SWEEPS, BETA0, BETAF = 2000, 0.1, 5.0
TRIALS, SEED = 200, 2024

BITS_AXIS = [2, 3, 4, 5, 6, 8]
SPAN_AXIS = [2.0, 3.0, 4.0, 6.0, 8.0]
RESET_AXIS = [1, 2, 3, 4, 5]


def enumerate_ground(problem):
    n = problem.n
    assert n <= 22
    best = np.inf
    for block_start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - block_start)
        ints = np.arange(block_start, block_start + m, dtype=np.int64)
        bits = ((ints[:, None] >> np.arange(n)) & 1).astype(np.int8)
        s = (2 * bits - 1).astype(np.float64)
        e = -0.5 * np.einsum("ij,ij->i", s @ problem.J.toarray(), s) \
            - s @ problem.h
        best = min(best, float(e.min()))
    return best


def run_one(problem, target, spec, jobs):
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=SWEEPS, update_mode="block")
    results = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                         n_trials=TRIALS, master_seed=SEED, n_jobs=jobs)
    return summarize_runs(results, target=target, sense="min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    odir = HERE / "results_circuit_ablation"
    odir.mkdir(exist_ok=True)

    problem = random_er_maxcut(n=ER_N, p=ER_P, sigma=1.0, seed=ER_SEED,
                               name=f"ER{ER_N}_p{ER_P:g}")
    target = enumerate_ground(problem)
    h_clip = float(np.ceil(np.abs(problem.J).sum(axis=1).max()))
    print(f"instance {problem.name}: E_min = {target:.6f}  h_clip = {h_clip}")

    measured_grid = None
    summ_path = TB / "update_chain_summary.json"
    if summ_path.exists():
        chain = json.loads(summ_path.read_text())
        measured_grid = [row["u"] for row in chain["per_bits"]["6"]["transfer"]]
        print(f"measured 6-bit grid: u in [{measured_grid[0]:.3f}, "
              f"{measured_grid[-1]:.3f}]")
    else:
        print("WARNING: update_chain_summary.json missing -> axis B skipped")

    runs = [("baseline", "ideal", 0, ("ideal", {}))]
    for b in BITS_AXIS:                                    # A
        runs.append(("dac_bits_fixed_u", "bits", b,
                     ("circuit_chain", dict(nbits=b, u_span=4.0,
                                            mode="fixed_u"))))
    if measured_grid:                                      # B
        runs.append(("dac_measured_6b", "bits", 6,
                     ("circuit_chain", dict(u_grid=measured_grid,
                                            mode="fixed_u"))))
    for sp in SPAN_AXIS:                                   # C
        runs.append(("dac_span_6b", "span", sp,
                     ("circuit_chain", dict(nbits=6, u_span=sp,
                                            mode="fixed_u"))))
    for b in BITS_AXIS:                                    # D
        runs.append(("dac_bits_beta_scaled", "bits", b,
                     ("circuit_chain", dict(nbits=b, u_span=h_clip,
                                            mode="beta_scaled"))))
    for k in RESET_AXIS:                                   # E
        runs.append(("reset_pulses", "k", k,
                     ("circuit_chain", dict(mode="none", n_reset=k))))
    if measured_grid:                                      # F
        runs.append(("combined_meas6b_k3", "k", 3,
                     ("circuit_chain", dict(u_grid=measured_grid,
                                            mode="fixed_u", n_reset=3))))

    rows, base_tts = [], None
    for axis, pname, pval, spec in runs:
        s = run_one(problem, target, spec, args.jobs)
        tts = s["tts_99_sweeps"]
        if axis == "baseline":
            base_tts = tts
        ratio = (tts / base_tts) if (base_tts and np.isfinite(tts)) \
            else float("inf")
        rows.append(dict(axis=axis, param=pname, value=pval,
                         p_success=s["p_success"], tts99_sweeps=tts,
                         tts99_ratio=ratio,
                         energy_min=s["energy_min"],
                         energy_median=s["energy_median"],
                         n_trials=TRIALS, n_sweeps=SWEEPS))
        print(f"[{axis:>22s} {pname}={pval:>4}] p_s={s['p_success']:.3f}  "
              f"TTS_sw={tts:12.1f}  ratio={ratio:8.3f}")

    with open(odir / "circuit_ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (odir / "circuit_ablation_config.json").write_text(json.dumps(dict(
        _label=("solver-side feedback of MEASURED chain constraints; "
                "protocol identical to Section 3.4.2 device ablation"),
        instance=problem.name, er=dict(n=ER_N, p=ER_P, seed=ER_SEED),
        target_energy=target, h_clip=h_clip, sweeps=SWEEPS,
        beta=[BETA0, BETAF], trials=TRIALS, master_seed=SEED,
        update_mode="block", r_reset=circuit_backends.R_RESET,
        measured_grid_source=str(summ_path.name) if measured_grid else None,
    ), indent=2))
    print(f"-> {odir / 'circuit_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
