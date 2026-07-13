#!/usr/bin/env python3
"""W5: feed the N=64 write-line IR-drop offset profile back into the Chapter-3 solver.

Ising context: on the Table-3.8 64x64 tile each column write line serves 64 rows; the
extraction flow (analyze_ir.py -> ir_drop_summary.json) gives every row a STATIC drive
deficit u_off(r) = dV(r)/VT on the calibrated sigmoid (far row: 1.51 u at row 64). This
script asks what that deficit costs the annealer, and whether 6-bit DAC predistortion
(per-row code offset, residual <= LSB/2) recovers it.

Protocol — EXACTLY the Section-3.4.2 device-ablation protocol (run_circuit_ablation.py):
fixed 14-spin ER Max-Cut instance random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0),
exact ground state by enumeration, SolverConfig(geometric, beta0=0.1, betaf=5.0,
n_sweeps=2000, update_mode="block"), multistart 200 trials, master_seed=2024;
summarize_runs -> p_success + tts_99_sweeps. Three scenarios:
  * baseline      — no offset (ideal delivery, u_offset = 0);
  * uncompensated — u_offset = -u_off(row_i): driver calibrated for the nearest row only;
  * predistorted  — u_offset = -resid_u(row_i): per-row compensation code
                    round(dV/LSB) applied, only the signed quantization residual remains.

Row mapping (documented, deterministic): the 14 spins are assigned to tile rows evenly
spread over 0..63 via row_i = round(i * 63 / 13), i = 0..13 ->
[0, 5, 10, 15, 19, 24, 29, 34, 39, 44, 48, 53, 58, 63]. Row k sits (k+1) cell pitches
from the column driver, i.e. analyze_ir.py row index r = k + 1 (rows list is r = 1..64).

Offsets enter through the registered backend eda/interface/circuit_backends.py:
spin_spec = ("circuit_chain", dict(mode="none", u_offset=<per-spin array>)) — u_offset
is in u units, ADDED to the drive after quantization (mode="none": no DAC grid here;
the pure IR-drop axis, same isolation style as the reset-pulse axis E).

RNG: only the solver's seeded generators (master_seed=2024 recorded below); nothing else.

Run (Windows or WSL, pure Python):
  python eda/extraction/writeline_ir/ir_solver_impact.py [--jobs 10]
Writes ir_solver_impact.csv next to this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))                       # isim, problems
sys.path.insert(0, str(ROOT / "eda" / "interface"))  # circuit_backends

from isim import SolverConfig, multistart, summarize_runs  # noqa: E402
from problems import random_er_maxcut                      # noqa: E402
import circuit_backends                                    # noqa: F401,E402

ER_N, ER_P, ER_SEED = 14, 0.30, 0
SWEEPS, BETA0, BETAF = 2000, 0.1, 5.0
TRIALS, SEED = 200, 2024
N_TILE = 64


def enumerate_ground(problem):
    """Exact E_min by full enumeration (same helper as run_circuit_ablation.py)."""
    n = problem.n
    assert n <= 22
    best = np.inf
    for block_start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - block_start)
        ints = np.arange(block_start, block_start + m, dtype=np.int64)
        bits = ((ints[:, None] >> np.arange(n)) & 1).astype(np.int8)
        s = (2 * bits - 1).astype(np.float64)
        e = -0.5 * np.einsum("ij,ij->i", s @ problem.J.toarray(), s) - s @ problem.h
        best = min(best, float(e.min()))
    return best


def load_profiles():
    """Per-spin offset arrays (u units) from ir_drop_summary.json, N=64 profile."""
    summ = json.loads((HERE / "ir_drop_summary.json").read_text())
    rows = summ["per_N"][str(N_TILE)]["rows"]        # r = 1..64
    tile_rows = np.rint(np.arange(ER_N) * (N_TILE - 1) / (ER_N - 1)).astype(int)
    u_off = np.array([rows[k]["u_off"] for k in tile_rows])       # row k -> r = k+1
    resid_u = np.array([rows[k]["resid_u"] for k in tile_rows])
    lsb_src = summ["provenance"]["dac_lsb"]["source"]
    return tile_rows, u_off, resid_u, lsb_src


def run_one(problem, target, u_offset, jobs):
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=SWEEPS, update_mode="block")
    spec = ("circuit_chain", dict(mode="none", u_offset=u_offset))
    results = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                         n_trials=TRIALS, master_seed=SEED, n_jobs=jobs)
    return summarize_runs(results, target=target, sense="min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    problem = random_er_maxcut(n=ER_N, p=ER_P, sigma=1.0, seed=ER_SEED,
                               name=f"ER{ER_N}_p{ER_P:g}")
    target = enumerate_ground(problem)
    tile_rows, u_off, resid_u, lsb_src = load_profiles()
    print(f"instance {problem.name}: E_min = {target:.6f}")
    print(f"tile rows (spin i -> row): {tile_rows.tolist()}")
    print(f"u_off profile:  max {u_off.max():.3f}  mean {u_off.mean():.3f}  [{lsb_src}]")
    print(f"resid_u profile: max|.| {np.abs(resid_u).max():.4f}")

    scenarios = [
        ("baseline", np.zeros(ER_N)),
        ("uncompensated", -u_off),        # drive deficit: p = sigma(u - u_off)
        ("predistorted", -resid_u),       # only the signed DAC quantization residual
    ]
    rows_out, base_tts = [], None
    for name, off in scenarios:
        s = run_one(problem, target, off, args.jobs)
        tts = s["tts_99_sweeps"]
        if name == "baseline":
            base_tts = tts
        ratio = (tts / base_tts) if (base_tts and np.isfinite(tts)) else float("inf")
        rows_out.append(dict(scenario=name, n_tile=N_TILE,
                             max_abs_u_off=float(np.abs(off).max()),
                             mean_abs_u_off=float(np.abs(off).mean()),
                             p_success=s["p_success"], tts99_sweeps=tts,
                             tts99_ratio=ratio, energy_min=s["energy_min"],
                             energy_median=s["energy_median"],
                             n_trials=TRIALS, n_sweeps=SWEEPS, master_seed=SEED))
        print(f"[{name:>14s}] max|u_off|={np.abs(off).max():.4f}  "
              f"p_s={s['p_success']:.3f}  TTS_sw={tts:12.1f}  ratio={ratio:8.3f}")

    out = HERE / "ir_solver_impact.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
