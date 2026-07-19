#!/usr/bin/env python3
"""RX-05 follow-up — solver cost of the LLG-informed effective reset residual.

The macrospin-LLG calibration (eda/testbenches/reset_correlation_llg_summary
.json) reproduces the single-pulse plateau (r1 = 0.70 vs measured 0.72) and
shows reset failures are NOT positively correlated, but resets that
succeeded back-hop on subsequent pulses with p ~ 0.14-0.21, so the
EFFECTIVE AP residual after k pulses decays as ~{0.30, 0.22, 0.19, 0.13}
instead of the independence model 0.28^k. This driver injects those
effective residuals (as sticky probability rho, n_reset=1 with
r_reset = 1 - rho_eff) into the solver on the mechanism/scale instances,
so the thesis can state the k trade-off with LLG-informed numbers.

Run:  python eda/interface/run_reset_llginformed.py
Appends nothing to other CSVs; writes reset_llginformed_summary.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from isim import SolverConfig, multistart, summarize_runs  # noqa: E402
from problems import random_er_maxcut                      # noqa: E402
import circuit_backends                                    # noqa: F401,E402
from stats import wilson, tts_ratio_ci                     # noqa: E402

LLG = json.loads((ROOT / "eda" / "testbenches" /
                  "reset_correlation_llg_summary.json").read_text())
# effective AP residual after k pulses, from the LLG train chain:
# k=1: 1 - r1; k>=2: n_fail_k(+backhop) fractions recorded in the chain
RHO_EFF = {1: 0.30, 2: 0.22, 3: 0.19, 4: 0.13}
REF_SWEEPS, REF_TRIALS, REF_SEED = 20000, 200, 20260720


def run(problem, spec, target, sweeps=2000, trials=200, seed=2024, jobs=10):
    cfg = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                       n_sweeps=sweeps, update_mode="block")
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=trials, master_seed=seed, n_jobs=jobs)
    return summarize_runs(res, target=target, sense="min")


def main():
    rows = []
    # ER14 (exact) + ER64 (LONGRUN_BEST target, same protocol as
    # run_reset_mechanism.py scale part)
    er14 = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0,
                            name="ER14_p0.3")
    t14 = -6.877554338927609
    er64 = random_er_maxcut(n=64, p=0.10, sigma=1.0, seed=0,
                            name="ER64_p0.1")
    cfg_ref = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                           n_sweeps=REF_SWEEPS, update_mode="block")
    ref = multistart(problem=er64, solver_config=cfg_ref,
                     spin_spec=("ideal", {}), n_trials=REF_TRIALS,
                     master_seed=REF_SEED, n_jobs=10)
    t64 = float(min(r.energy_final for r in ref))
    print(f"ER64 LONGRUN_BEST target = {t64:.12f}")

    for name, prob, target in (("ER14_p0.3", er14, t14),
                               ("ER64_p0.1", er64, t64)):
        base = run(prob, ("ideal", {}), target)
        k0, n = round(base["p_success"] * 200), 200
        rows.append(dict(instance=name, config="ideal", k_pulses=0,
                         rho_eff=0.0, p_success=base["p_success"],
                         wilson_lo=wilson(k0, n)[0],
                         wilson_hi=wilson(k0, n)[1],
                         tts99_sweeps=base["tts_99_sweeps"], tts_ratio=1.0,
                         ratio_lo="", ratio_hi="", target=target))
        for k, rho in RHO_EFF.items():
            s = run(prob, ("circuit_chain",
                           dict(mode="none", n_reset=1,
                                r_reset=1.0 - rho)), target)
            kk = round(s["p_success"] * 200)
            r = tts_ratio_ci(k0, n, kk, n, n_sweeps=2000) \
                if k0 > 0 and kk > 0 else dict(ratio=float("inf"),
                                               lo="", hi="")
            ratio = (s["tts_99_sweeps"] / base["tts_99_sweeps"]
                     if np.isfinite(s["tts_99_sweeps"]) else float("inf"))
            rows.append(dict(instance=name, config=f"llg_rho_k{k}",
                             k_pulses=k, rho_eff=rho,
                             p_success=s["p_success"],
                             wilson_lo=wilson(kk, n)[0],
                             wilson_hi=wilson(kk, n)[1],
                             tts99_sweeps=s["tts_99_sweeps"],
                             tts_ratio=ratio,
                             ratio_lo=r.get("lo", ""),
                             ratio_hi=r.get("hi", ""), target=target))
            print(f"{name} k={k} rho_eff={rho}: p_s={s['p_success']:.3f} "
                  f"ratio={ratio:.3f}")

    with open(HERE / "reset_llginformed_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-> reset_llginformed_summary.csv")


if __name__ == "__main__":
    main()
