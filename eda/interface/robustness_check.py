#!/usr/bin/env python3
"""Seed-robustness check for the two striking circuit-ablation findings
(run_circuit_ablation.py uses the canonical master seed 2024 only):

  * beta-scaled 3-bit apparent acceleration (canonical p_s = 0.73 vs 0.185)
  * sticky reset k = 1 near-baseline cost (canonical p_s = 0.170 vs 0.185)

Re-runs ideal / beta3 / k1 / k3 on the Section-3.4.2 instance with master
seeds 2025 and 4096 (200 trials each). Grounds the thesis footnote that the
beta-scaled acceleration direction is stable across seeds and the k = 1
penalty sits inside statistical fluctuation.

Run:  python eda/interface/robustness_check.py
Writes robustness_summary.json next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from isim import SolverConfig, multistart, summarize_runs  # noqa: E402
from problems import random_er_maxcut                      # noqa: E402
import circuit_backends                                    # noqa: F401,E402

TARGET = -6.877554338927609
CONFIGS = [
    ("ideal", ("ideal", {})),
    ("beta_scaled_3bit", ("circuit_chain",
                          dict(nbits=3, u_span=4.0, mode="beta_scaled"))),
    ("reset_k1", ("circuit_chain", dict(mode="none", n_reset=1))),
    ("reset_k3", ("circuit_chain", dict(mode="none", n_reset=3))),
]


def main():
    prob = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0,
                            name="ER14_p0.3")
    cfg = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                       n_sweeps=2000, update_mode="block")
    rows = []
    for seed in (2025, 4096):
        for name, spec in CONFIGS:
            r = multistart(problem=prob, solver_config=cfg, spin_spec=spec,
                           n_trials=200, master_seed=seed, n_jobs=10)
            s = summarize_runs(r, target=TARGET, sense="min")
            rows.append(dict(master_seed=seed, config=name,
                             p_success=s["p_success"],
                             tts99_sweeps=s["tts_99_sweeps"]))
            print(f"seed={seed} {name:18s} p_s={s['p_success']:.3f}")
    (HERE / "robustness_summary.json").write_text(json.dumps(dict(
        _label=("seed-robustness of the beta-scaled-3bit and reset-k "
                "findings; canonical seed 2024 lives in "
                "results_circuit_ablation/"),
        instance="ER14_p0.3", trials=200, sweeps=2000, beta=[0.1, 5.0],
        rows=rows), indent=2))
    print("-> robustness_summary.json")


if __name__ == "__main__":
    main()
