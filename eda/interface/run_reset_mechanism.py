#!/usr/bin/env python3
"""RX-05b/c — reset-story mechanism controls + scale check.

Part "mech" (RX-05b): is the benignity of the sticky reset due to
ASYMMETRY (only one transition direction saturates) or to the two-step
reset-write STRUCTURE? Four-way comparison on the Section 3.4.2 protocol
(ER14 instance, exact target), 2x2 in {structure} x {saturation}:

  config              structure  saturation   model (per update)
  ------------------  ---------  -----------  ---------------------------------
  sticky_k1 / k3      two-step   asymmetric   P(+1|+1)=rho+(1-rho)sig(u),
                                              P(+1|-1)=sig(u), rho=0.28^k
  asym_ceiling        one-step   asymmetric   P(+1)=min(sig(u),0.72), floor
                                              untouched, state-independent
  twostep_symsat_k*   two-step   symmetric    reset per-pulse success 0.72
                                              (rho=0.28^k) AND write pulse
                                              capped in its own direction:
                                              P(+1|+1)=rho+(1-rho)min(sig,0.72)
                                              P(+1|-1)=min(sig(u),0.72)
  sym_clip_pmax072    one-step   symmetric    behavioral_smtj p_max=0.72:
                                              p=clip(sig(u),0.28,0.72)
                                              (the Section 3.4.2 axis)

Exact write-pulse plateau model used by asym_ceiling / twostep_symsat (see
circuit_backends.CircuitChainSpin, write_ceiling): the probabilistic P->AP
write pulse saturates only in its OWN switching direction,
p_write = min(sigma(u), 0.72); a pulse that fails to switch leaves the
device in P, so the -1 side keeps the ideal tail (floor untouched).

Part "scale" (RX-05c): does the k=1 sticky residual (rho=0.28) stay benign
beyond 14 spins? Same protocol on ER n=20 (exact target by enumeration) and
ER n=64, p=0.10 (no exact ground truth at n=64 — target = best energy over a
long ideal reference run, independently seeded, labeled LONGRUN_BEST; p_s
against that reference is a hit rate at the reference energy, NOT a true
success probability, so the honest scale metric there is energy-median
degradation, also reported).

Protocol (Section 3.4.2 comparability anchor): geometric beta 0.1->5.0,
n_sweeps=2000, block update mode, 200 trials, master_seed=2024. Every p_s
carries a Wilson 95% CI; TTS ratios vs the same-instance ideal baseline
carry parametric-bootstrap CIs (eda/interface/stats.py, RX-01 convention).

Run (Windows python, pure Python, ~10-20 min at --jobs 10):
  python eda/interface/run_reset_mechanism.py [--jobs 10] [--part all]
Writes reset_mechanism_summary.csv + reset_mechanism_config.json into
eda/interface/results_reset_mechanism/.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from isim import SolverConfig, multistart, summarize_runs  # noqa: E402
from problems import random_er_maxcut                      # noqa: E402
from stats import wilson, tts_ratio_ci                     # noqa: E402
import circuit_backends                                    # noqa: F401,E402
import device_model                                        # noqa: F401,E402

SWEEPS, BETA0, BETAF = 2000, 0.1, 5.0
TRIALS, SEED = 200, 2024
REF_SWEEPS, REF_TRIALS, REF_SEED = 20000, 200, 20260720
R = circuit_backends.R_RESET                               # 0.72


def enumerate_ground(problem):
    n = problem.n
    assert n <= 22
    best = np.inf
    J = problem.J.toarray()
    for block_start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - block_start)
        ints = np.arange(block_start, block_start + m, dtype=np.int64)
        bits = ((ints[:, None] >> np.arange(n)) & 1).astype(np.int8)
        s = (2 * bits - 1).astype(np.float64)
        e = -0.5 * np.einsum("ij,ij->i", s @ J, s) - s @ problem.h
        best = min(best, float(e.min()))
    return best


def run_one(problem, spec, jobs, sweeps=SWEEPS, trials=TRIALS, seed=SEED):
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=sweeps, update_mode="block")
    t0 = time.perf_counter()
    results = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                         n_trials=trials, master_seed=seed, n_jobs=jobs,
                         progress=False)
    wall = time.perf_counter() - t0
    return results, wall


def mech_specs(n):
    """(config, structure, saturation, k, spec) for the four-way + extras."""
    beh = dict(g_dev=1.0, h_off=0.0, sigma_c2c=0.0, p_max=R,
               cv_gain=0.0, sigma_off=0.0, n_spins=n, d2d_seed=SEED + 1)
    return [
        ("ideal",              "-",        "none",       0, ("ideal", {})),
        ("sticky_k1",          "two_step", "asymmetric", 1,
         ("circuit_chain", dict(mode="none", n_reset=1))),
        ("sticky_k3",          "two_step", "asymmetric", 3,
         ("circuit_chain", dict(mode="none", n_reset=3))),
        ("asym_ceiling",       "one_step", "asymmetric", 0,
         ("circuit_chain", dict(mode="none", n_reset=0, write_ceiling=R))),
        ("twostep_symsat_k1",  "two_step", "symmetric",  1,
         ("circuit_chain", dict(mode="none", n_reset=1, write_ceiling=R))),
        ("twostep_symsat_k3",  "two_step", "symmetric",  3,
         ("circuit_chain", dict(mode="none", n_reset=3, write_ceiling=R))),
        ("sym_clip_pmax072",   "one_step", "symmetric",  0,
         ("behavioral_smtj", beh)),
    ]


def scale_specs():
    return [
        ("ideal",     "-",        "none",       0, ("ideal", {})),
        ("sticky_k1", "two_step", "asymmetric", 1,
         ("circuit_chain", dict(mode="none", n_reset=1))),
        ("sticky_k3", "two_step", "asymmetric", 3,
         ("circuit_chain", dict(mode="none", n_reset=3))),
    ]


def summarize_block(part, problem, target, target_kind, spec_rows, jobs):
    rows = []
    base_hits = base_tts = None
    for config, structure, saturation, k, spec in spec_rows:
        results, wall = run_one(problem, spec, jobs)
        s = summarize_runs(results, target=target, sense="min")
        hits = int(round(s["p_success"] * s["n_trials"]))
        lo, hi = wilson(hits, s["n_trials"])
        tts = s["tts_99_sweeps"]
        if config == "ideal":
            base_hits, base_tts = hits, tts
            ratio, rlo, rhi, fud = 1.0, "", "", ""
        else:
            ratio = (tts / base_tts) if (base_tts and np.isfinite(tts)) \
                else float("inf")
            ci = tts_ratio_ci(base_hits, TRIALS, hits, TRIALS,
                              n_sweeps=SWEEPS)
            rlo, rhi, fud = ci["lo"], ci["hi"], ci["frac_undefined"]
        rows.append(dict(
            part=part, instance=problem.name, n=problem.n,
            target_kind=target_kind, config=config, structure=structure,
            saturation=saturation, k=k,
            p_success=s["p_success"], hits=hits,
            wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
            tts99_sweeps=tts, tts99_ratio=ratio,
            ratio_lo=rlo, ratio_hi=rhi, frac_undefined=fud,
            energy_min=s["energy_min"], energy_median=s["energy_median"],
            residual_median=s["residual_median"], target=target,
            n_trials=TRIALS, n_sweeps=SWEEPS, wall_s=round(wall, 1)))
        print(f"[{part} {problem.name} {config:>18s}] "
              f"p_s={s['p_success']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"ratio={ratio:8.3f}  "
              f"E_med={s['energy_median']:.4f}  ({wall:.0f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--part", default="all", choices=["mech", "scale", "all"])
    args = ap.parse_args()

    odir = HERE / "results_reset_mechanism"
    odir.mkdir(exist_ok=True)
    rows, cfg_meta = [], {}

    if args.part in ("mech", "all"):
        er14 = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0,
                                name="ER14_p0.3")
        t14 = enumerate_ground(er14)
        print(f"ER14 target (enumerated) = {t14:.12f}")
        cfg_meta["er14"] = dict(n=14, p=0.30, seed=0, target=t14,
                                target_kind="ENUM")
        rows += summarize_block("mech", er14, t14, "ENUM",
                                mech_specs(er14.n), args.jobs)

    if args.part in ("scale", "all"):
        er20 = random_er_maxcut(n=20, p=0.30, sigma=1.0, seed=0,
                                name="ER20_p0.3")
        t20 = enumerate_ground(er20)
        print(f"ER20 target (enumerated) = {t20:.12f}")
        cfg_meta["er20"] = dict(n=20, p=0.30, seed=0, target=t20,
                                target_kind="ENUM")
        rows += summarize_block("scale", er20, t20, "ENUM",
                                scale_specs(), args.jobs)

        er64 = random_er_maxcut(n=64, p=0.10, sigma=1.0, seed=0,
                                name="ER64_p0.1")
        print(f"ER64 long-run reference: ideal, T={REF_SWEEPS}, "
              f"{REF_TRIALS} trials, seed {REF_SEED} ...")
        ref_results, ref_wall = run_one(er64, ("ideal", {}), args.jobs,
                                        sweeps=REF_SWEEPS, trials=REF_TRIALS,
                                        seed=REF_SEED)
        ref_e = np.array([r.energy_final for r in ref_results])
        t64 = float(ref_e.min())
        ref_hits = int((ref_e <= t64 + 1e-6).sum())
        tk64 = (f"LONGRUN_BEST(T={REF_SWEEPS},trials={REF_TRIALS},"
                f"seed={REF_SEED})")
        print(f"ER64 reference best = {t64:.12f} "
              f"(reached by {ref_hits}/{REF_TRIALS} reference trials, "
              f"{ref_wall:.0f}s) — NOT a certified ground state")
        cfg_meta["er64"] = dict(
            n=64, p=0.10, seed=0, target=t64, target_kind=tk64,
            ref_hits_at_best=ref_hits,
            note=("no exact ground truth at n=64; p_success rows for ER64 "
                  "are hit rates at the long-run reference energy, the "
                  "honest scale metric is energy-median degradation"))
        rows += summarize_block("scale", er64, t64, tk64,
                                scale_specs(), args.jobs)

    out = odir / "reset_mechanism_summary.csv"
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    cfgp = odir / "reset_mechanism_config.json"
    prev = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    instances = dict(prev.get("instances", {}))
    instances.update(cfg_meta)
    prev.update(dict(
        _label=("RX-05b mechanism controls (asymmetry vs two-step "
                "structure) + RX-05c scale check; protocol = Section "
                "3.4.2 device ablation"),
        sweeps=SWEEPS, beta=[BETA0, BETAF], trials=TRIALS,
        master_seed=SEED, update_mode="block", r_reset=R,
        write_ceiling_model=("p_write = min(sigma(u), 0.72) in the P->AP "
                             "direction only; -1 tail untouched"),
        stats=("Wilson 95% CI on p_s; parametric-bootstrap 95% CI on "
               "TTS ratio vs same-instance ideal baseline (stats.py)"),
        instances=instances,
    ))
    cfgp.write_text(json.dumps(prev, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
