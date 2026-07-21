#!/usr/bin/env python3
"""RX-04(b) — circuit-constraint ablation on real G-set landscapes.

Generalizes the Section 3.5.2/3.5.3 rail-span and bit-width rules beyond the
14-spin ER instance: the clip rule is landscape-coupled (truncated drive
fraction scales with the |h_eff| distribution ~ graph degree), so the span
saturation point measured on ER14 (degree ~3.9) is re-measured on G1
(n=800, degree ~48), G22 (n=2000, degree ~20) and optionally G14 (n=800,
degree ~11.7).

Protocol = the Section 3.3 G-set comparability anchor: T = 10000 sweeps,
geometric beta 0.1 -> 10.0, N_trial = 200, master_seed = 2024, block update
mode (custom backends are block-only), target = edge_sum/2 - BKS_cut.

Axes per instance (backend: circuit_backends.CircuitChainSpin):
  baseline   ideal Gibbs
  span_6b    rail span {2, 4, 6, 8, 10} V_T at 6 bits, fixed-u grid
  bits_span4 bit width {4, 6} at span 4 V_T, fixed-u grid (bits=6 is the
             identical config to span_6b@4 -> deduplicated, row copied)
  reset_k    k in {1, 3} reset pulses, no quantization, model residual
             rho = (1 - 0.72)^k (the beta-scaled axis is RX-08's, not rerun)

Metric convention: G1 sits at ideal p_s ~ 0.7 -> p_s/TTS with Wilson and
bootstrap CIs is decisive. G22 sits at p_s ~ 0.004 (RX-01) -> at N=200 the
hit counts are 0-2; the PRIMARY metric is the energy-median degradation vs
the ideal arm (bootstrap CI), p_s/Wilson still reported with their wide CIs.
G14 is p_s = 0 at T = 1e4 (first hits at T = 1e5) -> energy-median only.

Run:  python eda/interface/run_circuit_ablation_gset.py [--jobs 12]
          [--instances G1 G22 G14] [--probe-only]
Writes circuit_ablation_gset_summary.csv + circuit_ablation_gset_config.json
next to this script; per-arm trial energies cached (resumable) under
results_circuit_ablation_gset/.
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
from problems import load_gset                                # noqa: E402
from stats import wilson, tts_ratio_ci                        # noqa: E402
import circuit_backends                                       # noqa: F401,E402

BKS = {"G1": 11624, "G14": 3064, "G22": 13359}
METRIC_NOTE = {
    "G1":  "p_s/TTS primary (ideal p_s~0.7 at N=200)",
    "G22": "energy_median primary (ideal p_s~0.004; N=200 hit counts 0-2, "
           "Wilson CIs wide by design)",
    "G14": "energy_median only (p_s=0 regime at T=1e4)",
}

TRIALS, SEED = 200, 2024
SWEEPS, BETA0, BETAF = 10000, 0.1, 10.0
SPAN_AXIS = [2.0, 4.0, 6.0, 8.0, 10.0]
BITS_AXIS = [4, 6]                       # 6 == span_6b@4 config, deduped
RESET_AXIS = [1, 3]
BOOT_B = 10000
BOOT_SEED = 20260720
TOL = 1e-6


def boot_median_ci(x, conf=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x[rng.integers(0, len(x), size=(BOOT_B, len(x)))], axis=1)
    a = (1 - conf) / 2
    lo, hi = np.quantile(med, [a, 1 - a])
    return float(lo), float(hi)


def boot_dmedian_ci(x_arm, x_ideal, conf=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    xa = np.asarray(x_arm, dtype=np.float64)
    xi = np.asarray(x_ideal, dtype=np.float64)
    ma = np.median(xa[rng.integers(0, len(xa), size=(BOOT_B, len(xa)))], axis=1)
    mi = np.median(xi[rng.integers(0, len(xi), size=(BOOT_B, len(xi)))], axis=1)
    d = ma - mi
    a = (1 - conf) / 2
    lo, hi = np.quantile(d, [a, 1 - a])
    return float(lo), float(hi)


def run_grid(instance_name):
    """(axis, param, value, spec) list; the dedup row is added post-hoc."""
    runs = [("baseline", "ideal", 0, ("ideal", {}))]
    for sp in SPAN_AXIS:
        runs.append(("span_6b", "span", sp,
                     ("circuit_chain", dict(nbits=6, u_span=sp,
                                            mode="fixed_u"))))
    runs.append(("bits_span4", "bits", 4,
                 ("circuit_chain", dict(nbits=4, u_span=4.0,
                                        mode="fixed_u"))))
    for k in RESET_AXIS:
        runs.append(("reset_k", "k", k,
                     ("circuit_chain", dict(mode="none", n_reset=k))))
    return runs


def run_cached(odir, problem, axis, value, spec, jobs, force=False):
    key = f"{problem.name}_{axis}_{value:g}" if isinstance(value, float) \
        else f"{problem.name}_{axis}_{value}"
    fp = odir / f"{key}.json"
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("n_trials") == TRIALS:
            print(f"  [cache] {key} (wall {d['wall_s']:.0f}s)", flush=True)
            return d
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=SWEEPS, update_mode="block")
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=TRIALS, master_seed=SEED, n_jobs=jobs,
                     progress=False)
    wall = time.perf_counter() - t0
    d = dict(instance=problem.name, axis=axis, value=value, spec=repr(spec),
             n_trials=TRIALS, n_sweeps=SWEEPS, beta=[BETA0, BETAF],
             master_seed=SEED, update_mode="block", wall_s=wall,
             seed_derivation="np.random.SeedSequence(2024).spawn(200)[i]",
             energies=[float(r.energy_final) for r in res])
    fp.write_text(json.dumps(d))
    print(f"  [saved] {key} (wall {wall:.0f}s)", flush=True)
    return d


def summarize_row(problem, target, axis, pname, value, d, base, source):
    e = np.asarray(d["energies"], dtype=np.float64)
    k = int(round(p_success(e, target, atol=TOL) * TRIALS))
    ps = k / TRIALS
    lo, hi = wilson(k, TRIALS)
    tts = tts_at_confidence(float(SWEEPS), ps, 0.99)
    med_lo, med_hi = boot_median_ci(e)
    if base is None:
        ratio, r_lo, r_hi = 1.0, float("nan"), float("nan")
        dmed = d_lo = d_hi = 0.0
    else:
        e_base, k_base = base
        if k_base > 0 and k > 0:
            rc = tts_ratio_ci(k_base, TRIALS, k, TRIALS, n_sweeps=SWEEPS)
            ratio, r_lo, r_hi = rc["ratio"], rc["lo"], rc["hi"]
        elif k_base > 0:
            ratio, r_lo, r_hi = float("inf"), float("nan"), float("nan")
        else:
            ratio, r_lo, r_hi = float("nan"), float("nan"), float("nan")
        dmed = float(np.median(e) - np.median(e_base))
        d_lo, d_hi = boot_dmedian_ci(e, e_base)
    return dict(
        instance=problem.name, n=problem.n,
        degree_mean=round(problem.J.nnz / problem.n, 2),
        axis=axis, param=pname, value=value,
        n_trials=TRIALS, n_sweeps=SWEEPS,
        hits=k, p_s=ps, wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
        tts99_sweeps=(round(tts, 1) if np.isfinite(tts) else "inf"),
        tts_ratio_vs_ideal=ratio, ratio_lo=r_lo, ratio_hi=r_hi,
        energy_min=float(e.min()), energy_median=float(np.median(e)),
        emed_lo=round(med_lo, 4), emed_hi=round(med_hi, 4),
        dmed_vs_ideal=dmed, dmed_lo=round(d_lo, 4), dmed_hi=round(d_hi, 4),
        target=target, metric_note=METRIC_NOTE[problem.name],
        source=source, wall_s=round(d.get("wall_s", float("nan")), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--instances", nargs="+", default=["G1", "G22", "G14"])
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    odir = HERE / "results_circuit_ablation_gset"
    odir.mkdir(exist_ok=True)

    gdir = ROOT / "gset"
    try:
        from fetch_data import ensure_gset
        ensure_gset(args.instances, gdir)
    except Exception as err:
        print(f"auto-fetch warning: {err}", flush=True)

    t_all0 = time.perf_counter()
    rows = []
    for name in args.instances:
        prob = load_gset(gdir / name)
        target = prob.meta["edge_sum"] / 2.0 - BKS[name]
        print(f"== {name}: n={prob.n} deg_mean={prob.J.nnz/prob.n:.1f} "
              f"target={target:.1f}  [{METRIC_NOTE[name]}]", flush=True)
        if args.probe_only:
            cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0,
                               betaf=BETAF, n_sweeps=SWEEPS,
                               update_mode="block")
            t0 = time.perf_counter()
            multistart(prob, cfg,
                       ("circuit_chain", dict(nbits=6, u_span=6.0,
                                              mode="fixed_u")),
                       n_trials=2, master_seed=999, n_jobs=1, progress=False)
            print(f"  probe: {(time.perf_counter()-t0)/2:.2f} s/trial",
                  flush=True)
            continue

        base = None
        span4_d = None
        for axis, pname, value, spec in run_grid(name):
            d = run_cached(odir, prob, axis, value, spec, args.jobs,
                           args.force_recompute)
            row = summarize_row(prob, target, axis, pname, value, d, base,
                                source="this-run")
            if axis == "baseline":
                e = np.asarray(d["energies"], dtype=np.float64)
                base = (e, row["hits"])
                # re-emit baseline row now that base is set (ratio=1 def)
            if axis == "span_6b" and value == 4.0:
                span4_d = d
            rows.append(row)
            print(f"  [{axis:>10s} {pname}={value:>4}] hits={row['hits']:>3d}"
                  f"/{TRIALS} p_s={row['p_s']:.3f} "
                  f"CI[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}] "
                  f"E_med={row['energy_median']:.1f} "
                  f"dmed={row['dmed_vs_ideal']:+.1f} "
                  f"ratio={row['tts_ratio_vs_ideal']}", flush=True)
        # dedup: bits=6 @ span4 is the identical config (same seeds) as
        # span_6b @ 4.0 -> copy the cached energies, label the row
        if span4_d is not None:
            row = summarize_row(prob, target, "bits_span4", "bits", 6,
                                span4_d, base, source="dedup(span_6b@4)")
            rows.append(row)
            print(f"  [bits_span4 bits=   6] dedup of span_6b@4", flush=True)

    if args.probe_only:
        return

    csv_path = HERE / "circuit_ablation_gset_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "circuit_ablation_gset_config.json").write_text(json.dumps(dict(
        _label=("RX-04(b) circuit-constraint ablation on G-set landscapes; "
                "G-set protocol (T=1e4, beta 0.1->10, N=200, seed 2024), "
                "block mode; all rows MEASURED; bits=6 row deduplicated "
                "from the identical span_6b@4 config (same seed chain); "
                "beta-scaled axis deliberately absent (RX-08 owns it)"),
        instances=args.instances, bks=BKS, trials=TRIALS, sweeps=SWEEPS,
        beta=[BETA0, BETAF], master_seed=SEED, update_mode="block",
        span_axis=SPAN_AXIS, bits_axis=BITS_AXIS, reset_axis=RESET_AXIS,
        r_reset=circuit_backends.R_RESET,
        reset_model="rho = (1-0.72)^k independence model (NOT the RX-05 "
                    "LLG-informed chain; this axis generalizes the "
                    "canonical ER14 rule across landscapes)",
        boot_B=BOOT_B, boot_seed=BOOT_SEED, metric_notes=METRIC_NOTE,
        total_wall_s=time.perf_counter() - t_all0), indent=2))
    print(f"-> {csv_path}", flush=True)
    print(f"total wall: {(time.perf_counter()-t_all0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
