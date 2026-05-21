"""
bench_device_ablation.py — Device-level ablation: scan each
non-ideality knob of the BehavioralSMTJSpin model on a fixed
problem instance and report the impact on solver performance.

Five axes:
    g_dev      : drive gain  (ideal = 1.0). Scan: 0.5, 0.7, 1.0, 1.5, 2.0
    h_off      : drive offset (ideal = 0). Scan: 0, 0.05, 0.1, 0.2 (units of |J| max)
    sigma_c2c  : C2C Gaussian noise on drive (ideal = 0). Scan: 0, 0.5, 1.0, 2.0
    p_max      : back-hopping plateau ceiling (ideal = 1.0). Scan: 1.0, 0.9, 0.8, 0.72
    cv_gain    : D2D gain dispersion (ideal = 0). Scan: 0, 0.077, 0.15, 0.30, 0.60

For each setting the driver records:
    * p_success
    * TTS_99 (in CPU sweeps), divided by the ideal-device baseline
    * energy box-plot of final states across trials

The driver runs in update-mode=block to actually exercise the
behavioural backend (the JIT path bypasses sample_batch).

Usage:
    python bench_device_ablation.py --instance G1 --gset-dir ./gset \\
        --trials 100 --sweeps 10000 --output ./results_dev_ablation
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isim import (SolverConfig, multistart, summarize_runs,
                  cut_value_from_energy, get_logger,
                  save_results_json)
from problems import load_gset, random_er_maxcut
from plot_style import set_style, TSINGHUA_PURPLE
import device_model  # noqa: F401 — registers the behavioural backend


log = get_logger("bench_dev_ablation")


GSET_BKS = {
    "G1":  11624,  "G14": 3064,   "G22": 13359,
}


def _ideal_baseline(problem, args, cfg, target_energy):
    """Run the ideal Gibbs solver for the same configuration and
    return its summary. Used to normalise the ablation sweep."""
    log.info("Running ideal-Gibbs baseline ...")
    results = multistart(
        problem=problem,
        solver_config=cfg,
        spin_spec=("ideal", {}),
        n_trials=args.trials,
        master_seed=args.seed,
        n_jobs=args.jobs,
        progress=True,
    )
    summary = summarize_runs(results, target=target_energy, sense="min")
    log.info(f"  baseline p_s = {summary['p_success']:.3f}, "
             f"TTS_99 sweeps = {summary['tts_99_sweeps']:.2f}")
    return summary, results


def _device_run(problem, args, cfg, target_energy, knob_name, knob_value):
    """One device-config run. Builds the behavioural backend with the
    requested knob value and all others at their ideal default."""
    spec_kwargs = dict(g_dev=1.0, h_off=0.0, sigma_c2c=0.0, p_max=1.0,
                       cv_gain=0.0, sigma_off=0.0,
                       n_spins=problem.n, d2d_seed=int(args.seed) + 1)
    spec_kwargs[knob_name] = knob_value
    log.info(f"Device run: {knob_name}={knob_value!r}")
    results = multistart(
        problem=problem,
        solver_config=cfg,
        spin_spec=("behavioral_smtj", spec_kwargs),
        n_trials=args.trials,
        master_seed=args.seed,
        n_jobs=args.jobs,
        progress=False,
    )
    summary = summarize_runs(results, target=target_energy, sense="min")
    log.info(f"  p_s = {summary['p_success']:.3f}, "
             f"TTS_99 sweeps = {summary['tts_99_sweeps']:.2f}")
    return summary, results


def plot_ablation(records, baseline, instance_name, out_path):
    """Five-panel grid: one panel per knob axis, showing TTS_99 ratio
    vs ideal baseline on a log y-axis. The ideal point sits at 1.0
    on every panel (constant reference)."""
    set_style()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4),
                             sharey=True)
    axes[0].set_ylabel(r"TTS$_{99}$ ratio (device / ideal)")
    knob_titles = {
        "g_dev":     r"drive gain $g_\mathrm{dev}$",
        "h_off":     r"drive offset $h_\mathrm{off}$",
        "sigma_c2c": r"C2C noise $\sigma_\mathrm{C2C}$",
        "p_max":     r"plateau ceiling $p_\mathrm{max}$",
        "cv_gain":   r"D2D dispersion CV$(\Delta)$",
    }
    base_tts = baseline["tts_99_sweeps"]

    for ax, knob in zip(axes, ["g_dev", "h_off", "sigma_c2c",
                               "p_max", "cv_gain"]):
        rs = [r for r in records if r["knob"] == knob]
        rs.sort(key=lambda r: r["value"])
        xs = np.array([r["value"] for r in rs])
        tts = np.array([r["summary"]["tts_99_sweeps"] for r in rs],
                       dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = tts / base_tts
        finite = np.isfinite(ratio)
        if finite.any():
            ax.plot(xs[finite], ratio[finite], "-",
                    color=TSINGHUA_PURPLE["dark"], lw=2.0)
            ax.plot(xs[finite], ratio[finite], "o",
                    color=TSINGHUA_PURPLE["primary"],
                    markeredgecolor=TSINGHUA_PURPLE["darkest"],
                    markeredgewidth=0.9, markersize=8)
        ax.axhline(1.0, color=TSINGHUA_PURPLE["accent"],
                   ls="--", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel(knob_titles[knob])
        ax.grid(True, which="both", color="#E8E8E8", lw=0.6)
    # Establish a common y-range first, then place n/a markers above it.
    ymax_global = 0.0
    for ax in axes:
        ymax_global = max(ymax_global, ax.get_ylim()[1])
    ymax_global = max(ymax_global, 30.0)
    for ax in axes:
        ax.set_ylim(0.5, ymax_global)
    for ax, knob in zip(axes, ["g_dev", "h_off", "sigma_c2c",
                               "p_max", "cv_gain"]):
        rs = [r for r in records if r["knob"] == knob]
        rs.sort(key=lambda r: r["value"])
        xs = np.array([r["value"] for r in rs])
        tts = np.array([r["summary"]["tts_99_sweeps"] for r in rs],
                       dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = tts / base_tts
        finite = np.isfinite(ratio)
        for x, ok in zip(xs, finite):
            if not ok:
                ax.annotate("",
                            xy=(x, ymax_global * 0.95),
                            xytext=(x, ymax_global * 0.45),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=TSINGHUA_PURPLE["gray"],
                                            lw=1.4))
                ax.text(x, ymax_global * 0.32, "n/a",
                        ha="center", va="top",
                        color=TSINGHUA_PURPLE["gray"], fontsize=10)
    # Reverse plateau ceiling axis: monotonic non-ideality means
    # smaller p_max is "worse", so display the ideal end (1.0) on the
    # left to match the other panels.
    for ax, knob in zip(axes, ["g_dev", "h_off", "sigma_c2c",
                               "p_max", "cv_gain"]):
        if knob == "p_max":
            ax.invert_xaxis()

    fig.suptitle(f"Device non-ideality ablation on {instance_name}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def _enumerate_bks(problem):
    """Brute-force ground-state energy by full enumeration. Only
    valid for n <= 24 (16 M states). Used by the er problem kind."""
    n = problem.n
    if n > 24:
        raise ValueError(f"enumeration limited to n <= 24, got {n}")
    best = float("inf")
    # Iterate via integer; vectorise per row of the 2^n table
    for k in range(2 ** n):
        s = np.array([+1 if (k >> b) & 1 else -1 for b in range(n)],
                     dtype=np.int8)
        e = problem.energy(s)
        if e < best:
            best = e
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem-kind", default="gset",
                   choices=["gset", "er"],
                   help="gset: G-set instance (requires file). "
                        "er: synthetic Erdos-Renyi Max-Cut, "
                        "ground state found by enumeration "
                        "(requires --er-n <= 22).")
    p.add_argument("--instance", default="G1",
                   help="G-set instance name (gset mode only)")
    p.add_argument("--er-n", type=int, default=20,
                   help="ER vertex count (er mode only)")
    p.add_argument("--er-p", type=float, default=0.30,
                   help="ER edge density (er mode only)")
    p.add_argument("--er-seed", type=int, default=0)
    p.add_argument("--gset-dir", default="./gset")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sweeps", type=int, default=10000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.1)
    p.add_argument("--betaf", type=float, default=10.0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--update-mode", default="block",
                   choices=["block", "async_python"],
                   help="Solver update mode. The behavioural sMTJ "
                        "backend is only exercised on the non-JIT "
                        "paths; 'async_numba' is therefore rejected. "
                        "'block' is the default and faster choice.")
    p.add_argument("--output", default="./results_dev_ablation")
    p.add_argument("--auto-fetch", dest="auto_fetch",
                   action="store_true", default=True)
    p.add_argument("--no-auto-fetch", dest="auto_fetch",
                   action="store_false")
    args = p.parse_args(argv)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.problem_kind == "gset":
        gdir = Path(args.gset_dir)
        if args.auto_fetch:
            try:
                from fetch_data import ensure_gset
                ensure_gset([args.instance], gdir)
            except Exception as err:
                log.warning(f"auto-fetch warning: {err}")
        path = gdir / args.instance
        if not path.exists():
            log.error(f"missing G-set instance file: {path}")
            sys.exit(2)
        problem = load_gset(path)
        W = problem.meta["edge_sum"]
        cut_bks = GSET_BKS.get(problem.name)
        if cut_bks is None:
            log.error(f"no BKS table entry for {problem.name}")
            sys.exit(2)
        target_energy = W / 2.0 - cut_bks
    else:
        problem = random_er_maxcut(n=args.er_n, p=args.er_p,
                                   sigma=1.0, seed=args.er_seed,
                                   name=f"ER{args.er_n}_p{args.er_p:g}",
                                   normalize=False)
        log.info(f"Enumerating ground state of {problem.name} "
                 f"(2^{problem.n} states) ...")
        target_energy = _enumerate_bks(problem)
        log.info(f"  ground-state energy = {target_energy:.4f}")

    cfg = SolverConfig(
        schedule_shape=args.schedule, beta0=args.beta0, betaf=args.betaf,
        n_sweeps=args.sweeps,
        update_mode=args.update_mode,   # behavioural backend bypasses JIT
        dynamics="gibbs",
    )
    cfg.validate()

    base_summary, _ = _ideal_baseline(problem, args, cfg, target_energy)

    sweep_grid = {
        "g_dev":     [0.5, 0.7, 1.0, 1.5, 2.0],
        "h_off":     [0.0, 0.05, 0.1, 0.2],
        "sigma_c2c": [0.0, 0.5, 1.0, 2.0],
        "p_max":     [1.0, 0.9, 0.8, 0.72],
        "cv_gain":   [0.0, 0.077, 0.15, 0.30, 0.60],
    }

    records = []
    rows = []
    for knob, values in sweep_grid.items():
        for v in values:
            summary, _ = _device_run(problem, args, cfg,
                                     target_energy, knob, v)
            records.append({"knob": knob, "value": v, "summary": summary})
            rows.append({
                "instance": problem.name,
                "knob": knob,
                "value": v,
                "p_success": summary["p_success"],
                "tts99_sweeps": summary["tts_99_sweeps"],
                "tts99_sweeps_ratio": (summary["tts_99_sweeps"]
                                       / base_summary["tts_99_sweeps"]
                                       if np.isfinite(
                                           summary["tts_99_sweeps"])
                                       else float("inf")),
                "energy_min": summary["energy_min"],
                "energy_median": summary["energy_median"],
                "n_trials": args.trials,
                "n_sweeps": args.sweeps,
            })

    csv_path = odir / "device_ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"summary written: {csv_path}")

    plot_ablation(records, base_summary, problem.name,
                  odir / "device_ablation_panels.png")
    log.info(f"plot written: {odir / 'device_ablation_panels.png'}")


if __name__ == "__main__":
    main()
