"""
Ablation study for the Max-Cut benchmark.

Two independent sweeps isolate how solver performance responds to
the two principal solver-side hyperparameters:

  * Sweep budget T: fixes (beta0, betaf) and scans T over several
    orders of magnitude. Establishes whether an instance's zero or
    low success probability under a given budget is a budget issue
    (in which case p_s grows monotonically with T) or a
    landscape-level issue (in which case p_s saturates below 1).

  * Annealing endpoints (beta0, betaf): fixes T and scans the
    initial/final inverse temperatures. Verifies robustness of the
    main-table results to the choice of annealing schedule.

Usage (single line each):
    python bench_maxcut_ablation.py sweeps --instance G14 --Ts 1000 3000 10000 30000 100000 --trials 100 --update-mode async_numba --jobs 4

    python bench_maxcut_ablation.py beta --instance G1 --betaf-values 2 5 10 20 50 --trials 100 --update-mode async_numba --jobs 4

Outputs:
    <out>/sweeps_summary.csv or beta_summary.csv
    <out>/sweeps_scan.png    or beta_scan.png   dual-axis p_s and TTS_99
    <out>/run_<tag>.json     per-scan-point per-trial detail
    <out>/log_<tag>.txt      console log
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isim import (SolverConfig, multistart, tts_at_confidence, p_success,
                  cut_value_from_energy, save_results_json,
                  save_final_states, get_logger)
from problems import load_gset
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("mc_ablation")


GSET_BKS = {
    "G1":  11624, "G2":  11620, "G3":  11622,
    "G14": 3064,  "G15": 3050,  "G16": 3052,
    "G22": 13359, "G23": 13344, "G24": 13337,
    "G43": 6660,  "G44": 6650,  "G45": 6654,
}


def run_point(prob, cfg, args, point_tag: str):
    """Run one scan point: one full multistart batch."""
    log.info(f"=== scan point: {point_tag} "
             f"(T={cfg.n_sweeps}, beta0={cfg.beta0}, betaf={cfg.betaf}) ===")
    results = multistart(
        prob, cfg, spin_spec=("ideal", {}),
        n_trials=args.trials, master_seed=args.seed,
        n_jobs=args.jobs, progress=True,
    )
    W = prob.meta["edge_sum"]
    cut_bks = GSET_BKS.get(prob.name)
    if cut_bks is None:
        cut_bks = max(cut_value_from_energy(r.energy_final, W) for r in results)
    target_energy = W / 2.0 - cut_bks
    energies = np.array([r.energy_final for r in results])
    p_s = p_success(energies, target_energy)
    times = np.array([r.wall_time for r in results])
    t_med = float(np.median(times))
    tts = tts_at_confidence(t_med, p_s, 0.99)
    best_cut = W / 2.0 - energies.min()
    median_cut = W / 2.0 - np.median(energies)
    log.info(f"  best_cut={best_cut:.0f} median_cut={median_cut:.0f} "
             f"BKS={cut_bks} p_s={p_s:.3f} TTS99={tts:.2f}s t_med={t_med:.2f}s")
    return {
        "point_tag": point_tag,
        "n_sweeps": cfg.n_sweeps,
        "beta0": cfg.beta0, "betaf": cfg.betaf,
        "p_success": p_s, "tts_99_wall": tts,
        "time_median": t_med,
        "cut_best": float(best_cut), "cut_median": float(median_cut),
        "cut_bks": float(cut_bks),
        "energies": energies, "results": results,
    }


def plot_scan(records, M_label, xlabel, out_path, xvals_for_axis=None,
              x_is_log=True):
    """Dual-axis: p_s (bars, left) and TTS_99 (markers+line, right log).

    Robust to sparse-data edge cases:
    - left y-axis auto-fits to the actual p_s magnitude so low bars
      are still visible;
    - p_s values are annotated above each bar so very low bars are
      readable even at small visual height;
    - right y-axis TTS_99 range is set explicitly from the finite
      data, decoupled from matplotlib's auto-range;
    - scan points with p_s = 0 (infinite TTS_99) are shown as hollow
      circles at the top of the TTS axis with an `n/a` label just
      below, carrying the semantics "scan point exists but TTS_99
      is not defined".
    """
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    labels = [r["point_tag"] for r in records]
    p_vals = np.asarray([r["p_success"] for r in records])
    tts_vals = np.asarray([r["tts_99_wall"] for r in records])
    xs = np.arange(len(records))

    # Left axis: p_s bars with adaptive ylim and top-of-bar annotations
    bar_colors = [TSINGHUA_PURPLE["gray_lt"] if p == 0
                  else TSINGHUA_PURPLE["primary"] for p in p_vals]
    ax.bar(xs, p_vals, color=bar_colors,
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.9, width=0.66)

    p_max = float(p_vals.max())
    if p_max <= 0:
        y_top = 0.05
    elif p_max < 0.05:
        y_top = max(p_max * 2.0, 0.02)
    else:
        y_top = min(1.05, p_max * 1.30)
    ax.set_ylim(0.0, y_top)

    # Annotate each bar with its p_s value so low bars remain readable
    for x, p_val in zip(xs, p_vals):
        if p_val > 0:
            ax.text(x, p_val + y_top * 0.02,
                    f"{p_val:.3g}" if p_val < 0.01 else f"{p_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=10, color=TSINGHUA_PURPLE["darkest"])

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success probability $p_s$",
                  color=TSINGHUA_PURPLE["dark"])
    ax.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["dark"])

    # Right axis: TTS_99 with explicit log range from the finite data
    ax2 = ax.twinx()
    finite = np.isfinite(tts_vals) & (tts_vals > 0)
    tts_plot = np.where(finite, tts_vals, np.nan)

    if finite.any():
        tts_finite = tts_vals[finite]
        lo = float(tts_finite.min()) / 3.0
        hi = float(tts_finite.max()) * 3.0
        if hi <= lo * 1.5:     # degenerate case: single point or flat
            lo = lo / 3.0
            hi = hi * 3.0
        ax2.set_ylim(lo, hi)

    ax2.set_yscale("log")
    ax2.plot(xs, tts_plot, marker="o", linestyle="-",
             color=TSINGHUA_PURPLE["accent"],
             markersize=9, markeredgecolor=TSINGHUA_PURPLE["darkest"],
             markeredgewidth=0.8, linewidth=1.8)
    ax2.set_ylabel(r"TTS$_{99}$ (s)", color=TSINGHUA_PURPLE["accent"])
    ax2.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["accent"])
    ax2.grid(False)

    # Mark infinite TTS_99 points with a hollow marker near the top
    # of the TTS axis and a small "n/a" label just below. This makes
    # it clear that a scan point exists at this x-location even
    # though TTS_99 is not defined.
    if finite.any() and (~finite).any():
        ylo, yhi = ax2.get_ylim()
        y_marker = yhi / 1.8
        y_label = yhi / 4.5
        for i, ok in enumerate(finite):
            if not ok:
                ax2.plot(i, y_marker, marker="o",
                         markersize=9,
                         markerfacecolor="white",
                         markeredgecolor=TSINGHUA_PURPLE["gray"],
                         markeredgewidth=1.2, linestyle="none")
                ax2.text(i, y_label, "n/a",
                         ha="center", va="center",
                         fontsize=10, color=TSINGHUA_PURPLE["gray"])

    legend_elements = [
        Patch(facecolor=TSINGHUA_PURPLE["primary"],
              edgecolor=TSINGHUA_PURPLE["dark"],
              label="Success probability"),
        Line2D([0], [0], marker="o",
               color=TSINGHUA_PURPLE["accent"],
               markersize=9,
               markeredgecolor=TSINGHUA_PURPLE["darkest"],
               markeredgewidth=0.8,
               label=r"TTS$_{99}$"),
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_title(M_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "mc_ablation"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "mc_ablation"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def sweep_mode(args):
    path = Path(args.gset_dir) / args.instance
    if not path.exists():
        log.error(f"instance file not found: {path}")
        sys.exit(2)
    prob = load_gset(path)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    records = []
    for T in args.Ts:
        handler = attach_file_logger(odir, f"{args.instance}_T{T}")
        try:
            cfg = SolverConfig(
                schedule_shape=args.schedule,
                beta0=args.beta0, betaf=args.betaf,
                n_sweeps=T, update_mode=args.update_mode,
                dynamics="gibbs",
            )
            cfg.validate()
            rec = run_point(prob, cfg, args, point_tag=f"T={T:g}")
            save_results_json(
                rec["results"], odir / f"run_{args.instance}_T{T}.json",
                extras={
                    "instance": args.instance, "point_tag": rec["point_tag"],
                    "n_sweeps": T, "beta0": args.beta0, "betaf": args.betaf,
                    "p_success": rec["p_success"],
                    "tts_99_wall": rec["tts_99_wall"],
                    "cut_best": rec["cut_best"],
                    "cut_median": rec["cut_median"],
                    "cut_bks": rec["cut_bks"],
                    "config": {"n_trials": args.trials,
                               "update_mode": args.update_mode,
                               "master_seed": args.seed,
                               "schedule": args.schedule},
                })
            save_final_states(rec["results"],
                              odir / f"states_{args.instance}_T{T}.txt")
            records.append(rec)
        finally:
            detach_logger(handler)

    if records:
        with open(odir / "sweeps_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "instance", "n_sweeps", "beta0", "betaf",
                "p_success", "tts_99_wall", "time_median",
                "cut_best", "cut_median", "cut_bks",
                "n_trials", "update_mode", "schedule"])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "instance": args.instance,
                    "n_sweeps": r["n_sweeps"],
                    "beta0": r["beta0"], "betaf": r["betaf"],
                    "p_success": r["p_success"],
                    "tts_99_wall": r["tts_99_wall"],
                    "time_median": r["time_median"],
                    "cut_best": r["cut_best"],
                    "cut_median": r["cut_median"],
                    "cut_bks": r["cut_bks"],
                    "n_trials": args.trials,
                    "update_mode": args.update_mode,
                    "schedule": args.schedule,
                })
        plot_scan(records, f"Sweep-budget scan on {args.instance}",
                  "Sweep budget $T$",
                  odir / "sweeps_scan.png")
        log.info(f"done; outputs in {odir}")


def beta_mode(args):
    path = Path(args.gset_dir) / args.instance
    if not path.exists():
        log.error(f"instance file not found: {path}")
        sys.exit(2)
    prob = load_gset(path)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    records = []
    for betaf in args.betaf_values:
        tag = f"bf={betaf}"
        handler = attach_file_logger(odir, f"{args.instance}_bf{betaf}")
        try:
            cfg = SolverConfig(
                schedule_shape=args.schedule,
                beta0=args.beta0, betaf=float(betaf),
                n_sweeps=args.sweeps, update_mode=args.update_mode,
                dynamics="gibbs",
            )
            cfg.validate()
            rec = run_point(prob, cfg, args, point_tag=tag)
            save_results_json(
                rec["results"], odir / f"run_{args.instance}_bf{betaf}.json",
                extras={
                    "instance": args.instance, "point_tag": rec["point_tag"],
                    "n_sweeps": args.sweeps,
                    "beta0": args.beta0, "betaf": float(betaf),
                    "p_success": rec["p_success"],
                    "tts_99_wall": rec["tts_99_wall"],
                    "cut_best": rec["cut_best"],
                    "cut_median": rec["cut_median"],
                    "cut_bks": rec["cut_bks"],
                    "config": {"n_trials": args.trials,
                               "update_mode": args.update_mode,
                               "master_seed": args.seed,
                               "schedule": args.schedule},
                })
            save_final_states(rec["results"],
                              odir / f"states_{args.instance}_bf{betaf}.txt")
            records.append(rec)
        finally:
            detach_logger(handler)

    if records:
        with open(odir / "beta_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "instance", "n_sweeps", "beta0", "betaf",
                "p_success", "tts_99_wall", "time_median",
                "cut_best", "cut_median", "cut_bks",
                "n_trials", "update_mode", "schedule"])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "instance": args.instance,
                    "n_sweeps": r["n_sweeps"],
                    "beta0": r["beta0"], "betaf": r["betaf"],
                    "p_success": r["p_success"],
                    "tts_99_wall": r["tts_99_wall"],
                    "time_median": r["time_median"],
                    "cut_best": r["cut_best"],
                    "cut_median": r["cut_median"],
                    "cut_bks": r["cut_bks"],
                    "n_trials": args.trials,
                    "update_mode": args.update_mode,
                    "schedule": args.schedule,
                })
        plot_scan(records,
                  f"Annealing-endpoint scan on {args.instance} "
                  f"($\\beta_0={args.beta0}$, T={args.sweeps})",
                  r"Final inverse temperature $\beta_f$",
                  odir / "beta_scan.png")
        log.info(f"done; outputs in {odir}")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args shared between subcommands
    def common(sp):
        sp.add_argument("--instance", required=True)
        sp.add_argument("--gset-dir", default="./gset")
        sp.add_argument("--trials", type=int, default=100)
        sp.add_argument("--schedule", default="geometric",
                        choices=["linear", "geometric", "inverse_log"])
        sp.add_argument("--update-mode", default="async_numba",
                        choices=["async_numba", "block", "async_python"])
        sp.add_argument("--jobs", type=int, default=1)
        sp.add_argument("--seed", type=int, default=2024)

    ps_sweep = sub.add_parser("sweeps",
                              help="Scan sweep budget T at fixed beta range")
    common(ps_sweep)
    ps_sweep.add_argument("--Ts", nargs="+", type=int,
                          default=[1000, 3000, 10000, 30000, 100000])
    ps_sweep.add_argument("--beta0", type=float, default=0.1)
    ps_sweep.add_argument("--betaf", type=float, default=10.0)
    ps_sweep.add_argument("--output", default="./results_maxcut_sweeps")

    ps_beta = sub.add_parser("beta",
                             help="Scan final beta at fixed T and beta0")
    common(ps_beta)
    ps_beta.add_argument("--betaf-values", nargs="+", type=float,
                         default=[2.0, 5.0, 10.0, 20.0, 50.0])
    ps_beta.add_argument("--beta0", type=float, default=0.1)
    ps_beta.add_argument("--sweeps", type=int, default=10000)
    ps_beta.add_argument("--output", default="./results_maxcut_beta")

    args = p.parse_args(argv)
    if args.cmd == "sweeps":
        sweep_mode(args)
    else:
        beta_mode(args)


if __name__ == "__main__":
    main()
