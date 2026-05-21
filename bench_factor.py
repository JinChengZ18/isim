"""
Benchmark driver for Section 3.4.3: semi-prime integer factoring.

Usage (single line):
    python bench_factor.py --targets 15 21 33 35 51 65 77 91 143 --trials 200 --sweeps 20000
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

from isim import (SolverConfig, multistart, tts_at_confidence,
                  save_results_json, save_final_states, get_logger)
from problems import build_factoring_problem, decode_factors
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("bench_factor")


def run_one_target(M: int, cfg: SolverConfig, args):
    prob = build_factoring_problem(M)
    log.info(f"Built M={M}: N_spins={prob.n} (bp={prob.meta['bp']}, "
             f"bq={prob.meta['bq']}, penalty={prob.meta['penalty']:.1f})")

    results = multistart(
        problem=prob,
        solver_config=cfg,
        spin_spec=("ideal", {}),
        n_trials=args.trials,
        master_seed=args.seed,
        n_jobs=args.jobs,
        progress=True,
    )

    ps_arr, qs_arr, constraint_ok, product_match = [], [], [], []
    for r in results:
        p_val, q_val, ok = decode_factors(r.state_final, prob)
        ps_arr.append(p_val)
        qs_arr.append(q_val)
        constraint_ok.append(ok)
        product_match.append((p_val * q_val == M) and (p_val > 1) and (q_val > 1))
    ps_arr = np.asarray(ps_arr, dtype=int)
    qs_arr = np.asarray(qs_arr, dtype=int)
    constraint_ok = np.asarray(constraint_ok, dtype=bool)
    product_match = np.asarray(product_match, dtype=bool)
    success = constraint_ok & product_match
    p_s = float(success.mean())
    times = np.array([r.wall_time for r in results])
    sweeps = np.array([r.sweeps[-1] for r in results])
    tts_wall = tts_at_confidence(float(np.median(times)), p_s, 0.99)
    tts_sw = tts_at_confidence(float(np.median(sweeps)), p_s, 0.99)

    best_factors = None
    if success.any():
        k = int(np.argmax(success))
        best_factors = (int(ps_arr[k]), int(qs_arr[k]))

    log.info(f"  M={M}: constraint_ok={constraint_ok.mean()*100:.1f}% "
             f"product_match={product_match.mean()*100:.1f}% "
             f"p_success={p_s*100:.2f}%")
    if best_factors:
        log.info(f"  best (p, q) = {best_factors} "
                 f"(target {M} = {best_factors[0]} x {best_factors[1]})")
    log.info(f"  time_median={np.median(times):.2f}s TTS_99={tts_wall:.2f}s")

    return {
        "M": M, "prob": prob, "results": results,
        "ps": ps_arr, "qs": qs_arr,
        "constraint_ok": constraint_ok, "product_match": product_match,
        "success": success, "p_success": p_s,
        "tts_99_wall": tts_wall, "tts_99_sweeps": tts_sw,
        "time_median": float(np.median(times)),
        "best_factors": best_factors,
    }


def plot_success_bar(infos, out_path):
    """Dual-axis chart: success probability (bars, left) and
    TTS_99 (markers + line, right, log scale). TTS_99 is the
    principal scalability metric; p_s is shown as supplementary.

    Handles sparse-data edge cases (very low p_s, only a few finite
    TTS_99 values) via adaptive ylim and explicit `n/a` markers."""
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(max(6.2, 0.9 * len(infos) + 1.6), 4.0))
    labels = [f"M={info['M']}" for info in infos]
    p_vals = np.asarray([info["p_success"] for info in infos])
    tts_vals = np.asarray([info["tts_99_wall"] for info in infos])
    xs = np.arange(len(infos))

    # Left axis: adaptive ylim so low bars remain visible
    bar_colors = [TSINGHUA_PURPLE["primary"] if p > 0
                  else TSINGHUA_PURPLE["gray_lt"] for p in p_vals]
    ax.bar(xs, p_vals, color=bar_colors,
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Success probability $p_s$",
                  color=TSINGHUA_PURPLE["dark"])
    ax.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["dark"])
    p_max = float(p_vals.max())
    if p_max <= 0:
        y_top = 0.05
    elif p_max < 0.05:
        y_top = max(p_max * 2.0, 0.02)
    else:
        y_top = min(1.05, p_max * 1.30)
    ax.set_ylim(0.0, y_top)
    for x, p_val in zip(xs, p_vals):
        if p_val > 0:
            ax.text(x, p_val + y_top * 0.02,
                    f"{p_val:.3g}" if p_val < 0.01 else f"{p_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, color=TSINGHUA_PURPLE["darkest"])

    # Right axis: explicit log range from finite values
    ax2 = ax.twinx()
    finite = np.isfinite(tts_vals) & (tts_vals > 0)
    tts_plot = np.where(finite, tts_vals, np.nan)
    if finite.any():
        tts_finite = tts_vals[finite]
        lo = float(tts_finite.min()) / 3.0
        hi = float(tts_finite.max()) * 3.0
        if hi <= lo * 1.5:
            lo, hi = lo / 3.0, hi * 3.0
        ax2.set_ylim(lo, hi)
    ax2.set_yscale("log")
    ax2.plot(xs, tts_plot, marker="o", linestyle="-",
             color=TSINGHUA_PURPLE["accent"],
             markersize=8, markeredgecolor=TSINGHUA_PURPLE["darkest"],
             markeredgewidth=0.8, linewidth=1.6)
    ax2.set_ylabel(r"TTS$_{99}$ (s)", color=TSINGHUA_PURPLE["accent"])
    ax2.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["accent"])
    ax2.grid(False)

    if finite.any() and (~finite).any():
        ylo, yhi = ax2.get_ylim()
        y_marker = yhi / 1.8
        y_label = yhi / 4.5
        for i, ok in enumerate(finite):
            if not ok:
                ax2.plot(i, y_marker, marker="o",
                         markersize=8, markerfacecolor="white",
                         markeredgecolor=TSINGHUA_PURPLE["gray"],
                         markeredgewidth=1.2, linestyle="none")
                ax2.text(i, y_label, "n/a",
                         ha="center", va="center",
                         fontsize=9, color=TSINGHUA_PURPLE["gray"])

    legend_elements = [
        Patch(facecolor=TSINGHUA_PURPLE["primary"],
              edgecolor=TSINGHUA_PURPLE["dark"],
              label="Success probability"),
        Line2D([0], [0], marker="o",
               color=TSINGHUA_PURPLE["accent"],
               markersize=8,
               markeredgecolor=TSINGHUA_PURPLE["darkest"],
               markeredgewidth=0.8,
               label=r"TTS$_{99}$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title("Factoring success and time-to-solution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_run(info, args, out_dir):
    M = info["M"]
    save_results_json(info["results"], out_dir / f"run_M{M}.json",
                      extras={
                          "M": M,
                          "bp": info["prob"].meta["bp"],
                          "bq": info["prob"].meta["bq"],
                          "penalty": info["prob"].meta["penalty"],
                          "N_spins": info["prob"].n,
                          "p_success": info["p_success"],
                          "tts_99_wall": info["tts_99_wall"],
                          "tts_99_sweeps": info["tts_99_sweeps"],
                          "decoded_p": [int(x) for x in info["ps"]],
                          "decoded_q": [int(x) for x in info["qs"]],
                          "constraint_ok": [bool(x) for x in info["constraint_ok"]],
                          "product_match": [bool(x) for x in info["product_match"]],
                          "config": {
                              "n_trials": args.trials,
                              "n_sweeps": args.sweeps,
                              "schedule": args.schedule,
                              "beta0": args.beta0,
                              "betaf": args.betaf,
                              "update_mode": args.update_mode,
                              "dynamics": args.dynamics,
                              "master_seed": args.seed,
                          },
                      })
    save_final_states(info["results"], out_dir / f"states_M{M}.txt")


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "bench_factor"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "bench_factor"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", type=int,
                   default=[15, 21, 33, 35, 51, 65, 77, 91, 143])
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--sweeps", type=int, default=20000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.05)
    p.add_argument("--betaf", type=float, default=30.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    p.add_argument("--dynamics", default="gibbs",
                   choices=["gibbs", "metropolis"])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_factor")
    args = p.parse_args(argv)

    cfg = SolverConfig(
        schedule_shape=args.schedule, beta0=args.beta0, betaf=args.betaf,
        n_sweeps=args.sweeps, update_mode=args.update_mode,
        dynamics=args.dynamics,
    )
    try:
        cfg.validate()
    except ImportError as e:
        log.error(str(e))
        sys.exit(2)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    infos = []
    rows = []
    for M in args.targets:
        handler = attach_file_logger(odir, f"M{M}")
        try:
            info = run_one_target(M, cfg, args)
            infos.append(info)
            save_run(info, args, odir)
            row = {
                "M": M,
                "bp": info["prob"].meta["bp"],
                "bq": info["prob"].meta["bq"],
                "N_spins": info["prob"].n,
                "penalty": info["prob"].meta["penalty"],
                "p_success": info["p_success"],
                "constraint_rate": float(info["constraint_ok"].mean()),
                "product_match_rate": float(info["product_match"].mean()),
                "tts99_wall": info["tts_99_wall"],
                "tts99_sweeps": info["tts_99_sweeps"],
                "time_median": info["time_median"],
                "best_factors": (f"{info['best_factors']}"
                                 if info["best_factors"] else "None"),
                "n_trials": args.trials,
                "n_sweeps": args.sweeps,
                "update_mode": args.update_mode,
                "dynamics": args.dynamics,
                "schedule": args.schedule,
                "beta0": args.beta0,
                "betaf": args.betaf,
            }
            rows.append(row)
        finally:
            detach_logger(handler)

    if infos:
        plot_success_bar(infos, odir / "bar_success.png")

    if rows:
        csv_path = odir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Summary written to {csv_path}")


if __name__ == "__main__":
    main()
