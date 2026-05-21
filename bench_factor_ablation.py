"""
Ablation study for the factoring benchmark.

For a fixed target M, scan the bit budget allocated to the larger
factor (bq) while keeping all other hyperparameters fixed, and measure
single-trial success probability. This isolates the effect of encoding
size from solver dynamics.

Two outputs:
- Main scan plot: p_success vs bq, with N_spins annotated.
- Density plot for the underfit case: (p_hat, q_hat) distribution of
  constraint-satisfying trials, revealing the pseudo-optima that the
  solver converges to when the true factors cannot be represented.

Usage (single line):
    python bench_factor_ablation.py --M 51 --bq-range 4 7 --trials 200 --sweeps 20000 --update-mode async_numba --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isim import (SolverConfig, multistart, save_results_json,
                  save_final_states, get_logger, tts_at_confidence)
from problems import (build_factoring_problem, decode_factors,
                      suggest_factoring_bp_bq)
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("ablation")


def run_one_bq(M, bp, bq, cfg, args):
    """Build the factoring problem with a specified bit budget and run
    the full trial batch. Returns a record dict or None on failure."""
    try:
        prob = build_factoring_problem(M, bp=bp, bq=bq)
    except Exception as e:
        log.error(f"cannot build M={M}, bp={bp}, bq={bq}: {e}")
        return None

    log.info(f"M={M} bq={bq}: bp={prob.meta['bp']}, N_spins={prob.n}, "
             f"penalty={prob.meta['penalty']:.0f}")

    results = multistart(
        prob, cfg, spin_spec=("ideal", {}),
        n_trials=args.trials, master_seed=args.seed,
        n_jobs=args.jobs, progress=True,
    )

    ps_arr, qs_arr, ok_arr, match_arr = [], [], [], []
    for r in results:
        p_val, q_val, ok = decode_factors(r.state_final, prob)
        ps_arr.append(p_val)
        qs_arr.append(q_val)
        ok_arr.append(ok)
        match_arr.append(ok and p_val * q_val == M and p_val > 1 and q_val > 1)

    ps_arr = np.asarray(ps_arr, dtype=int)
    qs_arr = np.asarray(qs_arr, dtype=int)
    ok_arr = np.asarray(ok_arr, dtype=bool)
    match_arr = np.asarray(match_arr, dtype=bool)

    p_s = float(match_arr.mean())
    constraint_rate = float(ok_arr.mean())
    times = np.array([r.wall_time for r in results])
    tts = tts_at_confidence(float(np.median(times)), p_s, 0.99)

    log.info(f"  p_success={p_s*100:.2f}% "
             f"constraint_rate={constraint_rate*100:.1f}% "
             f"TTS99={tts:.2f}s")

    return {
        "M": M, "bp": prob.meta["bp"], "bq": bq,
        "N_spins": prob.n, "penalty": prob.meta["penalty"],
        "p_success": p_s, "constraint_rate": constraint_rate,
        "tts_99_wall": tts, "time_median": float(np.median(times)),
        "ps_decoded": ps_arr, "qs_decoded": qs_arr,
        "ok_arr": ok_arr, "match_arr": match_arr,
        "prob": prob, "results": results,
    }


def plot_scan(records, M, out_path):
    """Dual-axis bit-budget scan: success probability (bars, left)
    and TTS_99 (markers + line, right, log scale).

    Handles sparse-data edge cases via adaptive ylim, bar-top p_s
    annotations, and explicit hollow-marker + `n/a` labels for points
    where TTS_99 is not defined."""
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    bq_list = [r["bq"] for r in records]
    p_list = np.asarray([r["p_success"] for r in records])
    N_list = [r["N_spins"] for r in records]
    tts_list = np.asarray([r["tts_99_wall"] for r in records])
    xs = np.arange(len(bq_list))

    # Left axis: bars with adaptive ylim
    bar_colors = [TSINGHUA_PURPLE["gray_lt"] if p == 0
                  else TSINGHUA_PURPLE["primary"] for p in p_list]
    ax.bar(xs, p_list, color=bar_colors,
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.9)
    p_max = float(p_list.max())
    if p_max <= 0:
        y_top = 0.05
    elif p_max < 0.05:
        y_top = max(p_max * 2.0, 0.02)
    else:
        y_top = min(1.05, p_max * 1.35)
    ax.set_ylim(0.0, y_top)
    # Bar annotations: N= on top of each (moved up to avoid collision
    # with the p_s values just above), p_s value inside the bar at a
    # small offset above its top.
    for x, p_val, N in zip(xs, p_list, N_list):
        ax.text(x, y_top * 0.97, f"N={N}",
                ha="center", va="top",
                fontsize=11, color=TSINGHUA_PURPLE["darkest"])
        if p_val > 0:
            ax.text(x, p_val + y_top * 0.02,
                    f"{p_val:.3g}" if p_val < 0.01 else f"{p_val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, color=TSINGHUA_PURPLE["darkest"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$b_q={bq}$" for bq in bq_list])
    ax.set_ylabel("Success probability $p_s$",
                  color=TSINGHUA_PURPLE["dark"])
    ax.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["dark"])

    # Right axis: explicit log range from finite TTS values
    ax2 = ax.twinx()
    finite = np.isfinite(tts_list) & (tts_list > 0)
    tts_plot = np.where(finite, tts_list, np.nan)
    if finite.any():
        tts_finite = tts_list[finite]
        lo = float(tts_finite.min()) / 3.0
        hi = float(tts_finite.max()) * 3.0
        if hi <= lo * 1.5:
            lo, hi = lo / 3.0, hi * 3.0
        ax2.set_ylim(lo, hi)
    ax2.set_yscale("log")
    ax2.plot(xs, tts_plot, marker="o", linestyle="-",
             color=TSINGHUA_PURPLE["accent"],
             markersize=9, markeredgecolor=TSINGHUA_PURPLE["darkest"],
             markeredgewidth=0.8, linewidth=1.8)
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
                         markersize=9, markerfacecolor="white",
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
    ax.set_title(f"Bit-budget scan at M={M} "
                 f"($b_p={records[0]['bp']}$ fixed)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_density(record, M, out_path):
    """2D bubble plot of decoded (p, q) for constraint-satisfying
    trials, with the hyperbola p*q = M overlaid."""
    mask = record["ok_arr"]
    ps = record["ps_decoded"][mask]
    qs = record["qs_decoded"][mask]
    if len(ps) == 0:
        log.warning("No constraint-satisfying trials; skip density plot")
        return

    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    counter = Counter(zip(ps.tolist(), qs.tolist()))
    pairs = list(counter.keys())
    counts = np.array([counter[k] for k in pairs])
    xs = np.array([k[0] for k in pairs])
    ys = np.array([k[1] for k in pairs])

    sizes = 40 + 30 * counts
    sc = ax.scatter(xs, ys, s=sizes, c=counts,
                    cmap="Purples", vmin=0,
                    edgecolor=TSINGHUA_PURPLE["dark"],
                    linewidth=0.8, alpha=0.9)

    # True-factorization hyperbola
    x_max_grid = float(max(xs.max(), 1) * 1.3)
    x_curve = np.linspace(1.0, x_max_grid, 200)
    y_curve = M / x_curve
    ax.plot(x_curve, y_curve, "--",
            color=TSINGHUA_PURPLE["accent"], lw=1.6,
            label=f"$p q = {M}$")

    # Mark the true factors if they fit on the plot
    true_p, true_q = suggest_factoring_bp_bq_check(M)
    for tp, tq in [(true_p, true_q), (true_q, true_p)]:
        ax.plot([tp], [tq], marker="*",
                markersize=18, color=TSINGHUA_PURPLE["accent"],
                markeredgecolor=TSINGHUA_PURPLE["darkest"],
                markeredgewidth=0.8, linestyle="none",
                label=f"true factor" if (tp, tq) == (true_p, true_q) else None)

    ax.set_xlabel(r"decoded $\hat p$")
    ax.set_ylabel(r"decoded $\hat q$")
    ax.set_title(f"$(p,q)$ convergence at $b_q={record['bq']}$, M={M}")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Trial count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def suggest_factoring_bp_bq_check(M):
    """Just the trial-division factor finder."""
    for p in range(3, int(np.sqrt(M)) + 1, 2):
        if M % p == 0:
            return p, M // p
    return 1, M


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "ablation"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "ablation"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=51,
                   help="Semi-prime target to factor")
    p.add_argument("--bq-range", nargs=2, type=int, default=[4, 7],
                   help="Inclusive scan range for bq (min max)")
    p.add_argument("--bp", type=int, default=None,
                   help="Fix bp (default: auto from true factor)")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--sweeps", type=int, default=20000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.05)
    p.add_argument("--betaf", type=float, default=30.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_factor_ablation")
    args = p.parse_args(argv)

    cfg = SolverConfig(
        schedule_shape=args.schedule,
        beta0=args.beta0, betaf=args.betaf,
        n_sweeps=args.sweeps, update_mode=args.update_mode,
        dynamics="gibbs",
    )
    try:
        cfg.validate()
    except ImportError as e:
        log.error(str(e))
        sys.exit(2)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    # Resolve bp
    if args.bp is None:
        try:
            bp_auto, _ = suggest_factoring_bp_bq(args.M)
            bp = bp_auto
        except ValueError as e:
            log.error(f"cannot auto-resolve bp for M={args.M}: {e}")
            sys.exit(2)
    else:
        bp = args.bp

    log.info(f"Ablation: M={args.M}, bp={bp} (fixed), "
             f"bq ∈ [{args.bq_range[0]}, {args.bq_range[1]}]")

    records = []
    for bq in range(args.bq_range[0], args.bq_range[1] + 1):
        handler = attach_file_logger(odir, f"M{args.M}_bq{bq}")
        try:
            rec = run_one_bq(args.M, bp, bq, cfg, args)
            if rec is None:
                continue
            save_results_json(
                rec["results"], odir / f"run_M{args.M}_bq{bq}.json",
                extras={
                    "M": args.M, "bp": rec["bp"], "bq": bq,
                    "N_spins": rec["N_spins"],
                    "penalty": rec["penalty"],
                    "p_success": rec["p_success"],
                    "constraint_rate": rec["constraint_rate"],
                    "decoded_p": rec["ps_decoded"].tolist(),
                    "decoded_q": rec["qs_decoded"].tolist(),
                    "config": {
                        "n_trials": args.trials, "n_sweeps": args.sweeps,
                        "beta0": args.beta0, "betaf": args.betaf,
                        "schedule": args.schedule,
                        "update_mode": args.update_mode,
                        "master_seed": args.seed,
                    },
                })
            save_final_states(rec["results"],
                              odir / f"states_M{args.M}_bq{bq}.txt")
            records.append(rec)
        finally:
            detach_logger(handler)

    # Summary CSV
    if records:
        rows = [{
            "M": r["M"], "bp": r["bp"], "bq": r["bq"],
            "N_spins": r["N_spins"], "penalty": r["penalty"],
            "p_success": r["p_success"],
            "constraint_rate": r["constraint_rate"],
            "tts99_wall": r["tts_99_wall"],
            "time_median": r["time_median"],
            "n_trials": args.trials, "n_sweeps": args.sweeps,
        } for r in records]
        with open(odir / "ablation_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Scan plot
        plot_scan(records, args.M, odir / "ablation_scan.png")

        # Density plot: pick the smallest bq that showed zero success
        # (clearest underfit case); fall back to the smallest scanned bq.
        zero_records = [r for r in records if r["p_success"] == 0]
        target_rec = zero_records[0] if zero_records else records[0]
        plot_density(target_rec, args.M,
                     odir / f"ablation_density_bq{target_rec['bq']}.png")

        log.info(f"Done. Outputs in {odir}")


if __name__ == "__main__":
    main()
