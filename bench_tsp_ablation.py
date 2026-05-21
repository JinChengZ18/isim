"""
Ablation study for the TSP benchmark: penalty coefficient A.

The QUBO formulation of TSP (Section 3.1.4) uses a penalty coefficient
A to enforce the row/column constraints (each city visited exactly
once, each position assigned exactly once). For feasibility of the
ground state the inequality A > B * max(d) must hold. In practice the
choice of A trades off two effects:

  * small A (close to the feasibility threshold): the cost term
    B * sum d_ij x_ij x_ij+1 dominates, so the energy landscape
    follows the true tour-length contour. Feasibility is weakly
    enforced and many trials end at infeasible states.

  * large A: constraint violations become energetically prohibitive,
    so nearly all trials end feasible. But the cost term becomes
    negligible, so the solver cannot distinguish good tours from bad
    tours within the feasible region — it finds any feasible tour
    quickly but does not optimize it.

This scan measures both effects by scanning A = A_margin * B * max(d)
over a range of A_margin values on a single instance.

Usage (single line):
    python bench_tsp_ablation.py --instance ulysses16 --A-margins 1.2 1.5 2.0 3.0 5.0 10.0 --trials 100 --sweeps 50000 --update-mode async_numba --jobs 4
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
from problems import (load_tsplib, decode_tour, tour_length,
                      TSP_DEFAULT_MAX_N)
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("tsp_ablation")


TSPLIB_OPT = {
    "burma14":  3323,  "ulysses16": 6859, "gr17":     2085,
    "ulysses22":7013,  "gr24":     1272, "fri26":    937,
    "bayg29":   1610,  "bays29":   2020, "dantzig42":699,
    "swiss42":  1273,  "eil51":    426,  "berlin52": 7542,
}


def _canonical_name(name):
    """Strip any TSPLIB file extension that leaked into prob.name."""
    if not isinstance(name, str):
        return name
    for ext in (".tsp", ".TSP", ".atsp", ".ATSP"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def run_one_A(prob_path, A_margin, cfg, args):
    """Build the TSP problem with A = A_margin * B * D_max and run
    one full multistart batch."""
    # Load once to get D, then rebuild with the specified A
    prob0 = load_tsplib(prob_path, B=args.B,
                        allow_large=args.allow_large)
    D = prob0.meta["D"]
    dmax = float(D.max())
    A_value = A_margin * args.B * dmax
    prob = load_tsplib(prob_path, B=args.B, A=A_value,
                       allow_large=args.allow_large)
    n_cities = prob.meta["n_cities"]

    log.info(f"=== A_margin={A_margin:g} -> A={A_value:.1f}, "
             f"N_spins={prob.n} ===")
    results = multistart(
        prob, cfg, spin_spec=("ideal", {}),
        n_trials=args.trials, master_seed=args.seed,
        n_jobs=args.jobs, progress=True,
    )

    tours, lengths, feas_flags = [], [], []
    for r in results:
        tour, feas = decode_tour(r.state_final, n_cities)
        tours.append(tour)
        feas_flags.append(feas)
        lengths.append(tour_length(tour, D) if feas else np.inf)
    lengths = np.asarray(lengths)
    feas_flags = np.asarray(feas_flags)

    opt_len = TSPLIB_OPT.get(_canonical_name(prob.name))
    if opt_len is None:
        log.warning(f"no published OPT for {prob.name}, using best "
                    f"feasible observed")
        finite = np.isfinite(lengths)
        opt_len = float(lengths[finite].min()) if finite.any() else np.nan
    opt_len = float(opt_len)

    feas_rate = float(feas_flags.mean())
    finite = np.isfinite(lengths)
    if finite.any():
        best_len = float(lengths[finite].min())
        median_len = float(np.median(lengths[finite]))
        gap_best = (best_len - opt_len) / opt_len
        gap_median = (median_len - opt_len) / opt_len
    else:
        best_len = median_len = float("inf")
        gap_best = gap_median = float("inf")

    tol = args.opt_tol
    success_mask = finite & (lengths <= opt_len * (1 + tol))
    p_s = float(success_mask.mean())
    times = np.array([r.wall_time for r in results])
    t_med = float(np.median(times))
    tts = tts_at_confidence(t_med, p_s, 0.99)

    log.info(f"  feasible={feas_rate*100:.1f}%, best_len={best_len:.1f}, "
             f"gap_best={gap_best*100:.2f}%, p_s={p_s:.3f}, "
             f"TTS99={tts:.2f}s")

    return {
        "A_margin": A_margin, "A": A_value, "N_spins": prob.n,
        "feas_rate": feas_rate,
        "best_len": best_len, "median_len": median_len,
        "gap_best": gap_best, "gap_median": gap_median,
        "p_success": p_s, "tts_99_wall": tts, "time_median": t_med,
        "opt_len": opt_len, "tours": tours, "lengths": lengths,
        "feas_flags": feas_flags, "results": results, "prob": prob,
    }


def plot_scan(records, instance_name, out_path):
    """Three-panel-in-one dual-axis scan plot:
       left axis: feasibility rate (hollow bars, back) and gap_best
                  percent (solid bars, front);
       right axis: TTS_99 (markers, log).

    The dual-metric left axis makes the trade-off visible at a glance:
    small A => low feasibility but small gap, large A => high
    feasibility but large gap (or low resolving power). TTS_99 on
    the right traces which A minimises time-to-solution."""
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    A_margins = [r["A_margin"] for r in records]
    feas = np.asarray([r["feas_rate"] * 100.0 for r in records])
    gap = np.asarray([r["gap_best"] * 100.0 for r in records])
    tts = np.asarray([r["tts_99_wall"] for r in records])
    xs = np.arange(len(records))

    # Two sets of narrow bars: feasibility and gap.
    w = 0.36
    bar1 = ax.bar(xs - w / 2, feas, w,
                  color=TSINGHUA_PURPLE["primary"],
                  edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
                  label="Feasibility (%)")
    bar2 = ax.bar(xs + w / 2, np.minimum(gap, 200.0), w,
                  color=TSINGHUA_PURPLE["paler"],
                  edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
                  hatch="///",
                  label="gap$_\\mathrm{best}$ (%)")
    # Annotate gap values above bars (capped bars need exact value label)
    for x, g in zip(xs, gap):
        txt = f">200" if g > 200 else f"{g:.1f}"
        ax.text(x + w / 2, min(g, 200.0) + 3,
                txt, ha="center", va="bottom",
                fontsize=9, color=TSINGHUA_PURPLE["darkest"])
    for x, f_ in zip(xs, feas):
        ax.text(x - w / 2, f_ + 3,
                f"{f_:.0f}", ha="center", va="bottom",
                fontsize=9, color=TSINGHUA_PURPLE["darkest"])

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a:g}" for a in A_margins])
    ax.set_xlabel(r"Penalty margin $A / (B\,d_\mathrm{max})$")
    ax.set_ylabel("Percent (%)", color=TSINGHUA_PURPLE["dark"])
    ax.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["dark"])
    ax.set_ylim(0.0, max(115.0, min(210.0, gap.max() * 1.15)))

    # Right axis: TTS_99
    ax2 = ax.twinx()
    finite = np.isfinite(tts) & (tts > 0)
    tts_plot = np.where(finite, tts, np.nan)
    if finite.any():
        tf = tts[finite]
        ax2.set_ylim(float(tf.min()) / 3.0, float(tf.max()) * 3.0)
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
              label="Feasibility (%)"),
        Patch(facecolor=TSINGHUA_PURPLE["paler"],
              edgecolor=TSINGHUA_PURPLE["dark"],
              hatch="///",
              label=r"gap$_\mathrm{best}$ (%)"),
        Line2D([0], [0], marker="o",
               color=TSINGHUA_PURPLE["accent"],
               markersize=9,
               markeredgecolor=TSINGHUA_PURPLE["darkest"],
               markeredgewidth=0.8,
               label=r"TTS$_{99}$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title(f"Penalty-coefficient scan on {instance_name}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_run(info, args, out_dir):
    prob = info["prob"]
    save_results_json(
        info["results"],
        out_dir / f"run_{prob.name}_A{info['A_margin']:g}.json",
        extras={
            "instance": prob.name, "n_cities": prob.meta["n_cities"],
            "N_spins": prob.n, "A": info["A"], "A_margin": info["A_margin"],
            "B": prob.meta["B"], "opt_len": info["opt_len"],
            "feas_rate": info["feas_rate"],
            "best_len": info["best_len"],
            "median_len": info["median_len"],
            "gap_best": info["gap_best"],
            "p_success": info["p_success"],
            "tts_99_wall": info["tts_99_wall"],
            "tours": [list(map(int, t)) for t in info["tours"]],
            "tour_lengths": [float(x) for x in info["lengths"]],
            "feasibility": [bool(x) for x in info["feas_flags"]],
            "config": {
                "n_trials": args.trials, "n_sweeps": args.sweeps,
                "beta0": args.beta0, "betaf": args.betaf,
                "schedule": args.schedule,
                "update_mode": args.update_mode,
                "master_seed": args.seed,
            },
        })
    save_final_states(
        info["results"],
        out_dir / f"states_{prob.name}_A{info['A_margin']:g}.txt")


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "tsp_ablation"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "tsp_ablation"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--instance", required=True,
                   help="TSPLIB instance name (without .tsp extension)")
    p.add_argument("--tsplib-dir", default="./tsplib")
    p.add_argument("--A-margins", nargs="+", type=float,
                   default=[1.2, 1.5, 2.0, 3.0, 5.0, 10.0])
    p.add_argument("--B", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sweeps", type=int, default=50000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.05)
    p.add_argument("--betaf", type=float, default=20.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    p.add_argument("--opt-tol", type=float, default=0.01)
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_tsp_ablation")
    p.add_argument("--auto-fetch", dest="auto_fetch",
                   action="store_true", default=True)
    p.add_argument("--no-auto-fetch", dest="auto_fetch",
                   action="store_false")
    args = p.parse_args(argv)

    cfg = SolverConfig(
        schedule_shape=args.schedule, beta0=args.beta0, betaf=args.betaf,
        n_sweeps=args.sweeps, update_mode=args.update_mode,
        dynamics="gibbs",
    )
    try:
        cfg.validate()
    except ImportError as e:
        log.error(str(e))
        sys.exit(2)

    tdir = Path(args.tsplib_dir)
    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.auto_fetch:
        try:
            from fetch_data import ensure_tsplib
            ensure_tsplib([args.instance], tdir)
        except Exception as err:
            log.warning(f"auto-fetch warning: {err}")

    candidates = [tdir / args.instance, tdir / f"{args.instance}.tsp",
                  tdir / f"{args.instance}.TSP"]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        log.error(f"no TSPLIB file for {args.instance} in {tdir}")
        sys.exit(2)

    records = []
    for A_margin in args.A_margins:
        handler = attach_file_logger(odir, f"{args.instance}_A{A_margin:g}")
        try:
            info = run_one_A(path, A_margin, cfg, args)
            save_run(info, args, odir)
            records.append(info)
        finally:
            detach_logger(handler)

    if records:
        plot_scan(records, args.instance, odir / "tsp_A_scan.png")
        with open(odir / "tsp_A_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "instance", "A_margin", "A", "N_spins",
                "feas_rate", "best_len", "median_len",
                "gap_best_pct", "gap_median_pct", "p_success",
                "tts_99_wall", "time_median",
                "opt_len", "n_trials", "n_sweeps", "beta0", "betaf",
                "update_mode", "schedule",
            ])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "instance": args.instance, "A_margin": r["A_margin"],
                    "A": r["A"], "N_spins": r["N_spins"],
                    "feas_rate": r["feas_rate"], "best_len": r["best_len"],
                    "median_len": r["median_len"],
                    "gap_best_pct": 100.0 * r["gap_best"],
                    "gap_median_pct": 100.0 * r["gap_median"],
                    "p_success": r["p_success"],
                    "tts_99_wall": r["tts_99_wall"],
                    "time_median": r["time_median"],
                    "opt_len": r["opt_len"],
                    "n_trials": args.trials, "n_sweeps": args.sweeps,
                    "beta0": args.beta0, "betaf": args.betaf,
                    "update_mode": args.update_mode,
                    "schedule": args.schedule,
                })
        log.info(f"done; outputs in {odir}")


if __name__ == "__main__":
    main()
