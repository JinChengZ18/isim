"""
Side-by-side benchmark: QUBO single-spin Ising vs permutation-space
cluster-flip SA on TSPLIB instances.

Runs both methods on the same instance(s) with the same trial count
and random seeds, and produces a direct comparison of feasibility
rate, best/median gap, p_success, and TTS_99. Figures include:

  * comparison bar chart across instances (gap_best on linear axis,
    TTS_99 on log right axis), same dual-axis convention as the other
    drivers;
  * per-instance convergence traces overlaying the two methods.

Usage (single line):
    python bench_tsp_compare.py --instances burma14 ulysses16 gr17 --trials 100 --qubo-sweeps 50000 --perm-sweeps 5000 --update-mode async_numba --jobs 4
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
                  save_results_json, get_logger)
from problems import (load_tsplib, decode_tour, tour_length,
                      TSP_DEFAULT_MAX_N)
from perm_tsp import (PermutationSolverConfig, multistart_permutation,
                      summarize_permutation_runs)
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("tsp_compare")


TSPLIB_OPT = {
    "burma14":  3323,  "ulysses16": 6859, "gr17":     2085,
    "ulysses22":7013,  "gr24":     1272, "fri26":    937,
    "bayg29":   1610,  "bays29":   2020, "dantzig42":699,
    "swiss42":  1273,  "eil51":    426,  "berlin52": 7542,
}


def _canonical_name(name):
    """Strip any TSPLIB file extension that leaked into prob.name
    (caused by mirrors that pollute the NAME header with the
    filename). Acts as a safety net in addition to the
    _parse_tsplib fix; allows the lookup to succeed even if an old
    cached problem object is reused."""
    if not isinstance(name, str):
        return name
    for ext in (".tsp", ".TSP", ".atsp", ".ATSP"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def run_qubo(prob_path, args):
    """Run the QUBO single-spin Ising baseline."""
    prob0 = load_tsplib(prob_path, B=args.B,
                        allow_large=args.allow_large)
    D = prob0.meta["D"]
    A = args.A_margin * args.B * float(D.max())
    prob = load_tsplib(prob_path, B=args.B, A=A,
                       allow_large=args.allow_large)
    n_cities = prob.meta["n_cities"]

    cfg = SolverConfig(
        schedule_shape=args.schedule,
        beta0=args.qubo_beta0, betaf=args.qubo_betaf,
        n_sweeps=args.qubo_sweeps, update_mode=args.update_mode,
        dynamics="gibbs",
    )
    cfg.validate()
    log.info(f"--- QUBO on {prob.name}: N_spins={prob.n}, A={A:.1f}, "
             f"B={args.B}, T={args.qubo_sweeps} ---")
    results = multistart(prob, cfg, spin_spec=("ideal", {}),
                         n_trials=args.trials, master_seed=args.seed,
                         n_jobs=args.jobs, progress=True)

    tours, lengths, feas_flags = [], [], []
    for r in results:
        tour, feas = decode_tour(r.state_final, n_cities)
        tours.append(tour)
        feas_flags.append(feas)
        lengths.append(tour_length(tour, D) if feas else np.inf)
    lengths = np.asarray(lengths)
    feas_flags = np.asarray(feas_flags)
    opt_len = float(TSPLIB_OPT.get(_canonical_name(prob.name), np.nan))

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

    success_mask = finite & (lengths <= opt_len * (1 + args.opt_tol))
    p_s = float(success_mask.mean())
    times = np.array([r.wall_time for r in results])
    t_med = float(np.median(times))
    tts = tts_at_confidence(t_med, p_s, 0.99)

    return {
        "method": "qubo", "prob": prob, "opt_len": opt_len,
        "feas_rate": feas_rate, "best_len": best_len,
        "median_len": median_len, "gap_best": gap_best,
        "gap_median": gap_median, "p_success": p_s,
        "tts_99_wall": tts, "time_median": t_med,
        "lengths": lengths, "feas_flags": feas_flags,
        "tours": tours, "results": results, "D": D,
    }


def run_permutation(prob_path, args):
    """Run the permutation-space cluster-flip baseline."""
    prob = load_tsplib(prob_path, B=args.B,
                       allow_large=args.allow_large)
    D = prob.meta["D"]
    opt_len = float(TSPLIB_OPT.get(_canonical_name(prob.name), np.nan))

    cfg = PermutationSolverConfig(
        schedule_shape=args.schedule,
        beta0=args.perm_beta0, betaf=args.perm_betaf,
        n_sweeps=args.perm_sweeps,
    )
    cfg.validate()
    log.info(f"--- PERM on {prob.name}: n_cities={D.shape[0]}, "
             f"T={args.perm_sweeps} ---")
    results = multistart_permutation(
        D, cfg, n_trials=args.trials,
        master_seed=args.seed, n_jobs=args.jobs, progress=True,
    )
    summary = summarize_permutation_runs(results, opt_len,
                                          tol=args.opt_tol)
    return {
        "method": "perm", "prob": prob, "opt_len": opt_len,
        "feas_rate": 1.0,   # permutation-space is feasible by construction
        "best_len": summary["length_best"],
        "median_len": summary["length_median"],
        "gap_best": summary["gap_best"],
        "gap_median": summary["gap_median"],
        "p_success": summary["p_success"],
        "tts_99_wall": summary["tts_99_wall"],
        "time_median": summary["time_median"],
        "lengths": np.array([r.length_best for r in results]),
        "feas_flags": np.ones(len(results), dtype=bool),
        "accept_rate": summary["accept_rate"],
        "results": results, "D": D,
    }


def plot_compare_bars(qubo_infos, perm_infos, out_path):
    """Dual-axis comparison: gap_best (bars, left linear) and TTS_99
    (markers, right log). QUBO vs permutation side-by-side.

    Robust to NaN gap values (which indicate that the published OPT
    was not available for lookup): the corresponding bar is drawn as
    zero-height with a `?` annotation to signal missing reference,
    rather than leaving a silent blank column."""
    set_style()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(max(6.4, 1.2 * len(qubo_infos) + 2.0), 4.2))
    labels = [_canonical_name(info["prob"].name) for info in qubo_infos]
    gap_q = np.array([info["gap_best"] * 100.0 for info in qubo_infos])
    gap_p = np.array([info["gap_best"] * 100.0 for info in perm_infos])
    tts_q = np.array([info["tts_99_wall"] for info in qubo_infos])
    tts_p = np.array([info["tts_99_wall"] for info in perm_infos])
    xs = np.arange(len(labels))
    w = 0.36

    # Replace NaNs with 0 for bar plotting, then annotate `?`
    gap_q_plot_raw = np.where(np.isnan(gap_q), 0.0, np.minimum(gap_q, 200.0))
    gap_p_plot_raw = np.where(np.isnan(gap_p), 0.0, np.minimum(gap_p, 200.0))
    ax.bar(xs - w / 2, gap_q_plot_raw, w,
           color=TSINGHUA_PURPLE["gray_lt"],
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
           label=r"QUBO gap$_\mathrm{best}$")
    ax.bar(xs + w / 2, gap_p_plot_raw, w,
           color=TSINGHUA_PURPLE["primary"],
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
           label=r"Perm gap$_\mathrm{best}$")
    for x, g_q, g_p in zip(xs, gap_q, gap_p):
        if np.isnan(g_q):
            q_txt = "?"
        elif g_q > 200:
            q_txt = ">200"
        else:
            q_txt = f"{g_q:.1f}"
        if np.isnan(g_p):
            p_txt = "?"
        elif g_p > 200:
            p_txt = ">200"
        else:
            p_txt = f"{g_p:.2f}"
        y_q = 0.0 if np.isnan(g_q) else min(g_q, 200.0)
        y_p = 0.0 if np.isnan(g_p) else min(g_p, 200.0)
        ax.text(x - w / 2, y_q + 3, q_txt,
                ha="center", va="bottom",
                fontsize=9, color=TSINGHUA_PURPLE["darkest"])
        ax.text(x + w / 2, y_p + 3, p_txt,
                ha="center", va="bottom",
                fontsize=9, color=TSINGHUA_PURPLE["darkest"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"gap$_\mathrm{best}$ (%)",
                  color=TSINGHUA_PURPLE["dark"])
    ax.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["dark"])
    valid = np.concatenate([gap_q[~np.isnan(gap_q)],
                            gap_p[~np.isnan(gap_p)]])
    if valid.size > 0:
        ylim = max(valid.max() * 1.15, 25.0)
    else:
        ylim = 50.0
    ax.set_ylim(0.0, min(ylim, 210.0))

    # TTS_99 right axis
    ax2 = ax.twinx()
    all_tts = np.concatenate([tts_q, tts_p])
    finite_all = np.isfinite(all_tts) & (all_tts > 0)
    if finite_all.any():
        tf = all_tts[finite_all]
        ax2.set_ylim(float(tf.min()) / 3.0, float(tf.max()) * 3.0)
    ax2.set_yscale("log")
    # QUBO markers (hollow)
    for i, t in enumerate(tts_q):
        if np.isfinite(t) and t > 0:
            ax2.plot(i - w / 2, t, marker="s", markersize=9,
                     markerfacecolor="white",
                     markeredgecolor=TSINGHUA_PURPLE["accent"],
                     markeredgewidth=1.5, linestyle="none")
        else:
            # Infinite: hollow marker near top of axis
            ylo, yhi = ax2.get_ylim()
            ax2.plot(i - w / 2, yhi / 1.8, marker="s", markersize=9,
                     markerfacecolor="white",
                     markeredgecolor=TSINGHUA_PURPLE["gray"],
                     markeredgewidth=1.2, linestyle="none")
    # Perm markers (filled)
    for i, t in enumerate(tts_p):
        if np.isfinite(t) and t > 0:
            ax2.plot(i + w / 2, t, marker="o", markersize=9,
                     markerfacecolor=TSINGHUA_PURPLE["accent"],
                     markeredgecolor=TSINGHUA_PURPLE["darkest"],
                     markeredgewidth=0.8, linestyle="none")
        else:
            ylo, yhi = ax2.get_ylim()
            ax2.plot(i + w / 2, yhi / 1.8, marker="o", markersize=9,
                     markerfacecolor="white",
                     markeredgecolor=TSINGHUA_PURPLE["gray"],
                     markeredgewidth=1.2, linestyle="none")
    ax2.set_ylabel(r"TTS$_{99}$ (s)",
                   color=TSINGHUA_PURPLE["accent"])
    ax2.tick_params(axis="y", labelcolor=TSINGHUA_PURPLE["accent"])
    ax2.grid(False)

    legend_elements = [
        Patch(facecolor=TSINGHUA_PURPLE["gray_lt"],
              edgecolor=TSINGHUA_PURPLE["dark"],
              label="QUBO gap"),
        Patch(facecolor=TSINGHUA_PURPLE["primary"],
              edgecolor=TSINGHUA_PURPLE["dark"],
              label="Perm gap"),
        Line2D([0], [0], marker="s", color="none",
               markersize=9,
               markeredgecolor=TSINGHUA_PURPLE["accent"],
               markerfacecolor="white", markeredgewidth=1.5,
               label=r"QUBO TTS$_{99}$"),
        Line2D([0], [0], marker="o", color="none",
               markersize=9,
               markeredgecolor=TSINGHUA_PURPLE["darkest"],
               markerfacecolor=TSINGHUA_PURPLE["accent"],
               markeredgewidth=0.8,
               label=r"Perm TTS$_{99}$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=2,
              fontsize=10)
    ax.set_title("QUBO vs permutation-space SA")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_convergence(qubo_info, perm_info, out_path):
    """Per-instance convergence overlay. Left: QUBO energy trajectory
    (unnormalized QUBO Hamiltonian); right: permutation tour length."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    display_name = _canonical_name(qubo_info["prob"].name)

    # Left: QUBO energy (use median trial)
    q_results = qubo_info["results"]
    finals = np.array([r.energy_final for r in q_results])
    pick_q = np.argsort(finals)[len(finals) // 2]
    rq = q_results[pick_q]
    ax1.plot(np.maximum(rq.sweeps, 1), rq.energies,
             color=TSINGHUA_PURPLE["dark"], lw=1.8, label="QUBO energy")
    ax1.set_xscale("log")
    ax1.set_xlabel("Sweeps")
    ax1.set_ylabel("Energy (QUBO Hamiltonian)")
    ax1.set_title(f"{display_name}: QUBO trajectory (median trial)")
    ax1.legend(loc="best")

    # Right: permutation tour length
    p_results = perm_info["results"]
    finals = np.array([r.length_best for r in p_results])
    pick_p = np.argsort(finals)[len(finals) // 2]
    rp = p_results[pick_p]
    ax2.plot(np.maximum(rp.sweeps, 1), rp.lengths,
             color=TSINGHUA_PURPLE["primary"], lw=1.8,
             label="Perm tour length")
    if np.isfinite(perm_info["opt_len"]):
        ax2.axhline(perm_info["opt_len"],
                    color=TSINGHUA_PURPLE["accent"], linestyle="--",
                    linewidth=1.5,
                    label=f"OPT = {perm_info['opt_len']:g}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sweeps")
    ax2.set_ylabel("Tour length")
    ax2.set_title(f"{display_name}: perm trajectory (median trial)")
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_compare_run(qubo_info, perm_info, args, out_dir):
    prob_name = _canonical_name(qubo_info["prob"].name)
    save_results_json(
        qubo_info["results"], out_dir / f"qubo_{prob_name}.json",
        extras={
            "method": "qubo", "instance": prob_name,
            "n_cities": qubo_info["prob"].meta["n_cities"],
            "N_spins": qubo_info["prob"].n,
            "opt_len": qubo_info["opt_len"],
            "feas_rate": qubo_info["feas_rate"],
            "gap_best": qubo_info["gap_best"],
            "p_success": qubo_info["p_success"],
            "tts_99_wall": qubo_info["tts_99_wall"],
            "tour_lengths": [float(x) for x in qubo_info["lengths"]],
            "feasibility": [bool(x) for x in qubo_info["feas_flags"]],
            "config": {
                "n_trials": args.trials, "n_sweeps": args.qubo_sweeps,
                "beta0": args.qubo_beta0, "betaf": args.qubo_betaf,
                "A_margin": args.A_margin, "B": args.B,
                "update_mode": args.update_mode,
            },
        })
    import json
    perm_payload = {
        "meta": {
            "method": "perm", "instance": prob_name,
            "opt_len": perm_info["opt_len"],
            "feas_rate": perm_info["feas_rate"],
            "gap_best": perm_info["gap_best"],
            "p_success": perm_info["p_success"],
            "tts_99_wall": perm_info["tts_99_wall"],
            "accept_rate": perm_info["accept_rate"],
            "config": {
                "n_trials": args.trials, "n_sweeps": args.perm_sweeps,
                "beta0": args.perm_beta0, "betaf": args.perm_betaf,
            },
        },
        "runs": [r.to_dict() for r in perm_info["results"]],
    }
    (out_dir / f"perm_{prob_name}.json").write_text(
        json.dumps(perm_payload, indent=2))


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "permtsp", "tsp_compare"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "permtsp", "tsp_compare"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+",
                   default=["burma14", "ulysses16", "gr17"])
    p.add_argument("--tsplib-dir", default="./tsplib")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    # QUBO
    p.add_argument("--qubo-sweeps", type=int, default=50000)
    p.add_argument("--qubo-beta0", type=float, default=0.05)
    p.add_argument("--qubo-betaf", type=float, default=20.0)
    p.add_argument("--A-margin", type=float, default=2.0)
    p.add_argument("--B", type=float, default=1.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    # Permutation
    p.add_argument("--perm-sweeps", type=int, default=5000)
    p.add_argument("--perm-beta0", type=float, default=0.1)
    p.add_argument("--perm-betaf", type=float, default=50.0)
    # Common
    p.add_argument("--opt-tol", type=float, default=0.01)
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_tsp_compare")
    p.add_argument("--auto-fetch", dest="auto_fetch",
                   action="store_true", default=True)
    p.add_argument("--no-auto-fetch", dest="auto_fetch",
                   action="store_false")
    args = p.parse_args(argv)

    tdir = Path(args.tsplib_dir)
    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.auto_fetch:
        try:
            from fetch_data import ensure_tsplib
            ensure_tsplib(args.instances, tdir)
        except Exception as err:
            log.warning(f"auto-fetch warning: {err}")

    qubo_infos = []
    perm_infos = []
    rows = []
    for inst in args.instances:
        candidates = [tdir / inst, tdir / f"{inst}.tsp", tdir / f"{inst}.TSP"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            log.error(f"[skip] no TSPLIB file for {inst}")
            continue
        handler = attach_file_logger(odir, inst)
        try:
            qubo_info = run_qubo(path, args)
            perm_info = run_permutation(path, args)
            qubo_infos.append(qubo_info)
            perm_infos.append(perm_info)
            save_compare_run(qubo_info, perm_info, args, odir)
            plot_convergence(qubo_info, perm_info,
                             odir / f"trace_compare_{inst}.png")

            rows.append({
                "instance": inst,
                "n_cities": qubo_info["prob"].meta["n_cities"],
                "N_spins_qubo": qubo_info["prob"].n,
                "opt_len": qubo_info["opt_len"],
                # QUBO
                "qubo_feas_rate": qubo_info["feas_rate"],
                "qubo_best_len": qubo_info["best_len"],
                "qubo_gap_best_pct": 100.0 * qubo_info["gap_best"],
                "qubo_p_success": qubo_info["p_success"],
                "qubo_tts99": qubo_info["tts_99_wall"],
                "qubo_time_median": qubo_info["time_median"],
                # Perm
                "perm_best_len": perm_info["best_len"],
                "perm_median_len": perm_info["median_len"],
                "perm_gap_best_pct": 100.0 * perm_info["gap_best"],
                "perm_gap_median_pct": 100.0 * perm_info["gap_median"],
                "perm_p_success": perm_info["p_success"],
                "perm_tts99": perm_info["tts_99_wall"],
                "perm_time_median": perm_info["time_median"],
                "perm_accept_rate": perm_info["accept_rate"],
            })
        finally:
            detach_logger(handler)

    if qubo_infos:
        plot_compare_bars(qubo_infos, perm_infos,
                          odir / "tsp_compare_bars.png")
    if rows:
        with open(odir / "compare_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"comparison summary written to {odir / 'compare_summary.csv'}")


if __name__ == "__main__":
    main()
