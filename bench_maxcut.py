"""
Benchmark driver for Section 3.4.2: Max-Cut on G-set instances.

Usage (single line for cross-platform shells):
    python bench_maxcut.py --instances G1 G14 G22 --trials 200 --sweeps 10000

Outputs (all human-readable / git-diffable):
    <out>/summary.csv                       per-instance aggregate metrics
    <out>/run_<name>.json                   per-trial results (trajectory,
                                            final energy, wall time, seed)
    <out>/states_<name>.txt                 final spin configurations as
                                            compact 0/1 text matrix
    <out>/log_<name>.txt                    copy of the console log
    <out>/trace_<name>.png, hist_<name>.png  figures
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

from isim import (SolverConfig, multistart, summarize_runs,
                  cut_value_from_energy, save_results_json,
                  save_final_states, get_logger)
from problems import load_gset
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("bench_maxcut")


GSET_BKS = {
    "G1":  11624,  "G2":  11620,  "G3":  11622,
    "G14": 3064,   "G15": 3050,   "G16": 3052,
    "G22": 13359,  "G23": 13344,  "G24": 13337,
    "G43": 6660,   "G44": 6650,   "G45": 6654,
}


def run_one_instance(gset_path: Path, cfg: SolverConfig, args):
    prob = load_gset(gset_path)
    W = prob.meta["edge_sum"]
    log.info(f"Loaded {prob.name}: n={prob.n}, edges={prob.meta['raw_m']}, "
             f"edge_sum={W:.0f}")

    results = multistart(
        problem=prob,
        solver_config=cfg,
        spin_spec=("ideal", {}),
        n_trials=args.trials,
        master_seed=args.seed,
        n_jobs=args.jobs,
        progress=True,
    )

    cut_bks = GSET_BKS.get(prob.name)
    if cut_bks is None:
        cut_bks = max(cut_value_from_energy(r.energy_final, W) for r in results)
        log.warning(f"no published BKS for {prob.name}; using best observed "
                    f"cut={cut_bks:.0f}")
    target_energy = W / 2.0 - cut_bks

    summary = summarize_runs(results, target=target_energy, sense="min")
    best_cut = cut_value_from_energy(summary["energy_min"], W)
    median_cut = cut_value_from_energy(summary["energy_median"], W)
    rel_gap_best = 1.0 - best_cut / cut_bks

    log.info(f"  {prob.name} results: best_cut={best_cut:.0f} "
             f"median_cut={median_cut:.0f} BKS={cut_bks} "
             f"gap_best={rel_gap_best*100:.3f}%")
    log.info(f"  p_success={summary['p_success']:.3f} "
             f"time_median={summary['time_median']:.2f}s "
             f"TTS99_wall={summary['tts_99_wall']:.2f}s")
    return prob, results, summary, cut_bks


def plot_traces(prob, results, out_path: Path):
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    energies_final = np.array([r.energy_final for r in results])
    W = prob.meta["edge_sum"]
    order = np.argsort(energies_final)
    pick = [order[0], order[len(order) // 2], order[-1]]
    labels = ["best", "median", "worst"]
    colors = [TSINGHUA_PURPLE["dark"],
              TSINGHUA_PURPLE["medium"],
              TSINGHUA_PURPLE["light"]]
    for idx, lab, color in zip(pick, labels, colors):
        r = results[idx]
        cut_traj = W / 2.0 - r.energies
        ax.plot(np.maximum(r.sweeps, 1), cut_traj, color=color, lw=2.0,
                label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Sweeps")
    ax.set_ylabel("Cut value")
    ax.set_title(f"{prob.name}: convergence of cut value")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hist(prob, results, cut_bks, out_path: Path):
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    W = prob.meta["edge_sum"]
    cuts = np.array([W / 2.0 - r.energy_final for r in results])
    ax.hist(cuts, bins=25, color=TSINGHUA_PURPLE["primary"],
            edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8, alpha=0.9)
    ax.axvline(cut_bks, color=TSINGHUA_PURPLE["accent"], lw=2.0,
               linestyle="--", label=f"BKS = {cut_bks}")
    ax.set_xlabel("Cut value")
    ax.set_ylabel("Trials")
    ax.set_title(f"{prob.name}: cut-value distribution")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_run(prob, results, summary, cut_bks, args, out_dir: Path):
    save_results_json(results, out_dir / f"run_{prob.name}.json",
                      extras={
                          "instance": prob.name,
                          "n": prob.n,
                          "m": prob.meta["raw_m"],
                          "edge_sum": prob.meta["edge_sum"],
                          "cut_bks": float(cut_bks),
                          "summary": summary,
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
    save_final_states(results, out_dir / f"states_{prob.name}.txt")


def attach_file_logger(out_dir: Path, name: str):
    """Route all log output for this instance additionally to a file."""
    handler = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    logging.getLogger("isim").addHandler(handler)
    logging.getLogger("bench_maxcut").addHandler(handler)
    return handler


def detach_logger(handler):
    logging.getLogger("isim").removeHandler(handler)
    logging.getLogger("bench_maxcut").removeHandler(handler)
    handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+",
                        default=["G1", "G14", "G22"])
    parser.add_argument("--gset-dir", default="./gset",
                        help="Directory containing G-set text files")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--sweeps", type=int, default=10000)
    parser.add_argument("--schedule", default="geometric",
                        choices=["linear", "geometric", "inverse_log"])
    parser.add_argument("--beta0", type=float, default=0.1)
    parser.add_argument("--betaf", type=float, default=10.0)
    parser.add_argument("--update-mode", default="async_numba",
                        choices=["async_numba", "block", "async_python"],
                        help="async_numba (fastest, needs numba) | "
                             "block (fast, pure NumPy) | "
                             "async_python (slow reference)")
    parser.add_argument("--dynamics", default="gibbs",
                        choices=["gibbs", "metropolis"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", default="./results_maxcut")
    parser.add_argument("--auto-fetch", dest="auto_fetch",
                        action="store_true", default=True)
    parser.add_argument("--no-auto-fetch", dest="auto_fetch",
                        action="store_false")
    args = parser.parse_args(argv)

    cfg = SolverConfig(
        schedule_shape=args.schedule,
        beta0=args.beta0,
        betaf=args.betaf,
        n_sweeps=args.sweeps,
        update_mode=args.update_mode,
        dynamics=args.dynamics,
    )
    try:
        cfg.validate()
    except ImportError as e:
        log.error(str(e))
        log.error("Hint: either `pip install numba`, or rerun with "
                  "`--update-mode block`.")
        sys.exit(2)

    gdir = Path(args.gset_dir)
    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.auto_fetch:
        try:
            from fetch_data import ensure_gset
            ensure_gset(args.instances, gdir)
        except Exception as err:
            log.warning(f"auto-fetch warning: {err}")

    rows = []
    for inst in args.instances:
        path = gdir / inst
        if not path.exists():
            log.error(f"[skip] {path} not found")
            continue
        handler = attach_file_logger(odir, inst)
        try:
            prob, results, summary, cut_bks = run_one_instance(path, cfg, args)
            save_run(prob, results, summary, cut_bks, args, odir)
            plot_traces(prob, results, odir / f"trace_{prob.name}.png")
            plot_hist(prob, results, cut_bks, odir / f"hist_{prob.name}.png")

            W = prob.meta["edge_sum"]
            row = {
                "instance": prob.name,
                "n": prob.n,
                "m": prob.meta["raw_m"],
                "cut_bks": cut_bks,
                "cut_best": W / 2.0 - summary["energy_min"],
                "cut_median": W / 2.0 - summary["energy_median"],
                "gap_best_pct": 100.0 * (1.0 - (W / 2.0 -
                    summary["energy_min"]) / cut_bks),
                "gap_median_pct": 100.0 * (1.0 - (W / 2.0 -
                    summary["energy_median"]) / cut_bks),
                "p_success": summary["p_success"],
                "tts99_wall": summary["tts_99_wall"],
                "tts99_sweeps": summary["tts_99_sweeps"],
                "time_median": summary["time_median"],
                "n_trials": summary["n_trials"],
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

    if rows:
        csv_path = odir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Summary written to {csv_path}")


if __name__ == "__main__":
    main()
