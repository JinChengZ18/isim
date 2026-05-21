"""
Benchmark driver for Section 3.4.3: TSP on TSPLIB instances.

Usage (single line):
    python bench_tsp.py --instances burma14 ulysses16 --trials 100 --sweeps 50000

The n^2-spin QUBO encoding converges very slowly under plain SA for
n > 20. By default this driver refuses instances larger than that;
pass --allow-large to override.
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
                  p_success, save_results_json, save_final_states,
                  get_logger)
from problems import (load_tsplib, decode_tour, tour_length,
                      TSP_DEFAULT_MAX_N)
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("bench_tsp")


TSPLIB_OPT = {
    "burma14":  3323,
    "ulysses16":6859,
    "gr17":     2085,
    "ulysses22":7013,
    "gr24":     1272,
    "fri26":    937,
    "bayg29":   1610,
    "bays29":   2020,
    "dantzig42":699,
    "swiss42":  1273,
    "eil51":    426,
    "berlin52": 7542,
    "eil76":    538,
    "st70":     675,
}


def _canonical_name(name):
    """Strip any TSPLIB file extension that leaked into prob.name."""
    if not isinstance(name, str):
        return name
    for ext in (".tsp", ".TSP", ".atsp", ".ATSP"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def run_one_instance(tsplib_path: Path, cfg: SolverConfig, args):
    prob = load_tsplib(tsplib_path, B=args.B,
                       A=None if args.A is None else args.A,
                       allow_large=args.allow_large)
    if args.A is None:
        dmax = float(prob.meta["D"].max())
        A = args.A_margin * args.B * dmax
        prob = load_tsplib(tsplib_path, B=args.B, A=A,
                           allow_large=args.allow_large)
    n_cities = prob.meta["n_cities"]
    D = prob.meta["D"]
    log.info(f"Loaded {prob.name}: n_cities={n_cities}, "
             f"N_spins={prob.n}, A={prob.meta['A']:.1f}, B={prob.meta['B']:.1f}")

    results = multistart(
        problem=prob,
        solver_config=cfg,
        spin_spec=("ideal", {}),
        n_trials=args.trials,
        master_seed=args.seed,
        n_jobs=args.jobs,
        progress=True,
    )

    tours, lengths, feasible_flags = [], [], []
    for r in results:
        tour, feasible = decode_tour(r.state_final, n_cities)
        tours.append(tour)
        feasible_flags.append(feasible)
        lengths.append(tour_length(tour, D) if feasible else np.inf)
    lengths = np.asarray(lengths)
    feasible_flags = np.asarray(feasible_flags)

    opt_len = TSPLIB_OPT.get(_canonical_name(prob.name))
    if opt_len is None:
        if np.any(np.isfinite(lengths)):
            opt_len = float(lengths[np.isfinite(lengths)].min())
            log.warning(f"no published OPT for {prob.name}; using best "
                        f"feasible length={opt_len:.0f}")
        else:
            opt_len = float("nan")

    finite_mask = np.isfinite(lengths)
    n_feasible = int(finite_mask.sum())
    if n_feasible > 0:
        best_len = float(lengths[finite_mask].min())
        median_len = float(np.median(lengths[finite_mask]))
        gap_best = (best_len - opt_len) / opt_len
        gap_median = (median_len - opt_len) / opt_len
    else:
        best_len = median_len = float("inf")
        gap_best = gap_median = float("inf")

    tol = args.opt_tol
    success_mask = finite_mask & (lengths <= opt_len * (1 + tol))
    p_s = float(success_mask.mean())
    times = np.array([r.wall_time for r in results])
    sweeps = np.array([r.sweeps[-1] for r in results])
    tts_wall = tts_at_confidence(float(np.median(times)), p_s, 0.99)
    tts_sw = tts_at_confidence(float(np.median(sweeps)), p_s, 0.99)

    log.info(f"  {prob.name}: feasible={n_feasible}/{len(results)} "
             f"best_len={best_len:.1f} median_len={median_len:.1f} "
             f"OPT={opt_len}")
    log.info(f"  gap_best={gap_best*100:.2f}% gap_median={gap_median*100:.2f}% "
             f"p_s(tol={tol:g})={p_s:.3f} TTS99={tts_wall:.2f}s")

    info = {
        "prob": prob, "results": results, "tours": tours,
        "lengths": lengths, "feasible": feasible_flags,
        "opt_len": opt_len, "best_len": best_len, "median_len": median_len,
        "gap_best": gap_best, "gap_median": gap_median,
        "p_success": p_s, "tts_99_wall": tts_wall, "tts_99_sweeps": tts_sw,
        "time_median": float(np.median(times)),
    }
    return info


def plot_traces(info, out_path):
    prob = info["prob"]
    results = info["results"]
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lengths = info["lengths"]; feasible = info["feasible"]
    finite = feasible & np.isfinite(lengths)
    if finite.any():
        order = np.argsort(lengths + (~finite) * 1e18)
        pick = [order[0], order[len(order) // 2], order[-1]]
    else:
        pick = list(range(min(3, len(results))))
    labels = ["best", "median", "worst"]
    colors = [TSINGHUA_PURPLE["dark"], TSINGHUA_PURPLE["medium"],
              TSINGHUA_PURPLE["light"]]
    for idx, lab, c in zip(pick, labels, colors):
        r = results[idx]
        ax.plot(np.maximum(r.sweeps, 1), r.energies, color=c, lw=2.0, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Sweeps")
    ax.set_ylabel("Energy")
    ax.set_title(f"{_canonical_name(prob.name)}: energy trajectory")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hist(info, out_path):
    prob = info["prob"]
    lengths = info["lengths"]
    feasible = info["feasible"]
    opt_len = info["opt_len"]
    finite = feasible & np.isfinite(lengths)
    n_trials = len(lengths)
    n_feas = int(finite.sum())
    if not finite.any():
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(lengths[finite], bins=25,
            color=TSINGHUA_PURPLE["primary"],
            edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8, alpha=0.9)
    ax.axvline(opt_len, color=TSINGHUA_PURPLE["accent"], lw=2.0,
               linestyle="--", label=f"OPT = {opt_len:g}")
    ax.set_xlabel("Tour length")
    ax.set_ylabel("Feasible trials")
    ax.set_title(f"{_canonical_name(prob.name)}: tour-length distribution "
                 f"({n_feas}/{n_trials} feasible)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_run(info, args, out_dir):
    prob = info["prob"]
    results = info["results"]
    # Encode tours in meta
    tours_as_lists = [list(map(int, t)) for t in info["tours"]]
    save_results_json(results, out_dir / f"run_{prob.name}.json",
                      extras={
                          "instance": prob.name,
                          "n_cities": prob.meta["n_cities"],
                          "N_spins": prob.n,
                          "opt_len": info["opt_len"],
                          "best_len": info["best_len"],
                          "median_len": info["median_len"],
                          "p_success": info["p_success"],
                          "tts_99_wall": info["tts_99_wall"],
                          "tts_99_sweeps": info["tts_99_sweeps"],
                          "tours": tours_as_lists,
                          "tour_lengths": [float(x) for x in info["lengths"]],
                          "feasibility": [bool(x) for x in info["feasible"]],
                          "A": prob.meta["A"], "B": prob.meta["B"],
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


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "bench_tsp"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "bench_tsp"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["burma14", "ulysses16"])
    p.add_argument("--tsplib-dir", default="./tsplib")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sweeps", type=int, default=50000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.05)
    p.add_argument("--betaf", type=float, default=20.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    p.add_argument("--dynamics", default="gibbs",
                   choices=["gibbs", "metropolis"])
    p.add_argument("--B", type=float, default=1.0, help="Cost coefficient")
    p.add_argument("--A", type=float, default=None,
                   help="Penalty coefficient (if None, A=A_margin*B*max(d))")
    p.add_argument("--A-margin", type=float, default=2.0,
                   help="A = A_margin * B * max(d). Must be > 1 for feasibility.")
    p.add_argument("--opt-tol", type=float, default=0.01,
                   help="Relative tolerance to OPT for success counting")
    p.add_argument("--allow-large", action="store_true",
                   help=f"Override the n<={TSP_DEFAULT_MAX_N} default guard")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_tsp")
    p.add_argument("--auto-fetch", dest="auto_fetch",
                   action="store_true", default=True)
    p.add_argument("--no-auto-fetch", dest="auto_fetch", action="store_false")
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

    tdir = Path(args.tsplib_dir)
    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.auto_fetch:
        try:
            from fetch_data import ensure_tsplib
            ensure_tsplib(args.instances, tdir)
        except Exception as err:
            log.warning(f"auto-fetch warning: {err}")

    rows = []
    for inst in args.instances:
        candidates = [tdir / inst, tdir / f"{inst}.tsp", tdir / f"{inst}.TSP"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            log.error(f"[skip] no TSPLIB file for {inst} in {tdir}")
            continue
        handler = attach_file_logger(odir, inst)
        try:
            try:
                info = run_one_instance(path, cfg, args)
            except ValueError as e:
                log.error(f"[skip] {inst}: {e}")
                continue
            save_run(info, args, odir)
            plot_traces(info, odir / f"trace_{info['prob'].name}.png")
            plot_hist(info, odir / f"hist_{info['prob'].name}.png")
            row = {
                "instance": info["prob"].name,
                "n_cities": info["prob"].meta["n_cities"],
                "N_spins": info["prob"].n,
                "opt_len": info["opt_len"],
                "best_len": info["best_len"],
                "median_len": info["median_len"],
                "gap_best_pct": 100.0 * info["gap_best"],
                "gap_median_pct": 100.0 * info["gap_median"],
                "p_success": info["p_success"],
                "tts99_wall": info["tts_99_wall"],
                "tts99_sweeps": info["tts_99_sweeps"],
                "time_median": info["time_median"],
                "feasible_frac": float(info["feasible"].mean()),
                "n_trials": args.trials,
                "n_sweeps": args.sweeps,
                "opt_tol": args.opt_tol,
                "update_mode": args.update_mode,
                "dynamics": args.dynamics,
                "schedule": args.schedule,
                "beta0": args.beta0,
                "betaf": args.betaf,
                "A": info["prob"].meta["A"],
                "B": info["prob"].meta["B"],
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
