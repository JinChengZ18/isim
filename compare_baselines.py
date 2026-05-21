"""
Benchmark driver for Section 3.4: sMTJ-Gibbs vs classical simulated
annealing baseline.

The comparison contrasts two *dynamics* running on the same problem:

  * sMTJ-Gibbs : Glauber/Gibbs updates, which is the dynamical model of
                 a p-bit network built from ideal sMTJs.
  * Classical SA: Metropolis-Hastings updates, the reference algorithm
                  introduced by Kirkpatrick, Gelatt, and Vecchi (1983).

Both run with identical annealing schedules, sweep counts, and seeds;
the difference is strictly the update rule. Differences in final-energy
distributions, success probability, and TTS_99 can therefore be
attributed to the dynamics rather than to schedule tuning.

The three supported modes mirror the three benchmarks of Section 3.4:
  * maxcut : G-set Max-Cut instances (BKS from the G-set table)
  * factor : semiprime factoring (target = QUBO ground-state energy 0)
  * tsp    : TSPLIB symmetric TSP via n^2-spin one-hot QUBO encoding

Usage:
    python compare_baselines.py --mode maxcut --instances G1 G14 G22  --trials 200 --sweeps 10000
    python compare_baselines.py --mode factor --instances 15 21 33 35 51 65 77 91 143 --trials 200 --sweeps 20000 --beta0 0.05 --betaf 30
    python compare_baselines.py --mode tsp    --instances burma14 ulysses16 gr17 --trials 100 --sweeps 50000 --beta0 0.05 --betaf 20
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
                  p_success, save_results_json, get_logger)
from problems import (load_gset, load_tsplib, build_factoring_problem,
                      decode_factors, TSP_DEFAULT_MAX_N)
from plot_style import set_style, TSINGHUA_PURPLE

log = get_logger("compare")


GSET_BKS = {
    "G1": 11624, "G2": 11620, "G14": 3064, "G15": 3050,
    "G22": 13359, "G23": 13344, "G43": 6660, "G44": 6650,
}


def load_problem(mode, name, args):
    """Resolve an instance name to a (Problem, target_info) tuple.

    Supports three modes mirroring the three benchmarks of Section 3.4:
      - 'maxcut': name is a G-set instance (e.g. 'G1', 'G22'); target
                  is the published BKS cut value translated to the
                  matching ground-state energy of the spin-glass form.
      - 'factor': name is the integer M to factor (e.g. '15', '143');
                  target is energy 0 (the QUBO ground state for any
                  semiprime M = pq once the right (p, q) is found).
      - 'tsp':    name is a TSPLIB instance (with or without .tsp);
                  target is set to the best observed energy across the
                  two backends.
    """
    if mode == "maxcut":
        path = Path(args.gset_dir) / name
        return load_gset(path), ("maxcut", GSET_BKS.get(name))
    if mode == "tsp":
        for cand in [Path(args.tsplib_dir) / name,
                     Path(args.tsplib_dir) / f"{name}.tsp"]:
            if cand.exists():
                return load_tsplib(cand, B=1.0,
                                   allow_large=args.allow_large), ("tsp", None)
        raise FileNotFoundError(f"No TSPLIB file for {name}")
    if mode == "factor":
        try:
            M = int(name)
        except ValueError:
            raise ValueError(
                f"factor mode expects an integer M, got {name!r}")
        prob = build_factoring_problem(M)
        # The QUBO ground-state energy for a semiprime M = p*q is exactly
        # 0 once the auxiliary constraints z_{ij} = p_i*q_j are satisfied
        # AND (M - p_hat * q_hat)^2 = 0. Use 0.5 as a numerical tolerance
        # so floating-point drift does not exclude true solutions.
        return prob, ("factor", 0.5)
    raise ValueError(f"Unsupported mode/name: {mode}/{name}")


def run_one(mode, name, cfg_gibbs, cfg_metro, args):
    prob, kind_info = load_problem(mode, name, args)
    kind = kind_info[0]
    log.info(f"Problem {prob.name}: n={prob.n}, kind={kind}")

    log.info(f"  running sMTJ-Gibbs dynamics ...")
    res_smtj = multistart(prob, cfg_gibbs, ("ideal", {}),
                          n_trials=args.trials, master_seed=args.seed,
                          n_jobs=args.jobs, progress=True)
    log.info(f"  running Classical SA (Metropolis) dynamics ...")
    res_sa = multistart(prob, cfg_metro, ("metropolis", {}),
                        n_trials=args.trials, master_seed=args.seed,
                        n_jobs=args.jobs, progress=True)

    e_smtj = np.array([r.energy_final for r in res_smtj])
    e_sa = np.array([r.energy_final for r in res_sa])
    t_smtj = np.array([r.wall_time for r in res_smtj])
    t_sa = np.array([r.wall_time for r in res_sa])

    # Target / success criterion: the meaning differs by mode.
    #   maxcut: published BKS translated to the matching energy
    #   factor: decode the final state and check p_hat * q_hat == M
    #           with all auxiliary z-constraints satisfied
    #   tsp:    fall back to best observed energy across backends
    if kind == "maxcut" and kind_info[1] is not None:
        W = prob.meta["edge_sum"]
        target = W / 2.0 - kind_info[1]
        target_label = f"cut={kind_info[1]}"
        ps_smtj = p_success(e_smtj, target)
        ps_sa = p_success(e_sa, target)
    elif kind == "factor":
        M_target = int(prob.meta["M"])
        def _success_rate(results):
            ok = 0
            for r in results:
                p_hat, q_hat, c_ok = decode_factors(r.state_final, prob)
                if c_ok and (p_hat * q_hat == M_target) and p_hat > 1 and q_hat > 1:
                    ok += 1
            return ok / max(len(results), 1)
        ps_smtj = _success_rate(res_smtj)
        ps_sa = _success_rate(res_sa)
        target = None
        target_label = f"product pq=M={M_target}"
    else:
        target = float(min(e_smtj.min(), e_sa.min()))
        target_label = f"E_best_observed={target:.4f}"
        ps_smtj = p_success(e_smtj, target)
        ps_sa = p_success(e_sa, target)

    tts_smtj = tts_at_confidence(float(np.median(t_smtj)), ps_smtj, 0.99)
    tts_sa = tts_at_confidence(float(np.median(t_sa)), ps_sa, 0.99)

    log.info(f"  target: {target_label}")
    log.info(f"  sMTJ-Gibbs : E_min={e_smtj.min():.4f} "
             f"median={np.median(e_smtj):.4f} p_s={ps_smtj:.3f} "
             f"TTS99={tts_smtj:.2f}s (t_med={np.median(t_smtj):.2f}s)")
    log.info(f"  Classical SA: E_min={e_sa.min():.4f} "
             f"median={np.median(e_sa):.4f} p_s={ps_sa:.3f} "
             f"TTS99={tts_sa:.2f}s (t_med={np.median(t_sa):.2f}s)")

    return {
        "prob": prob, "kind": kind, "target": target,
        "target_label": target_label,
        "e_smtj": e_smtj, "e_sa": e_sa,
        "t_smtj": t_smtj, "t_sa": t_sa,
        "res_smtj": res_smtj, "res_sa": res_sa,
        "ps_smtj": ps_smtj, "ps_sa": ps_sa,
        "tts_smtj": tts_smtj, "tts_sa": tts_sa,
    }


def plot_energy_box(info, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    data = [info["e_smtj"], info["e_sa"]]
    bp = ax.boxplot(data, patch_artist=True,
                    labels=["sMTJ-Gibbs", "Classical SA"],
                    widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"],
                        [TSINGHUA_PURPLE["primary"],
                         TSINGHUA_PURPLE["accent_lt"]]):
        patch.set_facecolor(c); patch.set_alpha(0.9)
        patch.set_edgecolor(TSINGHUA_PURPLE["dark"])
        patch.set_linewidth(1.0)
    for line in bp["medians"]:
        line.set_color(TSINGHUA_PURPLE["darkest"]); line.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_color(TSINGHUA_PURPLE["gray"])
    for cap in bp["caps"]:
        cap.set_color(TSINGHUA_PURPLE["gray"])
    if info.get("target") is not None:
        ax.axhline(info["target"], color=TSINGHUA_PURPLE["gray"], lw=1.2,
                   linestyle="--", label=info["target_label"])
        ax.legend(loc="upper right")
    else:
        # factor mode: success criterion is decoded, not energy-based;
        # add the criterion to the title instead of as a horizontal line.
        ax.set_title(f"{info['prob'].name}: final energy per trial "
                     f"({info['target_label']})")
    ax.set_ylabel("Final energy")
    if info.get("target") is not None:
        ax.set_title(f"{info['prob'].name}: final energy per trial")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _display_label(prob_name):
    """Render a compact axis-tick label for one instance.

    Strips the .tsp suffix occasionally introduced by TSPLIB filename
    leakage, and rewrites the auto-generated factoring name
    'factor_M<n>' as the more readable 'M=<n>'."""
    s = str(prob_name)
    for ext in (".tsp", ".TSP", ".atsp", ".ATSP"):
        if s.endswith(ext):
            return s[:-len(ext)]
    if s.startswith("factor_M"):
        return f"M={s[len('factor_M'):]}"
    return s


def plot_tts_bar(infos, out_path):
    if not infos:
        return
    set_style()
    # Figure height fixed; width scales with instance count but with a
    # firm minimum so single-instance and three-instance plots both
    # render at sensible aspect ratios (avoiding the tall-narrow
    # collapse seen with very few columns under default rcParams).
    fig_w = max(5.4, 0.9 * len(infos) + 2.0)
    fig_h = 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    labels = [_display_label(info["prob"].name) for info in infos]

    def sanitize(x):
        return x if np.isfinite(x) and x > 0 else np.nan
    tts_smtj = np.asarray([sanitize(info["tts_smtj"]) for info in infos])
    tts_sa = np.asarray([sanitize(info["tts_sa"]) for info in infos])
    x = np.arange(len(infos))
    width = 0.38

    # Set log y-axis range explicitly from the finite data so all-NaN
    # columns do not collapse the canvas. If everything is NaN fall
    # back to a placeholder range.
    finite = np.concatenate([tts_smtj[~np.isnan(tts_smtj)],
                             tts_sa[~np.isnan(tts_sa)]])
    if finite.size > 0:
        lo = float(finite.min()) / 3.0
        hi = float(finite.max()) * 3.0
    else:
        lo, hi = 1.0, 10.0
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)

    ax.bar(x - width / 2, tts_smtj, width,
           color=TSINGHUA_PURPLE["primary"],
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
           label="sMTJ-Gibbs")
    ax.bar(x + width / 2, tts_sa, width,
           color=TSINGHUA_PURPLE["accent_lt"],
           edgecolor=TSINGHUA_PURPLE["dark"], linewidth=0.8,
           label="Classical SA (Metropolis)")

    # n/a annotations placed near the floor of the y-range so they
    # read as ground-zero rather than mid-canvas (which had given
    # the impression that the bar did exist at that height).
    y_label = lo * 1.5
    for i, (a, b) in enumerate(zip(tts_smtj, tts_sa)):
        if np.isnan(a):
            ax.text(i - width / 2, y_label, "n/a",
                    ha="center", va="bottom",
                    color=TSINGHUA_PURPLE["dark"], fontsize=9)
        if np.isnan(b):
            ax.text(i + width / 2, y_label, "n/a",
                    ha="center", va="bottom",
                    color=TSINGHUA_PURPLE["accent"], fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"TTS$_{99}$ (seconds)")
    ax.set_title("Time-to-solution (99% confidence)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_run(info, args, out_dir):
    prob = info["prob"]
    # Save both backends
    for tag, res in (("smtj", info["res_smtj"]), ("sa", info["res_sa"])):
        save_results_json(res, out_dir / f"compare_{prob.name}_{tag}.json",
                          extras={
                              "instance": prob.name,
                              "n": prob.n,
                              "backend": tag,
                              "target": info["target"],
                              "target_label": info["target_label"],
                              "config": {
                                  "n_trials": args.trials,
                                  "n_sweeps": args.sweeps,
                                  "schedule": args.schedule,
                                  "beta0": args.beta0,
                                  "betaf": args.betaf,
                                  "update_mode": args.update_mode,
                                  "master_seed": args.seed,
                              },
                          })


def attach_file_logger(out_dir, name):
    h = logging.FileHandler(out_dir / f"log_{name}.txt", mode="w")
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    for ln in ("isim", "compare"):
        logging.getLogger(ln).addHandler(h)
    return h


def detach_logger(h):
    for ln in ("isim", "compare"):
        logging.getLogger(ln).removeHandler(h)
    h.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="maxcut",
                   choices=["maxcut", "factor", "tsp"])
    p.add_argument("--instances", nargs="+", default=["G1", "G14", "G22"])
    p.add_argument("--gset-dir", default="./gset")
    p.add_argument("--tsplib-dir", default="./tsplib")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--sweeps", type=int, default=10000)
    p.add_argument("--schedule", default="geometric",
                   choices=["linear", "geometric", "inverse_log"])
    p.add_argument("--beta0", type=float, default=0.1)
    p.add_argument("--betaf", type=float, default=10.0)
    p.add_argument("--update-mode", default="async_numba",
                   choices=["async_numba", "block", "async_python"])
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output", default="./results_compare")
    p.add_argument("--auto-fetch", dest="auto_fetch",
                   action="store_true", default=True)
    p.add_argument("--no-auto-fetch", dest="auto_fetch", action="store_false")
    p.add_argument("--filter-no-solve", dest="filter_no_solve",
                   action="store_true", default=True,
                   help="Skip instances whose TTS_99 estimate is not "
                   "statistically supported from the bar plot (default: "
                   "enabled). Such instances remain in summary.csv.")
    p.add_argument("--no-filter-no-solve", dest="filter_no_solve",
                   action="store_false")
    p.add_argument("--min-successes-for-plot", type=int, default=2,
                   help="Minimum number of successful trials required "
                   "for an instance to appear in the bar plot. An "
                   "instance is plotted if at least one dynamics "
                   "reaches this count; otherwise both bars come from "
                   "zero or a single trial and the resulting TTS_99 is "
                   "not a reproducible statistic. Relevant for the TSP "
                   "mode in particular, where the success target is "
                   "defined as the best observed energy and a hit count "
                   "of one is structurally unavoidable. Default: 2.")
    args = p.parse_args(argv)

    # Build two SolverConfig: identical except for dynamics
    base = dict(
        schedule_shape=args.schedule, beta0=args.beta0, betaf=args.betaf,
        n_sweeps=args.sweeps, update_mode=args.update_mode,
    )
    cfg_gibbs = SolverConfig(dynamics="gibbs", **base)
    cfg_metro = SolverConfig(dynamics="metropolis", **base)
    try:
        cfg_gibbs.validate()
        cfg_metro.validate()
    except ImportError as e:
        log.error(str(e))
        sys.exit(2)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    if args.auto_fetch:
        try:
            if args.mode == "maxcut":
                from fetch_data import ensure_gset
                ensure_gset(args.instances, args.gset_dir)
            elif args.mode == "tsp":
                from fetch_data import ensure_tsplib
                ensure_tsplib(args.instances, args.tsplib_dir)
        except Exception as err:
            log.warning(f"auto-fetch warning: {err}")

    infos = []
    rows = []
    for name in args.instances:
        handler = attach_file_logger(odir, name)
        try:
            try:
                info = run_one(args.mode, name, cfg_gibbs, cfg_metro, args)
            except FileNotFoundError as err:
                log.error(f"[skip] {err}")
                continue
            except ValueError as err:
                log.error(f"[skip] {name}: {err}")
                continue
            infos.append(info)
            save_run(info, args, odir)
            plot_energy_box(info,
                            odir / f"energy_compare_{info['prob'].name}.png")
            row = {
                "instance": info["prob"].name, "n": info["prob"].n,
                "target": info["target"],
                "ps_smtj": info["ps_smtj"], "ps_sa": info["ps_sa"],
                "tts99_smtj": info["tts_smtj"], "tts99_sa": info["tts_sa"],
                "speedup_sa_over_smtj": (info["tts_sa"] / info["tts_smtj"]
                                         if (np.isfinite(info["tts_smtj"])
                                             and info["tts_smtj"] > 0)
                                         else float("nan")),
                "energy_min_smtj": float(info["e_smtj"].min()),
                "energy_min_sa": float(info["e_sa"].min()),
                "energy_median_smtj": float(np.median(info["e_smtj"])),
                "energy_median_sa": float(np.median(info["e_sa"])),
                "time_median_smtj": float(np.median(info["t_smtj"])),
                "time_median_sa": float(np.median(info["t_sa"])),
                "n_trials": args.trials, "n_sweeps": args.sweeps,
                "update_mode": args.update_mode,
                "schedule": args.schedule,
            }
            rows.append(row)
        finally:
            detach_logger(handler)

    if infos:
        # Filter out instances whose TTS_99 estimate rests on too few
        # successes to be reproducible. The threshold is expressed as a
        # minimum number of successful trials per dynamics; an instance
        # is plotted if at least one of the two dynamics reaches it.
        # The previous criterion (both p_s strictly zero) missed the
        # equally pathological "exactly one hit" regime that arises in
        # TSP runs where the target is the best observed energy. All
        # rows are nevertheless preserved in summary.csv: data is never
        # silently dropped, only excluded from the plot.
        if args.filter_no_solve:
            n_tr = max(1, args.trials)
            min_succ = max(1, args.min_successes_for_plot)
            def _enough_stats(info):
                n_smtj = int(round(info["ps_smtj"] * n_tr))
                n_sa = int(round(info["ps_sa"] * n_tr))
                return n_smtj >= min_succ or n_sa >= min_succ
            plot_infos = [i for i in infos if _enough_stats(i)]
            n_skipped = len(infos) - len(plot_infos)
            if n_skipped > 0:
                details = [
                    (i["prob"].name,
                     int(round(i["ps_smtj"] * n_tr)),
                     int(round(i["ps_sa"] * n_tr)))
                    for i in infos if not _enough_stats(i)
                ]
                log.info(f"plot_tts_bar: skipping {n_skipped} "
                         f"instance(s) with fewer than {min_succ} "
                         f"successful trials on both dynamics "
                         f"(name, n_succ_smtj, n_succ_sa): {details}")
        else:
            plot_infos = infos
        plot_tts_bar(plot_infos, odir / "tts_compare.png")

    if rows:
        csv_path = odir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Summary written to {csv_path}")


if __name__ == "__main__":
    main()
