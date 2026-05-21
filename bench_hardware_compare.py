"""
bench_hardware_compare.py — Project sweep-level solver metrics onto
distinct hardware platforms and produce a cross-architecture
TTS_99 / energy-per-solution comparison.

The driver reads a `summary.csv` file produced by any of the
benchmark drivers in this repository and re-projects each row
through the hardware-platform models defined in hardware_metrics.py.
No new solver runs are required: the underlying p_success values are
hardware-agnostic statistics of the algorithm; the platform model
determines the time and energy cost per sweep.

Usage:
    python bench_hardware_compare.py \\
        --summary results_maxcut/summary.csv \\
        --output ./results_hw_compare

The summary CSV must contain columns: instance, n (or n_spin),
n_sweeps, p_success, time_median (used for the cpu-numba calibration).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hardware_metrics import (SMTJ_ARRAY, CMOS_PBIT, FPGA_SBM,
                              cpu_platform_from_run,
                              HardwarePlatform)
from plot_style import set_style, TSINGHUA_PURPLE


PLATFORM_COLORS = {
    "sMTJ-array": TSINGHUA_PURPLE["dark"],
    "cmos-pbit":  TSINGHUA_PURPLE["primary"],
    "fpga-sbm":   TSINGHUA_PURPLE["accent"],
    "cpu-numba":  TSINGHUA_PURPLE["gray"],
}


def _read_summary(path):
    """Return a list of dicts with keys instance, n, n_sweeps,
    p_success, time_median, parsed from any of the bench drivers'
    summary CSVs. The drivers are not perfectly uniform on column
    naming; this reader normalises them.

    Recognised column families:
        bench_maxcut/tsp/factor : p_success, time_median, n_sweeps
        compare_baselines       : ps_smtj, time_median_smtj, n_sweeps
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            n = (r.get("n") or r.get("N_spins")
                 or r.get("n_cities") or r.get("N_spin"))
            if n is None:
                continue
            try:
                n = int(float(n))
            except ValueError:
                continue
            sw = int(float(r.get("n_sweeps", 10000)))
            ps = (r.get("p_success") or r.get("ps_smtj")
                  or r.get("ps") or "0")
            try:
                ps = float(ps) if ps not in ("", "nan") else 0.0
            except ValueError:
                ps = 0.0
            tm = (r.get("time_median") or r.get("time_median_smtj")
                  or 0)
            try:
                tm = float(tm) if tm not in ("", "nan") else 0.0
            except ValueError:
                tm = 0.0
            inst = (r.get("instance") or r.get("M") or "?")
            if r.get("M") and not str(r.get("instance", "")).startswith("M"):
                inst = f"M={r['M']}"
            rows.append(dict(instance=str(inst), n=n, n_sweeps=sw,
                             p_success=ps, time_median=tm))
    return rows


def _project_row(row, platforms, conf=0.99):
    out = []
    for plat in platforms:
        tts = plat.hardware_tts(row["n"], row["n_sweeps"],
                                row["p_success"], conf)
        en = plat.energy_per_solution(row["n"], row["n_sweeps"],
                                      row["p_success"], conf)
        out.append({"platform": plat.name, "tts": tts, "energy": en,
                    "instance": row["instance"], "n": row["n"]})
    return out


def plot_compare(records, out_path, instance_order=None):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax_t, ax_e = axes
    if instance_order is not None:
        # Preserve caller-provided order (matches CSV file order),
        # but de-duplicate while keeping first occurrence.
        seen = set()
        instances = []
        for inst in instance_order:
            if inst in seen:
                continue
            seen.add(inst)
            instances.append(inst)
        # Append any straggler instances that were in records but
        # missing from instance_order, sorted lexicographically as a
        # fallback.
        extra = sorted({r["instance"] for r in records} - seen)
        instances.extend(extra)
    else:
        instances = sorted({r["instance"] for r in records},
                           key=lambda s: (len(s), s))
    plats = ["sMTJ-array", "cmos-pbit", "fpga-sbm", "cpu-numba"]
    plats = [p for p in plats
             if any(r["platform"] == p for r in records)]

    n_inst = len(instances)
    bar_w = 0.20
    x = np.arange(n_inst)

    # Pre-compute global y-ranges from finite values for tight log axes
    finite_t = [r["tts"] for r in records
                if math.isfinite(r["tts"]) and r["tts"] > 0]
    finite_e = [r["energy"] for r in records
                if math.isfinite(r["energy"]) and r["energy"] > 0]
    t_min = min(finite_t) / 3 if finite_t else 1e-6
    t_max = max(finite_t) * 3 if finite_t else 1.0
    e_min = min(finite_e) / 3 if finite_e else 1e-6
    e_max = max(finite_e) * 3 if finite_e else 1.0

    n_p = len(plats)
    centre = (n_p - 1) / 2.0

    for k, plat in enumerate(plats):
        ts = []
        es = []
        for inst in instances:
            r = next((r for r in records if r["instance"] == inst
                      and r["platform"] == plat), None)
            ts.append(r["tts"] if r else float("inf"))
            es.append(r["energy"] if r else float("inf"))
        ts = np.array(ts, dtype=float)
        es = np.array(es, dtype=float)
        finite_t_mask = np.isfinite(ts) & (ts > 0)
        finite_e_mask = np.isfinite(es) & (es > 0)
        for kk, (t, ok) in enumerate(zip(ts, finite_t_mask)):
            if ok:
                ax_t.bar(x[kk] + (k - centre) * bar_w, t, bar_w,
                         color=PLATFORM_COLORS[plat],
                         edgecolor=TSINGHUA_PURPLE["darkest"],
                         linewidth=0.7, zorder=3,
                         label=plat if kk == 0 else None)
        for kk, (e_val, ok) in enumerate(zip(es, finite_e_mask)):
            if ok:
                ax_e.bar(x[kk] + (k - centre) * bar_w, e_val, bar_w,
                         color=PLATFORM_COLORS[plat],
                         edgecolor=TSINGHUA_PURPLE["darkest"],
                         linewidth=0.7, zorder=3)

    for ax, ymin, ymax in ((ax_t, t_min, t_max), (ax_e, e_min, e_max)):
        ax.set_xticks(x)
        ax.set_xticklabels(instances, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_axisbelow(True)
        ax.grid(True, which="both", color="#E8E8E8", lw=0.6, zorder=0)
    ax_t.set_ylabel(r"TTS$_{99}$ (s)")
    ax_e.set_ylabel("Energy per solution (J)")
    ax_t.set_title("Time-to-solution by platform")
    ax_e.set_title("Energy per solution by platform")
    ax_t.legend(loc="upper left", ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Project solver-level metrics through hardware "
                    "platform models. Accepts one or more summary "
                    "CSV files: a single bench_*.py output, or any "
                    "combination of compare_baselines.py outputs "
                    "across the three benchmark families. The "
                    "instance set in the resulting figure is the "
                    "union of instances across all input files.")
    p.add_argument("--summary", required=True, nargs="+",
                   help="One or more summary.csv files. Glob "
                        "patterns are expanded by the shell.")
    p.add_argument("--output", default="./results_hw_compare")
    p.add_argument("--cpu-tdp", type=float, default=28.0,
                   help="CPU TDP in watts for cpu-numba energy column")
    p.add_argument("--conf", type=float, default=0.99)
    p.add_argument("--skip-unsolved", dest="skip_unsolved",
                   action="store_true", default=True,
                   help="Skip instances whose p_success=0 on every "
                        "platform (default: enabled). Such instances "
                        "carry no platform-comparison information.")
    p.add_argument("--no-skip-unsolved", dest="skip_unsolved",
                   action="store_false")
    args = p.parse_args(argv)

    odir = Path(args.output)
    odir.mkdir(parents=True, exist_ok=True)

    # Read every input CSV and concatenate. De-duplicate by instance
    # name within each file (later files override earlier ones if a
    # name collision occurs); cross-file collisions keep the first
    # occurrence. The order of input files becomes the report order
    # of instance clusters in the figure.
    rows = []
    seen = set()
    for src in args.summary:
        src_path = Path(src)
        if not src_path.exists():
            print(f"error: summary file not found: {src_path}",
                  file=sys.stderr)
            sys.exit(2)
        for r in _read_summary(src_path):
            key = r["instance"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
        print(f"loaded {src_path} ({sum(1 for r in rows)} rows so far)",
              file=sys.stderr)

    if not rows:
        print(f"no usable rows in {args.summary}", file=sys.stderr)
        sys.exit(2)

    # Drop unsolved instances early, before platform projection. An
    # instance with p_success = 0 produces TTS_99 = inf on every
    # platform and thus contributes only an n/a marker in the figure.
    if args.skip_unsolved:
        kept = [r for r in rows if r["p_success"] > 0]
        n_drop = len(rows) - len(kept)
        if n_drop:
            dropped = [r["instance"] for r in rows if r["p_success"] <= 0]
            print(f"skipping {n_drop} unsolved instance(s) "
                  f"(p_success=0): {dropped}", file=sys.stderr)
        rows = kept

    if not rows:
        print("no solvable instance after filtering; nothing to plot",
              file=sys.stderr)
        sys.exit(2)

    # Calibrate cpu-numba from the first row with positive time_median
    cpu_seed = next((r for r in rows if r["time_median"] > 0), None)
    if cpu_seed is None:
        print("warning: no row has positive time_median; "
              "cpu-numba projection will be omitted", file=sys.stderr)
        platforms = [SMTJ_ARRAY, CMOS_PBIT, FPGA_SBM]
    else:
        cpu = cpu_platform_from_run(cpu_seed["time_median"],
                                    cpu_seed["n_sweeps"],
                                    cpu_seed["n"],
                                    tdp_watts=args.cpu_tdp)
        platforms = [SMTJ_ARRAY, CMOS_PBIT, FPGA_SBM, cpu]

    all_recs = []
    for row in rows:
        all_recs.extend(_project_row(row, platforms, conf=args.conf))

    # Write CSV
    out_csv = odir / "hw_compare_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "n", "platform",
                                          "tts", "energy"])
        w.writeheader()
        for r in all_recs:
            w.writerow(r)

    plot_compare(all_recs, odir / "hw_compare_panels.png",
                 instance_order=[r["instance"] for r in rows])
    print(f"wrote {out_csv}")
    print(f"wrote {odir / 'hw_compare_panels.png'}")


if __name__ == "__main__":
    main()
