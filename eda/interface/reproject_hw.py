#!/usr/bin/env python3
"""W7 — re-run the Section 3.4.3 hardware projection with the sMTJ-array
row upgraded from the device-only numbers (t = 0.75 ns, e = 0.78 pJ) to the
END-TO-END per-update cost measured by eda/testbenches/update_energy.py
(k reset pulses + probabilistic write + read + peripheral overheads).

Only the sMTJ-array platform changes; CMOS p-bit / FPGA SBM / CPU rows keep
their Section 3.4.3 definitions, so any shift in the comparison is exactly
the peripheral-cost correction. Reads the same canonical benchmark summaries
as bench_hardware_compare.py.

Run:  python eda/interface/reproject_hw.py [--k 3]
Writes results_reproject/reproject_summary.csv (+ .json header with the
exact upgraded constants and their provenance).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from hardware_metrics import (HardwarePlatform, SMTJ_ARRAY, CMOS_PBIT,   # noqa: E402
                              FPGA_SBM, cpu_platform_from_run)

# canonical benchmark summaries (Section 3.3), same inputs the original
# projection used via bench_hardware_compare.py
SUMMARIES = [
    ROOT / "results" / "results_compare_maxcut" / "compare_maxcut_summary.csv",
    ROOT / "results" / "results_compare_factor" / "compare_factor_summary.csv",
    ROOT / "results" / "results_compare_tsp" / "comapre_tsp_summary.csv",
]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3,
                    help="reset pulses per update in the end-to-end account")
    ap.add_argument("--energy-json",
                    default=str(ROOT / "eda" / "testbenches" /
                                "update_energy_summary.json"))
    ap.add_argument("--outdir", default="results_reproject",
                    help="output dir name under eda/interface/ (RX-03 "
                         "worst-corner runs use results_reproject_<corner>)")
    args = ap.parse_args()

    ej = json.loads(Path(args.energy_json).read_text())
    row = next(r for r in ej["table"] if r["k"] == args.k)
    t_e2e = row["t_update_ns"] * 1e-9
    # power-gated accounting: the most favourable defensible peripheral
    # account (always-on and supply-true are larger; quoted in prose)
    e_e2e = row["e_update_gated_pJ"] * 1e-12

    smtj_e2e = HardwarePlatform(name="sMTJ-array-e2e", t_update=t_e2e,
                                e_update=e_e2e,
                                parallel_n=SMTJ_ARRAY.parallel_n)
    platforms = [SMTJ_ARRAY, smtj_e2e, CMOS_PBIT, FPGA_SBM]

    out_rows = []
    cpu = None
    for path in SUMMARIES:
        for r in read_rows(path):
            inst = r.get("instance") or r.get("name")
            n = int(float(r.get("n") or r.get("n_spins") or 0))
            ps = float(r.get("ps_smtj") or r.get("p_success") or 0)
            nsw = int(float(r.get("n_sweeps") or 10000))
            tmed = float(r.get("time_median_smtj") or
                         r.get("time_median") or 0)
            if cpu is None and tmed > 0:
                cpu = cpu_platform_from_run(tmed, nsw, n)
            if ps <= 0:
                continue
            plats = platforms + ([cpu] if cpu else [])
            for p in plats:
                out_rows.append(dict(
                    instance=inst, n=n, platform=p.name,
                    tts=p.hardware_tts(n, nsw, ps),
                    energy=p.energy_per_solution(n, nsw, ps)))

    odir = HERE / args.outdir
    odir.mkdir(exist_ok=True)
    with open(odir / "reproject_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "n", "platform",
                                          "tts", "energy"])
        w.writeheader()
        w.writerows(out_rows)

    # headline: shift of the sMTJ row, device-only -> end-to-end
    shifts = {}
    for inst in {r["instance"] for r in out_rows}:
        dev = next(r for r in out_rows
                   if r["instance"] == inst and r["platform"] == "sMTJ-array")
        e2e = next(r for r in out_rows
                   if r["instance"] == inst
                   and r["platform"] == "sMTJ-array-e2e")
        shifts[inst] = dict(tts_x=e2e["tts"] / dev["tts"],
                            energy_x=e2e["energy"] / dev["energy"])
    (odir / "reproject_header.json").write_text(json.dumps(dict(
        _label=("Section 3.4.3 projection re-run; ONLY the sMTJ-array row "
                "upgraded to measured end-to-end per-update cost"),
        k_reset=args.k,
        t_update_e2e_ns=row["t_update_ns"],
        e_update_e2e_pJ=row["e_update_gated_pJ"],
        provenance=str(Path(args.energy_json).name),
        device_only=dict(t_update_ns=SMTJ_ARRAY.t_update * 1e9,
                         e_update_pJ=SMTJ_ARRAY.e_update * 1e12),
        shifts=shifts), indent=2))

    tx = np.array([s["tts_x"] for s in shifts.values()])
    ex = np.array([s["energy_x"] for s in shifts.values()])
    print(f"sMTJ end-to-end vs device-only (k={args.k}): "
          f"TTS x{tx.mean():.2f}, energy x{ex.mean():.2f} "
          f"(uniform across instances: {tx.std():.3f}/{ex.std():.3f} std)")
    print(f"-> {odir / 'reproject_summary.csv'}")


if __name__ == "__main__":
    main()
