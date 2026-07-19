#!/usr/bin/env python3
"""RX-03 — PVT corner sweep of the write chain (DC transfer + energy/timing).

Runs update_chain_dc.py (6-bit DC sweep) and update_energy.py once per sky130
corner in {tt, ss, ff, sf, fs} by setting SKY130_CORNER in the child env
(both scripts read it; non-tt runs write suffixed summaries so the canonical
tt files are never clobbered), then aggregates the per-corner metrics into
update_chain_corners_summary.json and prints the spread table.

The MTJ compact model has no process corners — its parameters are the ch2
calibration constants at every corner (stated in the JSON as ANALYTIC
constancy). Only the CMOS chain varies.

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/corners_sweep.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORNERS = ["tt", "ss", "ff", "sf", "fs"]


def run_corner(corner):
    env = dict(os.environ, SKY130_CORNER=corner)
    for script, args in (("update_chain_dc.py", ["--bits", "6"]),
                         ("update_energy.py", [])):
        # resumable: skip any (corner, script) whose summary already exists
        # (tt canonical files come from the W2/W3 runs; non-tt from a prior
        # interrupted sweep — delete the suffixed json to force a re-run)
        done = (chain_file(corner) if script == "update_chain_dc.py"
                else energy_file(corner))
        if done.exists():
            print(f"[{corner}] skip {script} (have {done.name})", flush=True)
            continue
        print(f"[{corner}] {script} {' '.join(args)}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / script), *args],
                           cwd=HERE, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed at corner {corner}")


def chain_file(corner):
    return HERE / ("update_chain_summary.json" if corner == "tt"
                   else f"update_chain_summary_{corner}.json")


def energy_file(corner):
    return HERE / ("update_energy_summary.json" if corner == "tt"
                   else f"update_energy_summary_{corner}.json")


def main():
    for c in CORNERS:
        run_corner(c)

    per_corner = {}
    for c in CORNERS:
        chain = json.loads(chain_file(c).read_text())
        b6 = chain["per_bits"]["6"]
        en = json.loads(energy_file(c).read_text())
        k3 = next(r for r in en["table"] if r["k"] == 3)
        per_corner[c] = dict(
            lsb_mV=b6["lsb_mV"], lsb_over_VT=b6["lsb_over_VT"],
            range_mV=b6["range_mV"], monotonic=b6["monotonic"],
            inl_lsb=b6["inl_lsb"], buffer_offset_mV=b6["buffer_offset_mV"],
            u_min=b6["u_min"], u_max=b6["u_max"],
            p_buf_mW=round(en["buffer_static"]["p_buf_W"] * 1e3, 4),
            t_settle_ns=en["settle"]["t_settle_ns"],
            e_dev_mid_pJ=en["write_pulse"]["mid"]["e_dev_pJ"],
            e_tg_mid_pJ=en["write_pulse"]["mid"]["e_tg_pJ"],
            vwr_flat_mid_V=en["write_pulse"]["mid"]["vwr_flat_mean_V"],
            t_update_k3_ns=k3["t_update_ns"],
            e_update_gated_k3_pJ=k3["e_update_gated_pJ"],
            e_update_supply_true_k3_pJ=k3["e_update_supply_true_pJ"],
            read_ok=all(s["resolved"] and s["correct"]
                        for s in en["read"]["states"]))

    mono_all = all(m["monotonic"] for m in per_corner.values())
    worst = max((c for c in CORNERS),
                key=lambda c: per_corner[c]["e_update_gated_k3_pJ"])
    out = dict(
        _label=("MEASURED per corner (ngspice, sky130, schematic-level); "
                "MTJ compact model is corner-invariant by construction "
                "(ch2 calibration constants, ANALYTIC constancy)"),
        corners=CORNERS, per_corner=per_corner,
        monotonic_all_corners=mono_all,
        worst_corner_by_gated_e_update=worst,
        spread=dict(
            lsb_mV=[min(m["lsb_mV"] for m in per_corner.values()),
                    max(m["lsb_mV"] for m in per_corner.values())],
            p_buf_mW=[min(m["p_buf_mW"] for m in per_corner.values()),
                      max(m["p_buf_mW"] for m in per_corner.values())],
            t_settle_ns=[min(m["t_settle_ns"] for m in per_corner.values()),
                         max(m["t_settle_ns"] for m in per_corner.values())],
            e_update_gated_k3_pJ=[
                min(m["e_update_gated_k3_pJ"] for m in per_corner.values()),
                max(m["e_update_gated_k3_pJ"] for m in per_corner.values())],
            t_update_k3_ns=[
                min(m["t_update_k3_ns"] for m in per_corner.values()),
                max(m["t_update_k3_ns"] for m in per_corner.values())]))
    (HERE / "update_chain_corners_summary.json").write_text(
        json.dumps(out, indent=2))

    hdr = ("corner  LSB(mV)  INL(LSB)  mono  off(mV)  P_buf(mW)  "
           "settle(ns)  e_k3_gated(pJ)  t_k3(ns)  read")
    print("\n" + hdr)
    for c in CORNERS:
        m = per_corner[c]
        print(f"{c:>6}  {m['lsb_mV']:7.3f}  {m['inl_lsb']:8.3f}  "
              f"{str(m['monotonic']):>4}  {m['buffer_offset_mV']:7.2f}  "
              f"{m['p_buf_mW']:9.3f}  {m['t_settle_ns']:10.3f}  "
              f"{m['e_update_gated_k3_pJ']:14.3f}  {m['t_update_k3_ns']:8.3f}  "
              f"{'OK' if m['read_ok'] else 'FAIL'}")
    print(f"\nmonotonic at all corners: {mono_all}; "
          f"worst corner (gated k=3 energy): {worst}")
    print("-> update_chain_corners_summary.json")


if __name__ == "__main__":
    main()
