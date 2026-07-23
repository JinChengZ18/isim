#!/usr/bin/env python3
"""RX-12 — Sigmoid emergence under the real transient waveform (macrospin LLG).

The Chapter-3 harness computes the write switching probability P_sw entirely on
the SPICE side as the compact model's observable node, which is the STATIC
sigmoid  P_sw = sigma((V_flat - Vth)/VT)  evaluated at the DC-delivered flat-top
V_wr; the Bernoulli draw and every RNG then live in the Python testbench
(section 3.5.1). This item asks whether that foundational move survives the real
transient: does a finite 0.75 ns write pulse (comparable to tau0 ~ 1 ns) at the
same flat-top level reach the device's steady-state switching probability, i.e.
the static sigmoid, or does finite-time truncation / back-hopping bias P_sw?

The write is P->AP (POSITIVE V(sot_p,sot_n); mirror of the RX-05a reset which is
AP->P at NEGATIVE V). Section 3.5.1 asserts this direction is "the clean single
segment sigmoid" measured in 2.3 -- RX-05a already showed the AP->P RESET LLG
plateaus far below its sigmoid mirror; this run tests the WRITE side directly.

Reuses the RX-05a harness (reset_correlation_llg.py): the recompiled OSDI of the
vendored Chapter-2 vgsot_llg.va, the Brown-1963 thermal field injected via
hx/hy/hz, the 1 ps Euler noise PWL, and the SeedSequence.spawn seed handling
(pass the CHILD SeedSequence objects, never int(child.entropy) -- see the RX-05a
trial-log note). The flat-top voltages per code are read from the committed
update_chain_summary.json per_bits["6"].transfer (the DC .op MEASURED v_wr), so
the drive levels are exactly the harness's assumed drive.

HONESTY GUARD (inherited from RX-05): the LLG is macrospin. If the write does not
reach the static sigmoid at 0.75 ns that is a real finite-time / back-hopping
finding -- report the measured curve, do NOT force-fit to the sigmoid.

Run INSIDE WSL (Ubuntu-24.04-EDA; openvaf + ngspice>=43 with OSDI):
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 \
      eda/testbenches/sigmoid_emergence_llg.py --mode pilot
  ... --mode run --seeds 600 --jobs 10
Writes eda/testbenches/sigmoid_emergence_llg_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

# Reuse the RX-05a harness verbatim: device constants, Brown-1963 amplitude,
# noise PWL builder, OSDI compile, ngspice runner, classifier, Wilson CI, pool.
from reset_correlation_llg import (
    WORK, OSDI, VTH, VT, RSOT, H_TH, T_STEP, TW, TR, T0, MZ_TOL,
    noise_pwl, ensure_osdi, run_deck, classify, wilson, run_pool)

HERE = Path(__file__).resolve().parent
CHAIN_SUMMARY = HERE / "update_chain_summary.json"
SUMMARY = HERE / "sigmoid_emergence_llg_summary.json"

# Relaxation window after the single write pulse. The pulse fall edge ends at
# T0 + 2*TR + TW = 1.05 ns; give the macrospin a quiet (undriven) tail to settle
# into a well, then classify. TAIL kept short so we read pulse-driven switching,
# not slow thermal over-barrier back-hop (which would need >> ns anyway).
TAIL = 1.5e-9
PER = 100e-9                       # PULSE period >> t_end => single pulse
T_PULSE_END = T0 + 2 * TR + TW     # 1.05 ns, end of the falling edge

# 12 codes from the committed 6-bit DC transfer spanning the sigmoid transition
# and into saturation (u ~ -3 .. +4.4, static P_sw ~ 0.05 .. 0.99) so the high-u
# end directly tests whether the write saturates at the sigmoid or plateaus.
CODES = [7, 11, 15, 20, 24, 30, 34, 39, 45, 51, 57, 63]


def load_flat_tops():
    """Return [(code, v_wr, u, sigma_static)] for CODES from the DC .op JSON."""
    d = json.loads(CHAIN_SUMMARY.read_text())
    transfer = {r["code"]: r for r in d["per_bits"]["6"]["transfer"]}
    out = []
    for c in CODES:
        r = transfer[c]
        v = float(r["v_wr"])
        u = (v - VTH) / VT
        sig = 1.0 / (1.0 + math.exp(-u))
        out.append(dict(code=c, v_wr=v, u=round(u, 5),
                        sigma_static=round(sig, 6)))
    return out


def build_deck(tag, seed_ss, vamp, t_end):
    """Single POSITIVE write pulse (P->AP) from the P state; Brown-1963 noise."""
    rng = np.random.default_rng(seed_ss)
    src = []
    for ax in ("hx", "hy", "hz"):
        src.append(f"Vn{ax} {ax} 0 {noise_pwl(rng, t_end)}")
    deck = f"""* RX-12 sigmoid-emergence write run {tag}
.model vgsot_llg vgsot_llg
N1 sot_p sot_n mtj_p mx my mz hx hy hz vgsot_llg
Vsot sot_p 0 PULSE(0 {vamp:.6f} {T0:.4e} {TR:.4e} {TR:.4e} {TW:.4e} {PER:.4e})
Vsn  sot_n 0 dc 0
Vmtj mtj_p 0 dc 0
{chr(10).join(src)}
.ic v(mx)=0.141 v(my)=0 v(mz)=0.99
.control
  set num_threads=1
  tran 1p {t_end:.4e} uic
  wrdata _{tag}.csv v(mz)
  quit
.endc
.end
"""
    (WORK / f"_{tag}.spice").write_text(deck)
    return tag


def write_worker(job):
    """job = (idx, code, child SeedSequence, vamp). Start P, one +pulse, classify
    at pulse-end and after relaxation. Switch (P->AP) = final state AP."""
    idx, code, child, vamp = job
    t_end = T_PULSE_END + TAIL
    tag = f"wr{idx:05d}"
    t0 = time.perf_counter()
    build_deck(tag, child, vamp, t_end)
    res = run_deck(tag)
    wall = time.perf_counter() - t0
    if res is None:
        return dict(idx=idx, code=code, ok=False, wall=wall)
    t, mz = res
    sample_t = [T0 - 0.05e-9,          # pre-pulse: confirm P start
                T_PULSE_END,           # end of pulse (may be mid-flight)
                T_PULSE_END + 0.5e-9,  # early relax
                t_end - 0.05e-9]       # final (post-relax) classification
    mz_s = [float(np.interp(ts, t, mz)) for ts in sample_t]
    states = [classify(m) for m in mz_s]
    return dict(idx=idx, code=code, ok=True, wall=wall,
                mz_samples=[round(m, 4) for m in mz_s], states=states)


def analyze(rows, flats, n_seeds):
    by_code = {}
    for r in rows:
        if r.get("ok"):
            by_code.setdefault(r["code"], []).append(r)
    # per-seed CSV
    with open(WORK / "sigmoid_emergence_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "code", "s_pre", "s_pulse_end", "s_relax", "s_final",
                    "mz_pre", "mz_pulse_end", "mz_relax", "mz_final"])
        for r in sorted([x for x in rows if x.get("ok")], key=lambda x: x["idx"]):
            w.writerow([r["idx"], r["code"], *r["states"], *r["mz_samples"]])

    per_code = []
    for fl in flats:
        c = fl["code"]
        rs = by_code.get(c, [])
        started_P = [r for r in rs if r["states"][0] == "P"]
        n = len(started_P)
        # empirical P_sw = fraction that ended AP (final sample)
        k_final = sum(1 for r in started_P if r["states"][3] == "AP")
        k_pend = sum(1 for r in started_P if r["states"][1] == "AP")
        k_relax = sum(1 for r in started_P if r["states"][2] == "AP")
        p = k_final / n if n else float("nan")
        lo, hi = wilson(k_final, n)
        sig = fl["sigma_static"]
        # equivalent-u: the u that the static sigmoid would need to reproduce p
        if 0 < p < 1:
            u_emp = math.log(p / (1 - p))
        else:
            u_emp = float("nan")
        per_code.append(dict(
            code=c, v_wr=round(fl["v_wr"], 7), u=fl["u"],
            i_sot_mA=round(fl["v_wr"] / RSOT * 1e3, 4),
            n=n, n_ambiguous=int(n - k_final -
                                 sum(1 for r in started_P if r["states"][3] == "P")),
            n_switched=k_final,
            p_sw_emp=round(p, 4), ci=[round(lo, 4), round(hi, 4)],
            sigma_static=round(sig, 4),
            dev_prob=round(p - sig, 4),
            u_emp_equiv=round(u_emp, 4) if np.isfinite(u_emp) else None,
            dev_u=round(u_emp - fl["u"], 4) if np.isfinite(u_emp) else None,
            p_sw_pulse_end=round(k_pend / n, 4) if n else None,
            p_sw_early_relax=round(k_relax / n, 4) if n else None))
    return per_code


def summarize_deviation(per_code):
    dp = [r["dev_prob"] for r in per_code]
    du = [r["dev_u"] for r in per_code if r["dev_u"] is not None]
    # affine fit u_emp = a*u + b over codes with finite u_emp (systematic-shift test)
    us = np.array([r["u"] for r in per_code if r["u_emp_equiv"] is not None])
    ue = np.array([r["u_emp_equiv"] for r in per_code if r["u_emp_equiv"] is not None])
    fit = None
    if len(us) >= 2:
        a, b = np.polyfit(us, ue, 1)
        resid = ue - (a * us + b)
        fit = dict(slope_a=round(float(a), 4), intercept_b=round(float(b), 4),
                   max_abs_resid_u=round(float(np.max(np.abs(resid))), 4),
                   note=("u_emp = a*u + b; a~1,b!=0 => pure Vth shift (b*VT), "
                         "a!=1 => VT (window) rescale, poor fit / saturation => "
                         "finite-time plateau not recalibratable"))
    return dict(
        max_abs_dev_prob=round(float(np.max(np.abs(dp))), 4),
        max_abs_dev_prob_code=int(per_code[int(np.argmax(np.abs(dp)))]["code"]),
        max_abs_dev_u=round(float(np.max(np.abs(du))), 4) if du else None,
        mean_dev_prob=round(float(np.mean(dp)), 4),
        all_dev_prob_same_sign=bool(np.all(np.array(dp) >= 0) or
                                    np.all(np.array(dp) <= 0)),
        affine_fit=fit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "run"])
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--master-seed", type=int, default=20260723)
    args = ap.parse_args()

    ensure_osdi()
    (WORK / ".spiceinit").write_text("osdi vgsot_llg.osdi\n")
    flats = load_flat_tops()

    if args.mode == "pilot":
        print(f"H_th (per 1 ps step, per axis) = {H_TH:.4e} A/m; TAIL={TAIL*1e9} ns")
        parent = np.random.SeedSequence(args.master_seed)
        children = parent.spawn(len(flats) * 3)
        for i, fl in enumerate(flats[:3]):
            for j in range(3):
                r = write_worker((i * 3 + j, fl["code"],
                                  children[i * 3 + j], fl["v_wr"]))
                print(f"code {fl['code']:2d} u={fl['u']:+.2f} "
                      f"sig={fl['sigma_static']:.3f} seed{j}: "
                      f"states={r.get('states')} mz={r.get('mz_samples')} "
                      f"wall={r['wall']:.1f}s")
        return

    n = args.seeds
    parent = np.random.SeedSequence(args.master_seed)
    children = parent.spawn(len(flats) * n)   # one child per (code, seed)
    jobs = []
    for i, fl in enumerate(flats):
        for j in range(n):
            k = i * n + j
            jobs.append((k, fl["code"], children[k], fl["v_wr"]))
    print(f"run: {len(flats)} codes x {n} seeds = {len(jobs)} runs, jobs={args.jobs}")
    t0 = time.perf_counter()
    rows = run_pool(write_worker, jobs, args.jobs)
    wall = time.perf_counter() - t0
    per_code = analyze(rows, flats, n)
    dev = summarize_deviation(per_code)

    summary = dict(
        _label=("RX-12 sigmoid emergence under the real transient write pulse; "
                "macrospin LLG (vendored Chapter-2 vgsot_llg.va, Linux-recompiled "
                "OSDI) + Brown-1963 thermal field via hx/hy/hz; ALL P_sw SIMULATED "
                "(macrospin), compared to the harness-assumed STATIC sigmoid "
                "sigma((V_flat-Vth)/VT) at the same DC flat-top."),
        direction="P->AP (write), positive V(sot_p,sot_n)",
        flat_tops_source="update_chain_summary.json per_bits['6'].transfer (DC .op MEASURED v_wr)",
        anchors=dict(vth=VTH, vt=VT, inv_vt_per_V=round(1.0 / VT, 3), r_sot=RSOT),
        pulse=dict(tw_ns=TW * 1e9, tr_ns=TR * 1e9, tail_ns=TAIL * 1e9,
                   t_pulse_end_ns=round(T_PULSE_END * 1e9, 3),
                   t_end_ns=round((T_PULSE_END + TAIL) * 1e9, 3),
                   ic="P state: mx=0.141 my=0 mz=+0.99",
                   classify="P: mz>+0.2, AP: mz<-0.2; P_sw = frac ending AP (final sample)"),
        n_seeds=n, master_seed=args.master_seed,
        h_th_per_axis_Apm=H_TH, t_step_s=T_STEP,
        wall_total_s=round(wall, 1),
        n_failed_runs=sum(1 for r in rows if not r.get("ok")),
        per_code=per_code,
        deviation=dev,
    )
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(per_code=per_code, deviation=dev), indent=2))
    print(f"-> {SUMMARY}  (wall {wall:.1f}s)")


if __name__ == "__main__":
    main()
