#!/usr/bin/env python3
"""RX-05a — LLG reset-pulse correlation: does the macrospin full-dynamics
engine reproduce the Chapter-2 back-hopping plateau (single-pulse AP->P
success ~0.72), and are consecutive reset-pulse failures independent
(the 0.28^k assumption of the sticky-reset model in circuit_backends.py)?

Engine: the vendored Chapter-2 LLG compact model
  04PBNNSim/smtj_pbnn_sim/eda/vendor/vgsot-sim/va/llg/vgsot_llg.va
(macrospin LLG + SOT/STT/VCMA, device parameters = the Chapter-2 Device-A
macrospin calibration baked into the .va defaults), recompiled here with the
Linux OpenVAF because the vendored .osdi is a Windows PE DLL. The vendor
repo is read-only for this harness; all build artifacts and outputs live in
eda/testbenches/llg_reset/.

Thermal field: Brown-1963 white field injected via the model's hx/hy/hz
nodes (1 V == 1 A/m), three independent N(0,1) draws per t_step per axis,
    H_th = sqrt(2 kB T alpha / (mu0 Ms gamma V_free t_step)),
sample-and-hold PWL at t_step (matching the vendor Python engine's Euler
noise; components NOT normalized — |H|^2 must be chi^2(3)). Seeds:
SeedSequence(master_seed).spawn per trial, all recorded.

Operating point (the Chapter-3 reset drive): the write chain resets at TOP
code, delivered flat-top ~0.999 V (update_energy.py CODE_TOP), opposite
polarity to the probabilistic write pulse. In the LLG model AP = m_z ~ -1,
P = m_z ~ +1, and NEGATIVE V(sot_p,sot_n) (I_SOT = V/R_SOT, R_SOT = 776)
drives AP->P; so the reset pulse is -0.999 V, t_w = 0.75 ns (0.05 ns
edges), repeated on a 2.0 ns cycle (the update_energy pulse-train timing).

Protocol:
  * train mode: 4-pulse reset train per seed, state sampled 50 ps before
    each pulse and 2.95 ns after the last; per-pulse success r_k, the
    conditional chain P(fail k+1 | fail k), back-hop rate
    P(AP at k+1 | P at k), and P(still AP after k) vs (1-r1)^k.
  * sweep mode: single-pulse P_sw(AP->P) vs amplitude — the direct
    plateau-reproduction check against the measured 0.72 anchor and the
    P->AP sigmoid mirror (VTH=0.895783, VT=0.023414).

HONESTY GUARD (from the RX-05 spec): if the macrospin LLG does NOT
reproduce the ~0.72 plateau, that is the finding — report the measured r1
and keep 0.28^k labeled a model assumption; do not force-fit.

Run INSIDE WSL (Ubuntu-24.04-EDA; needs openvaf + ngspice>=43 with OSDI):
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 \
      eda/testbenches/reset_correlation_llg.py --mode pilot
  ... --mode all --train-seeds 1000 --sweep-seeds 200 --jobs 10
Writes eda/testbenches/reset_correlation_llg_summary.json (+ per-seed CSVs
in eda/testbenches/llg_reset/).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # eda/testbenches
ROOT = HERE.parent.parent                       # isim_framework
WORK = HERE / "llg_reset"
VA = (ROOT.parent.parent / "04PBNNSim" / "smtj_pbnn_sim" / "eda" / "vendor"
      / "vgsot-sim" / "va" / "llg" / "vgsot_llg.va")
OSDI = WORK / "vgsot_llg.osdi"
SUMMARY = HERE / "reset_correlation_llg_summary.json"

# --- Chapter-2 committed operating-point mirrors (eda/testbenches/_common.py)
VTH, VT, RSOT = 0.895783, 0.023414, 776.0
V_RESET = -0.999            # top-code amplitude, AP->P polarity (see header)
TW, TR, CYCLE = 0.75e-9, 0.05e-9, 2.0e-9
T0 = 0.2e-9                 # first pulse start
N_PULSES = 4
TAIL = 2.95e-9              # relax window after the last pulse edge

# --- device constants for the Brown-1963 amplitude (vendor configs.py /
# vgsot_llg.va defaults — the Chapter-2 Device-A macrospin calibration)
KB = 1.380649e-23
MU0 = 4.0e-7 * math.pi
UB = 9.2740100783e-24
HBAR = 1.054571817e-34
GAMMA = 2.0 * MU0 * UB / HBAR          # ~2.2102e5 m/(A s)
MS = 0.625e6                           # A/m
ALPHA = 0.05
TF, D_ELEC = 1.1e-9, 65e-9
V_FREE = TF * math.pi * D_ELEC ** 2 / 4.0
TEMP = 300.0
T_STEP = 1e-12                         # vendor engine Euler step
H_TH = math.sqrt(2.0 * KB * TEMP * ALPHA
                 / (MU0 * MS * GAMMA * V_FREE * T_STEP))

MZ_TOL = 0.2                           # vendor success tolerance: P if
                                       # mz > 1-tol... we classify by sign
                                       # bands: P: mz>+0.2, AP: mz<-0.2

SWEEP_V = [-0.90, -0.95, -0.999, -1.05, -1.10, -1.20]


def ensure_osdi():
    WORK.mkdir(exist_ok=True)
    if OSDI.exists() and OSDI.stat().st_mtime >= VA.stat().st_mtime:
        return
    subprocess.run(["openvaf", str(VA), "-o", str(OSDI)], check=True,
                   cwd=WORK)


def noise_pwl(rng, t_end, t_step=T_STEP):
    """Sample-and-hold PWL string for one axis: value H_TH*xi per step."""
    n = int(math.ceil(t_end / t_step)) + 1
    xi = rng.standard_normal(n) * H_TH
    edge = 0.05e-12
    pts = []
    for i in range(n):
        t = i * t_step
        v = xi[i]
        pts.append(f"{t:.6e} {v:.5e}")
        pts.append(f"{(t + t_step - edge):.6e} {v:.5e}")
    # fold into continuation lines, 3 pairs per line
    lines = []
    for j in range(0, len(pts), 3):
        lines.append("+ " + " ".join(pts[j:j + 3]))
    return "PWL(\n" + "\n".join(lines) + " )"


def build_deck(tag, seed_ss, vamp, n_pulses, t_end, noise=True):
    rng = np.random.default_rng(seed_ss)
    src = []
    for ax in ("hx", "hy", "hz"):
        body = noise_pwl(rng, t_end) if noise else "dc 0"
        src.append(f"Vn{ax} {ax} 0 {body}")
    deck = f"""* RX-05a reset-correlation run {tag}
.model vgsot_llg vgsot_llg
N1 sot_p sot_n mtj_p mx my mz hx hy hz vgsot_llg
Vsot sot_p 0 PULSE(0 {vamp} {T0:.4e} {TR:.4e} {TR:.4e} {TW:.4e} {CYCLE:.4e})
Vsn  sot_n 0 dc 0
Vmtj mtj_p 0 dc 0
{chr(10).join(src)}
.ic v(mx)=0.141 v(my)=0 v(mz)=-0.99
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


def run_deck(tag):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run(["ngspice", "-b", f"_{tag}.spice"], cwd=WORK,
                       capture_output=True, text=True, env=env)
    out = WORK / f"_{tag}.csv"
    if r.returncode != 0 or not out.exists():
        (WORK / f"_{tag}.log").write_text(r.stdout + "\n---\n" + r.stderr)
        return None
    a = np.loadtxt(out)
    (WORK / f"_{tag}.spice").unlink(missing_ok=True)
    out.unlink()
    return a[:, 0], a[:, 1]


def classify(mz):
    if mz > MZ_TOL:
        return "P"
    if mz < -MZ_TOL:
        return "AP"
    return "X"


def train_worker(job):
    """job = (idx, child SeedSequence, vamp). The child carries its
    spawn_key, so trajectories are reproducible from
    SeedSequence(master_seed).spawn(N)[idx] — passing bare .entropy would
    collapse every trial onto the same stream."""
    idx, child, vamp = job
    t_end = T0 + (N_PULSES - 1) * CYCLE + TR + TW + TR + TAIL
    tag = f"tr{idx:04d}"
    t0 = time.perf_counter()
    build_deck(tag, child, vamp, N_PULSES, t_end)
    res = run_deck(tag)
    wall = time.perf_counter() - t0
    if res is None:
        return dict(idx=idx, ok=False, wall=wall)
    t, mz = res
    sample_t = [T0 - 0.05e-9 + k * CYCLE for k in range(N_PULSES)] \
        + [t_end - 0.05e-9]
    mz_s = [float(np.interp(ts, t, mz)) for ts in sample_t]
    states = [classify(m) for m in mz_s]
    return dict(idx=idx, ok=True, wall=wall,
                mz_samples=[round(m, 4) for m in mz_s], states=states)


def sweep_worker(job):
    idx, child, vamp = job
    t_end = T0 + TR + TW + TR + TAIL
    tag = f"sw{idx:04d}"
    t0 = time.perf_counter()
    build_deck(tag, child, vamp, 1, t_end)
    res = run_deck(tag)
    wall = time.perf_counter() - t0
    if res is None:
        return dict(idx=idx, vamp=vamp, ok=False, wall=wall)
    t, mz = res
    mzf = float(np.interp(t_end - 0.05e-9, t, mz))
    return dict(idx=idx, vamp=vamp, ok=True, wall=wall,
                mz_final=round(mzf, 4), state=classify(mzf))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def run_pool(worker, jobs, njobs):
    from multiprocessing import Pool
    out = []
    with Pool(njobs) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, jobs)):
            out.append(r)
            if (i + 1) % 25 == 0:
                w = np.mean([x["wall"] for x in out])
                print(f"  {i+1}/{len(jobs)} mean wall/run {w:.1f}s",
                      flush=True)
    return out


def analyze_train(rows, n_seeds_requested):
    ok = [r for r in rows if r.get("ok")]
    # per-seed CSV (seed identity = SeedSequence(master_seed).spawn(N)[idx])
    with open(WORK / "train_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "s0", "s1", "s2", "s3", "s_final",
                    "mz0", "mz1", "mz2", "mz3", "mz_final"])
        for r in sorted(ok, key=lambda x: x["idx"]):
            w.writerow([r["idx"], *r["states"], *r["mz_samples"]])
    st = np.array([r["states"] for r in ok])   # columns: s0..s3, s_final
    n = len(ok)
    valid0 = st[:, 0] == "AP"                  # started AP as intended
    res = dict(n_requested=n_seeds_requested, n_ok=n,
               n_failed_runs=len(rows) - n,
               n_started_AP=int(valid0.sum()),
               n_ambiguous_samples=int((st == "X").sum()))
    # success after pulse k (state sampled before pulse k+1 / final)
    ap_after = []
    for k in range(1, N_PULSES + 1):
        col = k if k < N_PULSES else N_PULSES   # s1..s3 then s_final
        ap_after.append(st[valid0, col] == "AP")
    r1_k = int((~ap_after[0]).sum())
    nv = int(valid0.sum())
    res["r1"] = r1_k / nv
    res["r1_ci"] = wilson(r1_k, nv)
    cond = []
    for k in range(1, N_PULSES):
        prev_fail = ap_after[k - 1]
        both = prev_fail & ap_after[k]
        nprev = int(prev_fail.sum())
        cond.append(dict(k=k, n_fail_k=nprev,
                         n_fail_k_plus_1=int(both.sum()),
                         p_fail_next_given_fail=(int(both.sum()) / nprev
                                                 if nprev else float("nan")),
                         ci=wilson(int(both.sum()), nprev)))
    res["conditional_chain"] = cond
    backhop = []
    for k in range(1, N_PULSES):
        prev_p = ~ap_after[k - 1]
        knocked = prev_p & ap_after[k]
        npv = int(prev_p.sum())
        backhop.append(dict(k=k, n_P_at_k=npv,
                            n_AP_at_k_plus_1=int(knocked.sum()),
                            p_backhop=(int(knocked.sum()) / npv
                                       if npv else float("nan")),
                            ci=wilson(int(knocked.sum()), npv)))
    res["backhop_chain"] = backhop
    surv = []
    for k in range(1, N_PULSES + 1):
        nk = int(ap_after[k - 1].sum())
        surv.append(dict(k=k, n_still_AP=nk, p_still_AP=nk / nv,
                         ci=wilson(nk, nv),
                         independence_pred=(1 - res["r1"]) ** k))
    res["still_AP_vs_independence"] = surv
    return res


def analyze_sweep(rows):
    out = []
    for v in SWEEP_V:
        rs = [r for r in rows if r.get("ok") and r["vamp"] == v]
        k = sum(1 for r in rs if r["state"] == "P")
        n = len(rs)
        sig = 1.0 / (1.0 + math.exp(-((abs(v) - VTH) / VT)))
        out.append(dict(vamp=v, i_sot_mA=round(v / RSOT * 1e3, 3), n=n,
                        n_switched=k, p_sw=(k / n if n else float("nan")),
                        ci=wilson(k, n),
                        sigmoid_mirror_P_to_AP=round(sig, 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot",
                    choices=["pilot", "train", "sweep", "all"])
    ap.add_argument("--train-seeds", type=int, default=1000)
    ap.add_argument("--sweep-seeds", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--master-seed", type=int, default=20260720)
    args = ap.parse_args()

    ensure_osdi()
    (WORK / ".spiceinit").write_text("osdi vgsot_llg.osdi\n")

    train_ch = np.random.SeedSequence(args.master_seed) \
        .spawn(args.train_seeds)
    sweep_ch = np.random.SeedSequence(args.master_seed + 1) \
        .spawn(args.sweep_seeds * len(SWEEP_V))

    if args.mode == "pilot":
        print(f"H_th (per 1 ps step, per axis) = {H_TH:.4e} A/m")
        for i in range(2):
            r = train_worker((i, train_ch[i], V_RESET))
            print("train pilot:", json.dumps(r))
        r = sweep_worker((0, sweep_ch[0], -0.90))
        print("sweep pilot:", json.dumps(r))
        return

    summary = dict(
        _label=("RX-05a LLG reset-pulse correlation; macrospin LLG "
                "(vendored Chapter-2 vgsot_llg.va, Linux-recompiled OSDI) "
                "+ Brown-1963 thermal field via hx/hy/hz; ALL numbers "
                "SIMULATED (macrospin), not measured"),
        va_source=str(VA), v_reset=V_RESET,
        i_reset_mA=round(V_RESET / RSOT * 1e3, 3),
        tw_s=TW, tr_s=TR, cycle_s=CYCLE, n_pulses=N_PULSES,
        h_th_per_axis_Apm=H_TH, t_step_s=T_STEP, temp_K=TEMP,
        device=dict(Ms=MS, alpha=ALPHA, tf=TF, D_elec=D_ELEC,
                    V_free_m3=V_FREE, R_sot=RSOT),
        anchors=dict(vth_P_to_AP=VTH, vt=VT, measured_plateau=0.72),
        master_seed=args.master_seed,
        classify=("P: mz>+0.2, AP: mz<-0.2, X otherwise; state sampled "
                  "50 ps before each next pulse and 2.95 ns after the "
                  "last edge"),
    )

    if args.mode in ("train", "all"):
        print(f"train: {args.train_seeds} seeds x {N_PULSES} pulses "
              f"at {V_RESET} V, jobs={args.jobs}")
        t0 = time.perf_counter()
        jobs = [(i, train_ch[i], V_RESET) for i in range(args.train_seeds)]
        rows = run_pool(train_worker, jobs, args.jobs)
        summary["train"] = analyze_train(rows, args.train_seeds)
        summary["train"]["wall_total_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(summary["train"], indent=2))

    if args.mode in ("sweep", "all"):
        print(f"sweep: {SWEEP_V} x {args.sweep_seeds} seeds")
        t0 = time.perf_counter()
        jobs = []
        j = 0
        for v in SWEEP_V:
            for _ in range(args.sweep_seeds):
                jobs.append((j, sweep_ch[j], v))
                j += 1
        rows = run_pool(sweep_worker, jobs, args.jobs)
        with open(WORK / "sweep_per_seed.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "vamp", "mz_final", "state"])
            for r in sorted([x for x in rows if x.get("ok")],
                            key=lambda x: x["idx"]):
                w.writerow([r["idx"], r["vamp"], r["mz_final"], r["state"]])
        summary["sweep"] = analyze_sweep(rows)
        summary["sweep_wall_total_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(summary["sweep"], indent=2))

    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"-> {SUMMARY}")


if __name__ == "__main__":
    main()
