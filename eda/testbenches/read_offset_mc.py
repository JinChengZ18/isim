#!/usr/bin/env python3
"""RX-06 — input-referred OFFSET of the read comparator (Pelgrom mismatch MC).

The Section 3.5.4 read path currently lives inside update_energy.py (function
run_read) and is characterized at tt/nominal only, where the decision is
trivially correct: the 0.2 V read rail drives Rref = Rp*(1+TMR/2) = 7350 ohm
into the device read branch, giving vsen ~ 0.080 V (st=0, P) / ~0.114 V (st=1,
AP) against a 0.100 V resistive midpoint reference.  The two static decision
margins are therefore only -20.0 mV and +14.3 mV -- the same order as a
small-device Pelgrom offset.  This harness extracts the comparator into a
standalone deck and measures the offset distribution, so "correct=True" can be
re-stated at mismatch instead of at nominal only.

Four blocks, all in one invocation:

  (0) NOMINAL     full read deck (divider + committed OSDI device + the
                  PMOS-input StrongARM), zero mismatch, st in {0,1}: reproduces
                  vsen/vref and records the two static margins.      MEASURED

  (K) THRESHOLD   zero-mismatch DECISION THRESHOLD of the full deck, measured as
                  the trim voltage in series with the sense input at which the
                  decision flips, swept over an added node capacitance C_node.
                  At C_node = 0 (the as-committed deck) the sense and reference
                  nodes are purely resistive (Zth = 2940 ohm at P / 4200 ohm at
                  AP, against a fixed 3675 ohm reference), so comparator
                  kickback displaces them unequally and the effective margin is
                  NOT the static divider margin.  Damping both nodes drives the
                  threshold back to the static margin: the conservative design
                  case, since a real reference would be decoupled.    MEASURED

  (1) MC-IDEAL    the comparator alone, ideal sources at the 0.100 V reference
                  common mode, differential sweep, N samples of per-device Vth
                  mismatch.  Input-referred offset = the differential input at
                  which the decision flips.  Headline sigma_off; direct port of
                  04PBNNSim/smtj_pbnn_sim/eda/hero/run_offset_mc.py (same AVT
                  assumption, same sweep-and-interpolate extraction, same
                  "report the ratio, not the absolute mV" convention).  MEASURED

  (2) MC-INSITU   the SAME mismatch draws on the FULL read deck (real divider
                  source impedance, real per-state common mode, real kickback),
                  at each C_node in --mc-cap.  The sign of the crossing is a
                  DIRECTLY MEASURED misread indicator, so the per-state misread
                  rate is not taken on faith from a Gaussian tail.    MEASURED

Mismatch model (ASSUMPTION, inherited from the sibling harness): per-device
threshold mismatch sigma_Vth = AVT / sqrt(W*L) with a sky130-class
AVT = 5.0 mV*um.  This is NOT the PDK statistical model; the transferable
quantity is the ratio sigma_off / V_T (and sigma_off / margin), not absolute mV.
Offset sources are injected on the input pair and on the two latch devices whose
sources are the integration nodes da/db -- device-for-device the same four
injection points as the sibling StrongARM netlist.  The cross-coupled pair
(XS5/XS6) is not injected, same convention as the sibling, so sigma_off is a
lower bound on the full-latch offset.

Every ngspice session parses the full sky130 library (~20 s), which dominates
the cost of a one-sample-per-process harness.  All samples of one arm are
therefore driven from a SINGLE session by `alter`ing the four offset sources
between sweeps; the transient always restarts from its own DC operating point,
so samples remain independent.

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/read_offset_mc.py
      [--n 120] [--seed 20260721] [--mc-cap 0 200f] [--smoke]
Writes read_offset_mc_summary.json next to this script.
"""
from __future__ import annotations

import argparse
import math
import re
import time

import numpy as np

from _common import (HERE, SKY130_LIB, VT, RP, TMR, load_wrdata, run_deck,
                     write_summary)

CORNER = "tt"
VDD = 1.8
VREAD = 0.2                       # read rail [V]
RREF = RP * (1.0 + TMR / 2.0)     # 7350 ohm midpoint reference resistor
TR = 50e-12
TCLK, TEVAL = 1e-9, 2e-9
MEAS_T = "2.9n"                   # decision sample, inside the eval phase
VCM_REF = 0.1                     # reference-node common mode [V]

AVT = 5.0e-3                      # sky130-class Pelgrom AVT [V*um] (ASSUMPTION)
# (device, W[um], L[um]) of each offset-injection point, in os0..os3 order
INJ = [("XS1_input_p", 8.0, 0.15), ("XS2_input_n", 8.0, 0.15),
       ("XS3_latch_p", 2.0, 0.15), ("XS4_latch_n", 2.0, 0.15)]

HALF, STEP = 0.080, 0.002         # sweep half-width and step [V]
CAP_AXIS = [0.0, 10e-15, 50e-15, 200e-15, 1e-12]   # threshold-vs-C_node axis
CAP_EPS = 1e-18                   # placeholder cap value, altered at run time

# PMOS-input StrongARM, verbatim device list from update_energy.SA_CORE with the
# four gate nodes broken out so a DC offset source can be inserted (mirror of
# the sibling comparators/strongarm.spice injection scheme).
SA_CORE = """Vo1 g1 {inp} DC 0
Vo2 g2 {inn} DC 0
Vo3 g3 outp  DC 0
Vo4 g4 outn  DC 0
XMtail ptail nclk vdd vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS1    da    g1    ptail vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS2    db    g2    ptail vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS3    outn  g3    da   vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
XS4    outp  g4    db   vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
XS5    outn  outp  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XS6    outp  outn  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp1   outp  nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp2   outn  nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp3   da    nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp4   db    nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
"""

HEAD = (f".lib {SKY130_LIB} {CORNER}\n"
        f"Vdd vdd 0 {VDD}\n"
        f"Vnclk nclk 0 PULSE({VDD} 0 {TCLK} {TR} {TR} {TEVAL} 10n)\n")

PT_RE = re.compile(r"PT\s+(\S+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")


def divider(st):
    """Read divider + committed OSDI device + midpoint reference + node caps."""
    return (f"Vread vread 0 {VREAD}\n"
            f"Rref vread vsen {RREF:.0f}\n"
            f"Rr1  vread vinn {RREF:.0f}\n"
            f"Rr2  vinn  0    {RREF:.0f}\n"
            f".model smtj_sot smtj_sot\n"
            f"N1 wr vsen com stn pswn taun sinfn smtj_sot\n"
            f"Vwr wr 0 0\nVcom com 0 0\nVst stn 0 {st}\n"
            f"Rpsw pswn 0 1e12\nRtau taun 0 1e12\nRsinf sinfn 0 1e12\n"
            f"Csen vsen 0 {CAP_EPS:.6e}\nCrefn vinn 0 {CAP_EPS:.6e}\n")


def sigma_list():
    """Per-injection-point Pelgrom sigma_Vth = AVT / sqrt(W*L) [V]."""
    return [AVT / math.sqrt(w * l) for _, w, l in INJ]


def batched_sweep(body, sweep_alter, jobs, tag):
    """One ngspice session; each job = (job_id, setup_alter_lines, center).

    Returns {job_id: [(swept_value, vop-von), ...]} sorted by swept value.
    """
    ctrl = [".control"]
    for jid, setup, center in jobs:
        ctrl += [f"  * job {jid}"] + [f"  {ln}" for ln in setup]
        ctrl += [f"  let vd = {center - HALF:.9f}",
                 f"  dowhile vd <= {center + HALF + STEP / 2:.9f}",
                 f"    {sweep_alter}",
                 "    tran 5p 4n",
                 f"    meas tran vop find v(outp) at={MEAS_T}",
                 f"    meas tran von find v(outn) at={MEAS_T}",
                 f"    echo PT {jid} $&vd $&vop $&von",
                 # every tran allocates a new plot; without this the session
                 # accumulates one plot per sweep point (>1 GB and slowing
                 # down badly by ~5000 points -- see TRIAL_LOG 2026-07-22)
                 "    destroy $curplot",
                 f"    let vd = vd + {STEP:.9f}",
                 "  end"]
    ctrl += ["  quit", ".endc", ".end", ""]
    out = run_deck(body + "\n".join(ctrl), tag)
    res = {jid: [] for jid, _, _ in jobs}
    for m in PT_RE.finditer(out):
        jid = m.group(1)
        if jid in res:
            res[jid].append((float(m.group(2)),
                             float(m.group(3)) - float(m.group(4))))
    for v in res.values():
        v.sort()
    return res


def crossing(pts):
    """Linear-interpolated zero crossing of (vop - von) vs the swept input."""
    if len(pts) < 3:
        return float("nan")
    x = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if not len(idx):
        return float("nan")
    i = idx[0]
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))


def off_alter(os):
    return [f"alter @vo{i+1}[dc] = {v:.9f}" for i, v in enumerate(os)]


# --------------------------------------------------------------- (0) nominal
def run_nominal(st, tag):
    """Full read deck, zero mismatch: vsen, vref, decision, static margin."""
    deck = (f"* RX-06 nominal read decision, st={st}\n" + HEAD + divider(st)
            + SA_CORE.format(inp="vsen", inn="vinn")
            + (f".control\n  tran 5p 4n\n"
               f"  wrdata _{tag}.csv v(outp) v(outn) v(vsen) v(vinn)\n"
               f"  quit\n.endc\n.end\n"))
    run_deck(deck, tag)
    t, (voutp, voutn, vsen, vref) = load_wrdata(HERE / f"_{tag}.csv", 4)
    i = int(np.searchsorted(t, TCLK + TEVAL - 0.1e-9))
    op, on = float(voutp[i]), float(voutn[i])
    resolved = abs(op - on) > 0.9 * VDD
    vs, vr = float(vsen[0]), float(vref[0])
    return dict(st=st, vsen_V=round(vs, 6), vref_V=round(vr, 6),
                static_margin_mV=round((vs - vr) * 1e3, 4),
                outp_V=round(op, 4), outn_V=round(on, 4),
                resolved=bool(resolved),
                correct=bool(resolved and ((op > on) == (st == 1))))


# ------------------------------------------------------------------ analytic
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def misread_gauss(margin_V, mean_V, sigma_V):
    """P(misread) for one state under a Gaussian input-referred offset.

    The comparator decides "AP" when (v_sense - v_ref) > V_os.  A state whose
    signed decision margin is `margin` is misread when margin < V_os (positive
    margin, st=1) or margin > V_os (negative margin, st=0).
    """
    z = (margin_V - mean_V) / sigma_V
    return (1.0 - _phi(z)) if margin_V > 0 else _phi(z)


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _stats(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    if len(a) < 2:
        return float("nan"), float("nan"), int(len(a))
    return float(a.mean()), float(a.std(ddof=1)), int(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--mc-cap", nargs="+", type=float, default=[0.0, 200e-15],
                    help="C_node values [F] at which the in-situ MC is run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_mc = 4 if args.smoke else args.n
    caps_mc = args.mc_cap[:1] if args.smoke else args.mc_cap
    t_all = time.perf_counter()

    # -------------------------------------------------------------- (0)
    print("[0] nominal read decision (zero mismatch, full deck)", flush=True)
    nominal = [run_nominal(0, "rd_nom_st0"), run_nominal(1, "rd_nom_st1")]
    for r in nominal:
        print(f"    st={r['st']}: vsen={r['vsen_V']:.5f} V  vref={r['vref_V']:.5f} V"
              f"  static margin={r['static_margin_mV']:+.2f} mV  "
              f"correct={r['correct']}")
    marg = {r["st"]: r["static_margin_mV"] * 1e-3 for r in nominal}
    sep = abs(marg[1] - marg[0])
    print(f"    separation |vsen(1)-vsen(0)| = {sep*1e3:.3f} mV;  "
          f"V_T = {VT*1e3:.3f} mV", flush=True)

    # -------------------------------------------------------------- (K)
    print("[K] zero-mismatch decision threshold vs node capacitance", flush=True)
    cap_axis = sorted(set(CAP_AXIS) | set(caps_mc))
    thr = {c: {} for c in cap_axis}
    for st in (0, 1):
        jobs = [(f"c{i}",
                 [f"alter @csen[capacitance] = {max(c, CAP_EPS):.6e}",
                  f"alter @crefn[capacitance] = {max(c, CAP_EPS):.6e}"],
                 0.0 if c >= 100e-15 else (0.06 if st == 0 else -0.04))
                for i, c in enumerate(cap_axis)]
        res = batched_sweep(f"* RX-06 threshold vs C_node, st={st}\n" + HEAD
                            + divider(st) + "Vtrim vinp vsen DC 0\n"
                            + SA_CORE.format(inp="vinp", inn="vinn"),
                            "alter @vtrim[dc] = vd", jobs, f"rd_thr_st{st}")
        for i, c in enumerate(cap_axis):
            thr[c][st] = crossing(res[f"c{i}"])
    for c in cap_axis:
        print(f"    C_node={c*1e15:8.1f} fF:  trim*(st=0)={thr[c][0]*1e3:+7.2f} mV"
              f"   trim*(st=1)={thr[c][1]*1e3:+7.2f} mV"
              f"   [static: {marg[0]*1e3:+.2f} / {marg[1]*1e3:+.2f} mV]",
              flush=True)

    # -------------------------------------------------------------- (1)
    sig = sigma_list()
    print(f"[mismatch] AVT={AVT*1e3:.1f} mV*um (ASSUMPTION); per-device "
          f"sigma_Vth = " + ", ".join(f"{nm}:{s*1e3:.2f} mV"
                                      for (nm, _, _), s in zip(INJ, sig)),
          flush=True)
    rng = np.random.default_rng(args.seed)
    draws = [[float(rng.normal(0.0, s)) for s in sig] for _ in range(n_mc)]

    print(f"[1] MC-IDEAL: {n_mc} samples, ideal drive at Vcm={VCM_REF} V",
          flush=True)
    t0 = time.perf_counter()
    res = batched_sweep(
        "* RX-06 offset MC, ideal drive at the reference common mode\n" + HEAD
        + f"Vinp vinp 0 {VCM_REF}\nVinn vinn 0 {VCM_REF}\n"
        + SA_CORE.format(inp="vinp", inn="vinn"),
        f"alter @vinp[dc] = {VCM_REF} + vd/2\n    alter @vinn[dc] = {VCM_REF} - vd/2",
        [(f"s{i}", off_alter(os_), 0.0) for i, os_ in enumerate(draws)],
        "rd_mc_ideal")
    off_ideal = [crossing(res[f"s{i}"]) for i in range(n_mc)]
    mean_i, sigma_i, nfin_i = _stats(off_ideal)
    print(f"    mean={mean_i*1e3:+.3f} mV  sigma={sigma_i*1e3:.3f} mV  "
          f"(finite {nfin_i}/{n_mc}, {time.perf_counter()-t0:.0f} s)")
    print(f"    sigma_off/V_T={sigma_i/VT:.3f}  "
          f"sigma_off/|separation|={sigma_i/sep:.3f}  "
          f"sigma_off/|margin(st=0)|={sigma_i/abs(marg[0]):.3f}  "
          f"sigma_off/|margin(st=1)|={sigma_i/abs(marg[1]):.3f}", flush=True)

    # -------------------------------------------------------------- (2)
    insitu = {}
    for cap in caps_mc:
        print(f"[2] MC-INSITU at C_node={cap*1e15:.1f} fF: {n_mc} samples x 2 states",
              flush=True)
        per_state = {}
        for st in (0, 1):
            c0 = thr[cap][st]
            t0 = time.perf_counter()
            setup = [f"alter @csen[capacitance] = {max(cap, CAP_EPS):.6e}",
                     f"alter @crefn[capacitance] = {max(cap, CAP_EPS):.6e}"]
            res = batched_sweep(
                f"* RX-06 in-situ offset MC, st={st}, C_node={cap:.3e}\n" + HEAD
                + divider(st) + "Vtrim vinp vsen DC 0\n"
                + SA_CORE.format(inp="vinp", inn="vinn"),
                "alter @vtrim[dc] = vd",
                [(f"s{i}", setup + off_alter(os_), c0)
                 for i, os_ in enumerate(draws)],
                f"rd_mc_insitu_st{st}")
            ts = [crossing(res[f"s{i}"]) for i in range(n_mc)]
            # correct decision at trim=0: trim*>0 for st=0, trim*<0 for st=1
            bad = sum(1 for t in ts if np.isfinite(t)
                      and not ((t > 0) if st == 0 else (t < 0)))
            nf = sum(1 for t in ts if np.isfinite(t))
            lo, hi = wilson(bad, nf)
            # offset around the zero-mismatch threshold, sign-normalized so a
            # POSITIVE value always erodes that state's margin
            devs = [(t - c0) * (-1.0 if st == 0 else +1.0) for t in ts]
            m_d, s_d, _ = _stats(devs)
            per_state[st] = dict(
                n=n_mc, n_finite=nf, misread=bad,
                misread_frac=(bad / nf if nf else float("nan")),
                wilson_lo=lo, wilson_hi=hi,
                threshold_zero_mismatch_mV=round(c0 * 1e3, 4),
                effective_margin_mV=round(abs(c0) * 1e3, 4),
                offset_mean_mV=round(m_d * 1e3, 4),
                offset_sigma_mV=round(s_d * 1e3, 4),
                trim_star_mV=[round(t * 1e3, 4) for t in ts])
            print(f"    st={st}: MEASURED misread {bad}/{nf} = "
                  f"{(bad/nf if nf else float('nan')):.4f} [{lo:.4f},{hi:.4f}]"
                  f"  in-situ offset mean={m_d*1e3:+.2f} sigma={s_d*1e3:.2f} mV"
                  f"  (eff. margin {abs(c0)*1e3:.1f} mV, "
                  f"{time.perf_counter()-t0:.0f} s)", flush=True)
        k = sum(per_state[st]["misread"] for st in (0, 1))
        n = sum(per_state[st]["n_finite"] for st in (0, 1))
        plo, phi = wilson(k, n)
        insitu[cap] = dict(states={str(st): per_state[st] for st in (0, 1)},
                           pooled=dict(k=k, n=n,
                                       frac=(k / n if n else float("nan")),
                                       wilson_lo=plo, wilson_hi=phi))
        print(f"    pooled: {k}/{n} = {(k/n if n else float('nan')):.4f} "
              f"[{plo:.4f},{phi:.4f}]", flush=True)

    # ------------------------------------------------------- (3) channel
    p_static = {st: misread_gauss(marg[st], mean_i, sigma_i) for st in (0, 1)}
    p_static_avg = 0.5 * (p_static[0] + p_static[1])
    c_ref = caps_mc[0]
    p_eff = {}
    for st in (0, 1):
        d = insitu[c_ref]["states"][str(st)]
        p_eff[st] = misread_gauss(abs(d["threshold_zero_mismatch_mV"]) * 1e-3,
                                  d["offset_mean_mV"] * 1e-3,
                                  d["offset_sigma_mV"] * 1e-3)
    p_eff_avg = 0.5 * (p_eff[0] + p_eff[1])
    print("[3] misread channel")
    print(f"    CONSERVATIVE (static margins, damped nodes): "
          f"st0={p_static[0]:.5f} st1={p_static[1]:.5f} avg={p_static_avg:.5f}")
    print(f"    AS-COMMITTED (C_node={c_ref*1e15:.0f} fF effective margins): "
          f"st0={p_eff[0]:.3e} st1={p_eff[1]:.3e} avg={p_eff_avg:.3e}",
          flush=True)

    summary = dict(
        _label=("RX-06 MEASURED input-referred offset of the Section 3.5.4 read "
                "comparator (ngspice tran, sky130 tt, schematic level) under a "
                "Pelgrom Vth-mismatch Monte-Carlo; the per-read misread "
                "probability fed to the solver is ANALYTIC (Gaussian tail of "
                "the MEASURED offset distribution) and is cross-checked against "
                "the directly MEASURED in-situ misread counts"),
        rng=dict(seed=args.seed, generator="numpy.random.default_rng",
                 n_samples=n_mc,
                 note="one 4-vector of Vth offsets per sample; the SAME draws "
                      "are reused for MC-IDEAL and every MC-INSITU arm"),
        mismatch_model=dict(
            label="ASSUMPTION", AVT_mV_um=AVT * 1e3,
            formula="sigma_Vth = AVT / sqrt(W*L)",
            injection=[dict(name=nm, W_um=w, L_um=l, WL_um2=round(w * l, 4),
                            sigma_Vth_mV=round(s * 1e3, 4))
                       for (nm, w, l), s in zip(INJ, sig)],
            note=("sky130-class AVT, NOT the PDK statistical model; the "
                  "transferable quantities are sigma_off/V_T and "
                  "sigma_off/margin. Cross-coupled pair XS5/XS6 not injected "
                  "(same 4-point convention as the sibling StrongARM netlist), "
                  "so sigma_off is a lower bound on the full-latch offset."),
            ported_from=("04PBNNSim/smtj_pbnn_sim/eda/hero/run_offset_mc.py "
                         "+ comparators/strongarm.spice injection scheme")),
        read_path=dict(
            label="MEASURED", rail_V=VREAD, rref_ohm=RREF, vcm_ref_V=VCM_REF,
            comparator="PMOS-input StrongARM (update_energy.SA_CORE, verbatim)",
            meas_t=MEAS_T, corner=CORNER, VT_mV=VT * 1e3,
            nominal=nominal,
            separation_mV=round(sep * 1e3, 4),
            static_margin_mV={str(st): round(marg[st] * 1e3, 4)
                              for st in (0, 1)},
            source_impedance_ohm={
                "0": round(RREF * RP / (RREF + RP), 1),
                "1": round(RREF * RP * (1 + TMR) / (RREF + RP * (1 + TMR)), 1),
                "ref": round(RREF / 2.0, 1)}),
        decision_threshold=dict(
            label="MEASURED",
            note=("zero-mismatch trim* = the series-trim voltage at which the "
                  "real decision flips; |trim*| is the EFFECTIVE decision "
                  "margin. At C_node=0 the sense-node source impedance is "
                  "state-dependent (2940 ohm at P, 4200 ohm at AP) against a "
                  "fixed 3675 ohm reference, so comparator kickback displaces "
                  "the two nodes unequally in the direction that REINFORCES "
                  "the correct decision; damping both nodes removes the effect "
                  "and returns trim* to the static divider margin. The damped "
                  "case is the conservative design case because a real "
                  "reference would be decoupled."),
            rows=[dict(c_node_fF=round(c * 1e15, 3),
                       trim_star_st0_mV=round(thr[c][0] * 1e3, 4),
                       trim_star_st1_mV=round(thr[c][1] * 1e3, 4))
                  for c in cap_axis]),
        mc_ideal=dict(
            label="MEASURED", n=n_mc, n_finite=nfin_i,
            sweep=dict(half_V=HALF, step_V=STEP,
                       extraction="linear interpolation of the (vop-von) zero "
                                  "crossing across the differential sweep"),
            offset_mean_mV=round(mean_i * 1e3, 4),
            offset_sigma_mV=round(sigma_i * 1e3, 4),
            sigma_over_VT=round(sigma_i / VT, 4),
            sigma_over_separation=round(sigma_i / sep, 4),
            sigma_over_static_margin={str(st): round(sigma_i / abs(marg[st]), 4)
                                      for st in (0, 1)},
            offsets_mV=[round(o * 1e3, 4) for o in off_ideal]),
        mc_insitu={f"{c*1e15:.0f}fF": insitu[c] for c in caps_mc},
        misread_channel=dict(
            label="ANALYTIC",
            model=("V_os ~ N(mean, sigma); state st is misread iff its signed "
                   "decision margin falls on the wrong side of V_os: "
                   "P(st=1) = 1 - Phi((margin_1 - mean)/sigma), "
                   "P(st=0) = Phi((margin_0 - mean)/sigma)"),
            conservative=dict(
                basis=("static divider margins with the MC-IDEAL offset "
                       "statistics; corresponds to a decoupled reference / "
                       "damped sense node, i.e. no kickback help"),
                p_misread_st0=p_static[0], p_misread_st1=p_static[1],
                p_read_flip=p_static_avg,
                solver_points=dict(x1=p_static_avg, x3=3 * p_static_avg,
                                   x10=10 * p_static_avg)),
            as_committed=dict(
                basis=(f"effective margins measured on the as-committed deck "
                       f"(C_node={c_ref*1e15:.0f} fF) with the in-situ offset "
                       f"statistics at that C_node"),
                p_misread_st0=p_eff[0], p_misread_st1=p_eff[1],
                p_read_flip=p_eff_avg),
            p_read_flip_note=(
                "equiprobable-state average. This is a POPULATION average over "
                "devices: a real static offset makes a small fraction of spins "
                "misread deterministically rather than making every spin "
                "misread occasionally. The i.i.d. per-read flip model used in "
                "circuit_backends.CircuitChainSpin therefore reproduces the "
                "average corruption rate but not its per-spin stickiness "
                "(see that backend's docstring)."),
            fed_to_solver="conservative"),
        wall_s=round(time.perf_counter() - t_all, 1))
    write_summary(HERE / "read_offset_mc_summary.json", summary)


if __name__ == "__main__":
    main()
