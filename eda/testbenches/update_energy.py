#!/usr/bin/env python3
"""W3 — transient energy + timing accounting of ONE Gibbs update on the write chain.

One update = k reset pulses (AP->P, chain driven at TOP code) + 1 probabilistic
write pulse (P->AP, characterized at MID code = worst-case current point) + 1
read strobe.  Every energy number here is an ngspice transient/op measurement
on the W2 chain (update_chain_dc.py) except the DAC-string static, which is the
committed analytic formula (labeled ANALYTIC).

DESIGN CORRECTION (measured in block 0 below, logged in .agents/TRIAL_LOG_eda.md):
gating the W2 chain by simply pulsing wen/wep OPENS the feedback loop (W2 takes
feedback from wr, after the enable TG), the buffer rails to Vdd between pulses
and cannot recover within 0.75 ns -> the device sees ~1.6 V for the whole pulse
(~2.7 pJ, psw pinned at 1).  Fix kept here: current-steering.  A matched replica
branch (8-finger TG copy into an Rsot-value resistor) conducts exactly when the
device branch is off (complementary gates), so the buffer load current is
constant and drv never moves; the feedback node is steered wr<->wrr by a small
TG pair so the loop never opens.  An intermediate always-on-replica variant was
also tried and rejected: the doubled load current during the pulse droops the
delivered level 30..90 mV (see trial log).

Consequence for the statics accounting: with steering, the supply current is the
same during and between pulses (replica burn <-> device burn swap), so the
literal prescribed sum k*(E_dev+E_tg)+P_buf*t_update double-counts the pulse
Ohmic energy under the always-on accounting; the JSON therefore also carries a
supply-true accounting (rail power x awake time, exact under steering).  Power
gating between updates is standard practice; the sibling repo reproduced 24-36%
primitive-level savings in design_survey/repro/picoram_gating.py (referenced,
not re-run).

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/update_energy.py [--smoke]
Writes update_energy_summary.json next to this script.
"""
from __future__ import annotations

import re
import sys

import numpy as np

from _common import (HERE, SKY130_LIB, VTH, VT, RSOT, RP, TMR, TW, grab, load_wrdata,
                     run_deck, trapz, write_summary)
from update_chain_dc import _tg, USPAN, RUNIT

VDD = 1.8
TR = 50e-12            # pulse rise/fall [s]
T0 = 2e-9              # pulse start (idle op settles first) [s]
TGAP = 0.25e-9         # inter-pulse gap in the timing model [s]
TREAD = 1e-9           # read strobe slot in the timing model [s]
NB = 6                 # characterization DAC resolution (W2 mid/top codes)
CODE_MID = 2 ** (NB - 1)          # v_wr ~ 0.901 V, worst-case current point
CODE_TOP = 2 ** NB - 1            # v_wr ~ 0.999 V, reset (AP->P) drive
VREAD = 0.2            # read rail [V]
RREF = RP * (1.0 + TMR / 2.0)     # 7350 ohm midpoint reference resistor
SETTLE_TOL = 0.1 * VT             # 2.3 mV settle band

# W2 BUFFER with the feedback node parameterized ({fb}); otherwise verbatim copy
# of update_chain_dc.BUFFER (same devices, same sizes, same Ccm).
BUFFER_FB = """* two-stage Miller unity buffer, NMOS input (Vcm ~ 0.9 V), drives write line
XMt1 btail bnb 0 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XMt2 btail bnb 0 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
Vnb  bnb 0 0.9
XM1  bd1 {inp} btail 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XM2  bd2 {fb}  btail 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XM3  bd1 bd2 vdd vdd sky130_fd_pr__pfet_01v8 W=20 L=0.5
XM4  bd2 bd2 vdd vdd sky130_fd_pr__pfet_01v8 W=20 L=0.5
XMo1 drv bd1 vdd vdd sky130_fd_pr__pfet_01v8 W=80 L=0.5
XMo2 drv bd1 vdd vdd sky130_fd_pr__pfet_01v8 W=80 L=0.5
Ccm  bd1 drv 2p
"""

# device-branch write-enable TG (8 fingers, W2 sizes) + committed OSDI device.
DEV_BRANCH = """XWEn1 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn2 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn3 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn4 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn5 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn6 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn7 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEn8 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XWEp1 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp2 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp3 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp4 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp5 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp6 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp7 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XWEp8 drv wep wr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
.model smtj_sot smtj_sot
N1 wr rd com st pswn taun sinfn smtj_sot
Vcom com 0 0
Vrd  rd 0 0
Vst  st 0 0
Rpsw  pswn 0 1e12
Rtau  taun 0 1e12
Rsinf sinfn 0 1e12
"""

# current-steering: replica TG conducts when the device TG is OFF (gates swapped:
# nfet gate=wep, pfet gate=wen) into an Rsot-value resistor; feedback node fbn is
# steered to wr during the pulse and to wrr when idle, so the loop never opens.
STEER = f"""* complementary replica branch + steered feedback (constant buffer load)
XRWn1 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn2 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn3 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn4 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn5 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn6 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn7 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWn8 drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
XRWp1 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp2 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp3 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp4 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp5 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp6 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp7 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XRWp8 drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
Rrep wrr 0 {RSOT:.0f}
* feedback steering TGs: fbn = wr during the pulse, wrr when idle
XFB1n wr  wen fbn 0   sky130_fd_pr__nfet_01v8 W=4 L=0.15
XFB1p wr  wep fbn vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XFB2n wrr wep fbn 0   sky130_fd_pr__nfet_01v8 W=4 L=0.15
XFB2p wrr wen fbn vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
"""


def dac_rails():
    return VTH - USPAN * VT, VTH + USPAN * VT


def chain_prefix(code, nbits=NB):
    """Rails + resistor-string DAC + code-select TG (verbatim W2 construction)."""
    vlo, vhi = dac_rails()
    ntap = 2 ** nbits
    s = (f"* W3 update-energy chain, code={code}/{ntap - 1} nbits={nbits}\n"
         f".lib {SKY130_LIB} tt\n"
         f"Vdd vdd 0 {VDD}\n"
         f"Vhi vhi 0 {vhi:.6f}\n"
         f"Vlo vlo 0 {vlo:.6f}\n")
    nodes = ["vhi"] + [f"rs{i}" for i in range(1, ntap)] + ["vlo"]
    for i in range(ntap):
        s += f"Rs{i} {nodes[i]} {nodes[i + 1]} {RUNIT}\n"
    s += _tg(nodes[ntap - 1 - code], "bin", 0)
    return s


def pulse_sources():
    return (f"Vwen wen 0 PULSE(0 {VDD} {T0} {TR} {TR} {TW} 100n)\n"
            f"Vwep wep 0 PULSE({VDD} 0 {T0} {TR} {TR} {TW} 100n)\n")


# ---------------------------------------------------------------- (0)+(a)+(b)
def run_pulse(code, tag, steer=True):
    """One gated 0.75 ns write pulse; trapz energy split from the waveforms."""
    deck = chain_prefix(code)
    if steer:
        deck += BUFFER_FB.format(inp="bin", fb="fbn") + STEER
    else:
        deck += BUFFER_FB.format(inp="bin", fb="wr")     # naive: prescribed gating, loop opens
    deck += pulse_sources() + DEV_BRANCH
    deck += (f".control\n  tran 2p 4.5n\n"
             f"  wrdata _{tag}.csv v(wr) v(drv) i(vdd)\n  quit\n.endc\n.end\n")
    run_deck(deck, tag)
    t, (vwr, vdrv, ivdd) = load_wrdata(HERE / f"_{tag}.csv", 3)
    idev = vwr / RSOT
    e_dev = float(trapz(vwr ** 2 / RSOT, t))                  # in the SOT branch
    e_tg = float(trapz((vdrv - vwr) * idev, t))               # in the enable TG
    e_drvstage = float(trapz((VDD - vdrv) * idev, t))         # class-A pfet drop
    flat = (t >= T0 + TR + 0.1e-9) & (t <= T0 + TW)           # flat-top window
    idle = (t >= 1.0e-9) & (t <= 1.9e-9)
    vf = vwr[flat]
    return dict(
        code=code,
        e_dev_pJ=round(e_dev * 1e12, 4), e_tg_pJ=round(e_tg * 1e12, 4),
        e_drvstage_pJ=round(e_drvstage * 1e12, 4),
        vwr_flat_mean_V=round(float(vf.mean()), 5),
        vwr_flat_min_V=round(float(vf.min()), 5),
        vwr_flat_max_V=round(float(vf.max()), 5),
        u_flat_mean=round(float((vf.mean() - VTH) / VT), 3),
        i_vdd_idle_mA=round(float(-ivdd[idle].mean()) * 1e3, 4),
        i_vdd_pulse_mA=round(float(-ivdd[flat].mean()) * 1e3, 4))


# --------------------------------------------------------------------- (c)
def run_op(tag, wen_on):
    """DC op of the steered chain (idle: replica conducts; active: device conducts)."""
    deck = chain_prefix(CODE_MID)
    deck += BUFFER_FB.format(inp="bin", fb="fbn") + STEER
    wen, wep = (VDD, 0.0) if wen_on else (0.0, VDD)
    deck += f"Vwen wen 0 {wen}\nVwep wep 0 {wep}\n" + DEV_BRANCH
    deck += (".control\n  op\n  print v(wr)\n  print v(wrr)\n  print v(drv)\n"
             "  print v(bin)\n  print i(vdd)\n  quit\n.endc\n.end\n")
    out = run_deck(deck, tag)
    m = dict(v_wr=grab(out, "wr"), v_wrr=grab(out, "wrr"), v_drv=grab(out, "drv"),
             v_bin=grab(out, "bin"))
    mm = re.search(r"i\(vdd\)\s*=\s*([-+0-9.eE]+)", out)
    m["i_vdd_A"] = -float(mm.group(1)) if mm else float("nan")
    m["p_rail_mW"] = round(VDD * m["i_vdd_A"] * 1e3, 4)
    return m


# --------------------------------------------------------------------- (d)
def run_settle(tag, upward=True):
    """Full-scale buffer-input step (code-0 tap <-> code-63 tap voltage), idle loop.

    Measures t_settle of the regulated node wrr to within 0.1*VT of final, and the
    supply energy of the settle window.  The DAC string + code TG contribute only
    ~1.6 kohm x ~20 fF ~ 30 ps and are replaced by an ideal stepped source at bin.
    """
    vlo, vhi = dac_rails()
    ntap = 2 ** NB
    v0 = vlo + (vhi - vlo) / ntap        # code-0 tap (one unit R above vlo)
    v1 = vhi                             # code-63 tap
    va, vb = (v0, v1) if upward else (v1, v0)
    tstep, tstop = 10e-9, 90e-9
    deck = (f"* W3 settle test, {'up' if upward else 'down'} full-scale step at bin\n"
            f".lib {SKY130_LIB} tt\n"
            f"Vdd vdd 0 {VDD}\n"
            f"Vbin bin 0 PULSE({va:.6f} {vb:.6f} {tstep} {TR} {TR} 200n 400n)\n")
    deck += BUFFER_FB.format(inp="bin", fb="fbn") + STEER
    deck += f"Vwen wen 0 0\nVwep wep 0 {VDD}\n" + DEV_BRANCH
    deck += (f".control\n  tran 10p {tstop}\n"
             f"  wrdata _{tag}.csv v(wrr) v(drv) i(vdd)\n  quit\n.endc\n.end\n")
    run_deck(deck, tag)
    t, (vwrr, vdrv, ivdd) = load_wrdata(HERE / f"_{tag}.csv", 3)
    vfinal = float(vwrr[t >= tstop - 1e-9].mean())
    off = np.abs(vwrr - vfinal) > SETTLE_TOL
    off[t <= tstep] = False
    t_settle = float(t[off][-1] - tstep) if off.any() else float("nan")
    if off.any() and t[off][-1] > tstop - 2e-9:
        t_settle = float("nan")          # did not settle inside the window
    win = (t >= tstep) & (t <= tstep + (t_settle if np.isfinite(t_settle) else tstop))
    e_supply = float(trapz(VDD * (-ivdd[win]), t[win]))
    p_pre = float(VDD * -ivdd[(t > tstep - 2e-9) & (t < tstep)].mean())
    p_post = float(VDD * -ivdd[t > tstop - 2e-9].mean())
    return dict(direction="up" if upward else "down",
                v_from_V=round(va, 6), v_to_V=round(vb, 6),
                v_final_V=round(vfinal, 6),
                t_settle_ns=round(t_settle * 1e9, 3) if np.isfinite(t_settle) else float("nan"),
                e_supply_window_pJ=round(e_supply * 1e12, 3),
                p_rail_pre_mW=round(p_pre * 1e3, 4), p_rail_post_mW=round(p_post * 1e3, 4))


# --------------------------------------------------------------------- (e)
# PMOS-input StrongARM (device-for-device mirror of the proven sibling NMOS SA in
# smtj_pbnn_sim/eda/hero/strongarm_sa.spice): the read divider sits at ~0.1 V
# common mode, far below the NMOS-input range, so n<->p, vdd<->gnd, clk<->nclk.
# Evaluate while nclk is LOW; outputs predischarge to 0 while nclk is high.
SA_CORE = """XMtail ptail nclk vdd vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS1    da    vsen  ptail vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS2    db    vref  ptail vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15
XS3    outn  outp  da   vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
XS4    outp  outn  db   vdd sky130_fd_pr__pfet_01v8 W=2 L=0.15
XS5    outn  outp  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XS6    outp  outn  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp1   outp  nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp2   outn  nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp3   da    nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
XSp4   db    nclk  0    0   sky130_fd_pr__nfet_01v8 W=2 L=0.15
"""


def run_read(st, tag):
    """One SA strobe on the device read divider; supply-charge integral = E_read.

    Divider: 0.2 V read rail -> Rref=7350 -> vsen -> device read branch (rd..com=0);
    reference: rail -> Rref -> vref -> Rref -> 0 = 0.1 V midpoint.  st=0 (P, Rp) gives
    vsen ~ 0.080 V < vref; st=1 (AP, Rap) gives vsen ~ 0.114 V > vref.
    """
    tclk, teval = 1e-9, 2e-9
    deck = (f"* W3 read: PMOS-input StrongARM over the device read divider, st={st}\n"
            f".lib {SKY130_LIB} tt\n"
            f"Vdd vdd 0 {VDD}\n"
            f"Vread vread 0 {VREAD}\n"
            f"Rref vread vsen {RREF:.0f}\n"
            f"Rr1  vread vref {RREF:.0f}\n"
            f"Rr2  vref  0    {RREF:.0f}\n"
            f".model smtj_sot smtj_sot\n"
            f"N1 wr vsen com stn pswn taun sinfn smtj_sot\n"
            f"Vwr wr 0 0\nVcom com 0 0\nVst stn 0 {st}\n"
            f"Rpsw pswn 0 1e12\nRtau taun 0 1e12\nRsinf sinfn 0 1e12\n"
            f"Vnclk nclk 0 PULSE({VDD} 0 {tclk} {TR} {TR} {teval} 10n)\n")
    deck += SA_CORE
    deck += (f".control\n  tran 5p 4n\n"
             f"  wrdata _{tag}.csv v(outp) v(outn) v(vsen) i(vdd) i(vread)\n"
             f"  quit\n.endc\n.end\n")
    run_deck(deck, tag)
    t, (voutp, voutn, vsen, ivdd, ivread) = load_wrdata(HERE / f"_{tag}.csv", 5)
    strobe = (t >= tclk - 0.1e-9) & (t <= tclk + teval + 0.3e-9)
    e_sa = float(trapz(VDD * (-ivdd[strobe]), t[strobe]))
    e_rail = float(trapz(VREAD * (-ivread[strobe]), t[strobe]))
    isamp = np.searchsorted(t, tclk + teval - 0.1e-9)          # resolved, inside eval
    op, on = float(voutp[isamp]), float(voutn[isamp])
    want_outp_high = st == 1                                    # vsen > vref -> outp
    resolved = abs(op - on) > 0.9 * VDD
    correct = resolved and ((op > on) == want_outp_high)
    return dict(st=st, vsen_V=round(float(vsen[0]), 5),
                e_sa_pJ=round(e_sa * 1e12, 5), e_readrail_pJ=round(e_rail * 1e12, 5),
                outp_V=round(op, 4), outn_V=round(on, 4),
                resolved=bool(resolved), correct=bool(correct))


# --------------------------------------------------------------------- (f)
def assemble(pulse_mid, pulse_top, op_idle, settles, reads, p_string):
    t_settle = max(s["t_settle_ns"] for s in settles) * 1e-9
    p_buf = op_idle["p_rail_mW"] * 1e-3                     # Vdd*I(vdd), DC op (idle)
    e_read = max(r["e_sa_pJ"] + r["e_readrail_pJ"] for r in reads) * 1e-12
    e_top = (pulse_top["e_dev_pJ"] + pulse_top["e_tg_pJ"]) * 1e-12
    e_mid = (pulse_mid["e_dev_pJ"] + pulse_mid["e_tg_pJ"]) * 1e-12
    t_pw = TW + 2 * TR                                      # one gated pulse window
    rows = []
    for k in range(1, 6):
        t_up = (k + 1) * (TW + TGAP) + t_settle + TREAD
        e_pulses = k * e_top + e_mid
        st_on = (p_buf + p_string) * t_up
        st_gate = (p_buf + p_string) * (k + 1) * t_pw
        e_true_on = (p_buf + p_string) * t_up + e_read      # rail burn contains pulses
        rows.append(dict(
            k=k, t_update_ns=round(t_up * 1e9, 3),
            e_pulses_pJ=round(e_pulses * 1e12, 4),
            e_read_pJ=round(e_read * 1e12, 4),
            statics_always_on_pJ=round(st_on * 1e12, 3),
            statics_gated_pJ=round(st_gate * 1e12, 3),
            e_update_always_on_pJ=round((e_pulses + e_read + st_on) * 1e12, 3),
            e_update_gated_pJ=round((e_pulses + e_read + st_gate) * 1e12, 3),
            e_update_supply_true_pJ=round(e_true_on * 1e12, 3)))
    return rows, t_settle, p_buf, e_read


def main():
    smoke = "--smoke" in sys.argv
    vlo, vhi = dac_rails()

    print("[0] naive prescribed gating (feedback opens between pulses) — for the record")
    naive = run_pulse(CODE_MID, "energy_naive_mid", steer=False)
    print(f"    E_dev={naive['e_dev_pJ']:.3f} pJ  flat v_wr="
          f"[{naive['vwr_flat_min_V']:.3f},{naive['vwr_flat_max_V']:.3f}] V  -> BROKEN")

    print("[a,b] steered write pulses")
    pulse_mid = run_pulse(CODE_MID, "energy_pulse_mid")
    pulse_top = run_pulse(CODE_TOP, "energy_pulse_top")
    for nm, p in (("mid", pulse_mid), ("top", pulse_top)):
        print(f"    {nm}: E_dev={p['e_dev_pJ']:.4f} pJ  E_tg={p['e_tg_pJ']:.4f} pJ  "
              f"E_drvstage={p['e_drvstage_pJ']:.4f} pJ  "
              f"flat v_wr={p['vwr_flat_mean_V']:.4f} V (u={p['u_flat_mean']:+.2f})  "
              f"i_vdd idle/pulse={p['i_vdd_idle_mA']:.3f}/{p['i_vdd_pulse_mA']:.3f} mA")
    sanity = pulse_mid["vwr_flat_mean_V"] ** 2 / RSOT * TW * 1e12
    print(f"    sanity: flat_mean^2/Rsot*TW = {sanity:.4f} pJ vs E_dev_mid = "
          f"{pulse_mid['e_dev_pJ']:.4f} pJ (edges + wander account for the rest)")
    if smoke:
        return

    print("[c] buffer static (DC op, mid code)")
    op_idle = run_op("energy_op_idle", wen_on=False)
    op_act = run_op("energy_op_active", wen_on=True)
    print(f"    idle:   P_rail={op_idle['p_rail_mW']:.4f} mW  v_wrr={op_idle['v_wrr']:.4f} V")
    print(f"    active: P_rail={op_act['p_rail_mW']:.4f} mW  v_wr={op_act['v_wr']:.4f} V "
          f"(W2 DC cross-check)")

    print("[d] DAC string static (ANALYTIC) + full-scale settle")
    p_string = (vhi - vlo) ** 2 / (2 ** NB * RUNIT)
    print(f"    P_string=(Vhi-Vlo)^2/(2^{NB}*{RUNIT:.0f}) = {p_string*1e6:.3f} uW")
    settles = [run_settle("energy_settle_up", True), run_settle("energy_settle_dn", False)]
    for s in settles:
        print(f"    {s['direction']:4s}: t_settle={s['t_settle_ns']:.2f} ns  "
              f"E_supply(window)={s['e_supply_window_pJ']:.2f} pJ  "
              f"P_rail pre/post={s['p_rail_pre_mW']:.3f}/{s['p_rail_post_mW']:.3f} mW")

    print("[e] read strobe (PMOS-input StrongARM on the device read divider)")
    reads = [run_read(0, "energy_read_st0"), run_read(1, "energy_read_st1")]
    for r in reads:
        print(f"    st={r['st']}: vsen={r['vsen_V']:.4f} V  E_sa={r['e_sa_pJ']:.4f} pJ  "
              f"E_rail={r['e_readrail_pJ']:.4f} pJ  outp/outn={r['outp_V']:.2f}/{r['outn_V']:.2f} V"
              f"  correct={r['correct']}")

    print("[f] per-update table")
    rows, t_settle, p_buf, e_read = assemble(pulse_mid, pulse_top, op_idle, settles,
                                             reads, p_string)
    hdr = (" k | t_upd(ns) | E_pulses | E_read | statics_on | statics_gate |"
           " E_upd_on | E_upd_gate | E_upd_supply")
    print(hdr)
    for r in rows:
        print(f" {r['k']} | {r['t_update_ns']:9.2f} | {r['e_pulses_pJ']:8.3f} |"
              f" {r['e_read_pJ']:6.4f} | {r['statics_always_on_pJ']:10.3f} |"
              f" {r['statics_gated_pJ']:12.3f} | {r['e_update_always_on_pJ']:8.3f} |"
              f" {r['e_update_gated_pJ']:10.3f} | {r['e_update_supply_true_pJ']:12.3f}")

    summary = dict(
        _label=("W3 MEASURED (ngspice tran/op, sky130 tt, schematic-level) energy+timing of "
                "one Gibbs update on the W2 write chain; DAC-string static is ANALYTIC"),
        rng=dict(seeds=[], note="no stochastic element in W3; all transients deterministic"),
        design_correction=dict(
            naive_prescribed_gating=naive,
            note=("W2 feedback is taken from wr (after the enable TG); pulsing wen/wep opens "
                  "the loop, buffer rails, device sees ~1.6 V all pulse. Fixed by "
                  "current-steering (complementary replica branch keeps buffer load constant; "
                  "feedback steered wr<->wrr). Always-on replica (no steering) was tried and "
                  "rejected: 30..90 mV droop. See .agents/TRIAL_LOG_eda.md 2026-07-13.")),
        write_pulse=dict(
            label="MEASURED", tw_ns=TW * 1e9, tr_ns=TR * 1e9,
            mid=pulse_mid, top=pulse_top,
            sanity_analytic_flatmean_pJ=round(sanity, 4)),
        buffer_static=dict(
            label="MEASURED", idle_op=op_idle, active_op=op_act,
            p_buf_W=round(p_buf, 8),
            note=("P_buf = Vdd*I(vdd) at the idle DC op (replica conducting). Steering makes "
                  "the rail current the same during pulses (device burn replaces replica "
                  "burn), so P_buf*t already CONTAINS the pulse Ohmic energy; the prescribed "
                  "always-on sum k*(E_dev+E_tg)+P_buf*t_update double-counts it by "
                  "<= e_pulses/e_update; e_update_supply_true is the exact rail accounting.")),
        power_gating=dict(
            note=("Power gating the buffer+replica between updates is standard; the sibling "
                  "repo reproduced 24-36% primitive-level savings in "
                  "design_survey/repro/picoram_gating.py (referenced, not re-run here)."),
            gated_accounting="P_buf applied over (k+1) pulse windows of TW+2*TR only, as prescribed",
            gated_caveat=("optimistic: the loop must be awake t_settle before the first pulse; "
                          "within one update the chain is busy for the whole t_update, so the "
                          "realistic saving is the idle time BETWEEN updates")),
        dac_static=dict(
            label="ANALYTIC", p_string_W=round(p_string, 10),
            formula=f"(Vhi-Vlo)^2/(2^{NB}*R_unit) with rails [{vlo:.6f},{vhi:.6f}] V, "
                    f"R_unit={RUNIT:.0f} ohm"),
        settle=dict(label="MEASURED", tol_V=round(SETTLE_TOL, 6), runs=settles,
                    t_settle_ns=round(t_settle * 1e9, 3),
                    note="worst of up/down full-scale step at bin, settle band 0.1*VT on wrr"),
        read=dict(
            label="MEASURED", rail_V=VREAD, rref_ohm=RREF,
            rationale=(f"0.2 V read rail keeps the divider current ~{VREAD/(RREF+RP)*1e6:.0f} uA "
                       f"and vsen ~0.08..0.11 V << Vth={VTH:.3f} V, so read disturb on the "
                       "write sigmoid is negligible; Rref=Rp*(1+TMR/2) is the midpoint "
                       "reference"),
            sa=("PMOS-input StrongARM, mirror of the proven sibling NMOS SA "
                "(smtj_pbnn_sim/eda/hero/strongarm_sa.spice); NMOS input does not work at "
                "the ~0.1 V divider common mode"),
            states=reads,
            e_read_pJ=round(e_read * 1e12, 5)),
        timing_model=dict(
            formula="t_update = (k+1)*(TW+t_gap) + t_settle + t_read",
            tw_ns=TW * 1e9, t_gap_ns=TGAP * 1e9, t_read_ns=TREAD * 1e9,
            t_settle_ns=round(t_settle * 1e9, 3),
            pulse_window_ns=round((TW + 2 * TR) * 1e9, 3),
            note=("k reset pulses at TOP code + 1 probabilistic pulse at the update code + "
                  "1 read strobe; t_settle charged once per update (DAC recode)")),
        update_energy_model=dict(
            formula=("e_update(k) = k*(E_dev_top+E_tg_top) + (E_dev_mid+E_tg_mid) + E_read "
                     "+ statics; statics_always_on=(P_buf+P_string)*t_update, "
                     "statics_gated=(P_buf+P_string)*(k+1)*(TW+2*TR); "
                     "e_update_supply_true=(P_buf+P_string)*t_update+E_read (exact under "
                     "steering, contains the pulse Ohmic energy)"),
            e_read_used="max over st in {0,1} of E_sa+E_readrail"),
        table=rows)
    write_summary(HERE / "update_energy_summary.json", summary)


if __name__ == "__main__":
    main()
