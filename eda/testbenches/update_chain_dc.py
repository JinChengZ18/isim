#!/usr/bin/env python3
"""W2 — Ising p-bit write chain, DC transfer (sky130 tt, schematic-level).

One Gibbs update wants  P(s_i=+1) = sigma(2*beta*h_i_eff) = sigma(u).  On the
pulse-programmed sMTJ this is realised by delivering a write voltage
V = Vth + u*VT to the SOT branch (P->AP direction, the clean calibrated
sigmoid), so the write DAC must cover a window centered on Vth = 895.8 mV and
resolve fractions of the probability window VT = 23.4 mV.

Chain simulated here (every stage a real sky130 device, DC operating point):

  resistor-string DAC (2^b unit R between rails Vth-u_span*VT .. Vth+u_span*VT)
    -> CMOS transmission gate (code select)
    -> two-stage Miller unity buffer (NMOS input pair: the ~0.9 V common mode
       sits too high for the PMOS-input buffer used at 0..0.2 V in the sibling
       PBNN flow, so the topology is mirrored)
    -> write-enable transmission gate
    -> SOT write branch of the committed OSDI device (776 ohm), whose
       *own* observable node psw reports the switching probability.

MEASURED (ngspice DC .op, sky130 tt, schematic-level): the delivered V(wr) and
device-reported P_sw at every code, for nbits in {4,5,6,8}; LSB (mV and VT
units), span in u units, INL, monotonicity, buffer offset.
The measured per-code u-grid feeds eda/interface/dac_quantized_backend.py.

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/update_chain_dc.py [--smoke]
Writes update_chain_summary.json next to this script.
"""
from __future__ import annotations

import sys

import numpy as np

from _common import (HERE, SKY130_LIB, VTH, VT, RSOT, grab, psw, run_deck,
                     write_summary)

USPAN = 4.0          # DAC rails at Vth +/- USPAN*VT  (probability clip window)
RUNIT = 100.0        # resistor-string unit R [ohm]
NBITS_LIST = [4, 5, 6, 8]

BUFFER = """* two-stage Miller unity buffer, NMOS input (Vcm ~ 0.9 V), drives write line
* (sky130 fd_pr devices come in fixed W bins -> parallel instances, not W=160)
* feedback is taken from node wr (after the write-enable TG), so the loop
* corrects the enable-gate drop at the ~1.2 mA write current.
XMt1 btail bnb 0 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XMt2 btail bnb 0 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
Vnb  bnb 0 0.9
XM1  bd1 {inp} btail 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XM2  bd2 wr    btail 0 sky130_fd_pr__nfet_01v8 W=20 L=0.5
XM3  bd1 bd2 vdd vdd sky130_fd_pr__pfet_01v8 W=20 L=0.5
XM4  bd2 bd2 vdd vdd sky130_fd_pr__pfet_01v8 W=20 L=0.5
XMo1 drv bd1 vdd vdd sky130_fd_pr__pfet_01v8 W=80 L=0.5
XMo2 drv bd1 vdd vdd sky130_fd_pr__pfet_01v8 W=80 L=0.5
Ccm  bd1 drv 2p
"""

TAIL = """* write-enable transmission gate (8 parallel fingers) + committed device (OSDI)
Vwen wen 0 1.8
Vwep wep 0 0
XWEn1 drv wen wr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15
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
.control
  op
  print v(wr)
  print v(pswn)
  print v(drv)
  print v(bin)
  quit
.endc
.end
"""


def _tg(node_in, node_out, idx):
    return (f"Vtgn{idx} tgn{idx} 0 1.8\n"
            f"Vtgp{idx} tgp{idx} 0 0\n"
            f"XTGn{idx} {node_in} tgn{idx} {node_out} 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
            f"XTGp{idx} {node_in} tgp{idx} {node_out} vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n")


def deck(code, nbits, uspan=USPAN):
    vhi = VTH + uspan * VT
    vlo = VTH - uspan * VT
    ntap = 2 ** nbits
    s = (f"* Ising p-bit write chain, code={code}/{ntap - 1} nbits={nbits}\n"
         f".lib {SKY130_LIB} tt\n"
         f"Vdd vdd 0 1.8\n"
         f"Vhi vhi 0 {vhi:.6f}\n"
         f"Vlo vlo 0 {vlo:.6f}\n")
    nodes = ["vhi"] + [f"rs{i}" for i in range(1, ntap)] + ["vlo"]
    for i in range(ntap):
        s += f"Rs{i} {nodes[i]} {nodes[i + 1]} {RUNIT}\n"
    tap = nodes[ntap - 1 - code]          # code 0 -> one R above vlo; max -> vhi
    s += _tg(tap, "bin", 0)
    s += BUFFER.format(inp="bin")
    s += TAIL
    return s


def run_code(code, nbits):
    out = run_deck(deck(code, nbits), f"chain_b{nbits}")
    return dict(code=code,
                v_wr=grab(out, "wr"),
                psw_dev=grab(out, "pswn"),
                v_drv=grab(out, "drv"),
                v_tap=grab(out, "bin"))


def metrics(rows, nbits):
    v = np.array([r["v_wr"] for r in rows])
    tap = np.array([r["v_tap"] for r in rows])
    dv = np.diff(v)
    lsb = float(np.mean(dv))
    ideal = np.linspace(v[0], v[-1], len(v))
    inl = float(np.max(np.abs(v - ideal)) / abs(lsb)) if lsb else float("nan")
    u = (v - VTH) / VT
    return dict(
        nbits=nbits,
        lsb_mV=round(lsb * 1e3, 4), lsb_over_VT=round(lsb / VT, 4),
        range_mV=round(float(v[-1] - v[0]) * 1e3, 3),
        monotonic=bool(np.all(dv > 0)), inl_lsb=round(inl, 4),
        u_min=round(float(u[0]), 4), u_max=round(float(u[-1]), 4),
        buffer_offset_mV=round(float(np.mean(v - tap)) * 1e3, 4),
        psw_span=[round(float(psw(v[0])), 5), round(float(psw(v[-1])), 5)],
        transfer=[dict(code=r["code"], v_wr=round(r["v_wr"], 7),
                       u=round((r["v_wr"] - VTH) / VT, 5),
                       psw_dev=round(r["psw_dev"], 6)) for r in rows])


def main():
    smoke = "--smoke" in sys.argv
    print(f"chain rails: Vth +/- {USPAN}*VT = "
          f"[{(VTH - USPAN * VT) * 1e3:.1f}, {(VTH + USPAN * VT) * 1e3:.1f}] mV; "
          f"load = SOT branch {RSOT:.0f} ohm")
    if smoke:
        nb = 6
        for code in (0, 2 ** (nb - 1), 2 ** nb - 1):
            r = run_code(code, nb)
            print(f"[smoke] b{nb} code={code:2d}: v_tap={r['v_tap']:.4f}  "
                  f"v_drv={r['v_drv']:.4f}  v_wr={r['v_wr']:.4f}  "
                  f"psw={r['psw_dev']:.4f}")
        return

    per_bits = {}
    for nb in NBITS_LIST:
        rows = [run_code(c, nb) for c in range(2 ** nb)]
        m = metrics(rows, nb)
        per_bits[str(nb)] = m
        print(f"[b{nb}] LSB={m['lsb_mV']:.3f} mV ({m['lsb_over_VT']:.3f} VT)  "
              f"range={m['range_mV']:.1f} mV  mono={m['monotonic']}  "
              f"INL={m['inl_lsb']:.3f} LSB  u=[{m['u_min']:.2f},{m['u_max']:.2f}]  "
              f"off={m['buffer_offset_mV']:.2f} mV")

    summary = dict(
        _label=("MEASURED, ngspice DC .op, sky130 tt, schematic-level; device = "
                "committed OSDI smtj_sot (psw read from its observable node)"),
        u_span_design=USPAN, vth_V=VTH, vt_V=VT, r_unit_ohm=RUNIT,
        rails_mV=[round((VTH - USPAN * VT) * 1e3, 3), round((VTH + USPAN * VT) * 1e3, 3)],
        chain=("resistor-string DAC -> TG -> NMOS-input two-stage Miller unity "
               "buffer -> write-enable TG -> SOT branch (776 ohm)"),
        per_bits=per_bits)
    write_summary(HERE / "update_chain_summary.json", summary)


if __name__ == "__main__":
    main()
