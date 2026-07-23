#!/usr/bin/env python3
"""RX-14 — bipolar (reverse-polarity) RESET driver: replace update_energy.py's
same-amplitude POSITIVE Ohmic proxy for the AP->P reset pulse with a measured
sky130 H-bridge, then re-assemble e_update(k) and re-run the projection.

The chapter costs the reset (AP->P) with a positive pulse into the SOT branch
through the write-enable TG (update_energy.py: E_dev_top + E_tg_top per reset).
A real reset needs REVERSE polarity across the SOT branch, which the committed
unipolar write chain cannot produce.  Here the write-enable TG + hard-grounded
`com` are replaced by a full H-bridge (two CMOS-inverter half-bridges) around
the OSDI SOT branch (wr/com): the RESET drives com HIGH and wr LOW, the device
sees -V_reset, and the current path is  rail -HS_R-> com -SOT-> wr -LS_L-> gnd
(TWO series switches vs the proxy's ONE enable TG).  Reset is always at the
TOP-code magnitude (~1.0 V), so only the right leg toggles per reset pulse; the
left leg holds wr low (it is the reset return path, its idle state too).

Because the OSDI SOT branch is purely Ohmic (I(wr,com)=V(wr,com)/Rsot), the
device dissipation is polarity-symmetric: at matched delivered magnitude E_dev
equals the proxy exactly, and the whole cost difference is the H-bridge overhead
--- the second series switch, right-leg shoot-through (crowbar) and gate drive.

TWO drive options are measured:
  [0] BUFFER-DRIVEN (naive reuse of the update_energy precision write buffer +
      current-steering for the reverse path): the delivered level DRIFTS across
      the 0.75 ns plateau (the write loop is mistuned for the 2-switch reverse
      path -- f=8 undershoots, f=16 overshoots).  Kept as the documented
      failure (see .agents/TRIAL_LOG_eda.md).
  [a] FIXED-RAIL (primary): the reset needs saturation, not the analog sigmoid
      precision of the write, so the H-bridge is driven from a fixed reverse
      rail auto-tuned to deliver |V_sot| = the proxy top-code magnitude.  Clean
      settled level; E_dev/E_switch/E_shoot cleanly separable.  The write buffer
      static P_buf is unchanged (shared, on for the write), so the accounting
      only swaps E_tg_top -> E_switch + E_shoot.

HONEST: schematic-level, sky130 tt, first-cut sizing.  The V_T-normalized ratios
of the chapter are unaffected by the absolute reset energy; expected outcome is
e_update UP and the absolute-energy story weakening further.

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/hbridge_reset.py [--fingers N] [--smoke]
Writes hbridge_reset_summary.json next to this script (update_energy.py JSON
shape: timing_model / buffer_static / dac_static / table, so reproject_hw.py
--energy-json consumes it directly).
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

from _common import (HERE, SKY130_LIB, VTH, VT, RSOT, load_wrdata, run_deck,
                     trapz)
from update_energy import (BUFFER_FB, VDD, TR, T0, TW, TGAP, TREAD, CODE_TOP,
                           chain_prefix, dac_rails)

CORNER = os.environ.get("SKY130_CORNER", "tt")
FINGERS = int(os.environ.get("HB_FINGERS", "8"))     # per H-bridge switch (=enable-TG)
V_TARGET = 0.9985                                     # proxy top-code delivered |V_sot|
UE_JSON = HERE / ("update_energy_summary.json" if CORNER == "tt"
                  else f"update_energy_summary_{CORNER}.json")


def _fingers(prefix, drain, gate, source, body, dev, w, n):
    return "".join(
        f"X{prefix}{i} {drain} {gate} {source} {body} {dev} W={w} L=0.15\n"
        for i in range(1, n + 1))


DEVICE = """.model smtj_sot smtj_sot
N1 wr rd com st pswn taun sinfn smtj_sot
Vrd  rd 0 0
Vst  st 0 1
Rpsw  pswn 0 1e12
Rtau  taun 0 1e12
Rsinf sinfn 0 1e12
"""


# =====================================================================  [a]
def railed_hbridge(rail, n):
    """H-bridge from a FIXED reverse rail `rail`.  Right leg (com) = inverter on
    wep; left leg (wr) held low (selh=VDD): LS_L on = reset return, HS_L off."""
    s = f"Vselh selh 0 {VDD}\n"
    s += _fingers("HSR", rail, "wep", "com", rail, "sky130_fd_pr__pfet_01v8", 8, n)
    s += _fingers("LSR", "com", "wep", "0",   "0",   "sky130_fd_pr__nfet_01v8", 4, n)
    s += _fingers("HSL", rail, "selh", "wr", rail, "sky130_fd_pr__pfet_01v8", 8, n)
    s += _fingers("LSL", "wr",  "selh", "0",  "0",   "sky130_fd_pr__nfet_01v8", 4, n)
    return s


def run_railed(vrail, tag, fingers, hold=False):
    """One 0.75 ns reverse reset from a fixed rail (hold=True: rail on the whole
    window, for the static-conduction control)."""
    deck = (f"* RX-14 fixed-rail H-bridge reset, Vrail={vrail:.4f}\n"
            f".lib {SKY130_LIB} {CORNER}\n"
            f"Vdd vdd 0 {VDD}\n"
            f"Vrail prail 0 {vrail:.6f}\n")
    if hold:
        deck += "Vwep wep 0 0\n"                       # reset held ON
    else:
        deck += f"Vwep wep 0 PULSE({VDD} 0 {T0} {TR} {TR} {TW} 100n)\n"
    deck += railed_hbridge("prail", fingers) + DEVICE
    deck += (f".control\n  tran 2p 4.5n\n"
             f"  wrdata _{tag}.csv v(com) v(wr) i(vrail) v(wep) i(vwep) i(vdd)\n"
             f"  quit\n.endc\n.end\n")
    run_deck(deck, tag)
    t, (vcom, vwr, irail, vwep, iwep, ivdd) = load_wrdata(HERE / f"_{tag}.csv", 6)
    vsot = vcom - vwr
    idev = vsot / RSOT
    flat = (t >= T0 + TR + 0.1e-9) & (t <= T0 + TW)
    e_dev = float(trapz(vsot ** 2 / RSOT, t))
    e_hs = float(trapz((vrail - vcom) * idev, t))                # rail -> com (HS_R)
    e_ls = float(trapz(vwr * idev, t))                           # wr -> gnd (LS_L)
    e_rail = float(trapz(vrail * (-irail), t))                   # supply from the rail
    # gate-drive: energy the wep driver SOURCES (charge-up edge) = ~C_g*VDD^2
    e_gate = float(trapz(np.clip(vwep * (-iwep), 0.0, None), t))
    vf = vsot[flat]
    vfin = float(vf.mean())
    band = 0.1 * VT
    pin = (t >= T0) & (t <= T0 + TW)
    off = (np.abs(vsot - vfin) > band) & pin
    t_set = float(t[off][-1] - T0) if off.any() else 0.0
    idle = (t >= 1.0e-9) & (t <= 1.9e-9)      # bridge idle (HS off): rail leakage
    return dict(
        vrail=round(vrail, 5), fingers=fingers,
        e_dev_pJ=round(e_dev * 1e12, 4),
        e_hs_pJ=round(e_hs * 1e12, 4), e_ls_pJ=round(e_ls * 1e12, 4),
        e_switch_pJ=round((e_hs + e_ls) * 1e12, 4),
        e_rail_pJ=round(e_rail * 1e12, 4),
        e_gate_pJ=round(e_gate * 1e12, 5),
        e_shoot_pJ=round((e_rail - e_dev - e_hs - e_ls) * 1e12, 4),
        vsot_flat_mean_V=round(vfin, 5),
        vsot_flat_min_V=round(float(vf.min()), 5),
        vsot_flat_max_V=round(float(vf.max()), 5),
        vcom_flat_mean_V=round(float(vcom[flat].mean()), 5),
        vwr_flat_mean_V=round(float(vwr[flat].mean()), 5),
        u_flat_mean=round((vfin - VTH) / VT, 3),
        reset_settle_ns=round(t_set * 1e9, 4),
        i_rail_flat_mA=round(float(-irail[flat].mean()) * 1e3, 4),
        i_rail_idle_uA=round(float(-irail[idle].mean()) * 1e6, 4),
        p_bridge_idle_uW=round(float(vrail * -irail[idle].mean()) * 1e6, 4))


def tune_rail(fingers, target=V_TARGET, tol=0.0015, itmax=6):
    """Secant iteration on Vrail so the settled |V_sot| == the proxy magnitude
    (V_sot is sub-linear in Vrail: the HS_R PMOS source-degenerates as the rail
    drops, so a single linear rescale overshoots)."""
    r0, p0 = 1.20, None
    p0 = run_railed(r0, "hb_tune", fingers)
    v0 = p0["vsot_flat_mean_V"]
    r1 = r0 * target / v0                    # first linear guess
    for _ in range(itmax):
        p1 = run_railed(r1, "hb_tune", fingers)
        v1 = p1["vsot_flat_mean_V"]
        if abs(v1 - target) <= tol:
            return p1
        slope = (v1 - v0) / (r1 - r0) if r1 != r0 else 1.0
        r0, v0 = r1, v1
        r1 = r1 + (target - v1) / (slope if abs(slope) > 1e-6 else 1.0)
        r1 = min(max(r1, target), VDD)       # keep the rail in a sane range
    return p1


# =====================================================================  [0]
# naive reuse of the update_energy precision write buffer + steering for the
# reverse path (documents the plateau drift); reset drives com, wr held low.
def buffered_hbridge(n):
    s = f"Vselh selh 0 {VDD}\n"
    s += _fingers("HSR", "drv", "wep", "com", "vdd", "sky130_fd_pr__pfet_01v8", 8, n)
    s += _fingers("LSR", "com", "wep", "0",   "0",   "sky130_fd_pr__nfet_01v8", 4, n)
    s += _fingers("HSL", "drv", "selh", "wr", "vdd", "sky130_fd_pr__pfet_01v8", 8, n)
    s += _fingers("LSL", "wr",  "selh", "0",  "0",   "sky130_fd_pr__nfet_01v8", 4, n)
    # steering replica + feedback mux (com<->wrr), update_energy sizes
    s += _fingers("RWn", "drv", "wep", "wrr", "0",   "sky130_fd_pr__nfet_01v8", 4, 8)
    s += _fingers("RWp", "drv", "wen", "wrr", "vdd", "sky130_fd_pr__pfet_01v8", 8, 8)
    s += f"Rrep wrr 0 {RSOT:.0f}\n"
    s += ("XFB1n com wen fbn 0   sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
          "XFB1p com wep fbn vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n"
          "XFB2n wrr wep fbn 0   sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
          "XFB2p wrr wen fbn vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n")
    return s


def run_buffered(tag, fingers):
    deck = chain_prefix(CODE_TOP) + BUFFER_FB.format(inp="bin", fb="fbn")
    deck += buffered_hbridge(fingers)
    deck += (f"Vwen wen 0 PULSE(0 {VDD} {T0} {TR} {TR} {TW} 100n)\n"
             f"Vwep wep 0 PULSE({VDD} 0 {T0} {TR} {TR} {TW} 100n)\n")
    deck += DEVICE
    deck += (f".control\n  tran 2p 4.5n\n"
             f"  wrdata _{tag}.csv v(com) v(wr) v(drv) i(vdd)\n  quit\n.endc\n.end\n")
    run_deck(deck, tag)
    t, (vcom, vwr, vdrv, ivdd) = load_wrdata(HERE / f"_{tag}.csv", 4)
    vsot = vcom - vwr
    flat = (t >= T0 + TR + 0.1e-9) & (t <= T0 + TW)
    vf = vsot[flat]
    # plateau drift = spread across the flat top
    i0 = np.searchsorted(t, T0 + TR + 0.1e-9)
    i1 = np.searchsorted(t, T0 + TW)
    return dict(fingers=fingers,
                vsot_start_V=round(float(vsot[i0]), 5),
                vsot_end_V=round(float(vsot[i1 - 1]), 5),
                vsot_flat_mean_V=round(float(vf.mean()), 5),
                plateau_drift_mV=round(float(vsot[i1 - 1] - vsot[i0]) * 1e3, 2))


# =====================================================================
def reassemble(reset_pp_pJ, t_settle_ns, p_buf_W, p_string_W, e_read_pJ,
               e_write_pulse_pJ, e_rail_reset_pJ=0.0):
    """Re-build e_update(k) with the reset per-pulse energy replaced by the
    measured bipolar value.  The k reset pulses are supplied by a SEPARATE
    reverse rail (not the write buffer), so unlike update_energy's supply-true
    accounting -- where the positive-proxy reset was contained inside P_buf via
    steering -- the reverse-rail energy k*E_rail is ADDED to supply-true."""
    t_settle = t_settle_ns * 1e-9
    t_pw = TW + 2 * TR
    rows = []
    for k in range(1, 6):
        t_up = (k + 1) * (TW + TGAP) + t_settle + TREAD
        e_pulses = (k * reset_pp_pJ + e_write_pulse_pJ) * 1e-12
        st_on = (p_buf_W + p_string_W) * t_up
        st_gate = (p_buf_W + p_string_W) * (k + 1) * t_pw
        e_read = e_read_pJ * 1e-12
        e_true = (p_buf_W + p_string_W) * t_up + e_read + k * e_rail_reset_pJ * 1e-12
        rows.append(dict(
            k=k, t_update_ns=round(t_up * 1e9, 3),
            e_pulses_pJ=round(e_pulses * 1e12, 4),
            e_read_pJ=round(e_read * 1e12, 4),
            statics_always_on_pJ=round(st_on * 1e12, 3),
            statics_gated_pJ=round(st_gate * 1e12, 3),
            e_update_always_on_pJ=round((e_pulses + e_read + st_on) * 1e12, 3),
            e_update_gated_pJ=round((e_pulses + e_read + st_gate) * 1e12, 3),
            e_update_supply_true_pJ=round(e_true * 1e12, 3)))
    return rows


def main():
    smoke = "--smoke" in sys.argv
    fingers = FINGERS
    if "--fingers" in sys.argv:
        fingers = int(sys.argv[sys.argv.index("--fingers") + 1])

    ue = json.loads(UE_JSON.read_text())
    top = ue["write_pulse"]["top"]
    mid = ue["write_pulse"]["mid"]
    proxy_reset_pp = top["e_dev_pJ"] + top["e_tg_pJ"]          # positive proxy
    e_write_pulse = mid["e_dev_pJ"] + mid["e_tg_pJ"]           # unipolar write (kept)
    p_buf_W = ue["buffer_static"]["p_buf_W"]
    p_string_W = ue["dac_static"]["p_string_W"]
    t_settle_ns = ue["timing_model"]["t_settle_ns"]
    e_read_pJ = ue["read"]["e_read_pJ"]

    print(f"[cfg] fingers/switch={fingers}  corner={CORNER}  V_target={V_TARGET} V")
    print(f"[proxy] positive reset per pulse = E_dev_top+E_tg_top = "
          f"{top['e_dev_pJ']:.4f}+{top['e_tg_pJ']:.4f} = {proxy_reset_pp:.4f} pJ "
          f"(delivered v_wr={top['vwr_flat_mean_V']:.4f} V)")

    if smoke:
        for f in (8, 16, 32):
            p = tune_rail(f)
            print(f"  f={f:2d}: Vrail={p['vrail']:.4f} |V_sot|={p['vsot_flat_mean_V']:.4f}"
                  f" E_dev={p['e_dev_pJ']:.4f} E_sw={p['e_switch_pJ']:.4f}"
                  f" E_shoot={p['e_shoot_pJ']:.4f} E_gate={p['e_gate_pJ']:.4f}"
                  f" settle={p['reset_settle_ns']:.3f}ns")
        return

    print("[0] naive reuse of the precision write buffer for the reverse path")
    for f in (8, 16):
        b = run_buffered(f"hb_buf_f{f}", f)
        print(f"    f={f:2d}: |V_sot| start={b['vsot_start_V']:.4f} -> "
              f"end={b['vsot_end_V']:.4f} V (mean {b['vsot_flat_mean_V']:.4f}); "
              f"plateau drift {b['plateau_drift_mV']:+.1f} mV -> unsettled")

    print("[a] fixed-rail H-bridge reset (primary), auto-tuned to V_target")
    rp = tune_rail(fingers)
    print(f"    Vrail={rp['vrail']:.4f} V -> |V_sot|={rp['vsot_flat_mean_V']:.4f} V "
          f"(u={rp['u_flat_mean']:+.2f}); settle={rp['reset_settle_ns']:.3f} ns "
          f"(spread {rp['vsot_flat_min_V']:.4f}..{rp['vsot_flat_max_V']:.4f})")
    print(f"    E_dev={rp['e_dev_pJ']:.4f}  E_hs={rp['e_hs_pJ']:.4f}  "
          f"E_ls={rp['e_ls_pJ']:.4f}  E_switch={rp['e_switch_pJ']:.4f}  "
          f"E_shoot={rp['e_shoot_pJ']:.4f}  E_gate={rp['e_gate_pJ']:.5f}  "
          f"E_rail={rp['e_rail_pJ']:.4f}")
    print(f"    bridge idle static (HS off, no replica): {rp['p_bridge_idle_uW']:.3f} uW "
          f"(rail leakage {rp['i_rail_idle_uA']:.3f} uA) vs shared write P_buf 2868 uW")

    print("[b] size variant (low-R switches, f=16)")
    rp16 = tune_rail(16)
    print(f"    Vrail={rp16['vrail']:.4f} |V_sot|={rp16['vsot_flat_mean_V']:.4f} "
          f"E_dev={rp16['e_dev_pJ']:.4f} E_switch={rp16['e_switch_pJ']:.4f} "
          f"E_shoot={rp16['e_shoot_pJ']:.4f} E_gate={rp16['e_gate_pJ']:.5f}")

    # bipolar reset per pulse (gated basis): device Ohmic + two series switches +
    # right-leg shoot-through + gate drive, at the matched proxy magnitude.  This
    # replaces the proxy's E_dev_top + E_tg_top (device + one enable TG).
    reset_pp = rp["e_dev_pJ"] + rp["e_switch_pJ"] + max(rp["e_shoot_pJ"], 0.0) + rp["e_gate_pJ"]
    print(f"[c] bipolar reset per pulse (gated) = E_dev+E_switch+E_shoot+E_gate = "
          f"{rp['e_dev_pJ']:.4f}+{rp['e_switch_pJ']:.4f}+{max(rp['e_shoot_pJ'],0):.4f}"
          f"+{rp['e_gate_pJ']:.4f} = {reset_pp:.4f} pJ")
    print(f"    vs proxy {proxy_reset_pp:.4f} pJ  ->  x{reset_pp/proxy_reset_pp:.3f} "
          f"(delta {reset_pp-proxy_reset_pp:+.4f} pJ/reset)")

    rows = reassemble(reset_pp, t_settle_ns, p_buf_W, p_string_W, e_read_pJ,
                      e_write_pulse, e_rail_reset_pJ=rp["e_rail_pJ"])
    ue_rows = {r["k"]: r for r in ue["table"]}
    print("[d] re-assembled e_update(k) gated (proxy -> H-bridge):")
    print(" k | reset_pp | e_pulses |  e_upd_gated proxy -> hbridge         | supply_true")
    for r in rows:
        pr = ue_rows[r["k"]]
        print(f" {r['k']} | {reset_pp:8.4f} | {r['e_pulses_pJ']:8.4f} | "
              f"{pr['e_update_gated_pJ']:8.3f} -> {r['e_update_gated_pJ']:8.3f} "
              f"(x{r['e_update_gated_pJ']/pr['e_update_gated_pJ']:.3f}) | "
              f"{r['e_update_supply_true_pJ']:8.3f}")

    summary = dict(
        _label=("RX-14 MEASURED (ngspice tran, sky130 %s, schematic-level, first-cut "
                "sizing): bipolar H-bridge AP->P reset (fixed reverse rail) replacing "
                "the same-amplitude positive Ohmic proxy of update_energy.py. Write "
                "pulse / read / statics / timing carried over unchanged. Shape matches "
                "update_energy_summary.json so reproject_hw --energy-json consumes it."
                % CORNER),
        rng=dict(seeds=[], note="deterministic transients; no stochastic element"),
        provenance=dict(reused_update_energy=UE_JSON.name, fingers_per_switch=fingers,
                        v_target_V=V_TARGET,
                        note=("H-bridge = two CMOS-inverter half-bridges around the SOT "
                              "branch; reset drives com HIGH, wr LOW; only the right leg "
                              "toggles per reset. Driven from a fixed reverse rail (a "
                              "reset needs saturation, not the write's analog precision). "
                              "The device Ohmic branch is polarity-symmetric so E_dev "
                              "equals the proxy at matched magnitude; overhead = second "
                              "series switch + right-leg shoot-through + gate drive.")),
        buffered_drift=dict(
            label="DOCUMENTED_FAILURE",
            note=("Naive reuse of the update_energy precision write buffer + steering "
                  "for the reverse path leaves the delivered level DRIFTING across the "
                  "0.75 ns plateau (write loop mistuned for the 2-switch reverse path). "
                  "See .agents/TRIAL_LOG_eda.md. Reset therefore driven from a fixed "
                  "rail."),
            runs=[run_buffered("hb_buf_f8", 8), run_buffered("hb_buf_f16", 16)]),
        proxy=dict(reset_per_pulse_pJ=round(proxy_reset_pp, 4),
                   e_dev_top_pJ=top["e_dev_pJ"], e_tg_top_pJ=top["e_tg_pJ"],
                   e_drvstage_top_pJ=top["e_drvstage_pJ"],
                   delivered_vwr_V=top["vwr_flat_mean_V"]),
        hbridge_reset=dict(
            label="MEASURED", primary=rp, size_variant_f16=rp16,
            reset_per_pulse_gated_pJ=round(reset_pp, 4),
            reset_ratio_vs_proxy=round(reset_pp / proxy_reset_pp, 4),
            overhead_breakdown=dict(
                e_dev_pJ=rp["e_dev_pJ"], e_switch_pJ=rp["e_switch_pJ"],
                e_shoot_pJ=round(max(rp["e_shoot_pJ"], 0.0), 4),
                e_gate_pJ=rp["e_gate_pJ"],
                second_switch_vs_single_tg=("proxy single enable TG E_tg=%.4f pJ; "
                    "H-bridge two series switches E_switch=%.4f pJ"
                    % (top["e_tg_pJ"], rp["e_switch_pJ"]))),
            note=("E_dev matches proxy (Ohmic, polarity-symmetric). E_shoot is the "
                  "right-leg crowbar drawn from the rail during the wep edges; E_gate "
                  "is the switch gate-charge from VDD.")),
        # --- update_energy.py-shaped fields for reproject_hw --energy-json ---
        write_pulse=dict(label="CARRIED_FROM_UPDATE_ENERGY", top=top, mid=mid,
                         e_write_pulse_pJ=round(e_write_pulse, 4),
                         note="reset only; write pulse kept unipolar per RX-14 spec"),
        buffer_static=dict(label="CARRIED", p_buf_W=p_buf_W,
                           note="write buffer static unchanged (shared, on for the write)"),
        dac_static=dict(label="CARRIED", p_string_W=p_string_W,
                        formula=ue["dac_static"]["formula"]),
        read=dict(label="CARRIED", e_read_pJ=e_read_pJ),
        timing_model=dict(
            formula="t_update = (k+1)*(TW+t_gap) + t_settle + t_read (unchanged)",
            tw_ns=TW * 1e9, t_gap_ns=TGAP * 1e9, t_read_ns=TREAD * 1e9,
            t_settle_ns=t_settle_ns, pulse_window_ns=round((TW + 2 * TR) * 1e9, 3),
            note=("fixed-rail reset settles within the 0.75 ns pulse "
                  "(reset_settle_ns=%.3f), so no extra t_update; DAC-recode settle "
                  "unchanged" % rp["reset_settle_ns"])),
        update_energy_model=dict(
            formula=("e_update(k) = k*(bipolar reset per pulse) + (write pulse) + E_read "
                     "+ statics; reset per pulse = E_dev+E_switch+E_shoot+E_gate (gated); "
                     "supply-true = (P_buf+P_string)*t_update + E_read"),
            reset_per_pulse_proxy_pJ=round(proxy_reset_pp, 4),
            reset_per_pulse_hbridge_pJ=round(reset_pp, 4),
            e_read_used="carried from update_energy read.e_read_pJ"),
        table=rows)
    out = HERE / ("hbridge_reset_summary.json" if CORNER == "tt"
                  else f"hbridge_reset_summary_{CORNER}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
