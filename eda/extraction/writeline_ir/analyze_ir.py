#!/usr/bin/env python3
"""W5: per-row write-line IR drop -> static drive offset on the Ising-tile probability window.

Ising context: the Table-3.8 hardware projection assumes N_par = 64 spins updated in parallel —
a 64x64 tile, row-sequential. Each COLUMN write line serves 64 rows at a 2 um cell pitch on
met2 (W = 1 um). The metal between the column driver and row r adds R_line(r) in series with
the 776-ohm SOT write branch; if the driver is calibrated to put Vth = 895.783 mV on the
NEAREST cell, row r receives Vth - dV(r) with (exact divider, I_write ~ 1.15 mA)

    dV(r) = Vth * R_line(r) / (Rsot + R_line(r)),   R_line(r) = 2 * Rs * r * pitch / W

(x2 = bit line out + source line return, same convention as the PBNN writeline flow whose
headline was N=16 1.0% / N=64 4.1% / N=256 16.5% of the 776-ohm device). On the calibrated
sigmoid this is a STATIC per-row drive offset u_off(r) = dV(r)/VT probability-window units
(VT = 23.414 mV): the row's update probability is sigma(u_drive - u_off(r)) instead of
sigma(u_drive). ir_solver_impact.py feeds the N=64 profile back into the Chapter-3 solver.

Predistortion view: the 6-bit update DAC (update_chain_dc.py) can pre-compensate each row by
adding code(r) = round(dV(r)/LSB) codes; what remains is the quantization residual
|dV(r) - code(r)*LSB| <= LSB/2, i.e. a residual offset <= LSB/(2*VT) in u units, at the cost
of code(r) codes of DAC headroom at the far rows.

Data provenance (INTEGRITY):
  * met2 sheet R  — MEASURED: Magic `extract do resistance` lumped net R from the .ext node
    records (cal strap, Rs = R*W/L over 400 squares), written by run_extresist.sh;
  * flow check    — MEASURED: Magic `extresist` two-port R of the 400-sq poly strap vs the
    sky130A.tech 48.2 ohm/sq (the -0.5% delta is the 0.5-um label inset: 398/400 squares);
  * per-row model — ANALYTIC: the committed divider formula above;
  * DAC LSB       — MEASURED from eda/testbenches/update_chain_summary.json
    (per_bits["6"].lsb_mV) when present, else FALLBACK 2.97 mV (ideal 8*VT/63), labeled.
No RNG anywhere in this script.

Run (Windows or WSL, pure Python; run gen_strap.py + run_extresist.sh first):
  python eda/extraction/writeline_ir/analyze_ir.py
Writes ir_drop_summary.json next to this script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TB = HERE.parent.parent / "testbenches"

# Chapter-2 calibration (mirrors eda/testbenches/_common.py; single source: models/smtj_sot.va)
VTH = 0.895783          # sigmoid center [V]
VT = 0.023414           # probability window = 1/beta_s [V]
RSOT = 776.0            # SOT write resistance [ohm]

PITCH_UM, W_UM, LAYER = 2.0, 1.0, "met2"
ROUNDTRIP = 2.0         # bit line + source line return (PBNN convention)
RS_TECHFILE = 0.125     # sky130A.tech met2 resist, ohm/sq (FALLBACK if extraction missing)
POLY_TECHFILE = 48.2    # sky130A.tech poly resist, ohm/sq (flow-validation reference)
LSB_FALLBACK_MV = 2.97  # ideal 6-bit LSB over Vth +/- 4*VT rails: 8*VT/63 = 2.973 mV
N_LIST = (16, 64, 256)


def read_met2_rs():
    """met2 ohm/sq from the Magic .ext lumped net R of the 400-sq cal strap (MEASURED)."""
    ext = HERE / "writeline_straps.ext"
    if not ext.exists():
        return RS_TECHFILE, "FALLBACK (techfile sky130A.tech met2 125 mohm/sq; .ext missing)"
    m = re.search(r'^node "cal_[ab]" (\d+(?:\.\d+)?) ', ext.read_text(), re.M)
    if not m:
        return RS_TECHFILE, "FALLBACK (techfile; cal node not found in .ext)"
    rs = float(m.group(1)) * 0.5 / 200.0                    # Rs = R * W / L
    return rs, "MEASURED (magic extract-do-resistance lumped R, cal strap 400 sq)"


def read_poly_validation():
    """extresist two-port R of the poly strap (MEASURED flow check vs techfile 48.2)."""
    sp = HERE / "wl_res.spice"
    if not sp.exists():
        return None
    rs = [(a, b, float(v)) for a, b, v in
          re.findall(r"^R\S*\s+(\S+)\s+(\S+)\s+([0-9.eE+-]+)", sp.read_text(), re.M)]
    tot = sum(v for a, b, v in rs if "polycal" in a or "polycal" in b)
    if not tot:
        return None
    rsq = tot * 0.5 / 200.0
    return dict(extracted_ohm_sq=rsq, techfile_ohm_sq=POLY_TECHFILE,
                delta_pct=(rsq / POLY_TECHFILE - 1.0) * 100.0,
                note="delta = 0.5-um label inset (398/400 sq); MEASURED (magic extresist)")


def read_lsb_mv():
    """6-bit DAC LSB from the W2 chain sweep; FALLBACK to the ideal value until it lands."""
    p = TB / "update_chain_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        pb6 = d.get("per_bits", {}).get("6", {})
        if "lsb_mV" in pb6:
            return float(pb6["lsb_mV"]), "MEASURED (update_chain_summary.json per_bits.6.lsb_mV)"
        tr = pb6.get("transfer")
        if tr and len(tr) > 2 and "v_wr" in tr[0]:
            lsb = float(np.median(np.diff([row["v_wr"] for row in tr]))) * 1e3
            return lsb, "MEASURED (median code-to-code v_wr step, update_chain_summary.json)"
    return LSB_FALLBACK_MV, "FALLBACK (ideal 8*VT/63 = 2.97 mV; update_chain_summary.json pending)"


def per_row_profile(n_rows, rs_ohm_sq, lsb_mv):
    """ANALYTIC per-row profile from the committed divider formula. Row r = 1..N."""
    r = np.arange(1, n_rows + 1)
    r_line = ROUNDTRIP * rs_ohm_sq * (r * PITCH_UM) / W_UM
    dv = VTH * r_line / (RSOT + r_line)                     # V short-fall at row r
    u_off = dv / VT
    i_ma = VTH / (RSOT + r_line) * 1e3                      # actual current, uncompensated
    code = np.rint(dv * 1e3 / lsb_mv).astype(int)           # predistortion codes
    resid_mv = dv * 1e3 - code * lsb_mv                     # signed residual, mV
    return dict(
        rows=[dict(r=int(rr), R_line_ohm=float(rl), dV_mV=float(d * 1e3),
                   u_off=float(u), comp_code=int(c), resid_mV=float(rm),
                   resid_u=float(rm / (VT * 1e3)))
              for rr, rl, d, u, c, rm in zip(r, r_line, dv, u_off, code, resid_mv)],
        far_row=dict(r=int(r[-1]), R_line_ohm=float(r_line[-1]), dV_mV=float(dv[-1] * 1e3),
                     u_off=float(u_off[-1]), I_write_mA=float(i_ma[-1]),
                     V_drv_needed_V=float(VTH * (RSOT + r_line[-1]) / RSOT)),
        predistortion=dict(lsb_mV=lsb_mv, max_comp_code=int(code.max()),
                           max_resid_mV=float(np.abs(resid_mv).max()),
                           max_resid_u=float(np.abs(resid_mv).max() / (VT * 1e3))),
    )


def main():
    rs, rs_src = read_met2_rs()
    poly = read_poly_validation()
    lsb, lsb_src = read_lsb_mv()

    print("=" * 96)
    print("W5: Ising-tile write-line IR drop -> per-row static drive offset (met2, "
          f"W={W_UM:.0f}um, pitch={PITCH_UM:.0f}um, BL+SL x{ROUNDTRIP:.0f})")
    print(f"  met2 Rs = {rs:.4f} ohm/sq   [{rs_src}]")
    if poly:
        print(f"  flow check: poly {poly['extracted_ohm_sq']:.2f} vs techfile "
              f"{poly['techfile_ohm_sq']} ohm/sq ({poly['delta_pct']:+.2f}%)")
    print(f"  6-bit DAC LSB = {lsb:.3f} mV   [{lsb_src}]")
    print("=" * 96)

    per_n = {}
    for n in N_LIST:
        prof = per_row_profile(n, rs, lsb)
        per_n[str(n)] = prof
        fr, pd = prof["far_row"], prof["predistortion"]
        print(f"N={n:>4}: far row R_line={fr['R_line_ohm']:6.1f} ohm  dV={fr['dV_mV']:7.2f} mV"
              f"  u_off={fr['u_off']:6.3f}  ({fr['dV_mV'] / VTH / 10:4.1f}% of Vth; "
              f"I={fr['I_write_mA']:.3f} mA, V_drv_needed={fr['V_drv_needed_V']:.4f} V)")
        print(f"        predistortion: max code {pd['max_comp_code']:3d} / 63  "
              f"max residual {pd['max_resid_mV']:.3f} mV = {pd['max_resid_u']:.4f} u")

    out = dict(
        _label=("Ising 64x64 tile column write-line IR drop; per-row static drive offset "
                "u_off(r)=dV(r)/VT and 6-bit predistortion residual; ported from the "
                "smtj_pbnn_sim writeline flow"),
        provenance=dict(
            met2_sheet_R=dict(value_ohm_sq=rs, source=rs_src),
            extresist_flow_validation_poly=poly if poly else "MISSING (run run_extresist.sh)",
            dac_lsb=dict(value_mV=lsb, source=lsb_src),
            per_row_model="ANALYTIC: dV(r)=VTH*R_line(r)/(RSOT+R_line(r)), "
                          "R_line(r)=2*Rs*r*pitch/W (committed formula, this script)",
        ),
        operating_point=dict(VTH_V=VTH, VT_V=VT, RSOT_ohm=RSOT, layer=LAYER,
                             pitch_um=PITCH_UM, w_um=W_UM, roundtrip_factor=ROUNDTRIP,
                             I_write_at_vth_mA=VTH / RSOT * 1e3),
        sign_convention=("u_off(r) stored POSITIVE = drive deficit: the row-r update "
                         "probability is sigma(u - u_off(r)); solver feed-in applies "
                         "u_offset = -u_off (see ir_solver_impact.py)"),
        per_N=per_n,
    )
    p = HERE / "ir_drop_summary.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
