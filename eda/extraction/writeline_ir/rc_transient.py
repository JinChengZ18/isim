#!/usr/bin/env python3
"""RX-13: write-line RC transient — does the met2 column line settle inside the
0.75 ns write pulse, and is the delivered far-row voltage data/position dependent
once several rows draw current simultaneously through the shared line?

Background (what §3.5.3 does): analyze_ir.py extracts only the met2 write-line
RESISTANCE (Magic extract-do-resistance, 0.125 ohm/sq, MEASURED) and models the
per-row IR drop as a STATIC DC offset u_off(r) = dV(r)/VT computed for ONE cell
active at a time (row r alone), dV(r) = VTH * R_line(r)/(RSOT + R_line(r)),
R_line(r) = 2*Rs*r*pitch/W. Predistortion cancels each u_off(r) with a per-row DAC
code; the residual is <= LSB/2. Two things that model never checks:
  (i)  BANDWIDTH — the line is an RC network; does the far node reach its DC value
       within the 0.75 ns pulse, or is there a dynamic short-fall on top of the
       static drop?
  (ii) COUPLING  — if a block/chromatic update fires SEVERAL rows on the same
       column line at once, the shared trunk carries their cumulative current, so
       the far-row drop is no longer its independent dV(r) but depends on WHICH and
       HOW MANY neighbours are co-active (occupancy/position) and on their drive
       level (reset top-code current > write mid-code current). The static model
       assumes the offsets superpose independently.

This script builds the distributed RC ladder from the SAME committed data the static
model uses (met2 Rs = 0.125 ohm/sq -> 0.5 ohm/row round-trip trunk; per-row line C)
and answers both with ngspice.

Capacitance provenance (INTEGRITY):
  * C_MEASURED  — Magic `extract all` node capacitance to substrate from the
    committed writeline_straps.ext (n64 = 9596.36 aF over the 128 um line,
    n256 = 45793.2 aF over the 512 um line). This is an UPPER bound on the isolated
    line: the straps sit 5 um apart in the extraction GDS, so the node C carries some
    neighbour coupling (node C grows super-linearly with length: n256/n64 = 4.77 vs
    length 4.0). Using the high value makes the settling test conservative.
  * C_ANALYTIC  — sky130A.tech defaultareacap allm2 metal2 = 17.5 aF/um^2 applied to
    L*W (area plate only), a LOWER bound (no fringe). The truth sits between the two;
    the RC verdict is the same across the whole range.
Resistance is the committed MEASURED 0.125 ohm/sq (unchanged from analyze_ir).

Taps: each active cell is the SOT write branch modelled as RSOT = 776 ohm from its
line node to the return rail (ground) — identical to the constant RSOT the static
analyze_ir model uses in its divider, so the DC endpoint is an exact cross-check of
that model rather than a re-parameterisation.

Run IN WSL (ngspice on PATH):
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/extraction/writeline_ir/rc_transient.py
Writes rc_transient_summary.json next to this script. No RNG anywhere.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---- committed Chapter-2 / extraction constants (single source: analyze_ir.py) ----
VTH = 0.895783           # sigmoid center [V]
VT = 0.023414            # probability window = 1/beta_s [V]
RSOT = 776.0             # SOT write-branch resistance [ohm]
RS_MET2 = 0.125          # met2 sheet R [ohm/sq]  (MEASURED, committed)
PITCH_UM = 2.0
W_UM = 1.0
ROUNDTRIP = 2.0          # bit line out + source line return
R_PER_ROW = ROUNDTRIP * RS_MET2 * PITCH_UM / W_UM   # 0.5 ohm per row spacing
USPAN = 4.0              # DAC rails Vth +/- 4*VT (from update_chain_dc.py)
TW = 0.75e-9             # write pulse width [s]
TR = 50e-12             # edge (matches update_energy.py) [s]
SETTLE_TOL = 0.1 * VT    # 2.3414 mV settle band (matches update_energy.py)

# sky130A.tech met2 area plate cap (defaultareacap allm2 metal2 17.5) [aF/um^2]
AREACAP_AF_UM2 = 17.5
# committed .ext node capacitances [aF] (MEASURED, Magic extract all)
EXT_NODE_C_AF = {64: 9596.36, 256: 45793.2}


def c_line_farad(n_rows, kind):
    """Total write-line capacitance to ground [F] for the two provenance models."""
    if kind == "MEASURED":
        return EXT_NODE_C_AF[n_rows] * 1e-18
    if kind == "ANALYTIC":
        length_um = n_rows * PITCH_UM
        return AREACAP_AF_UM2 * length_um * W_UM * 1e-18   # area plate only (lower bound)
    raise ValueError(kind)


# ---------------------------------------------------------------- ngspice plumbing
def run(deck, tag):
    p = HERE / f"_rc_{tag}.spice"
    p.write_text(deck)
    r = subprocess.run(["ngspice", "-b", p.name], cwd=HERE,
                       capture_output=True, text=True)
    (HERE / f"_rc_{tag}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"ngspice failed for {tag}; see _rc_{tag}.log")
    return r.stdout


def _load_wrdata(path, ncols):
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s[0] in "*#":
            continue
        parts = [x for x in s.replace(",", " ").split() if x]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, 0], [a[:, 2 * i + 1] for i in range(ncols)]


def ladder(n_rows, c_total_F, active_rows, v_drv, transient):
    """Build a per-row distributed RC ladder deck.

    ln0 = driver near end, ln1..lnN = row nodes. Trunk R_PER_ROW between neighbours,
    line C split evenly to ground at every row node, RSOT tap to ground on active rows.
    """
    c_node = c_total_F / n_rows
    lines = [f"* RC write line N={n_rows} active={len(active_rows)} v_drv={v_drv:.6f}"]
    if transient:
        lines.append(f"Vdrv ln0 0 PULSE(0 {v_drv:.6f} 0 {TR:.3e} {TR:.3e} {TW:.3e} 10n)")
    else:
        lines.append(f"Vdrv ln0 0 {v_drv:.6f}")
    for i in range(1, n_rows + 1):
        lines.append(f"Rseg{i} ln{i-1} ln{i} {R_PER_ROW:.6f}")
        lines.append(f"Cn{i} ln{i} 0 {c_node:.6e}")
    for r in active_rows:
        lines.append(f"Rsot{r} ln{r} 0 {RSOT:.1f}")
    return "\n".join(lines) + "\n"


def dc_farvoltage(n_rows, active_rows, v_drv):
    """DC .op: return delivered voltage at every active row (no C needed)."""
    deck = ladder(n_rows, 1e-30, active_rows, v_drv, transient=False)
    probes = "\n".join(f"  print v(ln{r})" for r in sorted(active_rows))
    deck += f".control\n  op\n{probes}\n  quit\n.endc\n.end\n"
    out = run(deck, f"dc_N{n_rows}_m{len(active_rows)}_{int(round(v_drv*1e4))}")
    v = {}
    for r in active_rows:
        m = re.search(rf"v\(ln{r}\)\s*=\s*([-+0-9.eE]+)", out)
        v[r] = float(m.group(1)) if m else float("nan")
    return v


def settle_transient(n_rows, c_total_F, v_drv, active_rows=None, tag=None):
    """Pulse the near end 0->v_drv; t_settle of the far node to its DC value.

    active_rows defaults to the far cell alone (the settling test); pass a multi-cell
    set to confirm the coupling sag is a fast resistive effect reached inside the pulse.
    """
    if active_rows is None:
        active_rows = [n_rows]
    tag = tag or f"settle_N{n_rows}"
    t_plateau_end = TR + TW                               # PULSE high over [TR, TR+TW]
    deck = ladder(n_rows, c_total_F, active_rows, v_drv, transient=True)
    deck += (f".control\n  tran 0.5p {t_plateau_end:.3e}\n"
             f"  wrdata _rc_{tag}.csv v(ln{n_rows})\n  quit\n.endc\n.end\n")
    run(deck, tag)
    t, (vfar,) = _load_wrdata(HERE / f"_rc_{tag}.csv", 1)
    # flat-top DC value = last 50 ps of the plateau (well before the fall)
    vfinal = float(vfar[t >= t_plateau_end - 50e-12].mean())
    off = np.abs(vfar - vfinal) > SETTLE_TOL
    off[t <= TR] = False                                  # ignore the rising edge itself
    # last instant on the plateau still outside the band, measured from the edge end
    idx = np.where(off & (t <= t_plateau_end))[0]
    t_settle = float(t[idx[-1]] - TR) if idx.size else 0.0
    # 10-90% rise time as a second bandwidth metric
    v10, v90 = 0.1 * vfinal, 0.9 * vfinal
    try:
        t10 = t[np.where(vfar >= v10)[0][0]]
        t90 = t[np.where(vfar >= v90)[0][0]]
        t_rise = float(t90 - t10)
    except IndexError:
        t_rise = float("nan")
    return dict(v_final_V=round(vfinal, 6),
                t_settle_ns=round(t_settle * 1e9, 5),
                t_rise_1090_ns=round(t_rise * 1e9, 5),
                pulse_ns=round(TW * 1e9, 4),
                settle_margin_x=round(TW / t_settle, 1) if t_settle > 0 else float("inf"))


def main():
    summ = json.loads((HERE / "ir_drop_summary.json").read_text())
    out = dict(
        _label=("RX-13 write-line RC transient: settling of the met2 column line vs the "
                "0.75 ns pulse, and data/position dependence of the far-row delivered "
                "voltage under simultaneous multi-row drive. R is committed MEASURED "
                "0.125 ohm/sq; C is MEASURED (.ext node C, high/coupling-contaminated) and "
                "ANALYTIC (techfile areacap, low/plate-only) as bounds. No RNG."),
        constants=dict(VTH_V=VTH, VT_V=VT, RSOT_ohm=RSOT, r_per_row_ohm=R_PER_ROW,
                       areacap_aF_um2=AREACAP_AF_UM2, ext_node_C_aF=EXT_NODE_C_AF,
                       tw_ns=TW*1e9, tr_ns=TR*1e9, settle_tol_mV=round(SETTLE_TOL*1e3, 4)),
        rng=dict(seeds=[], note="deterministic RC network; no stochastic element"),
        settling={}, dc_crosscheck={}, coupling={},
    )

    v_write = VTH                       # driver calibrated so near cell ~ Vth (analyze_ir)
    v_reset = VTH + USPAN * VT          # top-code reset drive (~1.0 V), higher current

    # ---- (i) settling + (ii) DC cross-check, per N and per C model ----
    for n_rows in (64, 256):
        far_static = summ["per_N"][str(n_rows)]["far_row"]
        # DC cross-check: single far cell must reproduce analyze_ir dV(N) exactly
        v_far_only = dc_farvoltage(n_rows, [n_rows], v_write)[n_rows]
        dv_meas_mV = (v_write - v_far_only) * 1e3
        dv_ref_mV = far_static["dV_mV"]
        out["dc_crosscheck"][str(n_rows)] = dict(
            v_far_V=round(v_far_only, 6), dV_meas_mV=round(dv_meas_mV, 5),
            dV_static_ref_mV=round(dv_ref_mV, 5),
            abs_err_uV=round(abs(dv_meas_mV - dv_ref_mV) * 1e3, 4))
        assert abs(dv_meas_mV - dv_ref_mV) < 1e-2, \
            f"RC DC endpoint {dv_meas_mV} != static {dv_ref_mV} at N={n_rows}"

        out["settling"][str(n_rows)] = {}
        for kind in ("MEASURED", "ANALYTIC"):
            c_tot = c_line_farad(n_rows, kind)
            s = settle_transient(n_rows, c_tot, v_write)
            s["C_total_fF"] = round(c_tot * 1e15, 4)
            s["C_source"] = kind
            out["settling"][str(n_rows)][kind] = s

    # ---- (iii) coupling: far-row drop vs simultaneous-occupancy pattern ----
    # Static model = far cell alone (its independent dV). Reality if a colour class
    # co-fires on the shared line = far cell + m spread neighbours drawing current.
    for n_rows in (64, 256):
        far_static = summ["per_N"][str(n_rows)]["far_row"]
        cc = dict(static_far_dV_mV=far_static["dV_mV"],
                  static_far_u_off=far_static["u_off"], patterns={})
        for level_name, v_drv in (("write_midcode", v_write), ("reset_topcode", v_reset)):
            # single far cell at this drive = the correct static reference for this level
            v_ref = dc_farvoltage(n_rows, [n_rows], v_drv)[n_rows]
            dv_ref_mV = (v_drv - v_ref) * 1e3
            rows_out = []
            # occupancy patterns: far row always active, plus evenly-spread co-active rows
            for frac_name, step in (("far_only", None), ("sparse_1_in_8", 8),
                                    ("checker_1_in_2", 2), ("all_rows", 1)):
                if step is None:
                    active = [n_rows]
                else:
                    active = sorted(set(range(step, n_rows + 1, step)) | {n_rows})
                vmap = dc_farvoltage(n_rows, active, v_drv)
                v_far = vmap[n_rows]
                v_min = min(vmap.values())
                r_min = min(vmap, key=vmap.get)
                dv_far_mV = (v_drv - v_far) * 1e3
                extra_mV = dv_far_mV - dv_ref_mV          # coupling beyond static
                rows_out.append(dict(
                    pattern=frac_name, n_active=len(active),
                    v_far_V=round(v_far, 6), dV_far_mV=round(dv_far_mV, 4),
                    extra_over_static_mV=round(extra_mV, 4),
                    extra_over_static_u=round(extra_mV / (VT * 1e3), 5),
                    worst_row=int(r_min), worst_row_V=round(v_min, 6),
                    worst_dV_mV=round((v_drv - v_min) * 1e3, 4)))
            cc["patterns"][level_name] = dict(
                v_drv_V=round(v_drv, 6), dV_static_ref_mV=round(dv_ref_mV, 4),
                occupancy=rows_out)
        # confirm the coupling sag is a fast resistive effect (reached inside the pulse),
        # not a slow bandwidth tail: transient settle of the checkerboard pattern (MEASURED C)
        checker = sorted(set(range(2, n_rows + 1, 2)) | {n_rows})
        ms = settle_transient(n_rows, c_line_farad(n_rows, "MEASURED"), v_write,
                              active_rows=checker, tag=f"checker_N{n_rows}")
        cc["multicell_settle_check"] = dict(
            pattern="checker_1_in_2", n_active=len(checker), C_source="MEASURED",
            v_final_V=ms["v_final_V"], t_settle_ns=ms["t_settle_ns"],
            settle_margin_x=ms["settle_margin_x"],
            note=("checkerboard far-node reaches its (sagged) DC value inside the pulse -> "
                  "the data-dependence is a steady IR/resistive effect, not an RC-bandwidth "
                  "one; the sag itself is what the row-sequential schedule exists to avoid"))
        out["coupling"][str(n_rows)] = cc

    p = HERE / "rc_transient_summary.json"
    p.write_text(json.dumps(out, indent=2))

    # ---- console digest ----
    print("=" * 92)
    print("RX-13 write-line RC transient")
    print(f"  R_per_row = {R_PER_ROW} ohm (round-trip, 0.125 ohm/sq MEASURED)")
    for n_rows in (64, 256):
        dcc = out["dc_crosscheck"][str(n_rows)]
        print(f"\nN={n_rows}:  DC cross-check far dV = {dcc['dV_meas_mV']:.4f} mV vs "
              f"static {dcc['dV_static_ref_mV']:.4f} mV (err {dcc['abs_err_uV']:.3f} uV)")
        for kind in ("MEASURED", "ANALYTIC"):
            s = out["settling"][str(n_rows)][kind]
            print(f"  settle [{kind:8s} C={s['C_total_fF']:6.2f} fF]: "
                  f"t_settle={s['t_settle_ns']*1e3:7.3f} ps  "
                  f"t_rise(10-90)={s['t_rise_1090_ns']*1e3:6.2f} ps  "
                  f"margin x{s['settle_margin_x']}")
        for level in ("write_midcode", "reset_topcode"):
            pat = out["coupling"][str(n_rows)]["patterns"][level]
            print(f"  {level} (v_drv={pat['v_drv_V']:.4f}, static far dV="
                  f"{pat['dV_static_ref_mV']:.3f} mV):")
            for row in pat["occupancy"]:
                print(f"     {row['pattern']:14s} n_act={row['n_active']:3d}  "
                      f"far dV={row['dV_far_mV']:8.3f} mV  "
                      f"extra={row['extra_over_static_mV']:8.3f} mV "
                      f"({row['extra_over_static_u']:+.3f} u)  "
                      f"worst row {row['worst_row']} dV={row['worst_dV_mV']:.2f} mV")
    print(f"\n-> {p.name}")


if __name__ == "__main__":
    main()
