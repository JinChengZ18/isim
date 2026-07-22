#!/usr/bin/env python3
"""RX-10 — put the Section 3.5.4 / Table 3.8 energy comparison on ONE caliber
and locate the sMTJ-vs-CMOS crossover.

Section 3.5.6 states the asymmetry plainly: the digital summation of
h_i^eff = sum_j J_ij s_j is absent from BOTH unit-level rows of Table 3.8
(sMTJ-array and CMOS p-bit) while the FPGA SBM and CPU rows carry it. Section
3.5.4 then defends the post-correction energy deficit to CMOS p-bit as an
accounting-boundary artefact without ever quantifying it. This driver closes
that gap with three measured ingredients:

  1. the per-accumulate energy of a sky130 accumulator
     (eda/testbenches/synapse_accum_energy_summary.json, MEASURED),
  2. the mean coupling degree of every canonical benchmark instance
     (eda/interface/graph_degrees.json, MEASURED from the Problem objects),
  3. the end-to-end per-update energy and timing of the write chain
     (eda/testbenches/update_energy_summary.json, MEASURED),

and sweeps the two engineering axes Section 3.5.4 names only qualitatively:
the analog-driver sharing factor and the parallel width N_par, plus the DAC
rail span that RX-04 forces to >= +/-10 V_T at array scale.

Run:  python eda/interface/energy_caliber.py
Writes eda/interface/energy_caliber_summary.csv
   and eda/interface/energy_caliber_config.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from hardware_metrics import (HardwarePlatform, SMTJ_ARRAY,        # noqa: E402
                              CMOS_PBIT, FPGA_SBM, cpu_platform_from_run)
from reproject_hw import (SUMMARIES, read_rows, load_components,   # noqa: E402
                          load_degrees, e_update_gated_J, t_update_s,
                          p_string_W, static_scale, DAC_RAIL_VT_BUILT,
                          DAC_NBITS_BUILT, VT_V)

SYN_JSON = ROOT / "eda" / "testbenches" / "synapse_accum_energy_summary.json"

K_CANON, NPAR_CANON, SHARE_CANON = 3, 64, 1.0
K_GRID = [1, 2, 3]
NPAR_GRID = [16, 64, 256]
SHARE_GRID = [1.0, 4.0, 16.0]
RAIL_GRID = [DAC_RAIL_VT_BUILT, 10.0]
GRID_INSTANCES = ["G1", "G22"]


# --------------------------------------------------------------- synapse --
def synapse_table(syn, degrees):
    """Per-instance accumulate energy: the arm follows the coupling alphabet
    (binary +/-J -> popcount counter; weighted -> full adder) and the width
    follows the instance's dynamic range. Widths the testbench measured
    directly are MEASURED; others are read off the measured linear fit."""
    meas = syn["e_accum_fJ"]
    fit = syn["fit_a_s_fJ"]
    out = {}
    for inst, d in degrees.items():
        arm = "count_core" if d["binary_coupling"] else "rand_core"
        b = d["b_req"]
        if str(b) in meas[arm]:
            e, prov = meas[arm][str(b)], "MEASURED"
        else:
            a, s = fit[arm]["intercept_fJ"], fit[arm]["slope_fJ_per_bit"]
            e, prov = a + s * b, "ANALYTIC (measured linear fit in b)"
        out[inst] = dict(arm=arm, b_accum=b, e_accum_fJ=e, provenance=prov,
                         deg_mean=d["deg_mean"],
                         synapse_pJ_per_update=e * 1e-15 * d["deg_mean"] * 1e12)
    return out


# ------------------------------------------------------------ projection --
def instances():
    """The canonical Section 3.3 rows with a non-zero success probability."""
    rows, cpu_seed = [], None
    for path in SUMMARIES:
        for r in read_rows(path):
            n = int(float(r.get("n") or 0))
            ps = float(r.get("ps_smtj") or 0)
            nsw = int(float(r.get("n_sweeps") or 10000))
            tmed = float(r.get("time_median_smtj") or 0)
            if cpu_seed is None and tmed > 0:
                cpu_seed = (tmed, nsw, n)
            if ps > 0:
                rows.append(dict(instance=r["instance"], n=n, p_s=ps,
                                 n_sweeps=nsw))
    return rows, cpu_platform_from_run(*cpu_seed)


def read_growth(comp):
    """RX-06 leaves the read decision failing under mismatch, so the 0.12 pJ
    read line item is a nominal-device number. Size the comparator that WOULD
    meet the G1 tolerance and re-price the read.

    sigma_off must fall until the Gaussian tail at the tighter static margin
    reaches the smallest flip rate RX-06 found indistinguishable on G1 (1e-5).
    Pelgrom gives sigma ~ 1/sqrt(WL), so the input devices grow by
    (sigma_meas/sigma_req)^2, and a StrongARM's switching energy is set by the
    capacitance it charges, i.e. approximately linear in that area."""
    from math import sqrt
    try:
        from scipy.stats import norm
        z = float(norm.isf(1e-5))
    except Exception:                                     # no scipy -> bisect
        from math import erfc
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * erfc(mid / sqrt(2.0)) > 1e-5:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
    mc = json.loads((ROOT / "eda" / "testbenches" /
                     "read_offset_mc_summary.json").read_text())
    ue = json.loads((ROOT / "eda" / "testbenches" /
                     "update_energy_summary.json").read_text())
    sig = mc["mc_ideal"]["offset_sigma_mV"]
    sm = mc["read_path"]["static_margin_mV"]
    margin = min(abs(float(m)) for m in
                 (sm.values() if isinstance(sm, dict) else sm))
    sig_req = margin / z
    e_sa = max(s["e_sa_pJ"] for s in ue["read"]["states"])
    e_rail = max(s["e_readrail_pJ"] for s in ue["read"]["states"])
    e_read0 = ue["read"]["e_read_pJ"]
    az = 2.36                       # ch-4 autozero suppression cited in 3.5.5
    out = {}
    for name, pre in (("device_growth_only", 1.0), ("autozero_plus_growth", az)):
        area = (sig / pre / sig_req) ** 2
        e_read = e_sa * area + e_rail
        out[name] = dict(sigma_meas_mV=sig, sigma_required_mV=sig_req,
                         z_at_1e_5=z, margin_mV=margin,
                         prescale=pre, area_factor=area,
                         e_read_pJ=e_read, delta_pJ=e_read - e_read0)
    out["_provenance"] = ("ANALYTIC — Pelgrom area scaling on MEASURED "
                          "sigma_off (read_offset_mc_summary.json) and "
                          "MEASURED e_sa (update_energy_summary.json); the "
                          "autozero suppression is the Section 3.5.5 "
                          "cross-reference to the chapter-4 comparator family, "
                          "not reproduced here, and its own overhead is not "
                          "priced")
    out["_e_read_committed_pJ"] = e_read0
    return out


def point(comp, inst, syn_pJ, k, n_par, share, rail_vt, cpu, extra_smtj_pJ=0.0):
    """One (instance, operating point) group: every platform row plus the two
    ratios against the CMOS p-bit reference. extra_smtj_pJ carries a change to
    an sMTJ-only line item (the RX-06 read-path re-pricing)."""
    syn = syn_pJ * 1e-12
    e_e2e = (e_update_gated_J(comp, k, share, rail_vt, DAC_NBITS_BUILT)
             + extra_smtj_pJ * 1e-12)
    plats = [
        HardwarePlatform("sMTJ-array", SMTJ_ARRAY.t_update,
                         SMTJ_ARRAY.e_update + syn, n_par),
        HardwarePlatform("sMTJ-array-e2e", t_update_s(comp, k, share),
                         e_e2e + syn, n_par),
        HardwarePlatform("cmos-pbit", CMOS_PBIT.t_update,
                         CMOS_PBIT.e_update + syn, n_par),
        FPGA_SBM, cpu,
    ]
    n, nsw, ps = inst["n"], inst["n_sweeps"], inst["p_s"]
    vals = {p.name: dict(e_update_pJ=p.e_update * 1e12,
                         tts_s=p.hardware_tts(n, nsw, ps),
                         energy_J=p.energy_per_solution(n, nsw, ps))
            for p in plats}
    ref = vals["cmos-pbit"]
    for v in vals.values():
        v["energy_ratio_vs_cmos"] = v["energy_J"] / ref["energy_J"]
        v["tts_ratio_vs_cmos"] = v["tts_s"] / ref["tts_s"]
    return vals


def crossover_share(comp, k, rail_vt, extra_smtj_pJ=0.0):
    """Smallest sharing factor at which the gated end-to-end update energy of
    the sMTJ row drops below the 5 pJ CMOS p-bit row. The synapse term is
    common to both rows and therefore cancels out of this condition — that is
    itself the quantitative form of Section 3.5.4's caliber argument."""
    row = comp["table"][k]
    fixed = (row["e_update_gated_pJ"] - row["statics_gated_pJ"]
             + extra_smtj_pJ) * 1e-12
    head = CMOS_PBIT.e_update - fixed
    if head <= 0:
        return float("inf")
    scale = static_scale(comp, 1.0, rail_vt, DAC_NBITS_BUILT)
    statics = row["statics_gated_pJ"] * 1e-12 * scale
    return statics / head


def main():
    comp = load_components()
    degrees = load_degrees()
    syn_json = json.loads(SYN_JSON.read_text())
    syn = synapse_table(syn_json, degrees)
    inst_rows, cpu = instances()

    out, seen = [], set()

    def emit(inst, k, n_par, share, rail_vt, with_syn, block, extra=0.0):
        key = (inst["instance"], k, n_par, share, rail_vt, with_syn, block)
        if key in seen:
            return
        seen.add(key)
        s = syn.get(inst["instance"], {})
        syn_pJ = s.get("synapse_pJ_per_update", 0.0) if with_syn else 0.0
        vals = point(comp, inst, syn_pJ, k, n_par, share, rail_vt, cpu, extra)
        for name, v in vals.items():
            out.append(dict(
                block=block, instance=inst["instance"], n=inst["n"],
                n_sweeps=inst["n_sweeps"], p_s=inst["p_s"],
                deg_mean=s.get("deg_mean", ""), coupling=s.get("arm", ""),
                b_accum=s.get("b_accum", ""),
                e_accum_fJ=round(s.get("e_accum_fJ", 0.0), 2),
                e_accum_provenance=s.get("provenance", ""),
                synapse_on=int(with_syn),
                synapse_pJ_per_update=round(syn_pJ, 4),
                k=k, n_par=n_par, share=share, rail_vt=rail_vt,
                platform=name, **{kk: v[kk] for kk in
                                  ("e_update_pJ", "tts_s", "energy_J",
                                   "energy_ratio_vs_cmos",
                                   "tts_ratio_vs_cmos")}))

    # block A — one caliber, every instance, at the chapter's operating point
    for inst in inst_rows:
        for with_syn in (False, True):
            emit(inst, K_CANON, NPAR_CANON, SHARE_CANON, DAC_RAIL_VT_BUILT,
                 with_syn, "caliber")

    # block B — sensitivity / crossover grid on the two Max-Cut headliners
    by_name = {r["instance"]: r for r in inst_rows}
    for name in GRID_INSTANCES:
        inst = by_name[name]
        for k in K_GRID:
            for n_par in NPAR_GRID:
                for share in SHARE_GRID:
                    for rail in RAIL_GRID:
                        for with_syn in (False, True):
                            emit(inst, k, n_par, share, rail, with_syn, "grid")

    # block C — RX-06: what the read path costs once it is sized to decide
    rg = read_growth(comp)
    for scen in ("device_growth_only", "autozero_plus_growth"):
        for name in GRID_INSTANCES:
            inst = by_name[name]
            for k in K_GRID:
                for share in SHARE_GRID:
                    emit(inst, k, NPAR_CANON, share, DAC_RAIL_VT_BUILT, True,
                         f"readgrow:{scen}", rg[scen]["delta_pJ"])

    fields = list(out[0].keys())
    with open(HERE / "energy_caliber_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    # ---------------------------------------------------------- config ----
    cross = {f"k={k},rail={r}": crossover_share(comp, k, r)
             for k in K_GRID + [4, 5] for r in RAIL_GRID}
    cfg = {
        "_label": "RX-10 same-caliber energy comparison — constants and "
                  "provenance for energy_caliber_summary.csv",
        "constants": {
            "e_accum": {
                "value": "per instance, see synapse_per_instance",
                "provenance": "MEASURED — eda/testbenches/"
                              "synapse_accum_energy.py (ngspice, sky130 tt, "
                              "sky130_fd_sc_hd cell netlists); widths outside "
                              "{8,12,16} read off the measured linear fit",
                "clock_floor_fJ_b12":
                    syn_json["e_accum_fJ_canonical_clock_floor"],
                "arms_fJ_b12": {
                    "weighted adder": syn_json["e_accum_fJ_canonical"],
                    "binary popcount": syn_json["e_accum_fJ_canonical_count"],
                    "weighted adder + weight register":
                        syn_json["e_accum_fJ_canonical_upper"]},
            },
            "deg_mean": {"provenance": "MEASURED from the Problem objects — "
                                       "eda/interface/graph_degrees.py"},
            "e_pulses, e_read, P_buf, t_settle": {
                "provenance": "MEASURED — eda/testbenches/update_energy.py "
                              "(sky130 tt, schematic level)",
                "p_buf_W": comp["p_buf_W"],
                "t_settle_ns": comp["t_settle_ns"]},
            "P_string": {
                "provenance": "ANALYTIC — (rail span)^2 / (2^b * R_unit), the "
                              "committed formula of update_energy_summary.json"
                              " re-evaluated per rail span",
                "V_T_V": VT_V, "R_unit_ohm": 100.0,
                "W_at_rail_vt": {f"+/-{h:g}V_T, {b}bit":
                                 p_string_W(h, b)
                                 for h in (4.0, 10.0) for b in (6, 4)}},
            "cmos_pbit_row": {
                "t_update_s": CMOS_PBIT.t_update,
                "e_update_J": CMOS_PBIT.e_update,
                "provenance": "LITERATURE — Camsari 2020 "
                              "doi:10.1109/JPROC.2020.2966869, unit-level, "
                              "peripherals excluded (the caliber this item "
                              "corrects)"},
            "fpga_cpu_rows": {
                "provenance": "LITERATURE (Goto 2021 "
                              "doi:10.1126/sciadv.abe7953) and runtime "
                              "measurement; both ALREADY contain the weighted "
                              "sum, so the synapse term is not added to them"},
        },
        "assumptions": [
            "one spin update needs exactly one pass over its neighbours, so "
            "the synapse cost per update is deg_mean x e_accum",
            "sharing factor S: S columns share one DAC + class-A buffer, so "
            "the analog static power is amortised over S updates and the "
            "shared front-end settles S codes per parallel block "
            "(t_update = S*t_settle + (k+1)*(t_w+t_gap) + t_read)",
            "the synapse accumulate is assumed pipelined behind the update, "
            "so it enters the energy axis only; at the measured 4 ns "
            "accumulate clock a degree-48 sum is 192 ns and would NOT hide "
            "behind a 14 ns update on a single accumulator",
            "the class-A buffer static power is held at its as-built value "
            "when the rail span is widened to +/-10 V_T; only the DAC string "
            "is re-evaluated, so the wide-rail driver power is a lower bound",
        ],
        "synapse_per_instance": syn,
        "crossover_share_min": cross,
        "read_path_growth": rg,
        "crossover_share_min_with_read_growth": {
            f"{scen},k={k}": crossover_share(comp, k, DAC_RAIL_VT_BUILT,
                                             rg[scen]["delta_pJ"])
            for scen in ("device_growth_only", "autozero_plus_growth")
            for k in K_GRID},
        "grid": {"k": K_GRID, "n_par": NPAR_GRID, "share": SHARE_GRID,
                 "rail_vt": RAIL_GRID, "instances": GRID_INSTANCES},
    }
    (HERE / "energy_caliber_config.json").write_text(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------ report --
    print("per-instance synapse term (mean degree x measured accumulate):")
    for name in [r["instance"] for r in inst_rows]:
        s = syn[name]
        print(f"  {name:<14s} deg {s['deg_mean']:7.2f}  {s['arm']:<10s} "
              f"b={s['b_accum']:<3d} {s['e_accum_fJ']:7.1f} fJ  -> "
              f"{s['synapse_pJ_per_update']:8.2f} pJ/update  "
              f"[{s['provenance'].split()[0]}]")

    print("\nsMTJ-e2e / CMOS energy ratio at the chapter operating point "
          f"(k={K_CANON}, share=1):")
    for name in [r["instance"] for r in inst_rows]:
        a = next(r for r in out if r["block"] == "caliber"
                 and r["instance"] == name and r["synapse_on"] == 0
                 and r["platform"] == "sMTJ-array-e2e")
        b = next(r for r in out if r["block"] == "caliber"
                 and r["instance"] == name and r["synapse_on"] == 1
                 and r["platform"] == "sMTJ-array-e2e")
        print(f"  {name:<14s} before {a['energy_ratio_vs_cmos']:6.3f}x  "
              f"after {b['energy_ratio_vs_cmos']:6.3f}x")

    print("\nminimum sharing factor for the sMTJ row to re-enter below CMOS:")
    for key, v in cross.items():
        print(f"  {key:<16s} S >= {v:8.2f}" if v != float("inf")
              else f"  {key:<16s} unreachable at any S")

    print("\nRX-06 read-path re-pricing (read energy sized to decide):")
    for scen in ("device_growth_only", "autozero_plus_growth"):
        d = rg[scen]
        print(f"  {scen:<22s} area x{d['area_factor']:5.1f}  e_read "
              f"{rg['_e_read_committed_pJ']:.3f} -> {d['e_read_pJ']:.3f} pJ")
        for k in K_GRID:
            v = cfg["crossover_share_min_with_read_growth"][f"{scen},k={k}"]
            print(f"      k={k}: " + (f"S >= {v:.2f}" if v != float("inf")
                                      else "unreachable at any S"))
    print(f"\n-> {HERE / 'energy_caliber_summary.csv'} ({len(out)} rows)")
    print(f"-> {HERE / 'energy_caliber_config.json'}")


if __name__ == "__main__":
    main()
