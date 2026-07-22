#!/usr/bin/env python3
"""W7 — re-run the Section 3.4.3 hardware projection with the sMTJ-array
row upgraded from the device-only numbers (t = 0.75 ns, e = 0.78 pJ) to the
END-TO-END per-update cost measured by eda/testbenches/update_energy.py
(k reset pulses + probabilistic write + read + peripheral overheads).

Only the sMTJ-array platform changes; CMOS p-bit / FPGA SBM / CPU rows keep
their Section 3.4.3 definitions, so any shift in the comparison is exactly
the peripheral-cost correction. Reads the same canonical benchmark summaries
as bench_hardware_compare.py.

RX-10 adds three optional axes, all defaulting to the committed behaviour:

  --synapse-fj E   put the digital h_eff accumulation on BOTH unit-level rows
                   (sMTJ and CMOS p-bit) at E fJ per accumulate x the
                   instance's mean coupling degree. Section 3.5.6 records that
                   this term is missing from both rows while the FPGA and CPU
                   rows contain it; supplying it puts the two on one caliber.
                   0 (default) reproduces the committed projection.
  --share S        S columns share one DAC + class-A buffer, so the analog
                   driver's static power is amortised over S spin updates and
                   the shared driver must recode S times per parallel block.
  --rail-vt H      rail half-span in V_T for the ANALYTIC DAC-string static
                   power (as built H = 4; RX-04 requires H >= 10 at array
                   scale, and P_string grows as the square of the span).

Run:  python eda/interface/reproject_hw.py [--k 3]
Writes results_reproject/reproject_summary.csv (+ .json header with the
exact upgraded constants and their provenance).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from hardware_metrics import (HardwarePlatform, SMTJ_ARRAY, CMOS_PBIT,   # noqa: E402
                              FPGA_SBM, cpu_platform_from_run)

# canonical benchmark summaries (Section 3.3), same inputs the original
# projection used via bench_hardware_compare.py
SUMMARIES = [
    ROOT / "results" / "results_compare_maxcut" / "compare_maxcut_summary.csv",
    ROOT / "results" / "results_compare_factor" / "compare_factor_summary.csv",
    ROOT / "results" / "results_compare_tsp" / "comapre_tsp_summary.csv",
]

ENERGY_JSON = ROOT / "eda" / "testbenches" / "update_energy_summary.json"
DEGREES_JSON = HERE / "graph_degrees.json"

# Chapter-2 probability window; the DAC reference rails are quoted in V_T
VT_V = 0.023414
# as-built resistor string (update_energy_summary.json dac_static.formula)
DAC_NBITS_BUILT = 6
DAC_R_UNIT = 100.0
DAC_RAIL_VT_BUILT = 4.0


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def p_string_W(rail_vt=DAC_RAIL_VT_BUILT, nbits=DAC_NBITS_BUILT,
               r_unit=DAC_R_UNIT):
    """ANALYTIC resistor-string static power: (rail span)^2 / (2^b * R_unit),
    the committed formula of update_energy_summary.json['dac_static'] written
    in V_T units. rail_vt is the HALF span, so the full span is 2*rail_vt*V_T."""
    return (2.0 * rail_vt * VT_V) ** 2 / ((2 ** nbits) * r_unit)


def load_components(energy_json=ENERGY_JSON):
    """Pull the measured per-update components out of update_energy_summary."""
    ej = json.loads(Path(energy_json).read_text())
    tm = ej["timing_model"]
    return dict(
        table={r["k"]: r for r in ej["table"]},
        p_buf_W=ej["buffer_static"]["p_buf_W"],
        p_string_built_W=ej["dac_static"]["p_string_W"],
        tw_ns=tm["tw_ns"], t_gap_ns=tm["t_gap_ns"],
        t_read_ns=tm["t_read_ns"], t_settle_ns=tm["t_settle_ns"],
        pulse_window_ns=tm["pulse_window_ns"],
        source=Path(energy_json).name)


def static_scale(comp, share=1.0, rail_vt=DAC_RAIL_VT_BUILT,
                 dac_nbits=DAC_NBITS_BUILT):
    """Factor by which the analog static power moves relative to the as-built
    point: sharing divides it by `share`, a wider rail span raises the DAC
    string term as the square of the span."""
    # the reference uses the SAME closed form as the numerator so that the
    # as-built point returns exactly 1.0 (a re-run must not perturb the
    # committed reproject_summary.csv by float round-off); the committed
    # p_string_built_W is kept as a cross-check of the formula
    built = comp["p_buf_W"] + p_string_W()
    assert abs(p_string_W() / comp["p_string_built_W"] - 1.0) < 1e-4
    return ((comp["p_buf_W"] + p_string_W(rail_vt, dac_nbits)) / share) / built


def e_update_gated_J(comp, k, share=1.0, rail_vt=DAC_RAIL_VT_BUILT,
                     dac_nbits=DAC_NBITS_BUILT):
    """Power-gated end-to-end update energy with the analog front-end shared by
    `share` columns and the DAC string re-evaluated at the requested rail span.
    Written as a perturbation of the PUBLISHED e_update_gated_pJ so that
    share = 1 at the as-built rail reproduces the committed number exactly."""
    row = comp["table"][k]
    sc = static_scale(comp, share, rail_vt, dac_nbits)
    return (row["e_update_gated_pJ"]
            + row["statics_gated_pJ"] * (sc - 1.0)) * 1e-12


def t_update_s(comp, k, share=1.0):
    """A shared analog front-end must settle one code per served column before
    the block can fire, so the settle term carries the sharing factor."""
    ns = (share * comp["t_settle_ns"]
          + (k + 1) * (comp["tw_ns"] + comp["t_gap_ns"])
          + comp["t_read_ns"])
    return ns * 1e-9


def load_degrees(path=DEGREES_JSON):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())["instances"]


def build_platforms(comp, k, args, degrees):
    """Return a factory: instance name -> list of HardwarePlatform rows, with
    the synapse term (if any) applied to the two unit-level rows only."""
    e_e2e = e_update_gated_J(comp, k, args.share, args.rail_vt, args.dac_nbits)
    t_e2e = t_update_s(comp, k, args.share)
    n_par = args.n_par if args.n_par else SMTJ_ARRAY.parallel_n

    def for_instance(inst):
        d = degrees.get(inst, {})
        syn = (args.synapse_fj * 1e-15) * d.get("deg_mean", 0.0)
        return [
            HardwarePlatform("sMTJ-array", SMTJ_ARRAY.t_update,
                             SMTJ_ARRAY.e_update + syn, n_par),
            HardwarePlatform("sMTJ-array-e2e", t_e2e, e_e2e + syn, n_par),
            HardwarePlatform("cmos-pbit", CMOS_PBIT.t_update,
                             CMOS_PBIT.e_update + syn, n_par),
            # FPGA and CPU rows already contain the weighted sum -> untouched
            FPGA_SBM,
        ], syn
    return for_instance, e_e2e, t_e2e, n_par


def project(args):
    """Core of the projection; returns (rows, header dict)."""
    comp = load_components(args.energy_json)
    degrees = load_degrees(args.degrees_json)
    for_instance, e_e2e, t_e2e, n_par = build_platforms(comp, args.k, args,
                                                        degrees)
    out_rows, cpu = [], None
    for path in SUMMARIES:
        for r in read_rows(path):
            inst = r.get("instance") or r.get("name")
            n = int(float(r.get("n") or r.get("n_spins") or 0))
            ps = float(r.get("ps_smtj") or r.get("p_success") or 0)
            nsw = int(float(r.get("n_sweeps") or 10000))
            tmed = float(r.get("time_median_smtj") or
                         r.get("time_median") or 0)
            if cpu is None and tmed > 0:
                cpu = cpu_platform_from_run(tmed, nsw, n)
            if ps <= 0:
                continue
            plats, syn = for_instance(inst)
            carries = {"sMTJ-array", "sMTJ-array-e2e", "cmos-pbit"}
            for p in plats + ([cpu] if cpu else []):
                out_rows.append(dict(
                    instance=inst, n=n, platform=p.name,
                    e_update_pJ=p.e_update * 1e12,
                    # the FPGA and CPU rows already contained the weighted sum
                    synapse_pJ=(syn * 1e12 if p.name in carries else ""),
                    tts=p.hardware_tts(n, nsw, ps),
                    energy=p.energy_per_solution(n, nsw, ps)))

    header = dict(
        _label=("Section 3.4.3 projection re-run; the sMTJ-array row carries "
                "the measured end-to-end per-update cost, and (RX-10) an "
                "optional synapse term is applied to the two unit-level rows"),
        k_reset=args.k, share=args.share, n_par=n_par,
        rail_vt=args.rail_vt, dac_nbits=args.dac_nbits,
        synapse_fJ_per_accumulate=args.synapse_fj,
        t_update_e2e_ns=t_e2e * 1e9, e_update_e2e_pJ=e_e2e * 1e12,
        p_string_W=p_string_W(args.rail_vt, args.dac_nbits),
        p_buf_effective_W=comp["p_buf_W"] / args.share,
        provenance=comp["source"],
        device_only=dict(t_update_ns=SMTJ_ARRAY.t_update * 1e9,
                         e_update_pJ=SMTJ_ARRAY.e_update * 1e12))
    return out_rows, header


def add_args(ap):
    ap.add_argument("--k", type=int, default=3,
                    help="reset pulses per update in the end-to-end account")
    ap.add_argument("--energy-json", default=str(ENERGY_JSON))
    ap.add_argument("--degrees-json", default=str(DEGREES_JSON))
    ap.add_argument("--synapse-fj", type=float, default=0.0,
                    help="energy of ONE h_eff accumulate [fJ]; multiplied by "
                         "the instance mean degree and added to the sMTJ and "
                         "CMOS p-bit rows. 0 = committed behaviour.")
    ap.add_argument("--share", type=float, default=1.0,
                    help="columns sharing one DAC + class-A buffer")
    ap.add_argument("--n-par", type=int, default=0,
                    help="override N_par of the sMTJ and CMOS rows (0 = 64)")
    ap.add_argument("--rail-vt", type=float, default=DAC_RAIL_VT_BUILT,
                    help="DAC rail HALF span in V_T (as built 4; RX-04 needs "
                         ">= 10 at array scale)")
    ap.add_argument("--dac-nbits", type=int, default=DAC_NBITS_BUILT)
    ap.add_argument("--outdir", default="results_reproject",
                    help="output dir name under eda/interface/ (RX-03 "
                         "worst-corner runs use results_reproject_<corner>)")
    return ap


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    out_rows, header = project(args)

    odir = HERE / args.outdir
    odir.mkdir(exist_ok=True)
    fields = ["instance", "n", "platform", "tts", "energy"]
    if args.synapse_fj:
        fields += ["e_update_pJ", "synapse_pJ"]
    with open(odir / "reproject_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # headline: shift of the sMTJ row, device-only -> end-to-end
    shifts = {}
    for inst in {r["instance"] for r in out_rows}:
        dev = next(r for r in out_rows
                   if r["instance"] == inst and r["platform"] == "sMTJ-array")
        e2e = next(r for r in out_rows
                   if r["instance"] == inst
                   and r["platform"] == "sMTJ-array-e2e")
        shifts[inst] = dict(tts_x=e2e["tts"] / dev["tts"],
                            energy_x=e2e["energy"] / dev["energy"])
    header["shifts"] = shifts
    (odir / "reproject_header.json").write_text(json.dumps(header, indent=2))

    tx = np.array([s["tts_x"] for s in shifts.values()])
    ex = np.array([s["energy_x"] for s in shifts.values()])
    print(f"sMTJ end-to-end vs device-only (k={args.k}): "
          f"TTS x{tx.mean():.2f}, energy x{ex.mean():.2f} "
          f"(uniform across instances: {tx.std():.3f}/{ex.std():.3f} std)")
    print(f"-> {odir / 'reproject_summary.csv'}")


if __name__ == "__main__":
    main()
