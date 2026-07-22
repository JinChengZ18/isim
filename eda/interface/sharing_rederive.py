#!/usr/bin/env python3
"""RX-10 correction — re-derive the energy-parity condition under the sharing
model the measured circuit actually supports.

The first RX-10 pass amortised the whole power-gated static term over S shared
columns. An audit of that model against this repo's own measurements rejects it:

  * `statics_gated_pJ` is the analog front-end's rail energy DURING the (k+1)
    pulse windows, not idle power (update_energy_summary.json/buffer_static
    note: under current steering the rail current is the same in and out of the
    pulse, so the gated term already contains the pulse Ohmic energy).
  * If S columns are served SEQUENTIALLY by one buffer — which is the reading
    the timing model itself takes, since it serialises the settle time by S —
    every column still needs its own (k+1) pulse windows with the buffer live,
    so the pulse-window energy does not amortise at all.
  * If they were served SIMULTANEOUSLY, one class-A stage would have to source
    S x ~1.16 mA against its measured 1.593 mA total rail current. The trial
    log's 2026-07-13 W3 entry records that exact stage failing on a mere 2x
    load step (delivered flat top 87 mV low), and sizing it up would raise its
    static power by ~S in turn.

So the only term sharing genuinely amortises is the resistor-string bias:
5.48 uW of the 2874 uW analog static budget, i.e. 0.19%. This script prints the
parity conditions under that corrected model, and the driver-static reduction
that WOULD reach parity, which is the honest replacement for the retracted
"S >= 1.9 / 5.5 / 102" claim.

Run:  python eda/interface/sharing_reredive.py   (writes sharing_reredive.json)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
E_JSON = ROOT / "eda" / "testbenches" / "update_energy_summary.json"
DEG_JSON = HERE / "graph_degrees.json"
CMOS_E_PJ = 5.0            # hardware_metrics.CMOS_PBIT.e_update (LITERATURE)
PULSE_WIN_NS = 0.85        # TW + 2*TR, update_energy_summary timing_model


def main():
    ej = json.loads(E_JSON.read_text())
    p_buf = ej["buffer_static"]["p_buf_W"]
    p_str = ej["dac_static"]["p_string_W"]
    table = {r["k"]: r for r in ej["table"]}
    deg = json.loads(DEG_JSON.read_text())["instances"]

    share_frac = p_str / (p_buf + p_str)      # the only amortisable fraction
    out = {
        "_label": ("RX-10 correction: parity under the sharing model the "
                   "measured circuit supports (only the DAC string amortises)"),
        "p_buf_mW": p_buf * 1e3, "p_string_uW": p_str * 1e6,
        "amortisable_fraction_of_static": share_frac,
        "cmos_e_update_pJ": CMOS_E_PJ,
        "cmos_uncertainty_note": ("hardware_metrics annotates the 5 pJ CMOS "
                                  "figure as carrying factor 2-3 absolute "
                                  "uncertainty; every ratio below inherits it"),
        "per_k": {}, "same_caliber": {},
    }

    for k, row in table.items():
        e = row["e_update_gated_pJ"]
        # sharing at S -> infinity removes only the string term
        e_inf = e - row["statics_gated_pJ"] * share_frac
        # driver static power that would bring THIS k to parity
        # e = pulses + read + (P_buf + P_str)*(k+1)*t_win  ->  solve for P_buf
        fixed = row["e_pulses_pJ"] + row["e_read_pJ"]
        win_s = (k + 1) * PULSE_WIN_NS * 1e-9
        p_needed = ((CMOS_E_PJ - fixed) * 1e-12) / win_s - p_str
        out["per_k"][k] = dict(
            e_gated_pJ=e, e_gated_share_inf_pJ=round(e_inf, 4),
            ratio_vs_cmos=round(e / CMOS_E_PJ, 4),
            p_buf_for_parity_mW=(round(p_needed * 1e3, 4)
                                 if p_needed > 0 else None),
            p_buf_reduction_needed=(round(p_buf / p_needed, 3)
                                    if p_needed > 0 else None))

    for inst in ("G22", "G1"):
        syn = deg[inst]["deg_mean"] * 366.759e-3      # pJ, b=8 count_core
        rows = {}
        for k, row in table.items():
            e = row["e_update_gated_pJ"]
            rows[k] = round((e + syn) / (CMOS_E_PJ + syn), 4)
        out["same_caliber"][inst] = dict(deg_mean=deg[inst]["deg_mean"],
                                         synapse_pJ=round(syn, 4),
                                         ratio_vs_cmos_by_k=rows)

    (HERE / "sharing_reredive.json").write_text(json.dumps(out, indent=2))
    print(f"amortisable fraction of the analog static term: "
          f"{share_frac*100:.2f}%  (string {p_str*1e6:.2f} uW of "
          f"{(p_buf+p_str)*1e6:.0f} uW)")
    for k, d in out["per_k"].items():
        print(f"k={k}: e_gated {d['e_gated_pJ']:.2f} pJ -> {d['e_gated_share_inf_pJ']:.2f} "
              f"at S->inf; ratio {d['ratio_vs_cmos']:.2f}; parity needs P_buf "
              f"{d['p_buf_for_parity_mW']} mW "
              f"(x{d['p_buf_reduction_needed']} reduction)")
    for inst, d in out["same_caliber"].items():
        print(f"{inst} same-caliber ratio by k: {d['ratio_vs_cmos_by_k']}")
    print("-> sharing_reredive.json")


if __name__ == "__main__":
    main()
