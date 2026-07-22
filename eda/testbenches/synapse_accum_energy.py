#!/usr/bin/env python3
"""RX-10 — MEASURED sky130 energy of ONE local-field accumulate step.

Section 3.5.6 records that the digital summation of h_i^eff = sum_j J_ij s_j is
absent from BOTH unit-level rows of Table 3.8 (sMTJ-array and CMOS p-bit) while
the FPGA and CPU rows contain it. This testbench grounds that missing term in
the same PDK as the rest of the chapter, so the two rows can be put on one
caliber instead of being defended qualitatively.

Datapath under test — one accumulate of a b-bit two's-complement adder-register,
which is what one neighbour of one spin costs:

    bx_k = w_k XOR sub                (conditional negate: s_j = -1 -> sub = 1)
    {cout_k, sum_k} = FA(q_k, bx_k, c_{k-1}),   c_{-1} = sub
    q_k <- sum_k on the rising clock edge

so ``acc <- acc + s_j * w_j`` in one clock. Cells are the real
sky130_fd_sc_hd netlists (xor2_1 / fa_1 / dfrtp_1 / inv), the clock is
distributed to the register through a two-stage inverter tree whose energy is on
the measured supply, and the supply charge is integrated over a window of
identical accumulate cycles driven by a recorded-seed random weight/sign stream.

Two accounting arms, both measured:
  * ``core``   — XOR + adder + accumulator register + local clock tree; the
                 weight bits arrive from ideal sources. This is the LOWER bound
                 and the number the projection uses, so the caliber conclusion
                 does not rest on the weight-storage assumption.
  * ``wreg``   — the same, plus a b-bit weight register (dfrtp_1) that is
                 reloaded every cycle, i.e. a local weight buffer whose read is
                 charged to the synapse. UPPER bound.
Excluded from both (stated, not estimated): the weight memory array itself, the
global clock distribution up to the local tree, address decode, and routing
parasitics beyond the cell netlists (no extracted layout).

Functional self-check: the final register word is compared against the
software-computed accumulator; a mismatch aborts rather than reporting energy
for a datapath that did not compute the sum.

Run inside WSL (native ngspice + sky130):
    wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/synapse_accum_energy.py
Writes eda/testbenches/synapse_accum_energy_summary.json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import SKY130_LIB, write_summary   # noqa: E402

SC_HD = ("/opt/pdk/sky130A/libs.ref/sky130_fd_sc_hd/spice/"
         "sky130_fd_sc_hd.spice")
VDD = 1.8
TCLK = 4.0e-9         # accumulate clock period [s]
TEDGE = 50e-12        # source rise/fall [s]
N_WARM = 2            # cycles discarded (reset release transient)
N_MEAS = 16           # cycles inside the energy window
SEED = 2024
# clock-tree second stage sized to the register it drives
CLKDRV = {8: "inv_8", 12: "inv_12", 16: "inv_16"}


def _pwl(name, node, bits, t0):
    """PWL source for one data bit: value changes a quarter cycle AFTER each
    rising clock edge, so it is stable for 3/4 of a cycle before capture."""
    pts, prev = [(0.0, float(bits[0]) * VDD)], float(bits[0])
    for m, b in enumerate(bits[1:], start=1):
        if float(b) == prev:
            continue
        t = t0 + (m - 1) * TCLK + TCLK / 4.0
        pts.append((t, prev * VDD))
        pts.append((t + TEDGE, float(b) * VDD))
        prev = float(b)
    body = " ".join(f"{t:.6e} {v:.4f}" for t, v in pts)
    return f"{name} {node} 0 PWL({body})"


def build_deck(b, words, subs, with_wreg, mode="add"):
    """Assemble the accumulate deck for a b-bit datapath.

    mode 'add'   — conditional-negate + ripple-carry adder + accumulator
                   register: q <- q +/- w, one general weighted neighbour.
    mode 'count' — enabled synchronous up-counter (prefix AND chain + toggle
                   XOR + register): the datapath a binary +/-J graph actually
                   needs, since h_eff = 2*n_aligned - deg is a popcount. subs[m]
                   is then the per-neighbour enable and words is ignored.

    words[m], subs[m] are the weight word and the sign/enable bit presented
    during cycle m (m = 0 .. N_WARM+N_MEAS-1). Capture edge m sits at
    t0 + m*TCLK and the cycle-m data arrives at t0 + (m-1)*TCLK + TCLK/4, i.e.
    3/4 of a cycle of carry-chain settling before the edge that captures it."""
    ncyc = len(words)
    t0 = 2 * TCLK                       # first rising edge
    # the clock is a finite train of exactly ncyc edges and is then held low,
    # so the tail of the transient is genuinely quiescent (the ripple carry has
    # settled, no register clocks) and can serve as the leakage baseline
    t_end = t0 + (ncyc + 1) * TCLK
    t_meas0 = t0 + (N_WARM - 0.5) * TCLK
    t_meas1 = t0 + (N_WARM + N_MEAS - 0.5) * TCLK
    t_lk0, t_lk1 = t_end - 0.5 * TCLK, t_end - 0.02 * TCLK

    ck = [(0.0, 0.0)]
    for m in range(ncyc):
        tr, tf = t0 + m * TCLK, t0 + m * TCLK + TCLK / 2
        ck += [(tr, 0.0), (tr + TEDGE, VDD), (tf, VDD), (tf + TEDGE, 0.0)]
    ck.append((t_end, 0.0))

    L = [f"* RX-10 synapse accumulate, b={b}, mode={mode}, wreg={with_wreg}",
         f".lib {SKY130_LIB} tt",
         f".include {SC_HD}",
         ".option scale=1u",
         f"Vdd vpwr 0 dc {VDD}",
         # clock: ideal source -> inv_2 -> inv_N -> register clock net
         "Vclk clkin 0 PWL(" +
         " ".join(f"{t:.6e} {v:.4f}" for t, v in ck) + ")",
         "Xcb0 clkin 0 0 vpwr vpwr clkb sky130_fd_sc_hd__inv_2",
         f"Xcb1 clkb 0 0 vpwr vpwr clk sky130_fd_sc_hd__{CLKDRV[b]}",
         # reset released before the first capture edge
         f"Vrb rstb 0 PULSE(0 {VDD} {t0 - 0.75*TCLK:.6e} {TEDGE:.2e} "
         f"{TEDGE:.2e} {10*ncyc*TCLK:.6e} {20*ncyc*TCLK:.6e})"]

    # data sources
    L.append(_pwl("Vsub", "sub_s", [s for s in subs], t0))
    if mode == "count":
        L.append("Rsub sub_s p0 0.001")
        for k in range(b):
            L.append(f"Xt{k} q{k} p{k} 0 0 vpwr vpwr s{k} "
                     "sky130_fd_sc_hd__xor2_1")
            if k < b - 1:
                L.append(f"Xp{k} p{k} q{k} 0 0 vpwr vpwr p{k+1} "
                         "sky130_fd_sc_hd__and2_1")
            L.append(f"Xq{k} clk s{k} rstb 0 0 vpwr vpwr q{k} "
                     "sky130_fd_sc_hd__dfrtp_1")
        L += [f".tran {TEDGE/5:.3e} {t_end:.6e}",
              f".meas tran qwin integ i(Vdd) from={t_meas0:.6e} "
              f"to={t_meas1:.6e}",
              f".meas tran qlk  integ i(Vdd) from={t_lk0:.6e} to={t_lk1:.6e}"]
        for k in range(b):
            L.append(f".meas tran fq{k} FIND v(q{k}) AT={t_end - 0.05*TCLK:.6e}")
        L.append(".end")
        return ("\n".join(L) + "\n",
                dict(t_meas0=t_meas0, t_meas1=t_meas1,
                     t_lk0=t_lk0, t_lk1=t_lk1, t_end=t_end))

    for k in range(b):
        L.append(_pwl(f"Vw{k}", f"w{k}_s", [(w >> k) & 1 for w in words], t0))

    if with_wreg:
        # local weight register: reloaded every cycle, its read drives the XOR
        L.append("Xwsub clk sub_s rstb 0 0 vpwr vpwr sub "
                 "sky130_fd_sc_hd__dfrtp_1")
        for k in range(b):
            L.append(f"Xwr{k} clk w{k}_s rstb 0 0 vpwr vpwr w{k} "
                     "sky130_fd_sc_hd__dfrtp_1")
    else:
        L.append("Rsub sub_s sub 0.001")
        for k in range(b):
            L.append(f"Rw{k} w{k}_s w{k} 0.001")

    # conditional negate + ripple-carry adder + accumulator register
    for k in range(b):
        cin = "sub" if k == 0 else f"c{k-1}"
        L.append(f"Xx{k} w{k} sub 0 0 vpwr vpwr bx{k} "
                 "sky130_fd_sc_hd__xor2_1")
        L.append(f"Xa{k} q{k} bx{k} {cin} 0 0 vpwr vpwr c{k} s{k} "
                 "sky130_fd_sc_hd__fa_1")
        L.append(f"Xq{k} clk s{k} rstb 0 0 vpwr vpwr q{k} "
                 "sky130_fd_sc_hd__dfrtp_1")

    L += [f".tran {TEDGE/5:.3e} {t_end:.6e}",
          f".meas tran qwin integ i(Vdd) from={t_meas0:.6e} to={t_meas1:.6e}",
          f".meas tran qlk  integ i(Vdd) from={t_lk0:.6e} to={t_lk1:.6e}"]
    for k in range(b):
        L.append(f".meas tran fq{k} FIND v(q{k}) AT={t_end - 0.05*TCLK:.6e}")
    L.append(".end")
    return ("\n".join(L) + "\n",
            dict(t_meas0=t_meas0, t_meas1=t_meas1,
                 t_lk0=t_lk0, t_lk1=t_lk1, t_end=t_end))


def run(b, with_wreg, wmode, rng):
    """wmode 'rand'  — general weighted graph: a fresh multi-bit weight word per
                       neighbour on the adder datapath.
       wmode 'unit'  — binary +/-J coupling run through the SAME general adder
                       (|w| = 1, only the sign toggles): what a shared datapath
                       costs on a Max-Cut graph.
       wmode 'count' — binary +/-J coupling on a dedicated enabled up-counter,
                       the popcount form h_eff = 2*n_aligned - deg.
       wmode 'idle'  — the adder datapath clocked with an all-zero addend: no
                       data node ever toggles, so this isolates the clock-tree
                       and register-internal floor that every accumulate pays."""
    ncyc = N_WARM + N_MEAS
    mode = "count" if wmode == "count" else "add"
    if wmode == "rand":
        words = [int(rng.integers(0, 1 << (b - 2))) for _ in range(ncyc)]
    elif wmode == "idle":
        words = [0] * ncyc
    else:
        words = [1] * ncyc
    subs = ([0] * ncyc if wmode == "idle"
            else [int(rng.integers(0, 2)) for _ in range(ncyc)])
    deck, tw = build_deck(b, words, subs, with_wreg, mode=mode)
    tag = f"syn_b{b}_{wmode}_{'wreg' if with_wreg else 'core'}"

    # run_deck() from _common lives in this directory and loads the OSDI;
    # the synapse datapath is pure CMOS, so call ngspice directly here.
    import subprocess
    path = HERE / f"_{tag}.spice"
    path.write_text(deck)
    r = subprocess.run(["ngspice", "-b", path.name], cwd=HERE,
                       capture_output=True, text=True)
    (HERE / f"_{tag}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    out = r.stdout
    if r.returncode != 0:
        raise RuntimeError(f"ngspice failed for {tag}; see _{tag}.log")

    def meas(name):
        m = re.search(rf"^{name}\s*=\s*([-\d.eE+]+)", out, re.M)
        if m is None:
            raise RuntimeError(f"missing .meas {name} in {tag}")
        return float(m.group(1))

    q_win, q_lk = abs(meas("qwin")), abs(meas("qlk"))
    t_win = tw["t_meas1"] - tw["t_meas0"]
    t_lk = tw["t_lk1"] - tw["t_lk0"]
    i_leak = q_lk / t_lk
    e_total = q_win * VDD / N_MEAS
    e_leak = i_leak * t_win * VDD / N_MEAS
    e_dyn = e_total - e_leak

    # Functional self-check against the software accumulator. With the weight
    # register in place the addend is one cycle late (both registers clock on
    # the same edge), so the last presented word never reaches the adder.
    pairs = list(zip(words, subs))
    if with_wreg:
        pairs = pairs[:-1]
    acc = 0
    for w, s in pairs:
        acc = ((acc + s) if mode == "count"
               else (acc + (-w if s else w))) % (1 << b)
    got = 0
    for k in range(b):
        got |= (1 if meas(f"fq{k}") > VDD / 2 else 0) << k
    if got != acc:
        raise RuntimeError(f"{tag}: datapath self-check FAILED — spice word "
                           f"{got} != software {acc} (b={b})")

    arm = f"{wmode}-{'wreg' if with_wreg else 'core'}"
    return dict(b=b, arm=arm, wmode=wmode, wreg=with_wreg,
                e_accum_J=e_total, e_accum_dyn_J=e_dyn, e_accum_leak_J=e_leak,
                i_leak_A=i_leak, q_window_C=q_win,
                n_cycles=N_MEAS, tclk_s=TCLK, selfcheck_word=got, ok=True)


def _fit(d):
    """Least-squares e_accum(b) = a + s*b over the measured widths [fJ]."""
    bs = np.array(sorted(d), float)
    es = np.array([d[b] for b in sorted(d)], float)
    s, a = np.polyfit(bs, es, 1)
    return float(a), float(s)


def main():
    rows = []
    for b in sorted(CLKDRV):
        for wmode, wreg in (("rand", False), ("rand", True),
                            ("unit", False), ("count", False),
                            ("idle", False)):
            rng = np.random.default_rng(SEED + b)   # same stream per width
            rows.append(run(b, wreg, wmode, rng))
            r = rows[-1]
            print(f"b={b:<3d} {r['arm']:<10s} e_accum = "
                  f"{r['e_accum_J']*1e15:7.1f} fJ  "
                  f"(dyn {r['e_accum_dyn_J']*1e15:6.1f}, "
                  f"leak {r['e_accum_leak_J']*1e15:5.1f})")

    core = {r["b"]: r for r in rows if r["arm"] == "rand-core"}
    wreg = {r["b"]: r for r in rows if r["arm"] == "rand-wreg"}
    unit = {r["b"]: r for r in rows if r["arm"] == "unit-core"}
    cnt = {r["b"]: r for r in rows if r["arm"] == "count-core"}
    idle = {r["b"]: r for r in rows if r["arm"] == "idle-core"}
    # canonical width: 6-bit weight magnitude (the measured write-DAC grid of
    # Section 3.5.1) + sign + 6 bits of headroom for a degree-64 tile
    B_CANON = 12
    fJ = {k: {str(b): v[b]["e_accum_J"] * 1e15 for b in v}
          for k, v in (("rand_core", core), ("rand_wreg", wreg),
                       ("unit_core", unit), ("count_core", cnt),
                       ("clock_floor", idle))}
    fits = {k: _fit({int(b): e for b, e in v.items()}) for k, v in fJ.items()}
    res = {
        "_label": ("RX-10 MEASURED (ngspice, sky130 tt, sky130_fd_sc_hd cell "
                   "netlists, schematic-level, no extracted parasitics) energy "
                   "of one local-field accumulate q <- q +/- w"),
        "method": {
            "datapath": "b x (xor2_1 conditional negate + fa_1 ripple carry + "
                        "dfrtp_1 accumulator bit) + inv_2/inv_N clock tree",
            "supply_V": VDD, "corner": "tt", "tclk_s": TCLK,
            "cycles_measured": N_MEAS, "cycles_discarded": N_WARM,
            "arms": {
                "rand_core": "general weighted graph: a fresh uniform random "
                             "(b-2)-bit weight word and a random sign bit per "
                             "neighbour; weights from ideal sources",
                "rand_wreg": "rand_core plus a b-bit local weight register "
                             "reloaded every cycle (weight read charged to the "
                             "synapse) -> upper bound",
                "unit_core": "binary +/-J coupling (all G-set Max-Cut "
                             "instances) run through the SAME general adder: "
                             "|w| = 1, only the sign toggles",
                "count_core": "binary +/-J coupling on the datapath it actually "
                              "needs — an enabled synchronous up-counter "
                              "(prefix AND chain + toggle XOR + register) "
                              "computing h_eff = 2*n_aligned - deg; random "
                              "enable stream -> the cheap regime",
                "clock_floor": "the adder datapath clocked with an all-zero "
                               "addend: no data node toggles, so this isolates "
                               "the clock-tree + register-internal energy that "
                               "every accumulate pays regardless of operand",
            },
            "rng_seed_base": SEED,
            "selfcheck": "final register word == software accumulator mod 2^b",
            "included": ["conditional negate", "ripple-carry adder",
                         "accumulator register", "local clock tree (2 stages)",
                         "carry-chain glitch power",
                         "cell leakage over the accumulate window"],
            "excluded": ["weight memory array read (core arms)",
                         "global clock distribution above the local tree",
                         "address decode", "extracted layout parasitics",
                         "the write-DAC recode (already in "
                         "update_energy_summary.json)"],
        },
        "b_canonical": B_CANON,
        "b_canonical_rationale": ("6-bit weight magnitude matching the measured "
                                  "6-bit write-DAC grid + sign + 6 bits of "
                                  "headroom for a degree-64 tile"),
        "e_accum_fJ": fJ,
        "fit_a_s_fJ": {k: {"intercept_fJ": a, "slope_fJ_per_bit": s}
                       for k, (a, s) in fits.items()},
        "e_accum_fJ_canonical": core[B_CANON]["e_accum_J"] * 1e15,
        "e_accum_fJ_canonical_upper": wreg[B_CANON]["e_accum_J"] * 1e15,
        "e_accum_fJ_canonical_unit": unit[B_CANON]["e_accum_J"] * 1e15,
        "e_accum_fJ_canonical_count": cnt[B_CANON]["e_accum_J"] * 1e15,
        "e_accum_fJ_canonical_clock_floor": idle[B_CANON]["e_accum_J"] * 1e15,
        "rows": rows,
    }
    write_summary(HERE / "synapse_accum_energy_summary.json", res)
    print(f"\ncanonical b={B_CANON}: {res['e_accum_fJ_canonical']:.1f} fJ "
          f"(weighted adder) / {res['e_accum_fJ_canonical_count']:.1f} fJ "
          f"(binary popcount) / {res['e_accum_fJ_canonical_upper']:.1f} fJ "
          f"(weighted, with weight register)")
    for k, (a, s) in fits.items():
        print(f"fit {k:<10s}: {a:7.1f} fJ + {s:6.1f} fJ/bit")


if __name__ == "__main__":
    main()
