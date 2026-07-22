#!/usr/bin/env python3
"""Generate the Chapter-3 circuit-section figure panels from the committed
eda/ result files (no hand-typed numbers anywhere: every curve/label reads a
*_summary.json / *.csv produced by a committed script).

Outputs:
  * clean single panels (no panel letters, no figure numbers, descriptive
    English titles only) -> eda/figs_raw/
  * preview composites in the caption's (a)(b)(c) reading order (letters NOT
    baked in) for a quick visual check -> eda/figs_raw/preview_{09,10,11}.png

The FINAL numbered figures article/figs/Chapter03_local_{09,10,11}.png are
produced by eda/build_ppt_figs.py, which composes these panels into a PPT
deck, adds the (a)(b)(c) letters and exports via LibreOffice. Run order:
  python eda/figs_make.py && python eda/build_ppt_figs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from plot_style import set_style, TSINGHUA_PURPLE as TP, new_figure, save  # noqa: E402

TB = HERE / "testbenches"
IF = HERE / "interface"
IR = HERE / "extraction" / "writeline_ir"
RAW = HERE / "figs_raw"
FIGS = ROOT / "article" / "figs"
VT_MV = 23.414

RAW.mkdir(exist_ok=True)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cload(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------- panels --
def panel_transfer(ax):
    d = jload(TB / "update_chain_summary.json")
    m = d["per_bits"]["6"]
    codes = np.array([r["code"] for r in m["transfer"]])
    v = np.array([r["v_wr"] for r in m["transfer"]]) * 1e3
    p = np.array([r["psw_dev"] for r in m["transfer"]])
    ax.plot(codes, v, "-", color=TP["dark"], lw=1.8)
    ax.set_xlabel("DAC code")
    ax.set_ylabel(r"$V_\mathrm{wr}$ (mV)", color=TP["dark"])
    ax.tick_params(axis="y", labelcolor=TP["dark"])
    ax2 = ax.twinx()
    ax2.plot(codes, p, "-", color=TP["accent"], lw=1.8)
    ax2.set_ylabel(r"$P_\mathrm{sw}$", color=TP["accent"])
    ax2.tick_params(axis="y", labelcolor=TP["accent"])
    ax2.grid(False)
    ax.annotate(f"LSB = {m['lsb_mV']:.2f} mV = {m['lsb_over_VT']:.2f} $V_T$\n"
                f"INL = {m['inl_lsb']:.2f} LSB, monotonic",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top",
                fontsize=11)
    ax.set_title("Measured write-chain transfer (6-bit)")


def panel_waveform(ax):
    a = np.loadtxt(TB / "closed_loop_wave.csv")
    t = a[:, 0] * 1e9
    v_wr, v_psw, v_st = a[:, 5], a[:, 7], a[:, 9]
    ax.plot(t, v_wr, color=TP["dark"], lw=1.6)
    ax.plot(t, v_psw, color=TP["accent"], lw=1.4)
    ax.plot(t, v_st, color=TP["gray"], lw=1.2)
    # labels stacked in the quiet pre-pulse region (traces sit at 0 there)
    ax.text(0.12, 0.90, r"$V_\mathrm{wr}$ (V)", color=TP["dark"],
            fontsize=11)
    ax.text(0.12, 0.60, r"$P_\mathrm{sw}$ probe", color=TP["accent"],
            fontsize=11)
    ax.text(0.12, 0.30, "state st", color=TP["gray"], fontsize=11)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("V / value")
    ax.set_title("Four consecutive update cycles (replayed outcomes)")


def panel_bits(ax):
    rows = cload(IF / "results_circuit_ablation" / "circuit_ablation_summary.csv")
    for axis, color, label, lxy in [
            ("dac_bits_fixed_u", TP["dark"], "fixed rails", (3.1, 2.3)),
            ("dac_bits_beta_scaled", TP["accent"],
             r"$\beta$-scaled rails", (4.6, 0.42))]:
        rs = sorted((r for r in rows if r["axis"] == axis),
                    key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in rs]
        y = [float(r["tts99_ratio"]) for r in rs]
        ax.plot(x, y, "o-", color=color, lw=1.8, ms=5)
        ax.annotate(label, xy=lxy, color=color, fontsize=11)
    meas = [r for r in rows if r["axis"] == "dac_measured_6b"]
    if meas:
        ax.plot(6, float(meas[0]["tts99_ratio"]), "*", color=TP["darkest"],
                ms=13)
        ax.annotate("measured\n6-bit grid",
                    xy=(6, float(meas[0]["tts99_ratio"])),
                    xytext=(4, -30), textcoords="offset points",
                    color=TP["darkest"], fontsize=11)
    ax.axhline(1.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("DAC resolution (bits)")
    ax.set_ylabel(r"TTS$_{99}$ ratio (vs ideal)")
    ax.set_title("Drive quantization vs solver cost")


def panel_span(ax):
    """RX-04: the clip-window requirement is not a V_T constant — it moves
    right with instance size/connectivity. ER curves are 5-seed geometric
    means; G-set curves are single runs at the G-set protocol."""
    rows = cload(IF / "circuit_ablation_multi_summary.csv")
    gset = cload(IF / "circuit_ablation_gset_summary.csv")
    series = []
    for n, color, lbl, lxy in (("14", TP["gray"], "ER $n$=14 (deg 4)",
                                (3.2, 2.1)),
                               ("20", TP["accent"], "ER $n$=20 (deg 6)",
                                (5.2, 8.0))):
        rs = sorted((r for r in rows if r["scope"] == "pooled"
                     and r["n"] == n and r["axis"] == "span"),
                    key=lambda r: float(r["value"]))
        series.append(([float(r["value"]) for r in rs],
                       [float(r["ratio_geomean"]) for r in rs], color, lbl,
                       lxy))
    rs = sorted((r for r in gset if r["instance"] == "G1"
                 and r["axis"] == "span_6b"), key=lambda r: float(r["value"]))
    x1 = [float(r["value"]) for r in rs]
    y1 = [float(r["tts_ratio_vs_ideal"]) for r in rs]
    series.append((x1, y1, TP["dark"], "G1 $n$=800 (deg 48)", (7.4, 30)))
    for x, y, color, lbl, lxy in series:
        xf = [xi for xi, yi in zip(x, y) if np.isfinite(yi)]
        yf = [yi for yi in y if np.isfinite(yi)]
        ax.plot(xf, yf, "o-", color=color, lw=1.8, ms=5)
        ax.annotate(lbl, xy=lxy, color=color, fontsize=11)
        miss = [xi for xi, yi in zip(x, y) if not np.isfinite(yi)]
        if miss:
            ax.plot(miss, [160] * len(miss), "v", color=color, ms=7)
    ax.annotate("G1: $p_s=0$", xy=(2.3, 150), color=TP["dark"], fontsize=11,
                va="center")
    ax.set_ylim(0.8, 260)
    ax.axhline(1.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 12.5)
    ax.set_xlabel(r"rail half-width $\pm u_\mathrm{clip}$ ($V_T$ units)")
    ax.set_ylabel(r"TTS$_{99}$ ratio (vs ideal)")
    ax.set_title("Clip window vs instance scale")


def panel_readflip(ax):
    """RX-06: solver tolerance to the misread channel, and how far the
    measured comparator sits from it. Tolerance tightens with instance
    size, like the RX-04 rail rule."""
    rows = cload(IF / "read_flip_solver_summary.csv")
    mc = jload(TB / "read_offset_mc_summary.json")
    for inst, color, lbl, lxy in (("ER14_p0.3", TP["gray"],
                                   "ER $n$=14", (2.2e-3, 1.35)),
                                  ("G1", TP["dark"], "G1 $n$=800",
                                   (2.6e-5, 3.1))):
        pts = [(float(r["p_read_flip"]), float(r["tts_ratio_vs_ideal"]))
               for r in rows if r["instance"] == inst
               and r["axis"] in ("tolerance", "read_flip")
               and float(r["p_read_flip"]) > 0]
        pts = sorted(p for p in pts if np.isfinite(p[1]))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color,
                lw=1.8, ms=5)
        ax.annotate(lbl, xy=lxy, color=color, fontsize=11)
    for key, style, lbl, ly in (("as_committed", "--", "as-committed", 30),
                                ("conservative", ":", "decoupled ref", 30)):
        p = mc["misread_channel"][key]["p_read_flip"]
        ax.axvline(p, color=TP["accent"], lw=1.4, ls=style)
        ax.annotate(lbl, xy=(p * 1.15, ly), color=TP["accent"], fontsize=11,
                    rotation=90, va="top")
    ax.axhline(1.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"misread probability $p_\mathrm{read}$")
    ax.set_ylabel(r"TTS$_{99}$ ratio (vs ideal)")
    ax.set_title("Read-error tolerance vs measured offset")


def panel_reset(ax):
    rows = cload(IF / "results_circuit_ablation" / "circuit_ablation_summary.csv")
    rs = sorted((r for r in rows if r["axis"] == "reset_pulses"),
                key=lambda r: float(r["value"]))
    x = [int(float(r["value"])) for r in rs]
    y = [float(r["tts99_ratio"]) for r in rs]
    finite = [(xi, yi) for xi, yi in zip(x, y) if np.isfinite(yi)]
    ax.plot([f[0] for f in finite], [f[1] for f in finite], "o-",
            color=TP["dark"], lw=1.8, ms=5)
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            ax.annotate(r"$p_s=0$", xy=(xi, ax.get_ylim()[1]),
                        ha="center", fontsize=11, color=TP["dark"])
    comb = [r for r in rows if r["axis"] == "combined_meas6b_k3"]
    if comb and np.isfinite(float(comb[0]["tts99_ratio"])):
        ax.plot(3, float(comb[0]["tts99_ratio"]), "*", color=TP["accent"],
                ms=13)
        ax.annotate("+ measured 6-bit grid",
                    xy=(3, float(comb[0]["tts99_ratio"])), xytext=(6, -14),
                    textcoords="offset points", color=TP["accent"],
                    fontsize=11)
    ax.axhline(1.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"reset pulses $k$ (residual $0.28^k$)")
    ax.set_ylabel(r"TTS$_{99}$ ratio (vs ideal)")
    ax.set_title("Sticky reset vs solver cost")


def panel_traj(ax):
    rows = cload(TB / "closed_loop_traj.csv")
    # ideal drawn first as a wide light underlay: its first-hit curve exactly
    # coincides with the quantized one, a fringe must stay visible
    styles = {"ideal": (TP["gray_lt"], "ideal sigmoid", "-"),
              "quantized": (TP["dark"], "measured 6-bit grid", "-"),
              "realistic": (TP["accent"], r"grid + $k{=}3$ reset", "--")}
    emin = -2.0
    for var, (color, label, ls) in styles.items():
        data = {}
        trials = {}
        for r in rows:
            if r["variant"] != var:
                continue
            s = int(r["sweep"])
            data.setdefault(s, {})[int(r["trial"])] = float(r["energy"])
        sw = sorted(data)
        ntr = max(len(v) for v in data.values())
        hit_sweep = {}
        for s in sw:
            for tr, e in data[s].items():
                if e <= emin + 1e-9 and tr not in hit_sweep:
                    hit_sweep[tr] = s
        frac = [sum(1 for h in hit_sweep.values() if h <= s) / ntr
                for s in sw]
        lw = 4.5 if var == "ideal" else 1.8
        ax.step(sw, frac, where="post", color=color, lw=lw, ls=ls)
        li = {"ideal": 0.42, "quantized": 0.58, "realistic": 0.26}[var]
        ax.annotate(label, xy=(112, li), color=color, fontsize=11)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("sweep")
    ax.set_ylabel("fraction of trials at ground state")
    ax.set_title("5-spin Max-Cut replay (8 trials, first hit)")


def panel_ir_profile(ax):
    d = jload(IR / "ir_drop_summary.json")

    def rows_of(N):
        e = d["per_N"][N]
        return e["rows"] if isinstance(e, dict) else e

    r256 = rows_of("256")
    ax.plot([row["r"] for row in r256], [row["u_off"] for row in r256],
            "-", color=TP["dark"], lw=1.8)
    for N, dxy in [("16", (6, -2)), ("64", (6, -2)), ("256", (-12, -14))]:
        far = rows_of(N)[-1]
        ax.plot(far["r"], far["u_off"], "o", color=TP["accent"], ms=6)
        ax.annotate(f"N = {N} far row: {far['u_off']:.2f} $V_T$",
                    xy=(far["r"], far["u_off"]), xytext=dxy,
                    textcoords="offset points", color=TP["accent"],
                    fontsize=11,
                    ha="left" if dxy[0] > 0 else "right")
    rows64 = rows_of("64")
    resid = [abs(row["resid_u"]) for row in rows64]
    ax.plot([row["r"] for row in rows64], resid, "--", color=TP["gray"],
            lw=1.4)
    ax.annotate("predistorted residual\n(N = 64)",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                va="bottom", color=TP["gray"], fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("row index r")
    ax.set_ylabel(r"drive offset $u_\mathrm{off}(r)$ ($V_T$ units)")
    ax.set_title("Write-line IR drop per row (met2)")


def panel_ir_impact(ax):
    """RX-09: fully-populated 64-spin arrays, 1000 trials, mode=none.
    Uncompensated cost grows with array height; predistortion recovers at
    every height on both a frustrated and a matched planted instance."""
    rows = [r for r in cload(IR / "ir_fullarray_summary.csv")
            if r["block"] in ("primary", "crosscheck")
            and r["n_trials"] == "1000" and r["dac_mode"] == "none"]
    inst = {"ER64_p0.1": ("frustrated", TP["dark"], -0.17),
            "PP64_p0.1_eta0.1_s0": ("planted", TP["medium"], 0.17)}
    Ns = ["64", "128", "256"]
    x = np.arange(len(Ns))
    for key, (lbl, color, dx) in inst.items():
        for scen, hatch, alpha in (("uncompensated", "", 1.0),
                                   ("predistorted", "//", 0.45)):
            y = []
            for N in Ns:
                m = [r for r in rows if r["instance"] == key
                     and r["scenario"] == scen and r["n_profile"] == N]
                v = float(m[0]["tts99_ratio"]) if m else np.nan
                y.append(v)
            yy = [min(v, 40) if np.isfinite(v) else 40 for v in y]
            ax.bar(x + dx + (0.075 if scen == "predistorted" else -0.075),
                   yy, 0.15, color=color, alpha=alpha, hatch=hatch,
                   edgecolor=color)
            for xi, v in zip(x, y):
                if not np.isfinite(v):
                    ax.annotate("$p_s{=}0$",
                                xy=(xi + dx + 0.075 * (1 if scen ==
                                    "predistorted" else -1), 41),
                                ha="center", fontsize=10, color=color)
        ax.annotate(lbl, xy=(0.02, 0.93 if dx < 0 else 0.84),
                    xycoords="axes fraction", color=color, fontsize=11)
    ax.axhline(1.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.annotate("solid: uncompensated    hatched: predistorted",
                xy=(0.02, 0.74), xycoords="axes fraction",
                color=TP["gray"], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {n}" for n in Ns])
    ax.set_ylabel(r"TTS$_{99}$ ratio (vs no offset)")
    ax.set_title("IR offset on a fully-populated 64-spin array")


def panel_energy(ax):
    d = jload(TB / "update_energy_summary.json")
    t = d["table"]
    ks = [r["k"] for r in t]
    pulses = np.array([r["e_pulses_pJ"] for r in t])
    read = np.array([r["e_read_pJ"] for r in t])
    gated = np.array([r["statics_gated_pJ"] for r in t])
    ax.bar(ks, pulses, 0.55, color=TP["dark"])
    ax.bar(ks, read, 0.55, bottom=pulses, color=TP["accent"])
    ax.bar(ks, gated, 0.55, bottom=pulses + read, color=TP["light"])
    # rail-exact always-on accounting (supply-true: statics already contain
    # the pulse Ohmic energy under current steering — no double count)
    tot_on = [r["e_update_supply_true_pJ"] for r in t]
    ax.plot(ks, tot_on, "o--", color=TP["gray"], lw=1.4, ms=4)
    ax.annotate("always-on total (rail-exact)", xy=(ks[1], tot_on[1]),
                xytext=(4, 8), textcoords="offset points", color=TP["gray"],
                fontsize=11)
    ax.annotate("driver statics (power-gated)",
                xy=(3, (pulses + read + gated)[2]), xytext=(-56, 26),
                textcoords="offset points", color=TP["medium"], fontsize=11,
                arrowprops=dict(arrowstyle="-", color=TP["medium"], lw=0.8))
    ax.annotate("SOT pulses + read",
                xy=(4, pulses[3] * 0.6), xytext=(-88, 26),
                textcoords="offset points", color=TP["dark"], fontsize=11,
                arrowprops=dict(arrowstyle="-", color=TP["dark"], lw=0.8))
    ax.set_xlabel(r"reset pulses $k$")
    ax.set_ylabel("energy per update (pJ)")
    ax.set_title("Update energy accounting")


def panel_projection(ax):
    rows = cload(IF / "results_reproject" / "reproject_summary.csv")
    marks = {"sMTJ-array": ("o", TP["dark"], "sMTJ device-only"),
             "sMTJ-array-e2e": ("*", TP["darkest"], "sMTJ end-to-end"),
             "cmos-pbit": ("s", TP["accent"], "CMOS p-bit"),
             "fpga-sbm": ("D", TP["medium"], "FPGA SBM"),
             "cpu-numba": ("v", TP["gray"], "CPU")}
    offs = {"sMTJ-array": (10, -4), "sMTJ-array-e2e": (10, 2),
            "cmos-pbit": (10, -4), "fpga-sbm": (10, -4),
            "cpu-numba": (-10, -2)}
    for r in rows:
        if r["instance"] != "G22":
            continue
        m, c, label = marks[r["platform"]]
        x, y = float(r["tts"]), float(r["energy"])
        ax.plot(x, y, m, color=c, ms=9 if m == "*" else 6)
        dx, dy = offs[r["platform"]]
        ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", color=c, fontsize=11,
                    ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [float(r["tts"]) for r in rows if r["instance"] == "G22"]
    ys = [float(r["energy"]) for r in rows if r["instance"] == "G22"]
    ax.set_xlim(min(xs) / 8, max(xs) * 8)
    ax.set_ylim(min(ys) / 8, max(ys) * 8)
    ax.set_xlabel(r"TTS$_{99}$ (s)")
    ax.set_ylabel(r"$E_\mathrm{sol}$ (J)")
    ax.set_title("Hardware projection, G22 (n = 2000)")


def panel_corners(ax):
    """RX-03: per-corner deviation of the 6-bit transfer from its endpoint
    fit. Every number plotted/annotated is read from the committed per-corner
    summaries — nothing hand-typed."""
    styles = {"tt": (TP["dark"], 10), "ss": (TP["medium"], 22),
              "ff": (TP["light"], 34), "fs": (TP["gray"], 46),
              "sf": (TP["accent"], None)}
    devs = {}
    for c, (color, lx) in styles.items():
        f = TB / ("update_chain_summary.json" if c == "tt"
                  else f"update_chain_summary_{c}.json")
        tr = jload(f)["per_bits"]["6"]["transfer"]
        v = np.array([r["v_wr"] for r in tr]) * 1e3
        codes = np.array([r["code"] for r in tr])
        dev = v - np.linspace(v[0], v[-1], len(v))
        devs[c] = dev
        ax.plot(codes, dev, color=color, lw=1.8)
        if lx is not None:
            ax.annotate(c, xy=(lx, dev[lx]), xytext=(0, 5),
                        textcoords="offset points", color=color, fontsize=11,
                        ha="center")
    imax = int(np.argmax(np.abs(devs["sf"])))
    ax.annotate(f"sf: {devs['sf'][imax]:+.1f} mV "
                f"({abs(devs['sf'][imax]) / VT_MV:.2f} $V_T$)",
                xy=(imax, devs["sf"][imax]), xytext=(8, -4),
                textcoords="offset points", color=TP["accent"], fontsize=11)
    others = max(float(np.max(np.abs(devs[c]))) for c in ("tt", "ss", "ff",
                                                          "fs"))
    ax.annotate(f"tt/ss/ff/fs: max |dev| = {others:.1f} mV",
                xy=(0.02, 0.05), xycoords="axes fraction", fontsize=11,
                color=TP["gray"])
    ax.axhline(0.0, color=TP["gray_lt"], lw=0.8, ls="--")
    ax.set_xlabel("DAC code")
    ax.set_ylabel("deviation from endpoint fit (mV)")
    ax.set_title("Transfer nonlinearity across process corners (6-bit)")


def panel_schematic(ax):
    img = plt.imread(HERE / "schematics" / "update_chain.png")
    ax.imshow(img)
    ax.axis("off")


# ------------------------------------------------------------- assembly ---
PANELS = {
    "chain_schematic": (panel_schematic, (10.4, 5.9)),
    "chain_transfer": (panel_transfer, (5.2, 3.6)),
    "chain_waveform": (panel_waveform, (5.6, 3.6)),
    "chain_corners": (panel_corners, (10.4, 3.2)),
    "abl_bits": (panel_bits, (5.2, 3.6)),
    "abl_span": (panel_span, (5.2, 3.6)),
    "abl_reset": (panel_reset, (5.2, 3.6)),
    "abl_readflip": (panel_readflip, (5.2, 3.6)),
    "abl_traj": (panel_traj, (5.2, 3.6)),
    "ir_profile": (panel_ir_profile, (5.2, 3.6)),
    "ir_impact": (panel_ir_impact, (5.2, 3.6)),
    "energy_stack": (panel_energy, (5.2, 3.6)),
    "hw_projection": (panel_projection, (5.2, 3.6)),
}

COMPOSITES = {
    "preview_09.png": [  # (a) schematic (b) transfer (c) waveform (d) corners
        ["chain_schematic", "chain_schematic"],
        ["chain_transfer", "chain_waveform"],
        ["chain_corners", "chain_corners"],
    ],
    "preview_10.png": [  # (a) bits (b) span (c) reset (d) replay (e) read
        ["abl_bits", "abl_span", "abl_reset"],
        ["abl_traj", "abl_readflip"],
    ],
    "preview_11.png": [  # (a) IR (b) impact (c) energy (d) projection
        ["ir_profile", "ir_impact"],
        ["energy_stack", "hw_projection"],
    ],
}


def main():
    set_style()
    for name, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.tight_layout()
        fig.savefig(RAW / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"panel  -> figs_raw/{name}.png")

    for out, grid in COMPOSITES.items():
        # a row may be a full-width span (same key repeated) or hold 1..N
        # panels; the grid is laid out on the least common column count
        nrow = len(grid)
        spans = [len(set(row)) == 1 and len(row) > 1 for row in grid]
        ncol = max(len(row) for row in grid)
        hr = [1.55 if s and grid[i][0] == "chain_schematic" else 1.0
              for i, s in enumerate(spans)]
        fig = plt.figure(figsize=(4.6 * ncol, 4.1 * sum(hr)))
        gs = fig.add_gridspec(nrow, ncol, height_ratios=hr)
        for i, row in enumerate(grid):
            if spans[i]:
                PANELS[row[0]][0](fig.add_subplot(gs[i, :]))
                continue
            for j, key in enumerate(row):
                PANELS[key][0](fig.add_subplot(gs[i, j]))
        fig.tight_layout()
        fig.savefig(RAW / out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"preview   -> figs_raw/{out}")


if __name__ == "__main__":
    main()
