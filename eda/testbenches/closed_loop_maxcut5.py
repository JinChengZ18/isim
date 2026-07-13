#!/usr/bin/env python3
"""W6 — closed-loop replay of the Section 3.2 5-node Max-Cut demo on the measured write chain.

Replicates demo/toy_demo_maxcut5.py EXACTLY (same graph, J = -w/2, 200-sweep geometric anneal
beta 0.1 -> 5.0, 8 trials seeded 2024+k, asynchronous random-permutation Gibbs, and the demo's
sigma(x) = 0.5*(1+tanh(x)) convention), then reruns IDENTICAL per-trial seed streams through
three update rules:

  ideal      p = sigma(2*beta*h_eff)                              (the demo, bit-for-bit;
             cross-checked against demo/out/toy_run.npz when present)
  quantized  u = 2*beta*h_eff clipped+snapped to the MEASURED 6-bit u-grid of the W2 write
             chain (update_chain_summary.json per_bits["6"].transfer[*].u), p = sigma(u_q)
  realistic  quantized + sticky reset with k = 3 reset pulses: before the probabilistic draw,
             a spin at +1 stays +1 with rho = (1-R_RESET)^3 (drawn from the same rng stream)

Part 2 drives one spin's physical chain (BUFFER/TAIL netlist blocks reused verbatim from
update_chain_dc.py) through an ngspice TRANSIENT of 4 consecutive 2-ns update cycles: node
bin replays the MEASURED DAC tap voltages of the 4 codes the quantized solver commanded on
spin 0 during sweeps 1-4 of trial 0; the write-enable TG is gated by a 0.75-ns pulse train;
st replays the harness-sampled outcomes of those 4 updates (stochastic outcomes are the
SEEDED HARNESS DRAWS, NOT spice noise — the RNG never lives in the model). Sanity: v(wr)
flat-tops are compared against fresh DC .op runs of the same codes.

MUST RUN IN WSL:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/closed_loop_maxcut5.py [--smoke]
If the W2 sweep is still running this script POLLS for update_chain_summary.json (default
120 x 30 s); only after the timeout does it fall back to an ideal uniform 64-level grid over
[-4, 4] labeled FALLBACK. Writes closed_loop_summary.json, closed_loop_traj.csv and
closed_loop_wave.csv next to this script.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from _common import HERE, R_RESET, RSOT, VTH, VT, grab, load_wrdata, run_deck, write_summary
from update_chain_dc import BUFFER, TAIL, deck as chain_deck

SUMMARY_IN = HERE / "update_chain_summary.json"

# ---- Section 3.2 instance, mirrored from demo/toy_demo_maxcut5.py (do not edit) -------------
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
N = 5
W = np.zeros((N, N), dtype=float)
for _i, _j in EDGES:
    W[_i, _j] = 1.0
    W[_j, _i] = 1.0
J_MATRIX = -0.5 * W          # J_ij = -w_ij/2, h_i = 0  (Section 3.1.3 mapping)

N_TRIALS = 8
N_SWEEPS = 200
BETA0, BETAF = 0.1, 5.0
SEED_BASE = 2024             # demo trial k uses np.random.default_rng(2024 + k)

K_RESET = 3
RHO_STICKY = (1.0 - R_RESET) ** K_RESET   # P(all 3 reset pulses fail) = 0.28^3

# ---- Part-2 waveform timing [s] -------------------------------------------------------------
T_LEAD = 0.5e-9              # settle before the first cycle
T_CYC = 2.0e-9               # one update cycle
T_RAMP = 0.1e-9              # bin (DAC tap) step ramp at cycle start
T_ON = 1.0e-9                # write-enable rise start, relative to cycle start
T_EDGE = 0.05e-9             # wen/wep/st edge time
T_PULSE = 0.75e-9            # write pulse plateau = calibrated t_p
N_CYCLES = 4


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(x))          # demo convention: sigma(x) = logistic(2x)


def beta_schedule(t, T, beta0, betaf):
    if T <= 0:
        return betaf
    return beta0 * (betaf / beta0) ** (t / T)


def energy(s):
    return 0.5 * sum(s[i] * s[j] for i, j in EDGES)


def enumerate_states():
    energies = np.zeros(2 ** N)
    for k in range(2 ** N):
        s = np.array([+1 if (k >> b) & 1 else -1 for b in range(N)], dtype=np.int8)
        energies[k] = energy(s)
    return energies


def snap_code(u, grid):
    """Nearest measured grid point (implicit clip at both ends). grid must be ascending."""
    j = int(np.searchsorted(grid, u))
    if j <= 0:
        return 0
    if j >= len(grid):
        return len(grid) - 1
    return j if (grid[j] - u) < (u - grid[j - 1]) else j - 1


def anneal(seed, variant, grid, spin0_log=None):
    """One annealing run; rng call order mirrors the demo exactly so that the ideal variant
    reproduces demo/out/toy_run.npz bit-for-bit. The sticky-reset branch of `realistic`
    consumes one extra draw from the SAME stream whenever the visited spin sits at +1."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=N).astype(np.int8)
    energies = [energy(s)]
    for t in range(1, N_SWEEPS + 1):
        beta = beta_schedule(t, N_SWEEPS, BETA0, BETAF)
        for i in rng.permutation(N):
            if variant == "realistic" and s[i] == +1 and rng.random() < RHO_STICKY:
                continue                       # all K_RESET reset pulses failed: stays +1
            h_eff = float(J_MATRIX[i] @ s)
            u = 2.0 * beta * h_eff
            if variant == "ideal":
                code, u_q = -1, u
            else:
                code = snap_code(u, grid)
                u_q = float(grid[code])
            p_plus = sigmoid(u_q)
            draw = rng.random()
            s_new = +1 if draw < p_plus else -1
            if spin0_log is not None and i == 0:
                spin0_log.append(dict(sweep=t, beta=round(beta, 6), u_cmd=round(u, 6),
                                      code=int(code), u_q=round(u_q, 6),
                                      p=round(float(p_plus), 6), draw=round(float(draw), 6),
                                      s_new=int(s_new)))
            s[i] = s_new
        energies.append(energy(s))
    return np.asarray(energies), s.copy()


def load_ugrid(max_polls=120, poll_s=30.0):
    """MEASURED 6-bit u-grid from the W2 sweep; poll while that sweep is still running."""
    for attempt in range(max_polls):
        if SUMMARY_IN.exists():
            try:
                js = json.loads(SUMMARY_IN.read_text())
                rows = js["per_bits"]["6"]["transfer"]
                grid = np.array([r["u"] for r in rows], float)
                if len(grid) == 64 and np.all(np.diff(grid) > 0):
                    return grid, "MEASURED", {r["code"]: r for r in rows}
                print(f"[grid] {SUMMARY_IN.name} present but 6-bit grid malformed; polling on")
            except (KeyError, ValueError) as e:
                print(f"[grid] {SUMMARY_IN.name} unreadable ({e}); polling on")
        if attempt < max_polls - 1:
            print(f"[grid] waiting for {SUMMARY_IN.name} (W2 sweep) "
                  f"... poll {attempt + 1}/{max_polls}", flush=True)
            time.sleep(poll_s)
    print("[grid] FALLBACK: ideal uniform 64 levels over [-4, 4] (W2 summary never appeared)")
    return np.linspace(-4.0, 4.0, 64), "FALLBACK", None


def crosscheck_demo(ideal_trajs):
    """Bit-for-bit check of the ideal variant against the committed demo output, if present."""
    npz = HERE.parent.parent / "demo" / "out" / "toy_run.npz"
    if not npz.exists():
        return "demo/out/toy_run.npz not found; cross-check skipped"
    ref = np.load(npz)["trial_energies"]
    ok = ref.shape == ideal_trajs.shape and np.array_equal(ref, ideal_trajs)
    return f"ideal variant vs demo/out/toy_run.npz trial_energies: {'IDENTICAL' if ok else 'MISMATCH'}"


# ---- Part 2: ngspice transient of 4 consecutive update cycles -------------------------------
def _pwl(pts):
    return "PWL(" + " ".join(f"{t:.5e} {v:.6f}" for t, v in pts) + ")"


def build_waveform_deck(taps, st_out):
    """taps: 4 measured DAC tap voltages; st_out: 4 harness-sampled outcomes in {0,1}."""
    t0 = [T_LEAD + k * T_CYC for k in range(N_CYCLES)]
    bin_pts = [(0.0, taps[0])]
    for k in range(1, N_CYCLES):
        bin_pts += [(t0[k], taps[k - 1]), (t0[k] + T_RAMP, taps[k])]
    wen_pts, wep_pts = [(0.0, 0.0)], [(0.0, 1.8)]
    for tk in t0:
        a = tk + T_ON
        wen_pts += [(a, 0.0), (a + T_EDGE, 1.8), (a + T_EDGE + T_PULSE, 1.8),
                    (a + 2 * T_EDGE + T_PULSE, 0.0)]
        wep_pts += [(a, 1.8), (a + T_EDGE, 0.0), (a + T_EDGE + T_PULSE, 0.0),
                    (a + 2 * T_EDGE + T_PULSE, 1.8)]
    st_pts, prev = [(0.0, 0.0)], 0.0
    for tk, out in zip(t0, st_out):
        if prev != 0.0:                       # reset phase returns the device to P (st = 0)
            st_pts += [(tk + 0.10e-9, prev), (tk + 0.15e-9, 0.0)]
        b = tk + T_ON + 2 * T_EDGE + T_PULSE  # after the write pulse has fully fallen
        st_pts += [(b, 0.0), (b + T_EDGE, float(out))]
        prev = float(out)

    dev = TAIL.split(".control")[0]
    dev = dev.replace("Vwen wen 0 1.8", "Vwen wen 0 " + _pwl(wen_pts))
    dev = dev.replace("Vwep wep 0 0", "Vwep wep 0 " + _pwl(wep_pts))
    dev = dev.replace("Vst  st 0 0", "Vst st 0 " + _pwl(st_pts))
    assert dev.count("PWL(") == 3, "TAIL template drifted; PWL substitution failed"

    # W6 design correction (.agents/TRIAL_LOG_eda.md, 2026-07-13): with the enable TG off the
    # wr-node feedback loop is OPEN (wr sags to 0 through Rsot) and drv rails to vdd; the
    # buffer cannot recover within the 0.75-ns pulse (measured flat-tops ~ +0.65 V vs DC).
    # Fix: a complementary-gated 8-finger replica TG + 776-ohm replica load parks the loop
    # between pulses, and a 1-finger feedback mux hands the buffer input pair between wrr
    # (idle) and the real wr node (pulse) — so during the plateau the topology is exactly the
    # DC-measured W2 chain.
    buf = BUFFER.format(inp="bin").replace("XM2  bd2 wr", "XM2  bd2 fb")
    assert "bd2 fb" in buf, "BUFFER template drifted; feedback-mux substitution failed"
    keeper = "Rrep wrr 0 776\n"
    for f in range(1, 9):
        keeper += (f"XRTn{f} drv wep wrr 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
                   f"XRTp{f} drv wen wrr vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n")
    keeper += ("XFBwn wr wen fb 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
               "XFBwp wr wep fb vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n"
               "XFBhn wrr wep fb 0 sky130_fd_pr__nfet_01v8 W=4 L=0.15\n"
               "XFBhp wrr wen fb vdd sky130_fd_pr__pfet_01v8 W=8 L=0.15\n")

    t_stop = T_LEAD + N_CYCLES * T_CYC
    return (f"* W6 closed-loop waveform: {N_CYCLES} consecutive 2-ns update cycles, one chain\n"
            f".lib /opt/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice tt\n"
            f"Vdd vdd 0 1.8\n"
            f"Vbin bin 0 " + _pwl(bin_pts) + "\n"
            + buf + keeper + dev +
            f".control\n"
            f"  tran 0.002n {t_stop * 1e9:.3f}n\n"
            f"  wrdata closed_loop_wave.csv v(bin) v(drv) v(wr) v(pswn) v(st)\n"
            f"  quit\n"
            f".endc\n"
            f".end\n")


def dc_reference(code):
    """Fresh MEASURED DC .op of the full W2 chain at `code` (6-bit); own tag, so it never
    collides with a still-running update_chain_dc sweep."""
    out = run_deck(chain_deck(code, 6), "clw_dcref")
    return dict(code=code, v_tap=grab(out, "bin"), v_wr=grab(out, "wr"),
                v_drv=grab(out, "drv"), psw=grab(out, "pswn"))


def run_waveform(spin0_log):
    picks = spin0_log[:N_CYCLES]              # spin-0 updates of sweeps 1..4, trial 0
    refs = {}
    for e in picks:
        if e["code"] not in refs:
            refs[e["code"]] = dc_reference(e["code"])
    taps = [refs[e["code"]]["v_tap"] for e in picks]
    st_out = [1 if e["s_new"] == +1 else 0 for e in picks]
    run_deck(build_waveform_deck(taps, st_out), "clw_tran")
    t, (v_bin, v_drv, v_wr, v_psw, v_st) = load_wrdata(HERE / "closed_loop_wave.csv", 5)

    cycles = []
    for k, e in enumerate(picks):
        tk = T_LEAD + k * T_CYC
        a, b = tk + T_ON + T_EDGE + T_PULSE - 0.23e-9, tk + T_ON + T_EDGE + T_PULSE - 0.02e-9
        mtop = (t >= a) & (t <= b)            # last ~0.2 ns of the plateau
        mpul = (t >= tk + T_ON) & (t <= tk + T_ON + 2 * T_EDGE + T_PULSE + 0.05e-9)
        r = refs[e["code"]]
        top = float(np.mean(v_wr[mtop]))
        cycles.append(dict(
            cycle=k, sweep=e["sweep"], u_cmd=e["u_cmd"], code=e["code"], u_q=e["u_q"],
            st_outcome=st_out[k], v_tap_dc=round(r["v_tap"], 6), v_wr_dc=round(r["v_wr"], 6),
            psw_dc=round(r["psw"], 6), flattop_mean_V=round(top, 6),
            delta_mV=round((top - r["v_wr"]) * 1e3, 3),
            peak_wr_V=round(float(np.max(v_wr[mpul])), 6),
            psw_flattop=round(float(np.mean(v_psw[mtop])), 6)))
    return cycles, refs


def main():
    smoke = "--smoke" in sys.argv
    emin_grid = enumerate_states()
    e_min = float(emin_grid.min())
    degeneracy = int((emin_grid == e_min).sum())
    assert e_min == -2.0, f"enumeration says E_min = {e_min}, expected -2 exactly"
    print(f"enumeration: 2^5 = 32 states, E_min = {e_min:+.1f} (degeneracy {degeneracy})")

    if smoke:
        grid, grid_label = np.linspace(-4.0, 4.0, 64), "FALLBACK(smoke)"
        rows6 = None
    else:
        grid, grid_label, rows6 = load_ugrid()
    print(f"u-grid [{grid_label}]: 64 levels, u in [{grid[0]:+.4f}, {grid[-1]:+.4f}], "
          f"sticky rho = (1-{R_RESET})^{K_RESET} = {RHO_STICKY:.6f}")

    variants, trajs, spin0_log = {}, {}, []
    for name in ("ideal", "quantized", "realistic"):
        tr, finals, first_hit = [], [], []
        for k in range(N_TRIALS):
            log = spin0_log if (name == "quantized" and k == 0) else None
            e_traj, _ = anneal(SEED_BASE + k, name, grid, log)
            tr.append(e_traj)
            finals.append(float(e_traj[-1]))
            hits_at = np.nonzero(e_traj == e_min)[0]
            first_hit.append(int(hits_at[0]) if len(hits_at) else None)
        tr = np.stack(tr)
        trajs[name] = tr
        n_hits = int(sum(1 for f in finals if f == e_min))
        variants[name] = dict(final_energies=finals, ground_state_hits=n_hits,
                              hit_rate=round(n_hits / N_TRIALS, 4),
                              mean_final_energy=round(float(np.mean(finals)), 4),
                              first_sweep_at_Emin=first_hit)
        print(f"[{name:9s}] hits = {n_hits}/{N_TRIALS}  finals = {finals}")

    check = crosscheck_demo(trajs["ideal"])
    print(f"cross-check: {check}")

    if smoke:
        print("[smoke] replay only; no files written, no ngspice run")
        return

    csv = HERE / "closed_loop_traj.csv"
    with csv.open("w") as f:
        f.write("variant,trial,sweep,energy\n")
        for name in trajs:
            for k in range(N_TRIALS):
                for t, e in enumerate(trajs[name][k]):
                    f.write(f"{name},{k},{t},{e:.1f}\n")
    print(f"  -> {csv.name}")

    print(f"waveform: replaying spin-0 updates of sweeps "
          f"{[e['sweep'] for e in spin0_log[:N_CYCLES]]} (trial 0, quantized variant)")
    cycles, refs = run_waveform(spin0_log)
    for c in cycles:
        print(f"  cycle {c['cycle']}: sweep {c['sweep']} code {c['code']:2d} "
              f"u_cmd {c['u_cmd']:+.3f} -> u_q {c['u_q']:+.3f}  st -> {c['st_outcome']}  "
              f"flat-top {c['flattop_mean_V']:.4f} V vs DC {c['v_wr_dc']:.4f} V "
              f"(delta {c['delta_mV']:+.2f} mV, peak {c['peak_wr_V']:.3f} V)")

    write_summary(HERE / "closed_loop_summary.json", dict(
        _label=("Part 1 replay: ANALYTIC solver loop on the MEASURED 6-bit u-grid "
                f"({grid_label}); Part 2 waveform: MEASURED, ngspice tran, sky130 tt, "
                "schematic-level, committed OSDI device; stochastic outcomes on node st are "
                "the seeded harness draws, NOT spice noise"),
        instance=dict(edges=EDGES, mapping="J_ij = -w_ij/2, h_i = 0", n_spins=N,
                      n_sweeps=N_SWEEPS, beta_schedule=f"geometric {BETA0} -> {BETAF}",
                      update="asynchronous random-permutation Gibbs (mirrors demo)",
                      sigma="0.5*(1+tanh(x)), demo convention", n_trials=N_TRIALS,
                      seeds=[SEED_BASE + k for k in range(N_TRIALS)],
                      seed_protocol=("per-trial np.random.default_rng(2024+k) exactly as "
                                     "demo/toy_demo_maxcut5.py; all three variants restart "
                                     "the identical stream per trial"),
                      demo_crosscheck=check),
        enumeration=dict(label="ANALYTIC (in-script full enumeration of 2^5 states)",
                         E_min=e_min, degeneracy=degeneracy),
        u_grid=dict(label=grid_label, n_levels=int(len(grid)),
                    u_min=round(float(grid[0]), 5), u_max=round(float(grid[-1]), 5),
                    source=(str(SUMMARY_IN.name) + " per_bits[6].transfer[*].u"
                            if grid_label == "MEASURED" else "np.linspace(-4, 4, 64)")),
        sticky_reset=dict(label="ANALYTIC (committed formula)", k_pulses=K_RESET,
                          r_reset_single=R_RESET, rho=RHO_STICKY,
                          rule="if s_prev = +1: stays +1 with prob rho, before the draw"),
        variants=variants,
        spin0_trace_trial0_quantized=spin0_log[:12],
        waveform=dict(label="MEASURED (ngspice tran 2 ps step; DC refs = fresh .op per code)",
                      cycle_ns=T_CYC * 1e9, pulse_plateau_ns=T_PULSE * 1e9,
                      lead_ns=T_LEAD * 1e9, enable_on_offset_ns=T_ON * 1e9,
                      picked_from="trial 0, quantized variant, spin 0, sweeps 1-4",
                      flattop_window="last 0.21 ns of each enable plateau",
                      cycles=cycles, wave_csv="closed_loop_wave.csv"),
    ))


if __name__ == "__main__":
    main()
