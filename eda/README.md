# eda/ — circuit-level workspace for the Chapter-3 Ising machine

Transistor-level grounding of the Chapter-3 sMTJ Ising solver: the write
chain that delivers the Gibbs drive to the device, the constraints that chain
imposes on the algorithm (quantization, clipping, reset reliability, IR
drop), and the per-update energy/timing that the Section 3.4.3 hardware
projection rests on. The flow, device model and integrity protocol are
shared with the sibling PBNN workspace (`04PBNNSim/smtj_pbnn_sim/eda`);
everything Ising-specific lives here.

## Toolchain

Open-source only, same as the sibling workspace: ngspice ≥ 43 (native WSL
build), sky130A PDK (`/opt/pdk/sky130A`, tt corner, 1.8 V devices), OpenVAF
(compiles `models/smtj_sot.va` → OSDI), Magic (resistance extraction),
KLayout (strap layout generation), Xschem (schematics). All circuit runs
execute inside WSL `Ubuntu-24.04-EDA`:

```
wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/<script>.py
```

Solver-side feedback scripts (`eda/interface/`) are pure Python and run
anywhere.

## Device anchor

`models/smtj_sot.va` is the Chapter-2-calibrated compact model (Device A,
P→AP, t_p = 0.75 ns): Vth = 895.783 mV, V_T = 23.414 mV (= 1/β_s),
Δ = 4.91, V_c0 = 0.857 V, R_P = 4.9 kΩ, TMR = 1.0, R_SOT = 776 Ω. The
Bernoulli draw and all timing state live in the Python harnesses (RNG never
inside the Verilog-A; every seed is recorded in the output JSON). The spec
anchor used throughout is the probability window V_T: a drive u = 2βh_eff
maps to a write voltage V = Vth + u·V_T.

## Layout

| Path | What it does | Output |
|---|---|---|
| `models/smtj_sot.va` | calibrated compact device (vendored, byte-identical analog core) | `testbenches/smtj_sot.osdi` |
| `testbenches/update_chain_dc.py` | write chain DC transfer: string DAC → TG → Miller buffer → enable TG → SOT branch; 4/5/6/8-bit | `update_chain_summary.json` |
| `testbenches/update_energy.py` | transient energy/timing of one update (k resets + write + read) | `update_energy_summary.json` |
| `testbenches/closed_loop_maxcut5.py` | measured-transfer-in-the-loop replay of the §3.2 demo + waveform | `closed_loop_summary.json`, `closed_loop_traj.csv`, `closed_loop_wave.csv` |
| `interface/circuit_backends.py` | registered solver backend: DAC quantization/clipping, β-scaled rails, sticky reset, static offsets | — |
| `interface/run_circuit_ablation.py` | §3.4.2-protocol ablation of the chain constraints | `results_circuit_ablation/circuit_ablation_summary.csv` |
| `interface/robustness_check.py` | seed-robustness of the two striking ablation findings | `robustness_summary.json` |
| `interface/reproject_hw.py` | §3.4.3 projection re-run with measured end-to-end sMTJ constants | `results_reproject/reproject_summary.csv` |
| `extraction/writeline_ir/` | KLayout strap → Magic extresist → per-row IR offset → solver impact | `ir_drop_summary.json`, `ir_solver_impact.csv` |
| `schematics/` | Xschem schematic of the update chain (headless SVG/PNG export) | `update_chain.svg/.png` |
| `figs_make.py` | clean thesis figure panels + letter-free preview composites from the committed results | `figs_raw/*.png`, `figs_raw/preview_{09..11}.png` |
| `build_ppt_figs.py` | composes the panels into a PPT deck, adds the (a)(b)(c) panel letters and exports the numbered figures via LibreOffice (run after `figs_make.py`) | `article/ppt/Chapter03_local.pptx`, `article/figs/Chapter03_local_{09..11}.png` |

## Integrity protocol

Inherited from the sibling `design_survey` README: every number quoted in
the thesis traces to a committed script plus its committed `*_summary.json`;
JSON blocks carry MEASURED / ANALYTIC / FALLBACK labels; seeds are recorded;
non-converged points are reported as NaN, never patched. Absolute sky130
numbers are schematic-level; headline claims are ratios normalized to V_T
wherever possible. Real build failures are logged in
`.agents/TRIAL_LOG_eda.md` and are the only admissible source for the
thesis's trial-and-error footnotes.
