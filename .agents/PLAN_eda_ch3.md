# Plan: ch3 circuit/EDA extension (new §3.5, old §3.5 → §3.6)

Date: 2026-07-13. Owner: Claude session e063faf4. User request: ch3 stays at
algorithm-simulator level; introduce real circuit/EDA work, referencing
`04PBNNSim/smtj_pbnn_sim` (its `eda/` workspace + ch4 §4.6 writing template).

## Anchors (must not drift)

- Device: ch2 Device A, P→AP, t_w = 0.75 ns. Committed VA model params
  (eda/models/smtj_sot.va, copied from PBNN): Vth=0.895783 V, VT=0.023414 V
  (=1/beta_s), Delta=4.91, Vc0=0.857 V, tau0=1 ns, Rp=4900, TMR=1.0, Rsot=776.
- Spec anchor for the whole section: probability window V_T = 23.4 mV;
  algorithm drive u = 2*beta*h_eff maps to (V_wr − Vth)/V_T.
- Solver-side comparability: quantization/offset ablations reuse the §3.4.2
  fixed instance (14-spin ER p=0.30, T=2000, beta 0.1→5, 200 trials/pt,
  block update mode, master seed 2024, exact E_min=−6.878, ideal ps≈0.19).
- Hardware projection today (tab 3.8): sMTJ-array t=0.75 ns, e=0.78 pJ (device
  only), N∥=64 — the new section grounds/extends these with end-to-end numbers.
- Toolchain: WSL `Ubuntu-24.04-EDA` (ngspice-46 /usr/local/bin, sky130A at
  /opt/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice, openvaf, magic, xschem,
  klayout, netgen). Run pattern: `wsl -d Ubuntu-24.04-EDA -- bash -lc
  'cd <repo>; python3 eda/...'`; `.spiceinit` with relative osdi path (repo
  path contains spaces). Windows-native ngspice-46 only for device-only decks.

## Work items

- W1 scaffold: eda/{README.md,models/smtj_sot.va,testbenches/} + osdi compile
  + smoke (osdi+sky130 co-load).
- W2 update_chain_dc.py: 6-bit resistor-string write-DAC (PBNN winning
  topology) → row driver → SOT branch; 64-code DC transfer with loading;
  outputs LSB/VT, span in u units, INL → update_chain_summary.json.
- W3 update_energy.py: transient one full update (DAC set, 0.75 ns write
  pulse, StrongARM read strobe); energy split device/driver/DAC/read;
  t_update decomposition → update_energy_summary.json.
- W4 quantized backend + run_dac_ablation.py: spin backend "dac_quantized"
  using W2 measured transfer, parametric bits {4,5,6,8} × span {±2,±4,±6} VT;
  TTS ratio vs ideal on §3.4.2 instance → dac_ablation_summary.csv.
- W5 writeline IR: port 3-step flow (klayout straps → magic extresist →
  analyze) for Ising tile column N∈{16,64,256}; ΔV → per-row static drive
  offset u_off = ΔV/VT → feed into behavioral backend as per-spin offsets →
  TTS impact; port per-row predistortion → residual. Ties to §3.4.2 h_off
  sensitivity (2nd most sensitive knob).
- W6 closed_loop_maxcut5.py: §3.2 5-spin instance, T=200, 8 trials, same seed
  protocol, measured-transfer-in-the-loop Bernoulli replay vs ideal behavioral
  trajectories + one true ngspice transient waveform of a few update cycles.
- W7 projection upgrade: rerun bench_hardware_compare with end-to-end
  e_update into results_rerun/results_hw_compare_e2e; quantify shift; old
  device-only numbers recorded in a process footnote.
- W8 xschem schematic of the update chain, headless SVG export (WSL).
- W9 figures: clean panels (plot_style.py, English/Arial, no panel letters,
  no fig numbers) into eda/figs_raw/; provisional composites (no letters)
  into article/figs/Chapter03_local_09..11.png pending user PPT pass.
- W10 writing: new §3.5 (spec anchor → update chain → DAC-bits ablation →
  IR/predistortion → energy accounting + projection correction → closed-loop
  replay → methodology-caveat paragraph); renumber old 3.5→3.6; update chapter
  intro (line 5) and handoff; new figs 3.9+, tables 3.9+; refs add Yoon EDL.

## Integrity protocol (from PBNN design_survey README)

- Every number in prose ⇐ committed script + committed *_summary.json; no
  hand-typed values. Seeds recorded in JSON. MEASURED vs ANALYTIC labels.
- RNG stays in Python harness; VA model is deliberately deterministic.
- Absolute sky130 numbers are schematic-level; headline claims as ratios
  normalized to V_T where possible.
- Trial-and-error log: .agents/TRIAL_LOG_eda.md — record REAL failures during
  this build; only these may ground new [^process-*] footnotes. Never invent.

## Verified new references (checked 2026-07-13 via NIST/arXiv/DOI)

- Cacoilo, Yoon, Madhavan, McClelland, Kanai, Fukami, Borders,
  "130-nm CMOS-Integrated Superparamagnetic Tunnel Junction-Based p-bit",
  IEEE Electron Device Letters 47(7), 2026. doi:10.1109/LED.2026.3696800
  (arXiv:2604.14446 has Yoon as first author and includes Ohno; the
  PUBLISHED EDL author list per NIST omits Ohno and leads with Cacoilo —
  cite the published version, first author Cacoilo.)

## Writing constraints (global + project conventions)

- ch3 precedes ch4 in the thesis: no forward references to ch4 results; the
  sky130 flow is introduced compactly and self-contained here; ch4's
  MAC/readout design space is NOT duplicated (ch3 = update chain + solver
  co-design only). Handoff in §3.6 gains the circuit-flow asset item.
- Figure captions 总分式; explanatory footnotes near paragraph; refs at end
  with DOI; no CJK in figures; de-AIGC prose rules; CJK-EN spacing rules.
