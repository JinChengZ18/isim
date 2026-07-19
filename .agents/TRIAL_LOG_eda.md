# Trial-and-error log — ch3 EDA build (real events only)

Purpose: ground any new [^process-*] footnote in the thesis. Only append
entries for failures/corrections that actually happened in this build, with
enough detail to verify (command, symptom, fix, commit). Never invent.

Format: `## YYYY-MM-DD <short title>` + what was tried / what broke / fix.

(empty — append as events occur)

## 2026-07-13 W5 writeline_ir: magic extresist refuses to output low-R met2 nets

Tried: direct port of the PBNN writeline flow with a met2-only strap GDS (cal 400 sq
plus n16/n64/n256 tile geometries), `extract do resistance -> extresist -> ext2spice`.
Symptom: `Nets extracted: 8, Nets output: 0` — wl_res.spice contained no resistors;
lowering `extresist tolerance` to 1e-6 changed nothing (met2 straps are 4–64 ohm).
Inspecting the surviving PBNN build dir showed the same behavior there: of their five
layers only the ~19.2 kohm poly strap was ever output (`R0 poly_a poly_b 19183.6`);
their li1/met1–met3 straps were extracted but dropped by the net filter, and the
published met2 numbers came from the techfile values validated by that one poly point.
Fix (design correction, not a typo): added a 400-sq poly calibration strap to the GDS
so the extresist two-port flow check still runs (47.96 vs techfile 48.2 ohm/sq,
-0.50% = the 0.5 um label inset, 398/400 sq), and took the met2 sheet R from the
lumped net resistance that `extract do resistance` writes into the .ext node records
(cal strap 50.0 ohm / 400 sq = 0.1250 ohm/sq, exact vs techfile; n16/n64/n256 all
consistent). analyze_ir.py labels the met2 value MEASURED (.ext lumped R) with the
poly extresist point as the flow validation.

## 2026-07-13 W6 closed_loop_maxcut5: open feedback loop between write pulses rails the buffer

Tried: pulsed transient of the W2 write chain as-is — bin PWL stepping measured DAC taps,
write-enable TG gated by the 0.75-ns pulse train (2-ns cycle), buffer feedback from node wr
(after the enable TG, exactly the DC-swept topology).
Symptom: with the enable TG off, wr sags to 0 through the 776-ohm SOT branch, the input
pair sees fb = 0 V and the two-stage Miller buffer rails drv to vdd; the loop then cannot
recover within the 0.75-ns pulse — measured flat-tops sat +0.60 to +0.69 V above the DC
transfer (v(pswn) pinned at 1.0), i.e. the chain wrote garbage on every cycle.
Fix (design correction, not a typo): keep the loop closed between pulses — added a
complementary-gated 8-finger replica TG into a 776-ohm replica load (wrr) plus a 1-finger
feedback mux that hands the buffer input pair from wrr (idle) to the real wr node (pulse).
During the plateau the topology is exactly the DC-measured W2 chain; flat-top deltas fell
from ~+650 mV to +3.3…+8.0 mV (residual = GBW-limited settling tail inside the 0.75-ns
plateau, visible as a +4…8 mV drift across the plateau in 50-ps window means).

## 2026-07-13 W3 update_energy: same open-loop railing, independently hit; always-on replica rejected for droop

Tried (update_energy.py block [0], kept in the script as the naive reference run): gating the
W2 chain by pulsing wen/wep as prescribed, feedback from wr. Symptom: identical to the W6
entry above (independently measured before seeing it) — buffer rails between pulses, device
sees 1.58–1.61 V for the whole 0.75-ns pulse, E_dev = 2.65 pJ instead of ~0.78 pJ, psw pinned
at 1 for every code.
Intermediate fix tried and REJECTED: always-on replica branch carrying the feedback (loop
never opens, drv pre-settled). Symptom: when the device TG fires, the buffer load current
doubles and the class-A output stage cannot supply the step — drv droops, delivered flat-top
sits 87 mV low (v_wr 0.809–0.817 at mid code); a 10-pF reservoir on drv trades this to a
43-mV in-pulse droop ramp (0.893→0.850) and would wreck code-change settle time (tau_down =
R_load*C_res).
Fix kept (converges with the W6 entry): current-steering — the replica TG is gated
complementarily (conducts exactly when the device branch is off) so the buffer load current
is constant (measured i_vdd 1.593 mA idle vs 1.593 mA in-pulse at mid code), plus the
wr<->wrr feedback-steering mux. Flat top 0.897–0.906 V at mid code (W2 DC value 0.9009),
E_dev_mid = 0.808 pJ. Energy-accounting consequence recorded in update_energy_summary.json:
with steering the rail power already contains the pulse Ohmic energy, so the prescribed
"k*(E_dev+E_tg) + P_buf*t_update" always-on sum double-counts it; the JSON carries a
supply-true accounting alongside.

## 2026-07-20 RX-05a: vendored LLG .osdi is a Windows PE DLL — not loadable by WSL ngspice

Tried: pointing the new reset-correlation harness at the vendored compiled model
`04PBNNSim/smtj_pbnn_sim/eda/vendor/vgsot-sim/va/llg/vgsot_llg.osdi` (the RX-05 spec said
"compiled .osdi exists"). Symptom: the file is a PE32+ Windows DLL built with MSVC link.exe
(its sibling `.exp`/`.spiceinit` embed `E:\EDA\ngspice-46\Spice64\bin\ngspice_con.EXE`);
a Linux ngspice inside Ubuntu-24.04-EDA cannot dlopen it. Fix: the harness recompiles the
`.va` source with the WSL-side OpenVAF into its own workspace
(`eda/testbenches/llg_reset/vgsot_llg.osdi`, ELF x86-64) and treats the vendor repo as
read-only; deterministic sanity deck (AP->P at -0.999 V, 0.75 ns) switches to mz=+0.998,
matching the vendor tb_switch.spice behavior.

## 2026-07-20 RX-05a: SeedSequence.spawn children share .entropy — all "seeded" LLG trials were identical

Tried: deriving per-trial noise streams as `default_rng(SeedSequence(int(child.entropy)))`
with `child = SeedSequence(master).spawn(N)[i]`, mirroring the entropy bookkeeping used for
logging elsewhere. Symptom (caught in the 2-seed pilot before the 1000-seed run): both
"independent" trials returned bit-identical mz trajectories — `.entropy` of a spawned child
is the PARENT's entropy; the child's identity lives in `spawn_key`, which the int() round-trip
discards. A 1000-seed run would have been 1000 copies of one trajectory. Fix: pass the child
SeedSequence objects themselves through the multiprocessing jobs (they pickle fine) and
document seed identity as `SeedSequence(master_seed).spawn(N)[idx]`; pilot re-run confirmed
distinct trajectories (one seed even showing three consecutive reset failures, the other a
back-hop knock-out of an already-switched device).

## 2026-07-13 (W8, xschem schematic) vendored fet symbol renders W blank in "@W/@L" sizing text

Tried: vendored the PBNN hero symbol set (sym/nfet.sym, sym/pfet.sym) unchanged and generated
update_chain.sch with per-instance W=/L= attributes, expecting the grey device-sizing text to
read "20/0.5" etc.
Symptom: exported SVG/PNG showed only the L value ("0.5", "0.15") next to every FET — xschem's
token parser does not stop at "/" directly after "@W", looks up a nonexistent attribute "W/@L",
and substitutes empty. Confirmed latent in the PBNN source figures too (their yoon_pbit_driver.svg
text dump contains only "0.15" strings, no W values), so copying the "known-good" template
silently dropped every device width from the figure.
Fix: patched the local symbol copies' sizing text from "T {@W/@L}" to "T {@W / @L}" (space
terminates the token), giving "20 / 0.5", "80 / 0.5", "4 / 0.15", "8 / 0.15" in the render.
Connectivity separately verified by netlisting the .sch (xschem -n -s) and matching every
device line against the update_chain_dc.py deck (M1-M4/Mt/Mo/Ccm/TG/WE all exact, feedback
gate of M2 = wr after the enable gate).

## 2026-07-19 RX-01: the G22 "3.71x core finding" fails its own CI audit; N=2000 rerun demotes it

Tried: adding Wilson/bootstrap intervals to every p_s table (new stats.py + ci_audit.py) as
table-stakes statistics. Symptom: the chapter's headline algorithmic claim — G22 speedup
3.71x — rested on 4-vs-1 hits at N_trial=200; the sweeps-basis bootstrap interval is
[0.33, 8.14] with 37% undefined replicates, i.e. the claim was never statistically
established. Verification before revision: reran G22 both dynamics at N_trial=2000 with the
same master seed; SeedSequence(2024).spawn(2000) children 0..199 are identical to the
original protocol and the rerun's first 200 trials reproduce the original 4-vs-1 hits and
per-trial energy multisets exactly (no protocol drift — pure upward fluctuation). Result:
p_s 0.004 vs 0.0025 (8 vs 5 hits), sweeps speedup 1.6x CI [0.5, 7.0], straddling 1.
Correction applied: 3.3.1/3.3小结/3.6 rewritten (grounds [^process-g22-power]); the G1
N=1000 control simultaneously showed SA significantly faster (0.84x [0.75, 0.94]),
consistent with the Peskun ordering — the chapter's Gibbs value proposition was moved from
algorithmic speed to physical sampling realization.
