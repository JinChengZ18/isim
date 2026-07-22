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

## 2026-07-20 RX-04: the ±6V_T rail rule and the "2-bit is better" reading were both single-instance artifacts

Tried: generalizing the §3.5.2 circuit-constraint sweeps (bits / rail span / reset-k), which had
been calibrated on one 14-spin ER instance with one instance-seed, to five ER seeds x n in {14,20}
and to the G-set instances G1/G14/G22 (new drivers run_circuit_ablation_multi.py and
run_circuit_ablation_gset.py).
Symptom 1 (rail span): the ±6V_T saturation point does not transfer. Pooled over five ER seeds the
cost at ±6V_T is 1.38x (n=14) and 1.59x (n=20) — already not saturated — and on G1 (n=800, mean
degree 47.9) ±6V_T degrades 63x, ±8V_T still 3.0x, with ±10V_T the first workable point (p_s 0.635
vs baseline 0.720). G22 (n=2000) has zero hits in 200 trials at ±8V_T and below. The published rule
was therefore ~2 orders of magnitude optimistic at array scale.
Symptom 2 (bit width): the canonical run's "2-bit is better than 8-bit" (1.36x vs 2.30x) is a
seed-0 artifact — per-seed 2-bit ratios are 1.36 / 5.83 / 6.52 / 2.37 / 3.51, geometric mean 3.36x,
i.e. WORSE than 8-bit (3.09x).
Confound found while checking: the G-set bits axis had been run at span=4, where every bit width
fails on G1/G22 (p_s=0) because the clip alone is fatal — bit width cannot be assessed there. Added
a bits sweep at span=10 on G1: 4/6/8 bit give p_s 0.610/0.635/0.620, Wilson intervals overlapping.
Fix: §3.5.2 rewritten — the rail-span rule is restated as size-dependent (with the mechanism: the
clip caps single-update certainty at sigma(u_clip) and anti-field flips accumulate with N*T), the
2-bit claim is deleted, and the design conclusion is inverted to "trade resolution for range"
(±10V_T at 4 bit beats ±4V_T at 6 bit by an unbounded margin at G-set scale). §3.6 updated.

## 2026-07-21 RX-04 (device knobs): the §3.4.2 sensitivity ranking reorders in its middle at array scale

Tried: rerunning the five-knob behavioural ablation (bench_device_ablation.py, already wired for
G-set mode but never exercised for the thesis) on G1 (n=800, mean degree 47.9, G-set protocol
T=1e4 / beta_f=10 / 200 trials / block mode), to test whether the ER14-derived ranking transfers.
Result: the two ends hold, the middle reorders. p_max stays the dominant threat and gets WORSE with
scale (0.9 -> 0/200 hits on G1, vs 10x on ER14; energy median -2036 -> -1171.5); sigma_C2C stays the
most tolerated (sigma=2 -> 1.04x, inside the baseline Wilson interval). But CV(Delta) rises to the
number-two constraint (CV=0.30 -> 1.66x with non-overlapping intervals on G1, vs an unremarkable
1.3x on ER14), while h_off's ER14 catastrophe does NOT reproduce (h_off=0.2 -> 40x on ER14 but
1.25x, statistically unresolved, on G1) and the g_dev>1 acceleration does not transfer either
(1.5x gain -> 0.62x on ER14, 1.16x and unresolved on G1).
Diagnosis (corrected 2026-07-21 after an audit caught the first version): the original diagnosis
blamed max|J| units and was WRONG — neither run normalizes J (load_gset/random_er_maxcut default
normalize=False), and max|J| is 0.5 on G1 versus 1.125 on ER14, so in those units the G1
perturbation is the LARGER one. The actual driver is the local field itself: h_off is an absolute
additive field of the same dimension as h_i, and the median |h_eff| under random configurations
grows from 0.53 (ER14) to 2.50 (G1), so the same absolute offset falls from 37% to 8% of the
typical local field. g_dev is dimensionless and needs its own account (both sides weaken on G1:
0.5 -> 1.37x, 1.5 -> 1.16x); only p_max and CV(Delta) act size-independently. Fix: §3.4.2 gained a generalization paragraph and §3.6's
device-priority sentence now states the invariant ends plus the degree-dependent middle.

## 2026-07-22 RX-09: the planted 64-spin instance had to be re-tuned before it had a known optimum

Tried: building the RX-09 fully-populated array instance as a planted-partition Max-Cut at
n=64 (balanced random partition, edges w.p. 0.10, |N(0,1)| magnitudes, sign rewarding the
planting, then a fraction eta of signs flipped to create frustration), so that the optimum
would be known by construction and no long-run reference target would be needed.
Symptom: at the first-choice frustration levels the planting is simply NOT the optimum. A
200-trial ideal run at T=20000 beats the planted energy by 1.23 (eta=0.15), 3.84 (eta=0.20)
and 13.56 (eta=0.30) energy units; using the planted energy as the target would have
reported p_s=0 for every arm against a target the solver had already passed.
Diagnosis: frustration and target certainty trade off directly — a planted state stays
optimal only while the flipped edges are too sparse to pay for a domain wall, and the
unfrustrated limit (eta=0) is gauge-equivalent to a ferromagnet, i.e. trivially solved
(p_s -> 1) and useless for ratio measurement.
Fix: the frustration level became a calibrated parameter rather than a free choice.
eta=0.10 is the largest level that survives verification (long-run min == planted energy to
the last digit, reached by 91/200 reference trials) while leaving the ideal baseline at
p_s=0.465, i.e. away from both 0 and 1 where the TTS ratios have resolution. The verification
is now an assertion inside eda/extraction/writeline_ir/ir_fullarray_impact.py (the run aborts
if a long ideal run ever goes below the planted energy), and the whole scenario grid is
repeated on ER64_p0.1 under the RX-05c LONGRUN_BEST convention so no conclusion depends on
the planted target alone.

## 2026-07-22 RX-09: the N=128 IR profile did not exist, and the README's N=256 code count is wrong

Tried: reading the N in {64, 128, 256} per-row offset profiles out of the committed
eda/extraction/writeline_ir/ir_drop_summary.json.
Symptom: there is no N=128 entry — analyze_ir.py's N_LIST is (16, 64, 256), so only three
profiles were ever written, and 128 was assumed to exist.
Fix: rather than re-running the extraction flow and mutating a committed artifact that §3.5.3
already cites, the driver imports analyze_ir.per_row_profile and calls it with the sheet
resistance and DAC LSB read back from the committed provenance block. N=64 and N=256 are then
asserted row-for-row equal to the committed JSON at run time (they are), and N=128 is labeled
ANALYTIC EXTENSION of the same committed formula rather than a new measurement.
Found while checking: writeline_ir/README.md's result table states 43/63 compensation codes
for N=256; the committed JSON and the formula both give 41/63 (round(126.84/3.0767) = 41).
The README's residual column (0.063 u for every N) is likewise slightly off from the JSON
(0.0648/0.0655 u) — §3.5.3's 0.066 V_T is the correct one. The README numbers need fixing;
no thesis number depends on them.

## 2026-07-22 RX-06: the read comparator's real decision margin is not the divider margin — and it is undamped-node artefact in both directions

Tried: extracting the §3.5.4 read path (0.2 V rail, Rref = 7350 Ω, PMOS-input StrongARM) from
update_energy.py into a standalone Pelgrom mismatch Monte-Carlo, on the assumption that the static
divider margins (−20.0 mV at P, +14.3 mV at AP against the 0.100 V midpoint) are the margins the
comparator actually has to beat.
Symptom: at zero mismatch the real deck does not flip at those values. Sweeping a trim source in
series with the sense input, the decision flips at +67 mV (st=0) and −43 mV (st=1) — 3x the static
margins, and asymmetric between states. Diagnosis: the sense node is resistively terminated with a
STATE-DEPENDENT Thevenin impedance (Rref‖Rp = 2940 Ω at P, Rref‖Rap = 4200 Ω at AP) against a fixed
Rref/2 = 3675 Ω reference, and neither node carries any capacitance in the committed schematic. The
comparator's kickback during regeneration therefore displaces the two inputs unequally, and because
the higher-impedance state is also the higher-voltage state, the displacement reinforces the correct
decision. Believing the static margin would have overstated the misread rate by three orders of
magnitude; believing the as-committed threshold would have understated it by the same factor, since
the help is an artefact of leaving both nodes undamped — a real reference would be decoupled.
Fix: the harness measures the zero-mismatch threshold as a function of an added node capacitance
instead of assuming it. The threshold collapses back onto the static margin by 50 fF (+21/−15 mV vs
static 20.0/14.29 mV) and is flat from there to 1 pF, so the damped case is the conservative design
case and both bounds are reported. Running the full N=120 mismatch MC at C_node = 0 and 200 fF then
validated the tail model rather than assuming it: the MEASURED damped misread rates, 18/120 = 0.150
[0.097, 0.225] at P and 30/120 = 0.250 [0.181, 0.334] at AP, bracket the Gaussian-tail predictions
0.132 and 0.232 computed from the independently measured σ_off = 18.53 mV.

Second correction, same run (throughput, no effect on any number): the sky130 `.lib` parse dominates
a one-ngspice-process-per-sample harness, so all samples of an arm were batched into a single session
with `alter` between sweeps. ngspice retains one plot per `tran`; by ~5000 sweep points the session
had grown to 1.18 GB RSS and had not finished 120 samples in 32 minutes. Adding `destroy $curplot`
inside the sweep loop made the same arm 4.9x faster (73 s -> 15 s on the 4-sample smoke) with
bit-identical output (mean −2.000 mV, σ 15.706 mV before and after).
