# Reviewer-driven experiment queue for chapter 3 (RX-01..RX-15)

Source: 2026-07-19 two-persona reviewer audit (device/circuit + Ising-machine
algorithm) of chapter03.md incl. the new §3.5. Each item below is a
self-contained spec: a fresh session should be able to execute it without any
other conversation context. Deliverable of every item = committed script(s) +
committed result file(s) + a decision applied to the thesis text per the
item's Decision rule.

## Claim protocol (parallel sessions)

1. Before starting an item, edit its `Status:` line to
   `claimed YYYY-MM-DD <branch-or-session-tag>` and commit that one-line
   change immediately (cheap lock; if two sessions collide, later commit
   rebases and picks another item).
2. On completion set `Status: done YYYY-MM-DD -> <artifact paths>` and add a
   dated line to `.agents/MEMORY.md` changelog.
3. If an item is abandoned, revert to `open` with a note.
4. Never edit another session's claimed item except to read.

## Shared ground rules (apply to every item)

- Repo root = `isim_framework` (this dir's parent). Solver runs on Windows
  python; ngspice/sky130/OSDI runs inside WSL:
  `wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/...`.
- Windows python with `n_jobs>1` REQUIRES the `if __name__ == "__main__":`
  guard (multiprocessing spawn) — see `eda/interface/robustness_check.py`.
- Custom spin backends are exercised ONLY in `update_mode="block"` (JIT path
  bypasses them). Registered kinds: `behavioral_smtj` (device_model.py),
  `circuit_chain` (eda/interface/circuit_backends.py).
- §3.4.2 ablation protocol (the comparability anchor for all feedback runs):
  `random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0)`; SolverConfig
  geometric, beta 0.1→5.0, n_sweeps=2000, block mode; multistart 200 trials,
  master_seed=2024; exact target E_min = -6.877554338927609 (enumeration);
  ideal baseline p_s=0.185, TTS_99=45023.55 sweeps.
- G-set protocol (§3.3 comparability): T=10000 sweeps, beta 0.1→10.0,
  N_trial=200, seeds via SeedSequence(master_seed).spawn. Loader
  `problems.load_gset` (+`fetch_data.ensure_gset`); target energy =
  edge_sum/2 - BKS_cut with BKS: G1=11624 (n=800), G14=3064 (n=800, p_s=0 at
  T=1e4; first hits at T=1e5), G22=13359 (n=2000, ideal-Gibbs p_s=0.02,
  Metropolis 0.005).
- Statistics convention (NEW, introduced by RX-01): every p_s that supports a
  ratio claim gets a Wilson 95% CI; TTS ratios reported with bootstrap CI
  over per-trial outcomes. Helper lands in RX-01 as `eda/interface/stats.py`
  (`wilson(k, n)`, `tts_ratio_ci(...)`) — later items import it.
- Integrity protocol: every prose number <= committed script + committed
  `*_summary.json`/`.csv` with recorded seeds and MEASURED/ANALYTIC/FALLBACK
  labels; non-converged = NaN, never patched. Real failures that force a
  design correction -> dated entry in `.agents/TRIAL_LOG_eda.md` (the only
  admissible source for new [^process-*] footnotes).
- Figures: NEVER write into `article/figs/` directly. Pipeline is
  `python eda/figs_make.py` (clean panels -> `eda/figs_raw/` + previews)
  then `python eda/build_ppt_figs.py` (adds (a)(b)(c) letters via the PPT
  deck + LibreOffice render into `article/figs/`). New panels = add a panel
  fn to figs_make.py + a slot in build_ppt_figs.py.
- Thesis text touchpoints are given per item by SECTION, not by exact string
  (the chapter was de-AIGC'd in commit 0885355; grep for the cited numbers
  to locate current wording). Chapter file:
  `article/chapter03.md` (§3.3 benchmarks; §3.4.2 device ablation; §3.4.3
  projection/表3.8; §3.5.1..3.5.5 circuit section; §3.6 summary).
- Local commits only. NEVER push.

---

## P0 — blocking-grade (defend existing headline claims; cheap)

### RX-01 Statistical power + confidence intervals for all p_s tables
- Status: done 2026-07-19 -> eda/interface/{stats.py,ci_audit.py,ci_audit_summary.csv}, results_rerun/results_compare_maxcut_{N2000,N1000_G1}/, chapter 3.3.1/3.3小结/3.6 rewritten (commit eb81325). OUTCOME: downgrade branch fired AND G1 flipped — N=2000 G22 speedup 1.6x CI [0.5,7.0] (was-3.71x = 4-vs-1-hit fluctuation, prefix-verified); G1 N=1000 SA significantly faster 0.84x [0.75,0.94], consistent with Peskun ordering (ref added). Chapter reframed: Gibbs value = physical sampling realization, not algorithmic speedup. NOTE for RX-02/RX-11: use these CIs; RX-11 premise (monotonic-growth prediction) was DELETED from 3.3.1 — RX-11 is now exploratory, not claim-defense.
- Effort: 0.5-1 day. Deps: none (unlocks the stats helper for all others).
- Objection: the §3.3.1 core finding "G22 3.71x Gibbs-vs-Metropolis speedup"
  is 4 hits vs 1 hit at N_trial=200; 95% CI on p_s=0.020 is ~[0.006,0.05]
  and OVERLAPS the SA value 0.005. No table in §3.3-3.5 carries intervals.
- Protocol:
  1. Write `eda/interface/stats.py`: `wilson(k,n,conf=0.95)` and a bootstrap
     `tts_ratio_ci(hits_a, n_a, hits_b, n_b, n_sweeps, B=10000, seed=...)`.
  2. Re-run G22 both dynamics (`compare_baselines.py` config: T=1e4,
     beta 0.1→10) at N_trial=2000 (block or numba path — ideal/metropolis
     kinds are JIT-safe, so async_numba is allowed and fast), master_seed
     2024; also G1 at N_trial=1000 as the well-powered control. Persist to
     `results_rerun/results_compare_maxcut_N2000/`.
  3. Post-hoc: bootstrap CIs for every existing table cell from the
     persisted per-trial JSONs in `results/` (no re-solve needed);
     write `eda/interface/ci_audit_summary.csv` (table, instance, p_s, lo,
     hi, tts_lo, tts_hi).
- Decision rule: if the N=2000 G22 CIs stay disjoint -> keep the 3.71x
  claim, add CI parentheses to 表3.3 and one sentence on statistical
  treatment in §3.3.1. If they overlap -> §3.3.1 "核心发现" and the §3.6
  algorithm-contribution paragraph get downgraded to "方向性一致但在
  n<=2000未达统计分辨" (honest-negative wording), and the speedup column
  gains CI bounds. Either way 表3.2/3.4/3.5 gain a CI note.
- Artifacts: stats.py, ci_audit_summary.csv, N2000 compare summary + JSONs.

### RX-02 Parallel-update semantics vs the N_par=64 projection assumption
- Status: claimed 2026-07-19 session-e063faf4-loop
- Effort: 1-2 days. Deps: RX-01 (use its CIs).
- Objection: 表3.8 divides sweeps by N_par=64 while p_s was measured under
  strictly ASYNC Gibbs. Simultaneous updates of coupled spins =
  parallel-Glauber, which breaks detailed balance (the chapter itself cites
  the sync-oscillation problem of pSA in §3.3.1). The projection silently
  assumes p_s is unchanged.
- Protocol: in `isim.py`'s block machinery add two schedules (new SolverConfig
  field or a wrapper): (a) baseline async (existing); (b) chromatic — greedy
  colour classes of the instance graph, one class per step, record mean
  class size = honest achievable width; (c) fixed 64-blocks — spins
  partitioned into ceil(n/64) fixed blocks ignoring adjacency, whole block
  sampled simultaneously from the pre-update field (this IS the 表3.8
  assumption). Run G1/G14/G22, ideal backend, G-set protocol, 200 trials
  (2000 for G22 per RX-01), same master seed. Driver:
  `eda/interface/run_parallel_semantics.py` ->
  `parallel_semantics_summary.csv` (instance, schedule, width, p_s, CI,
  tts_sweeps).
- Decision rule: if (c) p_s ~= async within CI -> add one validating
  sentence + footnote to §3.4.3 (positive result, retires the objection).
  If (c) degrades -> 表3.8's TTS gains an instance-dependent parallel
  penalty factor; §3.4.3 and the §3.5.4 reprojection both get corrected
  (multiply-through, scripts `bench_hardware_compare.py` /
  `eda/interface/reproject_hw.py` re-run with effective N_par or effective
  p_s); §3.6 wording adjusted. Report (b)'s mean colour-class size as the
  correctness-preserving width for G1/G22 (degree 48/20 graphs — expect
  far above 64 colours needed? verify, do not assume).
- Artifacts: run_parallel_semantics.py + summary csv (+ solver diff).

### RX-03 PVT corners for the write chain (DC + energy/timing)
- Status: done 2026-07-19 -> eda/testbenches/{update_chain,update_energy}_summary_{ss,ff,sf,fs}.json, update_chain_corners_summary.json, results_reproject_{ss,sf,fs}/, fig 3.9(d), 表3.9 note + 3.5.1/3.5.4/3.5.5 text. OUTCOME: monotonic at ALL corners (表3.9 validated); sf skew corner = outlier (INL 9.3 LSB / -39.4 mV bow / 18 mV offset, one-time per-code calibration absorbs it); settle 5.5-14.2 ns, gated k=3 e 13.7-16.3 pJ; projection correction x14-26 time / x17-21 energy across corners.
- Effort: ~1 day (compute-bound). Deps: none. Runs in WSL.
- Objection: every §3.5.1/§3.5.4 number is sky130 tt-only; "2.87 mW / 9.0 ns
  is an implementation upper bound" is asserted without ss/ff data;
  表3.9 "monotonic at all bit widths" is a tt-only row.
- Protocol: parameterize the corner string in
  `eda/testbenches/update_chain_dc.py` and `update_energy.py` deck builders
  (`.lib {SKY130_LIB} {corner}` — currently hardcoded tt); run
  {tt, ss, ff, sf, fs} x {6-bit DC sweep, mid/top pulse energy, settle}.
  MTJ/OSDI model has no corners (state that as ANALYTIC constancy).
  Write `update_chain_corners_summary.json` (per corner: LSB, INL,
  monotonic, buffer offset, u-window, P_buf, t_settle, E_dev/E_tg) and
  re-run `eda/interface/reproject_hw.py --k 3` with the WORST-corner
  gated e_update/t_update into `results_reproject_worstcorner/`.
- Decision rule: monotonicity break at any corner -> 表3.9 gains a corner
  column/note and §3.5.1 wording changes. Settle/static spread -> §3.5.4
  quotes tt with corner band (e.g. "9.0 ns (tt), x.x-y.y ns across
  corners"); §3.5.5 caveat paragraph drops "未含工艺角" and states the band.
- Artifacts: corner summary json, worst-corner reprojection csv.

---

## P1 — turns single-instance anecdotes into design rules; real novelty

### RX-04 Generalize the §3.4.2 + §3.5.2/3 rules beyond ER14
- Status: open
- Effort: 2-3 days (compute-bound sweeps). Deps: RX-01 stats helper.
- Objection: sensitivity ranking (§3.4.2), rail-span >=6V_T and >=4-bit
  rules, measured-grid==ideal-grid equivalence, IR 3.04x (§3.5.2/3.5.3)
  all come from ONE 14-spin ER instance with one instance-seed. The clip
  rule is landscape-coupled: truncated drive fraction scales with the
  |h_eff| distribution ~ graph degree.
- Protocol: two ladders.
  (a) ER robustness: er_seed in {0,1,2,3,4}, n in {14, 20} (n=20 still
      enumerable, `--er-n 20` supported), full circuit-ablation axes A/C/E
      of `eda/interface/run_circuit_ablation.py` (bits, span, reset-k)
      -> per-seed CSVs + a pooled `circuit_ablation_multi_summary.csv`.
  (b) real classes: G1, G14, G22 with target=BKS (no enumeration):
      span axis {2,4,6,8,10 V_T} + bits {4,6} + k {1,3} via a new driver
      `eda/interface/run_circuit_ablation_gset.py` reusing
      circuit_backends; G-set protocol, 200 trials (G14 will sit at
      p_s=0 — report energy-median degradation instead of TTS ratio,
      note in output).
  Also rerun the §3.4.2 five-knob ablation on G1 via existing
  `bench_device_ablation.py --problem-kind gset --instance G1` (wired,
  never run for the thesis).
- Decision rule: if span saturation moves to >=8-10 V_T on G22 -> §3.5.2's
  "6V_T" rule and the §3.6 restatement become degree-dependent (state the
  scaling); if the §3.4.2 ranking reorders on G14 (signed) -> qualify the
  ranking claim. If rules transfer -> upgrade wording from "该实例上" to
  multi-instance, cite the pooled CSV.
- Artifacts: multi-seed + gset ablation CSVs, G1 device-ablation dir.

### RX-05 Harden the reset story (LLG correlation + mechanism controls + scale)
- Status: open
- Effort: 2-4 days. Deps: none. Highest physics novelty; touches ch2 assets.
- Objection: §3.5.2's headline "plateau downgraded to sticky residual,
  k=1-2 suffice" rests on (i) reset-failure INDEPENDENCE (0.28^k), admitted
  uncalibrated in §3.5.5; (ii) an untested attribution — is the benignity
  due to ASYMMETRY (only one transition affected) or the two-step
  reset-write STRUCTURE?; (iii) 14-spin only.
- Protocol, three sub-items (separately claimable as RX-05a/b/c):
  (a) LLG correlation: use the vendored full-dynamics engine
      (`04PBNNSim/smtj_pbnn_sim/eda/vendor/vgsot-sim/va/llg/vgsot_llg.va`
      compiled .osdi exists; thermal field hx/hy/hz injected by harness,
      Brown-1963, recorded seeds — see that repo's llg harnesses for the
      injection pattern). New WSL harness
      `eda/testbenches/reset_correlation_llg.py`: drive the two-pulse
      AP->P reset sequence at t_w=0.75 ns, >=1000 seeds; extract per-pulse
      success r1 (vs the 0.72 plateau anchor) and conditional
      P(fail k+1 | fail k). Feed the MEASURED conditional chain into a
      `circuit_chain` variant (rho_k from data, not 0.28^k) and re-run the
      §3.4.2-protocol reset axis. CAUTION: the LLG device params are the
      ch2 macrospin calibration — back-hopping may need the high-V
      operating point; if the plateau does not reproduce in macrospin LLG,
      report that as a finding (plateau = beyond-macrospin physics) and
      keep 0.28^k labeled as model assumption — do NOT force-fit.
  (b) mechanism controls (pure Python, hours): two one-line backend
      variants in `circuit_backends.py`: asymmetric ceiling
      (P(+1)=min(sigma(u),0.72), floor untouched) and within-scheme
      symmetric saturation (0.72 on both reset and write). Run §3.4.2
      protocol; compare against sticky-k1 (1.10x) and symmetric-clip
      (catastrophic, §3.4.2).
  (c) scale check: sticky k=1 on a >=64-spin instance (reuse RX-09's
      64-node instance) — does the 0.28 residual accumulate like a mild
      h_off at scale?
- Decision rule: (a) correlated failures -> effective residual > 0.28^k,
  k recommendation moves to 3-4, e_update(k) and the §3.5.4 numbers shift
  (re-run reproject); independence confirmed -> §3.5.5 caveat replaced by
  a calibration sentence + possible [^process-*] footnote if the LLG hunt
  had real failures. (b) if asymmetric ceiling is also ~1.1x -> benignity
  is directionality, device-criterion relaxation is scheme-independent
  (strengthen §3.6); if catastrophic -> relaxation is contingent on the
  two-step scheme (qualify §3.5.2/§3.6). (c) growth with n -> "k=1已可接受"
  becomes size-qualified.
- Artifacts: reset_correlation_llg.py + summary json; backend variants +
  ablation rows; scale-check csv.

### RX-06 Read-decision offset MC -> misread channel (the 6th non-ideality)
- Status: open
- Effort: 1-2 days. Deps: none. WSL for the MC, Windows for the solver leg.
- Objection: §3.5.4's read energy claims `correct=True` at tt/nominal only.
  Margin is ~14-20 mV (vsen 0.080/0.114 V vs 0.100 V reference) on small
  devices; Pelgrom sigma could be comparable. Misreads corrupt h_eff of
  ALL neighbors — a channel absent from the 5-param model, and NOT
  averaged away like sigma_C2C.
- Protocol: extract the read path + PMOS-input StrongARM from
  `eda/testbenches/update_energy.py` into a standalone netlist; port the
  Pelgrom MC harness pattern from
  `04PBNNSim/smtj_pbnn_sim/eda/hero/run_offset_mc.py` (N=120, sky130
  AVT~5 mV*um, common mode at the ~0.1 V read node) ->
  `eda/testbenches/read_offset_mc.py` + summary (sigma_off in mV and in
  margin units; per-state misread probability under a Gaussian tail).
  Solver leg: add `p_read_flip` to `circuit_chain` (after sampling, the
  STORED state used for future h_eff flips with p_read_flip — note this
  needs the backend to model the stored-state channel; simplest faithful
  model: flip the returned sample with p_read_flip, document the
  equivalence assumption). Run §3.4.2 protocol at the measured flip rate
  and 3x/10x it.
- Decision rule: if sigma_off implies misread >1e-3 and the solver shows
  degradation -> §3.5.4 gains a read-margin paragraph and the §3.4.2
  "sigma_C2C most tolerated" discussion gains the contrast (misreads are
  sticky, not zero-mean); evaluate the autozero fallback qualitatively
  (the PBNN comparator family has `dong_autozero.spice` — cite as design
  direction, do not port unless needed). If sigma_off is comfortably small
  -> one validating sentence in §3.5.4 + summary json.
- Artifacts: read_offset_mc.py + summary, backend flag, ablation rows.

---

## P2 — completeness / boundary-setting

### RX-07 Full-stack realistic point + interactions + joint PDK mismatch
- Status: open
- Effort: 2-3 days. Deps: RX-01.
- Objection: §3.4.2 is one-factor-at-a-time; a real array carries all five
  device knobs at measured values PLUS the circuit constraints
  simultaneously; interactions (h_off x clip, h_off x p_max) untested.
  D2D is modeled as gain-only Gaussian; real PDK mismatch shifts
  (V_th, slope) JOINTLY.
- Protocol: (a) combined point: behavioral knobs at ch2 measured values
  (g_dev from measured slope ratio, h_off=0 nominal, sigma_C2C measured,
  p_max=1 with reset k=3 handling the plateau, cv_gain=0.077) + measured
  6-bit grid + span 4 + IR residual profile -> one "realistic full-stack"
  row vs the OFAT product. (b) fractional factorial: 2-level Res-IV over
  {g_dev 0.7/1.0, h_off 0/0.1, sigma 0/1, span 4/6, k 1/3} (16 runs) ->
  largest pairwise interaction. (c) joint mismatch: adapt
  `02MRAMSim/vgsot-sim/scripts/07_process_variability/` sampling to emit
  per-device (V_th_i, slope_i) -> map to per-spin (g_i, h_off_i) fed via
  circuit_chain's u_offset + a per-spin gain (needs a small backend
  extension mirroring ArrayDispersion). Driver
  `eda/interface/run_fullstack.py` -> `fullstack_summary.csv`.
- Decision rule: super-multiplicative combined degradation -> §3.4.2/§3.6
  stop presenting the ranking as additive, report the dominant interaction;
  approx-multiplicative -> state that as a positive decomposability result.
  Joint-mismatch h_off channel dominating -> the "CV 7.7% costs only 1.4x"
  reassurance gets qualified.
- Artifacts: run_fullstack.py + csv; backend extension.

### RX-08 Adversarial boundary for the beta-scaled 3-bit speedup
- Status: claimed 2026-07-19 session-e9b0f4b0-loop
- Effort: ~1 day. Deps: none.
- Objection: §3.5.2 reports 0.16x TTS for beta-scaled 3-bit and refuses to
  generalize, but never shows where it HURTS — unfalsifiable as written.
- Protocol: identical config (`circuit_chain` beta_scaled nbits=3,
  h_clip from instance) on landscapes where fine small-field resolution is
  load-bearing: G14 (energy-median metric, p_s=0 regime), factoring M=65
  (§3.3.2 deepest pseudo-product trap; factoring protocol T=2e4,
  beta 0.05→30, C=M^2+1), and one planted frustrated 3-regular n=16-20
  instance (enumerable); >=3 master seeds each.
- Decision rule: >=1 landscape with TTS>1x vs ideal -> §3.5.2 gains the
  boundary sentence with numbers ("在X类景观上退化Yx"); if it never hurts
  -> upgrade the §3.5.2 claim toward a usable low-bit guideline (stronger,
  and note it).
- Artifacts: rows appended to a `betascaled_boundary_summary.csv`.

### RX-09 IR predistortion on a fully-populated >=64-spin array
- Status: open
- Effort: 0.5-1 day. Deps: none.
- Objection: §3.5.3's 3.04x -> 1.06x recovery was shown with 14 spins
  spread over 64 rows (most rows empty, offsets never couple through a
  full J). N_par=64 premise implies 64 occupied rows.
- Protocol: generate ER n=64 (p~0.10, seed 0, enumeration impossible —
  use long-run best-of as reference or a planted instance for exact
  target; simplest defensible: planted-partition Max-Cut n=64 with known
  optimum) mapped one-spin-per-row to the measured N=64 offset profile
  from `eda/extraction/writeline_ir/ir_drop_summary.json`; run
  {no offset, uncompensated, predistorted} x 200 trials; extend to
  N in {128,256} profiles (analyze_ir.py already emits them).
- Decision rule: recovery worse than ~1.3x at N=256 -> §3.5.3's "可被完全
  消除/非工艺限制" conclusion gets size-qualified; clean recovery ->
  strengthen the sentence with the n=64 result.
- Artifacts: `eda/extraction/writeline_ir/ir_fullarray_summary.csv` +
  driver script.

### RX-10 Same-caliber energy comparison (synapse term + N_par sensitivity)
- Status: open
- Effort: 1-2 days. Deps: RX-03 (worst-corner numbers useful, not required).
- Objection: §3.5.4 defends the post-correction energy loss to CMOS p-bit
  as an accounting-boundary issue but never QUANTIFIES the fairness: the
  digital h_eff accumulator is excluded from both unit-level rows while
  FPGA/CPU rows include it.
- Protocol: estimate a sky130 per-update h_eff accumulation energy from
  the sibling per-op patterns (`04PBNNSim/.../eda/testbenches/
  dac_counter_energy.py` — adder/counter ~19-34 fJ/op class) x mean degree
  of each benchmark graph; add the SAME term to sMTJ and CMOS p-bit rows
  in `reproject_hw.py`; sweep N_par in {16,64,256} and P_buf in
  {2.87 mW, /4 (shared driver), /16} to locate the sMTJ-vs-CMOS energy
  crossover; one panel (figs pipeline) if the crossover is informative.
- Decision rule: if the synapse term dominates both rows -> the relative
  sMTJ-vs-CMOS gap narrows and §3.5.4's boundary argument gets its
  quantitative sentence; if the sMTJ advantage is restored under shared
  drivers -> add that as the engineering-direction quantification
  (currently only qualitative in §3.5.4/§3.6).
- Artifacts: reproject extension + `energy_caliber_summary.csv`.

### RX-11 Density/scale sweep for the Gibbs-vs-Metropolis prediction
- Status: open
- Effort: 1-2 days. Deps: RX-01 (CIs mandatory here).
- Objection: §3.3.1 predicts the Gibbs advantage grows monotonically with
  scale/|h_eff| variance from ONE supporting point (G22); G1 (same n
  class, dense) shows none.
- Protocol: (i) real ladder: G23-G27 (n=2000), G43-G47 (n=1000), G55
  (n=5000) — extend GSET_BKS in `compare_baselines.py` from the
  BenlicHao/Gset BKS tables (VERIFY each BKS value against the published
  table before committing; do not trust memory); (ii) synthetic ladder:
  d-regular +/-1 graphs, n=1000, degree {6,12,24,48,96}, 500 trials,
  target = long-run best-of (report relative metrics, label accordingly).
  Plot speedup vs degree and vs Var(|h_eff|).
- Decision rule: monotone rise -> §3.3.1 prediction substantiated (new
  panel + sentence); flat/non-monotone -> rewrite prediction as
  regime-dependent in §3.3.1 AND §3.6.
- Artifacts: extended compare summaries + `density_sweep_summary.csv`.

---

## P3 — caveat-closing (mostly future-work grade; do after P0-P2)

### RX-12 Sigmoid emergence under the real transient waveform (LLG)
- Status: open. Effort: 2-3 days. Deps: RX-05a harness.
- Drive the measured V_wr(t) pulse (from update_energy.py wrdata) into the
  LLG engine over many seeds; empirical P_sw vs static sigma((V_flat-Vth)/
  V_T) across codes. Decision: bias found -> §3.5.1's delivered-probability
  numbers get a dynamic-correction note; none -> validating sentence for
  the harness-RNG decoupling (§3.5.5 methodology).

### RX-13 Write-line RC transient (extract C, data-dependent settling)
- Status: open. Effort: 2-3 days. Deps: none.
- Extend writeline_ir flow to capacitance, build the RC line, transient
  with mixed reset/write row currents in 0.75 ns. Decision: line does not
  settle or drop is data-dependent -> §3.5.3's static-predistortion claim
  gets a bandwidth qualifier; else validating sentence.

### RX-14 H-bridge bipolar reset driver
- Status: open. Effort: 2-3 days. Deps: RX-03 useful.
- Minimal sky130 H-bridge around the OSDI SOT branch reusing the steering
  scaffolding; true bipolar reset energy + settling -> reassemble
  e_update(k), re-run reproject. Decision: replaces the §3.5.5 "energy
  proxy" caveat with measured numbers; expect e_update up, absolute-energy
  story weakens further (V_T-normalized results unaffected).

### RX-15 Encoding-side generality (factoring bit budget + permutation scale)
- Status: open. Effort: 0.5-1 day each. Deps: none.
- (a) repeat the §3.3.2 b_q sweep on M in {93, 129, 115} (3x31, 3x43,
  5x23); (b) push perm_tsp to att48/eil51/kroA100 at matched sweeps.
  Decision: confirms or size-qualifies the §3.3.2 "冗余比特有害" and
  §3.3.3 "置换空间p_s=1" claims.

---

## Suggested first wave (from the 2026-07-19 review synthesis)

One-week budget: RX-01, RX-02, RX-03, RX-05, RX-08. RX-01/02 defend the
chapter's two most attackable quantitative claims; RX-03 is the circuit
table-stakes; RX-05 converts the chapter's most distinctive claim from
assumption to calibration (and reuses the ch2 LLG asset); RX-08 makes the
beta-scaled finding falsifiable. Be prepared: RX-01 may force a downgrade of
the §3.3.1 "3.71x核心发现" — finding that ourselves is the point.
