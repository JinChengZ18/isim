# .agents/ — cross-session coordination channel (committed)

Orientation for any Claude session working on this repo. Keep thesis-facing
content OUT of here; this directory never ships.

## What lives where

| File | Role |
|---|---|
| `EXPERIMENTS_QUEUE.md` | actionable reviewer-driven experiment queue (RX-01..RX-15) with a claim protocol for parallel sessions — START HERE for "what next" |
| `PLAN_eda_ch3.md` | the (completed) build plan for §3.5 + eda/ workspace; kept for provenance of anchors/conventions |
| `TRIAL_LOG_eda.md` | dated REAL failure log — the only admissible source for new [^process-*] thesis footnotes; append, never invent |

## Repo ground truth (as of 2026-07-19)

- Chapter file `article/chapter03.md`: §3.1-3.3 algorithm benchmarks,
  §3.4 device ablation + hardware projection, §3.5 circuit-level update
  chain (sky130), §3.6 summary. De-AIGC'd in commit 0885355 — locate
  claims by section + number, not exact strings.
- `eda/` = circuit workspace (models/testbenches/interface/extraction/
  schematics/figs_raw + figs_make.py + build_ppt_figs.py). Every thesis
  number traces to a committed `*_summary.json`/`.csv`.
- Figure pipeline: `python eda/figs_make.py` (clean panels) then
  `python eda/build_ppt_figs.py` (letters via PPT+LibreOffice into
  `article/figs/`). Never write article/figs directly.
- Solver backends for circuit constraints: `eda/interface/circuit_backends.py`
  (kind `circuit_chain`), block mode only. §3.4.2 protocol constants and
  all shared ground rules are restated at the top of EXPERIMENTS_QUEUE.md.
- Git: local commits only, NEVER push. `chapter03.docx` regeneration is
  user-side. `article/figs/Chapter03_local_03(1).png` is an untracked user
  file — leave it alone.

## Changelog

- 2026-07-13: eda/ workspace + §3.5 written, verified, committed
  (6ab96f6, 57f4a8b). Trial log gained 4 real entries.
- 2026-07-13 (parallel session): de-AIGC pass over ch3 + panel letters +
  PPT compose workflow `eda/build_ppt_figs.py` (0885355); figs_make.py
  retargeted to clean panels + previews.
- 2026-07-19: two-persona reviewer audit distilled into
  `EXPERIMENTS_QUEUE.md` (RX-01..RX-15, claim protocol). Suggested first
  wave: RX-01/02/03/05/08.
- 2026-07-19: RX-01 DONE (commit eb81325) — G22 3.71x was low-count
  fluctuation (N=2000: 1.6x CI [0.5,7.0]); G1 SA significantly faster
  0.84x; 3.3.1/3.6 honestly rewritten, Peskun ref added, CI convention
  now chapter-wide. RX-03 corner sweep running.
- 2026-07-19: RX-03 DONE — five-corner sweep: monotonic everywhere, sf
  skew corner is the outlier (-39.4 mV bow, per-code cal absorbs); fig
  3.9 gained panel (d); corner bands in 3.5.4/表3.9.
- 2026-07-20: RX-02 DONE — fixed-64-block parallel Gibbs == async within
  CI on G1/G22; chromatic width G1=42/G22=167; 表3.8 N_par=64 validated
  (footnote in 3.4.3). All three P0 items closed.
- 2026-07-20: RX-05+RX-08 DONE (108425a) — reset benignity = hold-type
  residual (not asymmetry); LLG: plateau reproduced, back-hop 14-21%/pulse
  -> effective residual decays slowly, k guidance ~2-3 + device-side fix;
  beta-scaled boundary found on G14/reg3 (mid-rise grid erases small
  fields). P0 wave + RX-05/08 all closed.
- 2026-07-20: three-auditor verification of the whole RX wave applied
  37 fixes (e066e03) incl. a HIGH k-guidance contradiction in 3.5.4;
  TRIAL_LOG now grounds both new process footnotes. RX-04 claimed next.
- 2026-07-21: independent spot-acceptance of RX-01/02/03/05/08 (audit
  wf_822c7f8a, user-requested): ALL headline numbers recomputed from
  committed raw artifacts and traced into the chapter — accepted. Fixed
  the stale RX-08 status line (work was in 108425a by e063faf4-loop;
  the 2026-07-19 e9b0f4b0 claim delivered nothing). RX-04 confirmed
  alive (d515357 landed mid-audit).
- 2026-07-21: RX-04 DONE (d515357, 94c34ac) — falsified the +/-6V_T rail
  rule (G1 needs >=+/-10V_T) and the "2-bit better" reading (seed
  artifact); found the bits axis was confounded at span=4 and refilled it
  => design inverted to "trade resolution for range"; device ranking ends
  invariant but middle reorders (CV(Delta) up, h_off/g_dev down, both
  max|J|-relative). Acceptance-audit polish backlog cleared in 94c34ac.
- 2026-07-22: RX-09 DONE (2085e7a) — predistortion validated on a fully
  populated 64-spin array (claim strengthened; compatible with RX-04's
  resolution-for-range), but the uncompensated 3.04x anchor is
  landscape-signed at N=64 and was replaced by the N-scaling. Fig 3.11(b)
  replotted. RX-06 still running in the same session.
- 2026-07-22: RX-06 DONE (6050d9c) — read decision fails under mismatch
  (sigma_off 18.5 mV > AP margin; 5.87x on G1 even at the optimistic
  8.2e-4 misread rate; tolerance ~1e-5 at G1). Read margin, not write
  precision, is now the stated binding constraint at array scale. Fig
  3.10 gained panel (e). RX-06+RX-09 both closed this session.
- 2026-07-22: RX-07 DONE (76c1705) — decomposition valid at the operating
  point but super-multiplicative off-nominal (>400x on G1); rail x h_off
  synergy FALSIFIED RX-04's h_off-relief claim (it was an unclipped-drive
  artefact); joint PDK mismatch found the model's missing channel
  (per-device V_th shift, 6.0 V_T, needs 57/63 trim codes competing with
  IR predistortion). Also: 35 audit fixes on RX-06/09 landed in 554b299,
  which promoted the read path to its own subsection 3.5.5 (old 3.5.5 ->
  3.5.6). Queue: only RX-10, RX-11 and the four P3 items remain open.
- 2026-07-23: RX-10 DONE (22ea83b) — synapse term measured in-flow (the
  sibling's 19.4 fJ would have been ~20x wrong); same-caliber gap
  1.1-1.8x, and the sharing crossover is now a curve (k=1 S>=2, k=2
  S>=6, k=3 S>=102) exposing a k-tension with RX-05. Also: 21 audit
  fixes on RX-07 landed in 26f8cc2 — my falsification of RX-04's h_off
  relief was itself over-stated (it fails only below the rail the
  chapter prescribes). Queue: only RX-11 + four P3 items remain.
- 2026-07-23: RX-11 DONE (21c0b21) — re-scoped (its premise had been
  deleted by RX-01) and it SUBSTANTIATES the replacement claim: no Gibbs
  advantage at any degree 6-96, no degree trend. Earlier the same day
  RX-10's sharing model was RETRACTED (69ef38b) after an audit showed it
  contradicted this repo's own W3 trial-log measurement; parity is now
  stated as a driver-static-power target (1.9x at k=1, 5.5x at k=2) with
  a switching-mode driver as the lever, and fig 3.11(e) was replotted.
  ALL P0/P1/P2 QUEUE ITEMS ARE NOW CLOSED; only the four P3 items
  (RX-12..RX-15, future-work grade) remain open.
- 2026-07-23: RX-15 DONE (a3ddd07) — §3.3.2 bit-budget confirmed on 3
  more semiprimes; §3.3.3 permutation p_s=1 shown to be n<=17 only
  (best still hits/near-hits opt at n=48-100, per-trial p_s 0.08-0.24).
  Fourth P3 item (RX-12/13/14) still running as background agents.
- 2026-07-23: RX-13 DONE (1d400cb) — write-line RC settles in <=5.8 ps
  (130x margin), static IR model validated, no bandwidth qualifier;
  multi-cell same-line coupling (+15..+32 u) is the physical reason the
  schedule must be row-sequential. RX-12/RX-14 agents still running.
- 2026-07-23: RX-14 DONE (1dc9c38) — H-bridge bipolar reset measured:
  1.48 pJ/pulse (+19% vs proxy), e_update +4-5% at k=2-3, G22 projection
  18.7x->19.66x, TTS/V_T-ratios unchanged. 3.5.6 caveat upgraded from
  "not implemented" to measured. Only RX-12 (LLG sigmoid) still running.
- 2026-07-23: RX-12 DONE (159ec07) — macrospin LLG sigmoid-emergence. Under
  an IDEAL 0.75 ns flat-top at the DC-measured v_wr (no RC), LLG under-
  switches vs the ch2 measured-calibrated static sigmoid (max |dp|=0.54 @
  code 51; code 30 emp 0.19 vs 0.49; code 63 emp 0.57 vs 0.99), affine-
  recalibratable (u_emp=0.43u-1.54, resid ~0.20, ~2.3x window), switching
  completes in the 1.5 ns relaxation tail (tw~tau0). DECISION="none": no
  delivered-probability bias — harness draws Bernoulli in Python from the
  measured sigmoid (ground truth); the deviation is a macrospin-model +
  finite-pulse effect, and the RX-13 <=5.8 mV un-settled offset is absent
  from this ideal deck AND too small/wrong-sign (+0.06) to cause a 0.54 gap.
  -> §3.5.6 replay validating sentence + broadened LLG caveat. SCOPING: deck
  uses an ideal flat-top, so RX-12 does NOT test the RC transient (RX-13
  does). ALL RX-01..RX-15 NOW COMPLETE.
- 2026-07-23 COORDINATION: a parallel session is mid-way through a large
  uncommitted de-AIGC/restructuring rewrite of article/chapter03.md (~82/123
  diff vs HEAD, backup chapter03_backup_20260723-195127.md). RX-12's edit
  landed in §3.5.6 (which their rewrite had NOT reached) and was committed in
  ISOLATION via the git index (staged only HEAD+RX-12 blob), leaving their
  rewrite uncommitted on disk for them to commit. IF you are that session and
  rewrite §3.5.6, PRESERVE the two RX-12 sentences ("平顶电压到翻转概率的映射
  另经宏自旋LLG…" replay sentence + the broadened LLG caveat clause) — they
  are in git @159ec07 if lost. docx regen still user-side.
- 2026-08-07: Si2024 reference added (user request). Two edits to
  article/chapter03.md, committed in ISOLATION via the git index (same
  protocol as RX-12; the de-AIGC rewrite stays uncommitted on disk):
  (1) one anchor sentence in §3.3.3 "问题布置与规模限制" right after
  "而非强行套用QUBO求解器。" citing [^Si2024] (Si et al., Nat Commun
  15:3457, 2024 — 80-SMTJ all-to-all Ising annealer, GP+CTSP compression
  of 70-city TSP); (2) the [^Si2024] entry inserted between [^Aadit2022]
  and [^Reinelt1991] (first-use order). IF you are the rewrite session
  and touch §3.3.3 or the reference list, PRESERVE both. Differences
  analysis lives in notes/ref_si2024_comparison.md (standalone per user
  instruction — do NOT merge it into the chapter prose). Paper-side
  numbers there were adversarially verified against the PDF text; note
  Table 1's EXPERIMENTAL TSP70 time-to-solution is 40 s (50.54 s is the
  4Kb-simulation column — the mirrored table is easy to mis-decode).
- 2026-08-07 (2nd): public-knowledge citation rectification (user request,
  responds to reviewer's "方法出处未标注"). TEN new reference entries +
  anchors added to article/chapter03.md, again committed in ISOLATION via
  the index (rewrite still uncommitted on disk): Glauber1963, GemanGeman1984,
  Camsari2017 (3.1.1); Ronnow2014, Wilson1927 (3.3 preamble); Metropolis1953
  (3.3.1); Croes1958, LinKernighan1973 (3.3.3 2-opt parenthetical);
  Razavi2015 (3.5.1); Pelgrom1989 (3.5.5). [^Camsari2019] gained a second
  anchor at 3.1.1 and its DEFINITION MOVED up next to Camsari2017;
  [^Peskun1973] definition moved into first-use order (after
  Metropolis1953). All DOIs Crossref-verified (Pelgrom pages are 1433-1439,
  not -1440). IF you are the rewrite session: preserve all these markers
  and the reordered reference list; anchor sentences are identical in both
  versions except the 3.3 preamble Wilson sentence (wording differs per
  version, both patched). Reviewer letter + docx appendix live in notes/.
- 2026-08-07 (3rd): notes/appendix_si2024_comparison.docx is now HAND-
  FORMATTED by the user in WPS (headings shortened, §7 + the 统计协议 table
  row deleted, page break before 对照表) and has DIVERGED from the .md.
  NEVER regenerate it with pandoc — that clobbers his typesetting (it did
  once; he pushed back). To add content: unzip HIS docx, edit
  word/document.xml, rezip. Reuse existing styleIds (3=heading 2,
  5=heading 3, 45=body First Paragraph, 46=Compact/table, 40=Hyperlink
  char style); links need TargetMode="External" rels. The 信中所引文献
  section (15 refs, clickable DOIs) was appended that way and validated
  with the docx skill's validate.py (50→67 paragraphs). (LOW nits from the 2026-07-21 acceptance audit — apply
## when no session is actively editing chapter03.md; each is one line)

1. 表3.2/3.4/3.5 notes: RX-01's decision rule wanted a CI note at each
   table; only the §3.3 preamble sentence + 表3.3 note exist. Add one
   short CI sentence per note (or record the preamble as the chosen
   implementation in RX-01's Status line).
2. [^process-g22-power]: "逐一复现" overstates — first-200 energy
   MULTISETS are identical but stored per-index order differs (parallel
   completion-order storage). Suggest "结果集精确复现原4对1命中".
3. eda/interface/ci_audit.py comment: scope the prefix-identity claim to
   G22 (G1 N=1000 rerun has 1-of-200 prefix divergence; headline 0.84x
   unaffected — uses full-1000 counts, independently reproduced).
4. [^par-semantics] (§3.4.3): add half a sentence on why G14 is absent
   (all three schedules p_s=0 at this budget — comparison void; its
   artifacts ARE committed).
5. `article/figs/Chapter03_local_03(1).png`: untracked user file, likely
   an accidental "(1)" duplicate — ask user / recycle, never commit.
6. Rewrite-version §3.6 (working tree, ~line 652) uses banned word 落地
   ("到第一类系统任务的落地") — global CLAUDE.md ban list. Replace with
   落实 (or 完成…的衔接) when the rewrite session commits its version.
