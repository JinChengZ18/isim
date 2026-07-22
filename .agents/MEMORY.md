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

## Polish backlog (LOW nits from the 2026-07-21 acceptance audit — apply
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
