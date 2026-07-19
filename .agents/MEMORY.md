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
- 2026-07-19: RX-01 DONE (commit eb81325) — G22 3.71x was low-count
  fluctuation (N=2000: 1.6x CI [0.5,7.0]); G1 SA significantly faster
  0.84x; 3.3.1/3.6 honestly rewritten, Peskun ref added, CI convention
  now chapter-wide. RX-03 corner sweep running.
- 2026-07-19: RX-03 DONE — five-corner sweep: monotonic everywhere, sf
  skew corner is the outlier (-39.4 mV bow, per-code cal absorbs); fig
  3.9 gained panel (d); corner bands in 3.5.4/表3.9.
- 2026-07-19: two-persona reviewer audit distilled into
  `EXPERIMENTS_QUEUE.md` (RX-01..RX-15, claim protocol). Suggested first
  wave: RX-01/02/03/05/08. Nothing claimed yet.
