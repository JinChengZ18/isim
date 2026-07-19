#!/usr/bin/env python3
"""RX-01 — post-hoc confidence-interval audit of the chapter-3 tables.

Reads the CANONICAL benchmark summaries (the descriptive *_summary.csv files
the thesis tables were built from), reconstructs the binomial hit counts
(k = round(p_s * n_trials); exact because p_s was computed as k/n), and
attaches Wilson 95% intervals to every p_s plus parametric-bootstrap
intervals to every Gibbs-vs-Metropolis speedup. No re-solving.

Run:  python eda/interface/ci_audit.py
Writes ci_audit_summary.csv next to this script and prints the verdict for
the headline claims (G22 3.71x; the factoring/TSP ~1x band).
"""
from __future__ import annotations

import csv
from pathlib import Path

from stats import wilson, tts_ratio_ci

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

COMPARE = [
    ("表3.3-maxcut", ROOT / "results" / "results_compare_maxcut"
     / "compare_maxcut_summary.csv"),
    ("表3.3-factor", ROOT / "results" / "results_compare_factor"
     / "compare_factor_summary.csv"),
    ("表3.3-tsp", ROOT / "results" / "results_compare_tsp"
     / "comapre_tsp_summary.csv"),
    # RX-01 power reruns (seed-prefix-consistent with the canonical rows:
    # SeedSequence(2024).spawn(N) children 0..199 are identical to the
    # N=200 protocol, verified 8/2000 contains the original 4/200 hits)
    ("RX01-N2000", ROOT / "results_rerun" / "results_compare_maxcut_N2000"
     / "summary.csv"),
    ("RX01-N1000", ROOT / "results_rerun" / "results_compare_maxcut_N1000_G1"
     / "summary.csv"),
]
SINGLE = [
    ("表3.2-maxcut", ROOT / "results" / "results_maxcut" / "summary.csv"),
    ("表3.4-factor", ROOT / "results" / "results_factor" / "summary.csv"),
    ("表3.5-tsp", ROOT / "results" / "results_tsp" / "summary.csv"),
]


def fnum(row, *names, default=None):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    return default


def main():
    out = []
    verdicts = []
    for table, path in COMPARE:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        for row in csv.DictReader(open(path, newline="")):
            inst = row.get("instance") or row.get("name")
            n = int(fnum(row, "n_trials", default=200))
            nsw = int(fnum(row, "n_sweeps", default=10000))
            ps_a = fnum(row, "ps_smtj")
            ps_b = fnum(row, "ps_sa")
            if ps_a is None or ps_b is None:
                continue
            ka, kb = round(ps_a * n), round(ps_b * n)
            lo_a, hi_a = wilson(ka, n)
            lo_b, hi_b = wilson(kb, n)
            r = tts_ratio_ci(ka, n, kb, n, n_sweeps=nsw) \
                if ka > 0 and kb > 0 else dict(ratio=float("nan"),
                                               lo=float("nan"),
                                               hi=float("nan"),
                                               frac_undefined=1.0)
            overlap = not (hi_b < lo_a or hi_a < lo_b)
            out.append(dict(table=table, instance=inst, n_trials=n,
                            k_smtj=ka, ps_smtj=ps_a,
                            wilson_smtj=f"{lo_a:.4f}..{hi_a:.4f}",
                            k_sa=kb, ps_sa=ps_b,
                            wilson_sa=f"{lo_b:.4f}..{hi_b:.4f}",
                            ps_ci_overlap=overlap,
                            speedup=r["ratio"], speedup_lo=r["lo"],
                            speedup_hi=r["hi"],
                            speedup_frac_undef=r["frac_undefined"]))
            if inst in ("G1", "G22") or table.endswith("tsp") or table.startswith("RX01"):
                verdicts.append(
                    f"{table} {inst}: k={ka}vs{kb}/{n}  "
                    f"p_s CI overlap={overlap}  "
                    f"speedup={r['ratio']:.2f} CI=[{r['lo']:.2f},"
                    f"{r['hi']:.2f}] undef={r['frac_undefined']:.2%}")

    for table, path in SINGLE:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        for row in csv.DictReader(open(path, newline="")):
            inst = row.get("instance") or row.get("name")
            ps = fnum(row, "p_success", "ps")
            if ps is None:
                continue
            n = int(fnum(row, "n_trials", default=200))
            k = round(ps * n)
            lo, hi = wilson(k, n)
            out.append(dict(table=table, instance=inst, n_trials=n,
                            k_smtj=k, ps_smtj=ps,
                            wilson_smtj=f"{lo:.4f}..{hi:.4f}",
                            k_sa="", ps_sa="", wilson_sa="",
                            ps_ci_overlap="", speedup="", speedup_lo="",
                            speedup_hi="", speedup_frac_undef=""))

    with open(HERE / "ci_audit_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"-> ci_audit_summary.csv ({len(out)} rows)\n")
    for v in verdicts:
        print(v)


if __name__ == "__main__":
    main()
