#!/usr/bin/env python3
"""RX-10 helper — mean coupling degree of every canonical Section 3.3 benchmark
instance.

The synapse (local-field accumulation) term of the same-caliber energy account
is (mean degree) x (energy of one accumulate), under the assumption that one
spin update needs exactly one pass over that spin's non-zero couplings. This
script measures the degree from the SAME Problem objects the benchmarks solve
(``problems.load_gset`` / ``build_factoring_problem`` / ``load_tsplib``), so
the constant is not hand-typed.

Degree of spin i = #{j != i : J_ij != 0}. Reported statistics are the mean over
i (what the per-update cost scales with) and the max (what an accumulator must
not overflow on).

Run:  python eda/interface/graph_degrees.py
Writes eda/interface/graph_degrees.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from problems import (load_gset, load_tsplib,            # noqa: E402
                      build_factoring_problem)

GSET = ["G1", "G14", "G22"]
FACTOR = [15, 21, 33, 35, 51, 65, 77, 91, 143]
TSP = ["burma14", "ulysses16"]


def degree_stats(J) -> dict:
    # Problem.J is dense for the small instances and scipy-sparse for the
    # G-set ones; normalise to a dense array (n <= 2000 here).
    A = np.asarray(J.toarray() if hasattr(J, "toarray") else J, float).copy()
    np.fill_diagonal(A, 0.0)
    deg = (A != 0.0).sum(axis=1)
    nz = np.abs(A[A != 0.0])
    w_absmin = float(nz.min()) if nz.size else 1.0
    rowsum = float(np.abs(A).sum(axis=1).max())
    n_distinct = int(np.unique(np.abs(np.round(A, 9))).size - 1)
    # accumulator width: enough dynamic range to hold the largest |h_eff| at
    # the resolution of the smallest coupling, plus one sign bit; floored at
    # the narrowest width the testbench measures
    b_req = max(8, int(np.ceil(np.log2(rowsum / w_absmin))) + 1)
    return dict(n=int(A.shape[0]),
                deg_mean=float(deg.mean()),
                deg_max=int(deg.max()),
                deg_min=int(deg.min()),
                n_edges=int((A != 0.0).sum() // 2),
                w_absmax=float(nz.max()) if nz.size else 0.0,
                w_absmin=w_absmin,
                max_abs_rowsum=rowsum,
                w_distinct=n_distinct,
                binary_coupling=bool(n_distinct == 1),
                b_req=b_req)


def main():
    out = {"_label": "RX-10 MEASURED-from-problem-object coupling degrees of "
                     "the canonical Section 3.3 benchmark instances; the "
                     "synapse energy term is deg_mean x e_accumulate",
           "instances": {}}
    for name in GSET:
        p = load_gset(ROOT / "gset" / name)
        out["instances"][name] = degree_stats(p.J)
    for M in FACTOR:
        p = build_factoring_problem(M)
        out["instances"][f"factor_M{M}"] = degree_stats(p.J)
    for name in TSP:
        p = load_tsplib(ROOT / "tsplib" / f"{name}.tsp", B=1.0)
        out["instances"][name] = degree_stats(p.J)

    (HERE / "graph_degrees.json").write_text(json.dumps(out, indent=2))
    for k, v in out["instances"].items():
        print(f"{k:<14s} n={v['n']:<5d} deg_mean={v['deg_mean']:8.2f} "
              f"deg_max={v['deg_max']:<5d} |w|max={v['w_absmax']:.4g} "
              f"distinct|w|={v['w_distinct']:<4d} "
              f"{'binary' if v['binary_coupling'] else 'weighted':<8s} "
              f"b_req={v['b_req']}")
    print(f"-> {HERE / 'graph_degrees.json'}")


if __name__ == "__main__":
    main()
