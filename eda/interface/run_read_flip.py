#!/usr/bin/env python3
"""RX-06 (solver leg) — the read-misread channel fed back into the solver.

eda/testbenches/read_offset_mc.py measures the input-referred offset of the
Section 3.5.4 read comparator under sky130-class Pelgrom mismatch and converts
it into a per-read misread probability.  This driver switches that channel on
in `circuit_backends.CircuitChainSpin` (parameter `p_read_flip`) and re-runs the
two comparability anchors of the chapter at the measured rate and at 3x / 10x
it, so the read path joins the five behavioural non-idealities of Section 3.4.2
on the same axes.

Protocols (unchanged from the chapter, so the rows are directly comparable):
  ER14  Section 3.4.2 anchor: random_er_maxcut(n=14, p=0.30, sigma=1.0,
        seed=0); geometric beta 0.1 -> 5.0, T = 2000 sweeps, block mode,
        200 trials, master_seed = 2024; exact target by enumeration.
  G1    Section 3.3 G-set anchor: T = 10000 sweeps, geometric beta 0.1 -> 10.0,
        200 trials, master_seed = 2024, block mode; target = edge_sum/2 - 11624.

Arm design.  The read-flip axis is run with mode="none", i.e. NO DAC
quantization and NO rail clipping, so the measured degradation is attributable
to the misread channel alone.  Two further G1 arms combine the channel with the
DAC backend; per RX-04 the rail half-width there is +/-10 V_T, because +/-4 V_T
puts G1 in the p_s = 0 regime and any read-flip effect would be unobservable
underneath the clip failure.  Those two arms are run as a pair (span-10 without
and with the flip) so the marginal cost of the misread is read off directly.

Statistics: RX-01 convention — Wilson 95% interval on every p_s, parametric
bootstrap 95% interval on every TTS ratio against the same-instance ideal arm
(eda/interface/stats.py).

Run (Windows python):
  python eda/interface/run_read_flip.py [--jobs 12] [--instances ER14 G1]
                                        [--p-read-flip <override>]
Writes read_flip_solver_summary.csv + read_flip_solver_config.json next to this
script; per-arm trial energies cached (resumable) under results_read_flip/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isim import (SolverConfig, multistart, p_success,          # noqa: E402
                  tts_at_confidence)
from problems import load_gset, random_er_maxcut                # noqa: E402
from stats import wilson, tts_ratio_ci                          # noqa: E402
import circuit_backends                                         # noqa: F401,E402

MC_SUMMARY = ROOT / "eda" / "testbenches" / "read_offset_mc_summary.json"
TRIALS, SEED, TOL = 200, 2024, 1e-6
ER14_TARGET = -6.877554338927609        # Section 3.4.2 published anchor
G1_BKS = 11624
GSET_SPAN = 10.0                        # RX-04: >= +/-10 V_T at G-set scale
GSET_BITS = 6
P_FLIP_CAP = 0.5                        # max-entropy point of a flip channel
BOOT_B, BOOT_SEED = 10000, 20260721

PROTO = {
    "ER14": dict(sweeps=2000, beta0=0.1, betaf=5.0),
    "G1":   dict(sweeps=10000, beta0=0.1, betaf=10.0),
}


def enumerate_ground(problem):
    n = problem.n
    assert n <= 22
    best = np.inf
    J = problem.J.toarray()
    for start in range(0, 2 ** n, 65536):
        m = min(65536, 2 ** n - start)
        ints = np.arange(start, start + m, dtype=np.int64)
        bits = ((ints[:, None] >> np.arange(n)) & 1).astype(np.int8)
        s = (2 * bits - 1).astype(np.float64)
        e = -0.5 * np.einsum("ij,ij->i", s @ J, s) - s @ problem.h
        best = min(best, float(e.min()))
    return best


def boot_dmedian_ci(x_arm, x_ideal, conf=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    xa = np.asarray(x_arm, dtype=np.float64)
    xi = np.asarray(x_ideal, dtype=np.float64)
    ma = np.median(xa[rng.integers(0, len(xa), size=(BOOT_B, len(xa)))], axis=1)
    mi = np.median(xi[rng.integers(0, len(xi), size=(BOOT_B, len(xi)))], axis=1)
    d = ma - mi
    a = (1 - conf) / 2
    lo, hi = np.quantile(d, [a, 1 - a])
    return float(lo), float(hi)


def read_measured_p(override=None):
    """The two bounding flip rates from the committed MC summary.

    The offset MC produces two defensible per-read misread rates that bracket
    the design, three orders of magnitude apart, so both are carried as
    separate ladders rather than one being silently preferred:

      conservative  static divider margins (-20.0 / +14.3 mV) against the
                    measured offset distribution. This is the case in which the
                    reference node is decoupled, which is what a real array
                    would do; no help from comparator kickback.
      as_committed  effective margins measured on the as-committed deck, where
                    the undamped, state-dependent sense-node source impedance
                    lets comparator kickback widen the decision margin in the
                    direction that reinforces the correct decision.
    """
    if override is not None:
        return {"override": (float(override), "CLI override")}
    if not MC_SUMMARY.exists():
        raise SystemExit(f"missing {MC_SUMMARY}; run the WSL leg first "
                         f"(eda/testbenches/read_offset_mc.py)")
    ch = json.loads(MC_SUMMARY.read_text())["misread_channel"]
    return {k: (float(ch[k]["p_read_flip"]),
                f"{MC_SUMMARY.name}:misread_channel.{k}.p_read_flip "
                f"[{ch[k]['basis']}]")
            for k in ("as_committed", "conservative")}


def arms(instance, pmap):
    """(arm, axis, basis, p_read_flip, spec) rows for one instance.

    Ladders are deduplicated by flip rate: the 0.5 cap collapses the x3/x10
    points of the conservative ladder onto one another, and that collapse is
    recorded in the arm name rather than run twice.
    """
    rows = [("ideal", "baseline", "-", 0.0, ("ideal", {}))]
    wanted = []
    for basis, (p, _) in pmap.items():
        for mult in (1, 3, 10):
            # 0.5 is the maximum-entropy point of a binary flip channel (the
            # reported state carries no information); beyond it the channel is
            # anti-correlated, which no read path realizes, so the axis caps.
            q = min(P_FLIP_CAP, mult * p)
            tag = f"{basis}_x{mult}" + ("_cap" if mult * p > P_FLIP_CAP else "")
            wanted.append((tag, "read_flip", basis, q))
    # tolerance knee: where between the two bounding rates does the solver
    # start to care?  Chosen a priori, not tuned to the outcome.
    for q in ([1e-3, 1e-2, 3e-2] if instance == "ER14"
              else [1e-5, 1e-4, 3e-4, 1e-2]):
        wanted.append((f"tolerance_{q:g}", "tolerance", "sweep", q))

    seen = {}
    for tag, axis, basis, q in wanted:
        key = round(q, 12)
        if key in seen:
            i = seen[key]
            rows[i] = (rows[i][0] + "+" + tag,) + rows[i][1:]
            continue
        seen[key] = len(rows)
        rows.append((tag, axis, basis, q,
                     ("circuit_chain", dict(mode="none", p_read_flip=q))))

    if instance == "G1":
        # RX-04 constraint: combining the flip channel with the DAC backend at
        # G-set scale requires rail half-width >= +/-10 V_T.
        rows.append((f"dac_span{GSET_SPAN:g}_noflip", "dac_combined", "-", 0.0,
                     ("circuit_chain", dict(mode="fixed_u", nbits=GSET_BITS,
                                            u_span=GSET_SPAN,
                                            p_read_flip=0.0))))
        for basis in ("as_committed", "conservative"):
            if basis not in pmap:
                continue
            q = min(P_FLIP_CAP, pmap[basis][0])
            rows.append((f"dac_span{GSET_SPAN:g}_{basis}_x1", "dac_combined",
                         basis, q,
                         ("circuit_chain", dict(mode="fixed_u",
                                                nbits=GSET_BITS,
                                                u_span=GSET_SPAN,
                                                p_read_flip=q))))
    return rows


def run_cached(odir, problem, key, spec, proto, jobs, force=False):
    fp = odir / f"{problem.name}_{key}.json"
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("n_trials") == TRIALS and d.get("spec") == repr(spec):
            print(f"  [cache] {problem.name}/{key}", flush=True)
            return d
    cfg = SolverConfig(schedule_shape="geometric", beta0=proto["beta0"],
                       betaf=proto["betaf"], n_sweeps=proto["sweeps"],
                       update_mode="block")
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=TRIALS, master_seed=SEED, n_jobs=jobs,
                     progress=False)
    wall = time.perf_counter() - t0
    d = dict(instance=problem.name, arm=key, spec=repr(spec),
             n_trials=TRIALS, n_sweeps=proto["sweeps"],
             beta=[proto["beta0"], proto["betaf"]], master_seed=SEED,
             update_mode="block", wall_s=wall,
             seed_derivation=f"np.random.SeedSequence({SEED}).spawn({TRIALS})[i]",
             energies=[float(r.energy_final) for r in res])
    fp.write_text(json.dumps(d))
    print(f"  [saved] {problem.name}/{key} ({wall:.0f}s)", flush=True)
    return d


def summarize(problem, target, proto, arm, axis, basis, pflip, d, base):
    e = np.asarray(d["energies"], dtype=np.float64)
    k = int(round(p_success(e, target, atol=TOL) * TRIALS))
    ps = k / TRIALS
    lo, hi = wilson(k, TRIALS)
    tts = tts_at_confidence(float(proto["sweeps"]), ps, 0.99)
    if base is None:
        ratio = 1.0
        r_lo = r_hi = float("nan")
        dmed = d_lo = d_hi = 0.0
    else:
        e_base, k_base = base
        if k_base > 0 and k > 0:
            rc = tts_ratio_ci(k_base, TRIALS, k, TRIALS,
                              n_sweeps=proto["sweeps"])
            ratio, r_lo, r_hi = rc["ratio"], rc["lo"], rc["hi"]
        elif k_base > 0:
            ratio, r_lo, r_hi = float("inf"), float("nan"), float("nan")
        else:
            ratio = r_lo = r_hi = float("nan")
        dmed = float(np.median(e) - np.median(e_base))
        d_lo, d_hi = boot_dmedian_ci(e, e_base)
    return dict(
        instance=problem.name, n=problem.n, arm=arm, axis=axis, basis=basis,
        p_read_flip=pflip, n_trials=TRIALS, n_sweeps=proto["sweeps"],
        beta0=proto["beta0"], betaf=proto["betaf"], master_seed=SEED,
        hits=k, p_s=ps, wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
        tts99_sweeps=(round(tts, 1) if np.isfinite(tts) else "inf"),
        tts_ratio_vs_ideal=ratio, ratio_lo=r_lo, ratio_hi=r_hi,
        energy_min=float(e.min()), energy_median=float(np.median(e)),
        dmed_vs_ideal=dmed, dmed_lo=round(d_lo, 4), dmed_hi=round(d_hi, 4),
        target=target, spec=d["spec"], wall_s=round(d["wall_s"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--instances", nargs="+", default=["ER14", "G1"])
    ap.add_argument("--p-read-flip", type=float, default=None)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    pmap = read_measured_p(args.p_read_flip)
    for basis, (p, src) in pmap.items():
        print(f"p_read_flip[{basis}] = {p:.6g}   x3={min(P_FLIP_CAP, 3*p):.6g}"
              f"   x10={min(P_FLIP_CAP, 10*p):.6g}   (cap {P_FLIP_CAP})")
        print(f"    <- {src}")

    odir = HERE / "results_read_flip"
    odir.mkdir(exist_ok=True)
    rows, meta = [], {}
    t_all = time.perf_counter()

    for name in args.instances:
        proto = PROTO[name]
        if name == "ER14":
            prob = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0,
                                    name="ER14_p0.3")
            target = enumerate_ground(prob)
            assert abs(target - ER14_TARGET) < 1e-9, \
                f"ER14 target drift: {target} vs published {ER14_TARGET}"
            meta[name] = dict(kind="random_er_maxcut", n=14, p=0.30, seed=0,
                              target=target, target_kind="ENUM",
                              published_anchor=ER14_TARGET)
        else:
            gdir = ROOT / "gset"
            try:
                from fetch_data import ensure_gset
                ensure_gset([name], gdir)
            except Exception as err:
                print(f"auto-fetch warning: {err}", flush=True)
            prob = load_gset(gdir / name)
            target = prob.meta["edge_sum"] / 2.0 - G1_BKS
            meta[name] = dict(kind="gset", n=prob.n, bks=G1_BKS,
                              target=target, target_kind="BKS")
        print(f"== {name}: n={prob.n} target={target:.6f} "
              f"T={proto['sweeps']} beta {proto['beta0']}->{proto['betaf']}",
              flush=True)

        base = None
        for arm, axis, basis, pflip, spec in arms(name, pmap):
            d = run_cached(odir, prob, arm.split("+")[0], spec, proto,
                           args.jobs, args.force_recompute)
            if arm == "ideal":
                base = (np.asarray(d["energies"], dtype=np.float64),
                        int(round(p_success(np.asarray(d["energies"]), target,
                                            atol=TOL) * TRIALS)))
            row = summarize(prob, target, proto, arm, axis, basis, pflip, d,
                            None if arm == "ideal" else base)
            rows.append(row)
            print(f"  [{arm:>34s}] p_flip={pflip:.6g} hits={row['hits']:>3d}"
                  f"/{TRIALS} p_s={row['p_s']:.3f} "
                  f"CI[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}] "
                  f"ratio={row['tts_ratio_vs_ideal']} "
                  f"E_med={row['energy_median']:.4f} "
                  f"dmed={row['dmed_vs_ideal']:+.4f}", flush=True)

    csv_path = HERE / "read_flip_solver_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "read_flip_solver_config.json").write_text(json.dumps(dict(
        _label=("RX-06 solver leg: read-decision misread channel "
                "(circuit_backends.CircuitChainSpin p_read_flip) on the "
                "Section 3.4.2 (ER14) and Section 3.3 (G1) protocols; all rows "
                "MEASURED solver output, the flip rate itself is derived from "
                "the RX-06 offset MC"),
        p_read_flip_bases={k: dict(value=v[0], source=v[1])
                           for k, v in pmap.items()},
        multipliers=[1, 3, 10], p_flip_cap=P_FLIP_CAP,
        p_flip_cap_note=("0.5 is the maximum-entropy point of a binary flip "
                         "channel (the reported state carries no information); "
                         "requested multiples above it are capped, so a capped "
                         "row is the worst physically meaningful case rather "
                         "than the literal multiple"),
        protocols=PROTO, instances=meta,
        trials=TRIALS, master_seed=SEED, update_mode="block",
        read_flip_arms_mode="none (no DAC quantization, no rail clip) so the "
                            "degradation is attributable to the misread "
                            "channel alone",
        dac_combined_note=(f"RX-04 constraint: combining the read-flip channel "
                           f"with the DAC backend on G-set-scale instances "
                           f"requires rail half-width >= +/-10 V_T; +/-4 V_T "
                           f"sits in the p_s=0 regime. Arms use "
                           f"u_span={GSET_SPAN:g}, nbits={GSET_BITS}, run as a "
                           f"no-flip / flip pair so the marginal cost of the "
                           f"misread is isolated."),
        stats="Wilson 95% CI on p_s; parametric-bootstrap 95% CI on TTS ratio "
              "vs the same-instance ideal arm (stats.py, RX-01 convention); "
              "bootstrap CI on the energy-median shift",
        boot_B=BOOT_B, boot_seed=BOOT_SEED,
        total_wall_s=round(time.perf_counter() - t_all, 1)), indent=2))
    print(f"-> {csv_path}")
    print(f"total wall: {(time.perf_counter()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
