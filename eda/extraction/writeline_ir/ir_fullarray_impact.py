#!/usr/bin/env python3
"""RX-09 — write-line IR predistortion on a FULLY POPULATED >=64-spin array.

Section 3.5.3 measured the IR-drop cost (uncompensated 3.04x -> predistorted
1.06x) by spreading a 14-spin ER instance over the 64 rows of the tile. The
N_par = 64 projection of Table 3.8 assumes 64 OCCUPIED rows, so the per-row
quenched offsets there couple through a 64x64 J matrix, not a 14x14 one. This
driver repeats the measurement with one spin per row.

Instances (both n = 64, both run through the identical scenario grid so the
conclusion does not rest on a single target provenance):

  * PP64_p0.1_eta0.1_s0  (PRIMARY) — planted-partition Max-Cut. A balanced
    random partition sigma (part_seed = 1, drawn independently of the row
    index so the planted signs are uncorrelated with the monotone IR ramp) is
    planted: each of the n(n-1)/2 pairs carries an edge w.p. p = 0.10, weight
    magnitude |N(0,1)|, sign chosen to REWARD the planting (cut pairs get
    w > 0, uncut pairs w < 0), then flipped for a fraction eta = 0.10 of the
    edges to introduce frustration. The planted configuration is therefore a
    constructed candidate optimum, and it is VERIFIED here: a long ideal run
    (T = 20000, 200 trials, seed 20260722) never goes below E(sigma) and
    reaches it in a large share of trials. Target label PLANTED_VERIFIED.
    eta was fixed by the same probe over {0.10, 0.15, 0.20, 0.30}: eta >= 0.15
    is beaten by the long run (planting not clean -> unusable as an exact
    target), eta = 0.10 is clean AND leaves the ideal baseline at p_s ~ 0.47,
    i.e. away from both 0 and 1, which is what the TTS ratios need.
  * ER64_p0.1 (CROSS-CHECK) — random_er_maxcut(n=64, p=0.10, sigma=1.0,
    seed=0), the same instance RX-05c used, with the same LONGRUN_BEST target
    convention (T = 20000, 200 trials, seed 20260720) as
    eda/interface/run_reset_mechanism.py. Its p_s rows are hit rates at the
    reference energy, NOT certified ground-state probabilities.

The two instances turn out to be a MATCHED PAIR, and the driver asserts it:
both draw their edge mask and weight magnitudes from default_rng(0) in the
same order, so they share the same graph and the same |J| entrywise (max
entrywise difference 0) and differ ONLY in edge signs (43.7% of edges agree).
Whatever separates their responses to the same offset profile is therefore a
property of the sign structure — planted and near-unfrustrated versus random
and frustrated — not of degree, weight scale, or median |h_eff| (0.835 for
both under random configurations).

Offset profiles: eda/extraction/writeline_ir/analyze_ir.py per_row_profile()
is imported and called directly with the sheet resistance and DAC LSB read
back from the COMMITTED ir_drop_summary.json provenance block, so N = 64 and
N = 256 reproduce the committed rows exactly (asserted at run time) and
N = 128 — absent from the committed N_LIST = (16, 64, 256) — is an ANALYTIC
EXTENSION of the same committed formula, not a new measurement.

Spin -> row mapping: the 64 spins occupy 64 rows spread evenly over the N-row
write line, row index k_i = round(i * (N - 1) / 63) (0-based into the rows
list, i.e. analyze_ir row r = k_i + 1). For N = 64 this is the IDENTITY —
every row of the tile carries exactly one spin, which is the configuration
the Table-3.8 projection assumes. For N = 128 / 256 the instance still has 64
spins, so the mapping samples the larger array's ramp uniformly; the reported
N = 128 / 256 rows therefore answer "how does a 64-spin problem behave when
the drive deficit spans the range of a taller line", not "what does a
256-spin problem do".

Sign convention (ir_drop_summary.json): u_off is stored POSITIVE = drive
deficit, the row-r update probability is sigma(u - u_off(r)); the solver
feed-in is u_offset = -u_off (uncompensated) or -resid_u (predistorted).

Scenarios x profiles:
  none / uncompensated / predistorted  x  N in {64, 128, 256}
The primary and crosscheck blocks run mode="none" (pure offset injection, no
DAC grid), so the offset effect is not confounded by the rail clip — RX-04
showed the +/-4 V_T rail alone is fatal at array scale. A "combined" block
adds the update-DAC grid at the RX-04-compliant rail half-width +/-10 V_T
(mode="fixed_u", nbits=6, u_span=10) on both instances.

A fourth block sweeps the PREDISTORTION code step itself. The residual that
survives compensation is LSB/2, so it is set by the step of whichever code
path applies the per-row offset: at the committed 6-bit / +/-4 V_T update DAC
the step is the measured 3.0767 mV (residual <= 0.066 u, independent of N),
but RX-04 inverted the design rule toward 4 bit over +/-10 V_T rails, which
would coarsen the step by ~10x. The block therefore re-derives the residual
profile for {6b/+/-4 V_T (measured, = the predistorted arm above), 6b, 5b,
4b over +/-10 V_T (analytic ideal LSB)} and re-runs N in {64, 256} on both
instances, answering how fine the compensation step must be for the recovery
to survive the resolution-for-range trade.

Protocol: Section 3.4.2 anchor — SolverConfig(geometric, beta 0.1 -> 5.0,
n_sweeps = 2000, update_mode = "block"), master_seed = 2024. Every arm is run
at BOTH n_trials = 200 (the protocol anchor, directly comparable to the
14-spin numbers quoted in 3.5.3) and n_trials = 1000 (high power: seeds come
from SeedSequence(2024).spawn(n), whose first 200 children are identical, so
the 1000-trial arm is a strict superset of the 200-trial arm, not an
independent replicate). p_s carries a Wilson 95% interval, TTS ratios vs the
same-instance zero-offset baseline carry parametric-bootstrap intervals
(eda/interface/stats.py, RX-01 convention); the energy-median shift vs the
baseline carries its own bootstrap interval so arms that reach p_s = 0 still
report a signed effect.

Run (Windows python, pure Python):
  python eda/extraction/writeline_ir/ir_fullarray_impact.py [--jobs 24]
Writes ir_fullarray_summary.csv + ir_fullarray_config.json next to this file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))                        # isim, problems
sys.path.insert(0, str(ROOT / "eda" / "interface"))  # circuit_backends, stats
sys.path.insert(0, str(HERE))                        # analyze_ir

from isim import (SolverConfig, multistart, summarize_runs,   # noqa: E402
                  preprocess, Problem)
from problems import random_er_maxcut                          # noqa: E402
from stats import wilson, tts_ratio_ci                         # noqa: E402
from analyze_ir import per_row_profile, VT                     # noqa: E402
import circuit_backends                                        # noqa: F401,E402

SWEEPS, BETA0, BETAF = 2000, 0.1, 5.0
MASTER_SEED = 2024
TRIAL_LADDER = (200, 1000)
REF_SWEEPS, REF_TRIALS = 20000, 200
REF_SEED_PP, REF_SEED_ER = 20260722, 20260720
N_PROFILES = (64, 128, 256)
N_SPINS = 64
BOOT_B, BOOT_SEED = 10000, 20260722

# planted-partition instance parameters (fixed by the calibration probe)
PP = dict(n=64, p=0.10, eta=0.10, seed=0, part_seed=1)

# predistortion code-step ladder: (label, bits, rail half-width in V_T)
PREDIST_GRID = (("6b_span4_measured", 6, 4.0), ("6b_span10", 6, 10.0),
                ("5b_span10", 5, 10.0), ("4b_span10", 4, 10.0))
PREDIST_N = (64, 256)


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
def planted_partition_maxcut(n, p, eta, seed, part_seed):
    """Planted-partition Max-Cut with a controlled frustration fraction eta.

    Returns (problem, sigma) where sigma is the planted configuration.
    Max-Cut mapping J = -w/2 (problems.random_er_maxcut convention), so the
    Ising energy of a configuration s is E(s) = 0.5 * sum_{i<j} w_ij s_i s_j:
    a pair with w > 0 wants s_i != s_j (cut), w < 0 wants s_i == s_j.
    """
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(part_seed)
    sigma = np.ones(n, dtype=np.float64)
    sigma[prng.permutation(n)[: n // 2]] = -1.0
    tri_i, tri_j = np.triu_indices(n, k=1)
    mask = rng.random(len(tri_i)) < p
    src, dst = tri_i[mask], tri_j[mask]
    mag = np.abs(rng.normal(scale=1.0, size=int(mask.sum())))
    flip = rng.random(int(mask.sum())) < eta
    want = np.where(sigma[src] != sigma[dst], 1.0, -1.0)
    w = mag * want * np.where(flip, -1.0, 1.0)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([-0.5 * w, -0.5 * w])
    J = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    h = np.zeros(n, dtype=np.float64)
    J, h = preprocess(J, h, normalize=False)
    prob = Problem(name=f"PP{n}_p{p:g}_eta{eta:g}_s{seed}", n=n, J=J, h=h)
    prob.meta.update(dict(kind="maxcut", n_edges=int(mask.sum()),
                          n_frustrated_edges=int(flip.sum()),
                          eta=eta, part_seed=part_seed,
                          planted=[int(v) for v in sigma]))
    return prob, sigma


# ---------------------------------------------------------------------------
# measured offset profiles
# ---------------------------------------------------------------------------
def row_index(N):
    """0-based indices into the N-row profile occupied by the 64 spins."""
    return np.rint(np.arange(N_SPINS) * (N - 1) / (N_SPINS - 1)).astype(int)


def offset_profiles():
    """{N: dict(rows_used, u_off, resid_u, label)} on the N_SPINS mapping."""
    summ = json.loads((HERE / "ir_drop_summary.json").read_text())
    rs = summ["provenance"]["met2_sheet_R"]["value_ohm_sq"]
    lsb = summ["provenance"]["dac_lsb"]["value_mV"]
    out = {}
    for N in N_PROFILES:
        prof = per_row_profile(N, rs, lsb)["rows"]
        committed = summ["per_N"].get(str(N))
        if committed is not None:                    # integrity: reproduce
            assert prof == committed["rows"], f"N={N} profile drifted"
            label = "MEASURED-Rs/LSB + committed ANALYTIC per-row model"
        else:
            label = ("ANALYTIC EXTENSION of the committed per-row model "
                     "(N absent from analyze_ir.N_LIST; same Rs and LSB)")
        k = row_index(N)
        out[N] = dict(
            rows_used=[int(prof[j]["r"]) for j in k],
            u_off=np.array([prof[j]["u_off"] for j in k]),
            resid_u=np.array([prof[j]["resid_u"] for j in k]),
            max_comp_code=int(max(prof[j]["comp_code"] for j in k)),
            label=label,
            identity_mapping=bool(N == N_SPINS))
    return out, rs, lsb, summ["provenance"]


def predist_variants(N, rs, lsb_measured):
    """Residual profiles for coarser per-row compensation code steps.

    The surviving residual is LSB/2 of whichever code path carries the per-row
    offset, so the ladder is expressed as (bits, rail half-width in V_T). The
    6-bit / +/-4 V_T entry reuses the MEASURED update-chain LSB and therefore
    reproduces the 'predistorted' arm exactly; the +/-10 V_T entries use the
    ANALYTIC ideal step 2*span*V_T/(2**bits - 1), since no chain was measured
    at those rails.
    """
    k = row_index(N)
    out = []
    for label, bits, span in PREDIST_GRID:
        if bits == 6 and span == 4.0:
            lsb, src = lsb_measured, "MEASURED (update_chain 6-bit, +/-4 V_T)"
        else:
            lsb = 2.0 * span * VT * 1e3 / (2 ** bits - 1)
            src = f"ANALYTIC ideal step for {bits} bit over +/-{span:g} V_T"
        prof = per_row_profile(N, rs, lsb)["rows"]
        out.append(dict(
            label=label, bits=bits, span_uT=span, lsb_mV=lsb, lsb_src=src,
            resid_u=np.array([prof[j]["resid_u"] for j in k]),
            max_comp_code=int(max(prof[j]["comp_code"] for j in k)),
            code_budget=2 ** bits - 1,
            label_full=f"{label} (LSB {lsb:.3f} mV, residual bound "
                       f"{lsb / (2 * VT * 1e3):.4f} u)"))
    return out


# ---------------------------------------------------------------------------
# solver plumbing
# ---------------------------------------------------------------------------
def run_arm(problem, spec, trials, seed, jobs, sweeps=SWEEPS):
    cfg = SolverConfig(schedule_shape="geometric", beta0=BETA0, betaf=BETAF,
                       n_sweeps=sweeps, update_mode="block")
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=trials, master_seed=seed, n_jobs=jobs,
                     progress=False)
    return res, time.perf_counter() - t0


def boot_dmedian_ci(x_arm, x_base, conf=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    xa = np.asarray(x_arm, float)
    xb = np.asarray(x_base, float)
    ma = np.median(xa[rng.integers(0, len(xa), size=(BOOT_B, len(xa)))], axis=1)
    mb = np.median(xb[rng.integers(0, len(xb), size=(BOOT_B, len(xb)))], axis=1)
    d = ma - mb
    a = (1 - conf) / 2
    lo, hi = np.quantile(d, [a, 1 - a])
    return float(lo), float(hi)


def arm_row(block, problem, target, target_kind, scenario, N, prof,
            spec, trials, results, wall, base):
    e = np.array([r.energy_final for r in results])
    s = summarize_runs(results, target=target, sense="min")
    k = int(round(s["p_success"] * trials))
    lo, hi = wilson(k, trials)
    tts = s["tts_99_sweeps"]
    m = np.array([np.mean(r.state_final) for r in results])
    if base is None:
        ratio = 1.0
        rlo = rhi = fud = float("nan")
        dmed = dlo = dhi = 0.0
    else:
        bk, be, btts = base
        ratio = (tts / btts) if (btts and np.isfinite(tts)) else float("inf")
        if bk > 0 and k > 0:
            ci = tts_ratio_ci(bk, len(be), k, trials, n_sweeps=SWEEPS)
            rlo, rhi, fud = ci["lo"], ci["hi"], ci["frac_undefined"]
        else:
            rlo = rhi = float("nan")
            fud = 1.0
        dmed = float(np.median(e) - np.median(be))
        dlo, dhi = boot_dmedian_ci(e, be)
    off = np.asarray(spec[1].get("u_offset", 0.0), float)
    return dict(
        block=block, instance=problem.name, n_spins=problem.n,
        target_kind=target_kind, target=target, scenario=scenario,
        n_profile=(N if N else ""), mapping=("identity" if N == N_SPINS else
                                             ("even-sample" if N else "-")),
        profile_label=(prof["label"] if prof else "-"),
        max_comp_code=(prof["max_comp_code"] if prof else ""),
        dac_mode=spec[1].get("mode", "-"),
        dac_bits=spec[1].get("nbits", ""), dac_span_uT=spec[1].get("u_span", ""),
        max_abs_u_offset=float(np.abs(off).max()) if off.size else 0.0,
        mean_abs_u_offset=float(np.abs(off).mean()) if off.size else 0.0,
        n_trials=trials, n_sweeps=SWEEPS, master_seed=MASTER_SEED,
        hits=k, p_success=k / trials,
        wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
        tts99_sweeps=tts, tts99_ratio=ratio,
        ratio_lo=rlo, ratio_hi=rhi, frac_undefined=fud,
        energy_min=float(e.min()), energy_median=float(np.median(e)),
        dmedian_vs_baseline=dmed, dmed_lo=round(dlo, 4), dmed_hi=round(dhi, 4),
        residual_median=s["residual_median"],
        mean_magnetization=float(m.mean()), wall_s=round(wall, 1))


def scenario_specs(prof, dac):
    """(scenario, spin_spec) triples for one profile under one DAC setting."""
    base = dict(dac)
    return [
        ("no_offset", ("circuit_chain", dict(base))),
        ("uncompensated", ("circuit_chain", dict(base, u_offset=-prof["u_off"]))),
        ("predistorted", ("circuit_chain", dict(base, u_offset=-prof["resid_u"]))),
    ]


def sweep_instance(block, problem, target, target_kind, profiles, dac, jobs,
                   rows):
    """Run the {no offset, uncompensated, predistorted} x N grid.

    Returns {n_trials: (hits, baseline energies, baseline TTS)} so later
    blocks can reuse the same-instance zero-offset reference.
    """
    bases = {}
    for trials in TRIAL_LADDER:
        spec = ("circuit_chain", dict(dac))
        res, wall = run_arm(problem, spec, trials, MASTER_SEED, jobs)
        r = arm_row(block, problem, target, target_kind, "no_offset",
                    None, None, spec, trials, res, wall, None)
        rows.append(r)
        be = np.array([x.energy_final for x in res])
        base = (r["hits"], be, r["tts99_sweeps"])
        bases[trials] = base
        print(f"  [{block} {problem.name} n={trials:4d} "
              f"{'no_offset':>14s}          ] p_s={r['p_success']:.3f} "
              f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
              f"ratio=1.000  E_med={r['energy_median']:.4f} "
              f"({wall:.0f}s)", flush=True)
        for N in N_PROFILES:
            prof = profiles[N]
            for scen, spec in scenario_specs(prof, dac)[1:]:
                res, wall = run_arm(problem, spec, trials, MASTER_SEED, jobs)
                r = arm_row(block, problem, target, target_kind, scen, N,
                            prof, spec, trials, res, wall, base)
                rows.append(r)
                print(f"  [{block} {problem.name} n={trials:4d} "
                      f"{scen:>14s} N={N:3d}] p_s={r['p_success']:.3f} "
                      f"[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
                      f"ratio={r['tts99_ratio']:8.3f} "
                      f"[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] "
                      f"E_med={r['energy_median']:.4f} ({wall:.0f}s)",
                      flush=True)
    return bases


def sweep_predist_grid(problem, target, target_kind, bases, rs, lsb, jobs,
                       rows, trials=1000):
    """How fine must the per-row compensation step be? (RX-04 interaction)"""
    for N in PREDIST_N:
        for v in predist_variants(N, rs, lsb):
            spec = ("circuit_chain", dict(mode="none",
                                          u_offset=-v["resid_u"]))
            res, wall = run_arm(problem, spec, trials, MASTER_SEED, jobs)
            r = arm_row("predist_grid", problem, target, target_kind,
                        f"predistorted_{v['label']}", N,
                        dict(label=v["label_full"],
                             max_comp_code=v["max_comp_code"]),
                        spec, trials, res, wall, bases[trials])
            r.update(predist_bits=v["bits"], predist_span_uT=v["span_uT"],
                     predist_lsb_mV=round(v["lsb_mV"], 4),
                     predist_lsb_src=v["lsb_src"],
                     predist_code_budget=v["code_budget"])
            rows.append(r)
            print(f"  [predist_grid {problem.name} n={trials:4d} "
                  f"{v['label']:>18s} N={N:3d}] LSB={v['lsb_mV']:7.3f} mV "
                  f"max|resid|={np.abs(v['resid_u']).max():.4f} u  "
                  f"code={v['max_comp_code']}/{v['code_budget']}  "
                  f"p_s={r['p_success']:.3f} ratio={r['tts99_ratio']:8.3f} "
                  f"[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] ({wall:.0f}s)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--skip-combined", action="store_true")
    ap.add_argument("--skip-predist-grid", action="store_true")
    args = ap.parse_args()

    profiles, rs, lsb, prov = offset_profiles()
    print("=" * 92)
    for N in N_PROFILES:
        p = profiles[N]
        print(f"N={N:>3}: rows {p['rows_used'][0]}..{p['rows_used'][-1]} "
              f"({'identity' if p['identity_mapping'] else 'even-sample'})  "
              f"max u_off={p['u_off'].max():.3f}  "
              f"max|resid_u|={np.abs(p['resid_u']).max():.4f}  "
              f"max code={p['max_comp_code']}/63  [{p['label'][:28]}...]")
    print("=" * 92, flush=True)

    rows = []
    cfg_meta = {}

    # ---- primary instance: planted partition, exact-by-construction target
    pp, sigma = planted_partition_maxcut(**PP)
    e_planted = float(pp.energy(sigma.astype(np.int8)))
    ref, ref_wall = run_arm(pp, ("ideal", {}), REF_TRIALS, REF_SEED_PP,
                            args.jobs, sweeps=REF_SWEEPS)
    ref_e = np.array([r.energy_final for r in ref])
    ref_min = float(ref_e.min())
    clean = ref_min >= e_planted - 1e-9
    ref_hits = int((ref_e <= e_planted + 1e-6).sum())
    # control: the planted signs must not correlate with the monotone IR ramp
    rho_row = float(np.corrcoef(sigma, np.arange(pp.n))[0, 1])
    print(f"[{pp.name}] E_planted={e_planted:.12f}  longrun_min={ref_min:.12f} "
          f"clean={clean}  ref_hits={ref_hits}/{REF_TRIALS}  "
          f"corr(sigma, row)={rho_row:+.3f}  ({ref_wall:.0f}s)", flush=True)
    if not clean:
        raise SystemExit("planting not clean -> fall back to ER64 only")
    cfg_meta["PP64"] = dict(
        params=PP, name=pp.name, n_edges=pp.meta["n_edges"],
        n_frustrated_edges=pp.meta["n_frustrated_edges"],
        target=e_planted,
        target_kind=(f"PLANTED_VERIFIED(longrun T={REF_SWEEPS},"
                     f"trials={REF_TRIALS},seed={REF_SEED_PP})"),
        longrun_min=ref_min, longrun_hits_at_planted=ref_hits,
        corr_planted_sign_vs_row_index=rho_row,
        planted=pp.meta["planted"],
        note=("planted configuration is the best energy found by an "
              "independent long ideal run; eta=0.10 selected because "
              "eta>=0.15 is beaten by the long run (planting not clean)"))
    tk_pp = cfg_meta["PP64"]["target_kind"]
    base_pp = sweep_instance("primary", pp, e_planted, tk_pp, profiles,
                             dict(mode="none"), args.jobs, rows)

    # ---- cross-check instance: ER64, RX-05c LONGRUN_BEST convention
    er = random_er_maxcut(n=64, p=0.10, sigma=1.0, seed=0, name="ER64_p0.1")
    # matched-pair check: same graph, same |J|, signs only differ
    A, B = np.abs(er.J.toarray()), np.abs(pp.J.toarray())
    assert np.array_equal(A, B), "instances no longer share |J|"
    nz = A != 0
    sign_agree = float((np.sign(er.J.toarray()) ==
                        np.sign(pp.J.toarray()))[nz].mean())
    print(f"[matched pair] |J| identical (max diff "
          f"{np.abs(A - B).max():.1e}); edge-sign agreement "
          f"{sign_agree:.3f}", flush=True)
    ref2, ref2_wall = run_arm(er, ("ideal", {}), REF_TRIALS, REF_SEED_ER,
                              args.jobs, sweeps=REF_SWEEPS)
    ref2_e = np.array([r.energy_final for r in ref2])
    t_er = float(ref2_e.min())
    tk_er = (f"LONGRUN_BEST(T={REF_SWEEPS},trials={REF_TRIALS},"
             f"seed={REF_SEED_ER})")
    print(f"[{er.name}] longrun_best={t_er:.12f} "
          f"hits={int((ref2_e <= t_er + 1e-6).sum())}/{REF_TRIALS} "
          f"({ref2_wall:.0f}s)", flush=True)
    cfg_meta["ER64"] = dict(
        params=dict(n=64, p=0.10, sigma=1.0, seed=0), name=er.name,
        target=t_er, target_kind=tk_er,
        longrun_hits_at_best=int((ref2_e <= t_er + 1e-6).sum()),
        note=("no exact ground truth; p_success rows are hit rates at the "
              "long-run reference energy, identical convention and seed to "
              "eda/interface/run_reset_mechanism.py (RX-05c)"))
    cfg_meta["matched_pair"] = dict(
        abs_J_identical=True, max_abs_J_difference=0.0,
        edge_sign_agreement=sign_agree,
        median_abs_h_eff_random_config=0.835,
        note=("PP64 and ER64 share the same edge set and the same |J| "
              "entrywise (both consume default_rng(0) in the same order for "
              "mask then magnitude) and differ only in edge SIGNS, so the "
              "pair isolates sign structure from degree/weight scale"))
    base_er = sweep_instance("crosscheck", er, t_er, tk_er, profiles,
                             dict(mode="none"), args.jobs, rows)

    # ---- offsets combined with the update DAC at the RX-04-compliant rails
    if not args.skip_combined:
        for inst, tgt, tk in ((pp, e_planted, tk_pp), (er, t_er, tk_er)):
            sweep_instance("combined_6b_span10", inst, tgt, tk, profiles,
                           dict(mode="fixed_u", nbits=6, u_span=10.0),
                           args.jobs, rows)

    # ---- how fine must the per-row compensation step be?
    if not args.skip_predist_grid:
        for inst, tgt, tk, bs in ((pp, e_planted, tk_pp, base_pp),
                                  (er, t_er, tk_er, base_er)):
            sweep_predist_grid(inst, tgt, tk, bs, rs, lsb, args.jobs, rows)

    out = HERE / "ir_fullarray_summary.csv"
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    (HERE / "ir_fullarray_config.json").write_text(json.dumps(dict(
        _label=("RX-09: write-line IR drop + 6-bit per-row predistortion on a "
                "FULLY POPULATED 64-row array (one spin per row), vs the "
                "14-spin mapping of Section 3.5.3"),
        protocol=dict(schedule="geometric", beta=[BETA0, BETAF],
                      n_sweeps=SWEEPS, update_mode="block",
                      master_seed=MASTER_SEED, trial_ladder=list(TRIAL_LADDER),
                      seed_note=("SeedSequence(2024).spawn(n): the first 200 "
                                 "children of the 1000-trial arm are the "
                                 "200-trial arm, so the ladder is a strict "
                                 "power increase, not an independent "
                                 "replicate"),
                      stats=("Wilson 95% CI on p_s; parametric-bootstrap 95% "
                             "CI on the TTS ratio vs the same-instance "
                             "zero-offset baseline; bootstrap 95% CI on the "
                             "energy-median shift (eda/interface/stats.py)")),
        instances=cfg_meta,
        offset_source=dict(
            file="ir_drop_summary.json", met2_sheet_R_ohm_sq=rs,
            dac_lsb_mV=lsb, provenance=prov,
            n_profiles=list(N_PROFILES),
            n128_note=("N=128 is an ANALYTIC EXTENSION: analyze_ir."
                       "per_row_profile(128, Rs, LSB) with the committed Rs "
                       "and LSB; N=64/256 rows are asserted equal to the "
                       "committed ir_drop_summary.json rows at run time"),
            sign_convention=("u_off stored POSITIVE = drive deficit; solver "
                             "feed-in u_offset = -u_off (uncompensated) or "
                             "-resid_u (predistorted)"),
            mapping=("spin i -> rows list index round(i*(N-1)/63), "
                     "r = index+1; identity for N=64, even sampling of the "
                     "taller line for N=128/256"),
            rows_used={str(N): profiles[N]["rows_used"] for N in N_PROFILES},
            max_u_off={str(N): float(profiles[N]["u_off"].max())
                       for N in N_PROFILES},
            max_abs_resid_u={str(N): float(np.abs(profiles[N]["resid_u"]).max())
                             for N in N_PROFILES}),
        blocks=dict(
            primary=("mode='none': pure offset injection, no DAC grid, so "
                     "the offset effect is not confounded by the rail clip"),
            crosscheck="same grid on ER64 (LONGRUN_BEST target)",
            combined_6b_span10=("mode='fixed_u', 6 bit over +/-10 V_T — the "
                                "RX-04-compliant rail half-width; +/-4 V_T "
                                "sits in the p_s=0 regime at array scale; "
                                "run on both instances"),
            predist_grid=(
                "residual profile re-derived for coarser per-row "
                "compensation steps (bits, rail half-width) -> LSB -> "
                "residual bound LSB/(2 V_T); tests whether the RX-04 "
                "resolution-for-range inversion breaks the recovery; "
                "N in {64, 256}, 1000 trials, both instances")),
        predistortion_ladder=[
            dict(label=lab, bits=b, span_uT=s,
                 lsb_mV=(lsb if (b == 6 and s == 4.0)
                         else 2.0 * s * VT * 1e3 / (2 ** b - 1)),
                 residual_bound_u=((lsb if (b == 6 and s == 4.0)
                                    else 2.0 * s * VT * 1e3 / (2 ** b - 1))
                                   / (2 * VT * 1e3)))
            for lab, b, s in PREDIST_GRID],
    ), indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
