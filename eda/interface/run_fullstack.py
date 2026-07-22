#!/usr/bin/env python3
"""RX-07 — the whole stack at once: composition, interactions, joint mismatch.

Section 3.4.2 and Section 3.5.2/3.5.4 both present their non-idealities one
factor at a time, each against the same ideal baseline. A real array carries
all of them simultaneously. Three questions follow, one per part:

(a) COMPOSITION. Does the measured full stack cost what the product of the
    individual single-channel costs predicts? Seven channels are switched on
    together — the five behavioural device knobs of Section 3.4.2 at their
    Chapter-2 values, the measured write-DAC grid with its rails, the
    LLG-informed reset residual (RX-05) and the read-misread channel (RX-06)
    — and the resulting TTS_99 ratio is compared against the product of the
    seven one-at-a-time ratios measured on the identical protocol. Run on the
    Section 3.4.2 anchor (ER14) and on G1.

    At the Chapter-2 operating point three of the five device knobs sit at
    their ideal value (h_off = 0 and sigma_C2C = 0 in the P->AP direction,
    p_max = 1 because the reset-write scheme moves the plateau onto the reset
    channel), so the realistic stack has four live channels. A second,
    STRESS stack puts every one of the seven channels at a level that costs
    something on its own (g_dev = 0.7, h_off = 0.1, sigma_C2C = 1.0 — the
    Section 3.4.2 scan points), which is where a composition rule can
    actually be falsified.

(b) INTERACTIONS. A 2^(6-2) resolution-IV fractional factorial over the six
    channels that are live at array scale, response log(TTS_99 ratio),
    parametric-bootstrap intervals on every main effect and on every
    two-factor-interaction alias group.

(c) JOINT PDK MISMATCH. device_model models device-to-device dispersion as an
    independent Gaussian GAIN. Real mismatch moves (V_th, slope) together.
    Per-device (V_th, slope) pairs extracted from the sibling repository's
    process-variability ensemble are mapped to per-spin (g_i, delta_i) and
    fed through the new backend, against the independent-gain model at
    matched CV and against the 7.7% PDK baseline the chapter quotes.

Protocols are the two comparability anchors of the chapter, unchanged:
  ER14  random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0); geometric beta
        0.1 -> 5.0, T = 2000, block mode, 200 trials, master_seed 2024;
        exact target by enumeration.
  G1    T = 10000, geometric beta 0.1 -> 10.0, 200 trials, master_seed 2024,
        block mode; target = edge_sum/2 - 11624.

Rail half-width at G-set scale is >= +/-10 V_T throughout (RX-04: +/-4 V_T
puts G1 at p_s = 0, where no other channel is observable).

Statistics: RX-01 convention (Wilson 95% on p_s, parametric bootstrap 95% on
every TTS ratio). The factorial adds one extension, documented in
`effects_bootstrap`: run-level success probabilities are drawn from their
Jeffreys posterior Beta(k+1/2, n-k+1/2) rather than resampled from the plug-in
p, so an arm that lands on zero hits contributes a finite one-sided response
instead of dropping out of the design.

Run (Windows python; the multiprocessing guard is mandatory):
  python eda/interface/run_fullstack.py --selfcheck
  python eda/interface/run_fullstack.py --part a b c [--jobs 24]
Writes fullstack_summary.csv, fullstack_effects.csv and fullstack_config.json
next to this script; per-arm trial energies cached under results_fullstack/.
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
import device_model                                             # noqa: F401,E402

TB = ROOT / "eda" / "testbenches"
PDK_SCATTER = Path("D:/Documents/Graduation Project-2026/02MRAMSim/vgsot-sim"
                   "/scripts/07_process_variability/"
                   "device_threshold_scatter.json")

TRIALS, SEED, TOL = 200, 2024, 1e-6
N_BASE = 1000                      # ideal-arm trials for the composition test
D2D_SEED = 2025                    # bench_device_ablation.py uses seed + 1
ER14_TARGET = -6.877554338927609
G1_BKS = 11624
BOOT_B, BOOT_SEED = 10000, 20260722

PROTO = {
    "ER14": dict(sweeps=2000, beta0=0.1, betaf=5.0),
    "G1":   dict(sweeps=10000, beta0=0.1, betaf=10.0),
}

# -- measured / calibrated channel values ------------------------------------
# g_dev: the write chain normalises its drive by V_T = 23.414 mV (the value
# compiled into eda/models/smtj_sot.va and reported by update_chain_dc.py),
# while the Chapter-2 same-batch sigmoid fit of the reference device
# (Device A, P->AP, t_w = 0.75 ns) gives the scale parameter k = 22.43 mV.
# A drive of u chain-units therefore lands on the device as u * 23.414/22.43.
VT_CHAIN_MV = 23.414
VT_DEVICE_MV = 22.43
G_DEV_MEAS = VT_CHAIN_MV / VT_DEVICE_MV            # 1.0439
# reset: RX-05 LLG-informed effective AP residual after k pulses (NOT 0.28^k)
RHO_EFF = {1: 0.30, 2: 0.22, 3: 0.19, 4: 0.13}
RESET_K = 3
# read: RX-06 as-committed basis (the optimistic of the two bounding rates)
P_FLIP = 8.230995366729776e-4
CV_GAIN_PDK = 0.077
GSET_SPAN = 10.0                                   # RX-04 floor at G-set scale
ER14_SPAN = 4.0                                    # the as-built chain
NBITS = 6

# Section 3.4.2 scan points used for the STRESS stack (each individually
# resolvable on at least one of the two instances).
STRESS = dict(g_dev=0.7, h_off=0.1, sigma_c2c=1.0)

# -- (b) factorial ------------------------------------------------------------
# Levels chosen to keep every corner of the design inside the measurable
# regime on G1. The spec's low rail level of +/-6 V_T is not usable here:
# RX-04 measured 63x for that rail ALONE (4/200 hits), so every one of the
# eight low-rail runs would land on zero hits and the design would be fully
# censored. +/-8 (3.0x alone) and +/-12 bracket the +/-10 knee instead.
FACTORS = [
    ("rail",  "u_span",       8.0,   12.0),
    ("bits",  "nbits",        4,     6),
    ("resetk", "rho",         RHO_EFF[1], RHO_EFF[3]),   # k=1 low, k=3 high
    ("flip",  "p_read_flip",  P_FLIP, 0.0),              # on low, off high
    ("cv",    "cv_gain",      CV_GAIN_PDK, 0.0),         # on low, off high
    ("hoff",  "h_off",        0.1,   0.0),               # on low, off high
]
# 2^(6-2) resolution IV, generators E = ABC, F = BCD.
FACTORIAL_GENERATORS = {"cv": ("rail", "bits", "resetk"),
                        "hoff": ("bits", "resetk", "flip")}
ALIAS_GROUPS = [
    ("rail*bits", [("rail", "bits"), ("resetk", "cv")]),
    ("rail*resetk", [("rail", "resetk"), ("bits", "cv")]),
    ("rail*flip", [("rail", "flip"), ("cv", "hoff")]),
    ("rail*cv", [("rail", "cv"), ("bits", "resetk"), ("flip", "hoff")]),
    ("rail*hoff", [("rail", "hoff"), ("flip", "cv")]),
    ("bits*flip", [("bits", "flip"), ("resetk", "hoff")]),
    ("resetk*flip", [("resetk", "flip"), ("bits", "hoff")]),
]


# ===========================================================================
# problem / grid helpers
# ===========================================================================
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


def load_problem(name):
    if name == "ER14":
        prob = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0,
                                name="ER14_p0.3")
        target = enumerate_ground(prob)
        assert abs(target - ER14_TARGET) < 1e-9, "ER14 target drift"
        meta = dict(kind="random_er_maxcut", n=14, p=0.30, seed=0,
                    target=target, target_kind="ENUM")
    else:
        gdir = ROOT / "gset"
        try:
            from fetch_data import ensure_gset
            ensure_gset([name], gdir)
        except Exception as err:
            print(f"auto-fetch warning: {err}", flush=True)
        prob = load_gset(gdir / name)
        target = prob.meta["edge_sum"] / 2.0 - G1_BKS
        meta = dict(kind="gset", n=prob.n, bks=G1_BKS, target=target,
                    target_kind="BKS")
    return prob, target, meta


def measured_grid(span):
    """The measured 6-bit transfer, re-referenced to a +/-span V_T rail pair.

    The as-built chain is a +/-4 V_T design (eda/testbenches/update_chain_dc.py)
    and G-set-scale runs need >= +/-10 V_T (RX-04), so the measured grid has to
    be carried across a rail change. The deviation of the measured taps from
    the ideal uniform grid is kept in ABSOLUTE V_T units rather than scaled
    with the rails, because Section 3.5.1 attributes it to the buffer's
    code-dependent offset (2.8-3.1 mV, identical across bit widths) and not to
    the resistor string, so it does not follow the reference voltage.

    This is an EXTRAPOLATION, labelled as such: a +/-10 V_T chain has not been
    re-simulated, and its buffer would run at 2.5x the output-current swing.
    The ideal-grid arm at the same rails bounds the error of the extrapolation
    from the other side and is run alongside it.
    """
    chain = json.loads((TB / "update_chain_summary.json").read_text())
    u = np.array([row["u"] for row in chain["per_bits"][str(NBITS)]
                  ["transfer"]], dtype=np.float64)
    design = float(chain["u_span_design"])
    dev = u - np.linspace(-design, design, u.size)
    return np.linspace(-span, span, u.size) + dev, dev


def pdk_table(vt_norm_mV=VT_CHAIN_MV):
    """Per-spin (g_i, delta_i) from the sibling process-variability ensemble.

    Source: 02MRAMSim/vgsot-sim/scripts/07_process_variability/
    device_threshold_scatter.json — 16 PDK-mismatch device instances, 48
    switching trials at each of 10 write voltages, per device a threshold
    V_th,i (the P_sw = 0.5 crossing) and a transition width span_i (the
    25% -> 50% voltage interval). One device (index 12) is excluded upstream
    by the plateau gate and is excluded here too.

    Mapping. The DAC delivers V = V_th,nom + u * V_T,chain. Device i responds
    to (V - V_th,i)/V_T,i = g_i * (u - delta_i) with

        g_i     = V_T,nom / V_T,i = span_mean / span_i
        delta_i = (V_th,i - V_th,mean) / V_T,chain

    so the ensemble mean device is the nominal one and only the DISPERSION is
    transferred. delta is a fixed voltage divided by the chain's V_T, hence a
    u-domain (not beta-scaled) offset. g_i needs only the ratio of widths, so
    the conversion constant between span25to50 and the logistic scale
    parameter cancels and never has to be assumed.
    """
    d = json.loads(PDK_SCATTER.read_text())
    excl = set(d.get("excluded_devices", []))
    rows = [r for r in d["rows"] if r["device"] not in excl]
    vth = np.array([r["vth_mV"] for r in rows])
    span = np.array([r["span25to50_mV"] for r in rows])
    sig = np.array([r["sigma_mV"] for r in rows])
    g = span.mean() / span
    delta = (vth - vth.mean()) / float(vt_norm_mV)
    stats = dict(n_devices=len(rows), excluded=sorted(excl),
                 vth_mean_mV=float(vth.mean()), vth_std_mV=float(vth.std(ddof=1)),
                 vth_cv=float(vth.std(ddof=1) / vth.mean()),
                 span_mean_mV=float(span.mean()),
                 span_std_mV=float(span.std(ddof=1)),
                 span_cv=float(span.std(ddof=1) / span.mean()),
                 vth_estimation_sigma_mV_median=float(np.median(sig)),
                 g_mean=float(g.mean()), g_cv=float(g.std(ddof=1) / g.mean()),
                 delta_std_VT=float(delta.std(ddof=1)),
                 delta_min_VT=float(delta.min()), delta_max_VT=float(delta.max()),
                 vt_norm_mV=float(vt_norm_mV))
    return np.column_stack([g, delta]), stats


# ===========================================================================
# run + summarise
# ===========================================================================
def run_cached(odir, problem, key, spec, proto, jobs, force=False,
               trials=TRIALS):
    fp = odir / f"{problem.name}_{key}.json"
    if fp.exists() and not force:
        d = json.loads(fp.read_text())
        if d.get("n_trials") == trials and d.get("spec") == repr(spec):
            print(f"  [cache] {problem.name}/{key}", flush=True)
            return d
    cfg = SolverConfig(schedule_shape="geometric", beta0=proto["beta0"],
                       betaf=proto["betaf"], n_sweeps=proto["sweeps"],
                       update_mode="block")
    t0 = time.perf_counter()
    res = multistart(problem=problem, solver_config=cfg, spin_spec=spec,
                     n_trials=trials, master_seed=SEED, n_jobs=jobs,
                     progress=False)
    wall = time.perf_counter() - t0
    d = dict(instance=problem.name, arm=key, spec=repr(spec), n_trials=trials,
             n_sweeps=proto["sweeps"], beta=[proto["beta0"], proto["betaf"]],
             master_seed=SEED, update_mode="block", wall_s=wall,
             seed_derivation=f"np.random.SeedSequence({SEED}).spawn({trials})[i]",
             energies=[float(r.energy_final) for r in res])
    fp.write_text(json.dumps(d))
    print(f"  [saved] {problem.name}/{key} ({wall:.0f}s)", flush=True)
    return d


def hits_of(d, target):
    e = np.asarray(d["energies"], dtype=np.float64)
    n = int(d["n_trials"])
    return int(round(p_success(e, target, atol=TOL) * n)), e


def summarize(part, instance, n, arm, channel, label, d, target, proto, base):
    k, e = hits_of(d, target)
    nt = int(d["n_trials"])
    ps = k / nt
    lo, hi = wilson(k, nt)
    tts = tts_at_confidence(float(proto["sweeps"]), ps, 0.99)
    if base is None:
        ratio, r_lo, r_hi = 1.0, float("nan"), float("nan")
    else:
        k0 = base
        if k0 > 0 and k > 0:
            rc = tts_ratio_ci(k0, TRIALS, k, nt, n_sweeps=proto["sweeps"])
            ratio, r_lo, r_hi = rc["ratio"], rc["lo"], rc["hi"]
        elif k0 > 0:
            ratio, r_lo, r_hi = float("inf"), float("nan"), float("nan")
        else:
            ratio = r_lo = r_hi = float("nan")
    return dict(part=part, instance=instance, n=n, arm=arm, channel=channel,
                label=label, hits=k, n_trials=nt, p_s=ps,
                wilson_lo=round(lo, 4), wilson_hi=round(hi, 4),
                tts99_sweeps=(round(tts, 1) if np.isfinite(tts) else "inf"),
                tts_ratio_vs_ideal=ratio, ratio_lo=r_lo, ratio_hi=r_hi,
                energy_min=float(e.min()), energy_median=float(np.median(e)),
                n_sweeps=proto["sweeps"], wall_s=round(d["wall_s"], 1))


def _derived(part, instance, n, arm, label, triple, proto):
    """A row computed from other rows rather than from a solver run."""
    v, lo, hi = triple
    return dict(part=part, instance=instance, n=n, arm=arm, channel="all",
                label=label, hits="", n_trials=0, p_s="", wilson_lo="",
                wilson_hi="", tts99_sweeps="", tts_ratio_vs_ideal=v,
                ratio_lo=lo, ratio_hi=hi, energy_min="", energy_median="",
                n_sweeps=proto["sweeps"], wall_s=0.0)


def inactive_row(part, instance, n, arm, channel, label, proto):
    """A channel whose Chapter-2 value IS the ideal value.

    No solver run: the backend reduces to the ideal sampler identically, so
    the ratio is 1 by construction rather than by measurement. Labelled
    ANALYTIC so no reader mistakes it for a measured 1.00.
    """
    return dict(part=part, instance=instance, n=n, arm=arm, channel=channel,
                label=label, hits="", n_trials=0, p_s="", wilson_lo="",
                wilson_hi="", tts99_sweeps="", tts_ratio_vs_ideal=1.0,
                ratio_lo=1.0, ratio_hi=1.0, energy_min="", energy_median="",
                n_sweeps=proto["sweeps"], wall_s=0.0)


def _logratio_draws(kn, p0_log, rng, B):
    """log(TTS(arm)/TTS(ideal)) draws given pre-drawn baseline log(1-p0)."""
    k, n = kn
    p = rng.beta(k + 0.5, n - k + 0.5, size=B)
    return np.log(p0_log / np.log1p(-p))


def compose_ci(k0, n0, kn_full, singles, B=BOOT_B, seed=BOOT_SEED):
    """Product of single-channel TTS ratios, and the interaction factor.

    Returns (product, interaction) as (median, lo, hi) triples, where

        product     = prod_i TTS_i / TTS_ideal
        interaction = (TTS_full / TTS_ideal) / product

    Both are formed INSIDE each bootstrap replicate against a common baseline
    draw, so the baseline uncertainty that the product carries m times and the
    full-stack ratio carries once is correlated rather than compounded, and
    the interaction interval is the honest test of decomposability:
    it excludes 1 exactly when the composition is resolvably non-multiplicative.

    m-1 unshared copies of the baseline remain in the interaction (the product
    references it once per channel), which is why the ideal arm is run at
    N_BASE trials rather than 200.
    """
    rng = np.random.default_rng(seed)
    p0 = rng.beta(k0 + 0.5, n0 - k0 + 0.5, size=B)
    p0_log = np.log1p(-p0)
    logprod = np.zeros(B)
    for kn in singles:
        if kn is None:                      # analytically inactive channel
            continue
        logprod += _logratio_draws(kn, p0_log, rng, B)
    log_full = _logratio_draws(kn_full, p0_log, rng, B)
    q = lambda a: (float(np.median(a)), float(np.quantile(a, 0.025)),   # noqa
                   float(np.quantile(a, 0.975)))
    return q(np.exp(logprod)), q(np.exp(log_full - logprod))


# ===========================================================================
# (b) factorial effects
# ===========================================================================
def design_matrix():
    """16-run 2^(6-2) resolution-IV design in +/-1 coding."""
    base = [f[0] for f in FACTORS[:4]]
    rows = []
    for i in range(16):
        v = {base[j]: (1 if (i >> j) & 1 else -1) for j in range(4)}
        for gen, src in FACTORIAL_GENERATORS.items():
            s = 1
            for f in src:
                s *= v[f]
            v[gen] = s
        rows.append(v)
    return rows


def effects_bootstrap(k_runs, k_ideal, design, n_sweeps, B=BOOT_B,
                      seed=BOOT_SEED):
    """Main effects and 2FI alias-group effects on log(TTS ratio), with CIs.

    Success probabilities are drawn from the Jeffreys posterior
    Beta(k + 1/2, n - k + 1/2) instead of the plug-in bootstrap of
    stats.tts_ratio_ci: a factorial cell that draws zero hits still has a
    finite response, so a censored corner widens the interval instead of
    deleting the run and unbalancing the design.
    """
    rng = np.random.default_rng(seed)
    p0 = rng.beta(k_ideal + 0.5, TRIALS - k_ideal + 0.5, size=B)
    y = np.empty((len(k_runs), B))
    for i, k in enumerate(k_runs):
        p = rng.beta(k + 0.5, TRIALS - k + 0.5, size=B)
        y[i] = np.log(np.log1p(-p0) / np.log1p(-p))
    out = []
    for name, _, lo_v, hi_v in FACTORS:
        col = np.array([r[name] for r in design], dtype=float)
        eff = (y[col > 0].mean(axis=0) - y[col < 0].mean(axis=0))
        q = np.quantile(eff, [0.025, 0.5, 0.975])
        out.append(dict(term=name, kind="main",
                        aliased_with="", low=lo_v, high=hi_v,
                        effect_log=float(q[1]), effect_lo=float(q[0]),
                        effect_hi=float(q[2]),
                        factor=float(np.exp(q[1])),
                        factor_lo=float(np.exp(q[0])),
                        factor_hi=float(np.exp(q[2])),
                        resolved=bool(q[0] > 0 or q[2] < 0)))
    for label, group in ALIAS_GROUPS:
        a, b = group[0]
        col = np.array([r[a] * r[b] for r in design], dtype=float)
        eff = (y[col > 0].mean(axis=0) - y[col < 0].mean(axis=0))
        q = np.quantile(eff, [0.025, 0.5, 0.975])
        out.append(dict(term=label, kind="2fi",
                        aliased_with="=".join(f"{x}*{y_}" for x, y_ in group[1:]),
                        low="", high="",
                        effect_log=float(q[1]), effect_lo=float(q[0]),
                        effect_hi=float(q[2]),
                        factor=float(np.exp(q[1])),
                        factor_lo=float(np.exp(q[0])),
                        factor_hi=float(np.exp(q[2])),
                        resolved=bool(q[0] > 0 or q[2] < 0)))
    return out


# ===========================================================================
# self-check: the new backend must reproduce both parents bit for bit
# ===========================================================================
def selfcheck(jobs=4):
    from isim import multistart as ms
    prob = random_er_maxcut(n=14, p=0.30, sigma=1.0, seed=0, name="ER14_p0.3")
    cfg = SolverConfig(schedule_shape="geometric", beta0=0.1, betaf=5.0,
                       n_sweeps=500, update_mode="block")

    def energies(spec):
        return np.array([r.energy_final for r in
                         ms(problem=prob, solver_config=cfg, spin_spec=spec,
                            n_trials=12, master_seed=7, n_jobs=1,
                            progress=False)])

    cases = [
        ("ideal", ("ideal", {}), ("full_stack", {})),
        ("behavioral_smtj g_dev/h_off/sigma",
         ("behavioral_smtj", dict(g_dev=0.7, h_off=0.1, sigma_c2c=1.0)),
         ("full_stack", dict(g_dev=0.7, h_off=0.1, sigma_c2c=1.0))),
        ("behavioral_smtj D2D+p_max",
         ("behavioral_smtj", dict(cv_gain=0.077, p_max=0.9, n_spins=14,
                                  d2d_seed=2025)),
         ("full_stack", dict(cv_gain=0.077, p_max=0.9, n_spins=14,
                             d2d_seed=2025))),
        ("circuit_chain DAC",
         ("circuit_chain", dict(mode="fixed_u", nbits=6, u_span=4.0)),
         ("full_stack", dict(mode="fixed_u", nbits=6, u_span=4.0))),
        ("circuit_chain reset+flip",
         ("circuit_chain", dict(mode="none", n_reset=3, p_read_flip=1e-2)),
         ("full_stack", dict(mode="none", n_reset=3, p_read_flip=1e-2))),
    ]
    ok = True
    for label, ref, new in cases:
        a, b = energies(ref), energies(new)
        same = np.array_equal(a, b)
        ok &= same
        print(f"  [{'OK ' if same else 'FAIL'}] {label}: "
              f"max|diff| = {np.abs(a - b).max():.3e}")
    print("selfcheck", "PASSED" if ok else "FAILED")
    return ok


# ===========================================================================
# parts
# ===========================================================================
def part_a(prob, target, proto, odir, jobs, force, rows, specs):
    inst = "ER14" if prob.n == 14 else prob.name
    span = ER14_SPAN if inst == "ER14" else GSET_SPAN
    grid, _dev = measured_grid(span)
    n = prob.n

    ideal = run_cached(odir, prob, "ideal", ("ideal", {}), proto, jobs, force)
    k0, _ = hits_of(ideal, target)
    rows.append(summarize("a", inst, n, "ideal", "-", "MEASURED", ideal,
                          target, proto, None))
    # A better-powered baseline for the composition statistic only. The first
    # TRIALS seeds of SeedSequence(SEED).spawn(N_BASE) are the same objects as
    # spawn(TRIALS), so this arm is a strict superset of the row above.
    ideal_hi = run_cached(odir, prob, f"ideal_n{N_BASE}", ("ideal", {}), proto,
                          jobs, force, trials=N_BASE)
    k0b, e0b = hits_of(ideal_hi, target)
    lo_b, hi_b = wilson(k0b, N_BASE)
    rows.append(dict(part="a", instance=inst, n=n, arm=f"ideal_n{N_BASE}",
                     channel="-", label="MEASURED (baseline for the "
                                        "composition statistic)",
                     hits=k0b, n_trials=N_BASE, p_s=k0b / N_BASE,
                     wilson_lo=round(lo_b, 4), wilson_hi=round(hi_b, 4),
                     tts99_sweeps=round(tts_at_confidence(
                         float(proto["sweeps"]), k0b / N_BASE, 0.99), 1),
                     tts_ratio_vs_ideal=1.0, ratio_lo="", ratio_hi="",
                     energy_min=float(e0b.min()),
                     energy_median=float(np.median(e0b)),
                     n_sweeps=proto["sweeps"],
                     wall_s=round(ideal_hi["wall_s"], 1)))

    live = dict(g_dev=G_DEV_MEAS, cv_gain=CV_GAIN_PDK, n_spins=n,
                d2d_seed=D2D_SEED, mode="fixed_u", u_grid=list(grid),
                rho=RHO_EFF[RESET_K], p_read_flip=P_FLIP)

    # ---- realistic stack: single channels, then all of them together -------
    singles = [
        ("g_dev", dict(g_dev=G_DEV_MEAS),
         f"MEASURED (V_T ratio {VT_CHAIN_MV}/{VT_DEVICE_MV})"),
        ("h_off", None, "ANALYTIC (Chapter-2 P->AP offset is 0)"),
        ("sigma_c2c", None, "ANALYTIC (Chapter-2 slope absorbs C2C)"),
        ("cv_gain", dict(cv_gain=CV_GAIN_PDK, n_spins=n, d2d_seed=D2D_SEED),
         "MEASURED (Brinkman-PDK 7.7% baseline)"),
        ("dac", dict(mode="fixed_u", u_grid=list(grid)),
         f"MEASURED 6-bit grid, rails +/-{span:g} V_T"),
        ("reset", dict(rho=RHO_EFF[RESET_K]),
         f"MEASURED (RX-05 LLG rho_eff k={RESET_K})"),
        ("read_flip", dict(p_read_flip=P_FLIP),
         "MEASURED (RX-06 as-committed)"),
    ]
    kk = []
    for ch, kw, label in singles:
        if kw is None:
            rows.append(inactive_row("a", inst, n, f"single_{ch}", ch, label,
                                     proto))
            kk.append(None)
            continue
        spec = ("full_stack", kw)
        d = run_cached(odir, prob, f"single_{ch}", spec, proto, jobs, force)
        specs[f"{inst}/single_{ch}"] = kw
        k, _ = hits_of(d, target)
        kk.append((k, TRIALS))
        rows.append(summarize("a", inst, n, f"single_{ch}", ch, label, d,
                              target, proto, k0))

    d = run_cached(odir, prob, "fullstack", ("full_stack", live), proto, jobs,
                   force)
    specs[f"{inst}/fullstack"] = live
    kf, _ = hits_of(d, target)
    rows.append(summarize("a", inst, n, "fullstack", "all",
                          "MEASURED (all seven channels)", d, target, proto,
                          k0))
    prod, inter = compose_ci(k0b, N_BASE, (kf, TRIALS), kk)
    rows.append(_derived("a", inst, n, "product_of_singles",
                         f"DERIVED from the single rows (baseline "
                         f"n={N_BASE})", prod, proto))
    rows.append(_derived("a", inst, n, "interaction_factor",
                         "DERIVED fullstack / product_of_singles; excludes 1 "
                         "iff the composition is resolvably "
                         "non-multiplicative", inter, proto))

    # ---- ideal-grid control for the rail extrapolation ---------------------
    spec = ("full_stack", dict(mode="fixed_u", nbits=NBITS, u_span=span))
    d = run_cached(odir, prob, "single_dac_idealgrid", spec, proto, jobs, force)
    rows.append(summarize("a", inst, n, "single_dac_idealgrid", "dac",
                          f"CONTROL ideal {NBITS}-bit grid at +/-{span:g} V_T",
                          d, target, proto, k0))

    # ---- stress stack: every channel at a level that costs something -------
    stress_singles = [
        ("g_dev", dict(g_dev=STRESS["g_dev"]), "Section 3.4.2 scan point"),
        ("h_off", dict(h_off=STRESS["h_off"]), "Section 3.4.2 scan point"),
        ("sigma_c2c", dict(sigma_c2c=STRESS["sigma_c2c"]),
         "Section 3.4.2 scan point"),
    ]
    kk_s = []
    for ch, kw, label in stress_singles:
        spec = ("full_stack", kw)
        d = run_cached(odir, prob, f"stress_single_{ch}", spec, proto, jobs,
                       force)
        k, _ = hits_of(d, target)
        kk_s.append((k, TRIALS))
        rows.append(summarize("a", inst, n, f"stress_single_{ch}", ch, label,
                              d, target, proto, k0))
    kk_s += [kn for kn, (ch, kw, _l) in zip(kk, singles)
             if ch in ("cv_gain", "dac", "reset", "read_flip")]
    stress = dict(live)
    stress.update(g_dev=STRESS["g_dev"], h_off=STRESS["h_off"],
                  sigma_c2c=STRESS["sigma_c2c"])
    # the stressed stack lands near p_s = 0 by construction, so it is the one
    # arm that gets the deeper trial budget: at 200 trials its interval would
    # swamp the composition test it exists to support.
    d = run_cached(odir, prob, "fullstack_stress", ("full_stack", stress),
                   proto, jobs, force, trials=N_BASE)
    specs[f"{inst}/fullstack_stress"] = stress
    kfs, _ = hits_of(d, target)
    rows.append(summarize("a", inst, n, "fullstack_stress", "all",
                          "MEASURED (all seven channels, stressed)", d,
                          target, proto, k0))
    prod, inter = compose_ci(k0b, N_BASE, (kfs, N_BASE), kk_s)
    rows.append(_derived("a", inst, n, "product_of_singles_stress",
                         f"DERIVED from the stress single rows (baseline "
                         f"n={N_BASE})", prod, proto))
    rows.append(_derived("a", inst, n, "interaction_factor_stress",
                         "DERIVED fullstack_stress / "
                         "product_of_singles_stress", inter, proto))
    return k0


def part_b(prob, target, proto, odir, jobs, force, rows, effects, k0):
    inst = "ER14" if prob.n == 14 else prob.name
    n = prob.n
    design = design_matrix()
    k_runs = []
    for i, v in enumerate(design):
        kw = dict(mode="fixed_u", n_spins=n, d2d_seed=D2D_SEED)
        for name, key, lo_v, hi_v in FACTORS:
            kw[key] = hi_v if v[name] > 0 else lo_v
        spec = ("full_stack", kw)
        d = run_cached(odir, prob, f"fact{i:02d}", spec, proto, jobs, force)
        k, _ = hits_of(d, target)
        k_runs.append(k)
        code = "".join(("+" if v[f[0]] > 0 else "-") for f in FACTORS)
        rows.append(summarize("b", inst, n, f"fact{i:02d}", code,
                              "MEASURED", d, target, proto, k0))
    for e in effects_bootstrap(k_runs, k0, design, proto["sweeps"]):
        e = dict(instance=inst, **e)
        effects.append(e)
    return k_runs


def part_c(prob, target, proto, odir, jobs, force, rows, k0, pdk_stats):
    inst = "ER14" if prob.n == 14 else prob.name
    n = prob.n
    span = ER14_SPAN if inst == "ER14" else GSET_SPAN
    tab, st = pdk_table()
    pdk_stats.update(st)
    tab_l = [[float(a), float(b)] for a, b in tab]
    lsb = 2.0 * span / (2 ** NBITS - 1)

    arms = [
        ("pdk_joint", dict(pdk_table=tab_l, pdk_mode="joint", n_spins=n,
                           pdk_seed=SEED),
         "joint (V_th, slope) resampled from the 15-device ensemble"),
        ("pdk_shuffled", dict(pdk_table=tab_l, pdk_mode="shuffled", n_spins=n,
                              pdk_seed=SEED),
         "same marginals, pairing destroyed (correlation control)"),
        ("pdk_gain_only", dict(pdk_table=tab_l, pdk_mode="gain_only",
                               n_spins=n, pdk_seed=SEED),
         "empirical gain marginal, no threshold channel"),
        ("pdk_offset_only", dict(pdk_table=tab_l, pdk_mode="offset_only",
                                 n_spins=n, pdk_seed=SEED),
         "empirical threshold channel, no gain dispersion"),
        ("indep_gauss_matched",
         dict(cv_gain=float(st["g_cv"]), n_spins=n, d2d_seed=D2D_SEED),
         f"device_model independent Gaussian gain at matched CV="
         f"{st['g_cv']:.3f}"),
        ("pdk_calibrated",
         dict(pdk_table=tab_l, pdk_mode="calibrated", n_spins=n,
              pdk_seed=SEED, trim_lsb=lsb, trim_range=span),
         f"per-row trim on the {NBITS}-bit grid (LSB {lsb:.3f} V_T, "
         f"range +/-{span:g} V_T)"),
        # The threshold channel is expressed in chain V_T units, which assumes
        # the ensemble's devices have the reference device's transition width.
        # The ensemble's own widths are 2.25x wider (they are extracted from a
        # 10-point scan whose finest step, 62 mV, already exceeds the mean
        # 25->50 span), so normalising by the ensemble's own V_T is the
        # optimistic bound on the same data.
        ("pdk_offset_ensemble_VT",
         dict(pdk_table=[[1.0, float(b)] for b in
                         (tab[:, 1] * VT_CHAIN_MV
                          / (float(st["span_mean_mV"]) / np.log(3.0)))],
              pdk_mode="offset_only", n_spins=n, pdk_seed=SEED),
         "threshold channel normalised by the ensemble's own V_T "
         "(optimistic bound)"),
        # ... and the two arms that ask whether the trim is affordable at all:
        # an untrimmed threshold offset eats the drive window it shares with
        # the DAC, and a trimmed one only works where the code range reaches.
        ("pdk_joint_dac",
         dict(pdk_table=tab_l, pdk_mode="joint", n_spins=n, pdk_seed=SEED,
              mode="fixed_u", nbits=NBITS, u_span=span),
         f"joint mismatch behind the {NBITS}-bit DAC at +/-{span:g} V_T"),
        ("pdk_calibrated_dac",
         dict(pdk_table=tab_l, pdk_mode="calibrated", n_spins=n,
              pdk_seed=SEED, trim_lsb=lsb, trim_range=span,
              mode="fixed_u", nbits=NBITS, u_span=span),
         f"trimmed mismatch behind the {NBITS}-bit DAC at +/-{span:g} V_T"),
    ]
    for arm, kw, label in arms:
        d = run_cached(odir, prob, arm, ("full_stack", kw), proto, jobs, force)
        rows.append(summarize("c", inst, n, arm, "d2d", label, d, target,
                              proto, k0))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--part", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--instances", nargs="+", default=["ER14", "G1"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)

    odir = HERE / "results_fullstack"
    odir.mkdir(exist_ok=True)
    rows, effects, specs, pdk_stats = [], [], {}, {}
    t0 = time.perf_counter()

    for name in args.instances:
        prob, target, meta = load_problem(name)
        proto = PROTO[name]
        print(f"== {name}: n={prob.n} target={target:.6f} T={proto['sweeps']}",
              flush=True)
        k0 = None
        if "a" in args.part:
            k0 = part_a(prob, target, proto, odir, args.jobs,
                        args.force_recompute, rows, specs)
        if k0 is None:
            d = run_cached(odir, prob, "ideal", ("ideal", {}), proto,
                           args.jobs, args.force_recompute)
            k0, _ = hits_of(d, target)
        if "b" in args.part:
            part_b(prob, target, proto, odir, args.jobs,
                   args.force_recompute, rows, effects, k0)
        if "c" in args.part:
            part_c(prob, target, proto, odir, args.jobs,
                   args.force_recompute, rows, k0, pdk_stats)

    csv_path = HERE / "fullstack_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if effects:
        with open(HERE / "fullstack_effects.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(effects[0].keys()))
            w.writeheader()
            w.writerows(effects)

    _grid, dev = measured_grid(GSET_SPAN)
    (HERE / "fullstack_config.json").write_text(json.dumps(dict(
        _label=("RX-07: (a) full-stack composition vs the product of "
                "single-channel ratios, (b) 2^(6-2) resolution-IV interaction "
                "screen, (c) joint PDK (V_th, slope) mismatch vs the "
                "independent-gain model. All solver rows MEASURED; the "
                "channel VALUES are measured upstream and cited per row."),
        protocols=PROTO, trials=TRIALS, master_seed=SEED, update_mode="block",
        backend="circuit_backends.FullStackSpin (kind='full_stack')",
        channel_values=dict(
            g_dev=dict(value=G_DEV_MEAS, basis="V_T,chain / V_T,device",
                       vt_chain_mV=VT_CHAIN_MV, vt_device_mV=VT_DEVICE_MV,
                       source="eda/models/smtj_sot.va VT parameter (also "
                              "update_chain_summary.json vt_V) over the "
                              "Chapter-2 same-batch sigmoid scale parameter "
                              "k = 22.43 mV (Device A, P->AP, t_w = 0.75 ns)"),
            h_off=dict(value=0.0, basis="Chapter-2 nominal: the P->AP "
                                        "direction carries no write-bias "
                                        "offset"),
            sigma_c2c=dict(value=0.0, basis="Chapter-2 nominal: the fitted "
                                            "logistic slope already absorbs "
                                            "C2C jitter"),
            p_max=dict(value=1.0, basis="the plateau is carried by the reset "
                                        "channel, not by a symmetric clip, "
                                        "under the Section 3.5 reset-write "
                                        "scheme"),
            cv_gain=dict(value=CV_GAIN_PDK, basis="Brinkman-PDK baseline"),
            dac=dict(nbits=NBITS, er14_span_VT=ER14_SPAN,
                     gset_span_VT=GSET_SPAN,
                     source="update_chain_summary.json per_bits.6.transfer",
                     rail_extrapolation=("measured tap deviation kept in "
                                         "absolute V_T units and re-centred "
                                         "on the wider rails; EXTRAPOLATION, "
                                         "bounded by the ideal-grid control "
                                         "arm"),
                     deviation_VT=dict(min=float(dev.min()),
                                       max=float(dev.max()),
                                       mean=float(dev.mean()))),
            reset=dict(k=RESET_K, rho_eff=RHO_EFF[RESET_K],
                       source="RX-05 reset_correlation_llg_summary.json; "
                              "LLG-informed effective residual, NOT 0.28^k"),
            read_flip=dict(value=P_FLIP,
                           source="read_offset_mc_summary.json "
                                  "misread_channel.as_committed")),
        stress_levels=STRESS,
        omitted_channel=dict(
            name="write-line IR residual",
            reason=("RX-09 measured the predistorted residual at "
                    "[0.83, 1.14] with the interval containing 1 on a fully "
                    "occupied array, and the extraction only provides "
                    "profiles for N in {64, 128, 256}, which does not map "
                    "onto n = 800. Including it would add a channel whose "
                    "own single-channel ratio is not resolvable from 1.")),
        factorial=dict(design="2^(6-2) resolution IV, 16 runs",
                       generators={k: list(v) for k, v in
                                   FACTORIAL_GENERATORS.items()},
                       factors=[dict(name=f[0], param=f[1], low=f[2],
                                     high=f[3]) for f in FACTORS],
                       alias_groups={a: ["*".join(p) for p in g]
                                     for a, g in ALIAS_GROUPS},
                       level_note=("the spec's +/-6 V_T low rail was replaced "
                                   "by +/-8: RX-04 measured 63x for that rail "
                                   "alone on G1, so all eight low-rail cells "
                                   "would have been censored at zero hits"),
                       response="log(TTS_99 ratio vs the ideal arm)"),
        pdk_joint_mismatch=dict(source=str(PDK_SCATTER), **pdk_stats),
        arm_specs=specs,
        stats=("Wilson 95% on p_s; parametric bootstrap 95% on TTS ratios "
               "(stats.py, RX-01); Jeffreys-posterior bootstrap for the "
               "product-of-singles and the factorial effects, so zero-hit "
               "arms stay in the design"),
        boot_B=BOOT_B, boot_seed=BOOT_SEED,
        # this invocation may have served most arms from the cache, so the
        # honest compute cost is the sum of the per-arm solver times recorded
        # in results_fullstack/, not the wall time of the last run
        solver_wall_s_total=round(sum(
            json.loads(p.read_text()).get("wall_s", 0.0)
            for p in sorted(odir.glob("*.json"))), 1),
        n_cached_arms=len(list(odir.glob("*.json"))),
        invocation_wall_s=round(time.perf_counter() - t0, 1)), indent=2))
    print(f"-> {csv_path}")
    print(f"total wall: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
