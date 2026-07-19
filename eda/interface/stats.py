#!/usr/bin/env python3
"""RX-01 — statistics helpers for the chapter-3 tables (shared convention).

Every p_s that supports a ratio claim gets a Wilson 95% interval; TTS ratios
get a parametric-bootstrap interval over the binomial hit counts. Success
counts are exchangeable Bernoulli outcomes across trials (independent seeds
via SeedSequence.spawn), so the parametric bootstrap over (k, n) is exact for
p_s-derived quantities — per-trial JSONs are not needed for these intervals.

TTS convention matches isim.tts_at_confidence: TTS = t_unit * ln(0.01) /
ln(1 - p_s), infinite at p_s = 0.
"""
from __future__ import annotations

import math

import numpy as np


def wilson(k, n, conf=0.95):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(conf, 2))
    if z is None:
        raise ValueError(f"unsupported confidence {conf}")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _tts_sweeps(p, n_sweeps):
    with np.errstate(divide="ignore"):
        return np.where(p > 0,
                        n_sweeps * np.log(0.01) / np.log1p(-np.minimum(p, 1 - 1e-12)),
                        np.inf)


def tts_ratio_ci(k_a, n_a, k_b, n_b, n_sweeps=1, B=10000, conf=0.95,
                 seed=20260719):
    """Bootstrap CI for TTS_b / TTS_a (ratio > 1 means dynamics a faster).

    Parametric bootstrap: resample hit counts from Binomial(n, k/n) for both
    arms, form the TTS ratio per replicate. Replicates where either arm draws
    zero hits are kept as inf/0 and handled by percentile logic, so the
    interval honestly widens when an arm sits near p_s = 0.

    Returns dict(ratio, lo, hi, frac_undefined) where frac_undefined is the
    share of replicates with an infinite/zero ratio (both-arms-zero excluded
    from the percentile but counted here).
    """
    rng = np.random.default_rng(seed)
    pa, pb = k_a / n_a, k_b / n_b
    ka = rng.binomial(n_a, pa, size=B)
    kb = rng.binomial(n_b, pb, size=B)
    ta = _tts_sweeps(ka / n_a, n_sweeps)
    tb = _tts_sweeps(kb / n_b, n_sweeps)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = tb / ta
    finite = np.isfinite(r)
    point = (_tts_sweeps(np.array([pb]), n_sweeps)[0]
             / _tts_sweeps(np.array([pa]), n_sweeps)[0]) \
        if pa > 0 and pb > 0 else float("nan")
    if finite.sum() == 0:
        return dict(ratio=point, lo=float("nan"), hi=float("nan"),
                    frac_undefined=1.0)
    alpha = (1 - conf) / 2
    lo, hi = np.quantile(r[finite], [alpha, 1 - alpha])
    return dict(ratio=float(point), lo=float(lo), hi=float(hi),
                frac_undefined=float(1.0 - finite.mean()))


def fmt_ci(lo, hi, digits=3):
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"
