#!/usr/bin/env python3
"""Circuit-constrained spin backends: feed the measured write-chain limits
back into the Chapter-3 solver (eda -> algorithm, one-way).

Two constraints measured/derived from the eda/ workspace are modeled:

1. DAC quantization + clipping (from eda/testbenches/update_chain_dc.py):
   the Gibbs drive u = 2*beta*h_eff is delivered as a write voltage
   V = Vth + u*VT through a b-bit DAC spanning Vth +/- u_span*VT. The
   realizable drive is therefore CLIPPED to the rail window and QUANTIZED
   to the (measured or ideal) code grid. Two annealing realizations:
     * mode="fixed_u":     rails fixed, codes quantize u directly — the
       digital synapse computes 2*beta*h_eff, resolution is constant in u;
     * mode="beta_scaled": rails ramp with beta (the annealing schedule is
       carried by the DAC reference voltage), codes quantize h_eff on a
       fixed grid h in [-h_clip, +h_clip]; u = 2*beta*h_q.

2. Sticky reset (from the Chapter-2 back-hopping plateau): one Gibbs update
   on the pulse-programmed device is reset-to-P (AP->P direction, single-
   pulse success r_reset <= 0.72 measured) followed by the probabilistic
   P->AP pulse (clean calibrated sigmoid). k reset pulses leave the device
   stuck in AP with probability rho = (1 - r_reset)^k when the previous
   state was AP, breaking update independence:
       P(s'=+1 | s=+1) = rho + (1-rho)*sigma(u)
       P(s'=+1 | s=-1) = sigma(u)
   (+1 is mapped to AP: the probabilistic pulse is the P->AP direction.)

Backends are registered for the block update mode (the JIT path bypasses
sample_batch, same constraint as device_model.BehavioralSMTJSpin).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from isim import SpinBackend, register_spin_backend  # noqa: E402

R_RESET = 0.72          # Chapter-2 measured single-pulse AP->P plateau


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _quantize(x, grid):
    """Clip x to [grid[0], grid[-1]] and snap to the nearest grid point."""
    x = np.clip(x, grid[0], grid[-1])
    k = np.searchsorted(grid, x)
    k = np.clip(k, 1, len(grid) - 1)
    lo = grid[k - 1]
    hi = grid[k]
    return np.where(x - lo < hi - x, lo, hi)


class CircuitChainSpin(SpinBackend):
    """DAC-quantized (+ optionally sticky-reset) Gibbs spin.

    Parameters
    ----------
    u_grid : sequence of float, optional
        Measured drive grid in u units ((V_code - Vth)/VT), e.g. the
        transfer from update_chain_summary.json. Overrides nbits/u_span.
    nbits, u_span : int, float
        Ideal uniform grid: 2**nbits levels over [-u_span, +u_span]
        (fixed_u mode) or over [-h_clip, +h_clip] (beta_scaled mode,
        where u_span is interpreted as h_clip).
    mode : "fixed_u" | "beta_scaled" | "none"
        "none" disables quantization (for isolating the sticky-reset knob).
    n_reset : int
        Number of reset pulses per update; 0 = ideal reset (no stickiness).
    r_reset : float
        Single-pulse reset success probability (default: measured 0.72).
    u_offset : float or sequence
        Static drive offset(s) in u units (e.g. write-line IR drop per
        row, from the extraction flow); scalar or per-spin array.
    """

    kind = "circuit_chain"

    def __init__(self, u_grid=None, nbits=6, u_span=4.0, mode="fixed_u",
                 n_reset=0, r_reset=R_RESET, u_offset=0.0):
        if mode not in ("fixed_u", "beta_scaled", "none"):
            raise ValueError(f"bad mode: {mode}")
        self.mode = mode
        if u_grid is not None:
            self.grid = np.sort(np.asarray(u_grid, dtype=np.float64))
        elif mode != "none":
            self.grid = np.linspace(-float(u_span), float(u_span),
                                    2 ** int(nbits))
        else:
            self.grid = None
        self.n_reset = int(n_reset)
        self.r_reset = float(r_reset)
        self.rho = (1.0 - self.r_reset) ** self.n_reset if self.n_reset else 0.0
        self.u_offset = np.asarray(u_offset, dtype=np.float64)

    # -- drive path ---------------------------------------------------------
    def _p_plus(self, h_eff_arr, beta, idx=None):
        if self.mode == "beta_scaled":
            h = np.asarray(h_eff_arr, dtype=np.float64)
            u = 2.0 * beta * _quantize(h, self.grid)
        elif self.mode == "fixed_u":
            u = _quantize(2.0 * beta * np.asarray(h_eff_arr, np.float64),
                          self.grid)
        else:
            u = 2.0 * beta * np.asarray(h_eff_arr, dtype=np.float64)
        off = self.u_offset
        if off.ndim > 0 and idx is not None:
            off = off[idx]
        return _sigmoid(u + off)

    # -- SpinBackend API ----------------------------------------------------
    def sample(self, h_eff, s_cur, beta, rng):
        p = float(self._p_plus(np.array([h_eff]), beta)[0])
        if self.rho > 0.0 and s_cur == 1 and rng.random() < self.rho:
            return 1                          # reset failed: stuck in AP
        return 1 if rng.random() < p else -1

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        p = self._p_plus(h_eff_arr, beta, idx=idx)
        fresh = np.where(rng.random(size=p.shape) < p, 1, -1).astype(np.int8)
        if self.rho <= 0.0:
            return fresh
        stuck = (np.asarray(s_cur_arr) == 1) & (rng.random(size=p.shape)
                                                < self.rho)
        return np.where(stuck, np.int8(1), fresh)


def _factory(**kwargs):
    return CircuitChainSpin(**kwargs)


register_spin_backend("circuit_chain", _factory)
