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

3. Write-pulse saturation ceiling (RX-05b mechanism control): the
   probabilistic P->AP write pulse itself plateaus in its OWN switching
   direction, p_write = min(sigma(u), write_ceiling); the opposite
   direction is untouched (a write pulse that fails to switch leaves the
   device in P, so P(-1) -> 1 at strongly negative drive as in the ideal
   sigmoid). Two controls built from this flag:
     * asymmetric ceiling  (n_reset=0, write_ceiling=0.72): perfect reset
       + plateaued write. State-independent one-step map
           P(s'=+1) = min(sigma(u), 0.72),  floor untouched.
       Isolates ASYMMETRY (only one transition direction saturates)
       without the two-step reset-write structure.
     * within-scheme symmetric saturation (n_reset=k, write_ceiling=0.72):
       BOTH steps of the reset-write scheme saturate at 0.72 — the reset
       pulse per-pulse success stays r_reset=0.72 (sticky residual
       rho=(1-0.72)^k) AND the write pulse caps at 0.72:
           P(s'=+1 | s=+1) = rho + (1-rho)*min(sigma(u), 0.72)
           P(s'=+1 | s=-1) = min(sigma(u), 0.72).
       Keeps the two-step structure but removes the asymmetry, so the
       four-way comparison {sticky, asym ceiling, within-scheme symmetric,
       behavioral_smtj p_max=0.72 symmetric clip} separates asymmetry
       from two-step structure as the source of the reset benignity.

4. Read-decision misread (RX-06, p_read_flip): the state written into the
   device is recovered by a clocked comparator against a resistive midpoint
   reference, and the two decision margins are only -20.0 mV and +14.3 mV
   (eda/testbenches/read_offset_mc.py). A comparator offset comparable to
   those margins makes the decision report the wrong state with some
   probability per read.

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
    write_ceiling : float
        Saturation ceiling of the probabilistic write pulse in its own
        (P->AP, +1) direction: p = min(sigma(u), write_ceiling), floor
        untouched (state-independent, applied to the write sample).
        1.0 = clean sigmoid (default). See module docstring, item 3.
    rho : float, optional
        Direct override of the k-pulse stuck-in-AP residual probability
        (bypasses the (1-r_reset)**n_reset independence formula) — for
        feeding a MEASURED conditional failure chain, e.g. the RX-05a
        LLG P(still AP after k pulses), into the solver.
    p_read_flip : float
        Per-read probability that the comparator reports the opposite of
        the state the device actually holds (RX-06). Implemented by
        flipping the RETURNED sample: the sampled state is drawn from the
        correct write distribution, then inverted with probability
        p_read_flip before it is handed back to the solver.

        Modeling assumption, stated precisely. What a misread physically
        corrupts is the value the REST OF THE ARRAY sees: the neighbours'
        h_eff is accumulated from the reported state, so a misread spin
        contributes the wrong sign to every neighbour's field for that
        sweep. Flipping the returned sample reproduces exactly that, since
        the solver stores the returned value and builds all subsequent
        h_eff from it.

        This conflates "the stored state is wrong" with "the reported
        state is wrong": in the real chain the magnetisation is unchanged
        and only the readout errs, so the NEXT update of the same spin
        starts from the true state, whereas here the corrupted value is
        also the s_cur fed to the sticky-reset branch on the following
        update. The conflation is adequate and conservative in this
        regime. Adequate: h_eff enters the update only through the
        reported neighbour states, so the algorithmic channel — wrong
        field seen by neighbours — is modeled exactly; the stored/reported
        distinction only matters through s_cur, which is used solely by
        the sticky-reset branch (inactive whenever n_reset=0, as in the
        pure read-flip runs). Conservative: a corrupted stored value
        persists for one extra update instead of being corrected at the
        next read, so the model can only overstate the damage.

        A second, opposite-signed caveat that this parameter cannot
        capture: the measured misread rate comes from STATIC device
        mismatch, so it is a population average. A real array has a small
        fraction of spins whose comparator offset exceeds the margin and
        which therefore misread on EVERY read (a per-spin stuck/inverted
        channel, closer to a large h_off on those rows), while the rest
        never misread. The i.i.d. per-read flip modeled here reproduces
        the average corruption rate but not that stickiness, and unlike
        sigma_C2C the static channel does not average away over sweeps.
    """

    kind = "circuit_chain"

    def __init__(self, u_grid=None, nbits=6, u_span=4.0, mode="fixed_u",
                 n_reset=0, r_reset=R_RESET, u_offset=0.0,
                 write_ceiling=1.0, rho=None, p_read_flip=0.0):
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
        if rho is not None:
            self.rho = float(rho)
        self.u_offset = np.asarray(u_offset, dtype=np.float64)
        self.write_ceiling = float(write_ceiling)
        if not 0.0 < self.write_ceiling <= 1.0:
            raise ValueError(f"bad write_ceiling: {write_ceiling}")
        self.p_read_flip = float(p_read_flip)
        if not 0.0 <= self.p_read_flip <= 1.0:
            raise ValueError(f"bad p_read_flip: {p_read_flip}")

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
        p = _sigmoid(u + off)
        if self.write_ceiling < 1.0:
            p = np.minimum(p, self.write_ceiling)
        return p

    # -- SpinBackend API ----------------------------------------------------
    def _readback(self, s, rng, shape=None):
        """Apply the read-decision misread channel to the returned sample(s).

        RNG is only drawn when the channel is active, so p_read_flip=0 keeps
        every previously committed result bit-identical.
        """
        if self.p_read_flip <= 0.0:
            return s
        if shape is None:
            return -s if rng.random() < self.p_read_flip else s
        flip = rng.random(size=shape) < self.p_read_flip
        return np.where(flip, -s, s).astype(np.int8)

    def sample(self, h_eff, s_cur, beta, rng):
        p = float(self._p_plus(np.array([h_eff]), beta)[0])
        if self.rho > 0.0 and s_cur == 1 and rng.random() < self.rho:
            return self._readback(1, rng)     # reset failed: stuck in AP
        return self._readback(1 if rng.random() < p else -1, rng)

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        p = self._p_plus(h_eff_arr, beta, idx=idx)
        fresh = np.where(rng.random(size=p.shape) < p, 1, -1).astype(np.int8)
        if self.rho <= 0.0:
            return self._readback(fresh, rng, shape=p.shape)
        stuck = (np.asarray(s_cur_arr) == 1) & (rng.random(size=p.shape)
                                                < self.rho)
        return self._readback(np.where(stuck, np.int8(1), fresh), rng,
                              shape=p.shape)


def _factory(**kwargs):
    return CircuitChainSpin(**kwargs)


register_spin_backend("circuit_chain", _factory)


# ---------------------------------------------------------------------------
# RX-07: the whole stack at once.
# ---------------------------------------------------------------------------
class FullStackSpin(SpinBackend):
    """Every measured channel of Chapters 2 and 3 in one update.

    Section 3.4.2 scans five behavioural device knobs one at a time and
    Section 3.5.2/3.5.4 scans the circuit channels (DAC grid + rails, reset
    residual, read misread) one at a time, each against the same ideal
    baseline. A real array carries all of them simultaneously. This backend is
    the composition, ordered as the physical chain is:

        u_dac = quantize_clip(2 beta h_eff)          digital synapse + write DAC
        u_dev = g_dev g_i g_i^pdk (u_dac - delta_i)
                + 2 beta (h_off + h_off_i) + u_off   device transfer + static IR
        u_dev += eps,  eps ~ N(0, sigma_c2c^2)       cycle-to-cycle jitter
        p     = clip(sigma(u_dev), 1-p_max, p_max)   back-hopping plateau
        p     = min(p, write_ceiling)                write-pulse ceiling
        s     = Bernoulli(p), then sticky reset, then read misread

    Two conventions are inherited unchanged so that every arm stays
    numerically comparable with the committed single-channel rows:

    * h_off is an h-domain field (multiplied by 2 beta), exactly as in
      device_model.BehavioralSMTJSpin. That is what makes it an ABSOLUTE
      additive field whose relative weight falls as the degree-scaled
      |h_eff| grows (the Section 3.6 statement).
    * delta_i and u_offset are u-domain (V_T-normalized) voltage offsets,
      NOT beta-scaled, exactly as in CircuitChainSpin.u_offset. A device
      threshold shift is a fixed voltage, so it enters the delivered drive
      once, undivided by temperature.

    RNG draw order is byte-compatible with both parent backends: the
    Gaussian jitter draw first (only when sigma_c2c > 0), then the write
    uniform, then the sticky-reset uniform (only when rho > 0), then the
    read-flip uniform (only when p_read_flip > 0). Setting the device knobs
    to ideal reproduces CircuitChainSpin bit for bit; setting the circuit
    knobs to ideal reproduces BehavioralSMTJSpin bit for bit. Both
    equivalences are asserted by run_fullstack.py --selfcheck.

    Joint PDK mismatch (RX-07c). device_model models device-to-device
    dispersion as an independent Gaussian GAIN, g_i ~ N(1, CV^2), with no
    threshold channel. Real mismatch moves (V_th, slope) together. Passing
    `pdk_table` = [[g_i, delta_i], ...] draws per-spin pairs from that
    empirical table instead:

        pdk_mode="joint"        resample (g, delta) pairs together, keeping
                                whatever correlation the table carries
        pdk_mode="shuffled"     resample g and delta from the same marginals
                                but independently (destroys the pairing —
                                the control that isolates correlation)
        pdk_mode="gain_only"    delta_i = 0 (the device_model channel, but
                                with the empirical gain marginal)
        pdk_mode="offset_only"  g_i = 1
        pdk_mode="calibrated"   delta_i replaced by its residual after a
                                per-row trim on the DAC code grid, i.e. what
                                the Section 3.5.3 predistortion path would
                                leave behind (see `trim_lsb`, `trim_range`)

    Parameters not listed here carry the meaning documented on
    CircuitChainSpin and device_model.DeviceParams.
    """

    kind = "full_stack"

    def __init__(self,
                 # -- device (device_model.BehavioralSMTJSpin semantics) --
                 g_dev=1.0, h_off=0.0, sigma_c2c=0.0, p_max=1.0,
                 cv_gain=0.0, sigma_off=0.0, n_spins=None, d2d_seed=0,
                 # -- joint PDK mismatch (RX-07c) --
                 pdk_table=None, pdk_mode="joint", pdk_seed=0,
                 trim_lsb=None, trim_range=None,
                 # -- circuit chain (CircuitChainSpin semantics) --
                 u_grid=None, nbits=6, u_span=4.0, mode="none",
                 n_reset=0, r_reset=R_RESET, rho=None, u_offset=0.0,
                 write_ceiling=1.0, p_read_flip=0.0):
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

        self.g_dev = float(g_dev)
        self.h_off = float(h_off)
        self.sigma_c2c = float(sigma_c2c)
        self.p_max = float(p_max)
        self.n_spins = n_spins

        # D2D overlay, drawn exactly as device_model does (same seed, same
        # draw order) so a device-only arm is bit-identical to the
        # Section 3.4.2 rows.
        self._g_per = None
        self._h_off_per = None
        if cv_gain > 0.0 or sigma_off > 0.0:
            if n_spins is None:
                raise ValueError("cv_gain/sigma_off require n_spins")
            rng = np.random.default_rng(int(d2d_seed))
            self._g_per = rng.normal(loc=1.0, scale=cv_gain, size=n_spins)
            self._h_off_per = rng.normal(loc=0.0, scale=sigma_off,
                                         size=n_spins)

        # Joint PDK mismatch overlay.
        self.pdk_mode = pdk_mode
        self._g_pdk = None
        self._delta_pdk = None
        if pdk_table is not None:
            if n_spins is None:
                raise ValueError("pdk_table requires n_spins")
            tab = np.asarray(pdk_table, dtype=np.float64)
            if tab.ndim != 2 or tab.shape[1] != 2:
                raise ValueError("pdk_table must be [[g, delta], ...]")
            rng = np.random.default_rng(int(pdk_seed))
            i_g = rng.integers(0, tab.shape[0], size=n_spins)
            g = tab[i_g, 0]
            if pdk_mode == "shuffled":
                i_d = rng.integers(0, tab.shape[0], size=n_spins)
                delta = tab[i_d, 1]
            else:
                delta = tab[i_g, 1]
            if pdk_mode == "gain_only":
                delta = np.zeros(n_spins)
            elif pdk_mode == "offset_only":
                g = np.ones(n_spins)
            elif pdk_mode == "calibrated":
                if trim_lsb is None:
                    raise ValueError("pdk_mode='calibrated' needs trim_lsb")
                # per-row trim code on the write DAC grid: the nearest
                # available code cancels delta, leaving <= half an LSB —
                # unless delta exceeds the trim range, in which case the
                # code rails out and the excess survives uncompensated.
                lim = float("inf") if trim_range is None else float(trim_range)
                code = np.clip(np.round(delta / float(trim_lsb)),
                               -lim / float(trim_lsb), lim / float(trim_lsb))
                delta = delta - code * float(trim_lsb)
            elif pdk_mode not in ("joint", "shuffled"):
                raise ValueError(f"bad pdk_mode: {pdk_mode}")
            self._g_pdk = g
            self._delta_pdk = delta

        self.n_reset = int(n_reset)
        self.r_reset = float(r_reset)
        self.rho = (1.0 - self.r_reset) ** self.n_reset if self.n_reset else 0.0
        if rho is not None:
            self.rho = float(rho)
        self.u_offset = np.asarray(u_offset, dtype=np.float64)
        self.write_ceiling = float(write_ceiling)
        if not 0.0 < self.write_ceiling <= 1.0:
            raise ValueError(f"bad write_ceiling: {write_ceiling}")
        self.p_read_flip = float(p_read_flip)
        if not 0.0 <= self.p_read_flip <= 1.0:
            raise ValueError(f"bad p_read_flip: {p_read_flip}")

    # -- drive path ---------------------------------------------------------
    def _p_plus(self, h_eff_arr, beta, rng, idx=None):
        h = np.asarray(h_eff_arr, dtype=np.float64)
        if self.mode == "beta_scaled":
            u = 2.0 * beta * _quantize(h, self.grid)
        elif self.mode == "fixed_u":
            u = _quantize(2.0 * beta * h, self.grid)
        else:
            u = 2.0 * beta * h

        g = self.g_dev
        h_off_eff = self.h_off
        delta = 0.0
        if self._g_per is not None:
            g = g * (self._g_per if idx is None else self._g_per[idx])
            h_off_eff = h_off_eff + (self._h_off_per if idx is None
                                     else self._h_off_per[idx])
        if self._g_pdk is not None:
            g = g * (self._g_pdk if idx is None else self._g_pdk[idx])
            delta = (self._delta_pdk if idx is None else self._delta_pdk[idx])
        off = self.u_offset
        if off.ndim > 0 and idx is not None:
            off = off[idx]

        u = g * (u - delta) + 2.0 * beta * h_off_eff + off
        if self.sigma_c2c > 0.0:
            u = u + rng.normal(0.0, self.sigma_c2c, size=np.shape(u))
        p = _sigmoid(u)
        if self.p_max < 1.0:
            p = np.clip(p, 1.0 - self.p_max, self.p_max)
        if self.write_ceiling < 1.0:
            p = np.minimum(p, self.write_ceiling)
        return p

    def _readback(self, s, rng, shape=None):
        if self.p_read_flip <= 0.0:
            return s
        if shape is None:
            return -s if rng.random() < self.p_read_flip else s
        flip = rng.random(size=shape) < self.p_read_flip
        return np.where(flip, -s, s).astype(np.int8)

    # -- SpinBackend API ----------------------------------------------------
    def sample(self, h_eff, s_cur, beta, rng, idx=None):
        p = float(np.atleast_1d(
            self._p_plus(np.array([h_eff]), beta, rng,
                         idx=None if idx is None else np.array([idx])))[0])
        if self.rho > 0.0 and s_cur == 1 and rng.random() < self.rho:
            return self._readback(1, rng)
        return self._readback(1 if rng.random() < p else -1, rng)

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        p = self._p_plus(h_eff_arr, beta, rng, idx=idx)
        fresh = np.where(rng.random(size=p.shape) < p, 1, -1).astype(np.int8)
        if self.rho <= 0.0:
            return self._readback(fresh, rng, shape=p.shape)
        stuck = (np.asarray(s_cur_arr) == 1) & (rng.random(size=p.shape)
                                                < self.rho)
        return self._readback(np.where(stuck, np.int8(1), fresh), rng,
                              shape=p.shape)


register_spin_backend("full_stack", lambda **kw: FullStackSpin(**kw))
