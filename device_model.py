"""
device_model.py — Behavioral sMTJ p-bit model parametrised by the
device-level measurements reported in Section 2.3 of the thesis.

The ideal-Gibbs sMTJ samples
    p(s_i = +1 | h_i^eff) = sigma(2 beta h_i^eff).

A real device departs from this idealisation along four orthogonal
axes; each is captured here by a single dimensionless or unit-bearing
knob, designed so that setting all four to their nominal values
recovers the ideal device:

  1. finite logistic slope (beta_s_meas), measured per same-batch
     Sigmoid fits. The ideal sigma(2 beta h_eff) collapses on the
     unit time-scale; a real device has finite k = 1/beta_s, which
     scales the effective drive by g_dev = beta_s_meas / beta_s_ideal.

  2. write-bias offset V_off, equivalent to a static h_off field that
     biases all spins. Captured by a per-spin offset added to h_eff.

  3. cycle-to-cycle thermal jitter sigma_C2C, modelled as zero-mean
     Gaussian noise on the effective drive. Reduces the effective
     slope and turns p(+1) into a soft-Bernoulli with marginalised
     variance.

  4. back-hopping plateau, observed on Device A AP->P at high V where
     P_sw saturates near 0.7 instead of 1.0. Captured by a saturation
     ceiling p_max < 1 applied above a drive threshold.

Plus an array-level overlay:

  5. device-to-device dispersion: per-spin random gain and offset
     drawn from N(1, CV_d2d) and N(0, sigma_off^2) once per device,
     fixed across cycles. CV_d2d defaults to the 7.7% Brinkman-PDK
     baseline derived in the thesis.

The class is a subclass of isim.SMTJSpin so that the parallel worker
path resolves it via the standard registry. Setting every knob to
its nominal value restores ideal Gibbs behaviour to within Monte
Carlo precision.

For numerical accuracy in stress tests, a CALIBRATION_BETA_S_REF
constant defines the reference logistic slope of the ideal device
in the units used by the rest of the framework (h_eff has the
dimensions of energy and beta is its inverse, so the ideal device
has beta_s_ideal = 2 beta in those units; the device-level slope is
expressed as a ratio g_dev that is unit-free).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isim import SMTJSpin, register_spin_backend


# ---------------------------------------------------------------------------
# Parameter container.
#
# All four knobs default to their ideal-device values. The defaults
# below correspond to Section 2.3 main reference (Device A, P->AP,
# t_w = 0.75 ns) when used as a calibrated nominal device.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceParams:
    """Single-device behavioural parameters.

    g_dev: dimensionless gain on the drive 2 beta h_eff. g_dev = 1
        recovers the ideal sigmoid slope. The empirical slope ratio
        between same-batch Sigmoid measurement and Néel-Brown
        single-device prediction is captured by g_dev = eta_c when
        the device is operated at its calibrated bias point. In the
        chapter's notation g_dev relates to the dimensionless ratio
        beta_s_meas / beta_s_ideal, where beta_s_ideal corresponds
        to the framework's unit-time annealer.
    h_off: additive offset on h_eff, mirroring the write-bias offset
        V_off / V_th of the device. h_off has the same units as h_i
        in the Hamiltonian.
    sigma_c2c: standard deviation of zero-mean Gaussian noise added
        to the drive on every sample. Models cycle-to-cycle thermal
        jitter not absorbed by the slope. Same units as 2 beta h_eff,
        i.e. dimensionless.
    p_max: ceiling of the conditional sampling probability when the
        drive is saturating positive. Models the back-hopping
        plateau on Device A AP->P. Symmetric below: P(-1) saturates
        at p_max for saturating negative drive."""

    g_dev: float = 1.0
    h_off: float = 0.0
    sigma_c2c: float = 0.0
    p_max: float = 1.0

    def is_ideal(self) -> bool:
        return (math.isclose(self.g_dev, 1.0) and self.h_off == 0.0
                and self.sigma_c2c == 0.0 and math.isclose(self.p_max, 1.0))


@dataclass(frozen=True)
class ArrayDispersion:
    """Device-to-device dispersion overlay. cv_gain is the relative
    standard deviation of the per-spin gain g_i ~ N(1, cv_gain^2);
    sigma_off is the standard deviation of the per-spin additive
    offset h_off_i ~ N(0, sigma_off^2). Both are sampled once per
    device instance and held fixed across cycles, matching the
    Section 2.3.6 D2D parametrisation."""

    cv_gain: float = 0.0
    sigma_off: float = 0.0


# ---------------------------------------------------------------------------
# The behavioural backend itself.
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class BehavioralSMTJSpin(SMTJSpin):
    """sMTJ p-bit with Section 2.3 non-idealities.

    Mathematically the sample drawn for spin i is
        u_i = 2 beta (g_dev g_i h_eff + h_off + h_off_i)
              + epsilon_i,                  epsilon_i ~ N(0, sigma^2)
        p(+1) = clip(sigma(u_i), 1 - p_max, p_max)

    where g_i, h_off_i are device-level (D2D) per-spin coefficients
    fixed at construction, and sigma_c2c, h_off are common to all
    spins (single-device cycle noise and bias).

    n_spins is the number of spins in the target problem; required
    only when ArrayDispersion is non-trivial so that the per-spin
    coefficients can be sampled with a deterministic seed.
    """

    kind = "behavioral_smtj"

    def __init__(self,
                 g_dev: float = 1.0,
                 h_off: float = 0.0,
                 sigma_c2c: float = 0.0,
                 p_max: float = 1.0,
                 cv_gain: float = 0.0,
                 sigma_off: float = 0.0,
                 n_spins: Optional[int] = None,
                 d2d_seed: int = 0,
                 I_0: float = 1.0):
        super().__init__(I_0=I_0)
        self.params = DeviceParams(g_dev=g_dev, h_off=h_off,
                                   sigma_c2c=sigma_c2c, p_max=p_max)
        self.dispersion = ArrayDispersion(cv_gain=cv_gain,
                                          sigma_off=sigma_off)
        self.n_spins = n_spins
        self.d2d_seed = int(d2d_seed)
        self._g_per = None
        self._h_off_per = None
        if (cv_gain > 0.0 or sigma_off > 0.0):
            if n_spins is None:
                raise ValueError(
                    "ArrayDispersion (cv_gain or sigma_off) requires "
                    "n_spins to be specified at backend construction.")
            rng = np.random.default_rng(self.d2d_seed)
            self._g_per = rng.normal(loc=1.0, scale=cv_gain, size=n_spins)
            self._h_off_per = rng.normal(loc=0.0, scale=sigma_off,
                                         size=n_spins)

    # ------------------------------------------------------------------
    # The single-spin path is rarely on the hot path (only the
    # async_python mode ever calls it), so it is implemented in plain
    # Python for clarity. The batched path is the production hot path
    # under update-mode=block.
    # ------------------------------------------------------------------
    def _drive_single(self, h_eff, beta, idx=None):
        p = self.params
        g_eff = p.g_dev
        h_off_eff = p.h_off
        if self._g_per is not None and idx is not None:
            g_eff = g_eff * float(self._g_per[idx])
            h_off_eff = h_off_eff + float(self._h_off_per[idx])
        return 2.0 * beta * (g_eff * h_eff + h_off_eff)

    def sample(self, h_eff, s_cur, beta, rng, idx=None):
        u = self._drive_single(h_eff, beta, idx=idx)
        if self.params.sigma_c2c > 0.0:
            u = u + rng.normal(0.0, self.params.sigma_c2c)
        p = float(_sigmoid(u))
        if self.params.p_max < 1.0:
            p_lo = 1.0 - self.params.p_max
            p_hi = self.params.p_max
            p = min(max(p, p_lo), p_hi)
        return 1 if rng.random() < p else -1

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        p_par = self.params
        g_eff = p_par.g_dev
        h_off_eff = p_par.h_off
        if self._g_per is not None:
            # idx is the global spin index of each entry of h_eff_arr.
            # The block solver passes a colour-class slice; the
            # async_python solver passes the full vector or None.
            if idx is not None:
                g_arr = g_eff * self._g_per[idx]
                h_off_arr = h_off_eff + self._h_off_per[idx]
            elif self._g_per.shape[0] == h_eff_arr.shape[0]:
                g_arr = g_eff * self._g_per
                h_off_arr = h_off_eff + self._h_off_per
            else:
                # Index unknown and shapes don't align: fall back to
                # common-knob mode and emit a one-time warning.
                if not getattr(self, "_warned_d2d_skip", False):
                    import warnings
                    warnings.warn(
                        "BehavioralSMTJSpin: D2D dispersion bypassed "
                        "because backend received a partial-array call "
                        "without an idx argument. Pass idx=block_indices "
                        "from the solver to enable the per-spin path.",
                        RuntimeWarning, stacklevel=2)
                    self._warned_d2d_skip = True
                g_arr = np.full_like(h_eff_arr, g_eff)
                h_off_arr = np.full_like(h_eff_arr, h_off_eff)
        else:
            g_arr = g_eff
            h_off_arr = h_off_eff
        u = 2.0 * beta * (g_arr * h_eff_arr + h_off_arr)
        if p_par.sigma_c2c > 0.0:
            u = u + rng.normal(0.0, p_par.sigma_c2c, size=u.shape)
        p = _sigmoid(u)
        if p_par.p_max < 1.0:
            p = np.clip(p, 1.0 - p_par.p_max, p_par.p_max)
        rand_u = rng.random(size=p.shape)
        return np.where(rand_u < p, 1, -1).astype(np.int8)


# Register so spin_spec=("behavioral_smtj", {...}) resolves in the
# parallel worker.
register_spin_backend("behavioral_smtj",
                      lambda **kwargs: BehavioralSMTJSpin(**kwargs))


# ---------------------------------------------------------------------------
# Pre-baked parameter sets corresponding to Section 2.3 calibration
# points. Users can call these as drop-in nominal devices.
# ---------------------------------------------------------------------------
NOMINAL_DEVICE_A_P_AP = dict(
    g_dev=1.0,         # operating-point slope is the simulation reference
    h_off=0.0,         # symmetric P->AP direction has negligible offset
    sigma_c2c=0.0,     # logistic slope absorbs C2C jitter
    p_max=1.0,         # P->AP shows clean transition, no plateau
    cv_gain=0.077,     # PDK-Brinkman baseline
    sigma_off=0.0,
)

NOMINAL_DEVICE_A_AP_P = dict(
    g_dev=1.0,
    h_off=0.0,
    sigma_c2c=0.0,
    p_max=0.72,        # back-hopping plateau on Device A AP->P at high V
    cv_gain=0.077,
    sigma_off=0.0,
)


def make_spec(g_dev=1.0, h_off=0.0, sigma_c2c=0.0, p_max=1.0,
              cv_gain=0.0, sigma_off=0.0, n_spins=None, d2d_seed=0):
    """Return a (kind, kwargs) tuple for use as spin_spec=...
    in isim.multistart()."""
    return ("behavioral_smtj", dict(
        g_dev=g_dev, h_off=h_off, sigma_c2c=sigma_c2c, p_max=p_max,
        cv_gain=cv_gain, sigma_off=sigma_off,
        n_spins=n_spins, d2d_seed=d2d_seed))
