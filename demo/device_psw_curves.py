"""
device_psw_curves.py — Pedagogical sweep of the five non-ideality
parameters of the BehavioralSMTJSpin model. For each parameter the
script draws the resulting p(s = +1 | h_eff) curve as the parameter
varies continuously through its range, with the ideal-device curve
sigma(2 beta h_eff) overlaid as the reference (dashed orange).

Intended as a companion to the chapter section on hardware-aware
evaluation: a reader unfamiliar with the parameter set can see at
a glance how each non-ideality reshapes the sigmoidal switching
characteristic, and so build intuition for which parameter can be
expected to hurt which kind of solver behaviour.

Self-contained: depends only on NumPy and Matplotlib, no isim.
A separate script device_psw_solver_link.py performs the
"shape-change -> solver-performance" link by calling the actual
solver.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# Tsinghua-purple palette, kept in sync with plot_style.py
PURPLE = {
    "darkest":  "#4B1369",
    "dark":     "#6E2C91",
    "medium":   "#8E54A8",
    "primary":  "#A97DBE",
    "light":    "#C4A7D4",
    "paler":    "#DCC6E6",
    "palest":   "#EFE6F4",
    "accent":   "#D97706",
    "gray":     "#6B6B6B",
    "gray_lt":  "#B8B8B8",
}


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans",
                            "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# A common drive axis used for all panels. h_eff is in units of |J|max
# so that "drive offset = 0.1" is interpretable as 10% of the maximum
# coupling. beta is fixed at the calibration value used in the
# chapter's figures.
H_GRID = np.linspace(-3.0, 3.0, 401)
BETA = 1.0


def p_sw_curve(h, *, g=1.0, h_off=0.0, sigma=0.0, p_max=1.0,
               n_avg=2001, rng=None):
    """Mean p(s=+1 | h_eff) for the specified knob values, averaged
    over n_avg cycle-noise samples when sigma > 0. Other knobs
    apply analytically (no sampling needed)."""
    if sigma == 0.0:
        u = 2.0 * BETA * (g * h + h_off)
        p = sigmoid(u)
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        # Marginalise over zero-mean Gaussian drive noise. The closed
        # form is sigma((2 beta (g h + h_off)) / sqrt(1 + pi sigma^2 / 8)),
        # but Monte Carlo is more transparent in a teaching script.
        eps = rng.normal(0.0, sigma, size=(n_avg, h.size))
        u = 2.0 * BETA * (g * h[None, :] + h_off) + eps
        p = sigmoid(u).mean(axis=0)
    if p_max < 1.0:
        p = np.clip(p, 1.0 - p_max, p_max)
    return p


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def gradient_colors(n, start_hex, end_hex):
    """Linear interpolation between two hex colours in RGB space."""
    s = np.array([int(start_hex[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
    e = np.array([int(end_hex[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
    cs = []
    for k in range(n):
        t = k / max(n - 1, 1)
        rgb = (1 - t) * s + t * e
        cs.append(tuple(rgb))
    return cs


def plot_panel(ax, title, h, curves, values, formatter, ideal_idx,
               cbar_label):
    """Common rendering for one parameter panel.

    curves   : list of p_sw arrays, in the same order as `values`
    values   : list of parameter values
    formatter: callable producing a string from one value (legend key)
    ideal_idx: index of the ideal-parameter curve in `values` (drawn
               as a thicker reference line). The non-ideal curves are
               coloured along a gradient from medium purple to accent
               orange to suggest "moving away from ideal".
    """
    n = len(curves)
    # Build a gradient over only the non-ideal curves. Ordering by
    # the parameter's distance from the ideal value gives a clean
    # visual cue: the further from the ideal, the closer to orange.
    non_ideal = [(k, abs(k - ideal_idx)) for k in range(n) if k != ideal_idx]
    non_ideal.sort(key=lambda kv: kv[1])
    grad = gradient_colors(len(non_ideal), PURPLE["medium"],
                           PURPLE["accent"])
    color_of_k = {kv[0]: grad[i] for i, kv in enumerate(non_ideal)}

    # Reference: ideal curve, dashed orange
    ax.plot(h, sigmoid(2.0 * BETA * h), "--",
            color=PURPLE["accent"], lw=1.4, alpha=0.6,
            label="ideal $\\sigma(2\\beta h)$",
            zorder=2)

    # Non-ideal curves, in original order to preserve legend ordering
    for k in range(n):
        if k == ideal_idx:
            continue
        ax.plot(h, curves[k], "-", lw=1.8,
                color=color_of_k[k],
                label=formatter(values[k]),
                zorder=3)
    # Ideal curve drawn last in deep purple, thicker
    ax.plot(h, curves[ideal_idx], "-", lw=2.6,
            color=PURPLE["darkest"],
            label=formatter(values[ideal_idx]) + "  (ideal)",
            zorder=4)

    ax.set_xlabel(r"effective field $h_i^{\rm eff}$  (units of $|J|_{\max}$)")
    ax.set_ylabel(r"$p(s_i = +1)$")
    ax.set_title(title)
    ax.set_xlim(h.min(), h.max())
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_style()
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    axes_flat = axes.ravel()

    h = H_GRID

    # Panel A: drive gain g_dev
    g_values = [0.4, 0.7, 1.0, 1.5, 2.5]
    g_curves = [p_sw_curve(h, g=g) for g in g_values]
    plot_panel(axes_flat[0],
               r"(a) drive gain  $g_{\rm dev}$",
               h, g_curves, g_values,
               lambda v: f"$g_{{\\rm dev}}={v:g}$",
               ideal_idx=2,
               cbar_label="g_dev")

    # Panel B: drive offset h_off
    off_values = [-0.4, -0.2, 0.0, 0.2, 0.4]
    off_curves = [p_sw_curve(h, h_off=v) for v in off_values]
    plot_panel(axes_flat[1],
               r"(b) drive offset  $h_{\rm off}$",
               h, off_curves, off_values,
               lambda v: f"$h_{{\\rm off}}={v:+g}$",
               ideal_idx=2,
               cbar_label="h_off")

    # Panel C: cycle-to-cycle noise sigma_C2C
    sig_values = [0.0, 0.5, 1.0, 2.0, 4.0]
    sig_curves = [p_sw_curve(h, sigma=v) for v in sig_values]
    plot_panel(axes_flat[2],
               r"(c) C2C noise  $\sigma_{\rm C2C}$",
               h, sig_curves, sig_values,
               lambda v: f"$\\sigma_{{\\rm C2C}}={v:g}$",
               ideal_idx=0,
               cbar_label="sigma_C2C")

    # Panel D: plateau ceiling p_max
    pm_values = [1.0, 0.9, 0.8, 0.72, 0.6]
    pm_curves = [p_sw_curve(h, p_max=v) for v in pm_values]
    plot_panel(axes_flat[3],
               r"(d) plateau ceiling  $p_{\rm max}$",
               h, pm_curves, pm_values,
               lambda v: f"$p_{{\\rm max}}={v:g}$",
               ideal_idx=0,
               cbar_label="p_max")

    # Panel E: D2D dispersion CV(Delta) — illustrated as the
    # device-population mean over a population of devices with
    # gain ~ N(1, CV^2). The mean curve flattens monotonically with CV.
    cv_values = [0.0, 0.077, 0.20, 0.40, 0.80]
    rng = np.random.default_rng(2024)
    cv_curves = []
    n_dev = 4000
    for cv in cv_values:
        if cv == 0.0:
            cv_curves.append(p_sw_curve(h))
        else:
            gs = rng.normal(1.0, cv, size=n_dev)
            # Average over device population
            u = 2.0 * BETA * gs[:, None] * h[None, :]
            p = sigmoid(u).mean(axis=0)
            cv_curves.append(p)
    plot_panel(axes_flat[4],
               r"(e) D2D dispersion  CV($\Delta$)",
               h, cv_curves, cv_values,
               lambda v: f"CV={v:g}",
               ideal_idx=0,
               cbar_label="cv_gain")

    # Panel F: combined "calibrated nominal" device. The nominal
    # device A AP->P parameter set combines D2D PDK dispersion
    # (CV=0.077) with the back-hopping plateau (p_max=0.72). We
    # show the additive contribution of each non-ideality and the
    # combined curve to make the synthesis visually obvious.
    rng2 = np.random.default_rng(7)
    n_dev_f = 4000
    cv_pdk = 0.077

    # PDK only (CV=0.077, no plateau)
    gs = rng2.normal(1.0, cv_pdk, size=n_dev_f)
    u_pdk = 2.0 * BETA * gs[:, None] * h[None, :]
    p_pdk = sigmoid(u_pdk).mean(axis=0)

    # Plateau only (p_max=0.72, no D2D)
    p_plateau = p_sw_curve(h, p_max=0.72)

    # Combined: device-population mean of plateau-clipped sigmoid
    p_combined = np.clip(sigmoid(u_pdk), 1.0 - 0.72, 0.72).mean(axis=0)

    axes_flat[5].plot(h, sigmoid(2.0 * BETA * h), "--",
                      color=PURPLE["accent"], lw=1.4, alpha=0.6,
                      label="ideal $\\sigma(2\\beta h)$",
                      zorder=2)
    axes_flat[5].plot(h, p_pdk, "-", lw=1.8,
                      color=PURPLE["medium"],
                      label="PDK only  (CV=$0.077$)",
                      zorder=3)
    axes_flat[5].plot(h, p_plateau, "-", lw=1.8,
                      color=PURPLE["primary"],
                      label="plateau only  ($p_{\\rm max}=0.72$)",
                      zorder=3)
    axes_flat[5].plot(h, p_combined, "-", lw=2.6,
                      color=PURPLE["darkest"],
                      label="combined  (PDK + plateau)",
                      zorder=4)
    axes_flat[5].set_xlabel(r"effective field $h_i^{\rm eff}$  "
                            "(units of $|J|_{\\max}$)")
    axes_flat[5].set_ylabel(r"$p(s_i = +1)$")
    axes_flat[5].set_title("(f) calibrated nominal device A AP$\\to$P")
    axes_flat[5].set_xlim(h.min(), h.max())
    axes_flat[5].set_ylim(-0.02, 1.05)
    axes_flat[5].legend(loc="lower right", frameon=False)

    fig.suptitle("Effect of each non-ideality parameter on $p_{\\rm sw}$",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "device_psw_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
