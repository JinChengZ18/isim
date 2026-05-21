"""
Matplotlib style configuration for academic figures.

Design choices:
- English-only labels, Arial sans-serif (with DejaVu Sans as fallback
  so the style renders on Linux/macOS/Windows even when Arial is not
  installed; a small-caps warning prints once if Arial is absent).
- Tsinghua purple palette, leaning on the LIGHT shades so that long
  series (multi-instance bar charts, overlay curves) remain readable.
- Larger base font sizes (12-14 pt) tuned for direct insertion into
  thesis chapters at reproduction scale.
- Transparent grid, thin frame, inward ticks on all four sides.

Public API:
    set_style()              apply rcParams globally
    new_figure(figsize)      shortcut for (fig, ax) with style applied
    save(fig, path)          shortcut for high-DPI PNG export
    TSINGHUA_PURPLE          dict of named hex colors
    SERIES_COLORS            ordered list for matplotlib prop cycle
    LIGHT_FILL, DARK_LINE    convenience shades for fill+outline pairs
"""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Light-biased Tsinghua purple scheme. The primary fill is deliberately
# light (around 60% lightness) so that overlay labels remain legible.
TSINGHUA_PURPLE = {
    "darkest":  "#4B1369",
    "dark":     "#6E2C91",
    "medium":   "#8E54A8",
    "primary":  "#A97DBE",   # main fill color: light, but still clearly purple
    "light":    "#C4A7D4",
    "paler":    "#DCC6E6",
    "palest":   "#EFE6F4",
    "accent":   "#D97706",   # contrast orange for baseline comparisons
    "accent_lt":"#F6B26B",
    "gray":     "#6B6B6B",
    "gray_lt":  "#B8B8B8",
}

LIGHT_FILL = TSINGHUA_PURPLE["primary"]
DARK_LINE = TSINGHUA_PURPLE["darkest"]

# Ordered color cycle for multi-series plots. Starts dark and gets
# lighter, then introduces the orange accent, then gray.
SERIES_COLORS = [
    TSINGHUA_PURPLE["dark"],
    TSINGHUA_PURPLE["medium"],
    TSINGHUA_PURPLE["primary"],
    TSINGHUA_PURPLE["light"],
    TSINGHUA_PURPLE["accent"],
    TSINGHUA_PURPLE["gray"],
]


_STYLE_APPLIED = False
_ARIAL_WARNED = False


def _resolve_sans_family():
    """Return a font-family list with Arial first if available,
    otherwise DejaVu Sans. Emits a one-time warning if Arial is missing."""
    global _ARIAL_WARNED
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Arial", "Liberation Sans", "Helvetica",
                 "DejaVu Sans", "sans-serif"]
    if "Arial" not in available and not _ARIAL_WARNED:
        warnings.warn(
            "Arial font not found. Falling back to DejaVu Sans. Figures "
            "will render correctly but will not match Arial exactly. To "
            "restore Arial on Linux, install msttcorefonts "
            "(`sudo apt install ttf-mscorefonts-installer`) and clear "
            "matplotlib's font cache.",
            RuntimeWarning, stacklevel=2)
        _ARIAL_WARNED = True
    return preferred


def set_style():
    """Apply the paper-style matplotlib rcParams. Idempotent."""
    global _STYLE_APPLIED
    family = _resolve_sans_family()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "mathtext.fontset": "stixsans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })
    _STYLE_APPLIED = True


def new_figure(figsize=(5.2, 3.6), **kwargs):
    """Return (fig, ax) with the paper style applied. Default figsize
    is larger than before to match the bigger font."""
    if not _STYLE_APPLIED:
        set_style()
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def save(fig, path, **kwargs):
    fig.savefig(path, **kwargs)
