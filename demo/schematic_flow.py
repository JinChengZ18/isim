"""
schematic_flow.py — fancy 5-stage schematic of the Ising annealing
flow on the toy 5-node Max-Cut.

Stages, left to right:
    1. Problem (graph + objective)
    2. Encoding (J matrix, h, Hamiltonian)
    3. p-bit network (5 stochastic MTJ units + sigmoid sampling rule)
    4. Annealing schedule (geometric beta(t))
    5. Solution (graph with the optimal cut highlighted)

Layout strategy
---------------
Panel height is set so the principal graphical element of each stage
(node graph, J matrix, p-bit row, annealing curve) fills the upper
two-thirds of the panel; auxiliary text sits tight against the bottom
edge as a compact footer. This eliminates the large vertical
whitespace of an earlier draft.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle)
from pathlib import Path


PURPLE = {
    "darkest":  "#4B1369",
    "dark":     "#6E2C91",
    "medium":   "#8E54A8",
    "primary":  "#A97DBE",
    "light":    "#C4A7D4",
    "paler":    "#DCC6E6",
    "palest":   "#EFE6F4",
    "accent":   "#D97706",
    "accent_lt":"#F6B26B",
    "gray":     "#6B6B6B",
    "gray_lt":  "#B8B8B8",
}


# ---- Layout constants ----
PANEL_W   = 16.0
PANEL_H   = 14.0    # was 22.0 — compress vertical to remove dead space
GAP       = 1.2
TITLE_H   = 1.5
PAD_X     = 0.5
PAD_TOP   = 0.4
PAD_BOT   = 0.4


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans",
                            "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


NODES = {
    0: (-1.5,  0.9),
    1: (-1.5, -0.9),
    2: ( 0.0,  0.0),
    3: ( 1.4,  0.9),
    4: ( 1.4, -0.9),
}
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def panel(ax, x0, y0, w, h, title):
    body = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor=PURPLE["palest"],
        edgecolor=PURPLE["dark"],
        linewidth=1.5, zorder=1)
    ax.add_patch(body)
    strip = FancyBboxPatch(
        (x0, y0 + h - TITLE_H), w, TITLE_H,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor=PURPLE["dark"],
        edgecolor=PURPLE["dark"],
        linewidth=0.0, zorder=2)
    ax.add_patch(strip)
    ax.add_patch(Rectangle(
        (x0, y0 + h - TITLE_H), w, TITLE_H * 0.5,
        facecolor=PURPLE["dark"], edgecolor="none", zorder=2))
    ax.text(x0 + w / 2, y0 + h - TITLE_H / 2,
            title, ha="center", va="center",
            color="white", fontsize=16, fontweight="bold", zorder=3)
    return (x0 + PAD_X, y0 + PAD_BOT,
            x0 + w - PAD_X, y0 + h - TITLE_H - PAD_TOP)


def draw_arrow(ax, xa, ya, xb, yb):
    arr = FancyArrowPatch(
        (xa, ya), (xb, yb),
        arrowstyle="-|>", mutation_scale=24,
        color=PURPLE["dark"], linewidth=2.5, zorder=4)
    ax.add_patch(arr)


def draw_graph(ax, cx, cy, scale=1.0, partition=None,
               highlight_cut=False, label_size=12,
               node_radius=0.42):
    pos = {v: (cx + scale * x, cy + scale * y) for v, (x, y) in NODES.items()}
    cut_set = set()
    if partition is not None and highlight_cut:
        for i, j in EDGES:
            if partition[i] != partition[j]:
                cut_set.add(frozenset({i, j}))
    for i, j in EDGES:
        is_cut = frozenset({i, j}) in cut_set
        col = PURPLE["accent"] if is_cut else PURPLE["medium"]
        lw = 3.0 if is_cut else 1.7
        if partition is None:
            col = PURPLE["medium"]; lw = 1.8
        ax.plot([pos[i][0], pos[j][0]],
                [pos[i][1], pos[j][1]],
                color=col, lw=lw, zorder=5,
                solid_capstyle="round")
    for v, (x, y) in pos.items():
        if partition is None:
            face = PURPLE["paler"]; txt_col = PURPLE["darkest"]
        else:
            face = PURPLE["dark"] if partition[v] == +1 else PURPLE["light"]
            txt_col = "white" if partition[v] == +1 else PURPLE["darkest"]
        ax.add_patch(Circle((x, y), node_radius * scale,
                            facecolor=face,
                            edgecolor=PURPLE["darkest"],
                            linewidth=1.3, zorder=6))
        ax.text(x, y, str(v), ha="center", va="center",
                fontsize=label_size, fontweight="bold",
                color=txt_col, zorder=7)


def draw_pbit_unit(ax, cx, cy, w=0.9, h=1.6, top_dark=True):
    top_col = PURPLE["dark"] if top_dark else PURPLE["paler"]
    bot_col = PURPLE["paler"] if top_dark else PURPLE["dark"]
    ax.add_patch(Rectangle((cx - w / 2, cy + 0.05),
                           w, h / 2 - 0.05,
                           facecolor=top_col,
                           edgecolor=PURPLE["darkest"],
                           linewidth=1.3, zorder=6))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2),
                           w, h / 2 - 0.05,
                           facecolor=bot_col,
                           edgecolor=PURPLE["darkest"],
                           linewidth=1.3, zorder=6))
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy],
            color=PURPLE["darkest"], lw=1.6, zorder=7)
    arr_col_top = "white" if top_dark else PURPLE["darkest"]
    if top_dark:
        ax.annotate("", xy=(cx, cy + h / 2 - 0.15),
                    xytext=(cx, cy + 0.20),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=arr_col_top, lw=1.6),
                    zorder=8)
    else:
        ax.annotate("", xy=(cx, cy + 0.20),
                    xytext=(cx, cy + h / 2 - 0.15),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=arr_col_top, lw=1.6),
                    zorder=8)


def plot_schematic(out_path):
    set_style()

    n_panel = 5
    fig_w_units = n_panel * PANEL_W + (n_panel - 1) * GAP + 2.0
    fig_h_units = PANEL_H + 2.0

    aspect = fig_w_units / fig_h_units
    fig_h_inches = 3.4
    fig_w_inches = fig_h_inches * aspect

    fig, ax = plt.subplots(figsize=(fig_w_inches, fig_h_inches))
    ax.set_xlim(0, fig_w_units)
    ax.set_ylim(0, fig_h_units)
    ax.set_aspect("equal")
    ax.axis("off")

    y0 = 1.0
    x_origins = [1.0]
    for _ in range(n_panel - 1):
        x_origins.append(x_origins[-1] + PANEL_W + GAP)

    # ===================== Stage 1: Problem =====================
    xlo, ylo, xhi, yhi = panel(ax, x_origins[0], y0,
                               PANEL_W, PANEL_H, "Problem")
    cx = (xlo + xhi) / 2
    draw_graph(ax, cx, ylo + 7.5, scale=2.7,
               partition=None, highlight_cut=False,
               label_size=12, node_radius=0.40)
    ax.text(cx, ylo + 2.0,
            "5 nodes,  6 edges",
            ha="center", va="center", fontsize=13,
            color=PURPLE["darkest"])
    ax.text(cx, ylo + 0.7,
            "maximise the cut",
            ha="center", va="center", fontsize=12.5,
            color=PURPLE["darkest"])

    # ===================== Stage 2: Ising encoding =====================
    xlo, ylo, xhi, yhi = panel(ax, x_origins[1], y0,
                               PANEL_W, PANEL_H, "Ising encoding")
    cx = (xlo + xhi) / 2

    ax.text(cx, yhi - 1.2,
            r"$H(\mathbf{s})=-\!\!\sum_{i<j}\! J_{ij}\,s_is_j$",
            ha="center", va="center", fontsize=14,
            color=PURPLE["darkest"])
    ax.text(cx, yhi - 2.9,
            r"$J_{ij}=-w_{ij}/2,\quad h_i=0$",
            ha="center", va="center", fontsize=13,
            color=PURPLE["darkest"])

    n = 5
    cell = 1.10
    grid_w = n * cell
    grid_x0 = cx - grid_w / 2
    grid_y0 = ylo + 2.2
    edge_set = {frozenset({i, j}) for i, j in EDGES}
    for r in range(n):
        for c in range(n):
            if r == c:
                col = PURPLE["palest"]
            elif frozenset({r, c}) in edge_set:
                col = PURPLE["dark"]
            else:
                col = PURPLE["paler"]
            ax.add_patch(Rectangle(
                (grid_x0 + c * cell, grid_y0 + (n - 1 - r) * cell),
                cell, cell, facecolor=col,
                edgecolor=PURPLE["darkest"], linewidth=0.7, zorder=5))
    ax.text(cx, ylo + 1.0,
            r"coupling matrix $J$",
            ha="center", va="center", fontsize=12,
            color=PURPLE["darkest"])

    # ===================== Stage 3: p-bit network =====================
    xlo, ylo, xhi, yhi = panel(ax, x_origins[2], y0,
                               PANEL_W, PANEL_H, "p-bit network")
    cx = (xlo + xhi) / 2

    n_units = 5
    pb_y = ylo + 7.2
    pb_xs = np.linspace(xlo + 1.6, xhi - 1.6, n_units)
    states_show = [True, False, True, False, True]
    for k, x_u in enumerate(pb_xs):
        draw_pbit_unit(ax, x_u, pb_y, w=1.8, h=3.4,
                       top_dark=states_show[k])
        ax.text(x_u, pb_y + 2.05, f"$s_{k}$",
                ha="center", va="bottom", fontsize=13,
                color=PURPLE["darkest"])

    link_y = pb_y - 2.3
    for x_u in pb_xs:
        ax.plot([x_u, x_u], [pb_y - 1.8, link_y],
                color=PURPLE["gray"], lw=1.0, zorder=4)
    ax.plot([pb_xs[0], pb_xs[-1]], [link_y, link_y],
            color=PURPLE["gray"], lw=1.0, zorder=4)
    ax.text(cx, link_y - 1.3,
            r"$h_i^{\rm eff}=\sum_j J_{ij}s_j$",
            ha="center", va="center", fontsize=12,
            color=PURPLE["darkest"])
    ax.text(cx, ylo + 0.7,
            r"$p(s_i\!=\!{+}1\mid\mathbf{s}_{-i})="
            r"\sigma(2\beta\,h_i^{\rm eff})$",
            ha="center", va="center", fontsize=12.5,
            color=PURPLE["darkest"])

    # ===================== Stage 4: Annealing =====================
    xlo, ylo, xhi, yhi = panel(ax, x_origins[3], y0,
                               PANEL_W, PANEL_H, "Annealing")
    cx = (xlo + xhi) / 2

    ax.text(cx, yhi - 1.2,
            r"$\beta(t)=\beta_0(\beta_f/\beta_0)^{t/T}$",
            ha="center", va="center", fontsize=12.5,
            color=PURPLE["darkest"])

    t = np.linspace(0, 1, 100)
    beta0, betaf = 0.1, 5.0
    beta = beta0 * (betaf / beta0) ** t

    px0 = xlo + 1.6
    px1 = xhi - 0.7
    py0 = ylo + 3.0
    py1 = yhi - 1.7
    ax_w = px1 - px0
    ax_h = py1 - py0
    ax.add_patch(Rectangle((px0, py0), ax_w, ax_h,
                           facecolor="white",
                           edgecolor=PURPLE["gray"],
                           linewidth=1.0, zorder=4))
    for frac in [0.25, 0.5, 0.75]:
        ax.plot([px0, px1], [py0 + frac * ax_h] * 2,
                color="#E8E8E8", lw=0.7, zorder=4.5)
    xs_p = px0 + ax_w * t
    ys_p = py0 + ax_h * (beta - beta0) / (betaf - beta0)
    ax.plot(xs_p, ys_p, color=PURPLE["accent"], lw=2.8, zorder=6)
    ax.plot([xs_p[0]], [ys_p[0]], marker="o", markersize=8,
            color=PURPLE["accent"],
            markeredgecolor=PURPLE["darkest"], markeredgewidth=0.9,
            zorder=7)
    ax.plot([xs_p[-1]], [ys_p[-1]], marker="o", markersize=8,
            color=PURPLE["accent"],
            markeredgecolor=PURPLE["darkest"], markeredgewidth=0.9,
            zorder=7)
    ax.text(px0 - 0.3, py0,
            r"$\beta_0$", ha="right", va="center", fontsize=13,
            color=PURPLE["darkest"])
    ax.text(px0 - 0.3, py1,
            r"$\beta_f$", ha="right", va="center", fontsize=13,
            color=PURPLE["darkest"])
    ax.text((px0 + px1) / 2, py0 - 0.7,
            r"sweep $t$",
            ha="center", va="top", fontsize=11.5,
            color=PURPLE["darkest"])
    ax.text(cx, ylo + 0.7,
            "geometric schedule",
            ha="center", va="center", fontsize=12.5,
            color=PURPLE["darkest"])

    # ===================== Stage 5: Solution =====================
    xlo, ylo, xhi, yhi = panel(ax, x_origins[4], y0,
                               PANEL_W, PANEL_H, "Solution")
    cx = (xlo + xhi) / 2
    spin_optimal = {0: -1, 1: +1, 2: +1, 3: -1, 4: -1}
    draw_graph(ax, cx, ylo + 7.5, scale=2.7,
               partition=spin_optimal, highlight_cut=True,
               label_size=12, node_radius=0.40)
    ax.text(cx, ylo + 2.0,
            r"cut$=5$  (optimal)",
            ha="center", va="center", fontsize=13,
            color=PURPLE["darkest"])
    ax.text(cx, ylo + 0.7,
            r"$E_{\min}=-2$",
            ha="center", va="center", fontsize=12.5,
            color=PURPLE["darkest"])

    # ===================== Connecting arrows =====================
    arrow_y = y0 + (PANEL_H - TITLE_H) / 2 + 0.2
    for k in range(n_panel - 1):
        xa = x_origins[k] + PANEL_W
        xb = x_origins[k + 1]
        draw_arrow(ax, xa - 0.05, arrow_y,
                   xb + 0.05, arrow_y)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    plot_schematic(out_dir / "ising_flow_schematic.png")
    print(f"wrote {out_dir / 'ising_flow_schematic.png'}")


if __name__ == "__main__":
    main()
