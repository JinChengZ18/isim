"""
toy_demo_maxcut5.py

A self-contained 5-node Max-Cut demonstration of the Ising annealing
flow. Uses no dependencies beyond NumPy and Matplotlib so that a
reader can run it as a standalone reference of the minimal viable
Ising solver.

Outputs (next to this script):
    toy_energy_landscape.png   full enumeration of 2^5 = 32 spin states
    toy_annealing_traces.png   energy trajectories of independent runs
    toy_optimal_partition.png  the optimal cut visualised on the graph
    toy_run.npz                raw arrays for downstream re-plotting
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# ---------------------------------------------------------------------------
# Tsinghua-purple palette (matching plot_style.py used by the rest of the
# benchmark suite).
# ---------------------------------------------------------------------------
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


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans",
                            "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# The toy problem: 5-node graph with 6 unit-weight edges.
# Triangle 2-3-4 forces at least one edge per ground state to remain
# uncut, so the maximum cut is 5 out of 6 (proven by enumeration below).
# ---------------------------------------------------------------------------
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
N = 5
W = np.zeros((N, N), dtype=float)
for i, j in EDGES:
    W[i, j] = 1.0
    W[j, i] = 1.0

# Per Section 3.1.3, J_ij = -w_ij / 2, h_i = 0. Then
# H(s) = -sum_{i<j} J_ij s_i s_j = (1/2) sum_{(i,j) in E} s_i s_j.
J_MATRIX = -0.5 * W


def energy(s):
    return 0.5 * sum(s[i] * s[j] for i, j in EDGES)


def cut_value(s):
    return sum(1 for i, j in EDGES if s[i] != s[j])


def enumerate_states():
    states = np.zeros((2 ** N, N), dtype=np.int8)
    energies = np.zeros(2 ** N, dtype=float)
    cuts = np.zeros(2 ** N, dtype=int)
    for k in range(2 ** N):
        s = np.array([+1 if (k >> b) & 1 else -1 for b in range(N)],
                     dtype=np.int8)
        states[k] = s
        energies[k] = energy(s)
        cuts[k] = cut_value(s)
    return states, energies, cuts


# ---------------------------------------------------------------------------
# Minimal asynchronous Gibbs annealer. One sweep visits every spin once
# in a random order; each visit samples the new spin value from the
# conditional distribution sigma(2 beta h_i^eff). The temperature
# schedule is geometric.
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(x))


def beta_schedule(t, T, beta0, betaf):
    if T <= 0:
        return betaf
    return beta0 * (betaf / beta0) ** (t / T)


def anneal_once(seed, n_sweeps=200, beta0=0.1, betaf=5.0):
    """Single annealing run; returns sweep indices, energies, betas
    and the final state."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=N).astype(np.int8)
    sweeps = [0]
    energies = [energy(s)]
    betas = [beta0]
    for t in range(1, n_sweeps + 1):
        beta = beta_schedule(t, n_sweeps, beta0, betaf)
        for i in rng.permutation(N):
            h_eff = float(J_MATRIX[i] @ s)         # h_i^eff = sum_j J_ij s_j
            p_plus = sigmoid(2.0 * beta * h_eff)   # sigma(2 beta h_eff)
            s[i] = +1 if rng.random() < p_plus else -1
        sweeps.append(t)
        energies.append(energy(s))
        betas.append(beta)
    return (np.asarray(sweeps), np.asarray(energies),
            np.asarray(betas), s.copy())


# ---------------------------------------------------------------------------
# Plot 1: the full 2^N = 32 state energy landscape, sorted by energy
# and grouped by energy level.
# ---------------------------------------------------------------------------
def plot_energy_landscape(states, energies, cuts, out_path):
    set_style()
    order = np.argsort(energies, kind="stable")
    es = energies[order]
    cs = cuts[order]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(es))

    levels = np.unique(es)
    color_map = {
        levels[0]: PURPLE["dark"],     # ground
        levels[-1]: PURPLE["accent"],  # worst
    }
    for lev in levels[1:-1]:
        color_map[lev] = PURPLE["primary"]

    bar_colors = [color_map[e] for e in es]
    ax.bar(xs, es, color=bar_colors,
           edgecolor=PURPLE["dark"], linewidth=0.6, width=0.85)

    # Mark ground-state band
    ax.axhline(es.min(), color=PURPLE["darkest"], lw=1.2, ls="--",
               alpha=0.7)
    ax.text(len(es) - 0.5, es.min() - 0.18,
            f"ground state, $E=-2$, cut$=5$",
            ha="right", va="top",
            color=PURPLE["darkest"], fontsize=12)
    ax.text(0.5, es.max() + 0.18,
            f"worst, $E=+3$, cut$=0$",
            ha="left", va="bottom",
            color=PURPLE["accent"], fontsize=12)

    ax.set_xlabel("State index (sorted by energy)")
    ax.set_ylabel("Energy $H(\\mathbf{s})$")
    ax.set_title("Full enumeration of $2^5=32$ spin configurations")
    ax.set_xlim(-0.7, len(es) - 0.3)
    ax.set_ylim(es.min() - 0.6, es.max() + 0.7)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: energy trajectories overlaid with beta(t) on a secondary axis.
# ---------------------------------------------------------------------------
def plot_traces(records, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    # Light traces for individual trials
    for sweeps, en, betas, _ in records:
        ax.plot(sweeps, en, color=PURPLE["light"],
                lw=1.0, alpha=0.7)

    # Median trace highlighted
    all_e = np.stack([r[1] for r in records])
    sweeps = records[0][0]
    median = np.median(all_e, axis=0)
    ax.plot(sweeps, median, color=PURPLE["dark"], lw=2.6,
            label="median over trials")

    # Ground-state line
    ax.axhline(-2.0, color=PURPLE["darkest"], lw=1.2, ls="--",
               label="ground-state energy $E_{\\min}=-2$")

    ax.set_xlabel("Sweep index $t$")
    ax.set_ylabel("Energy $H(\\mathbf{s}^{(t)})$")
    ax.set_xlim(0, sweeps[-1])
    ax.set_ylim(-2.7, all_e.max() + 0.5)
    ax.set_title("Annealing trajectories on the toy Max-Cut")

    # Inset secondary axis: beta(t)
    ax2 = ax.twinx()
    ax2.plot(records[0][0], records[0][2],
             color=PURPLE["accent"], lw=1.6, ls=":",
             label="$\\beta(t)$ schedule")
    ax2.set_ylabel("Inverse temperature $\\beta$",
                   color=PURPLE["accent"])
    ax2.tick_params(axis="y", labelcolor=PURPLE["accent"])
    ax2.set_yscale("log")
    ax2.grid(False)

    # Combined legend
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    handles_left.append(Line2D([], [], color=PURPLE["light"], lw=1.5,
                               label="individual trials"))
    labels_left.append("individual trials")
    ax.legend(handles_left + handles_right,
              labels_left + labels_right,
              loc="upper right", framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: the graph itself with the optimal partition highlighted.
# ---------------------------------------------------------------------------
def _layout():
    """Hand-picked coordinates that keep edges from crossing and
    place the triangle 2-3-4 visually together."""
    return {
        0: (-1.6,  0.9),
        1: (-1.6, -0.9),
        2: ( 0.0,  0.0),
        3: ( 1.4,  0.9),
        4: ( 1.4, -0.9),
    }


def plot_partition(out_path):
    set_style()
    pos = _layout()
    # Optimal ground state: (-,+,+,-,-) gives cut = 5; equivalently
    # the partition {1, 2} vs {0, 3, 4}.
    # Choose the ground-state representative from enumeration that
    # matches a clean visual split.
    spin = {0: -1, 1: +1, 2: +1, 3: -1, 4: -1}

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for i, j in EDGES:
        cut_here = (spin[i] != spin[j])
        col = PURPLE["accent"] if cut_here else PURPLE["gray_lt"]
        lw = 3.0 if cut_here else 1.5
        ls = "-" if cut_here else "-"
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color=col, lw=lw, ls=ls,
                solid_capstyle="round", zorder=1)

    for v, (x, y) in pos.items():
        col = PURPLE["dark"] if spin[v] == +1 else PURPLE["light"]
        edge = PURPLE["darkest"]
        ax.scatter([x], [y], s=1500, color=col,
                   edgecolor=edge, linewidth=1.6, zorder=3)
        txt_color = "white" if spin[v] == +1 else PURPLE["darkest"]
        ax.text(x, y, str(v), ha="center", va="center",
                fontsize=18, fontweight="bold",
                color=txt_color, zorder=4)

    # Legend constructed by hand
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PURPLE["dark"],
               markeredgecolor=PURPLE["darkest"],
               markersize=14, label="$s_i=+1$ (set $S$)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PURPLE["light"],
               markeredgecolor=PURPLE["darkest"],
               markersize=14, label="$s_i=-1$ (set $\\bar S$)"),
        Line2D([0], [0], color=PURPLE["accent"], lw=3.0,
               label="cut edge (5 of 6)"),
        Line2D([0], [0], color=PURPLE["gray_lt"], lw=1.5,
               label="uncut edge"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=2, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.18))

    ax.set_xlim(-2.4, 2.2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_title("Optimal partition: $S=\\{1,2\\},\\;\\bar S=\\{0,3,4\\}$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    states, energies, cuts = enumerate_states()
    order = np.argsort(energies)
    print("Enumerated 2^5 = 32 spin states.")
    print(f"  ground-state energy = {energies[order[0]]:+.1f}, "
          f"max cut = {cuts[order[0]]}")
    print(f"  ground-state degeneracy = "
          f"{int((energies == energies.min()).sum())}")

    # Annealing
    n_trials = 8
    n_sweeps = 200
    beta0, betaf = 0.1, 5.0
    records = [anneal_once(seed=2024 + k,
                           n_sweeps=n_sweeps,
                           beta0=beta0, betaf=betaf)
               for k in range(n_trials)]
    final_e = np.array([r[1][-1] for r in records])
    n_hits = int((final_e == energies.min()).sum())
    print(f"  trials = {n_trials}, sweeps = {n_sweeps}, "
          f"beta in [{beta0}, {betaf}]")
    print(f"  ground-state hits: {n_hits}/{n_trials}")
    print(f"  final energies: {final_e.tolist()}")

    plot_energy_landscape(states, energies, cuts,
                          out_dir / "toy_energy_landscape.png")
    plot_traces(records, out_dir / "toy_annealing_traces.png")
    plot_partition(out_dir / "toy_optimal_partition.png")

    # Persist raw arrays
    np.savez(out_dir / "toy_run.npz",
             states=states, energies=energies, cuts=cuts,
             trial_sweeps=np.stack([r[0] for r in records]),
             trial_energies=np.stack([r[1] for r in records]),
             trial_betas=np.stack([r[2] for r in records]),
             trial_finals=np.stack([r[3] for r in records]),
             beta0=beta0, betaf=betaf, n_sweeps=n_sweeps,
             edges=np.array(EDGES))

    for fn in ("toy_energy_landscape.png", "toy_annealing_traces.png",
               "toy_optimal_partition.png", "toy_run.npz"):
        print(f"  wrote {out_dir / fn}")


if __name__ == "__main__":
    main()
