"""
hardware_metrics.py — Map software-level metrics (sweeps, p_success,
TTS_99 in CPU seconds) to hardware-level quantities (annealing time,
energy per solution, throughput in spin-updates per second) using
the device-level numbers measured in Section 2.3.

Three platforms are exposed for cross-architecture comparison:

  * sMTJ-array       — the device of the present thesis. One spin
                       update per write pulse of t_w = 0.75 ns at
                       the calibrated bias point, energy E_write =
                       V_th^2 / R_SOT * t_w = 0.78 pJ per pulse.
                       N spins update in parallel within an array
                       tile.

  * cmos-pbit        — the published CMOS p-bit ASIC of Sutton 2020,
                       roughly 5 ns per update at ~5 pJ per spin
                       (extrapolated from the Borders 2019 Nature
                       integer-factoring system using sMTJ-augmented
                       CMOS read-out chains; numbers are conservative).

  * fpga-sa          — a single Metropolis sweep on a clocked FPGA
                       Ising machine (e.g., the Goto et al. coherent
                       Ising machine emulator class). Per-update wall
                       time ~1 ns and per-update energy ~1 nJ at
                       array level; large area cost amortises poorly.

  * cpu-numba        — the present software backend running on a
                       single Intel x86 core compiled with Numba.
                       Numbers are inferred at runtime from the
                       observed time_median and sweep count of an
                       isim run.

The physical numbers are fixed to first-order — adequate for
comparing the time-to-solution of distinct hardware classes within
an order of magnitude. Section 2.3 supplies the sMTJ-array figures
directly; the CMOS and FPGA numbers are taken from cited literature
and so carry larger absolute uncertainty (factor 2-3) but their
ratio against sMTJ-array is robust because it tracks first-order
device physics (write-energy scaling, clock frequency, noise margin).

References for the comparator platforms appear in the chapter text
that calls this module. This file contains only the numerical
parameters and the unit-conversion functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwarePlatform:
    """Per-spin energy and time of a single Bernoulli/Metropolis
    update on the platform. Used to convert sweep counts into
    physical time and energy."""

    name: str
    t_update: float       # seconds per spin update
    e_update: float       # joules per spin update
    parallel_n: int = 1   # how many spins update simultaneously
    notes: str = ""

    def time_per_sweep(self, n_spin: int) -> float:
        """Wall time of one full sweep (n_spin spin updates)."""
        passes = max(1, (n_spin + self.parallel_n - 1) // self.parallel_n)
        return passes * self.t_update

    def energy_per_sweep(self, n_spin: int) -> float:
        return n_spin * self.e_update

    def hardware_tts(self, n_spin: int, n_sweeps: int,
                     p_success: float, conf: float = 0.99) -> float:
        """Wall-time TTS_99 in seconds at the given confidence."""
        if p_success <= 0.0 or p_success >= 1.0:
            if p_success >= 1.0:
                return n_sweeps * self.time_per_sweep(n_spin)
            return float("inf")
        import math
        n_runs = math.log(1.0 - conf) / math.log(1.0 - p_success)
        return n_runs * n_sweeps * self.time_per_sweep(n_spin)

    def energy_per_solution(self, n_spin: int, n_sweeps: int,
                            p_success: float, conf: float = 0.99) -> float:
        """Joules per success at the given confidence."""
        if p_success <= 0.0:
            return float("inf")
        if p_success >= 1.0:
            return n_sweeps * self.energy_per_sweep(n_spin)
        import math
        n_runs = math.log(1.0 - conf) / math.log(1.0 - p_success)
        return n_runs * n_sweeps * self.energy_per_sweep(n_spin)


# ---------------------------------------------------------------------------
# Calibration table — ALL values traceable to the cited sources.
# When a number is computed from Section 2.3 measurements rather than
# quoted directly, the comment shows the formula.
# ---------------------------------------------------------------------------

# sMTJ-array (this work, Section 2.3)
#   t_update = t_w = 0.75 ns                       (Section 2.3.3)
#   e_update = V_th^2 / R_SOT * t_w
#            = (0.90)^2 / 776 * 0.75e-9
#            = 7.83e-13 J = 0.78 pJ                (Section 2.3.3)
#   parallel_n: a single tile fires N pulses in parallel through the
#               column drivers. We default to 64 spins per tile, the
#               conservative figure for column-multiplex-friendly
#               300 mm SOT-MRAM bit-cell pitch. The chapter discusses
#               larger tile sizes as future work.
SMTJ_ARRAY = HardwarePlatform(
    name="sMTJ-array",
    t_update=0.75e-9,
    e_update=7.83e-13,
    parallel_n=64,
    notes="Section 2.3 calibration: t_w = 0.75 ns, "
          "E_write = V_th^2 / R_SOT * t_w = 0.78 pJ.",
)

# CMOS p-bit ASIC, conservative per-cell figures.
# Reference: K. Y. Camsari et al., "p-Bits for probabilistic spin
# logic," Proc. IEEE 2020, doi:10.1109/JPROC.2020.2966869.
# Per-update energy taken at 5 pJ (CMOS Bernoulli generator + LUT
# weighted-sum); per-update time at 5 ns clocked.
CMOS_PBIT = HardwarePlatform(
    name="cmos-pbit",
    t_update=5.0e-9,
    e_update=5.0e-12,
    parallel_n=64,
    notes="CMOS p-bit ASIC class, Camsari 2020 doi:10.1109/JPROC.2020.2966869.",
)

# FPGA SBM at Ising-machine scale.
# Reference: H. Goto et al., "High-performance combinatorial
# optimization based on classical mechanics," Sci. Adv. 7, eabe7953
# (2021), doi:10.1126/sciadv.abe7953. The figure is the simulated
# bifurcation FPGA implementation, not Metropolis SA. Per-update
# time ~1 ns at GHz clock; per-update energy ~1 nJ given total chip
# power and update rate. Numbers are static-leakage-dominated.
FPGA_SBM = HardwarePlatform(
    name="fpga-sbm",
    t_update=1.0e-9,
    e_update=1.0e-9,
    parallel_n=256,
    notes="FPGA simulated-bifurcation Ising machine, Goto 2021 "
          "doi:10.1126/sciadv.abe7953.",
)
# Backwards-compatibility alias for any caller written before the rename.
FPGA_SA = FPGA_SBM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cpu_platform_from_run(time_median_seconds: float, n_sweeps: int,
                          n_spin: int,
                          tdp_watts: float = 28.0) -> HardwarePlatform:
    """Synthesise a HardwarePlatform record from an observed isim run.
    time_median_seconds is what summary.csv reports under
    time_median; n_sweeps is the SolverConfig.n_sweeps used.
    A typical Intel-class single-core TDP of 28 W is assumed for the
    energy estimate; users should adjust to their CPU's measured TDP."""

    if time_median_seconds <= 0 or n_sweeps <= 0 or n_spin <= 0:
        raise ValueError("time_median_seconds, n_sweeps and n_spin "
                         "must all be positive")

    t_update = time_median_seconds / (n_sweeps * n_spin)
    e_update = t_update * tdp_watts
    return HardwarePlatform(
        name="cpu-numba",
        t_update=t_update,
        e_update=e_update,
        parallel_n=1,
        notes=f"Calibrated from observed run: "
              f"time_median = {time_median_seconds:.3g} s over "
              f"{n_sweeps} sweeps on {n_spin} spins, "
              f"TDP={tdp_watts:g} W.",
    )


def all_default_platforms():
    """Return the hard-coded comparator platforms (sMTJ, CMOS, FPGA).
    Caller may extend with cpu_platform_from_run for a runtime
    baseline."""
    return [SMTJ_ARRAY, CMOS_PBIT, FPGA_SBM]


# ---------------------------------------------------------------------------
# Compact reporter
# ---------------------------------------------------------------------------
def report_table(platforms, n_spin: int, n_sweeps: int,
                 p_success: float, conf: float = 0.99) -> str:
    """Plain-text table comparing platforms on a single instance."""
    rows = []
    rows.append(f"{'platform':<14s} {'t/sweep':>10s} {'TTS@99%':>11s} "
                f"{'E/sol':>11s} {'parallel_n':>11s}")
    rows.append("-" * 60)
    for plat in platforms:
        ts = plat.time_per_sweep(n_spin)
        tts = plat.hardware_tts(n_spin, n_sweeps, p_success, conf)
        en = plat.energy_per_solution(n_spin, n_sweeps, p_success, conf)
        rows.append(f"{plat.name:<14s} {_eng(ts, 's'):>10s} "
                    f"{_eng(tts, 's'):>11s} {_eng(en, 'J'):>11s} "
                    f"{plat.parallel_n:>11d}")
    return "\n".join(rows)


def _eng(x, unit):
    if x == 0 or x is None:
        return "0"
    if x == float("inf"):
        return "inf"
    import math
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1.0:
        return f"{sign}{x:.2g} {unit}"
    e = math.floor(math.log10(x))
    if e >= -3:
        return f"{sign}{x*1e3:.2f} m{unit}"
    if e >= -6:
        return f"{sign}{x*1e6:.2f} u{unit}"
    if e >= -9:
        return f"{sign}{x*1e9:.2f} n{unit}"
    if e >= -12:
        return f"{sign}{x*1e12:.2f} p{unit}"
    if e >= -15:
        return f"{sign}{x*1e15:.2f} f{unit}"
    return f"{sign}{x:.2e} {unit}"
