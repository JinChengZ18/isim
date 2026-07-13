#!/usr/bin/env python3
"""Shared plumbing for the isim eda/ testbenches (WSL ngspice + sky130 + OSDI).

Conventions (inherited from smtj_pbnn_sim/eda, same thesis):
  * every driver script is run INSIDE WSL (Ubuntu-24.04-EDA):
      wsl -d Ubuntu-24.04-EDA --cd <repo> -- python3 eda/testbenches/<script>.py
  * the compiled OSDI lives next to the testbenches; `.spiceinit` in this
    directory loads it with a RELATIVE path (the repo path contains spaces);
  * the RNG never lives in the Verilog-A model: harness owns seeds, and every
    seed is recorded in the *_summary.json it produces;
  * nothing is typed by hand: every number in a summary JSON is computed here.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EDA = HERE.parent
MODEL = EDA / "models" / "smtj_sot.va"
OSDI = HERE / "smtj_sot.osdi"
SKY130_LIB = "/opt/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice"

# Chapter-2 calibration (Device A, P->AP, t_p = 0.75 ns) — single source of
# truth is eda/models/smtj_sot.va; these mirrors are for harness-side math.
VTH = 0.895783          # sigmoid center [V]
VT = 0.023414           # probability window = 1/beta_s [V]
RSOT = 776.0            # SOT write resistance [ohm]
RP = 4900.0             # MTJ parallel resistance [ohm]
TMR = 1.0
TW = 0.75e-9            # write pulse width [s]
# Chapter-2 measured reset ceiling: Device A AP->P back-hopping plateau.
# A single reset pulse (AP->P direction) succeeds with probability <= R_RESET.
R_RESET = 0.72

trapz = getattr(np, "trapezoid", np.trapz)


def psw(v):
    """Committed operating-point switching probability (P->AP direction)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(v, float) - VTH) / VT))


def ensure_osdi():
    """Compile models/smtj_sot.va -> testbenches/smtj_sot.osdi if stale."""
    if OSDI.exists() and OSDI.stat().st_mtime >= MODEL.stat().st_mtime:
        return
    ov = shutil.which("openvaf") or shutil.which("openvaf-r")
    if not ov:
        raise RuntimeError("openvaf not found in PATH (run inside WSL Ubuntu-24.04-EDA)")
    subprocess.run([ov, str(MODEL), "-o", str(OSDI)], check=True, cwd=HERE)


def ensure_spiceinit():
    want = "osdi smtj_sot.osdi\n"
    si = HERE / ".spiceinit"
    if not si.exists() or si.read_text() != want:
        si.write_text(want)


def run_deck(text, tag):
    """Write a deck under testbenches/ and run `ngspice -b` there; return stdout."""
    ensure_osdi()
    ensure_spiceinit()
    deck = HERE / f"_{tag}.spice"
    deck.write_text(text)
    r = subprocess.run(["ngspice", "-b", deck.name], cwd=HERE,
                       capture_output=True, text=True)
    (HERE / f"_{tag}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"ngspice failed for {tag}; see _{tag}.log")
    return r.stdout


def grab(out, node):
    """Parse `print v(node)` output of an .op deck."""
    m = re.search(rf"v\({re.escape(node)}\)\s*=\s*([-+0-9.eE]+)", out)
    return float(m.group(1)) if m else float("nan")


def load_wrdata(path, ncols):
    """Load an ngspice `wrdata` CSV: rows of (t, v1, t, v2, ...)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "*#":
            continue
        parts = [x for x in line.replace(",", " ").split() if x]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    a = np.array(rows)
    t = a[:, 0]
    vs = [a[:, 2 * i + 1] for i in range(ncols)]
    return t, vs


def write_summary(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
    print(f"  -> {Path(path).name}")
