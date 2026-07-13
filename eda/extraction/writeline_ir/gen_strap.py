#!/usr/bin/env python3
"""W5: sky130 met2 write-line straps for the Ising-tile IR-drop extraction.

Ising context: the Table-3.8 hardware projection assumes N_par = 64 spins updated in
parallel — a 64x64 tile, row-sequential. Each COLUMN write line serves 64 rows at a
2 um cell pitch; the metal resistance between the column driver and row r drops the
write voltage by I_write * R_line(r) (I_write ~ Vth/(Rsot + R_line) ~ 1.15 mA), which
shifts the sigmoid drive of that row's sMTJ by a static per-row offset
u_off(r) = dV(r)/VT on the probability window (VT = 23.414 mV).

This script (ported from smtj_pbnn_sim/eda/extraction/writeline/gen_writeline_straps.py)
generates known-geometry rectangles with a label at each end for Magic extraction
(run_extresist.sh), from which we back out the effective sheet resistance Rs = R * W / L:

  * polycal : poly, L = 200 um, W = 0.5 um -> 400 squares — the PBNN-style calibration
              strap. Magic `extresist` outputs a two-port R only for high-R nets (the
              PBNN run showed li1/met1..met3 straps are extracted but dropped by the
              net filter at any tolerance), so poly (~19 kohm) is the strap that
              validates the extresist flow against the techfile 48.2 ohm/sq;
  * cal     : met2, same 400-square geometry — met2 Rs comes from the lumped net R
              that `extract do resistance` writes into the .ext node records
              (cross-checked against the techfile 0.125 ohm/sq);
  * n16/n64/n256 : met2, L = N * 2 um pitch, W = 1.0 um — the actual tile write-line
              geometries for N rows in {16, 64, 256} (64 = the Table-3.8 tile).

Run IN WSL via KLayout batch:
  wsl -d Ubuntu-24.04-EDA --cd <repo> -- klayout -b -r eda/extraction/writeline_ir/gen_strap.py
"""
import os
import pya

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "writeline_straps.gds")

PITCH_UM = 2.0                     # Ising tile cell pitch (same assumption as PBNN flow)
POLY = (66, 20, 5)                 # sky130 GDS layer: (layer number, draw dt, label dt)
MET2 = (69, 20, 5)
# name -> (layer tuple, length_um, width_um)
STRAPS = [
    ("polycal", POLY, 200.0,          0.5),  # 400-sq poly strap: extresist flow validation
    ("cal",     MET2, 200.0,          0.5),  # 400-sq met2 strap: lumped-R Rs measurement
    ("n16",     MET2, 16 * PITCH_UM,  1.0),  # 16-row column write line
    ("n64",     MET2, 64 * PITCH_UM,  1.0),  # 64-row column = the Table-3.8 tile
    ("n256",    MET2, 256 * PITCH_UM, 1.0),  # 256-row scaling point
]
ROW_PITCH_UM = 5.0                 # vertical spacing between straps in the GDS

ly = pya.Layout()
ly.dbu = 0.001
top = ly.create_cell("writeline_straps")

for i, (name, (lnum, draw_dt, lab_dt), l_um, w_um) in enumerate(STRAPS):
    draw = ly.layer(lnum, draw_dt)
    lab = ly.layer(lnum, lab_dt)
    y = int(i * ROW_PITCH_UM / ly.dbu)
    L = int(l_um / ly.dbu)
    W = int(w_um / ly.dbu)
    top.shapes(draw).insert(pya.Box(0, y, L, y + W))
    # a label at each end, on the sky130 label datatype so Magic attaches them as net names
    ta = pya.Text("%s_a" % name, pya.Trans(pya.Point(int(0.5 / ly.dbu), y + W // 2)))
    tb = pya.Text("%s_b" % name, pya.Trans(pya.Point(L - int(0.5 / ly.dbu), y + W // 2)))
    top.shapes(lab).insert(ta)
    top.shapes(lab).insert(tb)

ly.write(OUT)
bb = top.bbox()
print("GDS_WRITTEN %s" % OUT)
for name, layer, l_um, w_um in STRAPS:
    print("strap %-7s layer=%d L=%7.1fum W=%.2fum squares=%6.0f"
          % (name, layer[0], l_um, w_um, l_um / w_um))
print("bbox_um=%.1f x %.1f" % (bb.width() * ly.dbu, bb.height() * ly.dbu))
