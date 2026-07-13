#!/usr/bin/env python3
"""Generate a journal-style Xschem schematic of the Ising p-bit WRITE CHAIN.

Draws, left to right, the exact chain simulated by eda/testbenches/update_chain_dc.py:

  resistor-string DAC (abstracted: dashed box, 3 of the 2^b unit resistors drawn, rails
  vhi/vlo = Vth +/- 4*VT)  ->  code-select CMOS transmission gate (TGn 4/0.15, TGp 8/0.15)
  ->  two-stage Miller unity buffer at transistor level (NMOS input pair M1/M2 20/0.5,
  PMOS mirror M3/M4 20/0.5 with M4 diode-connected, tail Mt 2x 20/0.5 biased at 0.9 V,
  common-source PMOS output Mo 2x 80/0.5, Ccm = 2 pF Miller cap; the feedback wire runs
  from node wr AFTER the write-enable gate back to the inverting input M2)  ->
  write-enable transmission gate (x8 parallel fingers, per-finger 4/0.15 + 8/0.15)  ->
  sMTJ p-bit device (local sym/sot_mtj.sym extended with an st state-control pin;
  wr/rd/com/st are the committed OSDI model terminals, SOT branch wr->com to ground).

Device names are the update_chain_dc.py deck names minus the X spice-prefix (the symbol
re-adds it); net names (vhi, vlo, bin, bd1, bd2, btail, drv, wr, wen, wep, com, st) are
the deck's.  Rail voltages are computed here from the same chapter-2 calibration
constants as eda/testbenches/_common.py (VTH = 0.895783 V, VT = 0.023414 V), never
hand-typed.  House conventions follow the PBNN hero generators
(gen_yoon_pbit_driver_sch.py): gate labels exit left, NMOS bulks -> VSS gnd stubs right,
PMOS bulk tied to source only where source == vdd, TG pfets drawn with D/S vertically
swapped (symmetric devices; drawn net connectivity is exact).

Run (any Python, no EDA tools needed):   python3 eda/schematics/gen_update_chain_sch.py
Export (WSL, xschem + cairosvg):         bash eda/schematics/build_schematics.sh update_chain.sch
"""
import os

VTH, VT, USPAN = 0.895783, 0.023414, 4.0     # chapter-2 calibration (eda/testbenches/_common.py)
RSOT = 776                                   # SOT write-branch resistance [ohm]

NFET, PFET, RES, CAP, SOT = ("sym/nfet.sym", "sym/pfet.sym", "sym/res.sym",
                             "sym/cap.sym", "sym/sot_mtj.sym")
# pin offsets (rot 0), from the symbol bounding boxes
P = {"n": {"D": (20, -30), "G": (-20, 0), "S": (20, 30), "B": (20, 0)},
     "p": {"S": (20, -30), "G": (-20, 0), "D": (20, 30), "B": (20, 0)},
     "r": {"P": (0, -30), "M": (0, 30)},
     "c": {"P": (0, -30), "M": (0, 30)},
     "sot": {"Tin": (-40, 20), "Tsl": (40, 20), "Trd": (0, -40), "Tst": (30, 7)}}

VDD_Y = 110                                  # buffer supply rail
XL, XR = 620, 790                            # buffer left (M4/M2) / right (M3/M1) columns
XO = 950                                     # output-stage column
TG_Y = 340                                   # both transmission gates sit at this row

# (name, type, x, y, {attrs})
DEV = [
    ("Rs1", "r", 190, 270, ""), ("Rs2", "r", 190, 350, ""), ("RsN", "r", 190, 430, ""),
    ("TGn0", "n", 340, TG_Y, "W=4 L=0.15"),   # code-select TG
    ("TGp0", "p", 500, TG_Y, "W=8 L=0.15"),
    ("M4", "p", XL, 190, "W=20 L=0.5"),       # mirror, diode-connected (bd2)
    ("M3", "p", XR, 190, "W=20 L=0.5"),       # mirror output (bd1)
    ("M2", "n", XL, 310, "W=20 L=0.5"),       # inverting input (gate = wr feedback)
    ("M1", "n", XR, 310, "W=20 L=0.5"),       # non-inverting input (gate = bin)
    ("Mt", "n", 690, 430, "W=20 L=0.5"),      # tail, 2 parallel (x2)
    ("Mo", "p", XO, 190, "W=80 L=0.5"),       # common-source output, 2 parallel (x2)
    ("WEn", "n", 1080, TG_Y, "W=4 L=0.15"),   # write-enable TG, 8 parallel fingers (x8)
    ("WEp", "p", 1240, TG_Y, "W=8 L=0.15"),
    ("N1", "sot", 1420, 350, ""),             # committed OSDI device
]
DT = {n: t for n, t, *_ in DEV}
XY = {n: (x, y) for n, t, x, y, a in DEV}


def pin(n, p):
    x, y = XY[n]; dx, dy = P[DT[n]][p]; return x + dx, y + dy


def main():
    o = ["v {xschem version=3.4.4 file_version=1.2}\nG {}\nK {}\nV {}\nS {}\nE {}\n"]
    for n, t, x, y, a in DEV:
        if t in ("n", "p"):
            o.append("C {%s} %d %d 0 0 {name=%s model=sky130_fd_pr__%sfet_01v8 %s nf=1 m=1}\n"
                     % (NFET if t == "n" else PFET, x, y, n, t, a))
        elif t == "r":
            o.append("C {%s} %d %d 0 0 {name=%s value=R_u}\n" % (RES, x, y, n))
        elif t == "sot":
            o.append("C {%s} %d %d 0 0 {name=%s}\n" % (SOT, x, y, n))
    o.append("C {%s} 920 320 0 0 {name=Ccm value=2p}\n" % CAP)

    W = []
    # ---- resistor-string DAC: vhi -> Rs1 -> tap -> Rs2 -> ... -> RsN -> vlo ----
    W += [(190, 225, 190, 240)]                                  # vhi entry
    W += [(190, 300, 190, 310), (190, 310, 190, 320)]            # Rs1.M -> tap -> Rs2.P
    W += [(190, 380, 190, 400)]                                  # Rs2.M -> RsN.P
    W += [(190, 460, 190, 475)]                                  # RsN.M -> vlo
    W += [(40, 340, 80, 340)]                                    # b-bit code bus (conceptual)
    # ---- selected tap -> code-select TG (top = tap, bottom = bin) ----
    W += [(190, 310, 360, 310), (360, 310, 520, 310)]
    W += [(360, 370, 520, 370), (520, 370, 540, 370)]
    # ---- bin -> non-inverting input M1 (routes through the mirror/input gap) ----
    W += [(540, 370, 540, 285), (540, 285, 755, 285), (755, 285, 755, 310),
          (755, 310, 770, 310)]
    # ---- buffer: VDD rail + PMOS sources (bulk = source = vdd) ----
    W += [(XL + 20, VDD_Y, XR + 20, VDD_Y), (XR + 20, VDD_Y, XO + 20, VDD_Y)]
    for xc in (XL + 20, XR + 20, XO + 20):
        W += [(xc, VDD_Y, xc, 160), (xc, 190, xc, 160)]
    # ---- first stage: bd2 (diode) / bd1 columns, mirror gate bus ----
    W += [(XL + 20, 220, XL + 20, 245), (XL + 20, 245, XL + 20, 280)]      # bd2
    W += [(XR + 20, 220, XR + 20, 250), (XR + 20, 250, XR + 20, 280)]      # bd1
    W += [(600, 190, 585, 190), (585, 190, 585, 245), (585, 245, XL + 20, 245)]  # M4 diode tie
    W += [(XL + 20, 245, 740, 245), (740, 245, 740, 190), (740, 190, 770, 190)]  # mirror bus
    # ---- bd1 -> output gate + Miller cap Ccm (bd1 -> drv) ----
    W += [(XR + 20, 250, 920, 250), (920, 250, 920, 190), (920, 190, 930, 190)]
    W += [(920, 250, 920, 290)]                                  # bd1 -> Ccm.P
    W += [(920, 350, 920, 390), (920, 390, 1010, 390), (1010, 390, 1010, 310)]  # Ccm.M -> drv
    # ---- input-pair sources -> btail -> tail device -> VSS ----
    W += [(XL + 20, 340, XL + 20, 370), (XR + 20, 340, XR + 20, 370)]
    W += [(XL + 20, 370, 710, 370), (710, 370, XR + 20, 370)]
    W += [(710, 370, 710, 400), (710, 460, 710, 490)]
    # ---- output stage drain = drv -> write-enable TG (top = drv, bottom = wr) ----
    W += [(XO + 20, 220, XO + 20, 310), (XO + 20, 310, 1010, 310), (1010, 310, 1100, 310)]
    W += [(1100, 310, 1260, 310)]
    W += [(1100, 370, 1260, 370)]
    # ---- wr -> sMTJ Tin; feedback tap AFTER the enable gate -> M2 gate ----
    W += [(1260, 370, 1380, 370)]
    W += [(1260, 370, 1260, 560), (1260, 560, 575, 560), (575, 560, 575, 310),
          (575, 310, 600, 310)]
    # ---- sMTJ terminals: com -> gnd, rd stub, st stub ----
    W += [(1460, 370, 1530, 370)]
    W += [(1420, 310, 1420, 280)]
    W += [(1450, 357, 1490, 357)]
    for seg in W:
        o.append("N %d %d %d %d {}\n" % seg)

    nid = [0]

    def lab(px, py, rot, name, sym="lab_pin.sym"):
        nid[0] += 1
        o.append("C {sym/%s} %d %d 0 %d {name=l%d lab=%s}\n" % (sym, px, py, rot, nid[0], name))

    def stublab(pc, vec, rot, name, sym="lab_pin.sym"):
        px, py = pc; ex, ey = px + vec[0], py + vec[1]
        o.append("N %d %d %d %d {}\n" % (px, py, ex, ey)); lab(ex, ey, rot, name, sym)

    def txt(s, x, y, size=0.26, layer=None):
        tag = "{layer=%d}" % layer if layer else "{}"
        o.append("T {%s} %d %d 0 0 %g %g %s\n" % (s, x, y, size, size, tag))

    # ---- DAC rails / code input ----
    lab(190, 225, 1, "vhi")
    lab(190, 475, 0, "vlo")
    lab(40, 340, 2, "code")
    o.append("L 4 58 348 66 332 {}\n")                           # bus-width slash
    txt("b-bit", 40, 312, 0.18)
    # dashed abstraction box around the DAC
    for seg in ((80, 200, 300, 200), (300, 200, 300, 480), (80, 480, 300, 480),
                (80, 200, 80, 480)):
        o.append("L 4 %d %d %d %d {dash=4}\n" % seg)
    # ---- gate net-labels (exit LEFT, house convention) ----
    stublab(pin("TGn0", "G"), (-25, 0), 2, "sel")
    stublab(pin("TGp0", "G"), (-25, 0), 2, "selb")
    stublab(pin("WEn", "G"), (-25, 0), 2, "wen")
    stublab(pin("WEp", "G"), (-25, 0), 2, "wep")
    stublab(pin("Mt", "G"), (-25, 0), 2, "Vnb", "ipin.sym")
    # ---- internal net labels (on existing wires) ----
    lab(450, 370, 0, "bin")
    lab(XL + 20, 232, 0, "bd2")
    lab(XR + 20, 235, 0, "bd1")
    lab(990, 310, 0, "drv")
    lab(900, 560, 0, "wr")
    lab(1330, 370, 0, "wr")
    lab(1500, 370, 0, "com")
    lab(1490, 357, 1, "st")
    lab(1420, 280, 1, "rd", "opin.sym")
    # ---- bulk ties ----
    for n in ("TGn0", "M2", "M1", "Mt", "WEn"):                  # NMOS bulks -> VSS
        stublab(pin(n, "B"), (25, 0), 0, "VSS", "gnd.sym")
    for n in ("TGp0", "WEp"):                                    # TG pfet bulk = vdd
        stublab(pin(n, "B"), (25, 0), 0, "VDD", "vdd.sym")
    # ---- rails ----
    stublab((XL + 20, VDD_Y), (0, -28), 1, "VDD", "vdd.sym")
    lab(710, 490, 0, "VSS", "gnd.sym")
    lab(1530, 370, 0, "VSS", "gnd.sym")
    # ---- stage titles ----
    txt("resistor-string DAC", 85, 162, 0.28)
    txt("code-select TG", 350, 248)
    txt("two-stage Miller unity buffer", 660, 55, 0.28)
    txt("write-enable TG (x8)", 1050, 248)
    txt("sMTJ p-bit device", 1345, 240)
    # ---- annotations ----
    txt("V_th + 4V_T = %.1f mV" % ((VTH + USPAN * VT) * 1e3), 60, 178, 0.22)
    txt("V_th - 4V_T = %.1f mV" % ((VTH - USPAN * VT) * 1e3), 60, 498, 0.22)
    txt("R_u = 100 ohm", 205, 288, 0.2)
    txt("2^b taps", 205, 320, 0.2)
    txt("(0.9 V)", 590, 445, 0.18)
    txt("x2", 725, 468, 0.22, 13)
    txt("x2", 985, 228, 0.22, 13)
    txt("feedback: sensed after write-enable TG", 700, 540, 0.22)
    txt("V_wr = V_th + u*V_T", 1280, 415, 0.26, 7)
    txt("0.75 ns pulse, P->AP probabilistic write", 1280, 440, 0.22)
    txt("R_SOT = %d ohm" % RSOT, 1280, 462, 0.2, 13)

    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, "update_chain.sch"), "w", newline="\n").write("".join(o))
    print("wrote update_chain.sch (%d devices + Ccm)" % len(DEV))


if __name__ == "__main__":
    main()
