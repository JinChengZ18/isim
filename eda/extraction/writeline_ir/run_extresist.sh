#!/bin/bash
# W5: Magic resistance extraction of the Ising-tile met2 write-line straps.
# Ported from smtj_pbnn_sim/eda/extraction/writeline/run_extresist.sh.
# Runs from an ASCII ext4 build dir (the /tmp-tmpfs idle-wipe reason; repo path has spaces).
#
# Self-checks (two levels, both vs sky130A.tech `resist` values):
#   1. poly cal strap via extresist two-port R (the PBNN check: 47.96 vs 48.2 = 0.5%;
#      the 0.5% is the label inset — labels sit 0.5 um in from each end, 398/400 sq).
#      Magic extresist drops low-R nets at ANY tolerance (observed: met straps are
#      "extracted" but never "output"), so only the ~19 kohm poly strap survives.
#   2. met2 straps via the lumped net R that `extract do resistance` writes into the
#      .ext node records (vs techfile 125 milliohm/sq = 0.125 ohm/sq).
#
# INVOKE from this dir:
#   wsl -d Ubuntu-24.04-EDA -- bash -lc \
#     'cd "<repo>/eda/extraction/writeline_ir" && bash run_extresist.sh'
set -u
BUILD=/home/lenovo/isim_eda_build
RC=/opt/pdk/sky130A/libs.tech/magic/sky130A.magicrc
TOP=writeline_straps
SRC="$(pwd)"

mkdir -p "$BUILD"
cp writeline_straps.gds "$BUILD/wl.gds"
cp "$RC" "$BUILD/.magicrc"
sync
sz=$(wc -c < "$BUILD/wl.gds" 2>/dev/null || echo 0)
echo "staged $sz bytes -> $BUILD/wl.gds"
[ "$sz" -gt 200 ] || { echo "ERROR: GDS copy did not land; re-run."; exit 1; }

export PDK_ROOT=/opt/pdk PDK=sky130A
cd "$BUILD"
cat > extres.tcl <<EOF
gds read wl.gds
load $TOP
select top cell
extract do resistance
extract all
extresist tolerance 1
extresist all
ext2spice extresist on
ext2spice cthresh infinite
ext2spice -o wl_res.spice
puts "EXTRES_DONE"
quit -noprompt
EOF

magic -dnull -noconsole -rcfile .magicrc extres.tcl > extres.log 2>&1
echo "--- extres.log tail ---"; tail -15 extres.log
echo "--- wl_res.spice (resistor lines) ---"
grep -iE "^R|\.subckt|^\*" wl_res.spice 2>/dev/null | head -40 || echo "(no spice produced)"

# copy artifacts back so analyze_ir.py (Windows or WSL) can parse the MEASURED values
cp wl_res.spice "$SRC/wl_res.spice" 2>/dev/null && echo "copied wl_res.spice -> $SRC"
cp writeline_straps.ext "$SRC/writeline_straps.ext" 2>/dev/null && echo "copied .ext -> $SRC"
cp extres.log "$SRC/extres.log" 2>/dev/null && echo "copied extres.log -> $SRC"

# self-check 1: poly cal strap (extresist two-port) vs techfile 48.2 ohm/sq
# self-check 2: met2 straps (.ext lumped node R) vs techfile 0.125 ohm/sq
python3 - "$SRC/wl_res.spice" "$SRC/writeline_straps.ext" <<'PYEOF'
import re, sys
spice = open(sys.argv[1]).read()
rs = [(a, b, float(v)) for a, b, v in
      re.findall(r"^R\S*\s+(\S+)\s+(\S+)\s+([0-9.eE+-]+)", spice, re.M)]
poly = sum(v for a, b, v in rs if "polycal" in a or "polycal" in b)
if poly:
    rsq = poly * 0.5 / 200.0     # Rs = R * W / L (nominal 400 sq; labels inset 0.5 um)
    print("SELF-CHECK 1 (extresist, poly): R=%.1f ohm -> %.2f ohm/sq vs techfile 48.2 "
          "(delta %.2f%%)" % (poly, rsq, (rsq / 48.2 - 1.0) * 100.0))
else:
    print("SELF-CHECK 1: no polycal resistor in wl_res.spice — inspect extres.log")
ext = open(sys.argv[2]).read()
geom = {"cal": (200.0, 0.5), "n16": (32.0, 1.0), "n64": (128.0, 1.0), "n256": (512.0, 1.0)}
for name, (l_um, w_um) in geom.items():
    m = re.search(r'^node "%s_[ab]" (\d+(?:\.\d+)?) ' % name, ext, re.M)
    if not m:
        print("SELF-CHECK 2: met2 %s node not found in .ext" % name)
        continue
    r = float(m.group(1))        # lumped net R, ohms (rscale 1000 x milliohms)
    rsq = r * w_um / l_um
    print("SELF-CHECK 2 (extract-do-resistance, met2 %-4s): R=%6.1f ohm -> %.4f ohm/sq "
          "vs techfile 0.125 (delta %.2f%%)" % (name, r, rsq, (rsq / 0.125 - 1.0) * 100.0))
PYEOF
echo "  netlist: $BUILD/wl_res.spice"
echo "  log:     $BUILD/extres.log"
