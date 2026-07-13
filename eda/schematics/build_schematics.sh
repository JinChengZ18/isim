#!/bin/bash
# Export an Xschem .sch as a journal-grade figure: xschem -> SVG (vector, white bg + BLACK wires from
# the local xschemrc), then cairosvg rasterizes the SVG to a HIGH-RESOLUTION PNG and a PDF.
# Headless via the WSLg X server (DISPLAY=:0); timeout-guarded (always -q; never interactive).
#   wsl -d Ubuntu-24.04-EDA -- bash -lc 'cd "<repo>/eda/schematics" && bash build_schematics.sh update_chain.sch'
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
RC="$DIR/xschemrc"          # sky130 rc + local cleaned symbols + light scheme + black wires
PNG_W="${PNG_W:-2400}"      # high-res PNG width (px)
export DISPLAY=:0
cd "$DIR"
for sch in "$@"; do
  base="${sch%.sch}"
  timeout 90 xschem --rcfile "$RC" -q --svg --plotfile "$DIR/$base.svg" "$DIR/$sch" </dev/null >/dev/null 2>&1
  [ "$?" -eq 124 ] && { echo "$base: xschem TIMEOUT"; continue; }
  python3 -c "import cairosvg; b='$base'; cairosvg.svg2png(url=b+'.svg', write_to=b+'.png', output_width=$PNG_W); cairosvg.svg2pdf(url=b+'.svg', write_to=b+'.pdf')" 2>/dev/null
  dim=$(python3 -c "import struct; d=open('$base.png','rb').read(26); w,h=struct.unpack('>II',d[16:24]); print(str(w)+'x'+str(h))" 2>/dev/null || echo '?')
  echo "$base -> svg $([ -f $base.svg ] && echo ok || echo -), png $dim, pdf $([ -f $base.pdf ] && echo ok || echo -)"
done
