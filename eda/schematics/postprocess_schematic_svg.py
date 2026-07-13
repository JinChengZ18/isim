#!/usr/bin/env python3
"""Post-process an Xschem-exported SVG so that math-style labels render with
real sub/superscripts instead of literal ASCII underscores/carets.

Xschem text objects are plain strings, so a label written `V_wr = V_th + u*V_T`
comes out with literal underscores. This pass rewrites, inside every <text>
element, patterns `X_sub` -> X + subscript, `X^sup` -> X + superscript, and the
ASCII arrow `->` -> `→`, emitting cairosvg-compatible <tspan dy=...> runs. It
runs between the xschem SVG export and the cairosvg rasterization in
build_schematics.sh, so the tracked update_chain.svg/.png carry proper
subscripts (net-name identifiers without an underscore are untouched).

Usage: python3 postprocess_schematic_svg.py <file.svg>  (edits in place)
"""
import re
import sys

TEXT_RE = re.compile(r'(<text\b[^>]*?font-size="([0-9.]+)"[^>]*>)(.*?)(</text>)',
                     re.S)
SUB_RE = re.compile(r'([A-Za-z])_([A-Za-z0-9]+)')
SUP_RE = re.compile(r'([A-Za-z0-9])\^([A-Za-z0-9]+)')


def runs(content):
    """Yield (text, level) with level 0 normal, +1 subscript, -1 superscript."""
    out = []
    i = 0
    while i < len(content):
        ms = SUB_RE.match(content, i)
        mp = SUP_RE.match(content, i)
        if ms:
            out.append((ms.group(1), 0))
            out.append((ms.group(2), 1))
            i = ms.end()
        elif mp:
            out.append((mp.group(1), 0))
            out.append((mp.group(2), -1))
            i = mp.end()
        else:
            ch = content[i]
            if out and out[-1][1] == 0:
                out[-1] = (out[-1][0] + ch, 0)
            else:
                out.append((ch, 0))
            i += 1
    return out


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def convert(content, fs):
    # arrow first (SVG-escaped and raw)
    content = content.replace("-&gt;", "→").replace("->", "→")
    if "_" not in content and "^" not in content:
        return esc(content)
    sub_fs = round(fs * 0.72, 3)
    down = round(fs * 0.30, 3)
    up = round(fs * 0.34, 3)
    parts = []
    cur = 0.0
    for text, lvl in runs(content):
        want = down if lvl == 1 else (-up if lvl == -1 else 0.0)
        dy = round(want - cur, 3)
        cur = want
        attrs = []
        if dy != 0:
            attrs.append(f'dy="{dy}"')
        if lvl != 0:
            attrs.append(f'font-size="{sub_fs}"')
        else:
            attrs.append(f'font-size="{fs}"')
        parts.append(f'<tspan {" ".join(attrs)}>{esc(text)}</tspan>')
    return "".join(parts)


def repl(m):
    open_tag, fs, content, close = m.groups()
    return open_tag + convert(content, float(fs)) + close


def main(path):
    svg = open(path, encoding="utf-8").read()
    new = TEXT_RE.sub(repl, svg)
    open(path, "w", encoding="utf-8", newline="\n").write(new)
    n = len(TEXT_RE.findall(svg))
    print(f"postprocessed {n} <text> elements in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
