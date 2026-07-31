#!/usr/bin/env python3
"""Generate a shields-style coverage SVG from lcov.info or coverage.xml. No network."""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def pct_from_lcov(path: Path) -> float:
    found = hit = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("LF:"):
            found += int(line[3:] or 0)
        elif line.startswith("LH:"):
            hit += int(line[3:] or 0)
    if found <= 0:
        return 0.0
    return 100.0 * hit / found


def pct_from_cobertura(path: Path) -> float:
    root = ET.parse(path).getroot()
    rate = root.attrib.get("line-rate")
    if rate is not None:
        return float(rate) * 100.0
    return 0.0


def color_for(pct: float) -> str:
    if pct >= 90:
        return "#4c1"
    if pct >= 80:
        return "#97ca00"
    if pct >= 70:
        return "#a4a61d"
    if pct >= 50:
        return "#dfb317"
    if pct >= 30:
        return "#fe7d37"
    return "#e05d44"


def svg_badge(label: str, value: str, color: str) -> str:
    # Approximate character widths for DejaVu Sans-ish metrics
    def text_width(s: str) -> int:
        return max(1, int(round(len(s) * 6.5 + 10)))

    lw = text_width(label)
    rw = text_width(value)
    tw = lw + rw
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{tw}" height="20" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{tw}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="20" fill="#555"/>
    <rect x="{lw}" width="{rw}" height="20" fill="{color}"/>
    <rect width="{tw}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="{lw/2 * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(lw-10)*10}">{label}</text>
    <text x="{lw/2 * 10}" y="140" transform="scale(.1)" textLength="{(lw-10)*10}">{label}</text>
    <text aria-hidden="true" x="{(lw + rw/2) * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(rw-10)*10}">{value}</text>
    <text x="{(lw + rw/2) * 10}" y="140" transform="scale(.1)" textLength="{(rw-10)*10}">{value}</text>
  </g>
</svg>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lcov", type=Path)
    p.add_argument("--cobertura", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default="coverage")
    args = p.parse_args()
    if args.lcov and args.lcov.is_file():
        pct = pct_from_lcov(args.lcov)
    elif args.cobertura and args.cobertura.is_file():
        pct = pct_from_cobertura(args.cobertura)
    else:
        print("no coverage input found", file=sys.stderr)
        return 1
    value = f"{pct:.1f}%"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg_badge(args.label, value, color_for(pct)), encoding="utf-8")
    print(f"wrote {args.out} ({value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
