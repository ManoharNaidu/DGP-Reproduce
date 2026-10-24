#!/usr/bin/env python3
"""WCAG contrast check of palette.json (dark + light): text colours against the page and both card surfaces,
plus a no-hex-in-page check (index.html must use the palette variables)."""
import json, re, sys
def lum(h):
    c = [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= .03928 else ((x + .055) / 1.055) ** 2.4 for x in c]
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]
def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + .05) / (lb + .05)
NEED = {"fg": 4.5, "fg2": 4.5, "muted": 4.5, "accent": 4.5, "ok": 4.5, "warn": 4.5, "bad": 4.5}
p = json.load(open("palette.json")); bad = []; worst = {}
for mode, c in p.items():
    for k, need in NEED.items():
        for surf in ("bg", "card", "card2"):
            r = ratio(c[k], c[surf]); worst[(mode, k)] = min(worst.get((mode, k), 99), r)
            if r < need: bad.append(f"{mode}.{k} on {surf} {r:.2f} < {need}")
    r = ratio(c["bar2"], c["card"])
    if r < 3: bad.append(f"{mode}.bar2 on card {r:.2f} < 3 (non-text graphics)")
if re.search(r"#[0-9a-fA-F]{3,6}\b", open("index.html").read()): bad.append("index.html has a hex colour; use palette vars")
if "-v" in sys.argv: print("\n".join(f"{m}.{k}: min {r:.2f}" for (m, k), r in sorted(worst.items())))
print("OK" if not bad else "FAIL\n" + "\n".join(bad)); sys.exit(bool(bad))
