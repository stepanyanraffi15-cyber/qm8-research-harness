"""Generate the README figures from the results files. No plotting library.

Two charts, each emitted twice -- once for a light GitHub theme, once for dark --
because a fixed text colour is invisible on the other background and the README
uses <picture> to pick.

Palette is the dataviz reference instance, slots 1 and 2, validated in both modes
(worst adjacent CVD dE 24.7 light / 26.8 dark against a >=8 target).

    python analysis/figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figures"

THEME = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#8a8a85", grid="#e6e5e1",
                  surface="#fcfcfb", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(ink="#ffffff", ink2="#c3c2b7", muted="#82817a", grid="#2c2c2a",
                 surface="#1a1a19", s1="#3987e5", s2="#d95926"),
}


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# 1. label budget
# --------------------------------------------------------------------------


def label_budget(t: dict) -> str:
    d = json.loads((ROOT / "results" / "label_budget.json").read_text())
    rows = d["random"]["rows"]
    cheap = d["random"]["cheap"]
    xs = [r["n_train"] or r["n_actual"] for r in rows]
    direct = [r["direct"] for r in rows]
    delta = [r["delta"] for r in rows]

    W, H = 720, 380
    L, R, TP, B = 62, 150, 34, 52
    pw, ph = W - L - R, H - TP - B
    lx = lambda v: L + pw * (math.log10(v) - math.log10(100)) / (math.log10(20000) - math.log10(100))  # noqa: E731
    ymax = 0.70
    ly = lambda v: TP + ph * (1 - v / ymax)  # noqa: E731

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']

    o.append(f'<text x="{L}" y="20" font-size="14" font-weight="600" fill="{t["ink"]}">'
             f'174× fewer expensive labels</text>')
    o.append(f'<text x="{L}" y="{H-12}" font-size="11" fill="{t["muted"]}">'
             f'CC2 training labels — E1, PBE0/def2TZVP baseline, random split, 3 seeds</text>')

    for gv in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        y = ly(gv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" fill="{t["muted"]}">{gv:.1f}</text>')
    o.append(f'<text x="{L-46}" y="{TP-8}" font-size="10.5" fill="{t["muted"]}">MAE (eV)</text>')

    for v, lab in ((100, "100"), (500, "500"), (2500, "2.5k"), (10000, "10k"), (17429, "all")):
        x = lx(v)
        o.append(f'<text x="{x:.1f}" y="{TP+ph+18}" font-size="10.5" text-anchor="middle" fill="{t["muted"]}">{lab}</text>')

    ych = ly(cheap)
    o.append(f'<line x1="{L}" y1="{ych:.1f}" x2="{L+pw}" y2="{ych:.1f}" stroke="{t["muted"]}" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{L+pw+8}" y="{ych+4:.1f}" font-size="11" fill="{t["ink2"]}">no learning {cheap:.3f}</text>')

    for series, colour, name in ((direct, t["s2"], "direct"), (delta, t["s1"], "delta")):
        pts = " ".join(f"{lx(x):.1f},{ly(y):.1f}" for x, y in zip(xs, series))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in zip(xs, series):
            o.append(f'<circle cx="{lx(x):.1f}" cy="{ly(y):.1f}" r="4" fill="{colour}" '
                     f'stroke="{t["surface"]}" stroke-width="2"/>')
        o.append(f'<text x="{L+pw+8}" y="{ly(series[-1])+4:.1f}" font-size="12" font-weight="600" '
                 f'fill="{colour}">{name} {series[-1]:.3f}</text>')

    # the crossing: delta@100 below direct@all
    x0, y0 = lx(100), ly(delta[0])
    o.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="8" fill="none" stroke="{t["s1"]}" stroke-width="1.5"/>')
    o.append(f'<line x1="{x0:.1f}" y1="{y0-10:.1f}" x2="{x0:.1f}" y2="{ly(direct[-1]):.1f}" '
             f'stroke="{t["ink2"]}" stroke-width="1" stroke-dasharray="3 3"/>')
    ytop = ly(direct[-1])
    o.append(f'<text x="{x0+14:.1f}" y="{ytop-16:.1f}" font-size="11" font-weight="600" fill="{t["ink"]}">'
             f'delta on 100 labels ({delta[0]:.3f})</text>')
    o.append(f'<text x="{x0+14:.1f}" y="{ytop-3:.1f}" font-size="11" fill="{t["ink2"]}">'
             f'beats direct on all 17,429 ({direct[-1]:.3f})</text>')
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------
# 2. risk-coverage
# --------------------------------------------------------------------------


def risk_coverage(t: dict) -> str:
    # SEALED test set only. The validation curve in results/risk_coverage.json is
    # ~50% more optimistic at half coverage, and shipping it under a "sealed-set
    # verified" caption is the exact defect this file was corrected for.
    d = json.loads((ROOT / "results" / "risk_coverage_sealed.json").read_text())
    if d.get("eval_on") != "sealed_test":
        raise ValueError(
            f"risk-coverage figure must plot the sealed partition, got {d.get('eval_on')!r}"
        )
    rows = sorted(d["rows"], key=lambda r: -r["coverage"])
    n_eval = rows[0]["n_kept"]

    W, H = 720, 380
    L, R, TP, B = 62, 150, 34, 52
    pw, ph = W - L - R, H - TP - B
    cx = lambda c: L + pw * (1.0 - c) / 0.5  # noqa: E731  coverage 1.0 -> 0.5
    ymin, ymax = 0.010, 0.016
    cy = lambda v: TP + ph * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']
    o.append(f'<text x="{L}" y="20" font-size="14" font-weight="600" fill="{t["ink"]}">'
             f'Abstention beats random rejection at every coverage — sealed test set</text>')
    o.append(f'<text x="{L}" y="{H-12}" font-size="11" fill="{t["muted"]}">'
             f'coverage — fraction of molecules kept '
             f'(f1, delta, random split, sealed test n={n_eval:,})</text>')

    for gv in (0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016):
        y = cy(gv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" fill="{t["muted"]}">{gv:.3f}</text>')
    o.append(f'<text x="{L-52}" y="{TP-8}" font-size="10.5" fill="{t["muted"]}">MAE (a.u.)</text>')

    for r in rows:
        x = cx(r["coverage"])
        o.append(f'<text x="{x:.1f}" y="{TP+ph+18}" font-size="10.5" text-anchor="middle" '
                 f'fill="{t["muted"]}">{int(r["coverage"]*100)}%</text>')

    band_hi = " ".join(f"{cx(r['coverage']):.1f},{cy(r['random_ci95'][1]):.1f}" for r in rows)
    band_lo = " ".join(f"{cx(r['coverage']):.1f},{cy(r['random_ci95'][0]):.1f}" for r in reversed(rows))
    o.append(f'<polygon points="{band_hi} {band_lo}" fill="{t["s2"]}" opacity="0.16"/>')

    for key, colour, name in (("random_mae", t["s2"], "random"), ("selective_mae", t["s1"], "selective")):
        pts = " ".join(f"{cx(r['coverage']):.1f},{cy(r[key]):.1f}" for r in rows)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for r in rows:
            o.append(f'<circle cx="{cx(r["coverage"]):.1f}" cy="{cy(r[key]):.1f}" r="4" fill="{colour}" '
                     f'stroke="{t["surface"]}" stroke-width="2"/>')
        last = rows[-1][key]
        o.append(f'<text x="{L+pw+8}" y="{cy(last)+4:.1f}" font-size="12" font-weight="600" '
                 f'fill="{colour}">{name} {last:.5f}</text>')
    o.append(f'<text x="{L+pw+8}" y="{cy(rows[-1]["random_mae"])+20:.1f}" font-size="10.5" '
             f'fill="{t["muted"]}">band = random 95% CI</text>')

    half = rows[-1]
    drop = 100 * (1 - half["selective_mae"] / rows[0]["selective_mae"])
    o.append(f'<text x="{L+10}" y="{TP+ph-14}" font-size="11" fill="{t["ink"]}">'
             f'−{drop:.1f}% error at half coverage, with no CC2 required</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("label-budget", label_budget), ("risk-coverage", risk_coverage)):
        for mode, t in THEME.items():
            p = FIGS / f"{name}-{mode}.svg"
            p.write_text(fn(t))
            print(f"  wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
