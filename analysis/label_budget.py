"""How few expensive CC2 labels do you actually need?

This is the question the project is named after, and the one a company deciding
whether to run a CC2 campaign actually wants answered.

Sweep the number of CC2 training labels from 100 to the whole training set, for
direct and delta learning, on two splits. Three things come out of it:

  1 the crossing point   -- how many labels delta needs to beat direct trained on
                            everything, which is the label-efficiency headline
  2 the plateau          -- where more expensive compute stops buying accuracy
  3 h_budget_plateau     -- does the plateau MOVE under distribution shift? If
                            delta-learning fits a smooth local correction, the
                            scaffold split should need materially more labels.
                            If the curves superimpose, the correction is more
                            global than assumed -- a claim about the physics.

Every point is three seeds with a bootstrap interval, because the earlier version
of this curve was a single run per point and the headline was read off it.

    python analysis/label_budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402

SIZES = [100, 250, 500, 1000, 2500, 5000, 10000, None]  # None = all of train
SEEDS = [0, 1, 2]
SPLITS = ["random", "scaffold"]
TARGET = "E1"
CHEAP_LEVEL = "PBE0-TZVP"  # the better baseline, per the sealed-set comparison


def bootstrap_mean(vals: list[float], iters: int = 5000, seed: int = 0) -> list[float]:
    if len(vals) < 2:
        return [round(vals[0], 6), round(vals[0], 6)]
    rng = np.random.default_rng(seed)
    a = np.asarray(vals)
    means = np.sort(a[rng.integers(0, len(a), size=(iters, len(a)))].mean(axis=1))
    return [round(float(means[int(0.025 * iters)]), 6), round(float(means[int(0.975 * iters)]), 6)]


def sweep(split: str) -> dict:
    cheap = models.run(target=TARGET, method="cheap", cheap_level=CHEAP_LEVEL,
                       split=split, speed="fast", eval_on="validation").mae
    rows = []
    for n in SIZES:
        row = {"n_train": n, "label": "all" if n is None else str(n)}
        for method in ("direct", "delta"):
            vals = []
            for s in SEEDS:
                r = models.run(target=TARGET, method=method, cheap_level=CHEAP_LEVEL,
                               split=split, n_train=n, seed=s, speed="fast",
                               eval_on="validation")
                vals.append(r.mae)
                row["n_actual"] = r.n_train
            row[method] = round(float(np.mean(vals)), 6)
            row[f"{method}_ci"] = bootstrap_mean(vals)
        rows.append(row)
        print(f"  {split:9s} n={row['label']:>5s}  direct {row['direct']:.5f}  "
              f"delta {row['delta']:.5f}", flush=True)
    return {"split": split, "cheap": round(cheap, 6), "rows": rows}


def analyse(curve: dict) -> dict:
    rows, cheap = curve["rows"], curve["cheap"]
    full_direct = rows[-1]["direct"]
    full_delta = rows[-1]["delta"]

    # crossing point: fewest labels at which delta beats direct-trained-on-everything
    crossing = next((r for r in rows if r["delta"] < full_direct), None)
    # and the fewest at which delta beats doing nothing
    beats_cheap = next((r for r in rows if r["delta"] < cheap), None)

    # plateau: first size within 10% of the fully-trained delta error
    plateau = next((r for r in rows if r["delta"] <= full_delta * 1.10), None)

    out = {
        "cheap_baseline": cheap,
        "direct_full": full_direct,
        "delta_full": full_delta,
        "delta_beats_full_direct_at": crossing["label"] if crossing else None,
        "delta_beats_cheap_at": beats_cheap["label"] if beats_cheap else None,
        "plateau_within_10pct_at": plateau["label"] if plateau else None,
    }
    if crossing and crossing["n_train"]:
        out["label_efficiency_factor"] = round(rows[-1]["n_actual"] / crossing["n_train"], 1)
    return out


def main() -> int:
    print(f"label-budget sweep: {TARGET}, cheap={CHEAP_LEVEL}, {len(SEEDS)} seeds\n")
    curves = {}
    for split in SPLITS:
        curves[split] = sweep(split)
        curves[split]["analysis"] = analyse(curves[split])
        print()

    print(f"{'':10s} {'cheap':>9s} {'direct(all)':>12s} {'delta(all)':>11s} "
          f"{'delta<direct(all)':>18s} {'plateau':>9s}")
    for split, c in curves.items():
        a = c["analysis"]
        print(f"{split:10s} {a['cheap_baseline']:9.5f} {a['direct_full']:12.5f} "
              f"{a['delta_full']:11.5f} {str(a['delta_beats_full_direct_at']):>18s} "
              f"{str(a['plateau_within_10pct_at']):>9s}")

    # h_budget_plateau: does the plateau move under shift?
    pr, ps = (curves["random"]["analysis"]["plateau_within_10pct_at"],
              curves["scaffold"]["analysis"]["plateau_within_10pct_at"])
    print(f"\nh_budget_plateau: random plateaus at {pr}, scaffold at {ps} -> "
          f"{'MOVES' if pr != ps else 'curves superimpose'}")

    (ROOT / "results" / "label_budget.json").write_text(json.dumps(curves, indent=2) + "\n")
    print(f"wrote {ROOT / 'results' / 'label_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
