"""The five-method comparison, on the sealed set, with the baseline that was missing.

The published headline was: *a structure-only model is worse than doing nothing,
and delta-learning is the only method that survives distribution shift.* Both
halves are true of the table that was run, and the table that was run was not a
fair test. `cheap` and `delta` both get the TDDFT calculation; `direct` was the
only method denied it. So the comparison measured INPUT ACCESS and was reported as
a result about the delta construction.

`direct_aug` is that control: direct learning with the four cheap TDDFT numbers
(E1/E2/f1/f2 at the chosen level) appended to the feature matrix. One TDDFT
calculation returns all four, so this costs exactly what `cheap` and `delta` cost
and nothing more. On validation at speed="fast" it closes most of the gap:

    random    cheap 0.2721  direct 0.2297  direct_aug 0.0649  delta 0.0645
    scaffold  cheap 0.2027  direct 0.3826  direct_aug 0.0809  delta 0.0676

This script runs the whole grid -- five methods x two splits x four targets -- at
speed="full" on the SEALED test set, and pairs delta against direct_aug by a
bootstrap over molecules. Pairing matters here more than anywhere: the gap between
the two is small enough that an unpaired comparison of two MAEs cannot tell a real
difference from resampling noise.

Sign convention, stated once: the reported difference is

    MAE(delta) - MAE(direct_aug)

so NEGATIVE means delta is better and a CI excluding zero on the negative side is
the surviving form of the published claim.

This file holds models.AUDIT_TOKEN, for the same reason critic.py does: it writes a
sealed-set table, and nothing reachable from tools.py can run it.

    python analysis/method_comparison.py
    python analysis/method_comparison.py --speed fast --no-validation   # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import critic  # noqa: E402
import models  # noqa: E402

METHODS = ["cheap", "direct", "direct_aug", "delta", "zero"]
SPLITS = ["random", "scaffold"]
CHEAP_LEVEL = "PBE0-TZVP"  # the better cheap baseline, per the sealed-set comparison


def cell(target: str, split: str, speed: str, cheap_level: str,
         validation: bool) -> dict:
    """Every method on one (target, split), plus the delta vs direct_aug pairing."""
    sealed, errors = {}, {}
    for m in METHODS:
        r = models.run(target=target, method=m, cheap_level=cheap_level, split=split,
                       speed=speed, eval_on="sealed_test",
                       audit_token=models.AUDIT_TOKEN, return_errors=True)
        sealed[m] = round(r.mae, 6)
        errors[m] = r.errors

    out = {
        "target": target,
        "split": split,
        "cheap_level": cheap_level,
        "speed": speed,
        "n_train": int(r.n_train),
        "n_sealed": int(r.n_eval),
        "sealed_mae": sealed,
    }

    if validation:
        val = {
            m: round(models.run(target=target, method=m, cheap_level=cheap_level,
                                split=split, speed=speed, eval_on="validation").mae, 6)
            for m in METHODS
        }
        out["validation_mae"] = val
        out["val_to_test_gap"] = {m: round(sealed[m] - val[m], 6) for m in METHODS}

    # delta - direct_aug, paired over molecules. The whole point of the new
    # baseline is this one number, so it gets the full 5,000-iteration bootstrap
    # the referee uses rather than the cheap in-loop version.
    paired = critic.paired_bootstrap(errors["delta"], errors["direct_aug"])
    paired["interpretation"] = (
        "delta better" if paired["ci95"][1] < 0 else
        "direct_aug better" if paired["ci95"][0] > 0 else
        "indistinguishable at 95%"
    )
    out["delta_minus_direct_aug"] = paired

    # The two halves of the published headline, as booleans on this cell.
    out["published_claim"] = {
        "structure_only_worse_than_doing_nothing": bool(sealed["direct"] > sealed["cheap"]),
        "structure_plus_cheap_worse_than_doing_nothing":
            bool(sealed["direct_aug"] > sealed["cheap"]),
        "delta_beats_direct_aug": bool(paired["ci95"][1] < 0),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", default="full", choices=list(models.SPEED))
    ap.add_argument("--cheap-level", default=CHEAP_LEVEL, choices=models.CHEAP_LEVELS)
    ap.add_argument("--no-validation", action="store_true",
                    help="skip the matching validation fits (halves the runtime)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "method_comparison.json")
    args = ap.parse_args()

    t0 = time.time()
    cells = []
    for split in SPLITS:
        for target in models.TARGETS:
            print(f"  {split:9s} {target:3s} ...", end="", flush=True)
            c = cell(target, split, args.speed, args.cheap_level, not args.no_validation)
            cells.append(c)
            print(f" delta {c['sealed_mae']['delta']:.5f}  "
                  f"direct_aug {c['sealed_mae']['direct_aug']:.5f}  "
                  f"({c['delta_minus_direct_aug']['interpretation']})", flush=True)

    n_delta_wins = sum(c["published_claim"]["delta_beats_direct_aug"] for c in cells)
    n_direct_worse = sum(c["published_claim"]["structure_only_worse_than_doing_nothing"]
                         for c in cells)
    n_aug_worse = sum(c["published_claim"]["structure_plus_cheap_worse_than_doing_nothing"]
                      for c in cells)

    report = {
        "what": ("five methods x two splits x four targets on the SEALED test set, with "
                 "paired bootstrap CIs for MAE(delta) - MAE(direct_aug) over molecules"),
        "sign_convention": ("delta_minus_direct_aug is MAE(delta) - MAE(direct_aug); "
                            "negative means delta is better"),
        "methods": METHODS,
        "splits": SPLITS,
        "targets": models.TARGETS,
        "cheap_level": args.cheap_level,
        "speed": args.speed,
        "eval_on": "sealed_test",
        "bootstrap_iters": critic.BOOTSTRAP_ITERS,
        # Pin the table to the partitions it was measured on, the way a verdict is.
        "split_sha256": {s: critic.split_hashes()[s]["sealed_test"]["sha256"]
                         for s in SPLITS},
        "summary": {
            "n_cells": len(cells),
            "delta_significantly_beats_direct_aug": n_delta_wins,
            "direct_worse_than_cheap": n_direct_worse,
            "direct_aug_worse_than_cheap": n_aug_worse,
        },
        "cells": cells,
        "wall_seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    head = f"{'split':9s} {'target':7s}" + "".join(f"{m:>11s}" for m in METHODS)
    print(f"\n{head}  {'delta-direct_aug 95% CI':>28s}")
    for c in cells:
        row = f"{c['split']:9s} {c['target']:7s}" + "".join(
            f"{c['sealed_mae'][m]:11.5f}" for m in METHODS)
        p = c["delta_minus_direct_aug"]
        print(f"{row}  [{p['ci95'][0]:+.5f}, {p['ci95'][1]:+.5f}] {p['interpretation']}")

    print(f"\ndelta significantly beats direct_aug in {n_delta_wins}/{len(cells)} cells")
    print(f"direct (structure only) worse than cheap in {n_direct_worse}/{len(cells)}; "
          f"direct_aug worse than cheap in {n_aug_worse}/{len(cells)}")
    print(f"wrote {args.out}  ({report['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
