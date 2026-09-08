"""The state-ordering analysis: is 'state 1' the right label?

Ramakrishnan et al. 2015 found delta-learning fails for oscillator strengths and
conjectured the cause: TDDFT and CC2 do not always agree on which excited state
is which, so the correction is asked to learn a discontinuous relabelling.

This tests that directly, and it is the most novel claim in the project, so it
gets the most adversarial treatment:

  1 swap rate      how often does a swapped assignment fit better?
  2 gap separation are those molecules the near-degenerate ones? (the mechanism)
  3 oracle gain    what would a perfect assignment be worth? (the ceiling)
  4 brightness fix what does a FREE rule -- order by oscillator strength rather
                   than by energy -- actually recover?
  5 CONTROL        the same operation on energies, which are sorted by
                   construction and must therefore show no gain

Check 5 is the one that matters. Without it, any "improvement" from reshuffling
two columns could be overfitting, and the whole result collapses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402
CHEAP = ["PBE0-SVP", "PBE0-TZVP", "CAM"]


def _cols(names, prop, level):
    return names.index(f"{prop}-{level}")


def swap_analysis(level: str) -> dict:
    """Would a swapped state assignment fit CC2 better than the given one?"""
    env = models._env()
    names, T = env["names"], env["targets"]
    pos = env["positions"]

    e1c, e2c = T[pos, _cols(names, "E1", "CC2")], T[pos, _cols(names, "E2", "CC2")]
    f1c, f2c = T[pos, _cols(names, "f1", "CC2")], T[pos, _cols(names, "f2", "CC2")]
    e1t, e2t = T[pos, _cols(names, "E1", level)], T[pos, _cols(names, "E2", level)]
    f1t, f2t = T[pos, _cols(names, "f1", level)], T[pos, _cols(names, "f2", level)]

    # Does (f1t,f2t) match (f1c,f2c) better as given, or swapped?
    as_given = np.abs(f1t - f1c) + np.abs(f2t - f2c)
    swapped = np.abs(f2t - f1c) + np.abs(f1t - f2c)
    better_swapped = swapped < as_given

    gap_cc2 = e2c - e1c
    oracle = np.minimum(as_given, swapped)

    return {
        "level": level,
        "n": int(len(pos)),
        "swap_rate": round(float(better_swapped.mean()), 4),
        "mean_cc2_gap_when_swapped": round(float(gap_cc2[better_swapped].mean()), 4),
        "mean_cc2_gap_otherwise": round(float(gap_cc2[~better_swapped].mean()), 4),
        "f_mae_as_given": round(float((as_given / 2).mean()), 6),
        "f_mae_oracle": round(float((oracle / 2).mean()), 6),
        "oracle_gain_pct": round(float(100 * (1 - oracle.mean() / as_given.mean())), 2),
    }


def energy_control(level: str) -> dict:
    """CONTROL: the same swap test on ENERGIES.

    Energies are sorted by construction (E1 <= E2 at every level), so a swap can
    only ever hurt. If this shows a gain comparable to the oscillator strengths,
    the oscillator result is an artifact of reshuffling two correlated columns
    and means nothing.
    """
    env = models._env()
    names, T = env["names"], env["targets"]
    pos = env["positions"]
    e1c, e2c = T[pos, _cols(names, "E1", "CC2")], T[pos, _cols(names, "E2", "CC2")]
    e1t, e2t = T[pos, _cols(names, "E1", level)], T[pos, _cols(names, "E2", level)]

    as_given = np.abs(e1t - e1c) + np.abs(e2t - e2c)
    swapped = np.abs(e2t - e1c) + np.abs(e1t - e2c)
    oracle = np.minimum(as_given, swapped)
    return {
        "level": level,
        "swap_rate_energies": round(float((swapped < as_given).mean()), 4),
        "oracle_gain_pct_energies": round(float(100 * (1 - oracle.mean() / as_given.mean())), 2),
    }


def brightness_rule(level: str) -> dict:
    """The FREE fix: index the two states by oscillator strength, not energy rank.

    No learning, no new data -- the same two numbers in a different order. If the
    ordering conjecture is right, this recovers part of the error for nothing.
    """
    env = models._env()
    names, T = env["names"], env["targets"]
    pos = env["positions"]

    f1c, f2c = T[pos, _cols(names, "f1", "CC2")], T[pos, _cols(names, "f2", "CC2")]
    f1t, f2t = T[pos, _cols(names, "f1", level)], T[pos, _cols(names, "f2", level)]

    # order BOTH sides by brightness (descending), then compare like with like
    bt = np.sort(np.stack([f1t, f2t]), axis=0)[::-1]
    bc = np.sort(np.stack([f1c, f2c]), axis=0)[::-1]

    baseline = (np.abs(f1t - f1c) + np.abs(f2t - f2c)) / 2
    bright = (np.abs(bt[0] - bc[0]) + np.abs(bt[1] - bc[1])) / 2
    return {
        "level": level,
        "f_mae_energy_ordered": round(float(baseline.mean()), 6),
        "f_mae_brightness_ordered": round(float(bright.mean()), 6),
        "reduction_pct": round(float(100 * (1 - bright.mean() / baseline.mean())), 2),
    }


def main() -> int:
    out = {"swap": [], "energy_control": [], "brightness": []}
    for lvl in CHEAP:
        out["swap"].append(swap_analysis(lvl))
        out["energy_control"].append(energy_control(lvl))
        out["brightness"].append(brightness_rule(lvl))

    print("=== 1-3. swap analysis (oscillator strengths) ===")
    print(f"{'level':12s} {'swap rate':>10s} {'gap|swap':>9s} {'gap|else':>9s} {'oracle gain':>12s}")
    for r in out["swap"]:
        print(f"{r['level']:12s} {r['swap_rate']:10.1%} {r['mean_cc2_gap_when_swapped']:9.3f} "
              f"{r['mean_cc2_gap_otherwise']:9.3f} {r['oracle_gain_pct']:11.1f}%")

    print("\n=== 5. CONTROL: same test on energies (must show ~no gain) ===")
    for r in out["energy_control"]:
        print(f"{r['level']:12s} swap rate {r['swap_rate_energies']:6.1%}   "
              f"oracle gain {r['oracle_gain_pct_energies']:6.2f}%")

    print("\n=== 4. the free fix: order by brightness, not energy ===")
    print(f"{'level':12s} {'energy-ord':>11s} {'bright-ord':>11s} {'reduction':>10s}")
    for r in out["brightness"]:
        print(f"{r['level']:12s} {r['f_mae_energy_ordered']:11.5f} "
              f"{r['f_mae_brightness_ordered']:11.5f} {r['reduction_pct']:9.1f}%")

    (ROOT / "results" / "state_ordering.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {ROOT / 'results' / 'state_ordering.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# The part that matters: does fixing the LABEL rescue delta-learning?
# --------------------------------------------------------------------------
#
# Note first why brightness-ordering exactly equals the oracle above. For two
# elements, sorting both sides by the same key minimises the sum of absolute
# differences (rearrangement inequality). So "order by brightness" is not a
# partial recovery of the oracle -- it IS the oracle. The ~20% is a ceiling, not
# a step towards one.
#
# That also means it is a TASK REDEFINITION, not a model improvement. It says:
# predict the bright transition's strength and the dark one's, rather than the
# lower-energy transition's and the higher one's. That is well defined at
# deployment (a model emits two numbers, no CC2 required) and arguably closer to
# what a spectroscopist wants. But it changes what the benchmark measures, and
# saying "we reduced the error 20%" without saying that would be dishonest.
#
# The real test of the 2015 conjecture is different: with a consistent labelling,
# does delta-learning finally BEAT the cheap baseline for oscillator strengths?


def delta_in_brightness_space(level: str = "PBE0-SVP", split: str = "random") -> dict:
    """Re-run cheap / direct / delta with both sides ordered by brightness."""
    env = models._env()
    names, T = env["names"], env["targets"]
    saved = T.copy()
    try:
        for lv in CHEAP + ["CC2"]:
            f1, f2 = _cols(names, "f1", lv), _cols(names, "f2", lv)
            e1, e2 = _cols(names, "E1", lv), _cols(names, "E2", lv)
            swap = T[:, f2] > T[:, f1]
            for a, b in ((f1, f2), (e1, e2)):
                tmp = T[swap, a].copy()
                T[swap, a] = T[swap, b]
                T[swap, b] = tmp
        models._CACHE["env"]["targets"] = T
        models._CACHE = {k: v for k, v in models._CACHE.items() if not (isinstance(k, tuple) and k[0] == "fit")}
        out = {}
        for method in ("cheap", "direct", "delta"):
            out[method] = round(models.run(target="f1", method=method, cheap_level=level,
                                           split=split, speed="full",
                                           eval_on="validation").mae, 6)
        return out
    finally:
        models._CACHE["env"]["targets"] = saved
        models._CACHE = {k: v for k, v in models._CACHE.items() if not (isinstance(k, tuple) and k[0] == "fit")}


def energy_space(level: str = "PBE0-SVP", split: str = "random") -> dict:
    out = {}
    for method in ("cheap", "direct", "delta"):
        out[method] = round(models.run(target="f1", method=method, cheap_level=level,
                                       split=split, speed="full",
                                       eval_on="validation").mae, 6)
    return out
