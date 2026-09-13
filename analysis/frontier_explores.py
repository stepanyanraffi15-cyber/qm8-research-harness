"""Score `h_frontier_explores` exactly as preregister.json states it.

Written, and run against the Qwen arm, BEFORE any frontier-model rollout existed.
The registration fixes the thresholds; it does not say what happens when a seed
never measures E1, so that rule is fixed here rather than chosen after the fact:

    primary      mean best E1 over the seeds that ran E1 (what summarise() reports)
    conservative a seed with no E1 run is scored at the free `cheap` E1 MAE,
                 0.275021 eV (PBE0-SVP, random, validation) -- it found nothing
                 better than doing nothing

If the two disagree about the verdict, BOTH are printed and (b) is reported as
not decidable as registered. Neither is picked afterwards.

Criterion (a) counts `run_experiment` rows whose config.method is "delta", read
from each rollout's own trace file, which must still match the hash in its report
row. `run_control` rows do not count: a control on a delta config presupposes the
delta run was already made.

    python analysis/frontier_explores.py results/arms_agent8_clean.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PREREG = json.loads((ROOT / "preregister.json").read_text())
CLAIM = next(c for c in PREREG["claims"] if c["id"] == "h_frontier_explores")
GRID_E1 = 0.06446
MEAN_BEST_E1_MAX = round(GRID_E1 * 1.20, 5)   # 0.07735, as registered
SEEDS_TRYING_DELTA_MIN = 6
N_SEEDS = 8
CHEAP_E1 = 0.275021


def _trace(run: dict) -> list[dict]:
    """The rollout's own trace, verified against the hash in its report row.

    A trace written in a scratch worktree is looked up by filename under
    results/arms/ if its recorded absolute path no longer exists; the hash check
    is what makes that safe.
    """
    p = Path(run["log_path"])
    if not p.exists():
        p = ROOT / "results" / "arms" / p.name
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != run["log_sha256"]:
        raise SystemExit(f"{p.name}: hash {digest[:12]} != recorded {run['log_sha256'][:12]}")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def score(arms_path: Path) -> dict:
    report = json.loads(arms_path.read_text())
    runs = [r for r in report["runs"] if r.get("arm") == "agent"]
    per_seed = []
    for r in sorted(runs, key=lambda r: r["seed"]):
        methods = [(e.get("config") or {}).get("method")
                   for e in _trace(r) if e.get("tool") == "run_experiment"]
        per_seed.append({
            "seed": r["seed"],
            "methods": methods,
            "tried_delta": "delta" in methods,
            "best_E1": (r["best"].get("E1") or {}).get("mae"),
            "stopped_because": r.get("stopped_because"),
            "tokens": r.get("tokens"),
        })

    n_delta = sum(s["tried_delta"] for s in per_seed)
    with_e1 = [s["best_E1"] for s in per_seed if s["best_E1"] is not None]
    primary = sum(with_e1) / len(with_e1) if with_e1 else None
    conservative = sum(s["best_E1"] if s["best_E1"] is not None else CHEAP_E1
                       for s in per_seed) / len(per_seed)

    a_pass = n_delta >= SEEDS_TRYING_DELTA_MIN
    b_primary = primary is not None and primary <= MEAN_BEST_E1_MAX
    b_conservative = conservative <= MEAN_BEST_E1_MAX
    b_decidable = b_primary == b_conservative
    complete = len(per_seed) == N_SEEDS

    if not complete:
        verdict = f"INCOMPLETE -- {len(per_seed)} of {N_SEEDS} seeds"
    elif not b_decidable:
        verdict = "UNDECIDABLE as registered -- (b) depends on the missing-E1 rule"
    elif a_pass and b_primary:
        verdict = "CONFIRMED"
    else:
        verdict = "FALSIFIED"

    return {
        "hypothesis": CLAIM["id"],
        "arms": str(arms_path.relative_to(ROOT) if arms_path.is_relative_to(ROOT) else arms_path),
        "n_seeds": len(per_seed),
        "a_seeds_trying_delta": {"value": n_delta, "threshold": f">={SEEDS_TRYING_DELTA_MIN} of {N_SEEDS}",
                                 "pass": a_pass},
        "b_mean_best_E1": {"primary": None if primary is None else round(primary, 6),
                           "primary_n": len(with_e1),
                           "conservative": round(conservative, 6),
                           "threshold": f"<={MEAN_BEST_E1_MAX}",
                           "pass_primary": b_primary, "pass_conservative": b_conservative},
        "verdict": verdict,
        "per_seed": per_seed,
    }


def main() -> int:
    for arg in sys.argv[1:] or [str(ROOT / "results" / "arms_agent8_clean.json")]:
        s = score(Path(arg).resolve())
        a, b = s["a_seeds_trying_delta"], s["b_mean_best_E1"]
        print(f"\n{s['arms']}  ({s['n_seeds']} seeds)")
        for p in s["per_seed"]:
            e1 = f"{p['best_E1']:.5f}" if p["best_E1"] is not None else "   none"
            print(f"  seed {p['seed']}  delta={'yes' if p['tried_delta'] else 'no '}  "
                  f"best E1 {e1}  {p['stopped_because']:<14s} {p['methods']}")
        print(f"  (a) seeds trying delta   {a['value']} of {s['n_seeds']}   need {a['threshold']}   "
              f"{'PASS' if a['pass'] else 'FAIL'}")
        prim = "n/a" if b["primary"] is None else f"{b['primary']:.5f}"
        print(f"  (b) mean best E1         {prim} (n={b['primary_n']})   "
              f"conservative {b['conservative']:.5f}   need {b['threshold']}")
        print(f"  verdict: {s['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
