"""Score a research policy: accuracy, CI, pass^k, cost, failure histogram.

Adapted from ArmLLM 2026 Day 4 `evaluate.py`, including its bootstrap recipe
(resample with replacement 5,000 times, take the 2.5th and 97.5th percentiles --
no distributional assumption, and it behaves near 0 and 1).

    "+-15 percentage points, at 40 items, 95% confidence. A three-point
     improvement is noise."                                  Day 4, slide 37
    "Report the cost, or the number is half a result."        Day 4, slide 31

What this file adds, and what the earlier design had no answer for: a baseline
the AGENT has to beat. The earlier plan had a baseline the MODEL had to beat and
no way at all to tell an agent doing research from an agent sampling at random.
Three arms, same budget of experiments:

    random   uniform sampling from the configuration space
    grid     a fixed schedule, written before any results were seen
    agent    the LLM loop

Neither of these two baselines appears anywhere in the course literature. The
equal-budget FRAME is well evidenced (HAL: score is a property of
model x scaffold x harness x budget; AstaBench: cost-controlled Pareto), but
random-search and human-grid baselines are imported from experiment design and
are labelled reasoning, not evidence.

Two traps this file has to respect, both from Day 4 slide 39:
  * pass^k is uninformative at temperature 0 -- "there is nothing to vary, so the
    two are equal and you have learned nothing." Run the agent arm at t>0.
  * a batched server is not fully deterministic even at 0, so seed-pairing is
    leaky for the LLM arm. The model layer is genuinely deterministic, which is
    why the referee pairs over MOLECULES instead.

    uv run python evaluate.py --arms random,grid,agent --seeds 5
"""

from __future__ import annotations

import argparse
import json
import random as _random
import time
from pathlib import Path

import critic as critic_mod
import models
import tools

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

BOOTSTRAP_ITERS = 5000

# The space both null policies draw from. It is the human's contribution, and
# saying so is the point: an agent that cannot beat uniform sampling from this
# space has not added anything a for-loop could not.
SPACE = {
    "target": ["E1", "E2", "f1", "f2"],
    "method": ["cheap", "direct", "delta"],
    "cheap_level": ["PBE0-SVP", "PBE0-TZVP", "CAM"],
    "split": ["random", "scaffold", "tddft_gap"],
    "n_train": [None, 500, 2500, 10000],
}

# Written before any arm was run. This is the "competent fixed plan" an LLM has
# to beat to have earned its tokens.
GRID = [
    dict(target="E1", method="cheap", cheap_level="PBE0-SVP", split="random"),
    dict(target="E1", method="direct", cheap_level="PBE0-SVP", split="random"),
    dict(target="E1", method="delta", cheap_level="PBE0-SVP", split="random"),
    dict(target="E1", method="delta", cheap_level="PBE0-TZVP", split="random"),
    dict(target="E1", method="delta", cheap_level="CAM", split="random"),
    dict(target="E1", method="delta", cheap_level="PBE0-TZVP", split="scaffold"),
    dict(target="E2", method="delta", cheap_level="PBE0-TZVP", split="random"),
    dict(target="E1", method="delta", cheap_level="PBE0-TZVP", split="random", n_train=500),
    dict(target="E1", method="delta", cheap_level="PBE0-TZVP", split="random", n_train=2500),
    dict(target="f1", method="cheap", cheap_level="PBE0-SVP", split="random"),
    dict(target="f1", method="delta", cheap_level="PBE0-TZVP", split="random"),
    dict(target="f1", method="delta", cheap_level="PBE0-TZVP", split="tddft_gap"),
]


def bootstrap_ci(values: list[float], iters: int = BOOTSTRAP_ITERS, seed: int = 0
                 ) -> tuple[float, float]:
    """95% CI on a mean. Same recipe as the Day 4 original, on a continuous
    statistic rather than a proportion."""
    if not values:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def _score(cfg: dict, seed: int) -> float:
    return models.run(**cfg, seed=seed, speed="fast", eval_on="validation").mae


def _normalised(cfg: dict, best: dict | None, mae: float, best_mae: float) -> tuple[dict, float]:
    """Compare on a common footing: within a target, lower MAE is better, but MAE
    across targets is not comparable (eV against dimensionless). So each arm is
    scored on its own best *within each target family* and those are averaged."""
    return (cfg, mae) if best is None or mae < best_mae else (best, best_mae)


def run_random(seed: int, budget: int) -> dict:
    rng = _random.Random(seed)
    trace, best = [], {}
    for _ in range(budget):
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        try:
            mae = _score(cfg, seed=0)
        except Exception as exc:  # noqa: BLE001
            trace.append({"config": cfg, "error": str(exc)[:80]})
            continue
        fam = cfg["target"]
        if fam not in best or mae < best[fam]["mae"]:
            best[fam] = {"mae": mae, "config": cfg}
        trace.append({"config": cfg, "mae": mae, "best_so_far": best[fam]["mae"]})
    return {"arm": "random", "seed": seed, "trace": trace, "best": best, "tokens": 0}


def run_grid(seed: int, budget: int) -> dict:
    trace, best = [], {}
    for cfg in GRID[:budget]:
        mae = _score(cfg, seed=0)
        fam = cfg["target"]
        if fam not in best or mae < best[fam]["mae"]:
            best[fam] = {"mae": mae, "config": cfg}
        trace.append({"config": cfg, "mae": mae, "best_so_far": best[fam]["mae"]})
    return {"arm": "grid", "seed": seed, "trace": trace, "best": best, "tokens": 0}


def run_agent(seed: int, budget: int, temperature: float) -> dict:
    from agent import Agent
    from llm import LLM

    log = RESULTS / "arms" / f"agent_seed{seed}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)

    agent = Agent(llm=LLM(temperature=temperature), max_steps=budget + 4,
                  budget=budget, log_path=log)
    res = agent.run(
        "How few expensive CC2 labels are needed to predict excited-state properties, "
        "and where does the prediction break?"
    )
    entries = critic_mod.read_log(log)
    best: dict = {}
    trace = []
    for e in entries:
        mae = e.get("metric", {}).get("mae")
        cfg = e.get("config")
        if mae is None or cfg is None:
            continue
        fam = cfg["target"]
        if fam not in best or mae < best[fam]["mae"]:
            best[fam] = {"mae": mae, "config": cfg}
        trace.append({"config": cfg, "mae": mae, "best_so_far": best[fam]["mae"]})
    return {
        "arm": "agent", "seed": seed, "trace": trace, "best": best,
        "tokens": res.usage.total_tokens,
        "stopped_because": res.stopped_because,
        "claim": res.claim,
        "log_path": str(log),
    }


ARMS = {"random": run_random, "grid": run_grid}


# --------------------------------------------------------------------------


def summarise(runs: list[dict]) -> dict:
    """Per-arm summary: best-found error per target, cost, and reliability."""
    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)

    out = {}
    for arm, rs in by_arm.items():
        families = sorted({f for r in rs for f in r["best"]})
        per_family = {}
        for fam in families:
            vals = [r["best"][fam]["mae"] for r in rs if fam in r["best"]]
            lo, hi = bootstrap_ci(vals)
            entry = {
                "mean_best_mae": round(sum(vals) / len(vals), 6),
                "ci95": [round(lo, 6), round(hi, 6)],
                "n_seeds": len(vals),
            }
            # A fixed schedule over a deterministic model varies nothing across
            # seeds, so its bootstrap collapses to a point. Printing that as a
            # 95% CI would imply a precision that was never measured -- the
            # right statement is that this arm has no seed variance at all.
            if hi - lo == 0.0:
                entry["ci95"] = None
                entry["degenerate"] = (
                    "no variation across seeds; this arm is deterministic, so a "
                    "bootstrap over seeds measures nothing"
                )
            per_family[fam] = entry
        # pass^k: every seed found the target family at all. Reliability, not luck.
        universe = {"E1", "E2", "f1", "f2"}
        pass_k = sum(1 for r in rs if universe <= set(r["best"])) / len(rs)
        out[arm] = {
            "n_runs": len(rs),
            "per_family": per_family,
            "families_covered": len(families),
            f"pass^{len(rs)}": round(pass_k, 3),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs), 1),
            "claims": [r.get("claim") for r in rs if r.get("claim")],
            "stopped": [r.get("stopped_because") for r in rs if r.get("stopped_because")],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="random,grid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="agent arm only; pass^k is uninformative at 0")
    ap.add_argument("--out", type=Path, default=RESULTS / "arms.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip (arm, seed) pairs already present in --out")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    runs, t0 = [], time.time()

    # Checkpoint after every run. A long agent sweep gets killed -- by the OOM
    # killer on a laptop, by a timeout on a cluster -- and losing eight rollouts
    # because the report is only written at the end is a harness bug, not bad
    # luck. With --resume the same command picks up where it stopped.
    done = set()
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        runs = prev.get("runs", [])
        done = {(r["arm"], r["seed"]) for r in runs}
        print(f"  resuming: {len(done)} run(s) already complete")

    def checkpoint():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"budget_per_run": args.budget, "seeds": args.seeds,
             "temperature": args.temperature, "partial": True,
             "summary": summarise(runs), "runs": runs},
            indent=2, default=str) + "\n")

    for arm in arms:
        for seed in range(args.seeds):
            if (arm, seed) in done:
                continue
            print(f"  {arm} seed={seed} ...", flush=True)
            if arm == "agent":
                runs.append(run_agent(seed, args.budget, args.temperature))
            elif arm in ARMS:
                runs.append(ARMS[arm](seed, args.budget))
            else:
                raise SystemExit(f"unknown arm {arm!r}")
            checkpoint()

    report = {
        "budget_per_run": args.budget,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summarise(runs),
        "runs": runs,
    }
    report["partial"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"\n{'arm':8s} {'target':7s} {'best MAE':>10s} {'95% CI':>22s} {'tokens':>8s}")
    for arm, s in report["summary"].items():
        for fam, f in s["per_family"].items():
            ci = (f"[{f['ci95'][0]:.5f}, {f['ci95'][1]:.5f}]" if f["ci95"]
                  else "deterministic, no CI")
            print(f"{arm:8s} {fam:7s} {f['mean_best_mae']:10.5f} {ci:>22s} "
                  f"{s['mean_tokens']:8.0f}")
    print(f"\nwrote {args.out}  ({report['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
