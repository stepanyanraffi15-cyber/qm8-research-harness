"""Adjudicate the claims an agent sweep produced, one claim per rollout.

This was an ad-hoc command before, which is why results/audit_agent8.json was
written from results/arms_agent8.json -- the report whose first three rows are
byte-identical copies of an earlier sweep's -- and never re-derived from the clean
one. A number quoted in a write-up needs a command behind it, so here it is.

Two things this has to get right, and the ad-hoc version did not:

  * Evidence ids are RUN-scoped (`e_<prefix>_<n>`), so each claim is adjudicated
    against ITS OWN trace file, not the shared experiment log. Auditing a claim
    against another rollout's log is how a claim gets signed on a config it never
    ran.
  * A rollout that ended without a claim is not absent from the denominator. It
    is a `gave_up` rejection, because "produced nothing" is a research outcome
    and eight rollouts must report eight verdicts.

    python analysis/audit_agent_claims.py --arms results/arms_agent8_clean.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import critic  # noqa: E402


def audit_arms(arms_path: Path) -> list[dict]:
    report = json.loads(arms_path.read_text())
    rows = []
    for run in report.get("runs", []):
        if run.get("arm") != "agent":
            continue
        seed = run.get("seed")
        cid = f"agent_seed{seed}"
        claim = run.get("claim")
        if not claim:
            rows.append({
                "seed": seed, "claim_id": cid, "verdict": critic.REJECTED,
                "kind": "exploratory",
                "reasons": [f"gave_up: rollout produced no claim "
                            f"({run.get('stopped_because', 'unknown')})"],
                "checks": {},
            })
            continue
        log_path = run.get("log_path")
        log = critic.read_log(Path(log_path)) if log_path else []
        c = {**claim, "id": cid}
        v = critic.Critic(log=log).adjudicate(c)
        rows.append({"seed": seed, "trace": Path(log_path).name if log_path else None,
                     **v.as_dict()})
    return sorted(rows, key=lambda r: (r["seed"] is None, r["seed"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", type=Path, default=ROOT / "results" / "arms_agent8_clean.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = audit_arms(args.arms)
    counts = Counter(r["verdict"] for r in rows)
    histogram = Counter()
    for r in rows:
        for reason in r["reasons"]:
            label = reason.split(":")[0]
            if label in critic.FAILURES:
                histogram[label] += 1

    out = args.out or ROOT / "results" / f"audit_{args.arms.stem}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"{len(rows)} claims from {args.arms.name}")
    for r in rows:
        mark = {"signed": "OK  ", "narrowed": "NARR", "rejected": "REJ "}[r["verdict"]]
        print(f"  [{mark}] seed {r['seed']}  " + ("; ".join(r["reasons"])[:110] or "-"))
    print(f"\nVERDICTS           signed {counts['signed']} · narrowed {counts['narrowed']} "
          f"· rejected {counts['rejected']}")
    print("FAILURE HISTOGRAM  "
          + (" · ".join(f"{k} {n}" for k, n in histogram.most_common()) or "none"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
