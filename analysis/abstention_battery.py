"""Re-adjudicate the two selective-prediction claims, and publish the battery.

The flagship claim of this project was that a structure-only classifier predicts
which molecules the delta model will get wrong, so that abstaining on the risky
half beats random rejection. It was pre-registered, scored on a sealed test set,
and cleared a coin-flip control. It was still an artifact, because random
rejection is the wrong null: MAE on a non-negative target falls whenever you drop
the large values, so a score that merely correlates with |target| beats random
rejection while predicting nothing about error.

This script runs the comparison that settles it, for both targets:

    every rival risk rule at 50% coverage, on the sealed test set
    the magnitude-stratified null -- random rejection WITHIN |target| quintiles
    Spearman(risk, target) -- what the classifier was actually ranking
    bright retention -- how many of the molecules a screen wants the rule keeps

and writes three files: the two per-claim verdicts the write-up cites, and
results/abstention_battery.json, so the retraction quotes a file rather than
prose. The measurements come from critic.py's own verdicts rather than from a
parallel calculation -- a table that can disagree with the referee is how the
first version of this claim survived as long as it did.

    python analysis/abstention_battery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import critic  # noqa: E402

CLAIMS = ROOT / "results" / "claims_abstention.json"
BATTERY = ROOT / "results" / "abstention_battery.json"

# Which verdict file each claim owns. Both predate this script and are cited by
# the write-up, so they keep their names and their one-verdict shape.
VERDICT_FILES = {
    "h_misorder_signature": "audit_abstention.json",
    "h_misorder_signature_E1": "audit_abstention_E1.json",
}

RULE_ORDER = ["classifier", "cheap_gap", "residual_regressor",
              "predicted_correction", "oracle"]


def load_claims(path: Path = CLAIMS) -> list[dict]:
    doc = json.loads(path.read_text())
    return doc if isinstance(doc, list) else doc["claims"]


def _print_table(b: dict) -> None:
    unit = "eV" if b["target"].startswith("E") else "a.u."
    print(f"\n  {b['target']} · sealed · {b['coverage']:.0%} coverage · "
          f"full MAE {b['full_mae']:.6f} {unit}")
    print(f"    {'risk rule':<24s} {'selective MAE':>14s}   cost")
    for name in RULE_ORDER:
        r = b["risk_rules"][name]
        cost = "free" if r["free"] else ("ceiling" if name == "oracle" else "trained")
        print(f"    {name:<24s} {r['selective_mae']:14.6f}   {cost}")
    rr = b["random_rejection"]
    print(f"    {'random rejection':<24s} {rr['selective_mae']:14.6f}   CI {rr['ci95']}")

    s = b["magnitude_stratified"]
    print(f"    magnitude-stratified: {s['selective_mae']:.6f} vs null "
          f"{s['null_mae']:.6f} CI {s['null_ci95']}  beats_null {s['beats_null']}")
    print(f"    Spearman(risk, {b['target']}_CC2) = {b['spearman_risk_vs_target']:+.4f}"
          f"   beats free baselines: {b['beats_free_baselines']}")

    br = b["bright_retention"]
    if br["applicable"]:
        print(f"    bright retention (>= {br['threshold']}): rule "
              f"{br['kept_by_rule']}/{br['total_bright']} vs random "
              f"{br['kept_by_random_mean']}/{br['total_bright']} "
              f"CI {br['kept_by_random_ci95']}")
    else:
        print(f"    bright retention: n/a — {br['note']}")


def main() -> int:
    claims = load_claims()
    print(f"re-adjudicating {len(claims)} selective claims under the extended referee")

    report = critic.Critic().audit(claims)
    out = {"claims_file": str(CLAIMS.relative_to(ROOT)), "targets": {}, "verdicts": {}}

    for v in report["verdicts"]:
        cid = v["claim_id"]
        name = VERDICT_FILES.get(cid, f"audit_{cid}.json")
        (ROOT / "results" / name).write_text(json.dumps(v, indent=2) + "\n")
        print(f"\n[{v['verdict'].upper()}] {cid} ({v['kind']})  -> results/{name}")
        for reason in v["reasons"]:
            print(f"    {reason}")

        b = v["checks"].get("rival_risk_rules", {}).get("battery")
        if b is None:                       # rejected before the sealed re-run
            continue
        _print_table(b)
        out["targets"][b["target"]] = b
        out["verdicts"][cid] = {
            "target": b["target"],
            "verdict": v["verdict"],
            "kind": v["kind"],
            "reasons": v["reasons"],
            "verdict_file": f"results/{name}",
        }

    # The external review's E1 numbers reproduce on `validation` and reverse on
    # `sealed_test`. Recorded per target so the write-up cites the measurement.
    from analysis.misorder_signature import classifier_vs_free_gap  # noqa: E402

    out["classifier_vs_free_gap"] = {}
    for target in out["targets"]:
        cmp = classifier_vs_free_gap(target)
        out["classifier_vs_free_gap"][target] = cmp
        print(f"\n  {target}: classifier vs free cheap-gap rule, both partitions")
        for part in ("validation", "sealed_test"):
            c = cmp[part]
            print(f"    {part:12s} classifier {c['classifier']:.5f}  "
                  f"cheap_gap {c['cheap_gap']:.5f}  classifier wins: {c['classifier_wins']}")
        print(f"    ordering reverses val -> sealed: "
              f"{cmp['ordering_reverses_val_to_sealed']}")

    out["summary"] = {
        "signed": report["signed"],
        "narrowed": report["narrowed"],
        "rejected": report["rejected"],
        "failure_histogram": {k: n for k, n in report["failure_histogram"].items() if n},
    }
    BATTERY.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  signed {report['signed']}  narrowed {report['narrowed']}  "
          f"rejected {report['rejected']}")
    print(f"  wrote {BATTERY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
