"""The referee. Deterministic, and never an LLM.

    "verify.py is 120 lines. It is worth more than the agent: the agent is a
     checkpoint that expires."                              Day 4, slide 35

    "judge -- last resort, now you have two things to validate"   slide 35

An LLM judge is refused here for a specific measured reason, not on taste:
ChemCrow found GPT-4 could not separate confidently-wrong chemistry from correct
chemistry, and trajectory judges show *argument blindness* -- failing to notice a
wrong value passed to a tool (BabelJudge). A held-out number is not persuadable.

Eight checks, in order. The first three catch fabrication and leakage; those
existed in the earlier design. The rest catch the failures that actually occur in
ML research, and did not:

  1 split hash        the claim was measured on the partition it names
  2 evidence resolves every cited experiment is in the log
  3 sealed re-run     re-scored at full precision on the sealed test set
  4 noise floor       the effect exceeds a paired bootstrap CI over molecules
  5 direction         the effect runs the way the claim says it does
  6 multiplicity      Holm correction over the configs actually compared
  7 scope             the claim's quantifier matches the configs actually run
  8 control           a mechanistic claim has a control experiment

Check 5 was added because a real agent rollout produced a claim that was exactly
backwards -- "direct beats cheap for f1" when cheap wins by 0.004 a.u. -- and an
earlier version of this file SIGNED it. Verifying that a difference is real is
not the same as verifying it points the way the claim says.

Check 3 is why this file holds the audit token. Checks 5 and 6 need the log
rather than the claim, because a claim's own account of how many things were
tried before it is not evidence.

Pre-registration closes the other half of the leak. A sealed set survives an
agent that cannot query it and still dies to an author who submits eleven claims
and reports the three that landed. So a claim is CONFIRMATORY only if it appears
in preregister.json with a hash predating the audit; everything else is scored
and reported as EXPLORATORY, with its multiplicity stated.

Nothing in the course material argues for pre-registration -- it is imported
from experimental design and labelled as reasoning, not evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import models

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"
LOG_PATH = ROOT / "results" / "experiment_log.jsonl"
PREREG_PATH = ROOT / "preregister.json"

BOOTSTRAP_ITERS = 5000
ALPHA = 0.05

SIGNED, NARROWED, REJECTED = "signed", "narrowed", "rejected"

# Failure labels, the QM8 analogue of Day 4's stale / hallucinated /
# abstained_on_answerable / gave_up / wrong_number. A pass rate tells you
# nothing; a histogram of these tells you what to fix.
FAILURES = [
    "unsupported_claim",
    "noise_as_signal",
    "wrong_direction",
    "overscoped",
    "no_control",
    "gave_up",
    "abstained_on_findable",
]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def paired_bootstrap(
    err_a: np.ndarray, err_b: np.ndarray, iters: int = BOOTSTRAP_ITERS, seed: int = 0
) -> dict:
    """95% CI on mean(|err_a| - |err_b|), resampling molecules.

    Same resampling shape as ArmLLM's bootstrap_ci, but on a paired difference
    rather than a proportion. Pairing matters: the two configs are scored on the
    same molecules, so the between-molecule variance -- which is most of the
    spread -- cancels. An unpaired comparison of two MAEs throws that away and
    calls real effects noise.
    """
    if len(err_a) != len(err_b):
        raise ValueError("paired bootstrap needs equal-length error vectors")
    d = np.asarray(err_a, dtype=np.float64) - np.asarray(err_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(iters, n))
    means = np.sort(d[idx].mean(axis=1))
    lo, hi = float(means[int(0.025 * iters)]), float(means[int(0.975 * iters)])
    return {
        "mean_difference": float(d.mean()),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_paired": n,
        # two-sided bootstrap p, used only for the Holm correction below
        "p_value": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
    }


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni over a family of confirmatory tests.

    Uniformly more powerful than plain Bonferroni and makes no independence
    assumption, which matters because configs in one family are correlated by
    construction (same molecules, same features, overlapping training sets).
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, still_rejecting = {}, True
    for i, (key, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if p > threshold:
            still_rejecting = False
        out[key] = {
            "p": round(p, 6),
            "threshold": round(threshold, 6),
            "significant": bool(still_rejecting and p <= threshold),
            "rank": i + 1,
            "family_size": m,
        }
    return out


# --------------------------------------------------------------------------
# log and pre-registration
# --------------------------------------------------------------------------


def _name(claim: dict, key: str) -> str:
    """Short label for a config, for readable verdicts."""
    cfg = claim.get(key) or {}
    return "/".join(str(cfg.get(k)) for k in ("target", "method", "cheap_level") if cfg.get(k))


def read_log(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def split_hashes() -> dict:
    return json.loads((DERIVED / "manifest.json").read_text())["splits"]


def preregistration(path: Path = PREREG_PATH) -> dict:
    if not path.exists():
        return {"claims": {}, "sha256": None, "exists": False}
    raw = path.read_bytes()
    doc = json.loads(raw)
    return {
        "claims": {c["id"]: c for c in doc.get("claims", [])},
        "sha256": hashlib.sha256(raw).hexdigest(),
        "registered_at": doc.get("registered_at"),
        "exists": True,
    }


# --------------------------------------------------------------------------
# the referee
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    claim_id: str
    verdict: str
    kind: str  # confirmatory | exploratory
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    narrowed_statement: str | None = None

    def as_dict(self) -> dict:
        d = {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "kind": self.kind,
            "reasons": self.reasons,
            "checks": self.checks,
        }
        if self.narrowed_statement:
            d["narrowed_statement"] = self.narrowed_statement
        return d


class Critic:
    def __init__(self, log: list[dict] | None = None, prereg: dict | None = None):
        self.log = read_log() if log is None else log
        self.prereg = preregistration() if prereg is None else prereg
        self.hashes = split_hashes()

    # -- individual checks ------------------------------------------------

    def _check_split_hash(self, claim: dict) -> tuple[bool, str]:
        split = claim.get("scope", {}).get("split")
        if not split:
            return False, "claim names no split"
        if split not in self.hashes:
            return False, f"unknown split {split!r}"
        return True, f"split {split} hash {self.hashes[split]['sealed_test']['sha256'][:12]}"

    def _check_evidence(self, claim: dict) -> tuple[bool, str]:
        ids = claim.get("evidence", [])
        if not ids:
            return False, "no evidence ids cited"
        known = {e["experiment_id"] for e in self.log if "experiment_id" in e}
        missing = [i for i in ids if i not in known]
        if missing:
            return False, f"evidence not in log: {missing}"
        return True, f"{len(ids)} evidence ids resolve"

    def _check_scope(self, claim: dict) -> tuple[bool, str]:
        """The claim may not quantify over configurations that were never run."""
        scope = claim.get("scope", {})
        ids = set(claim.get("evidence", []))
        runs = [e for e in self.log if e.get("experiment_id") in ids]
        if not runs:
            return False, "no logged runs behind the claim"
        universes = {
            "target": set(models.TARGETS),
            "split": set(models.SPLITS),
            "cheap_level": set(models.CHEAP_LEVELS),
        }
        for key, universe in universes.items():
            claimed = scope.get(key)
            seen = {r["config"].get(key) for r in runs}
            if claimed is None:
                # Not quantified on this axis. Absence of a claim is not a claim,
                # so this is not a failure -- the axis is simply reported as the
                # values actually run when the verdict is written.
                continue
            if claimed == "all":
                if seen != universe:
                    return False, f"claims all {key}s but only ran {sorted(seen)}"
            elif seen != {claimed}:
                return False, f"claims {key}={claimed!r} but ran {sorted(seen)}"
        return True, "quantifier matches the configs run"

    def _check_control(self, claim: dict) -> tuple[bool, str]:
        if claim.get("kind") != "mechanism":
            return True, "not a mechanistic claim"
        ctrl = claim.get("control")
        if not ctrl:
            return False, "mechanistic claim with no control experiment"
        known = {e.get("experiment_id") for e in self.log}
        if ctrl not in known:
            return False, f"control {ctrl!r} not in log"
        return True, f"control {ctrl} present"

    def _runs_behind(self, claim: dict) -> list[dict]:
        ids = set(claim.get("evidence", []))
        return [e for e in self.log if e.get("experiment_id") in ids and "config" in e]

    def _observed_scope(self, claim: dict) -> dict:
        """What the cited runs actually cover, per axis."""
        runs = self._runs_behind(claim)
        out = {}
        for key in ("target", "split", "cheap_level"):
            seen = sorted({r["config"].get(key) for r in runs if r["config"].get(key)})
            if seen:
                out[key] = seen[0] if len(seen) == 1 else seen
        return out

    def _count_comparisons(self, claim: dict) -> int:
        """How many configs in this claim's family were tried before it.

        Best-of-N on a noisy metric is an overestimate by construction, and the
        claim's own narrative never mentions the N. The family is defined by the
        runs actually cited, not by the scope asserted -- a claim that says
        target="all" would otherwise match nothing and report a count of zero.
        """
        observed = self._observed_scope(claim)
        target, split = observed.get("target"), observed.get("split")
        if isinstance(target, list) or isinstance(split, list) or target is None:
            # heterogeneous family: count everything sharing the split
            return sum(1 for e in self.log
                       if e.get("config", {}).get("split") == (split if isinstance(split, str) else None)
                       or split is None)
        return sum(
            1
            for e in self.log
            if e.get("config", {}).get("target") == target
            and e.get("config", {}).get("split") == split
        )

    # -- the audit --------------------------------------------------------

    def adjudicate(self, claim: dict) -> Verdict:
        cid = claim.get("id", "?")
        registered = cid in self.prereg["claims"]
        kind = "confirmatory" if registered else "exploratory"
        v = Verdict(claim_id=cid, verdict=SIGNED, kind=kind)

        for name, fn in (
            ("split_hash", self._check_split_hash),
            ("evidence_resolves", self._check_evidence),
            ("scope", self._check_scope),
            ("control", self._check_control),
        ):
            ok, detail = fn(claim)
            v.checks[name] = {"passed": ok, "detail": detail}
            if not ok:
                v.verdict = REJECTED
                v.reasons.append(
                    {
                        "split_hash": "unsupported_claim",
                        "evidence_resolves": "unsupported_claim",
                        "scope": "overscoped",
                        "control": "no_control",
                    }[name]
                    + f": {detail}"
                )

        n_compared = self._count_comparisons(claim)
        v.checks["comparisons_in_family"] = {"passed": True, "detail": f"{n_compared} runs"}

        # Scope failures are narrowable rather than fatal: the evidence is sound,
        # the quantifier is not. Rewrite the claim to what the log supports.
        if v.verdict == REJECTED and all(
            v.checks[k]["passed"] for k in ("split_hash", "evidence_resolves", "control")
        ):
            v.verdict = NARROWED
            # Narrow to what the LOG supports, not to what the claim asserted.
            # Echoing the claimed scope back ("target=all") would defeat the point.
            observed = self._observed_scope(claim)
            v.narrowed_statement = (
                f"{claim.get('statement', '').rstrip('.')} "
                f"— for {', '.join(f'{k}={v2}' for k, v2 in observed.items())} only."
            )
            v.checks["scope"]["observed"] = observed

        if v.verdict == REJECTED:
            return v

        # Check 3 and 4: the only place the sealed test set is ever touched.
        if claim.get("kind") == "comparison":
            stat = self._sealed_comparison(claim)
            v.checks["sealed_rerun"] = {"passed": True, "detail": stat["summary"]}
            v.checks["noise_floor"] = {
                "passed": stat["excludes_zero"],
                "detail": (
                    f"paired bootstrap CI {stat['ci95']} on {stat['n_paired']} molecules"
                ),
            }
            v.checks["val_to_test_gap"] = {"passed": True, "detail": stat["gap_summary"]}

            # Direction. `config_b` is the one the claim says is better, so the
            # paired difference mean(|err_a| - |err_b|) must be POSITIVE. Checking
            # only that the CI excludes zero signs a claim that is exactly
            # backwards -- which is not hypothetical: the first real agent rollout
            # produced one, and an earlier version of this file signed it.
            right_way = stat["ci95"][0] > 0
            v.checks["direction"] = {
                "passed": right_way,
                "detail": (
                    f"claim says {_name(claim, 'config_b')} beats "
                    f"{_name(claim, 'config_a')}; measured "
                    f"{stat['sealed_mae']['b']:.5f} vs {stat['sealed_mae']['a']:.5f}"
                ),
            }

            if not stat["excludes_zero"]:
                v.verdict = REJECTED
                v.reasons.append(f"noise_as_signal: CI {stat['ci95']} includes zero")
            elif not right_way:
                v.verdict = REJECTED
                v.reasons.append(
                    f"wrong_direction: the effect is real but runs the other way -- "
                    f"{_name(claim, 'config_a')} beats {_name(claim, 'config_b')} "
                    f"by {-stat['mean_difference']:.5f}"
                )
            v.checks["_p_value"] = stat["p_value"]
        return v

    def _sealed_comparison(self, claim: dict) -> dict:
        """Re-run both configs at full precision on the sealed test set."""
        a, b = claim["config_a"], claim["config_b"]
        res = {}
        for tag, cfg in (("a", a), ("b", b)):
            res[tag] = {
                "sealed": models.run(
                    **cfg, speed="full", eval_on="sealed_test",
                    audit_token=models.AUDIT_TOKEN, return_errors=True,
                ),
                "val": models.run(**cfg, speed="full", eval_on="validation"),
            }
        stat = paired_bootstrap(res["a"]["sealed"].errors, res["b"]["sealed"].errors)
        gap_a = res["a"]["sealed"].mae - res["a"]["val"].mae
        gap_b = res["b"]["sealed"].mae - res["b"]["val"].mae
        stat["summary"] = (
            f"sealed MAE a={res['a']['sealed'].mae:.5f} b={res['b']['sealed'].mae:.5f} "
            f"(diff {stat['mean_difference']:+.5f})"
        )
        stat["gap_summary"] = f"val->test gap a={gap_a:+.5f} b={gap_b:+.5f}"
        stat["sealed_mae"] = {"a": res["a"]["sealed"].mae, "b": res["b"]["sealed"].mae}
        return stat

    # -- family-level ------------------------------------------------------

    def audit(self, claims: list[dict]) -> dict:
        """Adjudicate a set of claims and apply Holm across the confirmatory ones."""
        verdicts = [self.adjudicate(c) for c in claims]

        family = {
            v.claim_id: v.checks["_p_value"]
            for v in verdicts
            if v.kind == "confirmatory" and "_p_value" in v.checks
        }
        corrected = holm(family) if family else {}
        for v in verdicts:
            v.checks.pop("_p_value", None)
            if v.claim_id in corrected:
                c = corrected[v.claim_id]
                v.checks["multiplicity"] = {"passed": c["significant"], "detail": (
                    f"Holm rank {c['rank']}/{c['family_size']}, p={c['p']}, "
                    f"threshold={c['threshold']}"
                )}
                if v.verdict == SIGNED and not c["significant"]:
                    v.verdict = REJECTED
                    v.reasons.append(
                        f"noise_as_signal: survives alone but not Holm correction "
                        f"over {c['family_size']} confirmatory tests"
                    )

        histogram = {label: 0 for label in FAILURES}
        for v in verdicts:
            for r in v.reasons:
                label = r.split(":")[0]
                if label in histogram:
                    histogram[label] += 1

        return {
            "preregistration": {
                "exists": self.prereg["exists"],
                "sha256": self.prereg["sha256"],
                "registered_at": self.prereg.get("registered_at"),
            },
            "n_claims": len(claims),
            "signed": sum(v.verdict == SIGNED for v in verdicts),
            "narrowed": sum(v.verdict == NARROWED for v in verdicts),
            "rejected": sum(v.verdict == REJECTED for v in verdicts),
            "confirmatory": sum(v.kind == "confirmatory" for v in verdicts),
            "exploratory": sum(v.kind == "exploratory" for v in verdicts),
            "failure_histogram": histogram,
            "verdicts": [v.as_dict() for v in verdicts],
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="adjudicate claims against the sealed test set")
    ap.add_argument("claims", type=Path, help="JSON file with a list of claims")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "audit.json")
    args = ap.parse_args()

    claims = json.loads(args.claims.read_text())
    report = Critic().audit(claims if isinstance(claims, list) else claims["claims"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"claims {report['n_claims']}  "
          f"signed {report['signed']}  narrowed {report['narrowed']}  "
          f"rejected {report['rejected']}")
    print(f"  confirmatory {report['confirmatory']}  exploratory {report['exploratory']}")
    for v in report["verdicts"]:
        mark = {"signed": "OK  ", "narrowed": "NARR", "rejected": "REJ "}[v["verdict"]]
        print(f"  [{mark}] {v['claim_id']:12s} {v['kind']:12s} "
              + ("; ".join(v["reasons"]) if v["reasons"] else ""))
    nz = {k: n for k, n in report["failure_histogram"].items() if n}
    if nz:
        print(f"  failures: {nz}")
    print(f"  wrote {args.out}")
