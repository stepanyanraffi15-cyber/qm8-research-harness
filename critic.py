"""The referee. Deterministic, and never an LLM.

The referee outlives the agent. Models get replaced; the thing that decides whether
a claim is true is what you keep, which is why it is built first and kept small.

An LLM judge is refused here for a specific measured reason, not on taste:
ChemCrow found GPT-4 could not separate confidently-wrong chemistry from correct
chemistry, and trajectory judges show *argument blindness* -- failing to notice a
wrong value passed to a tool (BabelJudge). A held-out number is not persuadable.

Eleven checks, in order (0-10). The first three catch fabrication and leakage; those
existed in the earlier design. The rest catch the failures that actually occur in
ML research, and did not:

  0 well-formedness   the submission is a claim at all
  1 split hash        the claim was measured on the partition it names, or on the
                      one its cited evidence agrees on
  2 evidence resolves every cited experiment is in the log
  3 sealed re-run     re-scored at full precision on the sealed test set
  4 noise floor       the effect exceeds a paired bootstrap CI over molecules
  5 direction         the effect runs the way the claim says it does
  6 multiplicity      Holm over the confirmatory family (audit()); the runs tried
                      in a claim's family are counted and reported beside it
  7 scope             the evidence COVERS everything the claim quantifies over
  8 control           a mechanistic claim has a control experiment
  9 rival baselines   a selective rule beats free rules and a stratified null
 10 verifiable kind   the claim is of a kind this file can measure on sealed data

Check 10 exists because `kind="mechanism"` and `kind="value"` reached SIGNED
without a single sealed-set number being computed: only the `comparison` and
`selective` branches below touch data, so every other kind fell through to the
default verdict of SIGNED. A claim that cannot be falsified must not be signable,
so those kinds are now REJECTED as `unverifiable_kind` and removed from the
agent's tool enum. The alternative -- inventing a sealed measurement for them --
was rejected because neither kind carries the structure to support one: a
mechanism claim names an intervention (`reindex_states`, `ablate_features`) that
lives in tools.py as a mutation of the target table and has no representation in
models.run, so the referee cannot reproduce it on the sealed partition at all;
and a `value` claim carries no asserted number to compare a re-measurement
against. Refusing to sign is the honest verdict; adding a measurement that does
not exist would have been the dishonest one.

Check 0 exists because this file promised a verdict rather than a traceback and
did not deliver one: roughly half of the malformed claim shapes raised, and a
single `None` in the list took down the whole audit. `adjudicate` and `audit` are
now total -- every input yields a verdict, and an unexpected exception inside a
check becomes a `referee_error` rejection rather than a lost audit. This is the
same bug class as the one fixed in agent.py: the terminal path is the one the
happy path never exercises.

Checks 1 and 7 are deliberately permissive in one direction and strict in the
other. The claim tool tells the agent to omit an axis it is not claiming about,
so omitting `split` cannot also be an `unsupported_claim` -- the partition is
read off the cited runs, and only a disagreement between them (or no cited run at
all) is fatal. And check 7 tests COVERAGE, not equality: a claim scoped to
target="E1" needs the evidence to include E1 runs, not to consist only of them.
Set equality meant that citing one additional supporting experiment could flip a
claim from signed to overscoped, which punishes exactly the behaviour a referee
should reward.

Check 5 was added because a real agent rollout produced a claim that was exactly
backwards -- "direct beats cheap for f1" when cheap wins by 0.004 a.u. -- and an
earlier version of this file SIGNED it. Verifying that a difference is real is
not the same as verifying it points the way the claim says.

Check 9 was added because this file SIGNED the project's flagship claim, and the
claim was wrong. Selective prediction on f1 cleared every bar above -- a
pre-registered hypothesis, a sealed re-run, a random-rejection null it beat
comfortably, a coin-flip control that came out flat -- and it was still an
artifact. Random rejection is the wrong null: MAE on a non-negative target falls
whenever you drop the large values, so any score correlated with |target| clears
it without predicting a single error. The classifier was ranking molecules by
brightness (Spearman +0.38 against f1_CC2), and abstaining on the bright ones is
exactly what a photophysics screen must not do. Two questions settle it, and
neither is a statistical refinement of the first:

  is the rule better than one that costs nothing?  cheap_gap, predicted_correction
  does it survive with magnitude held fixed?       the stratified null

A rule beaten by a free rule is not worth its features; a rule inside the
stratified null is measuring magnitude, not error. Either is fatal, and the
labels say which -- `free_baseline_dominates` and `magnitude_artifact` are
different diagnoses with different fixes.

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

# Kinds this file can put a held-out number behind. `mechanism` and `value` are
# recognised -- a claim file or an older trace may carry them, and _check_control
# still applies to a mechanism claim -- but they are not signable, because nothing
# below measures them. See the module docstring, check 10.
SIGNABLE_KINDS = ("comparison", "selective")
KNOWN_KINDS = ("comparison", "selective", "mechanism", "value")


def _misorder_signature():
    """The sealed selective-prediction measurements, imported lazily.

    `analysis/` is a directory of scripts rather than an installed package, so
    the import is deferred to the one method that needs it: importing critic.py
    must not depend on the repository layout, and only selective claims pay for
    it. Sharing the module is deliberate -- the alternative is two copies of the
    sealed-scoring construction, and the copy that drifts is the one nobody is
    adjudicating against.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis import misorder_signature

    return misorder_signature

# Failure labels. A pass rate tells you nothing about what to fix; a histogram of
# named failure modes does, which is why every rejection carries one.
FAILURES = [
    "unsupported_claim",
    "noise_as_signal",
    "wrong_direction",
    "overscoped",
    "no_control",
    "gave_up",
    "abstained_on_findable",
    "magnitude_artifact",
    "free_baseline_dominates",
    # A submission that is not a claim, and a claim of a kind that cannot be
    # measured on held-out data, are both referee-side failures worth counting
    # separately: the first is a bug in whatever produced the claim, the second is
    # an agent asking to be believed without offering anything falsifiable.
    "malformed_claim",
    "unverifiable_kind",
    # Reserved for a check that raised. It should stay at zero; if it does not,
    # the histogram says so instead of the audit dying.
    "referee_error",
]


def _as_dict(value) -> dict:
    """A dict, or an empty one. Used wherever a field is optional in the schema."""
    return value if isinstance(value, dict) else {}


def shape_problems(claim) -> list[str]:
    """Everything wrong with a submission's SHAPE, before any measurement.

    Separate from the checks below because a malformed claim has no measurable
    content: there is nothing to adjudicate, only something to report. Returning a
    list rather than raising is the whole point -- `adjudicate` must be total.
    """
    if not isinstance(claim, dict):
        return [f"claim is {type(claim).__name__}, not an object"]

    problems = []
    cid = claim.get("id")
    if cid is not None and not isinstance(cid, str):
        problems.append(f"id is {type(cid).__name__}, not a string")

    kind = claim.get("kind")
    if not isinstance(kind, str) or not kind:
        problems.append(f"kind is {kind!r}, not a claim kind")
    elif kind not in KNOWN_KINDS:
        problems.append(f"kind {kind!r} is not one of {list(KNOWN_KINDS)}")

    if not isinstance(claim.get("statement", ""), str):
        problems.append(f"statement is {type(claim.get('statement')).__name__}, not a string")

    scope = claim.get("scope")
    if scope is not None and not isinstance(scope, dict):
        problems.append(f"scope is {type(scope).__name__}, not an object")
    elif isinstance(scope, dict):
        for key, val in scope.items():
            ok = isinstance(val, str) or (
                isinstance(val, (list, tuple))
                and all(isinstance(x, str) for x in val)
            )
            if not ok:
                problems.append(f"scope[{key!r}] is {val!r}, not a value or list of values")

    ev = claim.get("evidence")
    if ev is None:
        problems.append("no evidence field")
    elif isinstance(ev, (str, bytes)) or not isinstance(ev, (list, tuple)):
        problems.append(f"evidence is {type(ev).__name__}, not a list of experiment ids")
    elif not all(isinstance(i, str) for i in ev):
        problems.append("evidence contains an id that is not a string")

    for key in ("config_a", "config_b"):
        cfg = claim.get(key)
        if cfg is not None and not isinstance(cfg, dict):
            problems.append(f"{key} is {type(cfg).__name__}, not an object")

    ctrl = claim.get("control")
    if ctrl is not None and not isinstance(ctrl, str):
        problems.append(f"control is {type(ctrl).__name__}, not an experiment id")

    cov = claim.get("coverage")
    if cov is not None and (isinstance(cov, bool) or not isinstance(cov, (int, float))):
        problems.append(f"coverage is {cov!r}, not a number")
    elif isinstance(cov, (int, float)) and not 0.0 < float(cov) <= 1.0:
        problems.append(f"coverage {cov!r} is not a fraction in (0, 1]")

    return problems


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def paired_bootstrap(
    err_a: np.ndarray, err_b: np.ndarray, iters: int = BOOTSTRAP_ITERS, seed: int = 0
) -> dict:
    """95% CI on mean(|err_a| - |err_b|), resampling molecules.

    The nonparametric bootstrap (Efron 1979), on a paired difference rather than
    a proportion. Pairing matters: the two configs are scored on the
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
        """Pin the claim to a partition by hash, inferring it when the claim omits it.

        tools.py tells the agent "omit an axis you are not claiming about", and an
        earlier version of this check then rejected every claim that did so as
        `unsupported_claim`. The instruction and the referee have to agree, and the
        agent's instruction is the one worth keeping: an omitted `split` is not a
        refusal to name a partition, it is a claim that does not quantify over
        that axis. Every logged run carries `config.split`, so the partition is
        recoverable from the evidence. It is fatal only when the evidence cannot
        answer -- no cited run, or cited runs that disagree, in which case there is
        no single partition to hash and no honest way to pick one.
        """
        split = _as_dict(claim.get("scope")).get("split")
        inferred = False
        if not split:
            seen = sorted({c.get("split") for c in self._configs_behind(claim) if c.get("split")})
            if not seen:
                return False, "claim names no split and no cited run carries one"
            if len(seen) > 1:
                return False, (f"claim names no split and the cited runs disagree "
                               f"about it: {seen}")
            split, inferred = seen[0], True
        if not isinstance(split, str) or split not in self.hashes:
            return False, f"unknown split {split!r}"
        h = self.hashes[split]["sealed_test"]["sha256"][:12]
        return True, (f"split {split} (inferred from the cited evidence) hash {h}"
                      if inferred else f"split {split} hash {h}")

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
        """The claim may not quantify over configurations that were never run.

        COVERAGE, not equality. This check used to compare the set of values in the
        cited runs against the claimed scope with `==`, which made additional
        evidence a liability: a claim scoped to target="E1" passed while it cited
        one E1 run and flipped to `overscoped` the moment a second, supporting E1/f1
        pair was cited alongside it. More evidence must never make a claim worse, so
        the test is whether the evidence COVERS what the claim quantifies over.

        A claim remains overscoped when it quantifies beyond its evidence --
        target="all" with one target run, or target="E2" with no E2 run behind it.
        That is the failure this check was built for, and it still fires.
        """
        scope = _as_dict(claim.get("scope"))
        configs = self._configs_behind(claim)
        if not configs:
            return False, "no logged runs behind the claim"
        universes = {
            "target": set(models.TARGETS),
            "split": set(models.SPLITS),
            "cheap_level": set(models.CHEAP_LEVELS),
        }
        for key, universe in universes.items():
            claimed = scope.get(key)
            seen = {c.get(key) for c in configs if c.get(key) is not None}
            if claimed is None:
                # Not quantified on this axis. Absence of a claim is not a claim,
                # so this is not a failure -- the axis is simply reported as the
                # values actually run when the verdict is written.
                continue
            if claimed == "all":
                missing = sorted(universe - seen)
                if missing:
                    return False, (f"claims all {key}s but never ran {missing} "
                                   f"(ran {sorted(seen)})")
            elif isinstance(claimed, (list, tuple, set)):
                missing = sorted(set(claimed) - seen)
                if missing:
                    return False, (f"claims {key} in {sorted(set(claimed))} but never ran "
                                   f"{missing} (ran {sorted(seen)})")
            elif claimed not in seen:
                return False, f"claims {key}={claimed!r} but ran {sorted(seen)}"
        return True, "the cited runs cover everything the claim quantifies over"

    def _check_kind(self, claim: dict) -> tuple[bool, str]:
        """A claim the referee cannot measure on held-out data must not be signable.

        `mechanism` and `value` used to reach SIGNED with zero rows of sealed data
        touched, because only the two branches below compute anything and every
        other kind fell through to the default verdict. See the module docstring.
        """
        kind = claim.get("kind")
        if kind in SIGNABLE_KINDS:
            return True, f"kind {kind!r} has a sealed-set measurement"
        return False, (
            f"kind {kind!r} has no sealed-set measurement in this referee, so signing it "
            f"would assert a finding no held-out number stands behind"
        )

    def _check_configs(self, claim: dict) -> tuple[bool, str]:
        """A comparison claim must actually name the two things being compared.

        An agent can and does emit `kind: "comparison"` with a null config_b. That
        is a malformed claim, and it must produce a VERDICT rather than a
        traceback -- the referee has to be at least as robust as the loop it
        grades, or one bad claim takes down the whole audit.
        """
        if claim.get("kind") != "comparison":
            return True, "not a comparison claim"
        missing = [k for k in ("config_a", "config_b")
                   if not isinstance(claim.get(k), dict) or not claim.get(k)]
        if missing:
            return False, f"comparison claim missing {', '.join(missing)}"
        return True, "both configs present"

    def _check_control(self, claim: dict) -> tuple[bool, str]:
        """Mechanistic claims need a control experiment.

        Note on reachability: `verifiable_kind` already rejects kind="mechanism"
        because the referee cannot reproduce an intervention on the sealed set, so
        this check can no longer be the DECIDING reason for such a claim -- both
        reasons fire together and the verdict is rejected either way. It is kept,
        not deleted, because it is the check that must come back if `intervene`
        ever becomes sealed-reproducible, and because the two reasons are
        different diagnoses: "unverifiable by construction" versus "verifiable but
        uncontrolled". Removing it would lose that distinction from the histogram.
        """
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
        ev = claim.get("evidence")
        ids = set(ev) if isinstance(ev, (list, tuple, set)) else set()
        return [e for e in self.log
                if e.get("experiment_id") in ids and isinstance(e.get("config"), dict)]

    def _configs_behind(self, claim: dict) -> list[dict]:
        """The config dicts of the cited runs. One accessor, so a log row missing or
        mistyping `config` can never reach a check as a KeyError."""
        return [e["config"] for e in self._runs_behind(claim)]

    def _observed_scope(self, claim: dict) -> dict:
        """What the cited runs actually cover, per axis."""
        out = {}
        for key in ("target", "split", "cheap_level"):
            seen = sorted({c.get(key) for c in self._configs_behind(claim) if c.get(key)})
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
            # Heterogeneous family (scope spans several splits or targets, or
            # names none): count every logged run whose split is among those the
            # claim actually covers. The previous expression compared each row's
            # split against None and OR-ed with `split is None`, so for a list
            # scope BOTH sides were false for every row and the count came out 0.
            # That count is REPORTED, not enforced: it is the
            # `comparisons_in_family` detail beside each verdict, so the bug hid
            # the best-of-N context from a reader of exactly the multi-split
            # claims that most need it. It never reached Holm, which `audit()`
            # runs over the confirmatory claims' p-values independently of this.
            covered = (set(split) if isinstance(split, list)
                       else {split} if isinstance(split, str) else None)
            return sum(
                1 for e in self.log
                if covered is None
                or _as_dict(e.get("config")).get("split") in covered
            )
        return sum(
            1
            for e in self.log
            if _as_dict(e.get("config")).get("target") == target
            and _as_dict(e.get("config")).get("split") == split
        )

    # -- the audit --------------------------------------------------------

    def _claim_kind(self, cid: str) -> str:
        return "confirmatory" if cid in self.prereg["claims"] else "exploratory"

    def adjudicate(self, claim) -> Verdict:
        """Adjudicate one claim. TOTAL: every input returns a Verdict, nothing raises.

        The shape check runs first and outside the try, because a malformed claim has
        no content to measure; anything that escapes a real check becomes a
        `referee_error` rejection rather than a lost audit. The promise that a bad
        claim yields a verdict rather than a traceback is only worth making if the
        code enforces it, and for nine of nineteen malformed shapes it did not.
        """
        cid = claim.get("id", "?") if isinstance(claim, dict) else "?"
        if not isinstance(cid, str):
            cid = "?"

        problems = shape_problems(claim)
        if problems:
            detail = "; ".join(problems)
            return Verdict(
                claim_id=cid, verdict=REJECTED, kind=self._claim_kind(cid),
                reasons=[f"malformed_claim: {detail}"],
                checks={"wellformed": {"passed": False, "detail": detail}},
            )

        try:
            return self._adjudicate(claim)
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            return Verdict(
                claim_id=cid, verdict=REJECTED, kind=self._claim_kind(cid),
                reasons=[f"referee_error: {detail}"],
                checks={"wellformed": {"passed": True, "detail": "shape ok"},
                        "referee_error": {"passed": False, "detail": detail}},
            )

    def _adjudicate(self, claim: dict) -> Verdict:
        cid = claim.get("id", "?")
        v = Verdict(claim_id=cid, verdict=SIGNED, kind=self._claim_kind(cid))
        v.checks["wellformed"] = {"passed": True, "detail": "shape ok"}

        for name, fn in (
            ("verifiable_kind", self._check_kind),
            ("split_hash", self._check_split_hash),
            ("evidence_resolves", self._check_evidence),
            ("scope", self._check_scope),
            ("control", self._check_control),
            ("configs_present", self._check_configs),
        ):
            ok, detail = fn(claim)
            v.checks[name] = {"passed": ok, "detail": detail}
            if not ok:
                v.verdict = REJECTED
                v.reasons.append(
                    {
                        "verifiable_kind": "unverifiable_kind",
                        "split_hash": "unsupported_claim",
                        "evidence_resolves": "unsupported_claim",
                        "scope": "overscoped",
                        "control": "no_control",
                        "configs_present": "unsupported_claim",
                    }[name]
                    + f": {detail}"
                )

        n_compared = self._count_comparisons(claim)
        v.checks["comparisons_in_family"] = {"passed": True, "detail": f"{n_compared} runs"}

        # Scope failures are narrowable rather than fatal: the evidence is sound,
        # the quantifier is not. Rewrite the claim to what the log supports.
        # An unverifiable kind is NOT narrowable -- narrowing the quantifier on a
        # claim nothing can measure still leaves nothing measured.
        if v.verdict == REJECTED and all(
            v.checks[k]["passed"]
            for k in ("verifiable_kind", "split_hash", "evidence_resolves",
                      "control", "configs_present")
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
        if claim.get("kind") == "selective":
            stat = self._sealed_selective(claim)
            v.checks["sealed_rerun"] = {"passed": True, "detail": stat["summary"]}
            v.checks["noise_floor"] = {
                "passed": stat["beats_random"],
                "detail": (f"selective {stat['selective_mae']:.6f} vs random "
                           f"{stat['random_mae']:.6f}, random 95% CI {stat['random_ci95']}"),
            }
            v.checks["direction"] = {
                "passed": stat["beats_random"],
                "detail": (f"claim says abstention helps at {stat['coverage']:.0%} coverage; "
                           f"{'it does' if stat['beats_random'] else 'it does not'}"),
            }
            v.checks["control"] = {
                "passed": stat["control_null"],
                "detail": (f"coin-flip control AUC {stat['control_auc']:.4f} "
                           f"CI {stat['control_auc_ci']}"),
            }
            # Check 9, in two halves. A rule can fail either without failing the
            # other, and they are not the same finding: losing to a free rule
            # means the features bought nothing, while sitting inside the
            # stratified null means the rule was never ranking error at all.
            v.checks["free_baselines"] = {
                "passed": stat["beats_free_baselines"],
                "detail": (f"proposed rule {stat['selective_mae']:.6f} vs best free rule "
                           f"{stat['best_free']['name']} "
                           f"{stat['best_free']['selective_mae']:.6f}"),
            }
            v.checks["magnitude_control"] = {
                "passed": stat["beats_stratified_null"],
                "detail": stat["stratified_summary"],
            }
            # The full battery rides along, the way `scope` carries its observed
            # values: a verdict that only asserts a rule lost cannot be checked
            # without re-running the referee.
            v.checks["rival_risk_rules"] = {"passed": True, "detail": stat["rival_summary"],
                                            "battery": stat["battery"]}

            if not stat["beats_random"]:
                v.verdict = REJECTED
                v.reasons.append(
                    f"noise_as_signal: selective prediction does not beat random rejection "
                    f"at matched coverage on the sealed set"
                )
            elif not stat["control_null"]:
                v.verdict = REJECTED
                v.reasons.append(
                    f"no_control: the coin-flip control also separates "
                    f"(AUC {stat['control_auc']:.3f}), so the signal is not misordering"
                )
            else:
                if not stat["beats_stratified_null"]:
                    v.verdict = REJECTED
                    b = stat["battery"]["magnitude_stratified"]
                    v.reasons.append(
                        f"magnitude_artifact: with |{stat['target']}| held fixed within "
                        f"quintiles the rule scores {b['selective_mae']:.6f} against a "
                        f"stratified null of {b['null_ci95']} -- it is ranking magnitude "
                        f"(Spearman {stat['battery']['spearman_risk_vs_target']:+.3f}), "
                        f"not error"
                    )
                if not stat["beats_free_baselines"]:
                    v.verdict = REJECTED
                    v.reasons.append(
                        f"free_baseline_dominates: {stat['best_free']['name']} costs nothing "
                        f"and scores {stat['best_free']['selective_mae']:.6f} against the "
                        f"proposed rule's {stat['selective_mae']:.6f}"
                    )
            return v

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

    def _sealed_selective(self, claim: dict) -> dict:
        """Adjudicate a selective-prediction claim on the sealed test set.

        The abstention result does not fit the two-config comparison path, so it
        gets its own verifier rather than being reported as "strong evidence" and
        left unaudited.

        Everything is refit from `train` only; the risk classifier never sees the
        sealed partition it scores. The baseline used to be random rejection at
        the same coverage, and that baseline is kept -- but it is no longer
        sufficient, and the module docstring says why. Alongside it the rule is
        now scored against the rules it has to be better than to be worth
        anything: two that cost nothing, one that spends the same training budget
        on the question that matters, the oracle ceiling, and a null that
        randomises within |target| quintiles so magnitude cannot be the answer.

        The measurement lives in analysis.misorder_signature.selective_battery,
        which the published table is generated from too -- one implementation, so
        the referee's verdict and the write-up's table cannot disagree.
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        misorder = _misorder_signature()

        cfg = claim.get("config_b") or {}
        target = cfg.get("target", "f1")
        level = cfg.get("cheap_level", "PBE0-SVP")
        split = cfg.get("split", "random")
        coverage = float(claim.get("coverage", 0.5))

        env = models._env()
        pos, X = env["positions"], env["X"]
        parts = models._split(split)
        tr, te = parts["train"], parts["sealed_test"]

        y = misorder.misordered_labels(level)
        y_tr, y_te = y[tr], y[te]

        b = misorder.selective_battery(target, level, split, coverage)

        # CONTROL: the same pipeline predicting a coin flip. If this separates,
        # the classifier is memorising the training set rather than reading
        # structure, and no amount of downstream MAE means anything.
        rng = np.random.default_rng(999)
        ctrl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=0, n_jobs=-1, verbose=-1, force_row_wise=True)
        ctrl.fit(X[pos[tr]], (rng.random(len(y_tr)) < y_tr.mean()).astype(int))
        p_ctrl = ctrl.predict_proba(X[pos[te]])[:, 1]

        ctrl_auc = float(roc_auc_score(y_te, p_ctrl)) if len(set(y_te)) > 1 else 0.5
        ab = np.random.default_rng(1)
        idx = ab.integers(0, len(p_ctrl), size=(500, len(p_ctrl)))
        caucs = np.sort([roc_auc_score(y_te[j], p_ctrl[j]) for j in idx if len(set(y_te[j])) > 1])

        selective = b["risk_rules"]["classifier"]["selective_mae"]
        rand_mae = b["random_rejection"]["selective_mae"]
        lo, hi = b["random_rejection"]["ci95"]
        strat = b["magnitude_stratified"]
        best_free = b["best_free_baseline"]

        rivals = ", ".join(
            f"{n} {r['selective_mae']:.6f}" + (" (free)" if r["free"] else "")
            for n, r in b["risk_rules"].items() if n != "classifier"
        )

        return {
            "coverage": coverage,
            "target": target,
            "full_mae": b["full_mae"],
            "selective_mae": selective,
            "random_mae": rand_mae,
            "random_ci95": [lo, hi],
            "beats_random": bool(selective < lo),
            "classifier_auc": b["classifier_auc"],
            "control_auc": ctrl_auc,
            "control_auc_ci": [round(float(caucs[12]), 4), round(float(caucs[487]), 4)],
            "control_null": bool(caucs[487] > 0.45 and caucs[12] < 0.55),
            "battery": b,
            "beats_free_baselines": b["beats_free_baselines"],
            "best_free": best_free,
            "beats_stratified_null": strat["beats_null"],
            "rival_summary": rivals,
            "stratified_summary": (
                f"within |{target}| quintiles: {strat['selective_mae']:.6f} vs "
                f"stratified null {strat['null_mae']:.6f} CI {strat['null_ci95']}; "
                f"Spearman(risk, {target}_CC2) = {b['spearman_risk_vs_target']:+.3f}"
            ),
            "summary": (f"sealed: full {b['full_mae']:.6f}, selective@{coverage:.0%} "
                        f"{selective:.6f}, random {rand_mae:.6f}; rivals: {rivals}"),
        }

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

    def audit(self, claims) -> dict:
        """Adjudicate a set of claims and apply Holm across the confirmatory ones.

        TOTAL, like `adjudicate`: one bad claim in the list must not cost the other
        seven their verdicts. `Critic().audit([good_claim, None])` used to raise
        before the good claim was ever reported.
        """
        if isinstance(claims, dict) or not isinstance(claims, (list, tuple)):
            claims = [claims]

        verdicts = []
        for c in claims:
            try:
                verdicts.append(self.adjudicate(c))
            except Exception as exc:  # noqa: BLE001
                # adjudicate is already total; this is the belt to its braces, so a
                # future check added outside the try cannot take the audit down.
                detail = f"{type(exc).__name__}: {exc}"
                verdicts.append(Verdict(claim_id="?", verdict=REJECTED, kind="exploratory",
                                        reasons=[f"referee_error: {detail}"],
                                        checks={"referee_error": {"passed": False,
                                                                  "detail": detail}}))

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
    if isinstance(claims, dict) and isinstance(claims.get("claims"), list):
        claims = claims["claims"]
    report = Critic().audit(claims)

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
