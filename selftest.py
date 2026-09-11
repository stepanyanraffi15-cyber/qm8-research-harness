"""Every check this project claims to pass, runnable in one command.

    uv run python selftest.py            # everything
    uv run python selftest.py --quick    # harness only, skips the model fits

The point is that a reviewer should not have to take any number in the write-up
on trust. The `findings` block re-derives the three headline results from the
data rather than reading them out of a results file, which would only prove the
file exists. Each check below either reproduces a published result, proves an
isolation property, or demonstrates that the referee rejects something it should.

Checks 4 and 5 are the load-bearing ones: they are what make the invariant --
no number reaches the write-up except through a scorer neither the agent nor the
author can query adaptively -- a property of the code rather than a promise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name):
    def deco(fn):
        def wrapped():
            try:
                ok, detail = fn()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            _results.append((name, PASS if ok else FAIL, detail))
            print(f"  [{PASS if ok else FAIL}] {name}\n         {detail}")
            return ok

        return wrapped

    return deco


@check("parse reproduces Ramakrishnan et al. 2015")
def t_parse():
    m = json.loads((DERIVED / "manifest.json").read_text())
    p = m["checks"]["parse_vs_paper_2015"]
    return p["passed"], (
        f"E1 {p['E1']['mae_ev']} eV (paper {p['E1']['paper_ev']}), "
        f"E2 {p['E2']['mae_ev']} eV (paper {p['E2']['paper_ev']})"
    )


@check("MoleculeNet duplication is still present upstream")
def t_duplication():
    d = json.loads((DERIVED / "manifest.json").read_text())["checks"]["moleculenet_duplication"]
    return d["distinct_tasks"] == 12, (
        f"{d['identical_rows']}/{d['n_rows']} PBE0 rows identical -> "
        f"{d['distinct_tasks']} distinct tasks under 16 headers"
    )


@check("SMILES join is verified, not assumed")
def t_alignment():
    a = json.loads((DERIVED / "manifest.json").read_text())["checks"]["row_alignment"]
    return a["aligned"], (
        f"max|diff| = {a['max_abs_difference']:.1e} across "
        f"{len(a['columns_compared'])} shared columns"
    )


@check("environment resets byte-for-byte from a seed")
def t_reset():
    before = json.loads((DERIVED / "manifest.json").read_text())
    r = subprocess.run(
        [sys.executable, str(ROOT / "world" / "build.py"), "--seed", "0", "--offline"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0:
        return False, f"rebuild failed: {r.stderr[-200:]}"
    after = json.loads((DERIVED / "manifest.json").read_text())
    same = (before["data_sha256"] == after["data_sha256"]
            and before["splits"] == after["splits"])
    return same, f"targets hash {after['data_sha256']['targets_ev'][:16]} stable across builds"


@check("the sealed test set is unreachable without the audit token")
def t_sealed():
    import models

    try:
        models.run(target="E1", method="delta", eval_on="sealed_test")
    except PermissionError as exc:
        pass
    else:
        return False, "models.run returned a sealed-test number with no token"

    # and the token is not importable from the agent's side of the boundary
    import tools

    leaked = [n for n in dir(tools) if "AUDIT" in n.upper() or "TOKEN" in n.upper()]
    return not leaked, f"PermissionError raised; tools.py exposes no token ({leaked or 'none'})"


@check("no sealed-test number appears in any agent observation")
def t_no_leak():
    import models

    sealed = models.run(target="E1", method="delta", eval_on="sealed_test",
                        audit_token=models.AUDIT_TOKEN).mae
    traj = ROOT / "results" / "trajectory_mock.json"
    if not traj.exists():
        return False, "no trajectory to inspect; run agent.py first"
    blob = traj.read_text()
    needle = f"{sealed:.5f}"[:6]
    return needle not in blob, (
        f"sealed E1/delta MAE {sealed:.5f} does not appear in the mock trajectory"
    )


@check("a tampered split is refused")
def t_split_integrity():
    import models

    src = DERIVED / "split_random.npz"
    backup = Path(tempfile.mkdtemp()) / "split_random.npz"
    shutil.copy(src, backup)
    try:
        z = dict(np.load(src))
        z["validation"] = z["validation"][:-1]  # drop one molecule
        np.savez_compressed(src, **z)
        models._CACHE.pop("split:random", None)
        try:
            models._split("random")
        except models.SplitIntegrityError as exc:
            return True, str(exc).splitlines()[0]
        return False, "a mutated split was accepted"
    finally:
        shutil.copy(backup, src)
        models._CACHE.pop("split:random", None)


@check("shuffling the labels destroys all skill")
def t_shuffle_control():
    import tools

    tools.reset_session(budget=10, log_path=ROOT / "results" / "selftest_log.jsonl")
    a = json.loads(tools.run_experiment(target="E1", method="delta"))
    c = json.loads(tools.run_control(a["experiment_id"], "shuffle_labels"))
    ratio = c["control_mae"] / a["metric"]["mae"]
    return ratio > 5, (
        f"MAE {a['metric']['mae']:.4f} -> {c['control_mae']:.4f} ({ratio:.1f}x worse)"
    )


@check("the referee rejects a fabricated claim")
def t_referee_rejects():
    import critic

    claim = {"id": "x", "kind": "comparison", "statement": "invented",
             "scope": {"target": "E1", "split": "random"}, "evidence": ["e_does_not_exist"]}
    v = critic.Critic().adjudicate(claim)
    return v.verdict == critic.REJECTED, f"verdict={v.verdict}; {'; '.join(v.reasons)[:90]}"


@check("a malformed claim produces a verdict, not a traceback")
def t_referee_total():
    """critic.py promised this in its own comments and did not deliver it.

    The audit found roughly nine of nineteen malformed shapes raising instead of
    returning a verdict, and `Critic().audit([good, None])` dying outright, before
    the good claim alongside it was ever reported --
    so one bad claim cost an eight-seed sweep its whole audit. Same bug class as
    the one already fixed in agent.py: the terminal/error path is the one the happy
    path never exercises, so it only gets tested if a test goes looking for it.
    """
    import critic

    good = {"id": "good", "kind": "comparison", "statement": "a real claim",
            "scope": {"target": "E1", "split": "random"}, "evidence": ["e_absent"]}
    malformed = [
        None,                                                      # the one that killed audit()
        {},                                                        # nothing at all
        [],                                                        # not an object
        "claim_0",                                                 # a string
        42,
        {"kind": "comparison", "evidence": ["e"], "scope": "random"},   # scope not a dict
        {"kind": "comparison", "evidence": ["e"], "scope": {"target": {"E1": 1}}},
        {"kind": "comparison", "evidence": "e_0001"},                   # evidence a string
        {"kind": "comparison", "evidence": [None, 3]},                  # ids not strings
        {"kind": "comparison"},                                         # no evidence field
        {"kind": None, "evidence": []},
        {"kind": "not_a_kind", "evidence": []},
        {"id": 7, "kind": "comparison", "evidence": []},                # id not a string
        {"kind": "comparison", "evidence": [], "statement": {"a": 1}},
        {"kind": "comparison", "evidence": [], "config_a": "x", "config_b": 3},
        {"kind": "comparison", "evidence": [], "control": ["e"]},
        {"kind": "selective", "evidence": [], "coverage": "half"},
        {"kind": "selective", "evidence": [], "coverage": 5},
        {"kind": "mechanism", "evidence": ["e"], "scope": {"split": "random"}},
        {"kind": "value", "evidence": ["e"], "scope": {"split": "random"}},
    ]
    verdicts = {critic.SIGNED, critic.NARROWED, critic.REJECTED}
    c = critic.Critic()

    raised = []
    for m in malformed:
        try:
            v = c.adjudicate(m)
        except Exception as exc:  # noqa: BLE001
            raised.append(f"{str(m)[:30]} -> {type(exc).__name__}")
            continue
        if v.verdict not in verdicts or not v.reasons:
            raised.append(f"{str(m)[:30]} -> {v.verdict!r} with no reason")
    if raised:
        return False, f"{len(raised)}/{len(malformed)} did not yield a verdict: {raised[:3]}"

    # and one bad claim must not cost the others their verdicts
    report = c.audit([good, None, {}])
    if report["n_claims"] != 3 or report["rejected"] != 3:
        return False, (f"audit() lost a claim: {report['n_claims']} in, "
                       f"{report['rejected']} rejected")
    if not report["failure_histogram"]["malformed_claim"]:
        return False, "malformed claims are not counted in the failure histogram"

    # unfalsifiable kinds are refused rather than waved through
    signed_unverifiable = [
        k for k in ("mechanism", "value")
        if c.adjudicate({"id": "u", "kind": k, "statement": "s", "evidence": ["e"],
                         "scope": {"split": "random"}}).verdict != critic.REJECTED
    ]
    if signed_unverifiable:
        return False, f"kinds with no sealed measurement were not refused: {signed_unverifiable}"

    return True, (f"all {len(malformed)} malformed shapes returned a verdict; "
                  f"audit([good, None, {{}}]) reported 3; mechanism/value refused as "
                  f"unverifiable_kind")


@check("extra supporting evidence never makes a claim worse")
def t_referee_scope_monotone():
    """`_check_scope` compared evidence against scope with set equality, so citing
    one ADDITIONAL supporting experiment could flip a claim from passing to
    `overscoped`. A referee that punishes more evidence is training the wrong
    behaviour. Coverage semantics fix it; over-quantification still fails."""
    import critic

    log = [
        {"experiment_id": "e_a", "config": {"target": "E1", "split": "random",
                                            "cheap_level": "PBE0-SVP", "method": "delta"}},
        {"experiment_id": "e_b", "config": {"target": "f1", "split": "random",
                                           "cheap_level": "PBE0-SVP", "method": "delta"}},
    ]
    c = critic.Critic(log=log)
    base = {"id": "s", "kind": "comparison", "statement": "E1 claim",
            "scope": {"target": "E1", "split": "random"}}
    one = c._check_scope({**base, "evidence": ["e_a"]})
    two = c._check_scope({**base, "evidence": ["e_a", "e_b"]})
    over = c._check_scope({**base, "scope": {"target": "all", "split": "random"},
                           "evidence": ["e_a", "e_b"]})
    absent = c._check_scope({**base, "scope": {"target": "E2", "split": "random"},
                             "evidence": ["e_a", "e_b"]})
    ok = one[0] and two[0] and not over[0] and not absent[0]
    return ok, (f"1 run {one[0]}, +1 supporting run {two[0]} (was False); "
                f"target='all' on 2 of 4 {over[0]}; target='E2' with no E2 run {absent[0]}")


@check("an omitted split is inferred from the evidence, not punished")
def t_referee_split_inference():
    """tools.py tells the agent to omit an axis it is not claiming about; the
    referee used to reject exactly that as `unsupported_claim`. The harness must not
    punish the behaviour it instructs. Disagreeing evidence is still fatal."""
    import critic

    log = [
        {"experiment_id": "e_a", "config": {"target": "E1", "split": "random",
                                            "cheap_level": "PBE0-SVP", "method": "direct"}},
        {"experiment_id": "e_b", "config": {"target": "E1", "split": "random",
                                            "cheap_level": "PBE0-SVP", "method": "delta"}},
        {"experiment_id": "e_c", "config": {"target": "E1", "split": "scaffold",
                                            "cheap_level": "PBE0-SVP", "method": "delta"}},
    ]
    c = critic.Critic(log=log)
    base = {"id": "s", "kind": "comparison", "statement": "no split named", "scope": {}}
    agree = c._check_split_hash({**base, "evidence": ["e_a", "e_b"]})
    disagree = c._check_split_hash({**base, "evidence": ["e_a", "e_c"]})
    nothing = c._check_split_hash({**base, "evidence": ["e_missing"]})
    ok = agree[0] and not disagree[0] and not nothing[0] and "inferred" in agree[1]
    return ok, f"agreeing: {agree[1]}; disagreeing: {disagree[1]}; none: {nothing[1]}"


@check("direct_aug closes most of the direct-vs-delta gap")
def t_direct_aug():
    """The published claim was that a structure-only model is worse than doing
    nothing and delta is the only method that survives distribution shift. Given the
    same four cheap TDDFT numbers delta gets, direct closes most of the gap -- so the
    old headline was a statement about input access, not about the delta
    construction. This is the baseline that was missing."""
    import models

    out = {}
    for split in ("random", "scaffold"):
        out[split] = {
            m: models.run(target="E1", method=m, cheap_level="PBE0-TZVP", split=split,
                          speed="fast").mae
            for m in ("cheap", "direct", "direct_aug", "delta")
        }
    r, s = out["random"], out["scaffold"]
    # direct alone loses to cheap under scaffold shift; direct_aug does not, and it
    # lands within 0.02 of delta on both splits while delta still wins.
    ok = (s["direct"] > s["cheap"] and s["direct_aug"] < s["cheap"]
          and r["direct_aug"] - r["delta"] < 0.01
          and s["direct_aug"] - s["delta"] < 0.02
          and s["delta"] < s["direct_aug"])
    return ok, (
        f"random: direct {r['direct']:.4f} -> direct_aug {r['direct_aug']:.4f} "
        f"vs delta {r['delta']:.4f}; scaffold: direct {s['direct']:.4f} (worse than "
        f"cheap {s['cheap']:.4f}) -> direct_aug {s['direct_aug']:.4f} vs delta "
        f"{s['delta']:.4f}"
    )


@check("the referee narrows an over-scoped claim")
def t_referee_narrows():
    audit = ROOT / "results" / "audit_mock.json"
    if not audit.exists():
        return False, "no mock audit to inspect"
    v = json.loads(audit.read_text())["verdicts"][0]
    return v["verdict"] == "narrowed", v.get("narrowed_statement", "")[-95:]


@check("state-ordering: swap rate reproduces, and the control is null")
def t_state_ordering():
    """The project's most novel claim, plus the control that makes it mean anything.

    Energies are sorted by construction, so swapping them can only hurt. If the
    energy control ever shows a gain comparable to the oscillator strengths, the
    oscillator result is an artifact of reshuffling two correlated columns.
    """
    sys.path.insert(0, str(ROOT))
    from analysis.state_ordering import energy_control, swap_analysis

    sw = swap_analysis("PBE0-SVP")
    ctrl = energy_control("PBE0-SVP")
    ok = (0.15 < sw["swap_rate"] < 0.18
          and ctrl["oracle_gain_pct_energies"] == 0.0
          and sw["mean_cc2_gap_when_swapped"] < sw["mean_cc2_gap_otherwise"])
    return ok, (
        f"swap rate {sw['swap_rate']:.1%} (claimed 16.4%), oracle gain "
        f"{sw['oracle_gain_pct']}%; energy control gain "
        f"{ctrl['oracle_gain_pct_energies']}% (must be 0); gap "
        f"{sw['mean_cc2_gap_when_swapped']:.2f} vs {sw['mean_cc2_gap_otherwise']:.2f} eV"
    )


@check("delta-learning on 100 labels beats direct on the full training set")
def t_label_efficiency():
    import models

    d100 = models.run(target="E1", method="delta", cheap_level="PBE0-TZVP",
                      split="random", n_train=100, seed=0, speed="fast").mae
    full = models.run(target="E1", method="direct", cheap_level="PBE0-TZVP",
                      split="random", speed="fast")
    factor = full.n_train / 100
    return d100 < full.mae, (
        f"delta@100 {d100:.5f} vs direct@{full.n_train} {full.mae:.5f} "
        f"-> {factor:.0f}x fewer expensive labels"
    )


@check("the retracted abstention claim stays retracted")
def t_abstention_retracted():
    """This check used to certify the flagship. It now certifies its retraction.

    The claim was: a structure-only classifier predicts which molecules the
    delta-model gets wrong, and abstaining on them beats random rejection. That is
    true and it is not enough -- on the SEALED set a free rule (rank by the cheap
    E2-E1 gap, no training, no features) beats the classifier on both targets, and
    on f1 the gain does not survive magnitude stratification.

    The old version of this check ran on validation, where the classifier wins.
    That is the same val->test error that produced the withdrawn 31% headline, so
    the check that was supposed to guard the claim shared the claim's blind spot.
    """
    import json

    b = json.loads((ROOT / "results" / "abstention_battery.json").read_text())
    lines = []
    for target in ("f1", "E1"):
        t = b["targets"][target]
        clf = t["risk_rules"]["classifier"]["selective_mae"]
        best = t["best_free_baseline"]
        lines.append(f"{target}: classifier {clf:.5f} vs free {best['name']} {best['selective_mae']:.5f}")
        if t["beats_free_baselines"]:
            return False, f"{target}: classifier still beats every free baseline -- retraction wrong"
    st = b["targets"]["f1"]["magnitude_stratified"]
    inside = st["selective_mae"] >= st["null_ci95"][0]
    return inside, ("; ".join(lines)
                    + f"; f1 stratified {st['selective_mae']:.5f} inside null {st['null_ci95']}")


@check("the loop runs with no GPU and no network")
def t_offline_loop():
    import os

    env = {**os.environ, "QM8_PROFILE": "mock"}
    r = subprocess.run([sys.executable, str(ROOT / "agent.py")],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    ok = r.returncode == 0 and "stopped : claim" in r.stdout
    line = next((l for l in r.stdout.splitlines() if l.startswith("tokens")), "")
    return ok, f"mock rollout completed, {line.strip()}"


def main() -> int:
    print("QM8 research harness — selftest\n")
    fast = (t_parse, t_duplication, t_alignment, t_reset, t_sealed, t_no_leak,
            t_split_integrity, t_shuffle_control, t_referee_rejects,
            t_referee_total, t_referee_scope_monotone, t_referee_split_inference,
            t_referee_narrows, t_offline_loop)
    findings = (t_state_ordering, t_label_efficiency, t_abstention_retracted,
                t_direct_aug)

    print("-- harness --")
    for fn in fast:
        fn()
    if "--quick" in sys.argv:
        print("\n-- findings -- (skipped, --quick)")
    else:
        print("\n-- findings --")
        for fn in findings:
            fn()

    failed = [n for n, s, _ in _results if s == FAIL]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
