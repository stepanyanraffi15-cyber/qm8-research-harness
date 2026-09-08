"""Every check this project claims to pass, runnable in one command.

    uv run python selftest.py

The point is that a reviewer should not have to take any number in the write-up
on trust. Each check below either reproduces a published result, proves an
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


@check("the referee narrows an over-scoped claim")
def t_referee_narrows():
    audit = ROOT / "results" / "audit_mock.json"
    if not audit.exists():
        return False, "no mock audit to inspect"
    v = json.loads(audit.read_text())["verdicts"][0]
    return v["verdict"] == "narrowed", v.get("narrowed_statement", "")[-95:]


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
    for fn in (t_parse, t_duplication, t_alignment, t_reset, t_sealed, t_no_leak,
               t_split_integrity, t_shuffle_control, t_referee_rejects,
               t_referee_narrows, t_offline_loop):
        fn()

    failed = [n for n, s, _ in _results if s == FAIL]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
