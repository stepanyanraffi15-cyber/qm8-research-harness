"""h_misorder_signature — pre-registered 2026-09-08, before measurement.

Claim: molecules whose excited states are misordered between TDDFT and CC2 are
separable from the rest using structure-derived features ALONE.

Why it matters. Everything else in this project needs CC2 to know a molecule is
misordered, which is useless -- CC2 is the thing you are trying not to run. If
misordering is predictable from structure, the model can decline on the molecules
it is about to get wrong, before any expensive calculation. That is an
applicability domain, and it is the part of this work that maps onto screening
novel chemistry rather than onto a benchmark.

Falsification condition, as registered: no structure-only slice separates the
swapped population beyond the paired bootstrap interval.

Control, as registered: `random_reindex`. A classifier trained to predict a
COIN FLIP must show no separation. Without it, any apparent structure could be
the classifier memorising the training set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402

TARGET = "f1"
CHEAP_LEVEL = "PBE0-SVP"
SPLIT = "random"
N_BOOT = 5000


def misordered_labels(level: str) -> np.ndarray:
    """1 where a swapped state assignment fits CC2 better. Needs CC2 -- this is
    the label we are trying to predict WITHOUT it."""
    env = models._env()
    names, T, pos = env["names"], env["targets"], env["positions"]
    i = lambda p, l: names.index(f"{p}-{l}")  # noqa: E731
    f1c, f2c = T[pos, i("f1", "CC2")], T[pos, i("f2", "CC2")]
    f1t, f2t = T[pos, i("f1", level)], T[pos, i("f2", level)]
    as_given = np.abs(f1t - f1c) + np.abs(f2t - f2c)
    swapped = np.abs(f2t - f1c) + np.abs(f1t - f2c)
    return (swapped < as_given).astype(int)


def fit_and_score(y: np.ndarray, tag: str) -> dict:
    """Train a structure-only classifier on train, score on validation."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    env = models._env()
    X, pos = env["X"], env["positions"]
    parts = models._split(SPLIT)
    tr, va = parts["train"], parts["validation"]

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, n_jobs=-1, verbose=-1, force_row_wise=True)
    clf.fit(X[pos[tr]], y[tr])
    p = clf.predict_proba(X[pos[va]])[:, 1]
    auc = float(roc_auc_score(y[va], p)) if len(set(y[va])) > 1 else float("nan")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(p), size=(1000, len(p)))
    aucs = np.sort([roc_auc_score(y[va][k], p[k]) for k in idx if len(set(y[va][k])) > 1])
    return {
        "tag": tag,
        "prevalence_train": round(float(y[tr].mean()), 4),
        "prevalence_val": round(float(y[va].mean()), 4),
        "auc": round(auc, 4),
        "auc_ci95": [round(float(aucs[25]), 4), round(float(aucs[975]), 4)],
        "probs": p,
    }


def error_by_risk(p: np.ndarray, bins: int = 4) -> dict:
    """Does the delta-model actually do worse where the classifier says it will?"""
    r = models.run(target=TARGET, method="delta", cheap_level=CHEAP_LEVEL, split=SPLIT,
                   speed="full", eval_on="validation", return_errors=True)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            out.append({"bin": i, "n": int(m.sum()),
                        "risk_range": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
                        "mae": round(float(r.errors[m].mean()), 6)})

    lo = r.errors[(p >= edges[0]) & (p < edges[1])]
    hi = r.errors[(p >= edges[bins - 1])]
    rng = np.random.default_rng(0)
    ratios = np.sort([
        hi[rng.integers(0, len(hi), len(hi))].mean() / lo[rng.integers(0, len(lo), len(lo))].mean()
        for _ in range(N_BOOT)
    ])
    return {
        "overall_mae": round(r.mae, 6),
        "bins": out,
        "ratio_highest_over_lowest": round(float(hi.mean() / lo.mean()), 3),
        "ratio_ci95": [round(float(ratios[125]), 3), round(float(ratios[4875]), 3)],
        "separates": bool(ratios[125] > 1.0),
    }


def main() -> int:
    print("h_misorder_signature — pre-registered, testing now\n")

    y = misordered_labels(CHEAP_LEVEL)
    real = fit_and_score(y, "misordered")
    print(f"  real   : prevalence {real['prevalence_val']:.1%}  "
          f"AUC {real['auc']:.4f}  CI {real['auc_ci95']}")

    # CONTROL: the same pipeline predicting a coin flip
    rng = np.random.default_rng(999)
    y_ctrl = (rng.random(len(y)) < y.mean()).astype(int)
    ctrl = fit_and_score(y_ctrl, "random_reindex_control")
    print(f"  control: prevalence {ctrl['prevalence_val']:.1%}  "
          f"AUC {ctrl['auc']:.4f}  CI {ctrl['auc_ci95']}")

    predictable = real["auc_ci95"][0] > 0.5 and real["auc_ci95"][0] > ctrl["auc_ci95"][1]
    print(f"\n  structure-only prediction beats chance and beats the control: {predictable}")

    print("\n  does delta-model error follow the predicted risk?")
    risk = error_by_risk(real["probs"])
    for b in risk["bins"]:
        print(f"    q{b['bin']} risk {b['risk_range']}  n={b['n']:4d}  MAE {b['mae']:.6f}")
    print(f"    highest/lowest = {risk['ratio_highest_over_lowest']}x  "
          f"CI {risk['ratio_ci95']}  separates: {risk['separates']}")

    verdict = "SUPPORTED" if (predictable and risk["separates"]) else "FALSIFIED"
    print(f"\n  h_misorder_signature: {verdict}")

    out = {
        "hypothesis": "h_misorder_signature",
        "registered_before_measurement": True,
        "classifier": {k: v for k, v in real.items() if k != "probs"},
        "control": {k: v for k, v in ctrl.items() if k != "probs"},
        "structure_predictable": bool(predictable),
        "error_by_risk": risk,
        "verdict": verdict,
    }
    (ROOT / "results" / "misorder_signature.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  wrote {ROOT / 'results' / 'misorder_signature.json'}")
    return 0


# --------------------------------------------------------------------------
# The deliverable this was always for: a risk-coverage curve.
# --------------------------------------------------------------------------
#
# "abstention: on/off" returns one number and cannot express the finding. The
# object of interest is error against coverage across every rejection threshold,
# and its only honest baseline is RANDOM rejection at matched coverage -- a model
# that looks good discarding 30% of molecules has proven nothing until it beats
# throwing 30% away at random.


COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def _curve(err: np.ndarray, p: np.ndarray, n_boot: int) -> dict:
    """Error against coverage, with random rejection at matched coverage.

    `err` and `p` must be aligned to the SAME partition: p[i] is the predicted
    risk of the molecule whose delta-model error is err[i]. Getting that wrong is
    how a validation curve ends up labelled as a sealed one.
    """
    if len(err) != len(p):
        raise ValueError(f"errors ({len(err)}) and risks ({len(p)}) are different partitions")
    order = np.argsort(p)          # keep the lowest-risk molecules first
    rng = np.random.default_rng(0)

    rows = []
    for cov in COVERAGES:
        k = max(1, int(round(cov * len(err))))
        selective = float(err[order[:k]].mean())
        # random rejection at the SAME coverage, bootstrapped
        rand = np.sort([float(err[rng.choice(len(err), k, replace=False)].mean())
                        for _ in range(n_boot)])
        rows.append({
            "coverage": cov,
            "n_kept": k,
            "selective_mae": round(selective, 6),
            "random_mae": round(float(rand.mean()), 6),
            "random_ci95": [round(float(rand[int(.025 * n_boot)]), 6),
                            round(float(rand[int(.975 * n_boot)]), 6)],
            "beats_random": bool(selective < rand[int(.025 * n_boot)]),
        })
    aurc = float(np.mean([x["selective_mae"] for x in rows]))
    return {"rows": rows, "aurc": round(aurc, 6),
            "beats_random_everywhere": all(x["beats_random"] for x in rows if x["coverage"] < 1.0)}


def risk_coverage(p: np.ndarray, n_boot: int = 2000) -> dict:
    """The VALIDATION curve. `p` must come from `fit_and_score`, which scores the
    validation partition. Nothing computed here may be reported as sealed."""
    r = models.run(target=TARGET, method="delta", cheap_level=CHEAP_LEVEL, split=SPLIT,
                   speed="full", eval_on="validation", return_errors=True)
    return {"eval_on": "validation", **_curve(r.errors, p, n_boot)}


# --------------------------------------------------------------------------
# The same curve on the partition the claim is actually adjudicated against.
# --------------------------------------------------------------------------
#
# The validation curve above is what the agent is allowed to see, so it is what
# the exploratory analysis runs on -- and it is NOT the number the write-up may
# quote as verified. The sealed curve below is, and it is measurably weaker:
# selective prediction has to survive the val->test gap like everything else.
# Same construction as critic.py::_sealed_selective -- classifier fit on `train`
# alone, scored on a partition it has never seen -- so the curve and the
# referee's verdict cannot disagree.


def sealed_risk_scores(level: str = CHEAP_LEVEL, split: str = SPLIT) -> np.ndarray:
    """Predicted misordering risk for the sealed test partition."""
    import lightgbm as lgb

    env = models._env()
    X, pos = env["X"], env["positions"]
    parts = models._split(split)
    tr, te = parts["train"], parts["sealed_test"]
    y = misordered_labels(level)

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, n_jobs=-1, verbose=-1, force_row_wise=True)
    clf.fit(X[pos[tr]], y[tr])
    return clf.predict_proba(X[pos[te]])[:, 1]


def risk_coverage_sealed(target: str = TARGET, n_boot: int = 2000) -> dict:
    """The SEALED curve. Requires the audit token, like every sealed number."""
    p = sealed_risk_scores()
    r = models.run(target=target, method="delta", cheap_level=CHEAP_LEVEL, split=SPLIT,
                   speed="full", eval_on="sealed_test", audit_token=models.AUDIT_TOKEN,
                   return_errors=True)
    return {"eval_on": "sealed_test", "target": target, **_curve(r.errors, p, n_boot)}


def main_sealed() -> int:
    print("risk-coverage on the SEALED test set\n")
    out = risk_coverage_sealed()
    full = out["rows"][0]["selective_mae"]
    for r in out["rows"]:
        drop = 100 * (1 - r["selective_mae"] / full)
        print(f"  coverage {r['coverage']:.0%}  n={r['n_kept']:4d}  "
              f"selective {r['selective_mae']:.6f}  random {r['random_mae']:.6f} "
              f"CI {r['random_ci95']}  beats_random {str(r['beats_random']):5s}  −{drop:.1f}%")
    print(f"\n  AURC {out['aurc']:.6f}  beats random everywhere: "
          f"{out['beats_random_everywhere']}")
    path = ROOT / "results" / "risk_coverage_sealed.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_sealed() if "--sealed" in sys.argv else main())
