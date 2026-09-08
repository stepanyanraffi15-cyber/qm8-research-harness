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


if __name__ == "__main__":
    raise SystemExit(main())
