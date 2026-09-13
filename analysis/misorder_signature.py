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


_SCORES: dict = {}


def _risk_scores(part: str, level: str, split: str) -> np.ndarray:
    """Predicted misordering risk for one partition, classifier fit on `train`.

    Memoized because the fit is deterministic and target-independent -- the
    label is "did the two states swap", which is a property of the molecule, so
    the f1 and E1 adjudications are scoring the identical classifier and should
    not pay for it twice.
    """
    import lightgbm as lgb

    key = (part, level, split)
    if key in _SCORES:
        return _SCORES[key]

    env = models._env()
    X, pos = env["X"], env["positions"]
    parts = models._split(split)
    tr, ev = parts["train"], parts[part]
    y = misordered_labels(level)

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, n_jobs=-1, verbose=-1, force_row_wise=True)
    clf.fit(X[pos[tr]], y[tr])
    _SCORES[key] = clf.predict_proba(X[pos[ev]])[:, 1]
    return _SCORES[key]


def sealed_risk_scores(level: str = CHEAP_LEVEL, split: str = SPLIT) -> np.ndarray:
    """Predicted misordering risk for the sealed test partition."""
    return _risk_scores("sealed_test", level, split)


def risk_coverage_sealed(target: str = TARGET, n_boot: int = 2000) -> dict:
    """The SEALED curve. Requires the audit token, like every sealed number."""
    p = sealed_risk_scores()
    r = models.run(target=target, method="delta", cheap_level=CHEAP_LEVEL, split=SPLIT,
                   speed="full", eval_on="sealed_test", audit_token=models.AUDIT_TOKEN,
                   return_errors=True)
    return {"eval_on": "sealed_test", "target": target, **_curve(r.errors, p, n_boot)}


# --------------------------------------------------------------------------
# The controls this claim was never run against.
# --------------------------------------------------------------------------
#
# Random rejection is the wrong null, and it is the wrong null in a way that
# flatters every rule tested against it. MAE on a non-negative quantity falls
# whenever you drop the large values, so ANY score correlated with |target|
# clears random rejection without predicting a single error. The curve above
# proves the classifier is not noise; it does not prove the classifier is doing
# the thing the claim says it is doing.
#
# Two further questions decide that, and neither was asked before:
#
#   is it better than a rule that costs nothing?   cheap_gap, predicted_correction
#   does it survive with magnitude held fixed?     the stratified null
#
# `oracle` is here for a third reason: a risk-coverage plot without its ceiling
# invites the reader to score a rule against zero rather than against how much
# of the available separation it actually captured.

BRIGHT = 0.05        # f >= 0.05 a.u. is a molecule a photophysics screen exists to find
N_STRATA = 5


def _rows(part: str, split: str = SPLIT) -> np.ndarray:
    """Absolute feature-matrix rows for one partition of a split."""
    env = models._env()
    return env["positions"][models._split(split)[part]]


def _col(prop: str, level: str) -> int:
    return models._env()["names"].index(f"{prop}-{level}")


def sealed_cheap_gap_risk(level: str = CHEAP_LEVEL, split: str = SPLIT) -> np.ndarray:
    """Rank by the cheap E2-E1 gap: near-degenerate states are where the ordering
    is ambiguous, so a small gap is high risk.

    Free. Both numbers are already sitting in the TDDFT output that the delta
    model corrects, and nothing is fitted, so this rule costs a subtraction.
    """
    T = models._env()["targets"]
    rows = _rows("sealed_test", split)
    gap = T[rows, _col("E2", level)] - T[rows, _col("E1", level)]
    return -gap          # ascending risk == descending gap


def sealed_delta_correction(target: str, level: str = CHEAP_LEVEL, split: str = SPLIT,
                            seed: int = 0) -> np.ndarray:
    """The SIGNED correction the delta model predicts for the sealed partition.

    Refit rather than reached into, but identical by construction -- same
    features, same hyperparameters, same seed as models.run(method="delta",
    speed="full"). `selective_battery` asserts that identity against the errors
    models.run returns rather than assuming it.
    """
    import lightgbm as lgb

    env = models._env()
    T, X = env["targets"], env["X"]
    tr, te = _rows("train", split), _rows("sealed_test", split)
    y = T[:, _col(target, "CC2")] - T[:, _col(target, level)]
    m = lgb.LGBMRegressor(**models.SPEED["full"], random_state=seed, n_jobs=-1,
                          verbose=-1, force_row_wise=True)
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def sealed_residual_regressor_risk(target: str, level: str = CHEAP_LEVEL, split: str = SPLIT,
                                   seed: int = 0) -> np.ndarray:
    """Rank by a LightGBM regressor on |CC2 - cheap|, trained on `train` alone.

    Not free: it costs a second fit on the same 1036 features the classifier
    uses. That is the point of including it -- it is the classifier's own budget
    spent on the quantity the claim cares about (how wrong the cheap number is)
    instead of on a proxy for it (whether two states swapped).
    """
    import lightgbm as lgb

    env = models._env()
    T, X = env["targets"], env["X"]
    tr, te = _rows("train", split), _rows("sealed_test", split)
    y = np.abs(T[:, _col(target, "CC2")] - T[:, _col(target, level)])
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          random_state=seed, n_jobs=-1, verbose=-1, force_row_wise=True)
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def magnitude_strata(target: str, split: str = SPLIT, n: int = N_STRATA) -> list[np.ndarray]:
    """Sealed-set positions binned into |target_CC2| quantiles.

    Stratifying on the TRUE magnitude is deliberate. The question is not whether
    the rule can be defended as predicting magnitude; it is what is left of it
    once magnitude cannot be the answer.
    """
    T = models._env()["targets"]
    mag = np.abs(T[_rows("sealed_test", split), _col(target, "CC2")])
    edges = np.quantile(mag, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-12
    return [np.where((mag >= edges[i]) & (mag < edges[i + 1]))[0] for i in range(n)]


def _stratified_keep(risk: np.ndarray, strata: list[np.ndarray], coverage: float) -> np.ndarray:
    """Lowest-risk `coverage` fraction WITHIN each stratum."""
    keep = []
    for s in strata:
        k = max(1, int(round(coverage * len(s))))
        keep.append(s[np.argsort(risk[s])[:k]])
    return np.concatenate(keep)


def _stratified_null(err: np.ndarray, strata: list[np.ndarray], coverage: float,
                     n_boot: int, seed: int = 0) -> np.ndarray:
    """Random rejection matched within strata -- the null the claim needs to beat."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        keep = [rng.choice(s, max(1, int(round(coverage * len(s)))), replace=False)
                for s in strata]
        draws.append(float(err[np.concatenate(keep)].mean()))
    return np.sort(draws)


def selective_battery(target: str = TARGET, level: str = CHEAP_LEVEL, split: str = SPLIT,
                      coverage: float = 0.5, n_boot: int = 2000) -> dict:
    """Every rival risk rule at one coverage, on the sealed test set.

    One function so that the referee's verdict and the published table cannot
    disagree: critic.py::_sealed_selective calls this, and so does the battery
    driver that writes results/abstention_battery.json.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    r = models.run(target=target, method="delta", cheap_level=level, split=split,
                   speed="full", eval_on="sealed_test", audit_token=models.AUDIT_TOKEN,
                   return_errors=True)
    err = r.errors
    n = len(err)
    k = max(1, int(round(coverage * n)))

    T = models._env()["targets"]
    te = _rows("sealed_test", split)
    y_ref = T[te, _col(target, "CC2")]
    y_cheap = T[te, _col(target, level)]

    p = sealed_risk_scores(level, split)
    y_te = misordered_labels(level)[models._split(split)["sealed_test"]]
    raw = sealed_delta_correction(target, level, split)
    risks = {
        "classifier": p,
        "cheap_gap": sealed_cheap_gap_risk(level, split),
        "predicted_correction": np.abs(raw),
        "residual_regressor": sealed_residual_regressor_risk(target, level, split),
        "oracle": err,
    }
    free = {"cheap_gap", "predicted_correction"}

    def selective(risk):
        return float(err[np.argsort(risk)[:k]].mean())

    rules = {
        name: {"selective_mae": round(selective(risk), 6), "free": name in free}
        for name, risk in risks.items()
    }

    rb = np.random.default_rng(0)
    rand = np.sort([float(err[rb.choice(n, k, replace=False)].mean()) for _ in range(n_boot)])

    strata = magnitude_strata(target, split)
    keep = _stratified_keep(p, strata, coverage)
    null = _stratified_null(err, strata, coverage, n_boot)
    strat_mae = float(err[keep].mean())

    best_free = min((nm for nm in risks if nm in free),
                    key=lambda nm: rules[nm]["selective_mae"])

    out = {
        "target": target,
        "cheap_level": level,
        "split": split,
        "eval_on": "sealed_test",
        "coverage": coverage,
        "n_eval": n,
        "n_kept": k,
        "full_mae": round(r.mae, 6),
        "risk_rules": rules,
        "random_rejection": {
            "selective_mae": round(float(rand.mean()), 6),
            "ci95": [round(float(rand[int(0.025 * n_boot)]), 6),
                     round(float(rand[int(0.975 * n_boot)]), 6)],
        },
        "best_free_baseline": {"name": best_free,
                               "selective_mae": rules[best_free]["selective_mae"]},
        "beats_free_baselines": bool(rules["classifier"]["selective_mae"]
                                     < rules[best_free]["selective_mae"]),
        "magnitude_stratified": {
            "n_strata": len(strata),
            "stratum_sizes": [int(len(s)) for s in strata],
            "n_kept": int(len(keep)),
            "selective_mae": round(strat_mae, 6),
            "null_mae": round(float(null.mean()), 6),
            "null_ci95": [round(float(null[int(0.025 * n_boot)]), 6),
                          round(float(null[int(0.975 * n_boot)]), 6)],
            "beats_null": bool(strat_mae < null[int(0.025 * n_boot)]),
        },
        "classifier_auc": round(float(roc_auc_score(y_te, p)), 4) if len(set(y_te)) > 1 else None,
        "spearman_risk_vs_target": round(float(spearmanr(p, y_ref).statistic), 4),
        # The refit above must be the same model models.run scored, or
        # `predicted_correction` is a different model's opinion wearing its name.
        "correction_refit_matches_delta_model": bool(
            np.allclose(np.abs(raw + y_cheap - y_ref), err)
        ),
    }
    out["bright_retention"] = _bright_retention(y_ref, p, k, n_boot) if target.startswith("f") \
        else {"applicable": False,
              "note": f"a brightness threshold is meaningless for {target}; "
                      f"every molecule clears {BRIGHT} eV"}
    return out


def classifier_vs_free_gap(target: str = TARGET, level: str = CHEAP_LEVEL, split: str = SPLIT,
                           coverage: float = 0.5) -> dict:
    """The classifier against the free cheap-gap rule, on BOTH partitions.

    An external review reported E1 as surviving the retraction, with the
    classifier at 0.05182 against the free gap rule's 0.05749. Those are
    VALIDATION numbers -- the same partition that produced the 31% headline this
    project already withdrew. On the sealed partition the ordering reverses. So
    the comparison is computed side by side rather than argued about: whoever
    reads the write-up can see that the disagreement is a val->test gap and not a
    difference of method.
    """
    out = {}
    for part in ("validation", "sealed_test"):
        kw = {"audit_token": models.AUDIT_TOKEN} if part == "sealed_test" else {}
        r = models.run(target=target, method="delta", cheap_level=level, split=split,
                       speed="full", eval_on=part, return_errors=True, **kw)
        err = r.errors
        k = max(1, int(round(coverage * len(err))))
        rows = _rows(part, split)
        T = models._env()["targets"]
        gap = -(T[rows, _col("E2", level)] - T[rows, _col("E1", level)])
        p = _risk_scores(part, level, split)
        sel = lambda risk: round(float(err[np.argsort(risk)[:k]].mean()), 6)  # noqa: E731
        out[part] = {
            "n_eval": int(len(err)),
            "full_mae": round(r.mae, 6),
            "classifier": sel(p),
            "cheap_gap": sel(gap),
        }
        out[part]["classifier_wins"] = bool(out[part]["classifier"] < out[part]["cheap_gap"])
    out["ordering_reverses_val_to_sealed"] = bool(
        out["validation"]["classifier_wins"] != out["sealed_test"]["classifier_wins"]
    )
    return out


def _bright_retention(y_ref: np.ndarray, risk: np.ndarray, k: int, n_boot: int) -> dict:
    """How many of the molecules a screen is looking for does the keep-set keep?

    Selective MAE is indifferent to which molecules survive. A screen is not: a
    rule that abstains on the bright tail has thrown away the answer and been
    rewarded for it, because the bright tail is also where the errors are.
    """
    bright = y_ref >= BRIGHT
    total = int(bright.sum())
    kept = int(bright[np.argsort(risk)[:k]].sum())
    rng = np.random.default_rng(0)
    draws = np.sort([int(bright[rng.choice(len(y_ref), k, replace=False)].sum())
                     for _ in range(n_boot)])
    return {
        "applicable": True,
        "threshold": BRIGHT,
        "total_bright": total,
        "kept_by_rule": kept,
        "kept_by_random_mean": round(float(draws.mean()), 1),
        "kept_by_random_ci95": [int(draws[int(0.025 * n_boot)]), int(draws[int(0.975 * n_boot)])],
        "retention_rule": round(kept / total, 4) if total else None,
        "retention_random": round(float(draws.mean()) / total, 4) if total else None,
    }


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
