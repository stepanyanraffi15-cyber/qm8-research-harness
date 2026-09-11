"""Does the 2015 negative result depend on the metric?

Under MAE, reporting the cheap TDDFT number beats delta-learning for oscillator
strengths. That is the 2015 finding and this project reproduced it. But MAE is not
what a screening campaign optimises: nobody sorts a library by mean absolute error.
They rank it and look at the top.

So: cheap vs delta on the metrics a screen actually uses -- ranking (Spearman),
retrieval (AUC, average precision), and enrichment at a small top fraction.

This is the candidate replacement headline, and it gets the same adversarial
treatment that retracted the last two claims:

  * SEALED test set only, never validation. Both retracted claims died to a
    val->test gap, including the reviewer's proposed rescue for the first one.
  * paired bootstrap over molecules for every difference, resampling the SAME
    molecules for both methods so the between-molecule variance cancels
  * multiple thresholds, because a result that only exists at f >= 0.05 is a
    result about 0.05
  * the bright/dark decomposition, which is the mechanism claim rather than the
    headline claim, reported separately
  * delta_log as a candidate fix, reported as exploratory -- it was not
    pre-registered and cannot become confirmatory now

    python analysis/screening_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models  # noqa: E402

SPLIT = "random"
LEVEL = "PBE0-SVP"
N_BOOT = 2000
EPS = 1e-4


def _fit(target: str, method: str, eval_on: str, log_space: bool = False):
    """Return (pred, truth) on the requested partition."""
    import lightgbm as lgb

    env = models._env()
    names, T, X, pos = env["names"], env["targets"], env["X"], env["positions"]
    parts = models._split(SPLIT)
    tr, ev = parts["train"], parts[eval_on]
    col = names.index(f"{target}-CC2")
    cheap_col = names.index(f"{target}-{LEVEL}")
    y, c = T[:, col], T[:, cheap_col]

    if method == "cheap":
        return c[pos[ev]], y[pos[ev]]

    m = lgb.LGBMRegressor(**models.SPEED["full"], random_state=0, n_jobs=-1,
                          verbose=-1, force_row_wise=True)
    if log_space:
        # predict the correction in log space, then map back. The dark bulk is
        # where the raw-space model regresses to the mean and shreds an ordering
        # TDDFT already had for free.
        lt = np.log10(np.clip(c, 0, None) + EPS)
        ly = np.log10(np.clip(y, 0, None) + EPS)
        m.fit(X[pos[tr]], (ly - lt)[pos[tr]])
        pred = 10 ** (lt[pos[ev]] + m.predict(X[pos[ev]])) - EPS
        return np.clip(pred, 0, None), y[pos[ev]]

    m.fit(X[pos[tr]], (y - c)[pos[tr]])
    return m.predict(X[pos[ev]]) + c[pos[ev]], y[pos[ev]]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _auc(pred, lab):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(lab, pred)) if len(set(lab)) > 1 else float("nan")


def _ap(pred, lab):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(lab, pred)) if len(set(lab)) > 1 else float("nan")


def _spearman(pred, truth):
    from scipy.stats import spearmanr

    return float(spearmanr(pred, truth).statistic)


def _enrichment(pred, lab, frac=0.05):
    k = max(1, int(round(frac * len(pred))))
    top = np.argsort(-pred)[:k]
    base = lab.mean()
    return float(lab[top].mean() / base) if base > 0 else float("nan")


def paired_ci(fn, a, b, truth, seed=0, iters=N_BOOT):
    """Bootstrap the DIFFERENCE fn(b) - fn(a), resampling molecules once per draw
    and scoring both methods on the same resample."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(fn(b[idx], truth[idx]) - fn(a[idx], truth[idx]))
        except Exception:  # degenerate resample (e.g. one class)
            continue
    d = np.sort([x for x in diffs if np.isfinite(x)])
    return {
        "delta": round(float(fn(b, truth) - fn(a, truth)), 5),
        "ci95": [round(float(d[int(0.025 * len(d))]), 5),
                 round(float(d[int(0.975 * len(d))]), 5)],
        "excludes_zero": bool(d[int(0.025 * len(d))] > 0 or d[int(0.975 * len(d))] < 0),
    }


def compare(target: str, eval_on: str = "sealed_test") -> dict:
    kw = {"eval_on": eval_on}
    cheap, truth = _fit(target, "cheap", **kw)
    delta, _ = _fit(target, "delta", **kw)
    dlog, _ = _fit(target, "delta", log_space=True, **kw)

    out = {"target": target, "eval_on": eval_on, "n": int(len(truth)),
           "methods": {}, "differences": {}, "thresholds": {}}

    for name, p in (("cheap", cheap), ("delta", delta), ("delta_log", dlog)):
        out["methods"][name] = {
            "mae": round(float(np.abs(p - truth).mean()), 6),
            "spearman_all": round(_spearman(p, truth), 4),
        }

    # retrieval at several thresholds -- a result that lives at one cut is a
    # result about that cut
    for thr in (0.01, 0.05, 0.1):
        lab = (truth >= thr).astype(int)
        if lab.sum() < 20:
            continue
        row = {"n_positive": int(lab.sum())}
        for name, p in (("cheap", cheap), ("delta", delta), ("delta_log", dlog)):
            row[name] = {"auc": round(_auc(p, lab), 4), "ap": round(_ap(p, lab), 4),
                         "enrich@5%": round(_enrichment(p, lab), 3)}
        row["delta_minus_cheap"] = {
            "auc": paired_ci(_auc, cheap, delta, lab),
            "ap": paired_ci(_ap, cheap, delta, lab),
        }
        out["thresholds"][f"{thr}"] = row

    out["differences"]["spearman_all_delta_minus_cheap"] = paired_ci(
        _spearman, cheap, delta, truth)
    out["differences"]["spearman_all_deltalog_minus_cheap"] = paired_ci(
        _spearman, cheap, dlog, truth)

    # mechanism: where does the ranking change come from?
    bright = truth >= 0.01
    out["decomposition"] = {
        "n_bright": int(bright.sum()), "n_dark": int((~bright).sum()),
        "spearman_bright": {n: round(_spearman(p[bright], truth[bright]), 4)
                            for n, p in (("cheap", cheap), ("delta", delta), ("delta_log", dlog))},
        "spearman_dark": {n: round(_spearman(p[~bright], truth[~bright]), 4)
                          for n, p in (("cheap", cheap), ("delta", delta), ("delta_log", dlog))},
        "mae_bright": {n: round(float(np.abs(p[bright] - truth[bright]).mean()), 6)
                       for n, p in (("cheap", cheap), ("delta", delta), ("delta_log", dlog))},
    }
    return out


def main() -> int:
    report = {"split": SPLIT, "cheap_level": LEVEL, "targets": {}}
    for target in ("f1", "f2"):
        print(f"\n=== {target} (sealed test set) ===")
        r = compare(target)
        report["targets"][target] = r

        print(f"  {'method':11s} {'MAE':>9s} {'Spearman':>9s}")
        for n, m in r["methods"].items():
            print(f"  {n:11s} {m['mae']:9.5f} {m['spearman_all']:9.4f}")

        for thr, row in r["thresholds"].items():
            d = row["delta_minus_cheap"]
            print(f"  thr f>={thr} (n+={row['n_positive']}): "
                  f"AUC cheap {row['cheap']['auc']:.3f} delta {row['delta']['auc']:.3f} "
                  f"[{d['auc']['delta']:+.3f} CI {d['auc']['ci95']} sig={d['auc']['excludes_zero']}]")
            print(f"                AP  cheap {row['cheap']['ap']:.3f} delta {row['delta']['ap']:.3f} "
                  f"[{d['ap']['delta']:+.3f} CI {d['ap']['ci95']} sig={d['ap']['excludes_zero']}]")

        dc = r["decomposition"]
        print(f"  bright/dark (n={dc['n_bright']}/{dc['n_dark']}): "
              f"spearman bright {dc['spearman_bright']} dark {dc['spearman_dark']}")
        sp = r["differences"]
        print(f"  spearman all: delta {sp['spearman_all_delta_minus_cheap']['delta']:+.4f} "
              f"CI {sp['spearman_all_delta_minus_cheap']['ci95']} | "
              f"delta_log {sp['spearman_all_deltalog_minus_cheap']['delta']:+.4f} "
              f"CI {sp['spearman_all_deltalog_minus_cheap']['ci95']}")

    (ROOT / "results" / "screening_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {ROOT / 'results' / 'screening_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
