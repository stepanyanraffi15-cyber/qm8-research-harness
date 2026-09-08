"""The prediction layer: cheap / direct / delta. No LLM anywhere near this file.

    cheap    report the TDDFT number and do nothing. Free, and a real baseline.
    direct   predict CC2 from structure. What MoleculeNet-style models do.
    delta    predict the CC2 - TDDFT residual and add it back. Ramakrishnan's
             1988 idea applied to QM8, and the reason QM8 exists.

    zero     predict 0.0 everywhere. Only meaningful for oscillator strengths,
             where the target is non-negative and 32% of molecules sit below
             1e-3 -- so MAE rewards predicting nothing. Any claim that
             "delta-learning fails for f" has to clear this bar first, and the
             earlier round of this project never checked it.

The evaluation contract lives here and nowhere else:

    eval_on="validation"   what the agent is allowed to see
    eval_on="sealed_test"  what only critic.py may ask for

The agent's tools call this with the default. Passing "sealed_test" requires an
explicit audit token, so a leak has to be deliberate rather than accidental.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import features

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"

TARGETS = ["E1", "E2", "f1", "f2"]
CHEAP_LEVELS = ["PBE0-SVP", "PBE0-TZVP", "CAM"]
METHODS = ["cheap", "direct", "delta", "zero"]
SPLITS = ["random", "scaffold", "tddft_gap"]

# Two speeds, as the design calls for: fast to explore, full to claim.
SPEED = {
    "fast": dict(n_estimators=200, learning_rate=0.08, num_leaves=31),
    "full": dict(n_estimators=800, learning_rate=0.05, num_leaves=63),
}

# Only critic.py holds this. Nothing reachable from tools.py does.
AUDIT_TOKEN = "sealed-test-audit"

_CACHE: dict = {}


@dataclass
class Result:
    config: dict
    mae: float
    n_train: int
    n_eval: int
    eval_on: str
    fit_seconds: float
    extras: dict = field(default_factory=dict)

    # Per-molecule absolute errors, aligned to the eval partition's order, and
    # populated only when return_errors=True. critic.py pairs two configs on
    # these. Molecule-level pairing is stronger than seed-level pairing, and it
    # sidesteps the batched-server nondeterminism that makes seed-pairing leaky.
    errors: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "mae": round(self.mae, 6),
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "eval_on": self.eval_on,
            "fit_seconds": round(self.fit_seconds, 2),
            **({"extras": self.extras} if self.extras else {}),
        }


def _env():
    if "env" not in _CACHE:
        d = np.load(DERIVED / "qm8.npz", allow_pickle=True)
        _CACHE["env"] = {
            "targets": d["targets_ev"],
            "positions": d["positions"],
            "names": [str(s) for s in d["target_names"]],
            "X": features.load_or_build(),
        }
    return _CACHE["env"]


class SplitIntegrityError(RuntimeError):
    pass


def _split(name: str) -> dict:
    """Load a split, refusing it if its contents no longer match the manifest.

    A claim is tied to a partition by hash. If a split file can drift without
    anyone noticing, every downstream number is unfalsifiable -- so a mismatch
    is a hard failure, not a warning.
    """
    key = f"split:{name}"
    if key not in _CACHE:
        import hashlib
        import json

        z = np.load(DERIVED / f"split_{name}.npz")
        parts = {k: z[k] for k in ("train", "validation", "sealed_test")}
        expected = json.loads((DERIVED / "manifest.json").read_text())["splits"][name]
        for part, arr in parts.items():
            got = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            if got != expected[part]["sha256"]:
                raise SplitIntegrityError(
                    f"split {name!r} partition {part!r} does not match the manifest\n"
                    f"  expected {expected[part]['sha256'][:16]}\n"
                    f"  got      {got[:16]}\n"
                    f"Rebuild with `python world/build.py --seed 0` or restore the file."
                )
        _CACHE[key] = parts
    return _CACHE[key]


def _column(names: list[str], prop: str, level: str) -> int:
    return names.index(f"{prop}-{level}")


def run(
    target: str = "E1",
    method: str = "delta",
    cheap_level: str = "PBE0-SVP",
    split: str = "random",
    n_train: int | None = None,
    seed: int = 0,
    speed: str = "fast",
    eval_on: str = "validation",
    audit_token: str | None = None,
    return_errors: bool = False,
) -> Result:
    """Fit and score one configuration.

    Raises if asked for the sealed test set without the audit token. That check
    is the enforcement point for the whole project's invariant, so it is a hard
    failure rather than a warning.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if cheap_level not in CHEAP_LEVELS:
        raise ValueError(f"cheap_level must be one of {CHEAP_LEVELS}, got {cheap_level!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if speed not in SPEED:
        raise ValueError(f"speed must be one of {list(SPEED)}, got {speed!r}")
    if eval_on == "sealed_test" and audit_token != AUDIT_TOKEN:
        raise PermissionError(
            "sealed_test requires the audit token; the agent's tools cannot obtain it"
        )
    if eval_on not in ("validation", "sealed_test"):
        raise ValueError(f"eval_on must be validation or sealed_test, got {eval_on!r}")

    # Memoize on the full configuration. An audit re-runs the same config on both
    # partitions and across several claims; at speed="full" that is minutes of
    # refitting for an answer already computed. The key includes everything that
    # can change a number, so this cannot silently serve a stale result.
    ckey = (target, method, cheap_level, split, n_train, seed, speed, eval_on)
    hit = _CACHE.get(("fit", ckey))
    if hit is not None and (hit.errors is not None or not return_errors):
        return hit

    env = _env()
    parts = _split(split)
    pos, names = env["positions"], env["names"]

    tr_local, ev_local = parts["train"], parts[eval_on]
    if n_train is not None and n_train < len(tr_local):
        rng = np.random.default_rng(seed)
        tr_local = np.sort(rng.choice(tr_local, size=n_train, replace=False))

    tr, ev = pos[tr_local], pos[ev_local]
    y_ref = env["targets"][:, _column(names, target, "CC2")]
    y_cheap = env["targets"][:, _column(names, target, cheap_level)]

    t0 = time.perf_counter()
    if method == "cheap":
        pred = y_cheap[ev]
    elif method == "zero":
        pred = np.zeros(len(ev))
    else:
        import lightgbm as lgb

        X = env["X"]
        y_tr = y_ref[tr] if method == "direct" else (y_ref[tr] - y_cheap[tr])
        model = lgb.LGBMRegressor(
            **SPEED[speed], random_state=seed, n_jobs=-1, verbose=-1, force_row_wise=True
        )
        model.fit(X[tr], y_tr)
        raw = model.predict(X[ev])
        pred = raw if method == "direct" else raw + y_cheap[ev]
    fit_seconds = time.perf_counter() - t0

    err = np.abs(pred - y_ref[ev])
    result = Result(
        config=dict(
            target=target, method=method, cheap_level=cheap_level, split=split,
            n_train=int(len(tr)), seed=seed, speed=speed,
        ),
        mae=float(err.mean()),
        n_train=int(len(tr)),
        n_eval=int(len(ev)),
        eval_on=eval_on,
        fit_seconds=fit_seconds,
        extras={"rmse": float(np.sqrt((err**2).mean())), "max_err": float(err.max())},
        errors=err if return_errors else None,
    )
    _CACHE[("fit", ckey)] = result
    return result


def baseline_table(split: str = "random", speed: str = "fast", seed: int = 0) -> list[dict]:
    """Regenerate the cheap/direct/delta comparison for every target."""
    rows = []
    for target in TARGETS:
        methods = ["cheap", "direct", "delta"] + (["zero"] if target.startswith("f") else [])
        row = {"target": target, "split": split}
        for m in methods:
            row[m] = round(run(target=target, method=m, split=split, speed=speed, seed=seed).mae, 5)
        rows.append(row)
    return rows


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="regenerate the baseline table")
    ap.add_argument("--split", default="random", choices=SPLITS)
    ap.add_argument("--speed", default="fast", choices=list(SPEED))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-splits", action="store_true")
    args = ap.parse_args()

    out = {}
    for sp in (SPLITS if args.all_splits else [args.split]):
        rows = baseline_table(split=sp, speed=args.speed, seed=args.seed)
        out[sp] = rows
        print(f"\n=== {sp} split · speed={args.speed} · seed={args.seed} ===")
        print(f"{'target':7s} {'cheap':>10s} {'direct':>10s} {'delta':>10s} {'zero':>10s}")
        for r in rows:
            z = f"{r['zero']:10.5f}" if "zero" in r else " " * 10
            print(f"{r['target']:7s} {r['cheap']:10.5f} {r['direct']:10.5f} {r['delta']:10.5f} {z}")
    print()
    print(json.dumps(out))
