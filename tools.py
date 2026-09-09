"""The agent's action space.

A tool is a JSON schema and a function, and the description is prompt -- it is the
only thing telling the model when to reach for this rather than something else.

The earlier design offered seven enumerated axes and "a few thousand valid
combinations". Selecting cells from that is grid search with an LLM prior, and
it cannot express the moves that actually produce a finding. The evidence for
that is this project's own history: the state-indexing result came from noticing
an anomaly, retrieving a mechanism, designing a discriminating test, RUNNING A
CONTROL, and finding a mechanistic correlate. Three of those five moves were not
points in the config space.

So the surface is built from primitives instead:

    run_experiment   fit and score one configuration
    slice_error      error conditioned on a molecular property
    run_control      the same test where the effect must be absent
    intervene        change one thing and re-measure
    describe_dataset what is in the environment
    claim            the only route to the write-up; ends the run

`propose_features` is deliberately absent. The literature says sandbox code
execution rather than omit it, and the counter-evidence for omitting it is
heavier than the support. Its absence here is a SCOPE decision, not a safety one, and is
written that way rather than dressed up as principle.

Two honesty notes that belong in the write-up, not buried here:

1. `slice_error(by="state_gap"|"brightness")` and `intervene("reindex_states")`
   are supplied affordances. An agent that recovers the state-indexing result
   through them has done TOOL-ASSISTED HYPOTHESIS GENERATION, not independent
   discovery. The tools are still narrower than handing over the answer as a
   config flag, which is what the earlier design did.

2. Every number here comes from the VALIDATION split. Nothing reachable from
   this file can obtain the sealed test set -- models.run raises without an audit
   token that only critic.py holds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import models

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"
LOG_PATH = ROOT / "results" / "experiment_log.jsonl"

DEFAULT_BUDGET = 30

SLICE_BY = ["state_gap", "heavy_atoms", "brightness", "train_distance", "cheap_error"]
CONTROLS = ["shuffle_labels", "apply_to_energies", "random_reindex"]
INTERVENTIONS = ["reindex_states", "ablate_features", "subsample_train"]


class Session:
    """Holds the log, the budget and the best-so-far. One per agent run.

    The budget is visible to the agent on every observation. ResearchGym names
    poor resource management as a headline agent failure mode, and an agent that
    cannot see its own budget cannot manage it.
    """

    def __init__(self, budget: int = DEFAULT_BUDGET, log_path: Path = LOG_PATH, seed: int = 0):
        self.budget = budget
        self.calls_used = 0
        self.log_path = log_path
        self.seed = seed
        self.entries: list[dict] = []
        self.best: dict[str, tuple[float, np.ndarray, str]] = {}  # family -> (mae, errors, eid)
        self._noise: dict[str, float] = {}
        self.claim: dict | None = None
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- logging ----------------------------------------------------------

    def _next_id(self) -> str:
        return f"e_{len(self.entries):04d}"

    def _log(self, entry: dict) -> None:
        self.entries.append(entry)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def split_hash(self, split: str) -> str:
        m = json.loads((DERIVED / "manifest.json").read_text())
        return "sha256:" + m["splits"][split]["validation"]["sha256"][:16]

    # -- noise scale ------------------------------------------------------

    def noise_scale(self, target: str, split: str) -> float:
        """Seed-to-seed sd of the delta baseline, computed once per family.

        Gives the agent a scale for "is this difference real". Three seeds is
        thin, and it is reported as such -- the authoritative test is the paired
        bootstrap in vs_best, which needs no refit at all.
        """
        key = f"{target}:{split}"
        if key not in self._noise:
            maes = [
                models.run(target=target, method="delta", split=split, seed=s, speed="fast").mae
                for s in (0, 1, 2)
            ]
            self._noise[key] = float(np.std(maes, ddof=1))
        return self._noise[key]


_SESSION: Session | None = None


def session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session()
    return _SESSION


def reset_session(budget: int = DEFAULT_BUDGET, log_path: Path = LOG_PATH, seed: int = 0) -> Session:
    global _SESSION
    _SESSION = Session(budget=budget, log_path=log_path, seed=seed)
    return _SESSION


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _env():
    return models._env()


def _paired(a: np.ndarray, b: np.ndarray, iters: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap, same shape as critic.py's but cheaper for in-loop use."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    means = np.sort(d[idx].mean(axis=1))
    lo, hi = float(means[int(0.025 * iters)]), float(means[int(0.975 * iters)])
    return {
        "delta": round(float(d.mean()), 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "significant": bool(lo > 0 or hi < 0),
    }


def _noise_note(sd: float) -> dict:
    """Report the seed-to-seed scale honestly, including when it is zero.

    LightGBM on fixed data with force_row_wise is fully deterministic, so refits
    at different seeds return the same number and sd is exactly 0. That is not a
    noise floor of zero -- it means seeds are the wrong instrument here, and the
    paired bootstrap over molecules in `vs_best` is the one that measures
    anything. Saying "sd 0.0" without that sentence would invite an agent to
    treat any difference at all as real.
    """
    out = {"sd": round(sd, 6), "n_seeds": 3}
    if sd == 0.0:
        out["note"] = (
            "exactly 0: the fit is deterministic given the data, so seed variance "
            "measures nothing here. Use vs_best (paired bootstrap over molecules) "
            "to decide whether a difference is real."
        )
    else:
        out["note"] = "only 3 seeds; vs_best is the authoritative test"
    return out


def _budget_guard(s: Session) -> str | None:
    if s.calls_used >= s.budget:
        return (
            f"ERROR: budget exhausted ({s.budget} calls). Call `claim` with what you have, "
            "or the run ends with nothing."
        )
    return None


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def describe_dataset() -> str:
    m = json.loads((DERIVED / "manifest.json").read_text())
    p = m["profile"]
    return json.dumps(
        {
            "molecules": m["n_molecules"],
            "targets": m["target_names"],
            "units": m["units"],
            "levels_of_theory": {
                "reference": "RI-CC2/def2TZVP (expensive)",
                "cheap": ["PBE0/def2SVP", "PBE0/def2TZVP", "CAM-B3LYP/def2TZVP"],
            },
            "splits": {k: {p2: v2["n"] for p2, v2 in v.items()} for k, v in m["splits"].items()},
            "note": (
                "You are scored on validation. A sealed test set exists and you cannot "
                "reach it; the referee scores your claim there once."
            ),
            "structure": {
                "mean_abs_r_within_energies": p["mean_abs_r_within_energies"],
                "mean_abs_r_within_oscillator": p["mean_abs_r_within_oscillator"],
                "mean_abs_r_across_blocks": p["mean_abs_r_across_blocks"],
            },
            "f1_cc2_distribution": p["f1_cc2"],
            "methods": {
                "cheap": "report the TDDFT number, no learning",
                "direct": "predict CC2 from structure",
                "delta": "predict CC2 - TDDFT, add it back",
                "zero": "predict 0.0 (a real baseline for oscillator strengths)",
            },
        },
        indent=1,
    )


def run_experiment(
    target: str = "E1",
    method: str = "delta",
    cheap_level: str = "PBE0-SVP",
    split: str = "random",
    n_train: int | None = None,
    seed: int = 0,
) -> str:
    """Fit one configuration and score it on validation."""
    s = session()
    if (err := _budget_guard(s)) is not None:
        return err

    r = models.run(
        target=target, method=method, cheap_level=cheap_level, split=split,
        n_train=n_train, seed=seed, speed="fast", eval_on="validation", return_errors=True,
    )
    s.calls_used += 1
    eid = s._next_id()
    family = f"{target}:{split}"

    baselines = {}
    for m in ("cheap", "direct", "zero" if target.startswith("f") else "delta"):
        if m == method:
            continue
        baselines[m] = round(
            models.run(target=target, method=m, cheap_level=cheap_level, split=split,
                       seed=seed, speed="fast", eval_on="validation").mae,
            6,
        )

    prev = s.best.get(family)
    vs_best = None
    if prev is not None:
        vs_best = _paired(r.errors, prev[1])
        vs_best["against"] = prev[2]
        vs_best["best_so_far_mae"] = round(prev[0], 6)
    if prev is None or r.mae < prev[0]:
        s.best[family] = (r.mae, r.errors, eid)

    obs = {
        "experiment_id": eid,
        "config": r.config,
        "metric": {"mae": round(r.mae, 6), "unit": "eV" if target.startswith("E") else "a.u.",
                   "n_val": r.n_eval},
        "seed_variance": _noise_note(s.noise_scale(target, split)),
        "baselines": baselines,
        "vs_best": vs_best,
        "budget": {"calls_used": s.calls_used, "calls_left": s.budget - s.calls_used},
        "split_hash": s.split_hash(split),
        "fit_seconds": round(r.fit_seconds, 2),
    }
    s._log({**obs, "tool": "run_experiment", "t": time.time()})
    return json.dumps(obs, indent=1)


def slice_error(experiment_id: str, by: str = "state_gap", bins: int = 4) -> str:
    """Error conditioned on a molecular property. How you find WHERE a method breaks."""
    s = session()
    if (err := _budget_guard(s)) is not None:
        return err
    if by not in SLICE_BY:
        return f"ERROR: `by` must be one of {SLICE_BY}, got {by!r}."

    src = next((e for e in s.entries if e.get("experiment_id") == experiment_id), None)
    if src is None:
        return f"ERROR: no experiment with id {experiment_id!r}. Call run_experiment first."

    cfg = {k: v for k, v in src["config"].items() if k in
           ("target", "method", "cheap_level", "split", "seed")}
    r = models.run(**cfg, speed="fast", eval_on="validation", return_errors=True)
    s.calls_used += 1

    env, names = _env(), _env()["names"]
    parts = models._split(cfg["split"])
    pos = env["positions"][parts["validation"]]
    T = env["targets"]

    if by == "state_gap":
        v = T[pos, names.index(f"E2-{cfg['cheap_level']}")] - T[pos, names.index(f"E1-{cfg['cheap_level']}")]
        label = "cheap (TDDFT) E2-E1 gap, eV"
    elif by == "brightness":
        v = T[pos, names.index(f"f1-{cfg['cheap_level']}")]
        label = "cheap f1, a.u."
    elif by == "heavy_atoms":
        v = env["X"][pos, models.features.MORGAN_BITS + 1]
        label = "heavy atom count"
    elif by == "cheap_error":
        p = cfg["target"]
        v = np.abs(T[pos, names.index(f"{p}-{cfg['cheap_level']}")] - T[pos, names.index(f"{p}-CC2")])
        label = "|cheap - CC2|"
    else:  # train_distance
        Xv = env["X"][pos][:, : models.features.MORGAN_BITS]
        Xt = env["X"][env["positions"][parts["train"]]][:, : models.features.MORGAN_BITS]
        sub = Xt[np.random.default_rng(0).choice(len(Xt), size=min(2000, len(Xt)), replace=False)]
        inter = Xv @ sub.T
        union = Xv.sum(1)[:, None] + sub.sum(1)[None, :] - inter
        v = 1.0 - (inter / np.maximum(union, 1)).max(axis=1)
        label = "1 - max Tanimoto to training set"

    edges = np.quantile(v, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(bins):
        m = (v >= edges[i]) & (v < edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({
            "bin": i, "range": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
            "n": int(m.sum()), "mae": round(float(r.errors[m].mean()), 6),
        })

    obs = {"experiment_id": s._next_id(), "sliced": experiment_id, "by": by, "property": label,
           "overall_mae": round(r.mae, 6), "bins": out,
           "budget": {"calls_used": s.calls_used, "calls_left": s.budget - s.calls_used}}
    s._log({**obs, "tool": "slice_error", "config": src["config"], "t": time.time()})
    return json.dumps(obs, indent=1)


def run_control(experiment_id: str, control: str = "shuffle_labels") -> str:
    """Re-run where the effect must be ABSENT. Separates a finding from an artifact."""
    s = session()
    if (err := _budget_guard(s)) is not None:
        return err
    if control not in CONTROLS:
        return f"ERROR: `control` must be one of {CONTROLS}, got {control!r}."

    src = next((e for e in s.entries if e.get("experiment_id") == experiment_id), None)
    if src is None:
        return f"ERROR: no experiment with id {experiment_id!r}."
    cfg = {k: v for k, v in src["config"].items() if k in
           ("target", "method", "cheap_level", "split", "seed")}

    if control == "apply_to_energies":
        # the control that ruled out overfitting for the state-indexing result:
        # energies are sorted by construction, so a reindexing effect must vanish
        cfg2 = {**cfg, "target": "E1" if cfg["target"].startswith("f") else "f1"}
        r = models.run(**cfg2, speed="fast", eval_on="validation")
        detail = f"same configuration on {cfg2['target']} instead of {cfg['target']}"
        mae = r.mae
    elif control == "shuffle_labels":
        mae, detail = _shuffled_label_mae(cfg), "targets permuted; any skill here is leakage"
    else:  # random_reindex
        mae, detail = _reindex_mae(cfg, by="random"), "states reindexed at random"

    s.calls_used += 1
    obs = {"experiment_id": s._next_id(), "control_of": experiment_id, "control": control,
           "control_mae": round(float(mae), 6), "original_mae": src.get("metric", {}).get("mae"),
           "detail": detail,
           "budget": {"calls_used": s.calls_used, "calls_left": s.budget - s.calls_used}}
    s._log({**obs, "tool": "run_control", "config": src["config"], "t": time.time()})
    return json.dumps(obs, indent=1)


def _shuffled_label_mae(cfg: dict) -> float:
    env = _env()
    parts = models._split(cfg["split"])
    names = env["names"]
    rng = np.random.default_rng(12345)
    T = env["targets"].copy()
    col = names.index(f"{cfg['target']}-CC2")
    tr = env["positions"][parts["train"]]
    T[tr, col] = T[rng.permutation(tr), col]
    saved = env["targets"]
    try:
        models._CACHE["env"]["targets"] = T
        models._CACHE.pop(("fit", (cfg["target"], cfg["method"], cfg["cheap_level"],
                                   cfg["split"], None, cfg["seed"], "fast", "validation")), None)
        return models.run(**cfg, speed="fast", eval_on="validation").mae
    finally:
        models._CACHE["env"]["targets"] = saved
        models._CACHE.pop(("fit", (cfg["target"], cfg["method"], cfg["cheap_level"],
                                   cfg["split"], None, cfg["seed"], "fast", "validation")), None)


def _reindex_mae(cfg: dict, by: str = "brightness") -> float:
    """Swap (E1,f1) with (E2,f2) per molecule, then re-score.

    by="brightness": order the two states by cheap oscillator strength instead of
    by energy rank. by="random": the control -- a coin flip per molecule.
    """
    env = _env()
    names = env["names"]
    T = env["targets"].copy()
    lvl = cfg["cheap_level"]
    if by == "brightness":
        swap = T[:, names.index(f"f2-{lvl}")] > T[:, names.index(f"f1-{lvl}")]
    else:
        swap = np.random.default_rng(999).random(len(T)) < 0.5
    for prop_a, prop_b in (("E1", "E2"), ("f1", "f2")):
        for level in models.CHEAP_LEVELS + ["CC2"]:
            ia, ib = names.index(f"{prop_a}-{level}"), names.index(f"{prop_b}-{level}")
            a = T[swap, ia].copy()
            T[swap, ia] = T[swap, ib]
            T[swap, ib] = a
    saved = env["targets"]
    key = ("fit", (cfg["target"], cfg["method"], cfg["cheap_level"], cfg["split"],
                   None, cfg["seed"], "fast", "validation"))
    try:
        models._CACHE["env"]["targets"] = T
        models._CACHE.pop(key, None)
        return models.run(**cfg, speed="fast", eval_on="validation").mae
    finally:
        models._CACHE["env"]["targets"] = saved
        models._CACHE.pop(key, None)


def intervene(experiment_id: str, intervention: str = "reindex_states", value: str = "brightness") -> str:
    """Change one thing and re-measure."""
    s = session()
    if (err := _budget_guard(s)) is not None:
        return err
    if intervention not in INTERVENTIONS:
        return f"ERROR: `intervention` must be one of {INTERVENTIONS}, got {intervention!r}."

    src = next((e for e in s.entries if e.get("experiment_id") == experiment_id), None)
    if src is None:
        return f"ERROR: no experiment with id {experiment_id!r}."
    cfg = {k: v for k, v in src["config"].items() if k in
           ("target", "method", "cheap_level", "split", "seed")}
    before = src.get("metric", {}).get("mae")

    if intervention == "reindex_states":
        if value not in ("brightness", "energy"):
            return "ERROR: reindex_states takes value='brightness' or 'energy'."
        after = _reindex_mae(cfg, by=value) if value == "brightness" else \
            models.run(**cfg, speed="fast", eval_on="validation").mae
        detail = f"states ordered by {value} instead of energy rank"
    elif intervention == "ablate_features":
        if value not in features_blocks():
            return f"ERROR: ablate_features takes value in {list(features_blocks())}."
        after = _ablate_mae(cfg, value)
        detail = f"feature block {value!r} zeroed"
    else:  # subsample_train
        try:
            n = int(value)
        except ValueError:
            return "ERROR: subsample_train takes value as an integer training-set size."
        after = models.run(**cfg, n_train=n, speed="fast", eval_on="validation").mae
        detail = f"training set subsampled to {n}"

    s.calls_used += 1
    obs = {"experiment_id": s._next_id(), "intervention_on": experiment_id,
           "intervention": intervention, "value": value, "detail": detail,
           "mae_before": before, "mae_after": round(float(after), 6),
           "change": round(float(after) - before, 6) if before is not None else None,
           "budget": {"calls_used": s.calls_used, "calls_left": s.budget - s.calls_used}}
    s._log({**obs, "tool": "intervene", "config": src["config"], "t": time.time()})
    return json.dumps(obs, indent=1)


def features_blocks() -> dict:
    import features as F

    return F.FEATURE_BLOCKS


def _ablate_mae(cfg: dict, block: str) -> float:
    env = _env()
    lo, hi = features_blocks()[block]
    X = env["X"]
    saved = X[:, lo:hi].copy()
    key = ("fit", (cfg["target"], cfg["method"], cfg["cheap_level"], cfg["split"],
                   None, cfg["seed"], "fast", "validation"))
    try:
        X[:, lo:hi] = 0.0
        models._CACHE.pop(key, None)
        return models.run(**cfg, speed="fast", eval_on="validation").mae
    finally:
        X[:, lo:hi] = saved
        models._CACHE.pop(key, None)


def claim(statement: str, kind: str = "comparison", scope: dict | None = None,
          evidence: list[str] | None = None, config_a: dict | None = None,
          config_b: dict | None = None, control: str | None = None) -> str:
    """Submit a finding to the referee. This ends the run.

    Intercepted by agent.py before dispatch, so it doubles as the stopping rule.
    """
    s = session()
    s.claim = {
        "id": "claim_0", "statement": statement, "kind": kind, "scope": scope or {},
        "evidence": evidence or [], "config_a": config_a, "config_b": config_b,
        "control": control,
    }
    s._log({"tool": "claim", "claim": s.claim, "t": time.time()})
    return json.dumps({"submitted": s.claim}, indent=1)


TOOLS = {
    "describe_dataset": describe_dataset,
    "run_experiment": run_experiment,
    "slice_error": slice_error,
    "run_control": run_control,
    "intervene": intervene,
}


def _fn(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


TOOL_SCHEMAS = [
    _fn("describe_dataset",
        "What is in the QM8 environment: targets, units, levels of theory, split sizes, "
        "task correlations, and the distribution of oscillator strengths. Call this first.",
        {}, []),
    _fn("run_experiment",
        "Fit one configuration and score it on the validation set. Returns the MAE, the "
        "relevant baselines, the seed-to-seed noise scale, and a paired-bootstrap "
        "comparison against the best configuration you have found so far in this family. "
        "The comparison is the number that tells you whether an improvement is real.",
        {"target": {"type": "string", "enum": models.TARGETS,
                    "description": "E1/E2 are excitation energies (eV); f1/f2 oscillator strengths (a.u.)."},
         "method": {"type": "string", "enum": models.METHODS,
                    "description": "cheap = report TDDFT, no learning. direct = predict CC2 from "
                                   "structure. delta = predict CC2-TDDFT and add it back. zero = predict 0."},
         "cheap_level": {"type": "string", "enum": models.CHEAP_LEVELS,
                         "description": "Which TDDFT approximation to use as the cheap baseline."},
         "split": {"type": "string", "enum": models.SPLITS,
                   "description": "random; scaffold (structural shift); tddft_gap (trains on "
                                  "well-separated excited states, tests on near-degenerate ones)."},
         "n_train": {"type": "integer", "description": "Training-set size. Omit to use all of it."},
         "seed": {"type": "integer", "description": "Random seed (default 0)."}},
        ["target", "method"]),
    _fn("slice_error",
        "Break a finished experiment's error down by a molecular property. This is how you "
        "find WHERE a method fails rather than how much it fails on average.",
        {"experiment_id": {"type": "string", "description": "An id returned by run_experiment."},
         "by": {"type": "string", "enum": SLICE_BY,
                "description": "state_gap = cheap E2-E1 separation. brightness = cheap f1. "
                               "heavy_atoms = molecule size. train_distance = Tanimoto distance "
                               "to the training set. cheap_error = how wrong the cheap method is."},
         "bins": {"type": "integer", "description": "Number of quantile bins (default 4)."}},
        ["experiment_id", "by"]),
    _fn("run_control",
        "Re-run an experiment under a condition where the effect MUST be absent. A result "
        "without a control cannot be distinguished from an artifact, and the referee will "
        "reject a mechanistic claim that has no control behind it.",
        {"experiment_id": {"type": "string"},
         "control": {"type": "string", "enum": CONTROLS,
                     "description": "shuffle_labels = permute the targets; any skill left is "
                                    "leakage. apply_to_energies = run the same test on a quantity "
                                    "the effect should not touch. random_reindex = reorder states "
                                    "at random rather than by a rule."}},
        ["experiment_id", "control"]),
    _fn("intervene",
        "Change exactly one thing about an experiment and re-measure it.",
        {"experiment_id": {"type": "string"},
         "intervention": {"type": "string", "enum": INTERVENTIONS,
                          "description": "reindex_states = choose which excited state counts as "
                                         "state 1. ablate_features = zero a feature block. "
                                         "subsample_train = shrink the training set."},
         "value": {"type": "string",
                   "description": "reindex_states: 'brightness' or 'energy'. ablate_features: "
                                  "'morgan' or 'descriptors'. subsample_train: an integer."}},
        ["experiment_id", "intervention", "value"]),
    _fn("claim",
        "Submit one finding to the referee and END the run. The referee re-scores it on a "
        "sealed test set you have never seen, checks it against a paired bootstrap and a "
        "multiplicity correction, and verifies that your statement does not quantify over "
        "configurations you never ran. Claim narrowly and cite every experiment.",
        {"statement": {"type": "string", "description": "The finding, in one sentence."},
         "kind": {"type": "string", "enum": ["comparison", "mechanism", "value"],
                  "description": "comparison needs config_a and config_b; mechanism needs a control."},
         "scope": {"type": "object", "description": "e.g. {'target':'E1','split':'random'}. "
                                                    "Omit an axis you are not claiming about."},
         "evidence": {"type": "array", "items": {"type": "string"},
                      "description": "Experiment ids supporting the claim."},
         "config_a": {"type": "object", "description": "Baseline config (the one you say is worse)."},
         "config_b": {"type": "object", "description": "The config you say is better."},
         "control": {"type": "string", "description": "Control experiment id, for a mechanism claim."}},
        ["statement", "kind", "evidence"]),
]
