"""Build the QM8 environment — deterministically, from a seed.

Adapted in spirit from ArmLLM 2026 Day 4 `world/build.py`, which defines an
environment as five things: state, actions, transition, reward, reset. This file
owns *state* and *reset*. `tools.py` owns actions, `critic.py` owns reward.

    "reset -- regenerate from a seed, byte for byte"          Day 4, slide 46
    "No reset, no evaluation. And certainly no training."     Day 4, slide 34

Why this file exists at all: the earlier round of this project produced numbers
that were never written to disk. Nothing here is trusted unless this script can
regenerate it. Run it twice with the same seed and the manifest hashes must be
identical.

Two sources, and the reason for each:

  gdb8_22k_elec_spec.txt   the 2015 supplementary data, from the first author's
                           own repo. Carries ALL FOUR levels of theory.
  qm8.csv                  MoleculeNet's redistribution. Carries the SMILES,
                           which the raw release does not.

MoleculeNet's file is NOT used for any target value, because it is damaged: its
two PBE0 blocks are byte-identical across all 21,786 molecules, so it ships 12
distinct tasks under 16 column headers. The genuine PBE0/def2TZVP level is
recoverable only from the raw release. See check_moleculenet_duplication().

Because the raw release has no SMILES and MoleculeNet has no fourth level, the
two must be joined by row order -- and that join is *verified*, not assumed, by
comparing the columns the two files share.

    python world/build.py --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

HARTREE_EV = 27.211386245988  # CODATA 2018

SPEC_URL = (
    "https://raw.githubusercontent.com/raghurama123/ExcitedStatesQM8/main/"
    "22k_electronic_spectra_TDDFT_CC2/gdb8_22k_elec_spec.txt"
)
SMILES_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm8.csv"

SPEC_SHA256 = "9830940d7bf02f92f1b0ae7e00851652b08fdd81c31b0f61950e648e59fc2ea0"

N_MOLECULES = 21786

# The four levels of theory, in the column order of the raw file.
# CC2 is the expensive reference; the other three are the cheap approximations.
LEVELS = ["CC2", "PBE0-SVP", "PBE0-TZVP", "CAM"]
PROPS = ["E1", "E2", "f1", "f2"]

# Energies are hartree in the raw file and get converted to eV. Oscillator
# strengths are dimensionless (atomic units, length representation) and must NOT
# be converted -- mixing the two is how an averaged MAE over 16 targets becomes
# chemically meaningless.
IS_ENERGY = {"E1": True, "E2": True, "f1": False, "f2": False}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Hash an array by value, independent of memory layout."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# raw parsing
# --------------------------------------------------------------------------


def load_raw_spectra(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (gdb9_index[N], targets[N, 16]) in the raw file's own units (hartree/a.u.)."""
    idx, rows = [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 17:
                raise ValueError(f"expected 17 fields, got {len(parts)}: {s[:80]!r}")
            idx.append(int(parts[0]))
            rows.append([float(x) for x in parts[1:]])
    index = np.asarray(idx, dtype=np.int64)
    targets = np.asarray(rows, dtype=np.float64)
    if len(index) != N_MOLECULES:
        raise ValueError(f"expected {N_MOLECULES} molecules, parsed {len(index)}")
    return index, targets


def load_moleculenet(path: Path) -> tuple[list[str], np.ndarray]:
    """-> (smiles[N], targets[N, 16]). Used for SMILES only; targets are for the join check."""
    import csv

    smiles, rows = [], []
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader)  # header carries duplicate names; positional access only
        for row in reader:
            if not row:
                continue
            smiles.append(row[0])
            rows.append([float(x) for x in row[1:17]])
    return smiles, np.asarray(rows, dtype=np.float64)


def target_names() -> list[str]:
    return [f"{p}-{lvl}" for lvl in LEVELS for p in PROPS]


def to_ev(targets: np.ndarray) -> np.ndarray:
    """Convert energy columns to eV; leave oscillator strengths alone."""
    out = targets.copy()
    for j, name in enumerate(target_names()):
        if IS_ENERGY[name.split("-")[0]]:
            out[:, j] *= HARTREE_EV
    return out


# --------------------------------------------------------------------------
# checks -- these are the point of the file
# --------------------------------------------------------------------------


def check_moleculenet_duplication(mn_targets: np.ndarray) -> dict:
    """MoleculeNet's two PBE0 blocks are identical. Quantify it rather than assert it."""
    a, b = mn_targets[:, 4:8], mn_targets[:, 8:12]
    max_abs = float(np.abs(a - b).max())
    return {
        "identical_rows": int((np.abs(a - b).max(axis=1) == 0).sum()),
        "n_rows": int(len(mn_targets)),
        "max_abs_difference": max_abs,
        "distinct_tasks": 12 if max_abs == 0.0 else 16,
        "verdict": (
            "MoleculeNet qm8.csv ships 12 distinct tasks under 16 column headers; "
            "PBE0/def2TZVP is a verbatim copy of PBE0/def2SVP"
            if max_abs == 0.0
            else "blocks differ -- MoleculeNet may have been corrected upstream"
        ),
    }


def check_row_alignment(raw_targets: np.ndarray, mn_targets: np.ndarray) -> dict:
    """Prove the SMILES join is sound by comparing the columns both files share.

    Both files carry CC2 (cols 0-3), PBE0/def2SVP (4-7) and CAM (12-15). If those
    agree row-for-row, MoleculeNet's row i is the raw file's row i, and its SMILES
    can be attached. The PBE0/def2TZVP block is deliberately excluded -- that is
    the damaged one.
    """
    shared = list(range(0, 8)) + list(range(12, 16))
    diffs = np.abs(raw_targets[:, shared] - mn_targets[:, shared])
    max_abs = float(diffs.max())
    # the raw file has 8 decimals, MoleculeNet rounds some CAM values
    aligned = max_abs < 1e-6
    return {
        "columns_compared": [target_names()[j] for j in shared],
        "max_abs_difference": max_abs,
        "rows_exact": int((diffs.max(axis=1) == 0).sum()),
        "aligned": bool(aligned),
    }


def check_parse_against_paper(targets_ev: np.ndarray) -> dict:
    """Ramakrishnan et al. 2015 report PBE0/def2SVP vs CC2 at 0.27 / 0.37 eV.

    Reproducing those two numbers from our own parse is what proves the units,
    the column mapping and the reference assignment are right. If this drifts,
    nothing downstream means anything.
    """
    names = target_names()
    out = {}
    for prop, expected in (("E1", 0.27), ("E2", 0.37)):
        cc2 = targets_ev[:, names.index(f"{prop}-CC2")]
        pbe0 = targets_ev[:, names.index(f"{prop}-PBE0-SVP")]
        mae = float(np.abs(pbe0 - cc2).mean())
        out[prop] = {
            "mae_ev": round(mae, 4),
            "paper_ev": expected,
            "within_0.01": bool(abs(mae - expected) < 0.01),
        }
    out["passed"] = all(v["within_0.01"] for k, v in out.items() if k != "passed")
    return out


# --------------------------------------------------------------------------
# molecules
# --------------------------------------------------------------------------


def sanitize(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """-> (ok_mask, canonical_smiles). Records failures instead of silently dropping.

    DeepChem's loader drops the failures and returns 21,747. We keep all 21,786
    rows and carry a mask, so the count never changes underneath a result.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    ok = np.zeros(len(smiles), dtype=bool)
    canon = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canon.append("")
            continue
        ok[i] = True
        canon.append(Chem.MolToSmiles(mol))
    return ok, canon


def murcko_scaffolds(canon: list[str], ok: np.ndarray) -> list[str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out = []
    for i, smi in enumerate(canon):
        if not ok[i]:
            out.append("")
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False))
        except Exception:
            out.append("")
    return out


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def three_way_random(n_ok: int, seed: int, frac=(0.80, 0.10, 0.10)) -> dict[str, np.ndarray]:
    """Train / validation / sealed test.

    The agent sees validation. Only the critic ever scores against sealed test.
    Two sets rather than one is the whole defence against an agent that optimises
    against the number it is judged on, one scalar at a time, over thirty calls.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ok)
    n_train = int(round(frac[0] * n_ok))
    n_val = int(round(frac[1] * n_ok))
    return {
        "train": np.sort(perm[:n_train]),
        "validation": np.sort(perm[n_train : n_train + n_val]),
        "sealed_test": np.sort(perm[n_train + n_val :]),
    }


def three_way_scaffold(
    scaffolds: list[str], positions: np.ndarray, seed: int, frac=(0.80, 0.10, 0.10)
) -> dict[str, np.ndarray]:
    """Scaffold split, largest groups into train (the standard greedy construction).

    Recorded here without endorsement: at <=8 heavy atoms Bemis-Murcko scaffolds
    are near-degenerate, so this split is much weaker than it looks. The
    diagnostics in profile() quantify that, and a TDDFT-gap split is built
    separately as the honest distribution-shift protocol.
    """
    groups: dict[str, list[int]] = {}
    for local, pos in enumerate(positions):
        groups.setdefault(scaffolds[pos], []).append(local)
    # deterministic: size desc, then scaffold string
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n = len(positions)
    n_train, n_val = int(round(frac[0] * n)), int(round(frac[1] * n))
    out = {"train": [], "validation": [], "sealed_test": []}
    for _, members in ordered:
        if len(out["train"]) + len(members) <= n_train:
            out["train"].extend(members)
        elif len(out["validation"]) + len(members) <= n_val:
            out["validation"].extend(members)
        else:
            out["sealed_test"].extend(members)
    return {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in out.items()}


def three_way_tddft_gap(
    targets_ev: np.ndarray, positions: np.ndarray, frac=(0.80, 0.10, 0.10)
) -> dict[str, np.ndarray]:
    """Split on the *cheap* E2-E1 gap: train on well-separated states, test on
    near-degenerate ones.

    The CC2 gap would be the natural quantity, and it is the wrong one: it is only
    available after paying the cost the whole project exists to avoid. The TDDFT
    gap is computed for every molecule in the screening set already, so a model
    can actually be deployed behind this split.
    """
    names = target_names()
    e1 = targets_ev[positions, names.index("E1-PBE0-SVP")]
    e2 = targets_ev[positions, names.index("E2-PBE0-SVP")]
    gap = e2 - e1
    order = np.argsort(-gap, kind="stable")  # widest gap first -> train
    n = len(positions)
    n_train, n_val = int(round(frac[0] * n)), int(round(frac[1] * n))
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_val]),
        "sealed_test": np.sort(order[n_train + n_val :]),
    }


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def profile(targets_ev: np.ndarray, positions: np.ndarray, scaffolds: list[str]) -> dict:
    names = target_names()
    sub = targets_ev[positions]

    energy_cols = [j for j, nm in enumerate(names) if nm.startswith(("E1", "E2"))]
    osc_cols = [j for j, nm in enumerate(names) if nm.startswith(("f1", "f2"))]
    corr = np.corrcoef(sub.T)

    def mean_abs_offdiag(cols):
        block = np.abs(corr[np.ix_(cols, cols)])
        m = ~np.eye(len(cols), dtype=bool)
        return float(block[m].mean())

    cross = np.abs(corr[np.ix_(energy_cols, osc_cols)])

    scaf = [scaffolds[p] for p in positions]
    uniq: dict[str, int] = {}
    for s in scaf:
        uniq[s] = uniq.get(s, 0) + 1
    top20 = sorted(uniq.values(), reverse=True)[:20]

    # oscillator strengths are non-negative and heavily right-skewed; MAE on them
    # rewards predicting ~0, which is why "delta-learning fails for f" needs a
    # predict-zero baseline before it can be believed.
    f1 = sub[:, names.index("f1-CC2")]
    return {
        "n": int(len(positions)),
        "mean_abs_r_within_energies": round(mean_abs_offdiag(energy_cols), 3),
        "mean_abs_r_within_oscillator": round(mean_abs_offdiag(osc_cols), 3),
        "mean_abs_r_across_blocks": round(float(cross.mean()), 3),
        "scaffolds": {
            "unique": len(uniq),
            "acyclic_fraction": round(float(uniq.get("", 0) / len(scaf)), 4),
            "top20_coverage": round(float(sum(top20) / len(scaf)), 4),
            "singletons": int(sum(1 for v in uniq.values() if v == 1)),
        },
        "f1_cc2": {
            "mean": round(float(f1.mean()), 5),
            "median": round(float(np.median(f1)), 5),
            "frac_below_0.001": round(float((f1 < 1e-3).mean()), 4),
            "predict_zero_mae": round(float(np.abs(f1).mean()), 5),
            "note": "predict-zero MAE is the baseline any f-model must beat",
        },
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def ensure_raw(offline: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in ((RAW / "gdb8_22k_elec_spec.txt", SPEC_URL), (RAW / "qm8.csv", SMILES_URL)):
        if path.exists():
            continue
        if offline:
            raise SystemExit(f"missing {path} and --offline was given")
        import urllib.request

        print(f"  downloading {path.name} ...", flush=True)
        urllib.request.urlretrieve(url, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print(f"building QM8 environment  seed={args.seed}")
    ensure_raw(args.offline)

    spec_path, mn_path = RAW / "gdb8_22k_elec_spec.txt", RAW / "qm8.csv"
    spec_hash = sha256_file(spec_path)
    if spec_hash != SPEC_SHA256:
        print(f"  WARNING: raw spectra hash changed\n    expected {SPEC_SHA256}\n    got      {spec_hash}")

    index, raw_targets = load_raw_spectra(spec_path)
    smiles, mn_targets = load_moleculenet(mn_path)
    print(f"  parsed {len(index)} molecules, {raw_targets.shape[1]} targets")

    dup = check_moleculenet_duplication(mn_targets)
    print(f"  moleculenet: {dup['distinct_tasks']} distinct tasks "
          f"({dup['identical_rows']}/{dup['n_rows']} PBE0 rows identical)")

    align = check_row_alignment(raw_targets, mn_targets)
    if not align["aligned"]:
        raise SystemExit(f"row alignment failed, max|diff| = {align['max_abs_difference']}")
    print(f"  smiles join verified on {len(align['columns_compared'])} shared columns "
          f"(max|diff| = {align['max_abs_difference']:.2e})")

    targets_ev = to_ev(raw_targets)
    paper = check_parse_against_paper(targets_ev)
    print(f"  parse check vs 2015 paper: E1 {paper['E1']['mae_ev']} eV (paper 0.27), "
          f"E2 {paper['E2']['mae_ev']} eV (paper 0.37) -> "
          f"{'PASS' if paper['passed'] else 'FAIL'}")
    if not paper["passed"]:
        raise SystemExit("parse does not reproduce the paper; refusing to build")

    ok, canon = sanitize(smiles)
    positions = np.flatnonzero(ok)
    print(f"  rdkit sanitize: {ok.sum()} ok, {(~ok).sum()} failed "
          f"(deepchem drops these and reports 21747)")

    scaffolds = murcko_scaffolds(canon, ok)

    splits = {
        "random": three_way_random(len(positions), args.seed),
        "scaffold": three_way_scaffold(scaffolds, positions, args.seed),
        "tddft_gap": three_way_tddft_gap(targets_ev, positions),
    }

    DERIVED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DERIVED / "qm8.npz",
        gdb9_index=index,
        targets_ev=targets_ev,
        ok=ok,
        positions=positions,
        smiles=np.array(canon, dtype=object),
        scaffolds=np.array(scaffolds, dtype=object),
        target_names=np.array(target_names(), dtype=object),
    )

    split_hashes = {}
    for name, parts in splits.items():
        np.savez_compressed(DERIVED / f"split_{name}.npz", **parts)
        split_hashes[name] = {
            k: {"n": int(len(v)), "sha256": sha256_array(v)} for k, v in parts.items()
        }

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "units": {"energies": "eV (converted from hartree)", "oscillator_strengths": "a.u."},
        "hartree_to_ev": HARTREE_EV,
        "sources": {
            "spectra": {"url": SPEC_URL, "sha256": spec_hash},
            "smiles": {"url": SMILES_URL, "sha256": sha256_file(mn_path)},
        },
        "n_molecules": int(len(index)),
        "n_usable": int(ok.sum()),
        "target_names": target_names(),
        "checks": {
            "moleculenet_duplication": dup,
            "row_alignment": align,
            "parse_vs_paper_2015": paper,
        },
        "splits": split_hashes,
        "data_sha256": {
            "targets_ev": sha256_array(targets_ev),
            "positions": sha256_array(positions),
        },
        "profile": profile(targets_ev, positions, scaffolds),
    }
    (DERIVED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  wrote {DERIVED}/qm8.npz, 3 splits, manifest.json")
    print(f"  environment hash: {manifest['data_sha256']['targets_ev'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
