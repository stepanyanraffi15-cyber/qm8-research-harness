"""Featurization: Morgan fingerprints plus a named descriptor block.

Deliberately modest. The prediction layer is gradient boosting on fingerprints,
chosen for throughput (a fit in seconds, so an agent can run twenty experiments
in a session) rather than for accuracy. Published QM8 leaders are GNNs and would
beat this.

That choice has a consequence which is measured rather than hidden: fingerprints
discard the 3D geometry QM8 ships, and geometry determines most of an excitation
energy. The delta-model barely notices, because it only has to learn a smooth
correction to an already-physical number. The direct model is crippled by it.
Any comparison between the two is therefore partly a statement about this
featurization, and models.py reports it that way.

The earlier design said "12 RDKit descriptors" without ever naming them, which
is not reproducible. They are named here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"

MORGAN_RADIUS = 2
MORGAN_BITS = 1024

# 12 descriptors, cheap to compute and spanning size, polarity, flexibility and
# aromaticity -- the axes a fingerprint alone represents poorly.
DESCRIPTORS = [
    "MolWt",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "TPSA",
    "MolLogP",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "NumHeteroatoms",
    "BertzCT",
]

FEATURE_BLOCKS = {"morgan": (0, MORGAN_BITS), "descriptors": (MORGAN_BITS, MORGAN_BITS + len(DESCRIPTORS))}


def _descriptor_fns():
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    return {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomCount": lambda m: float(m.GetNumHeavyAtoms()),
        "NumHAcceptors": lambda m: float(Lipinski.NumHAcceptors(m)),
        "NumHDonors": lambda m: float(Lipinski.NumHDonors(m)),
        "NumRotatableBonds": lambda m: float(Lipinski.NumRotatableBonds(m)),
        "TPSA": rdMolDescriptors.CalcTPSA,
        "MolLogP": Crippen.MolLogP,
        "RingCount": lambda m: float(rdMolDescriptors.CalcNumRings(m)),
        "NumAromaticRings": lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NumHeteroatoms": lambda m: float(rdMolDescriptors.CalcNumHeteroatoms(m)),
        "BertzCT": lambda m: float(Descriptors.BertzCT(m)),
    }


def build(smiles: list[str]) -> np.ndarray:
    """-> X[n, 1024 + 12], float32."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fns = _descriptor_fns()

    X = np.zeros((len(smiles), MORGAN_BITS + len(DESCRIPTORS)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        X[i, :MORGAN_BITS] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        for j, name in enumerate(DESCRIPTORS):
            try:
                X[i, MORGAN_BITS + j] = float(fns[name](mol))
            except Exception:
                X[i, MORGAN_BITS + j] = 0.0
    return X


def load_or_build(force: bool = False) -> np.ndarray:
    """Cached featurization. Hash goes in the manifest so a claim can name it."""
    cache = DERIVED / "features.npz"
    if cache.exists() and not force:
        return np.load(cache)["X"]

    data = np.load(DERIVED / "qm8.npz", allow_pickle=True)
    smiles = [str(s) for s in data["smiles"]]
    X = build(smiles)
    np.savez_compressed(cache, X=X)
    return X


def feature_hash(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


if __name__ == "__main__":
    X = load_or_build(force=True)
    print(f"features {X.shape}  dtype={X.dtype}")
    print(f"  morgan     r={MORGAN_RADIUS} bits={MORGAN_BITS}  density={X[:, :MORGAN_BITS].mean():.4f}")
    print(f"  descriptors {len(DESCRIPTORS)}: {', '.join(DESCRIPTORS)}")
    print(f"  sha256 {feature_hash(X)[:16]}")
