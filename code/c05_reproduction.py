"""Package-local, hash-gated reproduction of the five frozen S32 runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
INPUTS = PKG / "chemrxiv_submission/reproducibility/inputs"
RESULTS = PKG / "chemrxiv_submission/reproducibility/runs"
SEEDS = (1, 2, 3, 4, 5)
ACCEPTED_INPUT_MANIFEST_SHA256 = "b41ee008865e228c1ad6b2087e145379835d7be0fd59c155b09d12019d373811"
ACCEPTED_METRICS_LOCK_SHA256 = "5dcd42bd624a8f6feba48ec369e696e553e67ec149bdf1491824c7a185c9f80d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_inputs():
    require(sha(INPUTS / "manifest.json") == ACCEPTED_INPUT_MANIFEST_SHA256, "Input manifest provenance changed")
    require(sha(PKG / "results/manuscript_metrics_lock.json") == ACCEPTED_METRICS_LOCK_SHA256, "C04 numerical lock changed")
    manifest = read_json(INPUTS / "manifest.json")
    require(manifest["schema_version"] == 1, "Unknown input manifest")
    for relative, expected in manifest["sha256"].items():
        path = (INPUTS / relative).resolve()
        require(path.is_relative_to(INPUTS.resolve()) and path.is_file(), "Missing or unsafe input: " + relative)
        require(sha(path) == expected, "Input hash mismatch: " + relative)
    require(set(manifest["seeds"]) == set(SEEDS), "Incomplete seed set")
    return manifest


def load_matrix(seed, part, position=None):
    """Rebuild the exact frozen numerical view; no estimator fitting occurs."""
    import joblib
    import numpy as np

    job = read_json(INPUTS / f"jobs/seed_{seed}.json")
    state = read_json(INPUTS / f"states/seed_{seed}.json")
    feature = INPUTS / f"features/{part}"
    ids = read_json(feature / "row_ids.json")
    index = np.arange(len(ids)) if position is None else np.asarray(position, dtype=int)
    require(index.ndim == 1 and len(index) > 0, "Empty feature selection")
    require(np.all((index >= 0) & (index < len(ids))), "Invalid row positions")
    names = job["feature_names"]
    require(len(names) == job["feature_columns"] and len(set(names)) == len(names), "Invalid feature names")
    require(names[:2048] == [f"morgan_bit_{i:04d}" for i in range(2048)], "Morgan column order changed")
    m = np.load(feature / "morgan.npy", allow_pickle=False)[index]
    require(m.shape == (len(index), 2048) and m.dtype == np.uint8 and np.isin(m, [0, 1]).all(), "Invalid Morgan block")
    raw = np.load(feature / "rdkit2d.npy", allow_pickle=False)[index]
    require(raw.shape == (len(index), 210) and not np.isinf(raw).any(), "Invalid descriptor block")
    require(state["available_indices"] == job["preprocessing"]["available_indices"], "Available columns differ")
    # The hash gate must precede joblib deserialization; joblib can execute code.
    processor = joblib.load(INPUTS / f"processors/seed_{seed}.joblib")
    desc = processor[:-1].transform(raw[:, state["available_indices"]])
    require(desc.shape == (len(index), job["feature_columns"] - 2048), "Processed descriptor width differs")
    require(names[2048:] == state["output_feature_names"], "Descriptor column order changed")
    x = np.concatenate((m.astype(np.float64), desc), axis=1)
    require(job["feature_transform"]["descriptor"] == "Ipc" and job["feature_transform"]["operation"] == "log1p", "Unknown transform")
    column = names.index("Ipc")
    require(np.isfinite(x).all() and (x[:, column] >= 0).all(), "Invalid pre-transform features")
    x[:, column] = np.log1p(x[:, column])
    require(np.isfinite(x).all() and np.isfinite(x.astype(np.float32)).all(), "Nonfinite model features")
    matrix_hash = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()
    return job, [ids[i] for i in index], x, matrix_hash


def prediction_rows(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["row_id", "prediction_tdc_scale"], "Prediction columns changed")
        rows = list(reader)
    ids = [row["row_id"] for row in rows]
    values = [float(row["prediction_tdc_scale"]) for row in rows]
    require(len(ids) == 1997 and len(set(ids)) == len(ids), "Prediction rows missing or duplicated")
    require(all(math.isfinite(value) for value in values), "Nonfinite predictions")
    return ids, values


def numeric_metrics(labels, predictions):
    import numpy as np
    from scipy.stats import spearmanr

    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    require(y.shape == p.shape and y.ndim == 1 and np.isfinite(y).all() and np.isfinite(p).all(), "Invalid metric arrays")
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(1.0 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "Spearman": float(spearmanr(y, p).statistic),
    }


def calculate_smiles(ids, smiles):
    """Apply the original RDKit registry and fingerprint settings to new structures."""
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    require(len(ids) == len(smiles) and len(ids) > 0 and len(set(ids)) == len(ids), "Invalid input row identities")
    schema = read_json(PKG / "manifests/c03/feature_schema.json")
    names = schema["rdkit2d"]["feature_names"]
    require(names == [name for name, _ in Descriptors.descList], "RDKit descriptor registry differs")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False, useBondTypes=True, countSimulation=False, includeRingMembership=True, includeRedundantEnvironments=False)
    morgan = np.empty((len(ids), 2048), dtype=np.uint8)
    desc = np.empty((len(ids), 210), dtype=np.float64)
    for row, (identity, smiles_text) in enumerate(zip(ids, smiles)):
        require(isinstance(smiles_text, str) and smiles_text.strip() != "", "Empty SMILES: " + str(identity))
        mol = Chem.MolFromSmiles(smiles_text)
        require(mol is not None and mol.GetNumAtoms() > 0, "Invalid SMILES: " + str(identity))
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), morgan[row])
        for column, (_, function) in enumerate(Descriptors.descList):
            try:
                value = float(function(mol))
                desc[row, column] = value if math.isfinite(value) else math.nan
            except Exception:
                desc[row, column] = math.nan
        require(np.isfinite(desc[row]).any(), "No finite descriptors: " + str(identity))
    return morgan, desc


def matrix_from_smiles(seed, ids, smiles):
    """Serve new structures with each seed's frozen processor, with no row fallback."""
    import joblib
    import numpy as np

    job = read_json(INPUTS / f"jobs/seed_{seed}.json")
    state = read_json(INPUTS / f"states/seed_{seed}.json")
    morgan, raw = calculate_smiles(ids, smiles)
    processor = joblib.load(INPUTS / f"processors/seed_{seed}.joblib")
    desc = processor[:-1].transform(raw[:, state["available_indices"]])
    require(job["feature_names"][2048:] == state["output_feature_names"], "Descriptor column order changed")
    x = np.concatenate((morgan.astype(np.float64), desc), axis=1)
    require(x.shape == (len(ids), job["feature_columns"]), "Feature width changed")
    column = job["feature_names"].index("Ipc")
    require(np.isfinite(x).all() and (x[:, column] >= 0).all(), "Invalid descriptor value")
    x[:, column] = np.log1p(x[:, column])
    require(np.isfinite(x).all() and np.isfinite(x.astype(np.float32)).all(), "Nonfinite model features")
    return x


def test_labels():
    with (INPUTS / "test.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["Drug_ID", "Drug", "Y"], "Test columns changed")
        rows = list(reader)
    require(len(rows) == 1997, "Test row count changed")
    return [float(row["Y"]) for row in rows]


def new_run(prefix):
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RESULTS / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
