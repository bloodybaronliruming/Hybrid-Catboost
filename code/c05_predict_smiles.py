"""Predict a package-local row_id,Drug CSV with all five frozen S32 models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from c05_reproduction import INPUTS, PKG, SEEDS, matrix_from_smiles, require, verified_inputs


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["row_id", "Drug"], "Input columns must be exactly row_id,Drug in that order")
        rows = list(reader)
    require(len(rows) > 0 and all(row["row_id"] and row["Drug"] for row in rows), "Empty input row")
    ids = [row["row_id"] for row in rows]
    require(len(set(ids)) == len(ids), "Duplicate row_id")
    return ids, [row["Drug"] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV inside the package root")
    parser.add_argument("--output", required=True, type=Path, help="New CSV inside the package root")
    args = parser.parse_args()
    verified_inputs()
    input_path = (PKG / args.input).resolve()
    output_path = (PKG / args.output).resolve()
    require(input_path.is_relative_to(PKG) and output_path.is_relative_to(PKG), "Input and output must stay inside package")
    require(input_path.is_file() and not output_path.exists(), "Missing input or existing output")
    ids, smiles = read_rows(input_path)
    predictions = []
    for seed in SEEDS:
        x = matrix_from_smiles(seed, ids, smiles)
        model = CatBoostRegressor()
        model.load_model(str(INPUTS / f"models/seed_{seed}.cbm"))
        p = np.asarray(model.predict(x, task_type="CPU"), dtype=np.float64)
        require(p.shape == (len(ids),) and np.isfinite(p).all(), "Nonfinite or missing prediction")
        predictions.append(p)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "prediction_mean_tdc_scale", *[f"seed_{seed}" for seed in SEEDS]])
        for row, identity in enumerate(ids):
            writer.writerow([identity, float(np.mean([p[row] for p in predictions])), *[float(p[row]) for p in predictions]])
    print(json.dumps({"status": "PASS", "rows": len(ids), "output": str(output_path.relative_to(PKG))}))


if __name__ == "__main__":
    main()
