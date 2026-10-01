"""R2: load hash-verified frozen CatBoost models and replay five test predictions."""

import csv
import json

import numpy as np
from catboost import CatBoostRegressor

from c05_reproduction import INPUTS, PKG, SEEDS, load_matrix, new_run, numeric_metrics, prediction_rows, read_json, require, test_labels, verified_inputs, write_json


def main():
    verified_inputs()
    contract = read_json(PKG / "chemrxiv_submission/reproducibility/contract.json")["R2"]
    lock = read_json(PKG / "results/manuscript_metrics_lock.json")["primary_full_precision"]
    labels = test_labels()
    run = new_run("r2_frozen_inference")
    results = []
    for seed in SEEDS:
        job, ids, x, matrix_hash = load_matrix(seed, "test")
        original_view = read_json(INPUTS / f"original_blind_views/seed_{seed}.json")
        require(matrix_hash == original_view["matrix_sha256"], f"Seed {seed} feature matrix differs")
        model = CatBoostRegressor()
        model.load_model(str(INPUTS / f"models/seed_{seed}.cbm"))
        predicted = np.asarray(model.predict(x, task_type="CPU"), dtype=np.float64)
        require(predicted.shape == (1997,) and np.isfinite(predicted).all(), f"Seed {seed} invalid predictions")
        original_ids, original = prediction_rows(INPUTS / f"predictions/seed_{seed}.csv")
        require(ids == original_ids, f"Seed {seed} row order differs")
        reference = np.asarray(original, dtype=np.float64)
        delta = np.abs(predicted - reference)
        relative = delta / np.maximum(np.abs(reference), 1e-8)
        max_absolute = float(np.max(delta))
        max_relative = float(np.max(relative[np.abs(reference) >= 1e-8]))
        require(max_absolute <= contract["prediction_max_absolute_difference"] and max_relative <= contract["prediction_max_relative_difference_above_1e-8"], f"Seed {seed} prediction tolerance failed")
        metrics = numeric_metrics(labels, predicted)
        for key, value in metrics.items():
            require(abs(value - lock[f"{key}_per_seed"][seed - 1]) <= contract["metric_absolute_tolerance"], f"Seed {seed} {key} differs")
        output = run / f"seed_{seed}_predictions.csv"
        with output.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "prediction_tdc_scale"])
            writer.writerows(zip(ids, predicted))
        results.append({"seed": seed, "rows": len(ids), "feature_matrix_sha256": matrix_hash, "max_absolute_prediction_difference": max_absolute, "max_relative_prediction_difference": max_relative, "metrics": metrics})
    report = {"status": "PASS", "level": "R2", "device": "CPU", "seeds": results, "contract": "chemrxiv_submission/reproducibility/contract.json", "input_manifest": "chemrxiv_submission/reproducibility/inputs/manifest.json"}
    write_json(run / "report.json", report)
    print(json.dumps({"status": "PASS", "level": "R2", "run": str(run.relative_to(PKG)), "maximum_absolute_difference": max(item["max_absolute_prediction_difference"] for item in results)}))


if __name__ == "__main__":
    main()
