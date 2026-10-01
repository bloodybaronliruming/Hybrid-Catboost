"""R3: independently refit the five frozen GPU CatBoost jobs in a new run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from threadpoolctl import threadpool_limits

from c05_reproduction import INPUTS, PKG, SEEDS, load_matrix, new_run, numeric_metrics, read_json, require, test_labels, verified_inputs, write_json


def training_data(seed):
    job = read_json(INPUTS / f"jobs/seed_{seed}.json")
    frame = pd.read_csv(INPUTS / f"train/seed_{seed}.csv", float_precision="round_trip")
    members = read_json(INPUTS / f"members/seed_{seed}.json")["train"]
    require(len(frame) == job["train_rows"] and frame.row_id.tolist() == members, "Training row identity differs")
    require(frame.row_id.is_unique and frame.source_partition.eq("train_val").all(), "Invalid training partition")
    require(np.isfinite(frame.Y.to_numpy(dtype=np.float64)).all(), "Invalid training labels")
    feature_ids = read_json(INPUTS / "features/train_val/row_ids.json")
    positions = frame.source_row_index.to_numpy(dtype=int)
    require([feature_ids[i] for i in positions] == members, "Training feature row mapping differs")
    actual_job, ids, matrix, matrix_hash = load_matrix(seed, "train_val", positions)
    require(ids == members and actual_job == job, "Training configuration differs")
    original_hash = read_json(INPUTS / f"original_feature_views/seed_{seed}.json")["matrix_sha256"]
    require(matrix_hash == original_hash, "Training feature matrix differs")
    return job, matrix, frame.Y.to_numpy(dtype=np.float64), matrix_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "smoke", "full"), required=True)
    args = parser.parse_args()
    verified_inputs()
    contract = read_json(PKG / "chemrxiv_submission/reproducibility/contract.json")
    require(contract["R3"]["new_models"] == 5, "Rebuild contract changed")
    require(all(read_json(INPUTS / f"jobs/seed_{seed}.json")["parameters"]["task_type"] == "GPU" for seed in SEEDS), "Original GPU setting changed")
    if args.mode == "check":
        matrices = []
        for seed in SEEDS:
            _, matrix, labels, matrix_hash = training_data(seed)
            matrices.append({"seed": seed, "rows": len(labels), "columns": matrix.shape[1], "matrix_sha256": matrix_hash})
        available = get_gpu_device_count()
        print(json.dumps({"status": "READY" if available > 0 else "BLOCKED_NO_GPU", "mode": "check", "available_gpu_devices": available, "planned_fits": 5, "frozen_training_views": matrices}))
        return
    require(get_gpu_device_count() > 0, "GPU device unavailable; no CPU training fallback")
    run = new_run("r3_" + args.mode)
    report = {"status": "IN_PROGRESS", "level": "R3", "mode": args.mode, "fits_completed": 0, "seeds": [], "contract": "chemrxiv_submission/reproducibility/contract.json"}
    write_json(run / "plan.json", report)
    try:
        selected = SEEDS if args.mode == "full" else (1,)
        for seed in selected:
            job, matrix, labels, matrix_hash = training_data(seed)
            params = dict(job["parameters"])
            require(params["random_seed"] == seed and params["devices"] == "0" and params["iterations"] == 1000, "Frozen fit configuration differs")
            if args.mode == "smoke":
                params["iterations"] = 2
                train_x, train_y = matrix[:32], labels[:32]
            else:
                train_x, train_y = matrix, labels
            model = CatBoostRegressor(**params)
            with threadpool_limits(limits=1):
                model.fit(train_x, train_y)
            model_path = run / f"seed_{seed}.cbm"
            model.save_model(str(model_path))
            reloaded = CatBoostRegressor()
            reloaded.load_model(str(model_path))
            probe = reloaded.predict(train_x[: min(32, len(train_x))], task_type="CPU")
            require(np.isfinite(probe).all(), "Nonfinite training probe")
            report["fits_completed"] += 1
            report["seeds"].append({"seed": seed, "rows_fit": len(train_y), "matrix_sha256": matrix_hash, "model_path": model_path.name, "mode": args.mode})
        if args.mode == "full":
            # Held-out labels become accessible only after all five fits have finished.
            y_test = test_labels()
            locked = read_json(PKG / "results/manuscript_metrics_lock.json")["primary_full_precision"]
            maes = []
            test_ids = read_json(INPUTS / "features/test/row_ids.json")
            for record in report["seeds"]:
                seed = record["seed"]
                _, ids, x, _ = load_matrix(seed, "test")
                require(ids == test_ids, "Test row order differs")
                model = CatBoostRegressor()
                model.load_model(str(run / record["model_path"]))
                p = np.asarray(model.predict(x, task_type="CPU"), dtype=np.float64)
                require(p.shape == (1997,) and np.isfinite(p).all(), "Invalid rebuilt predictions")
                metrics = numeric_metrics(y_test, p)
                record["metrics"] = metrics
                record["mae_delta_from_original"] = float(metrics["MAE"] - locked["MAE_per_seed"][seed - 1])
                maes.append(metrics["MAE"])
                with (run / f"seed_{seed}_predictions.csv").open("x", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["row_id", "prediction_tdc_scale"])
                    writer.writerows(zip(ids, p))
            report["rebuild_mae_mean"] = float(np.mean(maes))
            report["mae_mean_delta_from_original"] = report["rebuild_mae_mean"] - locked["MAE_mean"]
            require(all(abs(record["mae_delta_from_original"]) <= contract["R3"]["mae_per_seed_max_absolute_difference"] for record in report["seeds"]), "Per-seed rebuild MAE tolerance failed")
            require(abs(report["mae_mean_delta_from_original"]) <= contract["R3"]["mae_mean_max_absolute_difference"], "Mean rebuild MAE tolerance failed")
        report["status"] = "PASS_SMOKE" if args.mode == "smoke" else "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        write_json(run / "failure.json", report)
        raise
    write_json(run / "report.json", report)
    print(json.dumps({"status": report["status"], "mode": args.mode, "fits_completed": report["fits_completed"], "run": str(run.relative_to(PKG))}))


if __name__ == "__main__":
    main()
