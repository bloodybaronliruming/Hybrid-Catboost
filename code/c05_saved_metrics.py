"""R1: recompute the five S32 test metrics from accepted saved predictions."""

import json
import math

import numpy as np

from c05_reproduction import INPUTS, PKG, SEEDS, new_run, numeric_metrics, prediction_rows, read_json, require, test_labels, verified_inputs, write_json


def main():
    verified_inputs()
    contract = read_json(PKG / "chemrxiv_submission/reproducibility/contract.json")["R1"]
    numerical_lock = read_json(PKG / "results/manuscript_metrics_lock.json")
    locked = numerical_lock["primary_full_precision"]
    native = numerical_lock["native_TDC"]["S32_CATBOOST"]
    ids = read_json(INPUTS / "features/test/row_ids.json")
    y = test_labels()
    per_seed = []
    checks = 0
    for seed in SEEDS:
        got_ids, predictions = prediction_rows(INPUTS / f"predictions/seed_{seed}.csv")
        require(got_ids == ids, f"Seed {seed} prediction rows differ")
        measured = numeric_metrics(y, predictions)
        for metric, value in measured.items():
            expected = locked[f"{metric}_per_seed"][seed - 1]
            require(math.isclose(value, expected, rel_tol=contract["metric_relative_tolerance"], abs_tol=contract["metric_absolute_tolerance"]), f"Seed {seed} {metric} differs")
            checks += 1
        per_seed.append({"seed": seed, "rows": len(got_ids), "metrics": measured})
    summary = {}
    for metric in ("MAE", "RMSE", "R2", "Spearman"):
        values = np.array([r["metrics"][metric] for r in per_seed])
        mean = float(np.mean(values))
        sample_sd = float(np.std(values, ddof=1))
        require(math.isclose(mean, locked[f"{metric}_mean"], abs_tol=contract["metric_absolute_tolerance"], rel_tol=contract["metric_relative_tolerance"]), f"{metric} mean differs")
        require(math.isclose(sample_sd, locked[f"{metric}_sample_sd"], abs_tol=contract["metric_absolute_tolerance"], rel_tol=contract["metric_relative_tolerance"]), f"{metric} SD differs")
        checks += 2
        summary[metric] = {"mean": mean, "sample_sd": sample_sd, "population_sd": float(np.std(values))}
    native_mae = [round(item["metrics"]["MAE"], 3) for item in per_seed]
    require(native_mae == native["individual_MAE_round3"], "Native per-run MAE display differs")
    checks += len(SEEDS)
    native_summary = [round(float(np.mean(native_mae)), 3), round(float(np.std(native_mae)), 3)]
    require(native_summary == native["mean_population_sd_round3"], "Native rounded mean/population SD differs")
    checks += 2
    run = new_run("r1_saved_metrics")
    report = {"status": "PASS", "level": "R1", "seeds": list(SEEDS), "checks": checks, "per_seed": per_seed, "summary": summary, "native_TDC_rounded_summary": native_summary, "contract": "chemrxiv_submission/reproducibility/contract.json", "input_manifest": "chemrxiv_submission/reproducibility/inputs/manifest.json"}
    write_json(run / "report.json", report)
    print(json.dumps({"status": "PASS", "level": "R1", "checks": checks, "run": str(run.relative_to(PKG))}))


if __name__ == "__main__":
    main()
