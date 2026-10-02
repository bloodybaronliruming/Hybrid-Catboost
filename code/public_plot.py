"""Build manuscript tables and plots from frozen, verified study records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import LogFormatterMathtext
import numpy as np


PKG = Path(__file__).resolve().parents[1]
ROOT = PKG
OUT = FIG = TAB = LOCK = None
SRC = {}

GROUPS = [
    ("S00_MEAN", "Mean", "constant mean", "none"),
    ("S01_MEDIAN", "Median", "constant median", "none"),
    ("S11_RF", "Morgan RF", "random forest", "Morgan bits"),
    ("S12_CATBOOST", "Morgan CatBoost", "CatBoost", "Morgan bits"),
    ("S20_RIDGE", "RDKit2D Ridge", "ridge regression", "RDKit2D descriptors"),
    ("S21_RF", "RDKit2D RF", "random forest", "RDKit2D descriptors"),
    ("S22_CATBOOST", "RDKit2D CatBoost", "CatBoost", "RDKit2D descriptors"),
    ("S31_RF", "Hybrid RF", "random forest", "Morgan + RDKit2D"),
    ("S32_CATBOOST", "Hybrid-CatBoost", "CatBoost GPU Plain", "Morgan + RDKit2D"),
    ("S40_CHEMPROP", "D-MPNN", "Chemprop 2.x D-MPNN", "molecular graph"),
]
LABEL = {item[0]: item[1] for item in GROUPS}
COLOR = {"Mean": "#7A7A7A", "Median": "#7A7A7A", "Morgan RF": "#0072B2",
         "Morgan CatBoost": "#0072B2", "RDKit2D Ridge": "#D55E00", "RDKit2D RF": "#D55E00",
         "RDKit2D CatBoost": "#D55E00", "Hybrid RF": "#009E73",
         "Hybrid-CatBoost": "#009E73", "D-MPNN": "#CC79A7"}
MARKER = {"Mean": "o", "Median": "o", "Morgan RF": "D", "Morgan CatBoost": "h",
          "RDKit2D Ridge": "s", "RDKit2D RF": "D", "RDKit2D CatBoost": "h",
          "Hybrid RF": "D", "Hybrid-CatBoost": "h", "D-MPNN": "^"}
WIDTH = 180 / 25.4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decorate(ax, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", color="#D9D9D9", lw=0.5, zorder=0)


def panel(ax, label: str) -> None:
    ax.text(-0.04, 1.02, label, transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=10)


def save(fig, name: str) -> dict:
    paths = {}
    for extension in ("pdf", "svg", "png"):
        path = FIG / f"{name}.{extension}"
        fig.savefig(path, dpi=600, facecolor="white", bbox_inches="tight")
        paths[extension] = {"path": str(path.relative_to(PKG)), "sha256": digest(path)}
    plt.close(fig)
    return paths


def render(folder) -> None:
    global OUT, FIG, TAB, LOCK, SRC
    OUT = folder
    FIG = folder / 'figures'
    TAB = folder / 'tables'
    LOCK = folder / 'data/plot_metrics.json'
    SRC = {'lock': LOCK,
           'per_seed': folder / 'data/per_seed_metrics.csv',
           'validation_summary': folder / 'data/validation_summary.csv',
           'paired': folder / 'data/paired_validation_summary.csv',
           'compound': folder / 'data/final_compound_summary.csv',
           'quartile_summary': folder / 'data/target_range_performance_summary.csv',
           'quartile_thresholds': folder / 'data/target_range_thresholds.csv'}
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    frozen = json.loads(LOCK.read_text())
    assert frozen["status"] == "COMPUTED"
    assert frozen["public_model_name"] == "Hybrid-CatBoost"
    assert frozen["internal_group"] == "S32_CATBOOST"
    assert frozen["seeds"] == [1, 2, 3, 4, 5]
    assert frozen["test_rows_per_run"] == 1997
    assert len(frozen["all_groups_full_precision"]) == 10
    assert len(frozen["paired_validation_summary"]) == 7
    assert all(path.exists() for path in SRC.values())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    val = {x["experiment"]: x for x in read_csv(SRC["validation_summary"])}
    test = {x["experiment"]: x for x in frozen["all_groups_full_precision"]}
    assert set(val) == set(test) == set(LABEL)
    pair = frozen["paired_validation_summary"]
    runs = read_csv(SRC["per_seed"])
    assert len(runs) == 50 and all(int(x["rows"]) == 1997 for x in runs)

    # Tables preserve the full serialized precision of the frozen numeric lock.
    split_rows = [
        {"seed": i, "train_records": train, "validation_records": valid,
         "fixed_test_records": 1997, "dataset": "Solubility_AqSolDB",
         "target": "log10(S / (1 mol/L))"}
        for i, (train, valid) in enumerate([(6986, 999)] * 4 + [(5045, 2940)], start=1)
    ]
    write_csv(TAB / "table_01_dataset_splits.csv", list(split_rows[0]), split_rows)
    recipe_rows = [{"internal_id": code, "public_label": label, "learner": learner,
                    "representation": rep, "runs": 5} for code, label, learner, rep in GROUPS]
    write_csv(TAB / "table_02_model_inventory.csv", list(recipe_rows[0]), recipe_rows)
    metrics = []
    for code, label, _, _ in GROUPS:
        a, b = val[code], test[code]
        metrics.append({"public_label": label, "internal_id": code, "runs": 5,
                        "validation_MAE_mean": a["MAE_mean"],
                        "validation_MAE_sample_sd": a["MAE_sample_sd"],
                        **{f"test_{metric}_{stat}": b[f"{metric}_{stat}"]
                           for metric in ("MAE", "RMSE", "R2", "Spearman")
                           for stat in ("mean", "sample_sd")}})
    write_csv(TAB / "table_03_validation_test_metrics.csv", list(metrics[0]), metrics)
    pairs = [{"comparison": f"{LABEL[x['comparison']]} minus {LABEL[x['baseline']]}",
              "baseline": LABEL[x["baseline"]], "comparison_model": LABEL[x["comparison"]],
              "runs": x["n_runs"], "MAE_delta_mean": x["MAE_delta_mean"],
              "MAE_delta_sample_sd": x["MAE_delta_sample_sd"],
              "MAE_improved_seeds": x["MAE_improved_seeds"]} for x in pair]
    write_csv(TAB / "table_04_paired_validation.csv", list(pairs[0]), pairs)
    seed_rows = [{"public_label": LABEL[x["experiment"]], "internal_id": x["experiment"],
                  "seed": x["seed"], "test_records": x["rows"],
                  **{k: x[k] for k in ("MAE", "RMSE", "R2", "Spearman", "Spearman_reason")}}
                 for x in runs]
    write_csv(TAB / "table_s01_all_seed_metrics.csv", list(seed_rows[0]), seed_rows)
    quart = read_csv(SRC["quartile_summary"])
    selected = {"S21_RF", "S22_CATBOOST", "S31_RF", "S32_CATBOOST", "S40_CHEMPROP"}
    quart_rows = [{"public_label": LABEL[x["experiment"]], "internal_id": x["experiment"], **x}
                  for x in quart if x["experiment"] in selected]
    write_csv(TAB / "table_s02_target_quartiles.csv", list(quart_rows[0]), quart_rows)
    thresholds = read_csv(SRC["quartile_thresholds"])
    assert len(thresholds) == 5
    write_csv(TAB / "table_s03_train_quartile_thresholds.csv", list(thresholds[0]), thresholds)

    figure_files = {}
    # Render the accepted diagram geometry directly from public source.
    from public_pipeline_figure import render_pipeline
    figure_files['figure_01_actual_pipeline'] = render_pipeline(FIG)

    # Full ten-group result: ordinary models and the five extreme Ridge runs remain distinct.
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 6.7),
                              gridspec_kw={"height_ratios": [3.0, 1.5]}, layout="constrained")
    regular = [(code, label) for code, label, _, _ in GROUPS if code != "S20_RIDGE"]
    for pos, (code, label) in enumerate(regular):
        row = test[code]; values = row["MAE_per_seed"]
        a.scatter(values, np.repeat(pos, 5), marker=MARKER[label], s=23, alpha=.7,
                  color=COLOR[label], zorder=3)
        a.errorbar(row["MAE_mean"], pos, xerr=row["MAE_sample_sd"],
                   fmt="D", markersize=4, capsize=2, color="#202020", zorder=4)
    a.set_yticks(np.arange(len(regular)), [x[1] for x in regular]); a.invert_yaxis()
    a.set_xlim(left=0); decorate(a, "Fixed-test MAE (TDC logS)")
    panel(a, "a")
    ridge = test["S20_RIDGE"]
    b.scatter(ridge["MAE_per_seed"], range(1, 6), color=COLOR["RDKit2D Ridge"], marker="s", s=24)
    b.set_xscale("log"); b.set_ylim(.5, 5.5); b.set_yticks(range(1, 6))
    b.set_xlim(min(ridge["MAE_per_seed"])*.4, max(ridge["MAE_per_seed"])*2)
    decorate(b, "Fixed-test MAE (log axis)", "Seed")
    panel(b, "b")
    b.xaxis.set_major_formatter(LogFormatterMathtext())
    fig.supxlabel("Colored shapes: individual runs    |    Black diamonds and bars: mean ± sample SD (n = 5)", fontsize=8)
    figure_files["figure_02_full_test_mae"] = save(fig, "figure_02_full_test_mae")

    # Paired validation comparisons; different x ranges are explicitly labeled.
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 6.1), layout="constrained")
    for ax, subset, title in [(a, pair, "a  Full difference scale"),
                              (b, pair[1:2] + pair[3:], "b  Near-zero comparisons")]:
        for pos, x in enumerate(subset):
            v, sd = x["MAE_delta_mean"], x["MAE_delta_sample_sd"]
            ax.errorbar(v, pos, xerr=sd, fmt="o", color="#334455", markersize=4, capsize=3)
        ax.axvline(0, color="#555555", ls="--", lw=1)
        ax.set_yticks(range(len(subset)), [f"{LABEL[x['comparison']]} − {LABEL[x['baseline']]}" for x in subset])
        ax.invert_yaxis(); decorate(ax, "Paired validation ΔMAE (comparison − baseline)")
        panel(ax, title[0])
    a.set_xlim(-.82, .1); b.set_xlim(-.19, .055)
    figure_files["figure_03_paired_validation"] = save(fig, "figure_03_paired_validation")

    # Compound means only provide a visualization; metric labels refer to separate runs.
    compounds = [x for x in read_csv(SRC["compound"]) if x["experiment"] == "S32_CATBOOST"]
    assert len(compounds) == 1997 and len({x["row_id"] for x in compounds}) == 1997
    y = np.array([float(x["Y_true"]) for x in compounds]); pred = np.array([float(x["prediction_mean"]) for x in compounds])
    residual = pred - y
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.6), layout="constrained")
    fig.suptitle("Hybrid-CatBoost: 1,997 fixed-test records", fontsize=9)
    a.scatter(y, pred, s=7, alpha=.55, c="#009E73", edgecolors="none", rasterized=True)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    a.plot([lo, hi], [lo, hi], ls="--", color="#333333", lw=1)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    decorate(a, "Observed TDC logS", "Mean predicted TDC logS")
    panel(a, "a")
    b.scatter(y, residual, s=7, alpha=.55, c="#009E73", edgecolors="none", rasterized=True)
    b.axhline(0, ls="--", lw=1, color="#333333")
    decorate(b, "Observed TDC logS", "Mean signed residual (prediction − observed)")
    panel(b, "b")
    figure_files["figure_04_reference_diagnostics"] = save(fig, "figure_04_reference_diagnostics")

    # Train-defined quartiles: means and sample SD across five seed-specific bins.
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.8), layout="constrained")
    for code in ["S21_RF", "S22_CATBOOST", "S31_RF", "S32_CATBOOST", "S40_CHEMPROP"]:
        sub = sorted((x for x in quart_rows if x["experiment"] == code), key=lambda x: int(x["bin"]))
        assert len(sub) == 4 and all(int(x["n_runs"]) == 5 for x in sub)
        x = np.arange(1, 5); label = LABEL[code]
        a.errorbar(x, [float(t["MAE_mean"]) for t in sub],
                   yerr=[float(t["MAE_sample_sd"]) for t in sub], marker=MARKER[label],
                   markersize=5, capsize=2, lw=1.2, color=COLOR[label], label=label)
        b.errorbar(x, [float(t["signed_bias_mean"]) for t in sub],
                   yerr=[float(t["signed_bias_sample_sd"]) for t in sub], marker=MARKER[label],
                   markersize=5, capsize=2, lw=1.2, color=COLOR[label])
    b.axhline(0, color="#444444", ls="--", lw=1)
    for ax, ylabel in [(a, "Fixed-test MAE"), (b, "Mean signed residual")]:
        ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set(xlabel="Seed-specific train-label quartile", ylabel=ylabel)
        ax.grid(axis="y", color="#D9D9D9", lw=.5)
    panel(a, "a"); panel(b, "b")
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    figure_files["figure_05_train_defined_quartiles"] = save(fig, "figure_05_train_defined_quartiles")

    # Supporting plot preserves all fifty runs, including the extreme Ridge range.
    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 6.7),
                              gridspec_kw={"height_ratios": [3.0, 1.5]}, layout="constrained")
    for pos, (code, label) in enumerate(regular):
        rr = sorted((x for x in runs if x["experiment"] == code), key=lambda x: int(x["seed"]))
        assert len(rr) == 5
        for x in rr:
            a.scatter(float(x["MAE"]), pos, marker=MARKER[label], s=22, color=COLOR[label], alpha=.7)
    a.set_yticks(range(len(regular)), [x[1] for x in regular]); a.invert_yaxis()
    decorate(a, "Per-seed fixed-test MAE (TDC logS)")
    for x in sorted((x for x in runs if x["experiment"] == "S20_RIDGE"), key=lambda x: int(x["seed"])):
        b.scatter(float(x["MAE"]), int(x["seed"]), color=COLOR["RDKit2D Ridge"], marker="s", s=25)
    b.set_xscale("log"); b.set_yticks(range(1, 6)); b.set_ylim(.5, 5.5)
    b.set_xlim(min(ridge["MAE_per_seed"])*.4, max(ridge["MAE_per_seed"])*2)
    decorate(b, "Ridge MAE (log axis)", "Seed")
    panel(a, "a"); panel(b, "b")
    figure_files["figure_s01_all_seed_test_mae"] = save(fig, "figure_s01_all_seed_test_mae")

    manifest = {"status": "RENDERED_PENDING_VISUAL_REVIEW", "generator": "code/public_plot.py",
                "generator_sha256": digest(Path(__file__)),
                "analysis_style_reference": "code/public_plot.py",
                "analysis_style_sha256": digest(Path(__file__)),
                "python_version": platform.python_version(),
                "library_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                "input_hashes": {str(path.relative_to(ROOT)): digest(path) for path in SRC.values()},
                "figures": figure_files,
                "tables": {p.name: {"path": str(p.relative_to(PKG)), "sha256": digest(p)}
                           for p in sorted(TAB.glob("table_*.csv"))},
                "public_model_name": "Hybrid-CatBoost", "internal_model_id": "S32_CATBOOST",
                "test_rows": 1997, "runs_per_group": 5,
                "plot_statistics": "Sample SD (ddof=1) of run-level metrics; paired validation differences on matched seeds; mean prediction plotted only for descriptive compound visualization",
                "source_version": "New locally generated final-run metrics; validation origin is identified in the report manifest",
                "no_new_training_or_scoring": True}
    (OUT / "figure_table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "figures": len(figure_files),
                      "tables": len(manifest["tables"]), "test_compounds": len(compounds)}))

