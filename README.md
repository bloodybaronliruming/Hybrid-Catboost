# Hybrid-CatBoost: reproducible aqueous-solubility benchmark code

Public source code for the Hybrid-CatBoost study on the TDC `Solubility_AqSolDB` benchmark.

Author: Li Ruming, Beijing University of Chemical Technology
ORCID: https://orcid.org/0009-0004-0499-429X
Code license: [MIT](LICENSE)

Hybrid-CatBoost combines radius-2 Morgan fingerprints with RDKit 2D descriptors and uses CatBoost regression. The repository supports official benchmark data acquisition, five-seed scaffold splitting, feature generation, train-only preprocessing, model selection, model training, fixed-test prediction, evaluation, and manuscript figure generation.

The study includes 465 validation fits followed by 50 final model fits across ten comparison groups.

## Public repository and fixed version

Repository:
[Hybrid-CatBoost](https://github.com/bloodybaronliruming/Hybrid-Catboost)

Initial source release:
v1.0.0

## What is included

- `code/reproduce.py` and `public_*.py`: portable raw-data-to-result pipeline.
- `configs/`: candidate configurations, model recipes, source/split/feature hashes, graph schema, and environment information.
- `tests/`: scientific-invariant, failure-handling, and data-boundary checks.
- `requirements*.txt`, `environment.yml`, `environment.lock.txt`: dependency and environment specifications.
- `evidence/` and `results/`: numerical reference results and reproducibility checks.
- `docs/`: figure standards, source/redistribution notices, and citation information.
- Legacy reproduction scripts are retained for traceability; `code/reproduce.py` is the recommended entry point.

Raw datasets, per-molecule predictions, feature matrices, historical model weights, and fitted preprocessing objects are not distributed. The public pipeline obtains benchmark data from the original provider and regenerates preprocessing objects and models locally.

See [REPRODUCING.md](REPRODUCING.md) for installation instructions, ordered commands, expected outputs, verification criteria, and GPU execution.

## Reproducibility status

Core data acquisition, split generation, feature preparation, preprocessing, and a GPU smoke test have been checked in an isolated Linux environment.

The checks confirmed:

- all 7,985 training/validation feature rows;
- five independently fitted training-only preprocessing pipelines;
- agreement with the five primary training-matrix reference hashes; and
- a successful 32-row, two-tree GPU smoke fit.

Exact reconstruction depends on the pinned numerical-library builds and runtime settings documented in `environment.yml` and `configs/numerical_runtime.json`.

Full test-feature recomputation and complete five-model result reconstruction using the public raw-SMILES pipeline in a fresh environment remain pending. The earlier R3 check successfully refitted all five models using frozen features and processors and reproduced their fixed-test predictions. The smoke test verifies pipeline portability and is not a new benchmark evaluation or external validation.

## Method and interpretation

Original SMILES are parsed without additional salt removal, charge neutralization, or tautomer standardization.

The molecular inputs are:

- 2,048-bit binary Morgan fingerprints with radius 2 and no chirality; and
- 210 RDKit 2D descriptors.

For each training split, descriptor preprocessing fits median imputation and zero-variance filtering using training data only, retaining 207 descriptors for seeds 1, 2, 4, and 5 and 206 for seed 3. Tree-based descriptor models apply `log1p` to the retained `Ipc` descriptor. Ridge regression additionally uses a training-fitted scaler and does not use this `Ipc` transformation.

Hybrid-CatBoost uses GPU Plain boosting with 1,000 trees, depth 8, learning rate 0.1, L2 leaf regularization 3, RMSE loss, and no early stopping. Seeds 1–4 use 6,986/999 training/validation records, while seed 5 uses 5,045/2,940. Each final model predicts the same 1,997 fixed-test records on the original TDC logS scale.

The five-run Hybrid-CatBoost test MAE is:

- full-precision mean ± sample SD: `0.749631 ± 0.016216`;
- native rounded TDC-style display: `0.750 ± 0.014`.

Candidate selection was based on validation MAE before test evaluation. Hybrid-CatBoost was subsequently used as the reference model for detailed analysis. Model replay and refitting assess reproducibility within the benchmark workflow and do not constitute external or PBPK validation.

## Benchmark documentation

Official TDC ADMET benchmark documentation:
https://tdcommons.ai/benchmark/admet_group/overview/

Data/software attribution and release scope:
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)
