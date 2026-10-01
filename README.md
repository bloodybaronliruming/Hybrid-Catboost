# Hybrid-CatBoost: reproducible aqueous-solubility benchmark code

Public-source preparation for the frozen `solubility_trackA_v1` study on TDC
`Solubility_AqSolDB`. Author: Li Ruming, Beijing University of Chemical Technology;
ORCID: https://orcid.org/0009-0004-0499-429X. Code license: [MIT](LICENSE).

The repository implements official data acquisition, exact five-seed scaffold
splits, raw SMILES feature calculation, train-only preprocessing, validation
candidate selection, fresh model fitting, blind prediction, independent
evaluation and manuscript tables/figures. It covers the five Hybrid-CatBoost
reference fits and all ten comparison groups. The complete search has 465
validation fits, followed by 50 selected final fits. See
[REPRODUCING.md](REPRODUCING.md) for installation, ordered commands, resumability,
expected results, acceptance limits and GPU handoff.

## What is included

- `code/reproduce.py` and `public_*.py`: portable raw-data-to-result pipeline.
- `configs/`: frozen candidate lists, final recipes, source/split/feature hashes,
  native graph schema and recorded environment.
- `tests/`: independent scientific-invariant, failure and data-boundary checks.
- `requirements*.txt`, `environment.yml`, `environment.lock.txt`: recorded
  versions and installation instructions; a fresh installation is still pending.
- `evidence/` and `results/manuscript_metrics_lock.json`: historical numerical
  reference and earlier R1–R3 evidence, plus actual public-preparation checks.
- `code/c05_*.py` and `chemrxiv_submission/reproducibility/`: unchanged historical
  replay implementation and metadata. These legacy commands require excluded
  assets; use `reproduce.py` for the new reconstruction route.
- `docs/`: figure standard and source/redistribution notices; `CITATION.cff`.

Raw data, labels, per-molecule predictions, feature matrices, original/refitted
weights, fitted processor objects/statistics and private records are not bundled.
The code downloads data from the official provider and generates new processors
and models locally. Their source licenses remain separate from MIT. No approved
historical-weight download or access-on-request commitment is offered.

## Actual verification status

The original internal R1–R3 checks passed. R3 reused frozen features/processors;
it did not independently reconstruct raw feature preparation. The new public
pipeline has passed the checks recorded in
[evidence/public_preparation_checks.json](evidence/public_preparation_checks.json),
including isolated official acquisition/split checks and selected arithmetic
checks. Full raw-feature recomputation, the full GPU reconstruction and fresh
environment installation remain pending. Code being present is not a claim
that those complete workflows have already passed. No public GitHub URL or
fixed release has yet been created or verified.

## Method and interpretation

Original SMILES are parsed without extra salt, charge or tautomer standardization.
Inputs are 2,048 binary Morgan bits (radius 2, no chirality) and 210 RDKit 2D
descriptors. Each training subset fits its own median imputation and zero-variance
filter, retaining 207 descriptors except 206 for seed 3. Tree descriptor groups
apply log1p to Ipc after imputation/filtering; Ridge additionally uses its
train-fitted scaler without that Ipc transformation.

Hybrid-CatBoost uses GPU Plain boosting, 1,000 trees, depth 8, learning rate 0.1,
L2 leaf regularization 3, RMSE loss, no early stopping, seeds 1–5 and training
subsets only. Seeds 1–4 use 6,986/999 training/validation rows; seed 5 uses
5,045/2,940. Every final run predicts all 1,997 fixed-test rows on the original
TDC logS scale. Historical primary MAE is 0.749631 with sample SD 0.016216;
the native rounded TDC display is 0.750 with population SD 0.014. No separate
ensemble benchmark score or confidence interval is inferred from these values.

All ten final groups were frozen before testing. The candidate selection rule
uses validation MAE. A separate pre-test record uniquely naming S32 as the sole
manuscript reference was not found; the author reports that observed test scores
did not inform that reference choice. Historical test replay and model refitting
are reproducibility checks, not independent external or PBPK validation.

Official benchmark documentation:
https://tdcommons.ai/benchmark/admet_group/overview/ .
Data/software attribution and release scope:
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
