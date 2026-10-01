# Source attribution and redistribution scope

The MIT license applies to this project's distributed code. No third-party
library implementation, raw molecular dataset, pretrained weights or fitted
processor is bundled. Users obtain dependencies and data from their respective
providers, subject to those providers' terms. MIT does not relicense them.

- Therapeutics Data Commons: official ADMET BenchmarkGroup acquisition and
  scaffold splits, PyTDC 1.1.15. Documentation:
  https://tdcommons.ai/benchmark/admet_group/overview/
- AqSolDB: Sorkun, Khetan and Er, *AqSolDB, a curated reference set of aqueous
  solubility and 2D descriptors for a diverse set of compounds*, Scientific Data
  6, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1
- RDKit: original SMILES parsing, radius-2 binary Morgan fingerprints and the
  recorded two-dimensional descriptor registry. https://www.rdkit.org/
- scikit-learn: SimpleImputer, VarianceThreshold, StandardScaler, DummyRegressor,
  Ridge, RandomForestRegressor and metric implementations.
  https://scikit-learn.org/stable/
- CatBoost: GPU Plain boosting and CPU inference. https://catboost.ai/
- Chemprop: native D-MPNN featurizers, message passing, aggregation and regression
  head, version 2.2.1. https://github.com/chemprop/chemprop
- PyTorch: direct Adam/MSE training, deterministic configuration and serialization.
  https://pytorch.org/
- NumPy, pandas, SciPy, joblib, threadpoolctl and Matplotlib are runtime imports;
  their licenses remain those of their own distributions.

The TDC task page lists AqSolDB as CC BY 4.0, but the exact local benchmark archive
has not been cleared for redistribution. This repository provides an official
retrieval command and hashes rather than redistributing the source rows.
No raw data, per-molecule labels/predictions, historical processor statistics,
weights, private declarations or credentials are release assets.

AI assistance was used to inspect, extract and adapt code and draft documentation.
The human author remains responsible for verifying scientific behavior, source
attribution and the final public version. No authorship is attributed to AI.
