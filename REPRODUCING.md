# Reproducing the study

## Installation and status

Run all commands from the repository root. Python 3.11.16 is the recorded
version. The reproduction commands require the recorded scientific-library
versions and pinned PyTDC source files; they do not install or upgrade packages
automatically.

An isolated Linux core installation, full `train_val` feature reconstruction,
five train-only preprocessing fits, and one 32-row/two-tree GPU smoke test
passed on October 2, 2026. See
`evidence/public_portability_smoke.json`. Test-feature reconstruction and full
five-model/paper runs using the public raw-SMILES pipeline in a fresh
environment remain pending. The earlier R3 check successfully refitted all
five Hybrid-CatBoost models using frozen features and processors and
reproduced their fixed-test predictions. Earlier verification files
remain available under `evidence/` with their original scopes.

PyTDC 1.1.15 imports `pkg_resources`. The core requirements therefore pin
`setuptools==78.1.0`, matching the original environment, because Setuptools 82
and later remove that module. A successful `pip check` alone does not detect
this undeclared runtime import. See the
[Setuptools documentation](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html).

Exact historical feature reconstruction currently targets Linux x86_64.
`environment.yml` pins the original conda-forge build identifiers, including
NumPy, RDKit, and BLAS/LAPACK. Matching package version strings from other
builds do not guarantee numerical equivalence.
`configs/numerical_runtime.json` records the reference builds and the
20-thread BLAS setting used for descriptor calculation; training stages use
one thread. The original array and matrix hashes remain the numerical
reference. Other platforms and numerical builds have not yet been validated.

For core dependencies, create a separate environment:

```bash
conda env create -f environment.yml
conda activate hybrid-catboost
python -m pip check
python -B code/verify_release.py
python -B -m unittest discover -s tests -p test_public_protocol.py -v
```

The full paper additionally requires the recorded Torch CUDA build, Chemprop,
and Lightning:

```bash
python -m pip install -r requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip check
python -B -m unittest discover -s tests -p test_graph_protocol.py -v
```

The core Linux conda-build installation has been tested, but the full-paper
extra dependencies have not yet been verified in a fresh environment. If
dependency resolution or driver compatibility fails, keep the recorded
versions unchanged and report the discrepancy.
`environment.lock.txt` records distributions from the checked source
environment; it is not a wheel/hash lock validated in a clean environment.
`configs/environment_record.json` documents this distinction.

The original training GPU was an NVIDIA GeForce RTX 3060. Peak VRAM and the
exact driver version were not recorded, so no minimum-VRAM or specific-driver
requirement is claimed. Formal CatBoost and D-MPNN training use GPU execution.

## Data and train-only feature preparation

```bash
python -B code/reproduce.py download
python -B code/reproduce.py splits
python -B code/reproduce.py audit-data
python -B code/reproduce.py features --part train_val
python -B code/reproduce.py preprocessing
python -B code/reproduce.py check --scope primary
```

To use an existing matching TDC cache, replace the first command with:

```bash
python -B code/reproduce.py download --cache PATH_TO_TDC_CACHE
```

The cache must contain:

```text
admet_group/solubility_aqsoldb/train_val.csv
admet_group/solubility_aqsoldb/test.csv
```

Do not substitute the single-task ADME loader or a new random split. The
pipeline verifies the original 7,985/1,997 files, stable source-row IDs,
seed-specific split membership, serialized split CSVs, raw feature-array
hashes, and the five primary training-matrix hashes.

Features are calculated from the original SMILES without additional structural
standardization. Descriptor medians, constant filtering, and the Ridge scaler
are fitted separately on each seed's training rows. The scaler is bypassed for
tree models, and retained `Ipc` values receive `log1p` only in the four
descriptor-tree groups. Test labels are not used for model fitting or
selection. Newly fitted preprocessing objects are saved locally and
hash-verified before loading.

Each stage preserves partial output after failure. For a clean retry of data,
split, feature, or preprocessing stages, use a fresh checkout or output
location rather than overwriting previous results.

For a fresh checkout, the ordered workflows are also available as shell
scripts:

```bash
bash code/run_primary.sh
bash code/run_paper.sh
```

Activate the environment first. An optional first argument may provide a
matching official TDC cache. Each script stops if a stage fails and leaves the
existing output in place.

## Workflow A: five Hybrid-CatBoost reference fits

This workflow reproduces the selected Hybrid-CatBoost configuration without
repeating the candidate search:

```bash
python -B code/reproduce.py freeze --run primary_rebuild --scope primary
python -B code/reproduce.py train --run primary_rebuild --mode smoke
python -B code/reproduce.py train --run primary_rebuild --mode full
python -B code/reproduce.py features --part test
python -B code/reproduce.py predict --run primary_rebuild
python -B code/reproduce.py evaluate --run primary_rebuild
python -B code/reproduce.py report --run primary_rebuild
python -B code/reproduce.py predict-smiles --run primary_rebuild --input examples/smiles.csv --output generated/ethanol_predictions.csv
```

The smoke fit uses 32 rows and two trees and is not used as a scientific
result. The full stage fits five new 1,000-tree models on the original
training subsets without early stopping or merging validation rows.
Prediction is performed after all five final models have been trained
successfully. Held-out labels are read only during the evaluation stage.
Historical model weights and serialized preprocessing objects are not required
for this workflow.

## Workflow B: all ten groups and candidate selection

This workflow reproduces the full comparison, including 405 traditional-model
and 60 D-MPNN validation fits, followed by independent refitting of the 50
selected final models. Run it on a CUDA-capable training host:

```bash
python -B code/reproduce.py check --scope paper
python -B code/reproduce.py validate --run paper_rebuild --scope paper
python -B code/reproduce.py freeze --run paper_rebuild --scope paper --selection validation
python -B code/reproduce.py train --run paper_rebuild --mode full
python -B code/reproduce.py features --part test
python -B code/reproduce.py predict --run paper_rebuild
python -B code/reproduce.py evaluate --run paper_rebuild
python -B code/reproduce.py report --run paper_rebuild --validation-run paper_rebuild
```

Run the test-feature command only once per checkout; omit it if Workflow A has
already completed it. Exact ties in five-seed mean validation MAE are resolved
by sample SD and then by the prespecified candidate order.

D-MPNN validation uses the native Chemprop graph/MPNN with a direct Adam/MSE
training loop, patience 20, and at most 100 epochs. Final D-MPNN models are
initialized independently and trained for the seed-specific selected epoch
counts, without validation scoring during final fitting. The selected epoch
counts are 28/27/29/29/9. No scheduler, pretrained encoder, label
normalization, or Lightning Trainer is introduced.

To reproduce the 50 historical final models without repeating candidate
selection, use a separate run with:

```bash
python -B code/reproduce.py freeze --run paper_frozen --scope paper --selection historical
```

Then run the same train, predict, evaluate, and report stages for
`paper_frozen`. In this route, validation summaries are read from the historical
reference files and are labeled accordingly; they are not newly generated
validation results.

The report contains paired validation comparisons, all 50 per-seed metrics,
native TDC display summaries, prediction/residual plots, seed-specific
training-quartile analyses, quartile counts, provenance tables, and manuscript
and Supporting Information figures. It also recalculates difficult-record
ranks, maximum training-set Morgan Tanimoto similarity, descriptor
range/missingness flags, prediction-disagreement correlations, and descriptive
shrinkage measures. These diagnostics are descriptive and are not calibrated
uncertainty intervals or label-error diagnoses.

The `audit-data` stage records raw-SMILES, canonical-SMILES, and scaffold
overlaps together with empty-scaffold counts without changing the official
splits.

Interrupted `validate`, `train`, and `predict` stages support `--resume` only
when the source files, environment, input manifests, and specifications still
match. Incomplete or failed jobs remain available for inspection; use a new run
name for a clean retry. Numerical failures are retained rather than clipped or
replaced. The large historical Ridge errors are therefore included in the
reference results and plotted on a separate logarithmic scale. Constant
predictor Spearman values remain null. Candidate selection must not be revised
using observed test errors.

## Checking results

Outputs are written under `data/solubility_trackA/` and `generated/RUN/`, both
ignored by Git. Key outputs include:

- `final_plan.json`: selected configurations, seed-specific epoch counts, and
  comparison with the historical selection.
- `evaluation/per_seed_metrics.csv`: run-level metrics.
- `evaluation/historical_comparisons.csv`: comparison with the historical
  numerical reference under the predefined tolerances.
- `evaluation/summary.json`: overall numerical agreement and reference MAE
  tolerance checks.
- `report/`: CSV tables, PDF/SVG/600-dpi PNG figures, and input/output hashes.

After evaluation, inspect `evaluation/summary.json` and
`evaluation/historical_comparisons.csv` to assess agreement with the historical
reference results. Cross-device floating-point differences may prevent exact
equality. The predefined tolerances should be applied consistently; larger
discrepancies should be reported rather than adjusted post hoc.

Figures should also be inspected after rendering. Five-run SD is the sample SD,
not a confidence interval. The native TDC display instead uses rounded per-run
values and population SD. Averaged predictions are used for descriptive plots
only; no ensemble benchmark score, external validation, or PBPK validation is
implied.

## Release and reproducibility scope

The repository provides the source workflow for data acquisition, feature
generation, preprocessing, model training, prediction, evaluation, and
reporting. Some full-scale reconstruction steps remain unverified, as described
above. Repository availability should therefore not be interpreted as
independent external validation or as confirmation that every supported
environment reproduces the historical results exactly.

The code is distributed under the MIT License. Manuscript and preprint
licensing are handled separately.
