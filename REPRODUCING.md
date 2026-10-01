# Reproducing the frozen study

## Installation and status

Run all commands from the repository root. Python 3.11.16 is the recorded
version. The portable commands fail if recorded scientific-library versions
or the pinned PyTDC source files differ. They do not silently install/upgrade
packages. A new installation and a full public-pipeline GPU run have not yet
been completed. `evidence/public_preparation_checks.json` records the actual
checks; historical R1–R3 PASS files describe earlier internal runs.

For core dependencies, create a separate environment:

```bash
conda env create -f environment.yml
conda activate hybrid-catboost
python -m pip check
python -B code/verify_release.py
python -B -m unittest discover -s tests -p test_public_protocol.py -v
```

The full paper additionally needs the recorded Torch CUDA build, Chemprop and
Lightning:

```bash
python -m pip install -r requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip check
python -B -m unittest discover -s tests -p test_graph_protocol.py -v
```

These are installation instructions, not evidence that the current providers
supply every recorded wheel. If resolution or driver compatibility fails,
preserve the error; do not change versions and claim the same reproduction.
`environment.lock.txt` records installed distributions in the checked source
environment; it is not a wheel/hash lock validated in a clean environment.
`configs/environment_record.json` distinguishes this status explicitly.
The original training GPU was an RTX 3060. Peak VRAM and the exact driver version
were not recorded, so no minimum-VRAM or specific-driver guarantee is made.
No CPU replacement is allowed for formal CatBoost or D-MPNN training.

## Data and train-only feature preparation

```bash
python -B code/reproduce.py download
python -B code/reproduce.py splits
python -B code/reproduce.py audit-data
python -B code/reproduce.py features --part train_val
python -B code/reproduce.py preprocessing
python -B code/reproduce.py check --scope primary
```

To use an already downloaded, matching official TDC cache, replace the first
command with `python -B code/reproduce.py download --cache PATH_TO_TDC_CACHE`.
The cache must contain `admet_group/solubility_aqsoldb/train_val.csv` and
`test.csv`. Do not substitute the single-task ADME loader or a new random split.
The code verifies the original 7,985/1,997 files, stable source-row IDs,
exact seed-specific split memberships and serialized split CSVs, raw feature
array hashes and the five primary training-matrix hashes.

Features are calculated from original SMILES without extra structural
standardization. Descriptor medians, constant filtering and the Ridge scaler
are fitted on each seed's training rows only. The scaler is bypassed for tree
models; retained Ipc receives log1p only in the four descriptor-tree groups.
Test labels are never used for fitting or selection. Newly fitted processors
are saved locally and hash-verified before joblib loading. Do not restore
untrusted processor files.

A failed stage preserves its partial output. Do not delete evidence and silently
rerun. Start an independent checkout for a fresh attempt. Acquisition, split,
feature and processor stages intentionally refuse existing outputs.

For a fresh checkout, the ordered workflows are also provided as shell scripts:
`bash code/run_primary.sh` and `bash code/run_paper.sh`. Activate the environment
first. An optional first argument supplies a matching official TDC cache. Each
script stops on a failed stage and preserves its outputs; they do not delete
or silently skip prior scientific results. Use the stage commands below for
controlled continuation.

## Workflow A: five frozen Hybrid-CatBoost fits

This verifies the already selected reference recipe, without repeating the
candidate search:

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

The smoke fit uses 32 rows and two trees and is never evaluated as a scientific
result. The full stage fits five new 1,000-tree models on the original training
subsets, without early stopping or merging validation rows. The blind prediction
stage requires all five accepted fits. Only the separate evaluation stage reads
held-out labels. Newly generated models are used for serving; historical weights
and serialized processors are not needed or provided.

## Workflow B: all ten groups and candidate selection

This is the full, long GPU/CPU experiment, including 405 traditional and 60
D-MPNN validation fits. It also independently refits the 50 selected final
models. Run it on the user's training host:

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

Run the test-feature command only once per checkout; omit it if Workflow A
already completed it. Exact ties in five-seed validation MAE mean are resolved
by sample SD and then frozen candidate order. D-MPNN validation uses the native
Chemprop graph/MPNN and direct Adam/MSE loop, with patience 20 and at most 100
epochs. Final D-MPNN models are initialized afresh and fitted for the seed-specific
selected best epoch, with no validation scoring during final fitting.
Historical frozen D-MPNN epochs are 28/27/29/29/9. No scheduler, pretrained
encoder, label normalization or Lightning Trainer is introduced.

To reproduce the 50 historical frozen final jobs without repeating validation,
use a new run with `freeze --run paper_frozen --scope paper --selection historical`,
then train/predict/evaluate/report that run. In that shorter route, report
validation summaries come from explicitly labeled historical summary evidence.
They are not new validation results. The report contains paired validation
comparisons, all 50 per-seed metrics, native TDC display summaries, reference
prediction/residual plots, seed-specific training-quartile analyses, quartile
counts, provenance tables and all six manuscript/SI figure renderers.
The report also recalculates difficult-record ranks, maximum training-set
Morgan Tanimoto similarity, retained raw descriptor range/missingness flags,
prediction-disagreement correlations and descriptive shrinkage measures.
These flags are not calibrated uncertainty intervals or diagnoses of bad labels.
The separate audit-data stage records raw/canonical/scaffold overlaps and empty
scaffold counts without reading label values or changing the official splits.

For interrupted jobs, `validate`, `train` and `predict` support `--resume` only
when sources, environment, input manifests and specifications still match.
An incomplete or failed job is preserved and blocks resume; use a fresh run
name to retry. Formal numerical failures are not clipped, removed or silently
repaired. The very large historical Ridge errors remain in the expected
results and are plotted on a separately labeled logarithmic scale. Constant
predictor Spearman values remain null. Never select a replacement candidate
using observed test errors.

## Checking results

Outputs remain under `data/solubility_trackA/` and `generated/RUN/`, both ignored
by Git. In particular, inspect:

- `final_plan.json`: selections, seed-specific epoch counts and differences
  from the historical frozen selection.
- `evaluation/per_seed_metrics.csv`: newly measured run-level metrics.
- `evaluation/historical_comparisons.csv`: every metric against the historical
  numerical lock, using predeclared absolute/relative tolerances.
- `evaluation/summary.json`: strict historical metric agreement and the separate
  previously accepted primary R3 MAE tolerance (0.03 per seed; 0.02 for its mean).
- `report/`: CSV tables, PDF/SVG/600-dpi PNG figures and input/output hashes.

Metric computation completing is not evidence that historical results matched.
The evaluation status is COMPUTED and agreement flags must be inspected.
Cross-device floating-point differences may prevent exact equality. The stated
strict checks and broader primary R3 tolerances are distinct; passing the latter
does not establish agreement of all models or all metrics. Do not revise
acceptance tolerances after seeing results. Preserve discrepancies for review.
Figures require visual inspection after rendering. Five-run SD is sample SD,
not a confidence interval. The native TDC display separately uses per-run
rounding and population SD. Averaged predictions are plotted descriptively;
no ensemble benchmark score, external validation or PBPK validation is claimed.

## Before the GitHub release and ChemRxiv statement

For the current source-only publication scope, the author has deferred full
experimental reconstruction. No new full training or fresh-install verification
is required by this preparation task. State those unverified portions honestly;
readers can run the supplied workflows. Approve the exact files, publish this
source folder as a fixed commit/Release, verify anonymous download, and then
insert the actual version/URL into the manuscript. A DOI is optional and must not be invented.
`CITATION.cff` has no fabricated release version, repository URL or manuscript DOI.
The MIT code license is separate from the eventual ChemRxiv manuscript license.
The public repository/version and final author-review records remain pending.
A code release does not itself establish that the new raw-to-result pipeline has
passed full runtime reproduction. Manuscript PDF/DOCX and SI remain submission documents;
they are not required GitHub source assets.
