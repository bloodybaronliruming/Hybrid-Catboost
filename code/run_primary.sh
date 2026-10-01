#!/usr/bin/env bash
# Fresh-checkout five-model reconstruction. Preserve existing outputs on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
python -B code/verify_release.py
if [[ $# -gt 0 ]]; then
    python -B code/reproduce.py download --cache "$1"
else
    python -B code/reproduce.py download
fi
python -B code/reproduce.py splits
python -B code/reproduce.py audit-data
python -B code/reproduce.py features --part train_val
python -B code/reproduce.py preprocessing
python -B code/reproduce.py check --scope primary
python -B code/reproduce.py freeze --run primary_rebuild --scope primary
python -B code/reproduce.py train --run primary_rebuild --mode smoke
python -B code/reproduce.py train --run primary_rebuild --mode full
python -B code/reproduce.py features --part test
python -B code/reproduce.py predict --run primary_rebuild
python -B code/reproduce.py evaluate --run primary_rebuild
python -B code/reproduce.py report --run primary_rebuild
python -B code/reproduce.py predict-smiles --run primary_rebuild --input examples/smiles.csv --output generated/ethanol_predictions.csv
