#!/usr/bin/env bash
# Fresh-checkout full search and 50 final fits. This is a long GPU/CPU task.
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
python -B code/reproduce.py check --scope paper
python -B code/reproduce.py validate --run paper_rebuild --scope paper
python -B code/reproduce.py freeze --run paper_rebuild --scope paper --selection validation
python -B code/reproduce.py train --run paper_rebuild --mode full
python -B code/reproduce.py features --part test
python -B code/reproduce.py predict --run paper_rebuild
python -B code/reproduce.py evaluate --run paper_rebuild
python -B code/reproduce.py report --run paper_rebuild --validation-run paper_rebuild
