#!/bin/sh
# Rebuild every table, figure and number of the paper from the raw Intel Lab files.
# Cached per-run results in results/raw are reused when the simulator source is
# unchanged; delete that folder to recompute everything from scratch.
set -e
cd "$(dirname "$0")"
[ -d results/raw ] || { [ -f results/raw_runs.zip ] && unzip -q results/raw_runs.zip -d results/; } || true
python scripts/download_data.py
python scripts/preprocess.py
python -m pytest -q tests
python scripts/experiments.py tune
python scripts/experiments.py main
python scripts/experiments.py ablation
python scripts/experiments.py sens
python scripts/experiments.py signals
python scripts/analyze.py
