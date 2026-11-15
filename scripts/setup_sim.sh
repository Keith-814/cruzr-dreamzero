#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-sim
.venv-sim/bin/python -m pip install --upgrade pip
.venv-sim/bin/python -m pip install -e .
echo 'Simulation environment ready. Next: .venv-sim/bin/python scripts/import_assets.py --source-root /data1/oylx2'
