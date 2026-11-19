#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SIM_PYTHON="${SIM_PYTHON:-python3.11}"
"$SIM_PYTHON" - <<'PY'
import sys
if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise SystemExit(
        f"Python 3.11 or 3.12 is required; found {sys.version.split()[0]}"
    )
PY
"$SIM_PYTHON" -m venv .venv-sim
.venv-sim/bin/python -m pip install --upgrade pip
.venv-sim/bin/python -m pip install -e .
.venv-sim/bin/python -m pip check
echo 'Simulation environment ready. Next: .venv-sim/bin/python scripts/import_assets.py --source-root /data1/oylx2'
