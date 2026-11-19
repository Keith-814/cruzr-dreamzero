#!/usr/bin/env python3
"""Run the default pick, face-presentation, and return simulation task."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from run_scan_task import main


if __name__ == '__main__':
    # Keep one implementation and expose a task-specific, discoverable entry.
    sys.argv[1:1] = ['--mode', 'present']
    raise SystemExit(main())
