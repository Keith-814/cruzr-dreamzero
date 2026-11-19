#!/usr/bin/env python3
"""Run server regression tests; failures AND skipped tests fail acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def result_summary(result):
    return {
        'success': result.wasSuccessful() and not result.skipped and result.testsRun > 0,
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = 'disable'  # Physics tests do not render.
    if args.config:
        from cruzr_sim.config import load_config
        path = args.config.expanduser().resolve()
        config = load_config(path)
        os.environ['CRUZR_TEST_CONFIG'] = str(path)
        os.environ['CRUZR_MODEL_PATH'] = config['simulation']['model']
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = result_summary(result)
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
