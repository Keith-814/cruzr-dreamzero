#!/usr/bin/env python3
"""Bind sim evidence, commissioned hardware profile and taught poses; no ROS."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cruzr_sim.deployment import create_release, portable_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sim-report', 'task-report', 'profile', 'plan', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    receipt = portable_receipt(create_release(args.sim_report, args.task_report, args.profile, args.plan, ROOT), args.output)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == '__main__':
    main()
