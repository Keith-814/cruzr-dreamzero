#!/usr/bin/env python3
"""Server acceptance run: assets, tests, hold, repeated oracle task, RGB and policy timing.

Does not install dependencies, infer policies, contact a robot, or change GPU
settings. A JSON report and separate logs are saved even when a stage fails.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/mujoco.toml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--render', action='store_true', help='Include RGB validation; requires working GL backend')
    parser.add_argument('--gl', choices=('egl', 'osmesa', 'glfw'), default='egl')
    parser.add_argument('--timeout', type=float, default=600, help='Maximum wall seconds per stage')
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.timeout < float('inf'):
        parser.error('repeats and timeout must be positive')
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    child_env = os.environ.copy()
    child_env['MUJOCO_GL'] = args.gl if args.render else 'disable'
    child_env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + child_env.get('PYTHONPATH', '')
    report = {'success': False, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'render_requested': args.render, 'gl': args.gl if args.render else 'disable',
              'egl_device_id': child_env.get('MUJOCO_EGL_DEVICE_ID'),
              'results_scope': 'simulation foundation; oracle task; no DreamZero inference', 'stages': []}
    report_path = args.output_dir/'report.json'

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def stage(name, arguments):
        started = time.perf_counter()
        command = [sys.executable, *map(str, arguments)]
        log = args.output_dir/f'{name}.log'
        item = {'name': name, 'command': command, 'log': str(log), 'success': False}
        with log.open('w', encoding='utf-8') as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, env=child_env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                item.update(returncode=result.returncode, success=result.returncode == 0)
            except subprocess.TimeoutExpired:
                item['error'] = f'Stage timed out after {args.timeout} seconds'
        item['wall_time_s'] = time.perf_counter()-started
        report['stages'].append(item)
        save()
        print(f"{name}: {'PASS' if item['success'] else 'FAIL'}", flush=True)
        if not item['success']:
            raise RuntimeError(f'{name} failed; inspect {log}')

    save()
    try:
        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError('Use the Python 3.11 or 3.12 simulation environment')
        from cruzr_sim.config import load_config
        from check_assets import sha256
        cfg = load_config(args.config)
        from cruzr_sim.deployment import code_inventory
        report['code_inventory'] = code_inventory(ROOT)
        # A copied checkout need not contain .git. Runtime fingerprints are
        # always available and are what the release verifier actually needs.
        report['source_sha256'] = dict(report['code_inventory'])
        report['config'] = cfg
        report['config_sha256'] = sha256(args.config)
        report['versions'] = {name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'opencv-python-headless')}
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=30)
        if freeze.returncode:
            raise RuntimeError(f'Cannot record installed dependencies: {freeze.stderr}')
        (args.output_dir/'environment-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
        try:
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=30)
            report['git_commit'] = git.stdout.strip() if git.returncode == 0 else None
            dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=30)
            report['git_dirty'] = bool(dirty.stdout) if dirty.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            report['git_commit'] = report['git_dirty'] = None
        # Information only: simulation physics is CPU based and does not require
        # all four GPUs. Inference feasibility on 4x24 GB is a separate test.
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                                  '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
            report['gpus'] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['gpus'] = str(exc)
        model = Path(cfg['simulation']['model'])
        stage('dependencies', ['-m', 'pip', 'check'])
        stage('assets', ['scripts/check_assets.py', '--asset-dir', model.parent, '--scene-name', model.name])
        stage('unittest', ['scripts/run_tests.py', '--config', args.config,
                           '--output', args.output_dir/'unittest.json'])
        common = ['--config', args.config, '--gl', args.gl]
        stage('hold', ['scripts/run_sim.py', *common, '--mode', 'hold', '--seconds', '10'])
        for index in range(args.repeats):
            run_args = ['scripts/run_sim.py', *common, '--mode', 'pick-place', '--seconds', '8']
            if args.render and index == 0:
                run_args += ['--render-dir', args.output_dir/'pick-place-rgb']
            stage(f'pick-place-{index:03d}', run_args)
        contract_args = ['scripts/smoke_pi05_contract.py', *common,
                         '--output-dir', args.output_dir/'policy-contract']
        if args.render:
            contract_args += ['--render']
        stage('policy-contract', contract_args)
        present_args = ['scripts/run_scan_task.py', *common,
                        '--mode', 'present', '--output-dir', args.output_dir/'present-task']
        if args.render:
            present_args.append('--render')
        stage('present-task-motion', present_args)
        report['present_task_report_sha256'] = sha256(args.output_dir/'present-task/task-report.json')
        report['present_task_scope'] = 'oracle motion to face/camera waypoint; no scanner'
        report['success'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        save()
    print(str(report_path))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
