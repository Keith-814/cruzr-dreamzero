#!/usr/bin/env python3
"""MuJoCo pick/present/return motion task; no robot access."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('present', 'scan'), default='present')
    parser.add_argument('--scanner', choices=('fixture', 'opencv-qr', 'zbar'), default='fixture')
    parser.add_argument('--fixture-code', default='SIM-MOTION-ONLY')
    parser.add_argument('--expected-code')
    parser.add_argument('--scan-offset', nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument('--render', action='store_true', help='Capture RGB at the presentation waypoint')
    parser.add_argument('--gl', choices=('egl', 'osmesa', 'glfw'), default='egl')
    parser.add_argument('--policy', help='Optional module:factory() returning reset()/predict(obs,instruction,phase)')
    parser.add_argument('--policy-manifest', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.policy and not args.policy_manifest:
        parser.error('--policy requires --policy-manifest')
    os.environ['MUJOCO_GL'] = args.gl if args.render or args.policy or (args.mode == 'scan' and args.scanner != 'fixture') else 'disable'
    from cruzr_sim import CruzrEnvironment
    from cruzr_sim.scan_sim import SimScanBackend
    from cruzr_sim.task import PresentTaskConfig, ScanTaskConfig, run_pick_present_return, run_pick_scan_return
    env = None
    report = {'success': False, 'backend': 'mujoco-oracle', 'model_inference': False,
              'visual_scan_verified': False}
    try:
        scanner = None
        if args.mode == 'scan':
            from cruzr_sim.scanning import FixtureScanner, ImageScanner
            scanner = (FixtureScanner(args.fixture_code) if args.scanner == 'fixture' else
                       ImageScanner(args.scanner, args.output_dir/'scan-images'))
        env = CruzrEnvironment(args.config)
        backend = SimScanBackend(env, images=args.render or (args.mode == 'scan' and args.scanner != 'fixture'),
                                 scan_offset=args.scan_offset)
        if args.policy:
            from cruzr_sim.policy_runtime import PolicyMotionBackend
            module, name = args.policy.split(':', 1)
            backend = PolicyMotionBackend(backend, getattr(importlib.import_module(module), name)(), args.policy_manifest)
            report.update(backend='mujoco-phase-supervised-policy', model_inference=True)
        if args.mode == 'present':
            task = run_pick_present_return(backend, PresentTaskConfig(
                open_width_m=env.config['gripper']['max_width'],
                dwell_s=env.config['task']['presentation']['dwell_s']))
        else:
            task = run_pick_scan_return(backend, scanner, ScanTaskConfig(
                open_width_m=env.config['gripper']['max_width'], expected_code=args.expected_code))
        report.update(asdict(task), task_mode=args.mode,
                      visual_scan_verified=bool(task.barcode and task.scan_source == 'camera'),
                      scene_sha256=hashlib.sha256(env.scene_xml.encode()).hexdigest(),
                      simulation_time=float(env.data.time), config=env.config)
        if args.render and getattr(backend, 'last_capture', None):
            import cv2
            for name, frame in backend.last_capture['images'].items():
                path = args.output_dir / f'presentation-{name}.png'
                if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise OSError(f'Cannot write presentation image: {path}')
            report['presentation_images'] = [f'presentation-{name}.png'
                                            for name in backend.last_capture['images']]
    except Exception as exc:
        report.update(success=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        if env:
            try:
                env.close()
            except Exception as exc:
                report.update(success=False, close_error=f'{type(exc).__name__}: {exc}')
        (args.output_dir/'task-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
