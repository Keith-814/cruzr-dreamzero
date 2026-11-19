#!/usr/bin/env python3
"""SDK read-only probe/teaching/shadow, or explicitly requested hardware execution."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'teach', 'shadow', 'execute'), default='probe')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--waypoint', choices=('start', 'pregrasp', 'grasp', 'lift', 'present', 'scan', 'return_high', 'return_low', 'retreat'))
    parser.add_argument('--release', type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly enable SDK publishers; otherwise all modes are read-only')
    parser.add_argument('--task-mode', choices=('present', 'scan'), default='present')
    parser.add_argument('--scanner', choices=('opencv-qr', 'zbar'), default='zbar')
    parser.add_argument('--expected-code')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--transport', help='Optional module:factory(profile,writable=False), for validated site adapters')
    parser.add_argument('--placement-verifier', help='Optional module:function(transport,plan)->bool; calibrated object verification')
    args = parser.parse_args()
    if args.waypoint == 'present':
        # The persisted plan keeps the historical ``scan`` key for backwards
        # compatibility; user-facing teach/execute terminology is present.
        args.waypoint = 'scan'
    from cruzr_sim.real_robot import HardwareController, HardwareScanBackend, JOINT_NAMES, load_profile, load_plan
    from cruzr_sim.deployment import verify_release
    from cruzr_sim.task import PresentTaskConfig, ScanTaskConfig, run_pick_present_return, run_pick_scan_return

    writable = args.mode == 'execute' and args.execute
    if args.mode == 'execute' and not writable:
        parser.error('execute mode requires --execute; default is read-only')
    if args.execute and args.mode != 'execute':
        parser.error('--execute is only valid in execute mode')
    if args.mode in ('shadow', 'execute') and not args.plan:
        parser.error(f'{args.mode} requires --plan')
    if args.mode == 'teach' and (not args.plan or not args.waypoint):
        parser.error('teach requires --plan and --waypoint')
    profile = load_profile(args.profile, commissioned=writable, require_scanner=(writable and args.task_mode == 'scan'))
    plan = load_plan(args.plan, profile) if args.mode in ('shadow', 'execute') else None
    if writable:
        if not args.release:
            parser.error('Hardware execution requires --release')
        verify_release(args.release, args.profile, args.plan, ROOT, task_mode=args.task_mode)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def symbol(name):
        module, attr = name.split(':', 1)
        return getattr(importlib.import_module(module), attr)

    # Validate scanner dependencies BEFORE creating any transport/publisher.
    scanner = None
    if writable and args.task_mode == 'scan':
        from cruzr_sim.scanning import ImageScanner
        scanner = ImageScanner(args.scanner, args.output_dir/'scan-images')
    from cruzr_sim.ros2_transport import Ros2Transport
    factory = symbol(args.transport) if args.transport else Ros2Transport
    transport, controller = None, None
    report = {'success': False, 'mode': args.mode, 'publishers_enabled': writable}
    try:
        transport = factory(profile, writable=writable)
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            state = transport.read_state()
            if 'joint_position' in state and 'gripper' in state:
                break
            time.sleep(.05)
        else:
            raise TimeoutError('No arm/gripper feedback within 5 seconds')
        report['state'] = state
        if args.mode == 'probe':
            # Receipt alone is not commissioning or a healthy-motor verdict.
            report.update(success=True, note='Read-only sample; inspect stamps, names and health before commissioning')
        elif args.mode == 'teach':
            if not args.plan or not args.waypoint:
                parser.error('teach requires --plan and --waypoint; move externally before recording')
            from cruzr_sim.real_robot import finite_vector
            q = finite_vector(state['joint_position'], 7, 'taught pose')
            if not 0 <= time.monotonic()-state['arm_received_s'] <= profile['max_state_age_s']:
                raise RuntimeError('Cannot teach from stale feedback')
            if any(not lo <= value <= hi for value, lo, hi in zip(q, profile['lower'], profile['upper'])):
                raise ValueError('Taught pose outside configured bounds')
            taught = json.loads(args.plan.read_text()) if args.plan.exists() else {
                'joint_names': list(JOINT_NAMES), 'calibration_id': profile['commissioning']['calibration_id'], 'waypoints': {}}
            if taught['joint_names'] != list(JOINT_NAMES) or taught['calibration_id'] != profile['commissioning']['calibration_id']:
                raise ValueError('Existing plan belongs to another calibration')
            if args.waypoint in taught['waypoints']:
                raise ValueError('Waypoint already recorded; review/edit a new plan instead of silently overwriting')
            taught['waypoints'][args.waypoint] = q.tolist()
            args.plan.write_text(json.dumps(taught, ensure_ascii=False, indent=2)+'\n')
            report['success'] = True
        elif args.mode == 'shadow':
            controller = HardwareController(transport, profile)
            controller.state()  # Require fresh state and commissioned external health, no publishers.
            report.update(success=True, proposed_waypoints=plan['waypoints'],
                          note='Read-only proposal; no motion/collision/grasp validation is implied')
        else:
            verifier = symbol(args.placement_verifier) if args.placement_verifier else None
            controller = HardwareController(transport, profile, enabled=True)
            from cruzr_sim.real_robot import check_start_state
            check_start_state(controller.state(), plan, profile)
            # Reject missing/stale images before the first motion, rather than
            # discovering it while holding an object at the face waypoint.
            transport.capture_images()
            controller.start()
            backend = HardwareScanBackend(controller, plan,
                placement_verifier=(lambda: verifier(transport, plan)) if verifier else None)
            if args.task_mode == 'present':
                task = run_pick_present_return(backend, PresentTaskConfig(
                    open_width_m=profile['gripper_calibration']['geometric_max_m']))
            else:
                task = run_pick_scan_return(backend, scanner, ScanTaskConfig(
                    open_width_m=profile['gripper_calibration']['geometric_max_m'], expected_code=args.expected_code))
            report.update(asdict(task))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if controller:
            try:
                controller.stop()
            except Exception as exc:
                report.update(success=False, stop_error=str(exc))
        if transport:
            try:
                transport.close()
            except Exception as exc:
                report.update(success=False, close_error=str(exc))
        path = args.output_dir/'robot-report.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.tolist())+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.tolist()))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
