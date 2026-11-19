#!/usr/bin/env python3
"""Read-only deployment preflight; never starts physics, training or robot nodes."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('sim', 'model', 'robot'), required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--gpus', type=int, default=4, help='Expected visible GPU count for model target')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output report already exists; use a new filename')
    report = {'target': args.target, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'checks': [], 'success': False,
              'scope': 'preflight only; no physical, CUDA memory, ROS routing or model quality acceptance'}

    def check(name, function):
        try:
            detail = function()
            report['checks'].append({'name': name, 'success': True, 'detail': detail})
        except Exception as exc:
            report['checks'].append({'name': name, 'success': False, 'error': f'{type(exc).__name__}: {exc}'})

    def versions(packages):
        return {name: metadata.version(name) for name in packages}

    def require_python():
        minimum = (3, 10) if args.target == 'robot' else (3, 11)
        if sys.version_info[:2] < minimum or (args.target != 'robot' and sys.version_info[:2] >= (3, 13)):
            raise ValueError(f'Need Python {minimum[0]}.{minimum[1]}'+(' or 3.12' if args.target != 'robot' else '+'))
        return sys.version.split()[0]

    def simulation():
        from cruzr_sim.config import load_config
        from check_assets import check_assets
        config = load_config(args.config)
        model = Path(config['simulation']['model'])
        return {'assets': check_assets(model.parent, scene_name=model.name),
                'presentation': config['task']['presentation']}

    def gpu_runtime():
        import torch
        if args.gpus < 1 or not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
            raise ValueError(f'Need at least {args.gpus} visible CUDA GPUs; check driver and CUDA_VISIBLE_DEVICES')
        return {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'devices': [{'name': torch.cuda.get_device_name(i),
                             'memory_gib': round(torch.cuda.get_device_properties(i).total_memory/1024**3, 2)}
                            for i in range(torch.cuda.device_count())],
                'note': 'Device enumeration only, no weights or tensors allocated; 4x24GB is not automatic model sharding'}

    def robot_config():
        from cruzr_sim.real_robot import load_profile, load_plan
        if args.profile is None:
            raise ValueError('--profile is required for robot target')
        if args.plan is None:
            raise ValueError('--plan is required for robot target; execution requires start plus all waypoints')
        profile = load_profile(args.profile)
        plan = load_plan(args.plan, profile)
        return {'profile_valid': True, 'plan_valid': True,
                'commissioning': profile['commissioning']}

    def sdk_schema():
        # Import/type inspection only. No rclpy.init(), node or publishers.
        requirements = {
            'mc_state_msgs.msg.RobotState': ('header', 'joint_states'),
            'mc_task_msgs.msg.RobotCommand': ('header', 'joint_cmd'),
            'mc_task_msgs.msg.JointCmd': ('name', 'control_mode', 'position'),
            'ecat_task_msgs.msg.GripStatus': ('header', 'init_state', 'homed', 'error_code', 'grip_state', 'pos'),
            'ecat_task_msgs.msg.GripCmd': ('header', 'pos', 'vel', 'force', 'cur', 'mode'),
            'sensor_msgs.msg.Image': ('header', 'encoding', 'step', 'data'),
            'std_msgs.msg.Bool': ('data',), 'std_msgs.msg.Empty': (),
        }
        importlib.import_module('rclpy')
        for name, fields in requirements.items():
            module, symbol = name.rsplit('.', 1)
            value = getattr(importlib.import_module(module), symbol)()
            if any(not hasattr(value, field) for field in fields):
                raise ValueError(f'SDK message schema mismatch: {name}; expected {fields}')
        return list(requirements)

    check('python', require_python)
    if args.target == 'sim':
        check('packages', lambda: versions(('mujoco', 'numpy', 'opencv-python-headless')))
        check('config-and-assets', simulation)
    elif args.target == 'model':
        check('packages', lambda: versions(('torch', 'transformers', 'hydra-core', 'numpy', 'pandas', 'pyarrow', 'decord')))
        check('visible-cuda-gpus', gpu_runtime)
    else:
        check('profile-and-plan', robot_config)
        check('sdk-message-definitions', sdk_schema)
    report['success'] = bool(report['checks']) and all(row['success'] for row in report['checks'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
