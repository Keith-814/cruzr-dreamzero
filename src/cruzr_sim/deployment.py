"""Reproducible release receipts, not safety certification or robot authorization."""
import hashlib
import json
import os
from pathlib import Path
from .real_robot import load_plan, load_profile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_inventory(root):
    """Runtime source inventory, also usable in a checkout copied without Git."""
    root = Path(root).resolve()
    selected = {}
    for directory in ('src', 'scripts', 'tests', 'configs', 'third_party/dreamzero'):
        for path in sorted((root/directory).rglob('*')):
            if (path.is_file() and path.suffix in ('.py', '.sh', '.toml', '.json', '.yaml', '.yml')
                    and '__pycache__' not in path.parts and '.local.' not in path.name):
                selected[str(path.relative_to(root))] = digest(path)
    for name in ('pyproject.toml',):
        if (root/name).is_file():
            selected[name] = digest(root/name)
    return selected


def check_sources(report, root):
    records = report.get('source_sha256')
    if not records:
        raise ValueError('Simulation report must contain source fingerprints')
    root = Path(root).resolve()
    if 'code_inventory' in report and report['code_inventory'] != code_inventory(root):
        raise ValueError('Runtime source inventory changed; repeat simulation acceptance')
    for name, expected in records.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Simulation source changed or missing: {name}')


def create_release(sim_report, task_report, profile, plan, root):
    sim = json.loads(Path(sim_report).read_text())
    task = json.loads(Path(task_report).read_text())
    task_mode = task.get('task_mode', 'present')
    if task_mode not in ('present', 'scan'):
        raise ValueError(f'Unsupported task mode: {task_mode}')
    p = load_profile(profile, commissioned=True, require_scanner=(task_mode == 'scan'))
    load_plan(plan, p)
    if sim.get('success') is not True or sim.get('render_requested') is not True:
        raise ValueError('Successful RGB simulation acceptance is required')
    if not sim.get('code_inventory'):
        raise ValueError('Repeat simulation acceptance with the current runtime inventory checker')
    check_sources(sim, root)
    if sim.get('present_task_report_sha256') != digest(task_report):
        raise ValueError('Task report must be the exact artifact of this acceptance run')
    if task.get('model_inference') is not False or task.get('backend') != 'mujoco-oracle':
        raise ValueError('This release flow covers oracle motion only; model deployment needs separate acceptance')
    if not all(task.get(key) is True for key in ('success', 'motion_complete', 'placement_verified')):
        raise ValueError('Pick/present/return motion regression has not passed')
    if task.get('config') != sim.get('config'):
        raise ValueError('Task and acceptance must use the same simulation config')
    return {'schema': 1, 'calibration_id': p['commissioning']['calibration_id'],
            'scope': 'commissioned fixed-fixture pick/present/return motion; no claim of sim/real dynamics equivalence',
            'task_mode': task.get('task_mode', 'present'),
            'visual_scan_verified_in_sim': task.get('visual_scan_verified', False),
            'files': {name: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for name, path in
                      (('sim_report', sim_report), ('task_report', task_report), ('profile', profile), ('plan', plan))}}


def portable_receipt(receipt, destination):
    """Evidence paths relative to the receipt survive copying the outputs tree."""
    destination = Path(destination).resolve()
    receipt = json.loads(json.dumps(receipt))
    for name in ('sim_report', 'task_report'):
        receipt['files'][name]['path'] = os.path.relpath(receipt['files'][name]['path'], destination.parent)
    return receipt


def verify_release(receipt, profile, plan, root, *, task_mode=None):
    receipt_path = Path(receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 1:
        raise ValueError('Unsupported release receipt')
    if task_mode is not None and receipt.get('task_mode') != task_mode:
        raise ValueError(f'Release task mode is {receipt.get("task_mode")!r}, requested {task_mode!r}')
    for name, path in (('profile', profile), ('plan', plan)):
        if digest(path) != receipt['files'][name]['sha256']:
            raise ValueError(f'Release {name} changed; repeat release review')
    for name in ('sim_report', 'task_report'):
        row = receipt['files'][name]
        path = Path(row['path'])
        row['path'] = str(path if path.is_absolute() else (receipt_path.parent/path).resolve())
        if digest(row['path']) != row['sha256']:
            raise ValueError(f'Release evidence changed: {name}')
    expected = create_release(receipt['files']['sim_report']['path'], receipt['files']['task_report']['path'],
                              profile, plan, root)
    if expected['calibration_id'] != receipt.get('calibration_id'):
        raise ValueError('Release calibration id mismatch')
    if expected['task_mode'] != receipt.get('task_mode'):
        raise ValueError('Release task mode does not match acceptance evidence')
    return expected
