"""Read-only LeRobot v2 validation and explicit Cruzr semantic contract.

Heavy table/video libraries are imported only in the server-side inspection.
The metadata checker cannot prove acquisition timing or real units; those need
an operator-reviewed contract. LeRobot v3/raw bag inputs fail with guidance.
"""
import json
import math
from pathlib import Path
import numpy as np
from .real_robot import JOINT_NAMES, SDK_LOWER, SDK_UPPER
from .deployment import digest

CAMERAS = ('observation.images.head_left', 'observation.images.wrist_left')


def check_contract(contract):
    if contract.get('embodiment') != 'cruzr_s2' or contract.get('joint_names') != list(JOINT_NAMES)+['right_gripper']:
        raise ValueError('Contract embodiment/joint order mismatch')
    if (contract.get('action_semantics') != 'absolute_joint_position' or
            contract.get('joint_unit') != 'rad' or contract.get('gripper_unit') != 'm'):
        raise ValueError('This entry requires absolute SDK joint positions in rad and gripper in metres')
    fps = contract.get('fps')
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Contract requires a finite positive fps')
    if contract.get('semantics_verified') is not True or contract.get('observation_precedes_action') is not True:
        raise ValueError('Verify real units/joint order and pre-action observation timing before preparing training data')


def inspect_metadata(directory, contract):
    root = Path(directory).resolve()
    if not (root/'meta/info.json').is_file():
        raise ValueError('Expected LeRobot v2 meta/info.json; raw bags/video need acquisition-specific conversion first')
    info = json.loads((root/'meta/info.json').read_text())
    if not str(info.get('codebase_version', '')).startswith('v2.'):
        raise ValueError('Only LeRobot v2 is supported; v3 sharded video needs an explicit v3-to-v2 conversion')
    if not math.isclose(info.get('fps', 0), contract['fps'], rel_tol=0, abs_tol=1e-6):
        raise ValueError('Dataset fps differs from verified contract; resampling is not implicit')
    for name in ('observation.state', 'action'):
        feature = info['features'][name]
        if list(feature['shape']) != [8] or feature.get('names') != contract['joint_names']:
            raise ValueError(f'{name}: expected named eight-dimensional Cruzr order')
    for key in CAMERAS:
        if info['features'].get(key, {}).get('dtype') != 'video':
            raise ValueError(f'Missing video feature: {key}')
    if set(k for k, v in info['features'].items() if v.get('dtype') == 'video') != set(CAMERAS):
        raise ValueError('This profile expects exactly head-left and physical right-wrist video')
    return info


def check_vectors(state, action, timestamps, fps):
    state, action, timestamps = np.asarray(state, float), np.asarray(action, float), np.asarray(timestamps, float)
    if state.ndim != 2 or state.shape[1] != 8 or action.shape != state.shape or len(state) < 25:
        raise ValueError('Each episode needs >=25 aligned eight-dimensional state/action frames')
    if timestamps.shape != (len(state),) or not all(np.isfinite(a).all() for a in (state, action, timestamps)):
        raise ValueError('Invalid state/action/time values')
    for name, values in (('state', state), ('action', action)):
        if np.any(values[:, :7] < SDK_LOWER-1e-4) or np.any(values[:, :7] > SDK_UPPER+1e-4):
            raise ValueError(f'{name} outside SDK bounds; check degrees/radians and joint order')
        if np.any(values[:, 7] < -1e-6) or np.any(values[:, 7] > .05+1e-6):
            raise ValueError(f'{name} gripper outside SDK metre range')
    if not np.allclose(np.diff(timestamps), 1/fps, atol=0.002, rtol=0):
        raise ValueError('Irregular timestamps or wrong sample rate; resample explicitly before training')


def task_strings(table, tasks):
    if 'annotation.task' in table:
        values = table['annotation.task'].tolist()
    elif 'task_index' in table:
        values = [tasks[int(index)] for index in table['task_index']]
    else:
        raise ValueError('No annotation.task or task_index/task metadata; do not invent labels')
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError('Missing/non-text task labels')
    return values


def inspect_dataset(directory, contract, *, verify_video=True):
    check_contract(contract)
    root = Path(directory).resolve()
    info = inspect_metadata(root, contract)
    import pandas as pd
    tasks_path = root/'meta/tasks.jsonl'
    tasks = {row['task_index']: row['task'] for row in
             (json.loads(line) for line in tasks_path.read_text().splitlines())} if tasks_path.exists() else {}
    paths = sorted(root.glob('data/chunk-*/episode_*.parquet'))
    if len(paths) != info.get('total_episodes') or len(paths) < 2:
        raise ValueError('Need >=2 episodes for disjoint train/validation; episode count must match info')
    records, used_ids, labels = [], set(), set()
    chunk_size = info.get('chunks_size', 1000)
    for path in paths:
        table = pd.read_parquet(path)
        ep_ids = set(table['episode_index'].tolist())
        if len(ep_ids) != 1:
            raise ValueError('Expected one episode per v2 parquet')
        ep = int(next(iter(ep_ids)))
        if ep in used_ids or path.stem != f'episode_{ep:06d}':
            raise ValueError('Duplicate or mismatched episode ids')
        used_ids.add(ep)
        if table['frame_index'].tolist() != list(range(len(table))):
            raise ValueError('Frame indices must be consecutive from zero')
        check_vectors(np.stack(table['observation.state']), np.stack(table['action']), table['timestamp'], contract['fps'])
        labels.update(task_strings(table, tasks))
        videos = {}
        for camera in CAMERAS:
            pattern = info.get('video_path', 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
            video = (root/pattern.format(episode_chunk=ep//chunk_size, episode_index=ep, video_key=camera)).resolve()
            if not video.is_relative_to(root) or not video.is_file():
                raise ValueError(f'Missing/nonlocal camera video for episode {ep}: {camera}')
            if verify_video:
                import cv2
                cap = cv2.VideoCapture(str(video))
                try:
                    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-contract['fps']) > .1:
                        raise ValueError('Unreadable video or camera fps mismatch')
                    count = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if frame is None or frame.ndim != 3:
                            raise ValueError('Invalid decoded video frame')
                        count += 1
                    if count != len(table):
                        raise ValueError(f'Video/state frame count mismatch: {camera}, ep={ep}')
                finally:
                    cap.release()
            videos[camera] = {'path': str(video.relative_to(root)), 'sha256': digest(video)}
        records.append({'episode_index': ep, 'frames': len(table), 'parquet': str(path.relative_to(root)),
                        'sha256': digest(path), 'videos': videos})
    return {'success': True, 'dataset': str(root), 'info_sha256': digest(root/'meta/info.json'),
            'fps': contract['fps'], 'episodes': records, 'labels': sorted(labels), 'video_decoded': verify_video,
            'semantics': 'operator contract; not automatically proven by numerical checks',
            'warning': 'Photo/inspection demonstrations do not prove successful barcode decoding; scanner remains deterministic.'}
