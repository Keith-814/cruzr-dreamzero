#!/usr/bin/env python3
"""Validate/split LeRobot v2 data and build a Cruzr DreamZero training workspace.

Does not install packages, download weights, train, or modify input datasets.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cruzr_sim.dataset import CAMERAS, inspect_dataset, task_strings
from cruzr_sim.finetune import prepare_source
from cruzr_sim.deployment import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, output = args.dataset.resolve(), args.output_dir.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('Output and source must be separate trees')
    if output.exists() or not 0 < args.validation_fraction < 1:
        parser.error('Output must not exist; validation fraction must lie in (0,1)')
    contract = json.loads(args.contract.read_text())
    inspection = inspect_dataset(source, contract)
    output.mkdir(parents=True)
    # A failed prepare retains diagnostic evidence but cannot be launched.
    report = {'success': False, 'source_inspection': inspection, 'contract': contract,
              'contract_sha256': digest(args.contract)}
    try:
        workspace = output/'dreamzero'
        report['overlay'] = prepare_source(ROOT/'third_party/dreamzero', workspace, contract['fps'])
        import pandas as pd
        info = json.loads((source/'meta/info.json').read_text())
        task_path = source/'meta/tasks.jsonl'
        tasks = {r['task_index']: r['task'] for r in (json.loads(line) for line in task_path.read_text().splitlines())} if task_path.exists() else {}
        records = inspection['episodes'][:]
        random.Random(args.seed).shuffle(records)
        count = max(1, min(len(records)-1, round(len(records)*args.validation_fraction)))
        splits = {'train': sorted(records[count:], key=lambda r: r['episode_index']),
                  'validation': sorted(records[:count], key=lambda r: r['episode_index'])}
        report['split_ids'] = {name: [row['episode_index'] for row in rows] for name, rows in splits.items()}
        for name, rows in splits.items():
            destination = output/'datasets'/name
            (destination/'meta').mkdir(parents=True)
            subset_info = dict(info)
            subset_info.update(total_episodes=len(rows), total_frames=sum(row['frames'] for row in rows))
            # The upstream converter enumerates episode files from zero. Each
            # split is therefore a self-contained dataset with local contiguous
            # ids; the original ids remain in split_ids for traceability.
            subset_info['data_path'] = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
            subset_info['video_path'] = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
            subset_info['chunks_size'] = 1000
            discarded = set(info.get('discarded_episode_indices', []))
            subset_info['discarded_episode_indices'] = [i for i, row in enumerate(rows)
                                                        if row['episode_index'] in discarded]
            if len(subset_info['discarded_episode_indices']) == len(rows):
                raise ValueError(f'All episodes in {name} are marked discarded')
            # The sharded DreamZero loader requires a global ``index`` column
            # even when an input LeRobot export omitted it. Keep all metadata
            # columns explicit so parquet and info.json describe the same
            # schema after the split.
            subset_info['features'] = {**info['features'],
                                       'index': {'dtype': 'int64', 'shape': [1]},
                                       'annotation.task': {'dtype': 'string', 'shape': [1]}}
            subset_info['total_chunks'] = (len(rows) + 999) // 1000
            subset_info['total_videos'] = len(rows) * len(CAMERAS)
            subset_info['robot_type'] = 'cruzr_s2'
            subset_info.pop('splits', None)
            (destination/'meta/info.json').write_text(json.dumps(subset_info, indent=2)+'\n')
            local_rows = []
            split_labels = set()
            frame_offset = 0
            for local_index, row in enumerate(rows):
                table = pd.read_parquet(source/row['parquet'])
                table['annotation.task'] = task_strings(table, tasks)
                split_labels.update(table['annotation.task'])
                table['episode_index'] = local_index
                # Always rebuild the global index. Reusing a source index is
                # unsafe after episode filtering and fails in the sharded
                # loader when the source omitted the optional column.
                table['index'] = range(frame_offset, frame_offset+len(table))
                frame_offset += len(table)
                target = destination/'data'/f'chunk-{local_index//1000:03d}'/f'episode_{local_index:06d}.parquet'
                target.parent.mkdir(parents=True, exist_ok=True)
                table.to_parquet(target, index=False)
                for camera, video in row['videos'].items():
                    target = destination/'videos'/f'chunk-{local_index//1000:03d}'/camera/f'episode_{local_index:06d}.mp4'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source/video['path'], target)
                local_rows.append({'local_episode_index': local_index,
                                   'original_episode_index': row['episode_index'],
                                   'frames': row['frames']})
            report.setdefault('split_file_mapping', {})[name] = local_rows
            subset_info['total_tasks'] = len(split_labels)
            (destination/'meta/info.json').write_text(json.dumps(subset_info, indent=2)+'\n')
            # Run converter only on our new subset; never force-write the source.
            cmd = [sys.executable, str(workspace/'scripts/data/convert_lerobot_to_gear.py'),
                   '--dataset-path', str(destination), '--embodiment-tag', 'cruzr_s2',
                   '--state-keys', '{"joint_pos":[0,7],"gripper_pos":[7,8]}',
                   '--action-keys', '{"joint_pos":[0,7],"gripper_pos":[7,8]}',
                   '--relative-action-keys', 'joint_pos', 'gripper_pos', '--task-key', 'annotation.task',
                   '--action-horizon', '24']
            with (output/f'convert-{name}.log').open('w') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            # Converter episodes are intentionally local contiguous ids. The
            # source id mapping is retained in split_file_mapping above.
            episodes_path = destination/'meta/episodes.jsonl'
            generated = [json.loads(line) for line in episodes_path.read_text().splitlines()]
            if len(generated) != len(rows):
                raise ValueError('Converter episode count changed')
            if [ep['episode_index'] for ep in generated] != list(range(len(rows))):
                raise ValueError('Converter episode ids are not contiguous')
            # Rebuild task_index from the generated local vocabulary, rather
            # than retaining incompatible indices from the original dataset.
            generated_tasks = [json.loads(line) for line in (destination/'meta/tasks.jsonl').read_text().splitlines()]
            task_ids = {ep['task']: ep['task_index'] for ep in generated_tasks}
            for path in sorted((destination/'data').glob('chunk-*/episode_*.parquet')):
                table = pd.read_parquet(path)
                table['task_index'] = [task_ids[text] for text in table['annotation.task']]
                table.to_parquet(path, index=False)
        for name in ('stats.json', 'relative_stats_dreamzero.json'):
            shutil.copy2(output/'datasets/train/meta'/name, output/'datasets/validation/meta'/name)
        report['validation_normalization'] = 'training subset statistics only'
        report['prepared_files'] = {str(path.relative_to(output)): digest(path)
                                    for path in sorted((output/'datasets').rglob('*')) if path.is_file()}
        report['success'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        (output/'prepare-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(output/'prepare-report.json')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
