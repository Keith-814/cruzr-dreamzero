"""Optional data conversion regression, run in the model/data-tools environment.

python tests/integration_finetune.py
Not part of simulation unittest discovery: pandas/pyarrow/video tools are kept
out of the simulation environment. Uses synthetic data, never real weights.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cruzr_sim.dataset import CAMERAS
from cruzr_sim.deployment import digest


class PreparationIntegrationTest(unittest.TestCase):
    def test_split_renumbers_noncontiguous_episodes_and_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, prepared = root/'source', root/'prepared'
            (source/'meta').mkdir(parents=True)
            (source/'data/chunk-000').mkdir(parents=True)
            contract = json.loads((ROOT/'configs/dataset.example.json').read_text())
            contract.update(semantics_verified=True, observation_precedes_action=True)
            (root/'contract.json').write_text(json.dumps(contract))
            features = {key: {'dtype': 'float32', 'shape': [8], 'names': contract['joint_names']}
                        for key in ('observation.state', 'action')}
            features.update({key: {'dtype': 'int64', 'shape': [1]}
                             for key in ('episode_index', 'frame_index', 'task_index')})
            features['timestamp'] = {'dtype': 'float32', 'shape': [1]}
            features.update({key: {'dtype': 'video', 'shape': [3, 16, 16]} for key in CAMERAS})
            labels = ['Pick up, present at face, and return object A.',
                      'Pick up, present at face, and return object B.']
            info = {'codebase_version': 'v2.1', 'fps': 15, 'total_episodes': 4,
                    'total_frames': 100, 'chunks_size': 1000, 'features': features}
            (source/'meta/info.json').write_text(json.dumps(info))
            # Non-local task ids, and no annotation.task/global index column.
            (source/'meta/tasks.jsonl').write_text('\n'.join(
                json.dumps({'task_index': 10+i, 'task': text}) for i, text in enumerate(labels))+'\n')
            for i, episode in enumerate((2, 5, 9, 12)):
                state = np.zeros((25, 8), dtype=np.float32)
                state[:, -1] = .04
                table = pd.DataFrame({'observation.state': list(state), 'action': list(state),
                                      'timestamp': np.arange(25)/15, 'episode_index': episode,
                                      'frame_index': np.arange(25), 'task_index': 10+i%2})
                table.to_parquet(source/f'data/chunk-000/episode_{episode:06d}.parquet', index=False)
                for camera in CAMERAS:
                    target = source/f'videos/chunk-000/{camera}/episode_{episode:06d}.mp4'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 16))
                    self.assertTrue(writer.isOpened(), 'Synthetic video encoder unavailable')
                    for frame in range(25):
                        writer.write(np.full((16, 16, 3), frame+i, np.uint8))
                    writer.release()
            original = {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()}
            subprocess.run([sys.executable, str(ROOT/'scripts/prepare_finetune.py'),
                            '--dataset', str(source), '--contract', str(root/'contract.json'),
                            '--output-dir', str(prepared), '--validation-fraction', '0.5'],
                           check=True, capture_output=True, text=True)
            report = json.loads((prepared/'prepare-report.json').read_text())
            self.assertTrue(report['success'], report)
            for split, mapping in report['split_file_mapping'].items():
                data = prepared/'datasets'/split
                meta = json.loads((data/'meta/info.json').read_text())
                vocabulary = {t['task_index']: t['task'] for t in
                              map(json.loads, (data/'meta/tasks.jsonl').read_text().splitlines())}
                episodes = list(map(json.loads, (data/'meta/episodes.jsonl').read_text().splitlines()))
                self.assertEqual([ep['episode_index'] for ep in episodes], list(range(len(mapping))))
                offset = 0
                for row in mapping:
                    local, old = row['local_episode_index'], row['original_episode_index']
                    table = pd.read_parquet(data/f'data/chunk-000/episode_{local:06d}.parquet')
                    old_table = pd.read_parquet(source/f'data/chunk-000/episode_{old:06d}.parquet')
                    expected = [labels[int(i)-10] for i in old_table['task_index']]
                    self.assertEqual(table['episode_index'].tolist(), [local]*25)
                    self.assertEqual(table['index'].tolist(), list(range(offset, offset+25)))
                    self.assertEqual(table['frame_index'].tolist(), list(range(25)))
                    self.assertEqual(table['annotation.task'].tolist(), expected)
                    self.assertEqual([vocabulary[int(i)] for i in table['task_index']], expected)
                    offset += 25
                    for camera in CAMERAS:
                        self.assertEqual(digest(data/f'videos/chunk-000/{camera}/episode_{local:06d}.mp4'),
                                         digest(source/f'videos/chunk-000/{camera}/episode_{old:06d}.mp4'))
                self.assertEqual(meta['total_frames'], offset)
                self.assertEqual(meta['total_tasks'], len(vocabulary))
                self.assertEqual(meta['total_videos'], 2*len(mapping))
                self.assertEqual(meta['total_chunks'], 1)
                self.assertIn('index', meta['features'])
            for name in ('stats.json', 'relative_stats_dreamzero.json'):
                self.assertEqual(digest(prepared/'datasets/train/meta'/name),
                                 digest(prepared/'datasets/validation/meta'/name))
            self.assertEqual(original, {p.relative_to(source): digest(p) for p in source.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
