"""Release relocation and tamper refusal, without ROS or physics."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from cruzr_sim.deployment import code_inventory, create_release, digest, portable_receipt, verify_release
from cruzr_sim.task import WAYPOINTS

ROOT = Path(__file__).parents[1]


class DeploymentTest(unittest.TestCase):
    def test_receipt_survives_copy_and_refuses_added_source_or_changed_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'original'
            (root/'src').mkdir(parents=True)
            (root/'src/runtime.py').write_text('value = 1\n')
            outputs = root/'outputs'
            outputs.mkdir()
            profile = json.loads((ROOT/'configs/hardware.example.json').read_text())
            for key in profile['commissioning']:
                profile['commissioning'][key] = True
            profile['commissioning'].update(calibration_id='synthetic-test-only', scanner_verified=False)
            for key in ('health_topic', 'stop_request_topic', 'head_image_topic', 'wrist_image_topic'):
                profile['topics'][key] = '/test/'+key
            plan = {'joint_names': profile['joint_names'], 'calibration_id': 'synthetic-test-only',
                    'waypoints': {name: [0.]*7 for name in ('start', *WAYPOINTS)}}
            (root/'profile.json').write_text(json.dumps(profile))
            (root/'plan.json').write_text(json.dumps(plan))
            task = outputs/'task.json'
            task.write_text(json.dumps({'task_mode': 'present', 'model_inference': False,
                                       'backend': 'mujoco-oracle', 'success': True,
                                       'motion_complete': True, 'placement_verified': True,
                                       'config': {'test': 'fixture'}}))
            sim = outputs/'sim.json'
            inventory = code_inventory(root)
            sim.write_text(json.dumps({'success': True, 'render_requested': True,
                                      'source_sha256': inventory, 'code_inventory': inventory,
                                      'present_task_report_sha256': digest(task),
                                      'config': {'test': 'fixture'}}))
            receipt = outputs/'release.json'
            receipt.write_text(json.dumps(portable_receipt(create_release(
                sim, task, root/'profile.json', root/'plan.json', root), receipt)))
            copied = Path(directory)/'copied'
            shutil.copytree(root, copied)
            shutil.rmtree(root)  # No fallback to the original absolute paths.
            verify_release(copied/'outputs/release.json', copied/'profile.json', copied/'plan.json', copied,
                           task_mode='present')
            (copied/'src/new_runtime.py').write_text('extra = True\n')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                verify_release(copied/'outputs/release.json', copied/'profile.json', copied/'plan.json', copied)
            (copied/'src/new_runtime.py').unlink()
            record = json.loads((copied/'outputs/release.json').read_text())
            record['task_mode'] = 'scan'
            (copied/'outputs/release.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'task mode'):
                verify_release(copied/'outputs/release.json', copied/'profile.json', copied/'plan.json', copied)
