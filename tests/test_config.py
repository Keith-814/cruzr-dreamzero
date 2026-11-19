"""Configuration errors must fail before any renderer or robot is initialized."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
try:
    import tomllib
except ImportError:
    tomllib = None


@unittest.skipIf(tomllib is None, 'Config runtime requires Python 3.11+')
class ConfigTest(unittest.TestCase):
    def load_variant(self, old=None, new=None):
        from cruzr_sim.config import load_config
        path = Path(__file__).parents[1]/'configs/mujoco.toml'
        text = path.read_text()
        if old:
            self.assertIn(old, text)
            text = text.replace(old, new)
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/'variant.toml'
            config.write_text(text)
            return load_config(config)

    def test_default_accepts_negative_finger_travel(self):
        config = self.load_variant()
        self.assertLess(config['gripper']['zero_gap'], config['gripper']['max_width'])
        self.assertTrue(Path(config['simulation']['model']).is_absolute())

    def test_rejects_bad_physics_gains_and_fractional_image_sizes(self):
        for old, new in [('physics_dt = 0.002', 'physics_dt = 0.003'),
                         ('cube_mass = 0.04', 'cube_mass = -0.04'),
                         ('head_width = 640', 'head_width = 640.5'),
                         ('action_horizon = 50', 'action_horizon = 50.5')]:
            with self.subTest(change=new), self.assertRaises(ValueError):
                self.load_variant(old, new)

    def test_old_camera_profile_gets_compatible_defaults(self):
        config = self.load_variant('head_source = "front_cam"', '')
        self.assertEqual(config['camera']['head_source'], 'front_cam')

    def test_presentation_target_is_distinct_and_invalid_values_are_rejected(self):
        config = self.load_variant()
        table_z = config['task']['table_position'][2]+config['task']['table_half_size'][2]
        self.assertGreater(config['task']['presentation']['position'][2], table_z+0.12)
        for old, new in [('position = [0.05, 1.02, 1.22]', 'position = [0.05, 1.02]'),
                         ('dwell_s = 1.0', 'dwell_s = -1.0'), ('yaw = 0.0', 'yaw = inf')]:
            with self.subTest(change=new), self.assertRaises(ValueError):
                self.load_variant(old, new)
