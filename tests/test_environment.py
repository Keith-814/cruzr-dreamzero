import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class EnvironmentTest(unittest.TestCase):
    def test_named_joint_step_is_fixed_period_and_limited(self):
        from cruzr_sim import CruzrEnvironment

        env = CruzrEnvironment()
        try:
            obs = env.reset()
            target = obs["joint_position"].copy()
            target[0] += 0.1
            before = float(env.data.time)
            after = env.step(target, 0.04)
            self.assertAlmostEqual(float(env.data.time) - before, env.period)
            self.assertEqual(after["joint_position"].shape, (7,))
            self.assertGreater(after["gripper_width_m"], 0.0)
            self.assertLessEqual(after["gripper_width_m"], 0.045)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
