import unittest
from unittest.mock import patch
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class EnvironmentTest(unittest.TestCase):
    def make_env(self):
        from cruzr_sim import CruzrEnvironment
        env = CruzrEnvironment(os.environ.get('CRUZR_TEST_CONFIG'))
        self.addCleanup(env.close)
        return env

    def test_named_joint_step_is_fixed_period_and_limited(self):
        env = self.make_env()
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

    def test_pi05_action_contract_uses_absolute_targets(self):
        env = self.make_env()
        try:
            obs = env.observe_pi05(images=False)
            self.assertEqual(obs["observation.state"].shape, (8,))
            before = float(env.data.time)
            # The real Pi0.5 data uses 0.05 m for the fully open target while
            # this verified geometry is limited to 0.045 m; the adapter should
            # saturate that one value rather than reject the action.
            action = np.r_[obs["observation.state"][:7], 0.05]
            after = env.step_pi05(action)
            self.assertEqual(after["observation.state"].shape, (8,))
            self.assertAlmostEqual(float(env.data.time) - before, 33*env.model.opt.timestep)
            self.assertTrue(after["gripper_target_clipped"])
            self.assertLessEqual(float(after["observation.state"][-1]), 0.045 + 1e-7)
            chunk = np.repeat(action[None, :], 2, axis=0)
            observations = env.step_pi05_chunk(chunk)
            self.assertEqual(len(observations), 2)
            self.assertEqual(observations[-1]["observation.state"].shape, (8,))
            self.assertAlmostEqual(float(env.data.time)-before, 3/15)
            bad = chunk.copy()
            bad[-1, 0] = np.nan
            before_bad = float(env.data.time)
            with self.assertRaises(ValueError):
                env.step_pi05_chunk(bad)
            self.assertEqual(float(env.data.time), before_bad)
            env.reset()
            self.assertEqual(float(env.data.time), 0)
            self.assertEqual(env.policy_clock.index, 0)
        finally:
            env.close()

    def test_step_exception_latches_fault_and_rejected_call_keeps_clock(self):
        env = self.make_env()
        action = np.r_[env.command, 0.04]
        original_step = mujoco.mj_step

        def fail_after_step(model, data):
            original_step(model, data)
            raise RuntimeError('injected callback failure')

        with patch('mujoco.mj_step', side_effect=fail_after_step):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                env.step_pi05(action)
        self.assertTrue(env.faulted)
        before_time, before_clock = env.data.time, env.policy_clock.index
        with self.assertRaisesRegex(RuntimeError, 'faulted'):
            env.step_pi05(action)
        self.assertEqual(env.data.time, before_time)
        self.assertEqual(env.policy_clock.index, before_clock)
        env.reset()
        env.step_pi05(action)

    def test_derived_poses_are_current_after_step(self):
        env = self.make_env()
        target = env.command.copy()
        target[0] += 0.01
        env.step(target)
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        scratch.qvel[:] = env.data.qvel
        scratch.ctrl[:] = env.data.ctrl
        mujoco.mj_forward(env.model, scratch)
        np.testing.assert_allclose(env.data.xpos, scratch.xpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(env.data.xmat, scratch.xmat, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
