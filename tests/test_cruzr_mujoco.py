# Imported from the local Show-Harness snapshot; see docs/import-manifest.json.
# Modified 2026-11-15: standalone package imports, portable paths, and clarified scope.
"""Regression tests for the Cruzr S2 MuJoCo gripper and actuator setup."""

from pathlib import Path
import os
import unittest
from unittest import mock

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover - main environment intentionally lacks MuJoCo
    mujoco = None

if mujoco is not None:
    from cruzr_sim.mujoco_env import (  # noqa: E402
        CLOSED_FINGER_Q,
        MujocoCruzrRobot,
        OPEN_FINGER_Q,
    )
    from cruzr_sim.scene import gripper_tip  # noqa: E402


XML = Path(os.environ.get('CRUZR_MODEL_PATH', Path(__file__).parents[1] / 'models/cruzr_s2/scene.xml'))


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class CruzrMujocoTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("cruzr_sim.mujoco_env.XML", str(XML))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.robot = MujocoCruzrRobot(verbose=False)
        self.addCleanup(self.robot.close)

    def test_position_actuators_start_at_current_pose(self):
        for aid, jid in self.robot._arm_act.values():
            q = self.robot.data.qpos[self.robot.model.jnt_qposadr[jid]]
            self.assertAlmostEqual(float(self.robot.data.ctrl[aid]), float(q))
        for aid, jid in self.robot._finger_act:
            q = self.robot.data.qpos[self.robot.model.jnt_qposadr[jid]]
            self.assertAlmostEqual(float(q), OPEN_FINGER_Q)
            self.assertAlmostEqual(float(self.robot.data.ctrl[aid]), OPEN_FINGER_Q)

    def test_empty_gripper_closes_without_crossing(self):
        cube = mujoco.mj_name2id(
            self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        cube_joint = int(self.robot.model.body_jntadr[cube])
        qadr = int(self.robot.model.jnt_qposadr[cube_joint])
        self.robot.data.qpos[qadr:qadr + 3] = [0.4, 0.9, 0.76]
        mujoco.mj_forward(self.robot.model, self.robot.data)
        open_width = self.robot.get_gripper_position()[0]
        closed_width = self.robot.control_gripper(True)
        self.assertGreater(open_width, 0.04)
        self.assertLess(closed_width, 0.01)
        for _, jid in self.robot._finger_act:
            q = self.robot.data.qpos[self.robot.model.jnt_qposadr[jid]]
            self.assertGreater(float(q), OPEN_FINGER_Q)
            self.assertLessEqual(float(q), CLOSED_FINGER_Q + 1e-3)
        self.assertTrue(np.isfinite(self.robot.data.qpos).all())

    def test_ik_solve_does_not_teleport_live_arm(self):
        q_before = self.robot.data.qpos[self.robot.ik.qadr].copy()
        pos, rot = self.robot.ik.fk()
        # Restoring only qpos used to leave other physics caches at a trial pose.
        caches = {key: getattr(self.robot.data, key).copy()
                  for key in ("qpos", "qvel", "xpos", "xmat", "xipos", "subtree_com", "cdof")}
        q_target, pos_err, _ = self.robot.ik.solve(pos + [0.0, 0.0, 0.02], rot)
        for key, before in caches.items():
            np.testing.assert_array_equal(getattr(self.robot.data, key), before)
        np.testing.assert_allclose(
            self.robot.data.qpos[self.robot.ik.qadr], q_before, atol=1e-12
        )
        self.assertGreater(np.linalg.norm(q_target - q_before), 1e-4)
        self.assertLess(pos_err, 0.003)

    def test_ik_exception_preserves_live_state(self):
        before = self.robot.data.qpos.copy()
        with self.assertRaises(ValueError):
            self.robot.ik.solve(np.zeros(2), np.eye(3), q_seed=np.zeros(7))
        np.testing.assert_array_equal(self.robot.data.qpos, before)

    def test_physical_grasp_survives_controller_lift_step(self):
        model, data = self.robot.model, self.robot.data
        for _ in range(180):
            mujoco.mj_step(model, data)

        cube_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        cube_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        qadr = model.jnt_qposadr[cube_joint]
        dadr = model.jnt_dofadr[cube_joint]
        tip = gripper_tip(model, data, self.robot.side)
        pose = np.asarray(self.robot.get_ee_pose())
        pose[2] -= float(tip[2] - 0.753)
        self.robot.update_desired_ee_pose(pose)
        tip = gripper_tip(model, data, self.robot.side)
        data.qpos[qadr:qadr + 3] = [tip[0], tip[1], 0.741]
        data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(100):
            mujoco.mj_step(model, data)

        self.robot.control_gripper(True)
        z_grasp = float(data.xpos[cube_body][2])
        self.assertGreater(self.robot.get_gripper_position()[0], 0.03)

        pose = np.asarray(self.robot.get_ee_pose())
        pose[2] += 0.02
        self.robot.update_desired_ee_pose(pose)
        z_lift = float(data.xpos[cube_body][2])
        self.assertGreater(z_lift - z_grasp, 0.005)
        self.assertGreater(self.robot.get_gripper_position()[0], 0.03)


if __name__ == "__main__":
    unittest.main()
