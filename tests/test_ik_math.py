"""Rotation errors must remain meaningful at the 180-degree singularity."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from cruzr_sim.ik import ArmIK, rotation_vector


class IKMathTest(unittest.TestCase):
    def test_half_turn_has_nonzero_error_along_rotation_axis(self):
        for axis in (np.array([1., 0., 0.]), np.array([1., 2., -3.])/np.sqrt(14)):
            rotation = 2*np.outer(axis, axis)-np.eye(3)
            error = rotation_vector(rotation)
            self.assertAlmostEqual(np.linalg.norm(error), np.pi, places=7)
            self.assertAlmostEqual(abs(np.dot(error/np.pi, axis)), 1, places=7)

    def test_near_half_turn_and_identity(self):
        for angle in (0, 1e-8, 0.3, np.pi-1e-7, -np.pi+1e-7):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            np.testing.assert_allclose(rotation_vector(rotation), [0, 0, angle], atol=1e-8, rtol=0)

    def test_invalid_target_fails_before_accessing_physics(self):
        # Input validation is independent of a model and must run first.
        solver = ArmIK.__new__(ArmIK)
        for kwargs in ({'target_pos': [np.nan, 0, 0]},
                       {'target_rot': -np.eye(3)}, {'target_rot': np.zeros((3, 3))},
                       {'q_seed': [np.inf]*7}, {'iters': 0}):
            arguments = {'target_pos': np.zeros(3), 'target_rot': np.eye(3), **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.solve(**arguments)
