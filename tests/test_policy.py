"""Meaningful timing/input regression tests; no robot assets, GL, or MuJoCo."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cruzr_sim.policy import SampleClock, validate_chunk


class PolicyContractTest(unittest.TestCase):
    def test_15hz_runs_at_15hz_without_accumulated_drift(self):
        clock = SampleClock(0.002, 15)
        steps = 0
        for index in range(1, 15001):
            delta = clock.next_steps()
            self.assertIn(delta, (33, 34))
            steps += delta
            self.assertLessEqual(abs(steps * 0.002-index/15), 0.00100001)
        self.assertAlmostEqual(steps*0.002, 1000)

    def test_nan_or_invalid_last_action_is_rejected_before_execution(self):
        original = np.zeros((50, 8))
        original[-1, 1] = np.nan
        with self.assertRaises(ValueError):
            validate_chunk(original, [-1]*7, [1]*7)
        original[-1, 1] = 1.01
        with self.assertRaises(ValueError):
            validate_chunk(original, [-1]*7, [1]*7)

    def test_absolute_limits_and_input_copy(self):
        source = np.array([[0]*7+[0.05]], dtype=np.float32)
        chunk = validate_chunk(source, [-1]*7, [1]*7)
        self.assertAlmostEqual(chunk[0, -1], 0.05)
        chunk[0, 0] = 0.5
        self.assertEqual(source[0, 0], 0)
        for bad in (np.zeros(8), np.zeros((0, 8)), np.zeros((51, 8)), np.zeros((2, 7))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_chunk(bad, [-1]*7, [1]*7)

    def test_unrepresentable_sample_rates_are_rejected(self):
        for dt, hz in ((0.002, 501), (0.002, 0), (float('nan'), 15)):
            with self.assertRaises(ValueError):
                SampleClock(dt, hz)
