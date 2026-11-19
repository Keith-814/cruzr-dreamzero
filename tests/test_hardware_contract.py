"""No ROS/robot access: exercise refusal, freshness, rate and fault latching."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from cruzr_sim.real_robot import HardwareController, HardwareScanBackend, check_start_state, load_plan, load_profile
from cruzr_sim.task import WAYPOINTS
from cruzr_sim.deployment import check_sources, verify_release

ROOT = Path(__file__).parents[1]


class Transport:
    def __init__(self):
        now = time.monotonic()
        self.sample = {'arm_received_s': now, 'gripper_received_s': now, 'health_received_s': now,
                       'joint_position': [0.]*7, 'joint_velocity': [0.]*7, 'healthy': True,
                       'gripper': {'init_state': 1, 'homed': 1, 'grip_state': 1, 'error_code': 0, 'pos': .05}}
        self.published, self.stop_count = [], 0
    def read_state(self):
        return copy.deepcopy(self.sample)
    def publish(self, q, width):
        self.published.append((q, width))
        raise RuntimeError('injected send failure')
    def stop(self):
        self.stop_count += 1


class HardwareContractTest(unittest.TestCase):
    def setUp(self):
        self.p = load_profile(ROOT/'configs/hardware.example.json')
        self.transport = Transport()

    def test_default_cannot_publish_and_example_is_not_commissioned(self):
        c = HardwareController(self.transport, self.p)
        with self.assertRaisesRegex(RuntimeError, 'Read-only'):
            c.start()
        self.assertFalse(self.transport.published)
        with self.assertRaisesRegex(ValueError, 'commissioning'):
            load_profile(ROOT/'configs/hardware.example.json', commissioned=True)

    def test_present_commissioning_does_not_require_a_scanner(self):
        profile = copy.deepcopy(self.p)
        for key in profile['commissioning']:
            profile['commissioning'][key] = True
        profile['commissioning'].update(calibration_id='test-only', scanner_verified=False)
        for key in ('health_topic', 'stop_request_topic', 'head_image_topic', 'wrist_image_topic'):
            profile['topics'][key] = '/test/' + key
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.json'
            path.write_text(json.dumps(profile))
            load_profile(path, commissioned=True)
            with self.assertRaisesRegex(ValueError, 'commissioning'):
                load_profile(path, commissioned=True, require_scanner=True)

    def test_present_receipt_cannot_authorize_scan_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            path.write_text(json.dumps({'schema': 1, 'task_mode': 'present'}))
            # Refuse the different task before opening evidence or transport.
            with self.assertRaisesRegex(ValueError, 'task mode'):
                verify_release(path, None, None, ROOT, task_mode='scan')

    def test_missing_future_and_stale_receipt_times_are_rejected(self):
        for value in (None, time.monotonic()+1, time.monotonic()-1, float('nan')):
            self.transport.sample['arm_received_s'] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                HardwareController(self.transport, self.p).state()

    def test_gripper_or_supervisor_fault_blocks_execution(self):
        for key, value in (('homed', 0), ('error_code', 1), ('grip_state', 3), ('pos', float('nan'))):
            transport = Transport()
            transport.sample['gripper'][key] = value
            with self.assertRaises(RuntimeError):
                HardwareController(transport, self.p).state()
        self.transport.sample['healthy'] = False
        with self.assertRaises(RuntimeError):
            HardwareController(self.transport, self.p).state()

    def test_lease_expiry_sends_no_queued_targets(self):
        c = HardwareController(self.transport, self.p, enabled=True)
        c.lease = time.monotonic()-5
        c._loop()
        self.assertIn('lease expired', c.error)
        self.assertTrue(c.done.is_set())
        self.assertEqual(self.transport.published, [])
        self.assertEqual(self.transport.stop_count, 1)

    def test_native_send_failure_latches_and_cancels_continuation(self):
        c = HardwareController(self.transport, self.p, enabled=True)
        c.lease = time.monotonic()
        c.q_command = np.zeros(7)
        c.q_goal = np.array([.1]+[0.]*6)
        c.g_command = c.g_goal = .05
        c._loop()
        self.assertIn('injected send failure', c.error)
        self.assertLessEqual(self.transport.published[0][0][0], self.p['max_joint_speed']/self.p['command_hz']+1e-9)
        with self.assertRaises(RuntimeError):
            c.target(np.zeros(7), .045)

    def test_release_fingerprint_cannot_ignore_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                check_sources({}, directory)
            with self.assertRaisesRegex(ValueError, 'changed or missing'):
                check_sources({'source_sha256': {'missing.py': 'bad'}}, directory)

    def test_missing_start_or_moving_or_deviated_start_is_rejected(self):
        plan = {'joint_names': self.p['joint_names'],
                'calibration_id': self.p['commissioning']['calibration_id'],
                'waypoints': {name: [0.]*7 for name in WAYPOINTS}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'start'):
                load_plan(path, self.p)
            plan['waypoints']['start'] = [0.]*7
            path.write_text(json.dumps(plan))
            load_plan(path, self.p)
        state = {'joint_position': [0.]*7, 'joint_velocity': [0.]*7}
        check_start_state(state, plan, self.p)
        state['joint_position'][0] = .2
        with self.assertRaisesRegex(RuntimeError, 'start pose'):
            check_start_state(state, plan, self.p)
        state['joint_position'][0] = 0.
        state['joint_velocity'][0] = .04
        with self.assertRaisesRegex(RuntimeError, 'stationary'):
            check_start_state(state, plan, self.p)

    def test_measured_arrival_cannot_skip_pending_command_ramp(self):
        c = HardwareController(self.transport, self.p, enabled=True)
        c.worker = object()  # Predicate unit test; no publishing thread starts.
        c.q_command = c.q_goal = np.zeros(7)
        c.g_command = c.g_goal = .01
        backend = HardwareScanBackend(c, {'waypoints': {'lift': [.1]+[0.]*6}})
        def inspect_move(predicate):
            feedback = {'joint_position': np.array([.1]+[0.]*6), 'joint_velocity': np.zeros(7)}
            self.assertFalse(predicate(feedback))
            c.q_command = np.array([.1]+[0.]*6)
            self.assertTrue(predicate(feedback))
        backend._until = inspect_move
        backend.move('lift', .01)
        def inspect_gripper(predicate):
            feedback = {'gripper': {'grip_state': 1, 'pos': c.g_goal}}
            self.assertFalse(predicate(feedback))
            c.g_command = c.g_goal
            self.assertTrue(predicate(feedback))
        backend._until = inspect_gripper
        backend.grip(self.p['gripper_calibration']['geometric_max_m'])
