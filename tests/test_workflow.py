"""Task failure paths must not release early or fabricate barcode success."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from cruzr_sim.task import PresentTaskConfig, ScanTaskConfig, run_pick_present_return, run_pick_scan_return
from cruzr_sim.scanning import FixtureScanner


class FakeBackend:
    def __init__(self, fail=None):
        self.calls, self.sequence, self.fail = [], 0, fail
    def move(self, name, width, *, holding=False):
        self.calls.append(name)
        if name == self.fail:
            raise RuntimeError('injected motion fault')
    def grip(self, width, *, grasp=False):
        self.calls.append('close' if grasp else 'open')
        if grasp and self.fail == 'close':
            raise RuntimeError('empty grasp')
    def wait(self, seconds, *, holding=False):
        self.calls.append('wait')
    def capture(self):
        self.sequence += 1
        return {'id': self.sequence, 'images': {}}
    def verify_return(self):
        return True
    def stop(self):
        self.calls.append('stop')


class WorkflowTest(unittest.TestCase):
    def test_present_task_has_no_scanner_dependency(self):
        backend = FakeBackend()
        result = run_pick_present_return(backend, PresentTaskConfig(dwell_s=0))
        self.assertTrue(result.success)
        self.assertEqual(backend.calls[:3], ['open', 'pregrasp', 'grasp'])
        self.assertEqual(result.barcode, None)
        self.assertIn('present', result.phases)
        self.assertLess(backend.calls.index('scan'), backend.calls.index('return_high'))

    def test_return_is_after_scan_and_release_is_after_return_low(self):
        backend = FakeBackend()
        result = run_pick_scan_return(backend, FixtureScanner('ABC'))
        self.assertTrue(result.success)
        self.assertEqual(result.scan_source, 'fixture')
        self.assertLess(backend.calls.index('scan'), backend.calls.index('return_high'))
        self.assertLess(backend.calls.index('return_low'), len(backend.calls)-1-backend.calls[::-1].index('open'))
        self.assertEqual(backend.calls[-1], 'stop')

    def test_scan_failure_still_returns_but_is_not_success(self):
        backend = FakeBackend()
        def broken_scanner(_):
            raise RuntimeError('decoder unavailable')
        result = run_pick_scan_return(backend, broken_scanner)
        self.assertFalse(result.success)
        self.assertTrue(result.motion_complete)
        self.assertEqual(len(result.scan_attempts), 3)
        self.assertIn('return_low', backend.calls)
        self.assertIn('open', backend.calls)

    def test_drop_or_motion_fault_does_not_automatically_return_or_release(self):
        for failure in ('close', 'lift', 'scan'):
            backend = FakeBackend(failure)
            result = run_pick_scan_return(backend, FixtureScanner('ABC'))
            self.assertFalse(result.success)
            self.assertNotIn('return_low', backend.calls)
            self.assertEqual(backend.calls.count('open'), 1)  # Initial opening only; never release after fault.
            self.assertEqual(backend.calls[-1], 'stop')

    def test_ambiguous_or_wrong_barcode_is_not_accepted(self):
        for codes in ([{'text': 'A'}, {'text': 'B'}], [{'text': 'wrong'}]):
            backend = FakeBackend()
            result = run_pick_scan_return(backend, lambda _: {'codes': codes, 'source': 'camera'},
                                         ScanTaskConfig(expected_code='expected'))
            self.assertFalse(result.success)
            self.assertIsNone(result.barcode)
            self.assertTrue(result.motion_complete)

    def test_no_placement_evidence_is_not_task_success(self):
        backend = FakeBackend()
        backend.verify_return = lambda: None
        result = run_pick_scan_return(backend, FixtureScanner('A'))
        self.assertTrue(result.motion_complete)
        self.assertFalse(result.success)

    def test_repeated_frames_abort_without_silent_retries(self):
        backend = FakeBackend()
        backend.capture = lambda: {'id': 1, 'images': {}}
        result = run_pick_scan_return(backend, lambda _: {'codes': []})
        self.assertFalse(result.motion_complete)
        self.assertIn('Repeated', result.error)
