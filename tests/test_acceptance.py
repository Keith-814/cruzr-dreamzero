"""Server acceptance must not report success when tests were skipped."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_tests import result_summary


class AcceptanceTest(unittest.TestCase):
    def test_skip_failure_and_empty_suite_are_not_success(self):
        result = unittest.TestResult()
        self.assertFalse(result_summary(result)['success'])
        case = unittest.FunctionTestCase(lambda: None)
        result.startTest(case)
        result.addSuccess(case)
        result.stopTest(case)
        self.assertTrue(result_summary(result)['success'])
        result.addSkip(case, 'missing physics dependency')
        self.assertFalse(result_summary(result)['success'])
        self.assertEqual(result_summary(result)['skipped'][0]['reason'], 'missing physics dependency')
