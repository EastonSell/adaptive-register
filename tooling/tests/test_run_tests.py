import unittest
try: import run_tests
except ImportError: run_tests=None
class RunnerTests(unittest.TestCase):
 def test_default_includes_root_and_tooling(self):
  self.assertIsNotNone(run_tests,'aggregate runner is not implemented')
  self.assertEqual(run_tests.suites(False),[('tests','src'),('tooling/tests','pretrained:tooling')])
 def test_pretrained_is_explicit_and_additive(self):
  self.assertIsNotNone(run_tests,'aggregate runner is not implemented')
  self.assertEqual(run_tests.suites(True)[-1],('pretrained/tests','pretrained'))
  self.assertEqual(len(run_tests.suites(True)),3)
