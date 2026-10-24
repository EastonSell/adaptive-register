#!/usr/bin/env python3
"""Verify that runner output paths and preflight checks cannot mutate its input."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
RUNNER = Path(__file__).with_name('run_adversarial_tests.py')


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class RunnerSafety(unittest.TestCase):
    package_root = None

    def check_rejected(self, report_name=None, unlisted=False):
        with tempfile.TemporaryDirectory(prefix='reader-runner-safety-') as tmp:
            root = Path(tmp) / 'package'
            shutil.copytree(self.package_root, root)
            if unlisted:
                (root / 'unexpected.txt').write_text('unexpected\n', encoding='utf-8')
            before = snapshot(root)
            cmd = [sys.executable, '-B', str(RUNNER), '--package-root', str(root), '--mode', 'characterize']
            if report_name is not None:
                cmd += ['--report', str(root / report_name)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(before, snapshot(root), 'Runner changed its historical input')
            self.assertNotIn('PASS baseline', result.stdout, 'Rejected input should not be executed')

    def test_report_cannot_overwrite_historical_aggregate(self):
        self.check_rejected('results/aggregate.json')

    def test_report_cannot_add_unlisted_historical_file(self):
        self.check_rejected('new-report.json')

    def test_unlisted_source_file_is_rejected_before_execution(self):
        self.check_rejected(unlisted=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    args = parser.parse_args()
    RunnerSafety.package_root = args.package_root.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)
