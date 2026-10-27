"""Test outer time-limit policy without importing or running GPU workloads."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from artifact.mil_artifact import cpu_batch, cpu_schedule


class CpuTimeoutPolicyTest(unittest.TestCase):
    def invoke(self, module, arguments):
        with mock.patch('sys.argv', [module.__name__] + arguments), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return module.main()

    def test_single_run_waits_without_limit_by_default(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(cpu_schedule.subprocess, 'Popen') as popen:
            process = popen.return_value
            process.wait.return_value = 0
            result = self.invoke(cpu_schedule, [
                'run', '--requests', str(Path(directory) / 'requests.json'),
                '--output', str(Path(directory) / 'run')])
            self.assertEqual(result, 0)
            process.wait.assert_called_once_with(timeout=None)
            self.assertNotIn('--timeout', popen.call_args.args[0])
            process.send_signal.assert_not_called()

    def test_explicit_diagnostic_limit_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(cpu_schedule.subprocess, 'Popen') as popen:
            popen.return_value.wait.return_value = 0
            self.assertEqual(self.invoke(cpu_schedule, [
                'run', '--requests', str(Path(directory) / 'requests.json'),
                '--output', str(Path(directory) / 'run'), '--timeout', '7']), 0)
            popen.return_value.wait.assert_called_once_with(timeout=7)
            self.assertEqual(popen.call_args.args[0][-2:], ['--timeout', '7'])

    def test_failure_without_limit_records_null_not_timeout(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(cpu_schedule.subprocess, 'Popen') as popen:
            requests = Path(directory) / 'requests.json'
            requests.write_text('{}')
            output = Path(directory) / 'run'
            popen.return_value.wait.return_value = 1
            self.assertEqual(self.invoke(cpu_schedule, [
                'run', '--requests', str(requests), '--output', str(output)]), 1)
            report = json.loads((output / 'failure.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIsNone(report['timeout_seconds'])

    def test_explicit_diagnostic_timeout_remains_distinct(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(cpu_schedule.subprocess, 'Popen') as popen:
            requests = Path(directory) / 'requests.json'
            requests.write_text('{}')
            output = Path(directory) / 'run'
            popen.return_value.wait.side_effect = [subprocess.TimeoutExpired('worker', 7), 15]
            self.assertEqual(self.invoke(cpu_schedule, [
                'run', '--requests', str(requests), '--output', str(output),
                '--timeout', '7']), 1)
            report = json.loads((output / 'failure.json').read_text())
            self.assertEqual(report['status'], 'timeout')
            self.assertEqual(report['timeout_seconds'], 7)

    def test_batch_only_forwards_explicit_limits(self):
        for limit in (None, 7):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(cpu_batch.subprocess, 'call', return_value=0) as call:
                requests = Path(directory) / 'requests.json'
                requests.write_text(json.dumps({'application': 'llm_blender'}))
                output = Path(directory) / 'batch'
                arguments = ['--requests', str(requests), '--method', 'mil',
                             '--output', str(output)]
                if limit is not None:
                    arguments += ['--timeout', str(limit)]
                self.assertEqual(self.invoke(cpu_batch, arguments), 0)
                command = call.call_args.args[0]
                if limit is None:
                    self.assertNotIn('--timeout', command)
                else:
                    self.assertEqual(command[-2:], ['--timeout', str(limit)])
                report = json.loads((output / 'batch.json').read_text())
                self.assertEqual(report['tasks'][0]['command'], command)

    def test_nonpositive_explicit_limits_are_rejected_before_launch(self):
        for module in (cpu_batch, cpu_schedule):
            for limit in ('0', '-1'):
                with self.subTest(module=module.__name__, limit=limit), \
                        tempfile.TemporaryDirectory() as directory, \
                        mock.patch.object(module.subprocess, 'Popen') as popen, \
                        mock.patch.object(module.subprocess, 'call') as call:
                    output = Path(directory) / 'run'
                    arguments = ['--requests', str(Path(directory) / 'requests.json'),
                                 '--output', str(output), '--timeout', limit]
                    arguments = (['run'] if module is cpu_schedule else ['--method', 'mil']) + arguments
                    with self.assertRaises(SystemExit) as error:
                        self.invoke(module, arguments)
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(output.exists())
                    popen.assert_not_called()
                    call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
