"""Run a CPU method matrix with no wall-clock limit unless explicitly requested."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from .schema import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, action='append', required=True)
    parser.add_argument('--method', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=None,
                        help='Optional diagnostic time limit in seconds; default: no limit.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be between 1 and 4')
    if args.timeout is not None and args.timeout < 1:
        parser.error('timeout must be positive when explicitly supplied')
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for request in args.requests:
        app = json.loads(request.read_text())['application']
        for method in args.method:
            actual_method = 'mil_chain_summary' if method == 'mil' and app == 'chain_summary' else method
            tasks.append((request.resolve(), app, actual_method))
    if len({(a, m) for _, a, m in tasks}) != len(tasks):
        parser.error('duplicate application/method; use separate batches for separate settings')
    def run(task):
        request, app, method = task
        destination = args.output.resolve() / (app + '_' + method)
        command = [sys.executable, '-m', 'artifact.mil_artifact.cpu_schedule', 'run',
                   '--requests', str(request), '--method', method, '--output', str(destination),
                   '--cpus', '2']
        if args.timeout is not None:
            command += ['--timeout', str(args.timeout)]
        with destination.with_suffix('.stdout').open('x') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        record = {'application': app, 'method': method, 'exit_code': code, 'output': str(destination),
                  'command': command, 'status': 'completed_cpu_scheduling' if code == 0 else 'failed_or_timeout'}
        print(json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, tasks))
    atomic_write_json(args.output / 'batch.json', {'schema_version': 1, 'gpu_execution': False, 'tasks': results})
    return 0 if all(r['exit_code'] == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
