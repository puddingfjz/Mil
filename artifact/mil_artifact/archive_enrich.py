"""Extract all-file timing fields without loading or publishing request text."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .archive import bounded_metadata
from .config import repo_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSONL file, or - to stream to stdout')
    parser.add_argument('--reuse', type=Path, help='Completed metadata JSONL from an interrupted scan, or - for stdin')
    args = parser.parse_args()
    files = sorted(p for p in (repo_root() / 'benchmarks/test_end2end_schedule').glob('*.log') if not p.is_symlink())
    old = {}
    if args.reuse:
        previous = sys.stdin.read() if str(args.reuse) == '-' else args.reuse.read_text()
        for line in previous.splitlines():
            record = json.loads(line)
            if record['metadata_scan'] != 'all_file_numeric_markers':
                # Empty files in older scans used the bounded label; rescan
                # them, and never mistake a bounded nonempty scan for complete.
                continue
            old[record['path']] = record
    pattern = r'^(total running time:|Total search time:|Total time for preparation before search:|begin fusing similar models:|finish fusing similar models:|begin search:|other hybridflow res infor:|Best group seq:|update_workload_dict min_cost:|Cost_Query_|Plan_Group_|Iter_Search_Time|Select_Best_Time|real model_paths:)'
    matcher = (['rg', '--no-mmap', '--text', '--no-heading', '--no-filename', '--color', 'never']
               if shutil.which('rg') else ['grep', '--text', '--extended-regexp'])
    def read(path):
        relative = path.relative_to(repo_root()).as_posix()
        if relative in old:
            return old[relative]
        # Ripgrep scans the large files once with a compiled multi-pattern
        # matcher; only numeric diagnostics, never request text, reach Python.
        # Sequential reads avoid page-fault-heavy mmap traversal on the shared
        # filesystem; only numeric lines are retained, with bounded memory.
        result = subprocess.run(matcher + [pattern, str(path)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env={**os.environ, 'LC_ALL': 'C'})
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode(errors='replace'))
        with path.open('rb') as stream:
            head = stream.read(2 * 1024**2)
        record = bounded_metadata(head, result.stdout)
        record['metadata_scan'] = 'all_file_numeric_markers'
        import re
        updates = re.findall(rb'update_workload_dict min_cost:\s*([\d.eE+-]+)', result.stdout)
        if record['metrics'].get('initial_predicted_seconds') is not None:
            record['metrics']['incremental_predicted_seconds'] = record['metrics']['initial_predicted_seconds'] + sum(float(v) for v in updates)
        return record
    output_context = nullcontext(sys.stdout) if str(args.output) == '-' else args.output.open('x')
    with output_context as stream, ThreadPoolExecutor(4) as pool:
        for i, (path, record) in enumerate(zip(files, pool.map(read, files)), 1):
            record['path'] = path.relative_to(repo_root()).as_posix()
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if i % 100 == 0:
                print(f'enriched {i}/{len(files)}', file=sys.stderr, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
