"""Portable CPU reference data, with resources identified by role rather than host path."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .config import artifact_root, load_yaml
from .hashing import sha256_file
from .source_contract import source_path


RESOURCE_ROLES = {'cost_table', 'output_lengths', 'model_configurations'}
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def read_pinned(root, relative, digest):
    path = source_path(root, relative)
    if not isinstance(digest, str) or not SHA256.fullmatch(digest) or sha256_file(path) != digest:
        raise ValueError(f'input checksum mismatch: {relative}')
    return json.loads(path.read_text())


def validate_references(reference):
    if reference.get('schema_version') != 2 or reference.get('kind') != 'cpu_regression_reference':
        raise ValueError('unsupported CPU reference schema')
    resources = reference['resources']
    if set(resources) != RESOURCE_ROLES:
        raise ValueError('CPU references must identify all resource roles')
    for item in resources.values():
        if not SHA256.fullmatch(item['sha256']):
            raise ValueError('invalid resource checksum')
    identities, names = set(), set()
    if not reference['cases']:
        raise ValueError('empty CPU reference')
    for case in reference['cases']:
        identity = (case['method'], case['request_set_sha256'])
        if identity in identities or case['case_id'] in names:
            raise ValueError('duplicate CPU reference')
        identities.add(identity)
        names.add(case['case_id'])
        if not SHA256.fullmatch(case['request_set_sha256']):
            raise ValueError('invalid request checksum')
        recorded = case['recorded_resource_hashes']
        if not {'cost_table','output_lengths'}.issubset(recorded) or set(recorded) - RESOURCE_ROLES:
            raise ValueError('missing historical resource identity')
        if any(resources[role]['sha256'] != digest for role,digest in recorded.items()):
            raise ValueError('conflicting historical resource identity')
        plan = case['actual']
        if not isinstance(plan['plans'], list) or not plan['plans'] or not all(isinstance(g,list) and g for g in plan['plans']):
            raise ValueError('invalid reference plans')
        cost = plan['predicted_seconds']
        numeric = lambda n: type(n) in (int,float) and math.isfinite(n) and n >= 0
        if cost is not None and not numeric(cost):
            raise ValueError('invalid reference cost')
        if 'stage_times' in plan:
            times = plan['stage_times']
            if (len(times) != len(plan['plans']) or not all(numeric(t) for t in times)
                    or cost is None or not math.isclose(sum(times), cost, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError('inconsistent reference stage costs')
    return reference


def load_reference(path):
    """Read an explicitly supplied reference; no experiment results are bundled."""
    from .plot_entry import safe_path
    path = safe_path(path)
    return validate_references(json.loads(path.read_text())), path


def validate_request(request):
    if request.get('kind') != 'normal_request_set':
        raise ValueError('expected a normal request set')
    nodes = set(range(len(request['model_paths'])))
    if not nodes:
        raise ValueError('empty model set')
    for key in ('input_lengths', 'input_edges'):
        if set(request[key]) != {str(n) for n in nodes}:
            raise ValueError('incomplete node fields: ' + key)
    external = set()
    graph = {}
    for node, edges in request['input_edges'].items():
        if any(type(v) is not int or (v >= 0 and v not in nodes) for v in edges):
            raise ValueError('unknown dependency node')
        graph[int(node)] = {v for v in edges if v >= 0}
        external.update(v for v in edges if v < 0)
    if set(request['inp_seq_ids_dict']) != {str(n) for n in nodes | external}:
        raise ValueError('incomplete node fields: inp_seq_ids_dict')
    for node in nodes:
        lengths = request['input_lengths'][str(node)]
        ids = request['inp_seq_ids_dict'][str(node)]
        if len(lengths) != len(ids) or any(type(v) is not int or v < 0 for v in lengths):
            raise ValueError('invalid input lengths or request IDs')
    pending = set(nodes)
    while pending:
        ready = {n for n in pending if not (graph[n] & pending)}
        if not ready:
            raise ValueError('cyclic request dependency graph')
        pending -= ready
