"""Location-independent fingerprints for release source verification."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

from .hashing import sha256_file


DIRECTIVE = re.compile(r'^#\s*(?:.*?\bcoding[:=]|type:|noqa\b|fmt:|isort:|pylint:|'
                       r'pyright:|mypy:|ruff:|pragma:|Copyright\b|SPDX-|Licensed under\b)', re.I)
SHEBANG = re.compile(r'^#![ \t]*/\S+')


def protected_path_parts(parts):
    """Shared lexical guard; never inspect a protected filesystem object."""
    return any('backup' in p.lower() or 'camera-ready' in p.lower()
               or p.lower() in ('.git', '.git_ori') or p.lower().endswith('_copy') for p in parts)


def source_path(root, relative):
    """Validate a repository-relative path before inspecting its filesystem objects."""
    relative = Path(relative)
    root = Path(root).absolute()
    parts = relative.parts
    if (relative.is_absolute() or not parts or '..' in (root.parts + parts)
            or protected_path_parts(root.parts + parts)):
        raise ValueError('invalid or protected source path')
    path = root
    for part in ('', *parts):
        path = path / part
        if path.is_symlink():
            raise ValueError('source symlinks are not accepted')
    return path


def fingerprint(source):
    """Keep executable AST, docstrings and special directives, but not positions."""
    if isinstance(source, bytes):
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        source = source.decode(encoding)
    tree = ast.parse(source, type_comments=True, feature_version=9)
    for node in ast.walk(tree):
        if isinstance(node, ast.TypeIgnore):
            node.lineno = 0  # The directive's token anchor below preserves its owner.
    # Empty type_params is an AST addition in newer Python, not a source change.
    def canonical(node):
        if isinstance(node, ast.AST):
            return [type(node).__name__, [(key, canonical(value)) for key, value in ast.iter_fields(node)
                    if not (key == 'type_params' and not value)]]
        if isinstance(node, list):
            return [canonical(value) for value in node]
        if isinstance(node, (bytes, complex)) or node is Ellipsis:
            return [type(node).__name__, repr(node)]
        return node
    directives, position = [], 0
    lines = source.splitlines()
    ignored = {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            shebang = token.start == (1, 0) and SHEBANG.match(token.string)
            if shebang or DIRECTIVE.match(token.string):
                inline = bool(lines[token.start[0] - 1][:token.start[1]].strip())
                prologue = token.start[0] if shebang or re.match(r'^#.*\bcoding[:=]', token.string) else None
                directives.append([position, inline, prologue, token.string])
        elif token.type not in ignored:
            position += 1
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True).encode()).hexdigest()
    header = []
    for line in source.splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            break
        header.append(line)
    legal = '\n'.join(header).strip()
    if not re.search(r'copyright|licensed under|SPDX|redistribution and use|permission is hereby', legal, re.I):
        legal = ''
    return {'ast_sha256': digest(canonical(tree)), 'directives_sha256': digest(directives),
            'legal_header_sha256': digest(legal)}


def verify_source_contract(root, contract):
    if contract.get('schema_version') != 1 or contract.get('kind') != 'source_semantics':
        raise ValueError('unsupported source contract')
    files = contract['files']
    if not files or len({f['path'] for f in files}) != len(files):
        raise ValueError('empty or duplicate source contract')
    issues = []
    for record in files:
        path = source_path(root, record['path'])
        try:
            if record['comparison'] == 'python_ast_and_directives':
                valid = fingerprint(path.read_bytes()) == record['fingerprint']
            elif record['comparison'] == 'bytes':
                valid = sha256_file(path) == record['sha256']
            else:
                raise ValueError('unsupported source comparison')
        except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as error:
            issues.append({'path': record['path'], 'error': str(error)})
            continue
        if not valid:
            issues.append({'path': record['path'], 'error': 'source contract mismatch'})
    return {'passed': not issues, 'checked_files': len(files), 'issues': issues,
            'scope': 'Source equivalence, not new GPU or performance validation.'}
