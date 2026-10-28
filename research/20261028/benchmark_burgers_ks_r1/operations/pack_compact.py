#!/usr/bin/env python3
"""Pack only bounded code/JSON/CSV/log evidence; reuse an existing verified pack."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

CORE = ('CORE_MANIFEST.json',)
OPERATIONS = ('operations/replay_and_aggregate.py', 'operations/pack_compact.py')
TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.log', '.txt', '.md', '.yaml', '.yml'}
BANNED_WORDS = ('credential', 'cookie', 'password', 'secret', 'token', 'session')
MAX_RECORD_BYTES = 5 * 1024 * 1024
MAX_LOG_BYTES = 512 * 1024
INVENTORY = '_compact_inventory.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def verify_archive(archive, receipt):
    if archive.stat().st_size != receipt['archive_bytes'] or file_sha(archive) != receipt['archive_sha256']:
        raise ValueError('Existing compact archive integrity failed')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or INVENTORY not in names or any(not m.isfile() for m in members):
            raise ValueError('Invalid archive member set/types')
        raw = tar.extractfile(INVENTORY).read()
        if digest(raw) != receipt['inventory_sha256']:
            raise ValueError('Compact inventory identity mismatch')
        inventory = json.loads(raw)
        if set(names) != set(inventory['files']) | {INVENTORY}:
            raise ValueError('Compact exact member set mismatch')
        for name, entry in inventory['files'].items():
            member = tar.getmember(name)
            if name.startswith('/') or '..' in Path(name).parts:
                raise ValueError('Unsafe archive path')
            data = tar.extractfile(member).read()
            if member.size != entry['bytes'] or digest(data) != entry['sha256']:
                raise ValueError('Compact member size/hash mismatch: ' + name)
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    root, analysis, archive = a.root.resolve(strict=True), a.analysis_dir.resolve(strict=True), a.archive.resolve()
    if not analysis.is_relative_to(root) or archive.is_relative_to(analysis) or any(archive.is_relative_to(root / name) for name in ('runs', 'preflight')):
        raise ValueError('Analysis must be inside root; archive must be outside analysis and immutable runs/preflight')
    report_path = analysis / 'AGGREGATE_REPORT.json'
    if not report_path.is_file():
        raise ValueError('Replay/aggregate record required before packing')
    report_sha = file_sha(report_path)
    receipt_path = archive.with_name(archive.name + '.receipt.json')
    if archive.exists() or receipt_path.exists():
        if not archive.exists() or not receipt_path.exists():
            raise FileExistsError('Partial existing pack: inspect instead of replacing')
        receipt = json.loads(receipt_path.read_text())
        verify_archive(archive, receipt)
        if receipt.get('root') != str(root) or receipt.get('analysis_dir') != str(analysis) or receipt.get('aggregate_report_sha256') != report_sha:
            raise ValueError('Existing verified pack belongs to a different root/analysis/report; choose a new archive path without replacing it')
        print(json.dumps({'status': 'REUSED_VERIFIED_EXISTING_COMPACT_ARCHIVE', **receipt}, indent=2))
        return
    selected = {}
    notes = []
    def add(path, required=False):
        path = Path(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Symlink/outside-root evidence rejected')
        name = path.relative_to(root).as_posix()
        if any(word in path.name.lower() for word in BANNED_WORDS):
            return
        if path.suffix not in TEXT_SUFFIXES | {'.py', '.sh'} or path.name.endswith('.pending'):
            return
        if path.stat().st_size > MAX_RECORD_BYTES and path.suffix != '.log':
            raise ValueError('Oversized compact record: ' + name)
        raw = path.read_bytes()
        raw.decode('utf-8')  # Refuse binaries hidden behind a text extension.
        if path.suffix == '.log' and len(raw) > MAX_LOG_BYTES:
            retained = raw[-MAX_LOG_BYTES:]
            # Start at a complete UTF-8/log line so an offline viewer can read it.
            retained = retained.split(b'\n', 1)[-1]
            notes.append({'file': name, 'source_bytes': len(raw), 'source_sha256': digest(raw),
                          'retention': 'last complete lines up to 512KiB'})
            raw = retained
        selected[name] = raw
    core_names = tuple(json.loads((root / 'CORE_MANIFEST.json').read_text())['files'])
    for name in CORE + core_names + OPERATIONS:
        add(root / name, required=True)
    add(root / 'CORE_MANIFEST.json')
    for path in root.glob('*.sh'):
        add(path)
    for dirname in ('runs', 'preflight', 'operations', 'logs', 'data'):
        directory = root / dirname
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                permitted = TEXT_SUFFIXES | ({'.py', '.sh'} if dirname == 'operations' else set())
                if path.suffix in permitted and 'matplotlib-cache' not in path.parts and not any(part.startswith('env-') for part in path.parts):
                    add(path)
    for path in sorted(analysis.rglob('*')):
        if path.suffix in TEXT_SUFFIXES:
            add(path)
    if notes:
        selected['_compact_log_retention.json'] = json_bytes(notes)
    inventory = {'schema': 'burgers_ks_compact_inventory_v1',
        'root': str(root), 'analysis_dir': str(analysis), 'aggregate_report_sha256': report_sha,
        'scope': 'Code, configuration, hashes, compact curves/metrics, failures and bounded logs; checkpoint and raw prediction bytes excluded',
        'files': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in sorted(selected.items())}}
    inventory_raw = json_bytes(inventory)
    selected[INVENTORY] = inventory_raw
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name, data in sorted(selected.items()):
                entry = tarfile.TarInfo(name); entry.size = len(data); entry.mode = 0o644; entry.mtime = 0
                tar.addfile(entry, io.BytesIO(data))
    receipt = {'schema': 'burgers_ks_compact_pack_receipt_v1', 'archive': str(archive),
        'root': str(root), 'analysis_dir': str(analysis), 'aggregate_report_sha256': report_sha,
        'archive_bytes': archive.stat().st_size, 'archive_sha256': file_sha(archive),
        'inventory_sha256': digest(inventory_raw), 'member_count': len(selected),
        'checkpoint_and_prediction_bytes_included': False}
    verify_archive(archive, receipt)
    with receipt_path.open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': 'VERIFIED_COMPACT_ARCHIVE_CREATED', **receipt}, indent=2))


if __name__ == '__main__':
    main()
