#!/usr/bin/env python3
"""Restore or verify the recorded public DUE source/data; never launch training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / 'work/due_official_vorticity'
SOURCE_MANIFEST = OFFICIAL / 'operations/official_source_manifest.json'
DATA_MANIFEST = OFFICIAL / 'data/download_manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_source(root):
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    root = root.resolve(strict=True)
    for name, expected in manifest['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError(f'Source mismatch: {name}')
    print(f'Verified {len(manifest["files"])} source files at {root}')


def verify_data(root):
    manifest = json.loads(DATA_MANIFEST.read_text())
    root = root.resolve(strict=True)
    for item in manifest['files']:
        path = root / item['filename']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Data mismatch: {path.name}')
    print(f'Verified {len(manifest["files"])} public datasets at {root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='store_true', help='Fetch the pinned upstream commit into the default sibling path')
    parser.add_argument('--data', action='store_true', help='Download public author data and check recorded hashes')
    parser.add_argument('--verify-source', type=Path)
    parser.add_argument('--verify-data', type=Path)
    args = parser.parse_args()
    if not any((args.source, args.data, args.verify_source, args.verify_data)):
        parser.error('Choose a restore or verify operation')
    if args.source:
        manifest = json.loads(SOURCE_MANIFEST.read_text())
        target = HERE / 'work/literature_baselines/due'
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(target)], check=True)
        for command in [
            ['remote', 'add', 'origin', manifest['author_repository']],
            ['fetch', '--depth', '1', 'origin', manifest['commit']],
            ['checkout', '--detach', 'FETCH_HEAD'],
        ]:
            subprocess.run(['git', '-c', 'http.version=HTTP/1.1', '-C', str(target), *command], check=True, timeout=180)
        verify_source(target)
    if args.data:
        saved_manifest = DATA_MANIFEST.read_bytes()
        try:
            subprocess.run([sys.executable, str(OFFICIAL / 'download_public_data.py')], check=True)
            verify_data(OFFICIAL / 'data')
        finally:
            # The downloader writes a new receipt; retain the original integrity reference.
            DATA_MANIFEST.write_bytes(saved_manifest)
    if args.verify_source:
        verify_source(args.verify_source)
    if args.verify_data:
        verify_data(args.verify_data)


if __name__ == '__main__':
    main()
