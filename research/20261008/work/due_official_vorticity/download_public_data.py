#!/usr/bin/env python3
"""Download the two public author datasets, without credentials or source edits."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FILES = (
    ('VorticityOSG_train.mat', '1g6RU34K4JFm4e3OqJfKnVYcK75Wd_-Jn', 93123263),
    ('VorticityOSG_test.mat', '1bjCYz5jCAmx8t4xK6J_WYJvmyfqRf2Qq', 181969568),
)
FOLDER = 'https://drive.google.com/drive/folders/1efL-RR_H43Pe6P5BLtcEPFgz7ZmXnl5a?usp=sharing'

def download(item):
    name, identifier, expected_bytes = item
    target = DATA / name
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing {target}')
    partial = target.with_suffix('.mat.part')
    if partial.exists():
        raise FileExistsError(f'Refusing to replace existing incomplete download {partial}')
    public_url = 'https://drive.usercontent.google.com/download?' + urllib.parse.urlencode({'id': identifier, 'export': 'download', 'confirm': 't'})
    request = urllib.request.Request(public_url, headers={'User-Agent': 'Mozilla/5.0'})
    begin = time.perf_counter()
    digest = hashlib.sha256()
    received, last_progress = 0, 0
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get('Content-Type', '')
        content_length = response.headers.get('Content-Length')
        if response.status != 200 or 'text/html' in content_type:
            raise RuntimeError(f'{name}: expected binary HTTP 200, got {response.status}, {content_type}')
        if content_length is not None and int(content_length) != expected_bytes:
            raise RuntimeError(f'{name}: Content-Length differs from public folder listing')
        metadata = {'http_status': response.status, 'content_type': content_type, 'content_length': content_length,
                    'content_disposition': response.headers.get('Content-Disposition'),
                    'final_host': urllib.parse.urlsplit(response.url).netloc}
        first = response.read(1024 * 1024)
        if not first.startswith(b'MATLAB 5.0 MAT-file'):
            raise RuntimeError(f'{name}: response is not a MATLAB v5 binary')
        with partial.open('xb') as stream:
            stream.write(first)
            digest.update(first)
            received += len(first)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received - last_progress >= 32 * 1024 * 1024:
                    print(f'{name}: {received}/{expected_bytes} bytes', flush=True)
                    last_progress = received
    if received != expected_bytes:
        raise RuntimeError(f'{name}: incomplete download {received}/{expected_bytes}; kept {partial}')
    partial.rename(target)
    result = {'filename': name, 'path': str(target), 'google_drive_file_id': identifier,
              'author_folder_url': FOLDER, 'public_file_url': f'https://drive.google.com/file/d/{identifier}/view',
              'public_download_url': public_url, 'anonymous_download': True,
              'public_listing_expected_bytes': expected_bytes, 'bytes': received, 'sha256': digest.hexdigest(),
              'mat_header': first[:116].decode('ascii').strip(), 'http': metadata,
              'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - begin}
    print(f'{name}: complete, sha256={result["sha256"]}', flush=True)
    return result

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(download, FILES))
    manifest = {'schema_version': 'due-vorticity-official-public-download-v1',
                'author_repository': 'https://github.com/AI4Equations/due',
                'author_commit': '1e629104a12a9c3e3064996cbd4e301cec641ad8',
                'dataset_link_source': 'https://github.com/AI4Equations/due/blob/1e629104a12a9c3e3064996cbd4e301cec641ad8/README.md#dataset',
                'author_data_folder': FOLDER, 'files': records,
                'scope': 'Anonymous download of publicly shared author data; no training, no source modifications. The computed hashes are local integrity receipts; no author-published cryptographic checksum was available.'}
    (DATA / 'download_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('Wrote', DATA / 'download_manifest.json', flush=True)

if __name__ == '__main__':
    main()
