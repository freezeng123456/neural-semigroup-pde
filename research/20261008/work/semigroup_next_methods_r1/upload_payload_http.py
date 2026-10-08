#!/usr/bin/env python3
"""Upload via the official web application's observed file-transfer protocol.

One-MiB POST slices, closed Content-Range, and per-slice server MD5 are copied
from upload_web.js. Login cookies remain in RAM. Existing files are protected.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    helper_path = Path(__file__).resolve().parent.parent / 'semigroup_midpoint_stage_b/transport-tools/official_http_recovery.py'
    spec = importlib.util.spec_from_file_location('bscc_http_helper', helper_path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.event_log = a.output / 'TRANSFER_EVENTS.jsonl'
    h.headers['para_version'] = 'BSCCCloud_web_3.35.0 on MacIntel'
    recorded = json.loads(a.archive.with_suffix(a.archive.suffix+'.json').read_text())
    size = a.archive.stat().st_size
    expected = hashlib.sha256(a.archive.read_bytes()).hexdigest()
    assert size == recorded['bytes'] and expected == recorded['sha256']
    assert a.archive.name == '20261005-next-methods-r1.tar.gz'
    if not h.login():
        raise SystemExit('ORDINARY_LOGIN_REJECTED')
    paths, info = h.inspect_json(h.request(h.CLOUD+'/api/scc/account/paths'))
    assert info['http_status'] == 200 and isinstance(paths, list)
    assigned = [r for r in paths if r.get('zone') == 'BSCC-N26' and r.get('clusterUser') == 'scv7tsq']
    assert len(assigned) == 1
    assigned = assigned[0]
    base = assigned['endpoint']['ipv4'][0].rstrip('/')+'/api'
    directory = assigned['path']+'/data02/home/scv7tsq/run/'
    target = directory+a.archive.name
    rows, info = h.inspect_json(h.request(base+'/file/manager/ls?'+urllib.parse.urlencode({'path': directory})))
    assert info['http_status'] == 200
    if isinstance(rows, dict):
        candidates = [v for k, v in rows.items() if k in ('data', 'files', 'list', 'items', 'result') and isinstance(v, list)]
        assert len(candidates) == 1
        rows = candidates[0]
    assert isinstance(rows, list)
    assert not any(r.get('name', r.get('fileName')) == a.archive.name for r in rows), 'Existing remote archive; do not overwrite'
    result, info = h.inspect_json(h.request(base+'/file/manager/disk/check',
                                           {'path': directory, 'size': math.ceil(size/1024)}))
    assert info['http_status'] == 200 and result.get('result') in (True, 'true'), 'Disk check did not authorize upload'
    intent = {'schema': 'bscc_payload_upload_v1', 'created_unix_seconds': time.time(),
              'target': target, 'assigned_api': base, 'bytes': size, 'sha256': expected,
              'status': 'UPLOAD_INTENT', 'training_submitted': False}
    with (a.output / 'UPLOAD_INTENT.json').open('x') as f:
        json.dump(intent, f, indent=2)
    h.emit({'status': 'UPLOAD_READY', 'file': a.archive.name, 'bytes': size, 'assigned_api': base})
    upload_url = base+'/file/stream/upload?'+urllib.parse.urlencode({'path': target})
    completed = 0
    began = time.monotonic()
    with a.archive.open('rb') as source:
        while completed < size:
            payload = source.read(1024*1024)
            last = completed+len(payload)-1
            request = urllib.request.Request(upload_url, data=payload, method='POST', headers={
                **h.headers, 'Content-Type': 'application/octet-stream',
                'Content-Range': f'bytes {completed}-{last}/{size}'})
            # On an unknown outcome, preserve the remote partial and do not
            # repeat a write automatically. The receipt records acknowledged slices.
            with h.opener.open(request, timeout=35) as response:
                result, info = h.inspect_json(response)
            digest = hashlib.md5(payload).hexdigest()
            assert info['http_status'] == 200 and isinstance(result, dict) and result.get('md5') == digest, info
            with (a.output / 'ACKNOWLEDGED_SLICES.jsonl').open('a') as f:
                f.write(json.dumps({'first': completed, 'last': last, 'bytes': len(payload), 'server_md5': digest})+'\n')
            completed = last+1
            if completed % (10*1024*1024) == 0 or completed == size:
                h.emit({'status': 'UPLOAD_PROGRESS', 'acknowledged_bytes': completed, 'total_bytes': size,
                        'elapsed_seconds': time.monotonic()-began})
    roundtrip = a.output / a.archive.name
    h.run_command({'action': 'download', 'base': base, 'path': target, 'output': str(roundtrip),
                   'bytes': size, 'sha256': expected, 'resume': True})
    assert roundtrip.is_file() and roundtrip.stat().st_size == size
    actual = hashlib.sha256(roundtrip.read_bytes()).hexdigest()
    assert actual == expected
    receipt = {**intent, 'status': 'REMOTE_ARCHIVE_ROUNDTRIP_VERIFIED',
               'acknowledged_bytes': completed, 'roundtrip_sha256': actual,
               'duration_seconds': time.monotonic()-began,
               'remote_shell_execution_verified': False, 'training_submitted': False}
    with (a.output / 'UPLOAD_RECEIPT.json').open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    h.emit({'status': receipt['status'], 'bytes': size, 'sha256': actual, 'training_submitted': False})


if __name__ == '__main__':
    main()
