#!/usr/bin/env python3
"""Verify every archived file and produce a separate local engineering gate."""
import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
from launch_task import sha


def canonical(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and str(p) == name
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-manifest-sha', required=True)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text())
    assert a.archive.stat().st_size == receipt['bytes'] and sha(a.archive) == receipt['sha256']
    assert receipt['payload_manifest_sha256'] == a.expected_manifest_sha
    assert receipt['job'].isdigit()
    with tarfile.open(a.archive) as tar:
        members = tar.getmembers()
        assert len(members) == receipt['archive_members'] and len({m.name for m in members}) == len(members)
        for m in members:
            canonical(m.name); assert m.isfile()
        prefixes = {canonical(m.name).parts[0] for m in members}; assert len(prefixes) == 1
        prefix = prefixes.pop()
        inventory_path = prefix+'/'+receipt['inventory_relative_path']
        inventory = json.load(tar.extractfile(inventory_path))
        expected_names = {prefix+'/'+n for n in inventory['files']} | {inventory_path}
        assert expected_names == {m.name for m in members}
        for name, item in inventory['files'].items():
            m = tar.getmember(prefix+'/'+name)
            import hashlib
            h = hashlib.sha256()
            with tar.extractfile(m) as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    h.update(block)
            assert m.size == item['bytes'] and h.hexdigest() == item['sha256'], name
        a.output.mkdir(parents=True, exist_ok=False)
        tar.extractall(a.output)
    root = a.output/prefix
    report_path = root/receipt['run_relative_path']/'VALIDATION_REPORT.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'PASS' and report['device'] == 'cuda' and 'V100' in report['gpu_name'] and report['gpu_capability'] == [7, 0]
    assert report['payload_manifest_sha256'] == a.expected_manifest_sha and str(report['slurm_job_id']) == receipt['job']
    assert report['actual_optimizer_updates'] == 168 and len(report['checks']) == 29 and all(c['passed'] is True for c in report['checks'])
    assert sha(root/'PAYLOAD_MANIFEST.json') == a.expected_manifest_sha
    gate = {'schema': 'next_methods_local_recovery_gate_v1', 'status': 'PASS', 'preflight_job_id': receipt['job'],
            'payload_manifest_sha256': a.expected_manifest_sha, 'validation_report_sha256': sha(report_path),
            'checks': [{'name': 'remote_archive_sha256', 'passed': True}, {'name': 'exact_internal_inventory', 'passed': True},
                       {'name': 'all_file_sha256_and_bytes', 'passed': True}, {'name': 'v100_optimizer_and_resume_gates', 'passed': True}],
            'recovered_root': str(root.resolve())}
    with (a.output/'LOCAL_PREFLIGHT_GATE.json').open('x') as f:
        json.dump(gate, f, indent=2)
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
