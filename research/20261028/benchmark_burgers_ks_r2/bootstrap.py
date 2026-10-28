#!/usr/bin/env python3
"""Verify frozen source and isolated installed data dependencies; no model work."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'operations'))
from common import require_remote, verify_core, tasks, write, sha, now
require_remote(); core = verify_core(); rows = tasks()
runtime = json.loads((ROOT/'RUNTIME_CONFIG.json').read_text())
expected = {}
for line in (ROOT/'requirements-data.txt').read_text().splitlines():
    line = line.strip()
    if line and not line.startswith('#'):
        name, version = line.split('=='); expected[name] = version
program = 'import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in '+repr(list(expected))+'}))'
reply = subprocess.run([runtime['data_python'], '-c', program], text=True, capture_output=True, check=True)
installed = json.loads(reply.stdout)
assert installed == expected, (installed,expected)
source = ROOT.parent.parent/'20261028-burgers-ks-r2-source.tar.gz'
record = {'status':'PASS','verified_at_utc':now(),'source_archive_sha256':sha(source),
          'core_manifest_sha256':sha(ROOT/'CORE_MANIFEST.json'),'core_files':len(core['files']),
          'frozen_tasks':len(rows),'data_requirements':installed,'optimizer_updates':0,
          'training_python':runtime['training_python'],'data_python':runtime['data_python']}
write(ROOT/'operations/REMOTE_BOOTSTRAP_RECEIPT.json',record)
print(json.dumps(record,indent=2))
