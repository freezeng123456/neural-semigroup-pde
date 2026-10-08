"""Ordinary BSCC HTTP session using endpoints observed in official public code.

Credentials are loaded only by the authorized helper; cookies stay in memory.
This worker emits selected public metadata, not authentication responses.
"""
import importlib.util
import json
from pathlib import Path
import sys

helper_path = Path(__file__).resolve().parent.parent/'semigroup_midpoint_stage_b/transport-tools/official_http_recovery.py'
spec = importlib.util.spec_from_file_location('http_helper', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.headers['para_version'] = 'BSCCCloud_web_3.35.0 on MacIntel'

if not helper.login():
    raise SystemExit('AUTHENTICATION_FAILED')
print('{"status":"CONTROL_READY"}', flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command['action'] == 'quit':
        break
    if command['action'] == 'public_query':
        endpoint = command['endpoint']
        if endpoint not in ('app/user/app/list', 'app/list', 'scc/account/paths', 'scc/account/list'):
            raise ValueError('Endpoint has not been inspected in official source')
        url = helper.CLOUD+'/api/'+endpoint+'?'+helper.urllib.parse.urlencode(command.get('params', {}))
        parsed, info = helper.inspect_json(helper.request(url))
        info['endpoint'] = endpoint
        if info['http_status'] == 200 and command.get('output'):
            target = Path(command['output'])
            with target.open('x') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            info['saved'] = str(target)
        elif isinstance(parsed, dict):
            info['service_message'] = str(parsed.get('msg', ''))[:200]
        helper.emit(info)
    else:
        helper.run_command(command)
