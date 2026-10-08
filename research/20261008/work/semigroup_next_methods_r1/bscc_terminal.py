"""Authorized Web SSH via the official application's Socket.IO polling transport.

The public app/ssh bundle specifies /pterm/socket.io, cluster_code,
cluster_login_name, and the input/output/resize events. Its delivered client
is Socket.IO 2.2.0 (Engine.IO 3). Cookies and transport identifiers stay in RAM.
No SSH host, credential or browser profile is guessed or extracted.
"""
import base64
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import threading
import time
import traceback
import urllib.request
import urllib.parse
import uuid

helper_path = Path(__file__).resolve().parent.parent/'semigroup_midpoint_stage_b/transport-tools/official_http_recovery.py'
spec = importlib.util.spec_from_file_location('http_helper', helper_path)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
helper.headers['para_version'] = 'BSCCCloud_web_3.35.0 on MacIntel'
if not helper.login():
    raise SystemExit('AUTHENTICATION_FAILED')

base = 'https://ssh.blsc.cn/pterm/socket.io/'
assigned, info = helper.inspect_json(helper.request(helper.CLOUD+'/api/scc/account/list'))
assert info['http_status'] == 200 and isinstance(assigned, list)
matches = [r for r in assigned if r.get('cluster_code') == 'BSCC-N26' and r.get('cluster_login_name') == 'scv7tsq']
assert len(matches) == 1
# The public launcher replaces account_id with the selected cluster/name before
# opening its iframe. Match that actual UI route after checking the assignment.
params = {'EIO': '3', 'transport': 'polling', 'cluster_code': 'BSCC-N26',
          'cluster_login_name': 'scv7tsq'}
sid = None
connected = threading.Event()
terminal_created = threading.Event()
shell_verified = threading.Event()
stopped = threading.Event()
post_lock = threading.Lock()
active = None
active_lock = threading.Lock()
text_buffer = ''
startup_id = None


def url():
    return base+'?'+urllib.parse.urlencode({**params, **({'sid': sid} if sid else {}), 't': str(int(time.time()*1000))})


def packets(raw):
    text = raw.decode('utf-8')
    pos = 0
    while pos < len(text):
        sep = text.find(':', pos)
        if sep < 0 or not text[pos:sep].isdigit():
            raise ValueError('Unexpected Engine.IO 3 polling framing')
        size = int(text[pos:sep]); start = sep+1
        yield text[start:start+size]
        pos = start+size


def post(packet):
    data = (str(len(packet))+':'+packet).encode()
    with post_lock:
        req = urllib.request.Request(url(), data=data, headers={**helper.headers, 'Content-Type': 'text/plain;charset=UTF-8'})
        with helper.opener.open(req, timeout=30) as r:
            if r.status != 200 or r.read() != b'ok':
                raise RuntimeError('Engine.IO input was not acknowledged')


def output(data):
    global text_buffer, active
    with active_lock:
        if active is None:
            # Terminal banners and prompts are summarized without exposing any
            # platform session strings or unrelated previous terminal contents.
            if 'password' in data.lower() or 'permission denied' in data.lower():
                helper.emit({'status': 'TERMINAL_AUTHENTICATION_ATTENTION'})
            return
        text_buffer += data
        clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text_buffer).replace('\r', '')
        begin = '__TASK_BEGIN_'+active+'__'
        end = '__TASK_END_'+active+'__'
        if begin+'\n' in clean and end in clean:
            body = clean.split(begin+'\n', 1)[1].split(end, 1)[0]
            if active == startup_id:
                expected = 'BSCC_SHELL_ID=scv7tsq'
                if expected in body.splitlines() and any(line.startswith('BSCC_SHELL_HOST=') for line in body.splitlines()):
                    helper.emit({'status': 'SHELL_IDENTITY_VERIFIED', 'output': body})
                    shell_verified.set()
                else:
                    helper.emit({'status': 'SHELL_IDENTITY_NOT_VERIFIED'})
                    stopped.set()
            else:
                helper.emit({'status': 'REMOTE_COMMAND_FINISHED', 'command_id': active, 'output': body})
            active = None; text_buffer = ''


def consume(packet):
    helper.emit({'status': 'TRANSPORT_PACKET', 'packet_type': packet[:2], 'packet_characters': len(packet)})
    if packet == '40':
        connected.set()
    elif packet.startswith('42'):
        event = json.loads(packet[2:])
        if event[0] == 'output':
            output(event[1])
        elif event[0] in ('exit', 'error'):
            helper.emit({'status': 'TERMINAL_EVENT', 'event': event[0]}); stopped.set()
        elif event[0] == 'termCreated':
            terminal_created.set()
            helper.emit({'status': 'TERMINAL_CREATED'})
    elif packet == '1' or packet == '41':
        stopped.set(); helper.emit({'status': 'TERMINAL_DISCONNECTED'})
    elif packet == '2':
        post('3')
    elif packet.startswith('44'):
        detail = json.loads(packet[2:])
        message = str(detail)
        code_match = re.search(r'ERROR\s*(\d{3,6})', message, re.I)
        helper.emit({'status': 'SOCKET_NAMESPACE_ERROR',
                     'service_code': int(code_match.group(1)) if code_match else None,
                     'authentication_error': 'authentication error' in message.lower()})
        stopped.set()


with helper.request(url()) as response:
    first = list(packets(response.read()))
if not first or not first[0].startswith('0'):
    raise SystemExit('NO_ENGINE_IO_HANDSHAKE')
handshake = json.loads(first[0][1:]); sid = handshake['sid']
for packet in first[1:]:
    consume(packet)
helper.emit({'status': 'POLLING_TRANSPORT_CONNECTED', 'cluster': 'BSCC-N26', 'engine_io': 3})
# Socket.IO 2.2.0 does not send a CONNECT packet for the default namespace.
# Its server sends the connection packet; wait for polling to deliver it.


def poll():
    try:
        while not stopped.is_set():
            with helper.request(url()) as response:
                if response.status != 200:
                    raise RuntimeError('Polling HTTP status '+str(response.status))
                for packet in packets(response.read()):
                    consume(packet)
    except BaseException as e:
        helper.emit({'status': 'TERMINAL_POLL_FAILED', 'error_type': type(e).__name__}); stopped.set()


def ping():
    try:
        while not stopped.wait(handshake.get('pingInterval', 25000)/1000*.7):
            post('2')
    except BaseException as e:
        helper.emit({'status': 'TERMINAL_PING_FAILED', 'error_type': type(e).__name__}); stopped.set()


threading.Thread(target=poll, daemon=True).start()
threading.Thread(target=ping, daemon=True).start()
for _ in range(18):
    if stopped.is_set() or connected.wait(1):
        break
if not connected.is_set() or stopped.is_set():
    try:
        post('1')
    except Exception:
        pass
    raise SystemExit('SOCKET_NAMESPACE_NOT_CONNECTED')
for _ in range(18):
    if stopped.is_set() or terminal_created.wait(1):
        break
if not terminal_created.is_set() or stopped.is_set():
    stopped.set()
    raise SystemExit('TERMINAL_NOT_CREATED')
post('42'+json.dumps(['resize', {'col': 160, 'row': 50}], separators=(',', ':')))
# Verify a real shell command before accepting any project command. This probe
# changes no remote files and cannot submit a job.
startup_id = uuid.uuid4().hex
with active_lock:
    active = startup_id
    probe = "import getpass,socket;print('__TASK_BEGIN_%s__',flush=True);print('BSCC_SHELL_ID='+getpass.getuser(),flush=True);print('BSCC_SHELL_HOST='+socket.gethostname(),flush=True);print('__TASK_END_%s__',flush=True)" % (startup_id, startup_id)
    probe_shell = 'python3 -c '+shlex.quote(probe)+'\n'
post('42'+json.dumps(['input', probe_shell], separators=(',', ':')))
for _ in range(18):
    if stopped.is_set() or shell_verified.wait(1):
        break
if not shell_verified.is_set() or stopped.is_set():
    stopped.set()
    raise SystemExit('SHELL_COMMAND_NOT_VERIFIED')
helper.emit({'status': 'TERMINAL_READY', 'cluster': 'BSCC-N26'})
for line in sys.stdin:
    command = json.loads(line)
    if command['action'] == 'quit':
        stopped.set(); break
    if command['action'] != 'exec' or stopped.is_set():
        raise ValueError('Terminal is unavailable or unsupported action')
    with active_lock:
        if active is not None:
            helper.emit({'status': 'COMMAND_STILL_ACTIVE'}); continue
        active = uuid.uuid4().hex
        code = command['code'] if 'code' in command else Path(command['code_file']).read_text()
        encoded = base64.b64encode(code.encode()).decode()
        wrapper = "import base64,traceback;print('__TASK_BEGIN_%s__',flush=True)\ntry:\n exec(compile(base64.b64decode('%s'),'<authorized-project-command>','exec'))\nexcept BaseException:\n traceback.print_exc()\n print('REMOTE_COMMAND_ERROR',flush=True)\nfinally:\n print('__TASK_END_%s__',flush=True)" % (active, encoded, active)
        shell = 'python3 -c '+shlex.quote(wrapper)+'\n'
        text_buffer = ''
    post('42'+json.dumps(['input', shell], separators=(',', ':')))
    helper.emit({'status': 'REMOTE_COMMAND_DISPATCHED', 'command_id': active})
