"""Read-only Slurm resource audit adapted from the completed Stage B protocol."""
import hashlib,json,re,subprocess
from pathlib import Path
USER=ACCOUNT="scv7tsq"
CLUSTER="bscc26a"
CPUS,GPUS,WALL_SECONDS=10,1,21600

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def positive_limit(value):
    value = value.strip()
    if value.upper() in ('', 'UNLIMITED', 'NONE', 'N/A', '-1'):
        return None
    require(value.isdigit(), 'Unrecognized Slurm limit: ' + value)
    return int(value)


def time_seconds(value):
    if value.upper() in ('', 'UNLIMITED', 'NONE', 'N/A', '-1'):
        return None
    days = 0
    if '-' in value:
        day_text, value = value.split('-', 1); days = int(day_text)
    parts = [int(x) for x in value.split(':')]
    if len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    else:
        seconds = parts[0] * 60
    return days * 86400 + seconds


def tres(value):
    result = {}
    for item in value.split(','):
        if '=' not in item:
            continue
        name, count = item.strip().split('=', 1)
        if name == 'cpu' or name == 'gres/gpu' or name.startswith('gres/gpu:'):
            require(count.isdigit(), 'Unrecognized resource limit: ' + item)
            result[name] = int(count)
    return result


def gpu_count(value):
    if value in ('', 'N/A', '(null)', 'None'):
        return None
    # squeue %b: gres/gpu:1 or gres/gpu:tesla_v100:1, optionally other GRES.
    total = 0
    for entry in value.split(','):
        if entry.startswith(('gres/gpu:', 'gres:gpu:', 'gpu:')):
            tail = entry.rsplit(':', 1)[1]
            require(tail.isdigit(), 'Unrecognized requested GRES: ' + value)
            total += int(tail)
    require(total > 0 or 'gpu' not in value, 'Unknown GPU resource spelling')
    return total


def job_gpu_count(job_id, evidence):
    """Resolve missing squeue GRES from explicit live Slurm resource fields.

    This cluster's --gpus=1 jobs expose %b=N/A but scontrol TRES=...gres/gpu=1.
    Keep resource metadata only; broad job output also contains command/paths.
    """
    command = ['scontrol', 'show', 'job', job_id, '-o']
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    fields = dict(re.findall(r'(\w+)=([^\s]+)', reply.stdout))
    allowed = ('JobId', 'ArrayJobId', 'ArrayTaskId', 'JobState', 'NumNodes',
               'TRES', 'ReqTRES', 'AllocTRES', 'TresPerJob', 'TresPerNode')
    kept = {key: fields[key] for key in allowed if key in fields}
    evidence.append({'command': command, 'returncode': reply.returncode,
                     'persisted_fields': kept,
                     'stderr': reply.stderr if reply.returncode else ''})
    require(reply.returncode == 0, 'Could not resolve GPU count for job ' + job_id)
    if '_' in job_id:
        array_id, task_id = job_id.split('_', 1)
        require(fields.get('ArrayJobId') == array_id and fields.get('ArrayTaskId') == task_id,
                'Slurm GPU metadata returned a different array task: ' + job_id)
    else:
        require(fields.get('JobId') == job_id, 'Slurm GPU metadata returned a different job: ' + job_id)
    counts = []
    explicit_tres_seen = False
    for key in ('TRES', 'ReqTRES', 'AllocTRES'):
        raw = fields.get(key, '')
        if raw in ('', 'N/A', '(null)', 'None'):
            continue
        explicit_tres_seen = True
        values = tres(raw)
        if 'gres/gpu' in values:
            counts.append(values['gres/gpu'])
        else:
            typed = [value for name, value in values.items() if name.startswith('gres/gpu:')]
            if typed:
                counts.append(sum(typed))
    for key in ('TresPerJob', 'TresPerNode'):
        raw = fields.get(key, '')
        if 'gpu' not in raw:
            continue
        count = gpu_count(raw)
        require(count is not None and count > 0, 'Unresolved ' + key + ' for job ' + job_id)
        if key == 'TresPerNode':
            require(fields.get('NumNodes', '').isdigit() and int(fields['NumNodes']) > 0,
                    'Cannot scale per-node GPU resources for job ' + job_id)
            count *= int(fields['NumNodes'])
        counts.append(count)
    if not counts:
        # A nonempty explicit total resource list without any GPU request is a
        # CPU job. N/A alone never establishes a zero-GPU allocation.
        require(explicit_tres_seen, 'GPU count remains unknown for job ' + job_id)
        return 0
    require(len(set(counts)) == 1, 'Conflicting explicit GPU counts for job ' + job_id)
    return counts[0]


def resolve_queue_gpus(rows, evidence):
    for row in rows:
        if row['gpus'] is None:
            row['gpus'] = job_gpu_count(row['id'], evidence)
    return rows


def parse_pipe(text, fields):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.strip().split('|')
        if len(values) == len(fields) + 1 and values[-1] == '':
            values.pop()
        require(len(values) == len(fields), 'Unexpected Slurm columns: ' + line)
        rows.append(dict(zip(fields, (v.strip() for v in values))))
    return rows


def parse_queue(text):
    rows = parse_pipe(text, ('id', 'state', 'account', 'partition', 'cpus', 'gres', 'qos'))
    for row in rows:
        require(re.fullmatch(r'\d+(?:_\d+)?', row['id']) is not None,
                'squeue must expand every array task; compressed/unknown ID: ' + row['id'])
        require(row['cpus'].isdigit(), 'Invalid requested CPU count')
        row['cpus'] = int(row['cpus']); row['gpus'] = gpu_count(row['gres'])
    require(len({row['id'] for row in rows}) == len(rows), 'Duplicate squeue task rows')
    return rows


def all_pass(report):
    checks = report.get('checks')
    return (report.get('status') == 'PASS' and isinstance(checks, list) and bool(checks)
            and all(isinstance(c, dict) and c.get('name') and c.get('passed') is True for c in checks)
            and len({c['name'] for c in checks}) == len(checks))


def query(command, evidence):
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    evidence.append({'command': command, 'returncode': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr})
    require(reply.returncode == 0, 'Read-only Slurm query failed: ' + ' '.join(command))
    return reply.stdout


def resources(mode, evidence):
    # Persist only ClusterName from this broad configuration read; do not copy
    # authentication/configuration fields into the public evidence receipt.
    cluster_reply = subprocess.run(['scontrol', 'show', 'config'], capture_output=True, text=True, timeout=60)
    cluster_match = re.search(r'^\s*ClusterName\s*=\s*(\S+)\s*$', cluster_reply.stdout, re.MULTILINE)
    current_cluster = cluster_match.group(1) if cluster_match else None
    evidence.append({'command': ['scontrol', 'show', 'config'], 'returncode': cluster_reply.returncode,
                     'persisted_fields': {'ClusterName': current_cluster},
                     'stderr': cluster_reply.stderr})
    require(cluster_reply.returncode == 0 and current_cluster == CLUSTER, 'Current Slurm controller is not the authorized Beijing cluster')
    fields = ('Cluster', 'Account', 'User', 'Partition', 'QOS', 'DefaultQOS', 'MaxJobs', 'MaxSubmit', 'GrpTRES', 'MaxTRES', 'MaxWall')
    raw = query(['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=' + USER, 'account=' + ACCOUNT, 'format=' + ','.join(fields)], evidence)
    assocs = [r for r in parse_pipe(raw, fields) if r['Cluster'] == CLUSTER and r['Account'] == ACCOUNT and r['User'] == USER and r['Partition'] in ('', 'gpu')]
    require(bool(assocs), 'No authorized Beijing GPU association')
    partition_raw = query(['scontrol', 'show', 'partition', 'gpu', '-o'], evidence)
    parts = dict(re.findall(r'(\w+)=([^\s]+)', partition_raw))
    require(parts.get('PartitionName') == 'gpu' and parts.get('State') == 'UP', 'GPU partition is not UP')
    max_time = time_seconds(parts.get('MaxTime', ''))
    if mode == 'full':
        require(max_time is None and parts.get('MaxTime', '').upper() == 'UNLIMITED',
                'Full submission requests unlimited time, but partition MaxTime is not explicitly UNLIMITED')
    else:
        require(max_time is None or max_time >= WALL_SECONDS, 'Partition wall limit below six hours')
    qos_names = {parts.get('QoS', parts.get('QOS', ''))}
    for row in assocs:
        qos_names.update(row['QOS'].split(',')); qos_names.add(row['DefaultQOS'])
    qos_names -= {'', 'N/A', '(null)'}
    require(qos_names and all(re.fullmatch(r'[A-Za-z0-9_-]+', q) for q in qos_names), 'Unresolved QoS names')
    qfields = ('Name', 'MaxWall', 'MaxJobsPU', 'MaxSubmitPU', 'MaxTRESPU', 'GrpTRES')
    qos_raw = query(['sacctmgr', '-nP', 'show', 'qos', 'where', 'name=' + ','.join(sorted(qos_names)), 'format=' + ','.join(qfields)], evidence)
    qos_rows = parse_pipe(qos_raw, qfields)
    require({row['Name'] for row in qos_rows} == qos_names, 'Incomplete QoS limits')
    queue_raw = query(['squeue', '--array', '--noheader', '--user=' + USER, '--format=%i|%T|%a|%P|%C|%b|%q'], evidence)
    jobs = resolve_queue_gpus(parse_queue(queue_raw), evidence)
    # --array gives every potential task, so six pending elements consume six
    # MaxSubmit slots even though this matrix can run at most two together.
    active = [row for row in jobs if row['state'] != 'PENDING']
    potential, concurrent = (6, 2) if mode == 'full' else (1, 1)
    maxjobs = [positive_limit(r['MaxJobs']) for r in assocs]
    maxsubmit = [positive_limit(r['MaxSubmit']) for r in assocs]
    cpulimits, gpulimits = [], []
    for row in assocs:
        group = tres(row['GrpTRES'])
        if 'cpu' in group: cpulimits.append(group['cpu'])
        if 'gres/gpu' in group: gpulimits.append(group['gres/gpu'])
        individual = tres(row['MaxTRES'])
        require(individual.get('cpu', CPUS) >= CPUS and individual.get('gres/gpu', GPUS) >= GPUS, 'Per-job association resource limit too small')
        limit = time_seconds(row['MaxWall'])
        if mode == 'full':
            require(limit is None, 'Full submission requests unlimited time, but association MaxWall is finite')
        else:
            require(limit is None or limit >= WALL_SECONDS, 'Association wall limit below six hours')
    for row in qos_rows:
        maxjobs.append(positive_limit(row['MaxJobsPU'])); maxsubmit.append(positive_limit(row['MaxSubmitPU']))
        limits = tres(row['MaxTRESPU'])
        if 'cpu' in limits: cpulimits.append(limits['cpu'])
        if 'gres/gpu' in limits: gpulimits.append(limits['gres/gpu'])
        limit = time_seconds(row['MaxWall'])
        if mode == 'full':
            require(limit is None, 'Full submission requests unlimited time, but QoS MaxWall is finite')
        else:
            require(limit is None or limit >= WALL_SECONDS, 'QoS wall limit below six hours')
        group = tres(row['GrpTRES'])
        if group:
            # A finite QoS group limit covers other users as well. Inspect only
            # scheduler resource metadata, never names, paths, or commands.
            scope_flag = '--partition=gpu' if row['Name'] == parts.get('QoS', parts.get('QOS')) else '--qos=' + row['Name']
            global_jobs = resolve_queue_gpus(parse_queue(query(['squeue', '--array', '--noheader', scope_flag, '--format=%i|%T|%a|%P|%C|%b|%q'], evidence)), evidence)
            group_active = [r for r in global_jobs if r['state'] != 'PENDING']
            require(sum(r['cpus'] for r in group_active) + concurrent * CPUS <= group.get('cpu', 10**12), 'QoS group CPU headroom insufficient')
            require(sum(r['gpus'] for r in group_active) + concurrent * GPUS <= group.get('gres/gpu', 10**12), 'QoS group GPU headroom insufficient')
    maxjobs = [x for x in maxjobs if x is not None]; maxsubmit = [x for x in maxsubmit if x is not None]
    require(maxjobs and maxsubmit and cpulimits and gpulimits, 'Finite account limits must be visible; inherited/unknown limits need inspection')
    limits = {'max_jobs': min(maxjobs), 'max_submit': min(maxsubmit), 'cpus': min(cpulimits), 'gpus': min(gpulimits)}
    usage = {'submitted_tasks': len(jobs), 'active_tasks': len(active),
             'active_cpus': sum(r['cpus'] for r in active), 'active_gpus': sum(r['gpus'] for r in active)}
    require(usage['submitted_tasks'] + potential <= limits['max_submit'], 'MaxSubmit headroom insufficient for all potential array tasks')
    require(usage['active_tasks'] + concurrent <= limits['max_jobs'], 'MaxJobs headroom insufficient')
    require(usage['active_cpus'] + concurrent * CPUS <= limits['cpus'], 'Account CPU headroom insufficient')
    require(usage['active_gpus'] + concurrent * GPUS <= limits['gpus'], 'Account GPU headroom insufficient')
    return {'cluster': CLUSTER, 'account': ACCOUNT, 'partition': 'gpu',
            'limits': limits, 'current_usage': usage, 'new_potential_tasks': potential,
            'new_max_concurrent_tasks': concurrent, 'qos_checked': sorted(qos_names),
            'queue_snapshot': jobs,
            'limitation': 'This is a live pre-submit snapshot, not a reservation; Slurm enforces subsequent changes and physical availability.'}
