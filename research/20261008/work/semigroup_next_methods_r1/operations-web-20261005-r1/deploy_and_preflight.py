"""Deploy the verified freeze and submit exactly one authorized preflight.

Extra live TRES inspection handles generic --gpus jobs whose squeue %b is N/A.
The frozen submission script retains its own durable intent and other guards.
"""
import datetime,getpass,hashlib,json,os,re,subprocess,tarfile
from pathlib import Path
BASE=Path('/data02/run01/scv7tsq')
ROOT=BASE/'neural-semigroup/20261005-next-methods-r1'
ARCHIVE=BASE/'20261005-next-methods-r1.tar.gz'
PYTHON=BASE/'apolarity/20261002-v100-paper-r1/env/runtime/bin/python'
DEPS=BASE/'neural-semigroup/20261004-due-vorticity-1e62910-r1/deps'
assert getpass.getuser()=='scv7tsq' and PYTHON.is_file() and DEPS.is_dir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
assert ARCHIVE.stat().st_size==200326164 and sha(ARCHIVE)=='1e3e22752148d6f3d233fa0aa99d0dfabf32cd7ade0a6a5163a4cdd1b9673b9c'
with tarfile.open(ARCHIVE,'r:gz') as tar:
    manifest_data=tar.extractfile(ROOT.name+'/PAYLOAD_MANIFEST.json').read()
    assert hashlib.sha256(manifest_data).hexdigest()=='ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'
    manifest=json.loads(manifest_data)
    names={ROOT.name+'/'+n for n in manifest['files']}|{ROOT.name+'/PAYLOAD_MANIFEST.json'}
    members=tar.getmembers()
    assert len(members)==100 and {m.name for m in members}==names and all(m.isfile() for m in members)
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    if not ROOT.exists():
        for n,item in manifest['files'].items():
            b=tar.extractfile(ROOT.name+'/'+n).read()
            assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],n
        tar.extractall(ROOT.parent,members=members)
for n,item in manifest['files'].items():
    p=ROOT/n
    assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],n
assert (ROOT/'PAYLOAD_MANIFEST.json').read_bytes()==manifest_data
ops=ROOT/'operations';ops.mkdir(exist_ok=True)
assert not (ops/'validate_submission_intent.json').exists(),'Inspect existing submission intent; never resubmit automatically'

def query(command):
    r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=50)
    assert r.returncode==0,(command,r.stderr)
    return r.stdout
assoc=query(['sacctmgr','-nP','show','assoc','where','user=scv7tsq','account=scv7tsq','format=Cluster,Account,User,MaxJobs,MaxSubmit,GrpTRES'])
rows=[r.split('|') for r in assoc.splitlines() if r.startswith('bscc26a|scv7tsq|scv7tsq|')]
assert rows
limits=[]
for row in rows:
    t=dict(p.split('=',1) for p in row[5].split(',') if '=' in p)
    assert row[3].isdigit() and row[4].isdigit() and t.get('cpu','').isdigit() and t.get('gres/gpu','').isdigit()
    limits.append({'jobs':int(row[3]),'submitted':int(row[4]),'cpu':int(t['cpu']),'gpu':int(t['gres/gpu'])})
limit={k:min(r[k] for r in limits) for k in limits[0]}
queue=query(['squeue','--array','--noheader','--user=scv7tsq','--format=%i|%T|%C'])
jobs=[]
for line in queue.splitlines():
    jid,state,cpus=line.strip().split('|')
    assert re.fullmatch(r'\d+(?:_\d+)?',jid) and cpus.isdigit()
    details=query(['scontrol','show','job',jid,'-o'])
    fields=dict(re.findall(r'(\w+)=([^\s]+)',details))
    resource=fields.get('ReqTRES','') if state=='PENDING' else fields.get('AllocTRES',fields.get('ReqTRES',''))
    t=dict(p.split('=',1) for p in resource.split(',') if '=' in p)
    gpu=int(t['gres/gpu']) if 'gres/gpu' in t else sum(int(v) for k,v in t.items() if k.startswith('gres/gpu:'))
    assert resource and all(v.isdigit() for k,v in t.items() if k.startswith('gres/gpu'))
    jobs.append({'job_id':jid,'state':state,'cpus':int(cpus),'gpus':gpu,'resource_tres':resource})
active=[j for j in jobs if j['state']!='PENDING']
assert len(jobs)+1<=limit['submitted'] and len(active)+1<=limit['jobs']
assert sum(j['cpus'] for j in active)+10<=limit['cpu']
assert sum(j['gpus'] for j in active)+1<=limit['gpu']
gate={'schema':'next_methods_web_deployment_v1','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'user':getpass.getuser(),'root':str(ROOT),'payload_manifest_sha256':hashlib.sha256(manifest_data).hexdigest(),
      'archive_sha256':sha(ARCHIVE),'verified_files':99,'live_limits':limit,'live_jobs_with_tres':jobs,
      'generic_gpus_note':'squeue %b returned N/A for existing --gpus jobs; independently inspect ReqTRES/AllocTRES before submitting.',
      'status':'DEPLOYED_RESOURCE_GATE_PASS'}
with (ops/'browser_deployment_r1.json').open('x') as f:json.dump(gate,f,indent=2)
env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(DEPS)+':'+str(ROOT/'due'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
result=subprocess.run([str(PYTHON),'-B','-u',str(ROOT/'submit_experiments.py'),'--root',str(ROOT),'--mode','validate'],
                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,timeout=240)
receipt=ops/'validate_submission_receipt.json'
summary={'returncode':result.returncode,'receipt_exists':receipt.is_file(),'stderr':result.stderr[-4000:]}
if receipt.is_file():
    r=json.loads(receipt.read_text());summary.update(job_id=r.get('job_id'),status=r.get('status'))
    d=sha(receipt)
    print('SUBMISSION_RECEIPT_BYTES='+str(receipt.stat().st_size),flush=True)
    print('SUBMISSION_RECEIPT_SHA_A='+d[:32],flush=True)
    print('SUBMISSION_RECEIPT_SHA_B='+d[32:],flush=True)
print(json.dumps(summary,ensure_ascii=False),flush=True)
if result.returncode!=0 and not receipt.is_file():print(result.stdout[-4000:],flush=True)
