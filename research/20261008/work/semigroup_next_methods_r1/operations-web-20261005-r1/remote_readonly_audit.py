import datetime, getpass, hashlib, json, os, re, socket, subprocess
from pathlib import Path

ACCOUNT = 'scv7tsq'
BASE = Path('/data02/run01/scv7tsq')
ARCHIVE = BASE/'20261005-next-methods-r1.tar.gz'
ROOT = BASE/'neural-semigroup/20261005-next-methods-r1'
RUNTIME = BASE/'apolarity/20261002-v100-paper-r1/env/runtime/bin/python'
DEPS = BASE/'neural-semigroup/20261004-due-vorticity-1e62910-r1/deps'
assert getpass.getuser() == ACCOUNT
report = {'schema':'next_methods_web_shell_readonly_v1',
          'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'user':getpass.getuser(),'hostname':socket.gethostname(),
          'archive_exists':ARCHIVE.is_file(),'root_exists':ROOT.exists(),
          'runtime_exists':RUNTIME.is_file(),'dependency_root_exists':DEPS.is_dir(),
          'scheduler_queries':[]}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
if ARCHIVE.is_file():
    report['archive_bytes']=ARCHIVE.stat().st_size
    report['archive_sha256']=sha(ARCHIVE)
    report['archive_verified']=report['archive_bytes']==200326164 and report['archive_sha256']=='1e3e22752148d6f3d233fa0aa99d0dfabf32cd7ade0a6a5163a4cdd1b9673b9c'
commands=[['scontrol','show','config'],
 ['squeue','--array','--noheader','--user='+ACCOUNT,'--format=%i|%T|%a|%P|%C|%b|%q'],
 ['scontrol','show','partition','gpu','-o'],
 ['sacctmgr','-nP','show','assoc','where','user='+ACCOUNT,'account='+ACCOUNT,'format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmit,GrpTRES,MaxTRES,MaxWall'],
 ['sacctmgr','-nP','show','qos','format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES']]
for command in commands:
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=50)
        row={'command':command,'returncode':r.returncode,'stderr':r.stderr}
        if command==commands[0]:
            m=re.search(r'^\s*ClusterName\s*=\s*(\S+)\s*$',r.stdout,re.M)
            row['ClusterName']=m.group(1) if m else None
        else:row['stdout']=r.stdout
        report['scheduler_queries'].append(row)
    except Exception as e:
        report['scheduler_queries'].append({'command':command,'error_type':type(e).__name__})
report['scheduler_read_verified']=all(r.get('returncode')==0 for r in report['scheduler_queries']) and report['scheduler_queries'][0].get('ClusterName')=='bscc26a'
path=BASE/'next-web-shell-audit-20261005-r1.json'
data=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
with path.open('xb') as f:f.write(data)
print(json.dumps({'status':'WEB_SHELL_READONLY_RECEIPT','file':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'archive_verified':report.get('archive_verified',False),'scheduler_verified':report['scheduler_read_verified']},ensure_ascii=False),flush=True)
