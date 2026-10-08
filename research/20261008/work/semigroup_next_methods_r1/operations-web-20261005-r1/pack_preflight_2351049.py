import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
runtime=Path('/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python')
deps='/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps'
sidecar=root/'recovery/job2351049-job2351049-COMPLETE.archive.json'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
if not sidecar.is_file():
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=deps+':'+str(root/'due'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    r=subprocess.run([str(runtime),'-B','-u',str(root/'pack_evidence.py'),'--root',str(root),'--job','2351049'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,timeout=240)
    assert r.returncode==0,r.stderr
receipt=json.loads(sidecar.read_text());archive=Path(receipt['archive'])
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
for label,path in [('SIDECAR',sidecar),('ARCHIVE',archive)]:
    digest=sha(path)
    print(label+'_NAME='+path.name,flush=True)
    print(label+'_BYTES='+str(path.stat().st_size),flush=True)
    print(label+'_SHA_A='+digest[:32],flush=True)
    print(label+'_SHA_B='+digest[32:],flush=True)
