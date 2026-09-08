from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
root=Path('/data/semigroup-h20-20260908')
for name in ('launcher.exit','extension-launcher.exit','audit.exit','replay.exit','numerical-floor.exit'):
 assert (root/name).read_text().strip()=='0',name
assert json.loads((root/'AUDIT.json').read_text())['full_cells']==132
assert json.loads((root/'CHECKPOINT_REPLAY.json').read_text())['selected_checkpoints_replayed']==132
source=(root/'publication-source-commit').read_text().strip()
publish=root/'publish'
if not publish.exists():
 subprocess.run(['git','--git-dir='+str(root/'repo.git'),'worktree','add','-b','results/generator-coverage-h20-20260908',str(publish),source],check=True)
else:
 assert subprocess.check_output(['git','-C',str(publish),'rev-parse','HEAD'],universal_newlines=True).strip()==source
dest=publish/'results/generator-coverage-h20-20260908'
dest.mkdir(parents=True,exist_ok=True)
for obsolete in ('assembly.log','assembly.exit'):
 if (dest/obsolete).exists():(dest/obsolete).unlink()
dirs=['wave1-euler','wave2-midpoint','oracle','long-L4','rate-L4','long-L8','smoke-r1','smoke-r2','extension-smoke-rate','extension-smoke-L8','legacy','legacy-diagnosis','numerical-floor','analysis-first','analysis-long-L4','analysis-rate-L4','analysis-final']
for name in dirs:
 if not (dest/name).exists():shutil.copytree(root/name,dest/name)
for path in sorted(root.iterdir()):
 if not path.is_file() or path.name.startswith('assembly'): continue
 if path.suffix in ('.log','.exit','.status','.sh','.py','.json','.bundle','.pid') or path.name in ('run.status','extension.status','audit.status','legacy-inputs.tar.gz'):
  if path.name in ('publication-source-commit',): continue
  shutil.copy2(path,dest/path.name)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
worktrees={}
for name in ['code','code-ec8ec77','code-439473a','code-audit','code-mechanisms','code-replay']:
 d=root/name
 worktrees[name]={'commit':subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],universal_newlines=True).strip()}
 if (d/'experiments/run_generator_coverage.py').exists():worktrees[name]['entry_sha256']=digest(d/'experiments/run_generator_coverage.py')
manifest={'publication_source_commit':source,'worktrees':worktrees,'bundles':{p.name:digest(p) for p in dest.glob('*.bundle')}}
(dest/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in dirs:
 d=dest/name
 for sha in d.glob('artifacts.sha256'):
  for line in sha.read_text().splitlines():
   expected,rel=line.split('  ',1)
   assert digest(d/rel)==expected,(name,rel)
records={str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(dest.rglob('*')) if p.is_file()}
(dest/'ROOT_MANIFEST.json').write_text(json.dumps({'file_count':len(records),'files':records},indent=2)+'\n')
subprocess.run(['git','-C',str(publish),'add','-f','results/generator-coverage-h20-20260908'],check=True)
subprocess.run(['git','-C',str(publish),'-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Publish all 132 H20 generator coverage experiments with audited checkpoints'],check=True)
commit=subprocess.check_output(['git','-C',str(publish),'rev-parse','HEAD'],universal_newlines=True).strip()
(root/'publication-result-commit').write_text(commit+'\n')
archive=root/'generator-coverage-h20-complete.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(dest.iterdir()):tar.add(p,arcname=p.name)
receipt={'source_commit':source,'results_commit':commit,'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),'manifest_files':len(records)}
(root/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
