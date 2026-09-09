#!/usr/bin/env python3
"""Audit all cells and independently replay primary endpoints on CPU."""
import argparse,csv,json,math
from pathlib import Path
import torch
import run_semigroup_confirmation as d

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    cache_hash=d.base.digest(a.root/'cache.pt')
    data=torch.load(a.root/'cache.pt',map_location='cpu')
    evidence=[]
    for i,c in enumerate(d.matrix()):
        out=a.root/'cells'/f'{i:03d}'
        assert (out/'done').is_file() and not (out/'failed').exists(),f'missing cell {i}'
        config=json.loads((out/'config.json').read_text())
        summary=json.loads((out/'summary.json').read_text())
        assert all(config[k]==v for k,v in c.items())
        assert config['cache_sha256']==cache_hash and config['updates']==20000
        metrics=list(csv.DictReader((out/'metrics.csv').open()))
        assert int(metrics[-1]['update'])==20000
        assert all(math.isfinite(float(row[k])) for row in metrics for k in ('loss','validation_mse','training_seconds'))
        initial=torch.load(out/'initial.pt',map_location='cpu')
        final=torch.load(out/'final.pt',map_location='cpu')
        best=torch.load(out/'best.pt',map_location='cpu')
        assert all(bool(torch.isfinite(t).all()) for state in (initial,final,best) for t in state.values())
        assert any(not torch.equal(initial[k],final[k]) for k in initial)
        chosen=min(metrics,key=lambda r:float(r['validation_mse']))
        assert summary['best_update']==int(chosen['update'])
        extra=list(out.glob('fixed_*.pt'))+list(out.glob('best_*.pt'))
        assert len(extra)==8
        for checkpoint in extra:
            state=torch.load(checkpoint,map_location='cpu')
            assert all(bool(torch.isfinite(t).all()) for t in state.values())
        model=d.Flow(c['model']);model.load_state_dict(best);model.eval()
        pool=data['pools'][f"test-{c['track']}-0.02"]
        diffs=[]
        for t in (1.2,2.4):
            for lag in (.06,.12):
                pred,_,_=d.rollout(model,pool['u'],t,lag)
                mse=float((pred-pool['targets'][str(t)]).square().mean())
                original=next(r['mse'] for r in summary['rows'] if r['split']=='test' and r['nu']==.02 and r['horizon']==t and r['lag']==lag)
                delta=abs(mse-original);limit=max(1e-8,.02*original)
                assert delta<=limit,(i,t,lag,mse,original)
                diffs.append(dict(horizon=t,lag=lag,cpu_mse=mse,recorded_gpu_mse=original,absolute_difference=delta,tolerance=limit))
        evidence.append(dict(cell=i,model=c['model'],replays=diffs,best_update=summary['best_update'],
            checkpoint_sha256={name:d.base.digest(out/(name+'.pt')) for name in ('initial','final','best')}))
        print(f'cell {i}: checkpoint audit and four endpoint replays passed',flush=True)
    d.base.dump(a.root/'CPU_REPLAY_AUDIT.json',dict(status='PASSED',cells=len(evidence),checkpoints=len(evidence)*11,base_checkpoint_hashes=len(evidence)*3,additional_finite_checkpoint_checks=len(evidence)*8,
        primary_endpoint_replays=len(evidence)*4,cache_sha256=cache_hash,evidence=evidence))
if __name__=='__main__':main()
