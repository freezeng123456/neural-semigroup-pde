#!/usr/bin/env python3
"""Post-screen information-matched control; see the explicit extension protocol."""
import argparse,shutil
from pathlib import Path
import run_semigroup_discovery as d

def matrix():
    return [dict(track='forced',seed=s,n=128,noise=0.,model='clock_query') for s in d.SEEDS]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-cache',type=Path);p.add_argument('--cell',type=int,default=0);p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='prepare':
        if a.source_cache is None:raise ValueError('Frozen original cache required')
        a.root.mkdir(parents=True,exist_ok=False);shutil.copy2(a.source_cache,a.root/'cache.pt')
        d.base.dump(a.root/'cache_metadata.json',dict(source=str(a.source_cache),sha256=d.base.digest(a.root/'cache.pt'),
            post_screen_extension=True,shared_data=True))
        d.base.dump(a.root/'matrix.json',matrix())
    else:
        d.matrix=matrix
        d.run(a.root,a.cell,a.device,a.updates,a.smoke)
if __name__=='__main__':main()
