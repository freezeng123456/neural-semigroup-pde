"""Fail before compute when uploaded source/data bytes differ from the run card."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--require-preflight',action='store_true')
    args=p.parse_args(argv)
    root=args.root.resolve()
    card=json.loads((root/'run_card.json').read_text())
    for rel,expected in card['files'].items():
        path=(root/rel).resolve()
        if root not in path.parents or digest(path)!=expected:
            raise RuntimeError('bundle hash/path mismatch: '+rel)
    if args.require_preflight:
        receipt=json.loads((root/'preflight/PREFLIGHT_PASSED.json').read_text())
        if receipt['status']!='passed' or receipt['device']!='cuda':
            raise RuntimeError('allocated CUDA preflight did not pass')
        if receipt['cache_sha256']!=card['files']['cache.pt']:
            raise RuntimeError('preflight cache does not match run card')
        if receipt['protocol_sha256']!=card['files']['source/docs/research/BURGERS_OSG_STAGE_A_PROTOCOL.md']:
            raise RuntimeError('preflight protocol does not match run card')
    print(json.dumps({'bundle_verified':True,'source_commit':card['source_commit'],
        'files_checked':len(card['files']),'preflight_required':args.require_preflight}),flush=True)


if __name__=='__main__':main()
