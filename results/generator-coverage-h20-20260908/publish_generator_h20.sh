#!/usr/bin/env bash
set -euo pipefail
IFS= read -r GH_TOKEN
export GH_TOKEN
ASKPASS=$(mktemp /tmp/generator-askpass.XXXXXX)
trap 'rm -f "$ASKPASS"; unset GH_TOKEN GIT_ASKPASS' EXIT
cat >"$ASKPASS" <<'ASKPASS_EOF'
#!/bin/sh
case "$1" in
  *Username*) printf %s x-access-token ;;
  *) printf %s "$GH_TOKEN" ;;
esac
ASKPASS_EOF
chmod 700 "$ASKPASS"
export GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0
/usr/bin/python3 - <<'PY_REMOTE'
import os,urllib.request,json
for url in ('https://api.github.com/user','https://api.github.com/repos/freezeng123456/neural-semigroup-pde'):
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=30) as response:
  assert response.status==200
print('GitHub account and repository authorization verified.')
PY_REMOTE
publish=/data/semigroup-h20-20260908/publish
source_commit=$(cat /data/semigroup-h20-20260908/publication-source-commit)
git -C "$publish" -c credential.helper= push https://github.com/freezeng123456/neural-semigroup-pde.git "$source_commit:refs/heads/experiment/generator-coverage-h20-20260908"
git -C "$publish" -c credential.helper= push https://github.com/freezeng123456/neural-semigroup-pde.git HEAD:refs/heads/results/generator-coverage-h20-20260908
git -C "$publish" rev-parse HEAD
git -C "$publish" -c credential.helper= ls-remote https://github.com/freezeng123456/neural-semigroup-pde.git refs/heads/results/generator-coverage-h20-20260908 refs/heads/experiment/generator-coverage-h20-20260908
