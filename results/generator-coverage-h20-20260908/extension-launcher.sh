#!/usr/bin/env bash
set -uo pipefail
root=/data/semigroup-h20-20260908
printf 'WAITING_FOR_FIRST_BATCH\n' > "$root/extension.status"
while [ ! -f "$root/launcher.exit" ] || [ ! -f "$root/extension-tests.exit" ]; do sleep 10; done
if [ "$(cat "$root/launcher.exit")" != 0 ] || [ "$(cat "$root/extension-tests.exit")" != 0 ]; then
  printf 'PREDECESSOR_FAILED\n' > "$root/extension.status"
  printf '1\n' > "$root/extension-launcher.exit"
  exit 1
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python -u "$root/extension-driver.py"
result=$?
printf '%s\n' "$result" > "$root/extension-launcher.exit"
exit "$result"
