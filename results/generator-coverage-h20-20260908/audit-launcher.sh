#!/usr/bin/env bash
set -uo pipefail
root=/data/semigroup-h20-20260908
printf 'WAITING_FOR_TRAINING\n' > "$root/audit.status"
while [ ! -f "$root/extension-launcher.exit" ]; do sleep 10; done
if [ "$(cat "$root/extension-launcher.exit")" != 0 ]; then
 printf 'TRAINING_FAILED\n' > "$root/audit.status"
 printf '1\n' > "$root/audit.exit"
 exit 1
fi
printf 'AUDITING\n' > "$root/audit.status"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python -u "$root/code-audit/experiments/audit_generator_coverage.py" "$root"
result=$?
if [ "$result" = 0 ]; then
 /usr/bin/python3 "$root/code-audit/experiments/analyze_generator_coverage.py" "$root/wave1-euler" "$root/wave2-midpoint" "$root/oracle" "$root/long-L4" "$root/rate-L4" "$root/long-L8" --output "$root/analysis-final" > "$root/analysis-final.log"
 result=$?
fi
printf '%s\n' "$result" > "$root/audit.exit"
if [ "$result" = 0 ]; then printf 'COMPLETE\n' > "$root/audit.status"; else printf 'FAILED\n' > "$root/audit.status"; fi
exit "$result"
