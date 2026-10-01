#!/usr/bin/env bash
# Offline e2e: save in one process (dies), restore in a second, same tokens.
set -uo pipefail
exec > /tmp/zp_offline_e2e.log 2>&1
cd /mnt/e/inference-engineering/projects/ZeroPrefill
source /root/.venvs/zeroprefill/bin/activate
export HF_HOME=/root/.cache/huggingface
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export ZP_SUMMARY=/tmp/zp_summary.jsonl
export ZP_CACHE=/root/zeroprefill-cache
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

rm -rf "$ZP_CACHE" /tmp/zp_summary.jsonl

echo "########## PHASE 1: COLD PREFILL + SAVE (process A) ##########"
python scripts/save.py --tokens 20000 --gpu-util 0.70 --out /tmp/zp_save.json
pkill -9 -f "from multiprocessing" 2>/dev/null
sleep 3
echo "segments on disk: $(find $ZP_CACHE -mindepth 2 -maxdepth 2 -type d | wc -l)"
echo "files:   $(find $ZP_CACHE -type f | wc -l)"
echo "size:    $(du -sh $ZP_CACHE | cut -f1)"
echo "--- connector save record ---"
cat /tmp/zp_summary.jsonl

echo
echo "########## PHASE 2: RESTORE (process B, cold GPU) ##########"
python scripts/restore.py --tokens 20000 --gpu-util 0.70 \
  --save-report /tmp/zp_save.json --out /tmp/zp_restore.json
grep -oE "ZP_HIT tokens=[0-9]+ of [0-9]+ \([0-9.]+%\)" /tmp/zp_offline_e2e.log | tail -2

echo
echo "########## COMPARISON ##########"
python - <<'PY'
import json
s = json.load(open("/tmp/zp_save.json"))
r = json.load(open("/tmp/zp_restore.json"))
cold, warm = s["cold_prefill_s"], r["restore_prefill_s"]
print(f"prompt tokens      : {s['prompt_tokens']}")
print(f"cold prefill       : {cold:.2f} s")
print(f"restored prefill   : {warm:.2f} s")
print(f"speedup            : {cold/warm:.2f}x")
print(f"time saved         : {cold-warm:.2f} s")
print(f"first token cold   : {s['first_token']!r}")
print(f"first token warm   : {r['first_token']!r}")
print(f"IDENTICAL OUTPUT   : {s['first_token'] == r['first_token']}")
PY
