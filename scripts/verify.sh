#!/usr/bin/env bash
# Correctness: cold output vs cache-restored output must match token for token.
# Each stage is a separate process; 4 GiB of VRAM cannot hold two engines.
set -uo pipefail
exec > /tmp/zp_verify.log 2>&1
cd /mnt/e/inference-engineering/projects/ZeroPrefill
source /root/.venvs/zeroprefill/bin/activate
export HF_HOME=/root/.cache/huggingface
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export ZP_SUMMARY=/tmp/zp_verify_summary.jsonl
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
rm -f /tmp/zp_verify_summary.jsonl

for stage in cold populate warm; do
  echo "########## STAGE: $stage ##########"
  # Each stage writes its own log; the pipe here would swallow vLLM's output.
  python scripts/verify_correctness.py "$stage" --tokens "${ZP_TOKENS:-4096}" \
    --gpu-util "${ZP_GPU_UTIL:-0.70}" > "/tmp/zp_verify_$stage.log" 2>&1
  echo "  rc=$?"
  pkill -9 -f "from multiprocessing" 2>/dev/null
  sleep 5
  grep -oE "ZP_HIT tokens=[0-9]+ of [0-9]+ \([0-9.]+%\)" "/tmp/zp_verify_$stage.log" | tail -1
  grep -oE "COLD_DONE.*|POPULATE_DONE.*|IDENTICAL|MISMATCH" "/tmp/zp_verify_$stage.log" | tail -2
done
