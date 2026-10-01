#!/usr/bin/env bash
# Start the local vLLM server for pi. Logs to /root/zp-logs/serve.log.
set -uo pipefail

# Create the log dir before any redirection into it, or the shell fails to open
# the file and the server never starts.
LOGDIR=/root/zp-logs
mkdir -p "$LOGDIR"

CACHE_ROOT="${ZP_CACHE:-/root/zeroprefill-cache}"
PORT="${ZP_PORT:-8000}"
MODEL="${ZP_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

cd /mnt/e/inference-engineering/projects/ZeroPrefill
source /root/.venvs/zeroprefill/bin/activate
export HF_HOME=/root/.cache/huggingface
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export ZP_SUMMARY="$LOGDIR/summary.jsonl"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# /tmp is tmpfs in WSL2 and is lost whenever the distro stops, which is exactly
# when a session ends. Keep logs and the summary on the ext4 volume.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --dtype float16 \
  --enforce-eager \
  --max-model-len "${ZP_MAX_LEN:-16384}" \
  --gpu-memory-utilization "${ZP_GPU_UTIL:-0.70}" \
  --max-num-seqs 1 \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --kv-transfer-config "{\"kv_connector\":\"ZeroPrefillConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"zeroprefill.connector\",\"kv_connector_extra_config\":{\"zeroprefill\":{\"cache_root\":\"$CACHE_ROOT\",\"quantization\":\"none\"}}}" \
  > "$LOGDIR/serve.log" 2>&1