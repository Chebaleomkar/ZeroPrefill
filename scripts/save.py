"""Phase 1: prefill a 20k-token prompt and let the connector persist KV blocks.

Run this, let it exit, then run restore.py. The two are separate processes, so
the restore is a genuine cold start against an empty GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from zeroprefill.prompt import token_ids_for

CACHE_ROOT = os.environ.get("ZP_CACHE", "/root/zeroprefill-cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--out", default="./cache/save_report.json")
    args = ap.parse_args()

    ids = token_ids_for(args.tokens, args.model)
    print(f"[save] prompt tokens: {len(ids)}")

    llm = LLM(
        model=args.model,
        dtype="float16",          # Turing (sm_75) has no bf16 datapath
        enforce_eager=True,       # avoid CUDA-graph capture issues on sm_75
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.tokens + 64,
        max_num_seqs=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ZeroPrefillConnector",
            kv_role="kv_both",
            kv_connector_module_path="zeroprefill.connector",
            kv_connector_extra_config={
                "zeroprefill": {"cache_root": CACHE_ROOT, "quantization": "none"}
            },
        ),
    )

    # vLLM 0.30 takes {"prompt_token_ids": [...]} as a TokensPrompt dict.
    t0 = time.perf_counter()
    out = llm.generate(
        [{"prompt_token_ids": ids}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )
    cold_s = time.perf_counter() - t0

    # Count segments as directories under the layout fingerprint, not every
    # file: each segment holds one safetensors file per layer.
    fp = CACHE_ROOT
    n_segments = sum(len(dirs) for _, dirs, _ in os.walk(fp))
    n_files = sum(len(files) for _, _, files in os.walk(fp))
    bytes_on_disk = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(fp)
        for f in fs
    )

    report = {
        "phase": "save",
        "prompt_tokens": len(ids),
        "cold_prefill_s": round(cold_s, 3),
        "cache_root": os.path.abspath(fp),
        "segments": n_segments,
        "files": n_files,
        "bytes_on_disk": bytes_on_disk,
        "mib_on_disk": round(bytes_on_disk / 1024**2, 2),
        "first_token": out[0].outputs[0].text,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
