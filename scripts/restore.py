"""Phase 2: cold-start restore. New process, empty GPU, disk-only KV.

The headline number is warm_prefill_s vs the cold figure recorded by save.py.
Grep the log for "ZeroPrefill external hit" to confirm the load path fired.
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
    ap.add_argument("--gpu-util", type=float, default=0.70)
    ap.add_argument("--save-report", default="/tmp/zp_save.json")
    ap.add_argument("--out", default="/tmp/zp_restore.json")
    args = ap.parse_args()

    ids = token_ids_for(args.tokens, args.model)

    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
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

    t0 = time.perf_counter()
    out = llm.generate(
        [{"prompt_token_ids": ids}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )
    warm_s = time.perf_counter() - t0

    cold_s = None
    if os.path.exists(args.save_report):
        with open(args.save_report) as f:
            cold_s = json.load(f).get("cold_prefill_s")

    report = {
        "phase": "restore",
        "prompt_tokens": len(ids),
        "restore_prefill_s": round(warm_s, 3),
        "cold_prefill_s": cold_s,
        "speedup": round(cold_s / warm_s, 2) if cold_s else None,
        "first_token": out[0].outputs[0].text,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
