"""Correctness: does a restored cache produce the same tokens as a cold run?

Timing alone proves nothing. This generates a fixed prompt, decodes N tokens
cold (empty cache) and warm (populated cache), and compares the outputs token
for token. Any divergence means the onloaded KV is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zeroprefill.prompt import token_ids_for  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 24
OUT = "/tmp/zp_verify.json"

# Three engines cannot share 4 GiB of VRAM, so each stage runs as its own
# process: `python verify_correctness.py cold|populate|warm`.
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["cold", "populate", "warm"])
ap.add_argument("--tokens", type=int, default=4096)
ap.add_argument("--gpu-util", type=float, default=0.70)
A = ap.parse_args()
STAGE = A.stage

TOKENS = A.tokens


def build(cache_root: str) -> LLM:
    return LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=A.gpu_util,
        max_model_len=TOKENS + 256,
        max_num_seqs=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ZeroPrefillConnector",
            kv_role="kv_both",
            kv_connector_module_path="zeroprefill.connector",
            kv_connector_extra_config={
                "zeroprefill": {"cache_root": cache_root, "quantization": "none"}
            },
        ),
    )


def decode(llm: LLM, ids: list[int]) -> list[int]:
    out = llm.generate(
        [{"prompt_token_ids": ids}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_NEW),
    )
    return list(out[0].outputs[0].token_ids)


ids = token_ids_for(TOKENS, MODEL)
print(f"[verify] stage={STAGE} prompt tokens: {len(ids)}", flush=True)

COLD_ROOT = "/tmp/zp_verify_cold"
WARM_ROOT = "/tmp/zp_verify_warm"
COLD_OUT = "/tmp/zp_verify_cold.json"
WARM_OUT = "/tmp/zp_verify_warm.json"

if STAGE == "cold":
    os.system(f"rm -rf {COLD_ROOT}")
    ids_out = decode(build(COLD_ROOT), ids)
    json.dump({"ids": ids_out, "prompt_tokens": len(ids)}, open(COLD_OUT, "w"))
    print("COLD_DONE", ids_out[:8])
elif STAGE == "populate":
    os.system(f"rm -rf {WARM_ROOT}")
    ids_out = decode(build(WARM_ROOT), ids)
    print("POPULATE_DONE", ids_out[:8])
elif STAGE == "warm":
    warm = decode(build(WARM_ROOT), ids)
    cold = json.load(open(COLD_OUT))["ids"]
    rep = {"prompt_tokens": len(ids), "max_new_tokens": MAX_NEW,
           "cold_ids": cold, "warm_ids": warm, "identical": cold == warm}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("IDENTICAL" if cold == warm else "MISMATCH")
else:
    raise SystemExit(f"unknown stage {STAGE}")
