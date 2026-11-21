"""Smoke test on FrontiersMind's Lumma-0.6B-Instruct: does the connector work on
a model vLLM only runs through its Transformers backend?

Lumma is a custom architecture (trust_remote_code) with no native vLLM model.
lumma_vllm.py serves it through vLLM's Transformers backend, adding the
factorized output projection the generic wrapper lacks. Lumma caches one
pre-RoPE tensor per token and uses it as both key and value; under vLLM it
stores keys and values separately like any other model. Other unknowns: head
size 90 and fp16 on a model trained in bf16 (T4 has no bf16).

  1. fresh plain vLLM server -> a short sanity question, then the long prompt
     twice                                                              (cold)
     fp16 first; float32 if fp16 fails to start or does not answer "Paris"
  2. connector on, empty cache -> long prompt, KV written to disk      (save)
  3. fresh server, connector on -> same prompt restored from disk       (warm)

Pass (the "passed" field): the sanity answer names Paris, cold answers are
non-empty and identical, the warm server restored tokens from disk (ZP_HIT and
ZP_RESTORED), and warm output == cold output. If warm != cold but save == cold
and cold is deterministic, suspect fp16 batch-shape drift before the connector.
"""

import glob
import json
import os
import re
import shutil

os.environ.update({
    "ZP_MODEL": "FrontiersMind/Lumma-0.6B-Instruct",
    "ZP_TP": "1",
    "ZP_QUANT": "none",
    "ZP_MAX_LEN": "12288",       # Lumma's max_position_embeddings
    "ZP_SMOKE_TOKENS": "11000",  # leaves room for the chat template and answer
})

import lib  # noqa: E402  reads the settings above at import

CACHE = "/root/zp-cache"
LUMMA_ARGS = ("--trust-remote-code",)
DTYPES = ("half", "float32")
SANITY = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


def install_plugin():
    """Expose lumma_vllm.register as a vllm.general_plugins entry point.

    vLLM loads general plugins in every process it starts (API server, engine
    core, workers). A dist-info directory on the servers' PYTHONPATH is enough
    for importlib.metadata to find the entry point, without a pip install.
    """
    dist = f"{lib.CODE_DIR}/lumma_vllm-0.0.0.dist-info"
    os.makedirs(dist, exist_ok=True)
    with open(f"{dist}/METADATA", "w") as f:
        f.write("Metadata-Version: 2.1\nName: lumma-vllm\nVersion: 0.0.0\n")
    with open(f"{dist}/entry_points.txt", "w") as f:
        f.write("[vllm.general_plugins]\nlumma = lumma_vllm:register\n")


def zp_events(tag, event):
    out = []
    for line in lib.grep(f"{lib.OUT}/server_{tag}.log", f"{event} "):
        try:
            out.append(json.loads(line.split(f"{event} ", 1)[1]))
        except ValueError:
            pass
    return out


def hit_tokens(tag):
    hits = lib.grep(f"{lib.OUT}/server_{tag}.log", "ZP_HIT")
    m = re.search(r"ZP_HIT tokens=(\d+)", hits[-1]) if hits else None
    return int(m.group(1)) if m else 0


def answer_tokens(run):
    return (run.get("usage") or {}).get("completion_tokens")


def cold_run(messages, res):
    """Serve plain vLLM at the first dtype that starts and answers."""
    for dtype in DTYPES:
        lib.DTYPE = dtype
        tag = f"cold_{dtype}"
        try:
            srv = lib.start_server(tag, prefix_caching=False, extra_args=LUMMA_ARGS)
        except (RuntimeError, TimeoutError) as e:
            res[f"start_failed_{dtype}"] = str(e)[-3000:]
            lib.log(f"{dtype}: server did not start")
            continue
        try:
            sanity = lib.chat(SANITY, max_tokens=32)
            res[f"sanity_{dtype}"] = sanity["text"]
            lib.log(f"{dtype} sanity: {sanity['text']!r}")
            # A wrongly loaded model still answers, with output that ignores its
            # input, and then warm == cold holds whatever KV was restored.
            if "paris" not in sanity["text"].lower():
                continue
            res["cold1"] = lib.chat(messages)
            res["cold2"] = lib.chat(messages)
        except Exception as e:  # the server crashed or returned an error mid-request
            res[f"request_failed_{dtype}"] = repr(e)[-3000:]
            lib.log(f"{dtype}: request failed: {e!r}")
            continue
        finally:
            lib.stop_server(srv)
        res["dtype"] = dtype
        res["resolved_model"] = [line for line in lib.grep(f"{lib.OUT}/server_{tag}.log", "Resolved architecture")
                                 + lib.grep(f"{lib.OUT}/server_{tag}.log", "Transformers modeling backend")][-3:]
        res["attention_backend"] = [line for line in lib.grep(f"{lib.OUT}/server_{tag}.log", "Using ")
                                    if "backend" in line.lower()][-3:]
        return True
    return False


def connector_run(messages, res):
    """Save with the connector on, then restore into a fresh server."""
    srv = lib.start_server("save", cache_root=CACHE, extra_args=LUMMA_ARGS)
    try:
        res["save"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)
    res["cache_mib"] = sum(os.path.getsize(p) for p in glob.glob(f"{CACHE}/**/*", recursive=True)
                           if os.path.isfile(p)) / 2**20
    res["cache_files"] = len(glob.glob(f"{CACHE}/**/*.safetensors", recursive=True))

    srv = lib.start_server("warm", cache_root=CACHE, extra_args=LUMMA_ARGS)
    try:
        res["warm"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    install_plugin()
    messages = lib.build_messages()
    shutil.rmtree(CACHE, ignore_errors=True)
    res = {"model": lib.MODEL, "passed": False}

    if not cold_run(messages, res):
        lib.save_json("lumma.json", res)
        lib.log("Lumma did not serve at any dtype; see start_failed_*, request_failed_* and sanity_* in lumma.json")
        return
    lib.save_json("lumma.json", res)

    try:
        connector_run(messages, res)
    except Exception as e:  # keep the cold results whatever broke
        res["connector_failed"] = str(e)[-3000:]
        lib.save_json("lumma.json", res)
        lib.log("connector run failed; see connector_failed in lumma.json")
        return

    restored = zp_events("warm", "ZP_RESTORED")
    res["prompt_tokens"] = (res["save"]["usage"] or {}).get("prompt_tokens")
    res["warm_hit_tokens"] = hit_tokens("warm")
    res["warm_restored_tokens"] = sum(r.get("tokens", 0) for r in restored)
    res["warm_restore_s"] = [r.get("total_s") for r in restored]
    # Includes the answer segments saved at session end, so an upper bound.
    res["cache_kib_per_saved_token"] = (
        res["cache_mib"] * 1024 / (res["prompt_tokens"] + answer_tokens(res["save"]))
        if res["prompt_tokens"] and answer_tokens(res["save"]) is not None else None)
    res["zp_warnings"] = [line for tag in ("save", "warm")
                          for line in lib.grep(f"{lib.OUT}/server_{tag}.log", "[ZP]")
                          if "could not place" in line or "mismatch" in line
                          or "final save failed" in line]
    res["cold_deterministic"] = res["cold1"]["text"] == res["cold2"]["text"]
    res["warm_equals_cold"] = (res["warm"]["text"] == res["cold1"]["text"]
                               and answer_tokens(res["warm"]) == answer_tokens(res["cold1"]))
    res["save_equals_cold"] = res["save"]["text"] == res["cold1"]["text"]
    res["passed"] = bool(res["cold1"]["text"].strip()
                         and res["cold_deterministic"]
                         and res["warm_hit_tokens"] > 0
                         and res["warm_restored_tokens"] > 0
                         and res["warm_equals_cold"]
                         and not res["zp_warnings"])
    lib.save_json("lumma.json", res)

    lib.log(f"dtype {res['dtype']}  model {res['resolved_model']}  backend {res['attention_backend']}")
    lib.log(f"prompt {res['prompt_tokens']} tokens  ZP_HIT {res['warm_hit_tokens']}  "
            f"restored {res['warm_restored_tokens']} in {res['warm_restore_s']}s  "
            f"cache {res['cache_mib']:.0f} MiB in {res['cache_files']} files "
            f"(<= {res['cache_kib_per_saved_token']} KiB/token)")
    lib.log(f"ttft cold {res['cold1']['ttft_s']}s  warm {res['warm']['ttft_s']}s  "
            f"energy cold {res['cold1']['energy_j']:.0f}J  warm {res['warm']['energy_j']:.0f}J")
    lib.log(f"cold deterministic: {res['cold_deterministic']}  warm == cold: "
            f"{res['warm_equals_cold']}  save == cold: {res['save_equals_cold']}  "
            f"warnings: {res['zp_warnings']}")
    lib.log(f"PASSED: {res['passed']}")


if __name__ == "__main__":
    main()
