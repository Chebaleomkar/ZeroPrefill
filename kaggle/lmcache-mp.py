"""LMCache multi-process (MP) mode baseline: same prompt, model and GPUs as smoke.py.

MP mode runs LMCache as its own server process that vLLM connects to. It is the
mode LMCache supports going forward; the in-process connector measured in
lmcache-smoke.py returned wrong output on vLLM 0.30.0's LBNHC layout (LMCache
#6156, closed: in-process mode is deprecated). MP mode registers vLLM's KV
memory directly and does not use the gather kernel behind #6156.

  1. LMCache server A (empty fs L2) + vLLM -> prefill, KV stored to L1 and L2   (save)
  2. stop both. Fresh LMCache server B (empty L1, same L2 dir) + fresh vLLM
     -> same prompt, served from disk                                         (warm)
     This is the "everything restarted" case ZeroPrefill measures.
  3. keep server B, restart only vLLM -> same prompt, served from B's L1      (warm_l1)
     The deployment LMCache recommends; open issue #6257 reports corruption
     in this case under container PID reuse.
  4. fresh plain vLLM server -> same prompt computed twice                     (cold)

Results go to lmcache_mp.json in the same shape as smoke.json.
"""

import glob
import os
import shutil

import lib
from lmc import (L2_DIR, LMCACHE_SPEC, install_lmcache, lmc_lines, start_lmcache,
                 stop_lmcache, vllm_server, wait_for_l2)


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    messages = lib.build_messages()  # before LMCache install: same tokenizer as smoke.py
    install_lmcache()
    shutil.rmtree(L2_DIR, ignore_errors=True)
    os.makedirs(L2_DIR, exist_ok=True)
    res = {"lmcache": LMCACHE_SPEC, "mode": "mp", "l2": "fs"}

    # 1. save
    lmc = start_lmcache("a")
    try:
        srv = vllm_server("save")
        try:
            res["save"] = lib.chat(messages)
            res["l2_files"] = wait_for_l2()
        finally:
            lib.stop_server(srv)
    finally:
        stop_lmcache(lmc)
    res["l2_mib"] = sum(os.path.getsize(p) for p in glob.glob(f"{L2_DIR}/**/*", recursive=True)
                        if os.path.isfile(p)) / 2**20
    res["save_log"] = lmc_lines(f"{lib.OUT}/server_save.log", f"{lib.OUT}/lmc_a.log")
    lib.save_json("lmcache_mp.json", res)

    # 2. everything restarted, KV only on disk; 3. vLLM restarted, server kept
    lmc = start_lmcache("b")
    try:
        srv = vllm_server("warm")
        try:
            res["warm"] = lib.chat(messages)
        finally:
            lib.stop_server(srv)
        res["warm_log"] = lmc_lines(f"{lib.OUT}/server_warm.log", f"{lib.OUT}/lmc_b.log")
        lib.save_json("lmcache_mp.json", res)

        srv = vllm_server("warm_l1")
        try:
            res["warm_l1"] = lib.chat(messages)
        finally:
            lib.stop_server(srv)
        res["warm_l1_log"] = lmc_lines(f"{lib.OUT}/server_warm_l1.log", f"{lib.OUT}/lmc_b.log")
    finally:
        stop_lmcache(lmc)
    lib.save_json("lmcache_mp.json", res)

    # 4. cold
    srv = lib.start_server("cold", prefix_caching=False)
    try:
        res["cold1"] = lib.chat(messages)
        res["cold2"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)

    res["cold_deterministic"] = res["cold1"]["text"] == res["cold2"]["text"]
    for arm in ("save", "warm", "warm_l1"):
        res[f"{arm}_equals_cold"] = res[arm]["text"] == res["cold1"]["text"]
    res["prompt_tokens"] = (res["save"]["usage"] or {}).get("prompt_tokens")
    lib.save_json("lmcache_mp.json", res)

    for arm in ("cold1", "cold2", "save", "warm", "warm_l1"):
        lib.log(f"{arm}: ttft {res[arm]['ttft_s']}s  energy {res[arm]['energy_j']}J  "
                f"output tokens {(res[arm]['usage'] or {}).get('completion_tokens')}  "
                f"equals cold: {res.get(arm + '_equals_cold', '-')}")
    lib.log(f"L2: {res['l2_mib']:.0f} MiB in {res['l2_files']} files")
    for line in res["warm_log"][-10:]:
        lib.log(f"warm log: {line}")


if __name__ == "__main__":
    main()
