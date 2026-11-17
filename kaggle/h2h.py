"""Head-to-head: ZeroPrefill vs LMCache MP mode, same machine, prompt and GPUs.

Earlier numbers came from different Kaggle sessions (ZeroPrefill warm 1.60 s in
September, 14.2 s on 2026-11-17; LMCache MP 15.1 s), so they cannot be compared.
This run measures both in one session:

  1. ZeroPrefill: save, then restore twice, each in a fresh vLLM server
  2. LMCache MP:  save, then restore twice, each with fresh LMCache + vLLM
  3. plain vLLM:  same prompt computed twice (cold)

Before every restore the OS page cache is dropped if the container allows it,
so both read the cache from disk. Whether the drop worked and the free memory
are recorded per arm. Restore time is split into disk read and the rest using
ZP_RESTORED (ZeroPrefill) and the prefetch / retrieve lines (LMCache).
"""

import glob
import os
import shutil
import subprocess

import lib
from lmc import (L2_DIR, LMCACHE_SPEC, install_lmcache, lmc_lines, start_lmcache,
                 stop_lmcache, vllm_server, wait_for_l2)

ZP_CACHE = "/root/zp-cache"
REPS = 2


def drop_page_cache():
    """Evict file pages so the next restore reads from disk. Returns True if it worked."""
    r = subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches", shell=True,
                       capture_output=True, text=True)
    ok = r.returncode == 0
    lib.log(f"drop page cache: {'ok' if ok else 'not allowed: ' + r.stderr.strip()}")
    return ok


def free_mib():
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")[1]
    _, total, used, free, shared, cache, avail = out.split()[:7]
    return {"total": int(total), "free": int(free), "buff_cache": int(cache),
            "available": int(avail)}


def dir_mib(root):
    return sum(os.path.getsize(p) for p in glob.glob(f"{root}/**/*", recursive=True)
               if os.path.isfile(p)) / 2**20


def arm_env():
    return {"dropped": drop_page_cache(), "mem_mib": free_mib()}


def zp_arms(res, messages):
    shutil.rmtree(ZP_CACHE, ignore_errors=True)
    srv = lib.start_server("zp_save", cache_root=ZP_CACHE)
    try:
        res["zp_save"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)
    res["zp_cache_mib"] = dir_mib(ZP_CACHE)

    for i in range(REPS):
        tag = f"zp_warm{i}"
        res[f"{tag}_env"] = arm_env()
        srv = lib.start_server(tag, cache_root=ZP_CACHE)
        try:
            res[tag] = lib.chat(messages)
        finally:
            lib.stop_server(srv)
        log = f"{lib.OUT}/server_{tag}.log"
        res[f"{tag}_hit"] = lib.grep(log, "ZP_HIT")
        res[f"{tag}_restored"] = lib.grep(log, "ZP_RESTORED")
        lib.save_json("h2h.json", res)


def lmc_arms(res, messages):
    shutil.rmtree(L2_DIR, ignore_errors=True)
    os.makedirs(L2_DIR, exist_ok=True)
    lmc = start_lmcache("save")
    srv = vllm_server("lmc_save")
    try:
        res["lmc_save"] = lib.chat(messages)
        res["lmc_l2_files"] = wait_for_l2()
    finally:
        # LMCache holds GPU contexts; stop it first so stop_server's VRAM wait passes.
        stop_lmcache(lmc)
        lib.stop_server(srv)
    res["lmc_l2_mib"] = dir_mib(L2_DIR)

    for i in range(REPS):
        tag = f"lmc_warm{i}"
        res[f"{tag}_env"] = arm_env()
        lmc = start_lmcache(tag)
        srv = vllm_server(tag)
        try:
            res[tag] = lib.chat(messages)
        finally:
            stop_lmcache(lmc)
            lib.stop_server(srv)
        res[f"{tag}_log"] = lmc_lines(f"{lib.OUT}/server_{tag}.log", f"{lib.OUT}/lmc_{tag}.log")
        lib.save_json("h2h.json", res)


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    messages = lib.build_messages()  # before LMCache install: same tokenizer as smoke.py
    install_lmcache()
    res = {"lmcache": LMCACHE_SPEC, "reps": REPS, "mem_start_mib": free_mib()}

    zp_arms(res, messages)
    lmc_arms(res, messages)

    srv = lib.start_server("cold", prefix_caching=False)
    try:
        res["cold1"] = lib.chat(messages)
        res["cold2"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)

    ref = res["cold1"]["text"]
    res["cold_deterministic"] = res["cold2"]["text"] == ref
    arms = ["zp_save", "lmc_save"] + [f"{t}_warm{i}" for t in ("zp", "lmc") for i in range(REPS)]
    for arm in arms:
        res[f"{arm}_equals_cold"] = res[arm]["text"] == ref
    res["prompt_tokens"] = (res["cold1"]["usage"] or {}).get("prompt_tokens")
    lib.save_json("h2h.json", res)

    for arm in ["cold1", "cold2"] + arms:
        env = res.get(f"{arm}_env", {})
        lib.log(f"{arm}: ttft {res[arm]['ttft_s']}s  energy {res[arm]['energy_j']}J  "
                f"equals cold: {res.get(arm + '_equals_cold', '-')}  "
                f"page cache dropped: {env.get('dropped', '-')}")
    for i in range(REPS):
        lib.log(f"zp_warm{i} restore: {res[f'zp_warm{i}_restored']}")
        lib.log(f"lmc_warm{i} log: {[l for l in res[f'lmc_warm{i}_log'] if 'Prefetch' in l or 'Retrieved' in l]}")
    lib.log(f"cache size: zp {res['zp_cache_mib']:.0f} MiB, lmc {res['lmc_l2_mib']:.0f} MiB")


if __name__ == "__main__":
    main()
