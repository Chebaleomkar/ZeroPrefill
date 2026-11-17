"""LMCache baseline for the smoke test: same prompt, model and GPUs as smoke.py.

  1. LMCache fs backend, empty dir -> prefill, KV chunks written to disk   (save)
  2. fresh server, same dir         -> same prompt restored from disk      (warm)
  3. fresh plain vLLM server        -> same prompt computed twice          (cold)

Backend choice. LMCache 0.5.6's local_disk tier keeps its index only in memory,
so a restarted server cannot see files a previous process wrote (LMCache issue
#1175, fix in open PR #4778; run 1 measured 0 of 16,031 tokens restored). The fs
remote backend looks files up on disk, so it is the LMCache configuration that
can do what ZeroPrefill does.

blocking_timeout_secs is raised from LMCache's default of 10 s. In run 1 one TP
rank's read of its 1.45 GB took longer than 10 s, and vLLM failed the request.

Results go to lmcache_smoke.json in the same shape as smoke.json.
"""

import glob
import os
import shutil
import sys
import time

import lib

LMCACHE_SPEC = "lmcache==0.5.6"  # wheel built against torch 2.13.0 ...
TORCH_SPEC = "torch==2.13.0"     # ... which vllm 0.30.0 pins
CHUNK = 256
FS_DIR = "/root/lmc-fs"
CONNECTOR = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}

FS_CONFIG = f"""\
chunk_size: {CHUNK}
local_cpu: false
max_local_cpu_size: 2
save_unfull_chunk: false
remote_serde: "naive"
blocking_timeout_secs: 120
remote_storage_plugins: ["fs"]
extra_config:
  remote_storage_plugin.fs.base_path: "{FS_DIR}"
  save_chunk_meta: false
"""



def install_lmcache():
    """Install LMCache without letting pip move vllm or torch off the tested pins."""
    lib.sh(f'{sys.executable} -m pip install -q {LMCACHE_SPEC} "{lib.VLLM_SPEC}" "{TORCH_SPEC}"')
    lib.sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E '^(vllm|torch|lmcache|transformers) '")
    # Fail in seconds, not after a server start, if the compiled ops do not load.
    lib.sh(f"{sys.executable} -c 'import lmcache.integration.vllm.vllm_v1_adapter'")


def lmcache_env(name, config):
    path = f"/root/lmcache_{name}.yaml"
    with open(path, "w") as f:
        f.write(config)
    # vLLM seeds its block-hash chain from os.urandom unless PYTHONHASHSEED is
    # set, and LMCache chains its chunk keys from that seed, so without it a
    # restarted server computes different keys and can never hit.
    return {"LMCACHE_CONFIG_FILE": path, "PYTHONHASHSEED": "0"}


def wait_for_files(root, pattern, settle_s=15, max_s=300):
    """LMCache stores asynchronously: wait until the chunk count stops growing."""
    last, stable_since, t0 = -1, time.time(), time.time()
    while time.time() - t0 < max_s:
        n = len(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if n != last:
            last, stable_since = n, time.time()
        elif time.time() - stable_since >= settle_s:
            break
        time.sleep(1)
    lib.log(f"{root}: {last} {pattern} files after {time.time() - t0:.0f}s")
    return last


def dir_mib(root):
    return sum(os.path.getsize(p) for p in glob.glob(f"{root}/**/*", recursive=True)
               if os.path.isfile(p)) / 2**20


def save_then_restore(res, tag, env, cache_dir, pattern, messages):
    """Prefill in one server, kill it, send the same prompt to a fresh one."""
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    save, warm = f"{tag}save", f"{tag}warm"

    srv = lib.start_server(save, kv_transfer=CONNECTOR, extra_env=env)
    try:
        res[save] = lib.chat(messages)
        res[f"{tag}cache_files"] = wait_for_files(cache_dir, pattern)
    finally:
        lib.stop_server(srv)
    res[f"{tag}cache_mib"] = dir_mib(cache_dir)

    srv = lib.start_server(warm, kv_transfer=CONNECTOR, extra_env=env)
    try:
        res[warm] = lib.chat(messages)
    finally:
        lib.stop_server(srv)
    res[f"{tag}lmc_hit"] = lib.grep(f"{lib.OUT}/server_{warm}.log", "LMCache hit tokens")
    res[f"{tag}lmc_retrieved"] = lib.grep(f"{lib.OUT}/server_{warm}.log", "Retrieved")
    res[f"{tag}lmc_errors"] = [line for line in
                               lib.grep(f"{lib.OUT}/server_{save}.log", "ERROR")
                               + lib.grep(f"{lib.OUT}/server_{warm}.log", "ERROR")
                               if "LMCache" in line or "lmcache" in line][:20]


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    messages = lib.build_messages()  # before LMCache install: same tokenizer as smoke.py
    install_lmcache()
    res = {"lmcache": LMCACHE_SPEC, "chunk_size": CHUNK}

    save_then_restore(res, "", lmcache_env("fs", FS_CONFIG), FS_DIR, "*.data", messages)
    lib.save_json("lmcache_smoke.json", res)

    srv = lib.start_server("cold", prefix_caching=False)
    try:
        res["cold1"] = lib.chat(messages)
        res["cold2"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)

    res["cold_deterministic"] = res["cold1"]["text"] == res["cold2"]["text"]
    res["warm_equals_cold"] = res["warm"]["text"] == res["cold1"]["text"]
    res["save_equals_cold"] = res["save"]["text"] == res["cold1"]["text"]
    res["prompt_tokens"] = (res["save"]["usage"] or {}).get("prompt_tokens")
    lib.save_json("lmcache_smoke.json", res)

    for arm in ("cold1", "cold2", "save", "warm"):
        lib.log(f"{arm}: ttft {res[arm]['ttft_s']}s  energy {res[arm]['energy_j']}J  "
                f"output tokens {(res[arm]['usage'] or {}).get('completion_tokens')}")
    lib.log(f"fs cache {res['cache_mib']:.0f} MiB in {res['cache_files']} files")
    lib.log(f"hit: {res['lmc_hit']}")
    lib.log(f"retrieved: {res['lmc_retrieved']}")
    lib.log(f"errors: {res['lmc_errors']}")
    lib.log(f"cold deterministic: {res['cold_deterministic']}  "
            f"warm == cold: {res['warm_equals_cold']}  save == cold: {res['save_equals_cold']}")

if __name__ == "__main__":
    main()
