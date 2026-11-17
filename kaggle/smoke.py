"""Run 0: prove the connector works at TP=2 on Kaggle before any agent run.

  1. connector on, empty cache  -> prefill a long prompt, KV written to disk
  2. fresh server, connector on -> same prompt restored from disk     (warm)
  3. fresh plain vLLM server    -> same prompt computed twice         (cold)
  4. pi headless: plain reply + one bash tool call

Pass: warm output == cold output, ZP_HIT present, pi tool call works.
"""

import glob
import os
import shutil

import lib

CACHE = "/root/zp-cache"  # local disk, regenerated every run


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    messages = lib.build_messages()
    shutil.rmtree(CACHE, ignore_errors=True)
    res = {}

    srv = lib.start_server("save", cache_root=CACHE)
    try:
        res["save"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)
    res["cache_mib"] = sum(os.path.getsize(p) for p in glob.glob(f"{CACHE}/**/*", recursive=True)
                           if os.path.isfile(p)) / 2**20
    res["cache_files"] = len(glob.glob(f"{CACHE}/**/*.safetensors", recursive=True))

    srv = lib.start_server("warm", cache_root=CACHE)
    try:
        res["warm"] = lib.chat(messages)
    finally:
        lib.stop_server(srv)
    res["zp_hit"] = lib.grep(f"{lib.OUT}/server_warm.log", "ZP_HIT")
    res["zp_saved"] = lib.grep(f"{lib.OUT}/server_save.log", "ZP_SAVED")
    res["zp_warnings"] = (lib.grep(f"{lib.OUT}/server_save.log", "could not place")
                          + lib.grep(f"{lib.OUT}/server_warm.log", "mismatch"))

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
    lib.save_json("smoke.json", res)
    lib.log(f"cold ttft {res['cold1']['ttft_s']:.2f}s  warm ttft {res['warm']['ttft_s']:.2f}s  "
            f"energy cold {res['cold1']['energy_j']:.0f}J warm {res['warm']['energy_j']:.0f}J")
    lib.log(f"prompt tokens {res['prompt_tokens']}  cache {res['cache_mib']:.0f} MiB "
            f"in {res['cache_files']} files")
    lib.log(f"ZP_HIT: {res['zp_hit']}  warnings: {res['zp_warnings']}")
    lib.log(f"cold deterministic: {res['cold_deterministic']}  "
            f"warm == cold: {res['warm_equals_cold']}  save == cold: {res['save_equals_cold']}")

    # pi: plumbing and tool calling against the plain server.
    lib.install_pi()
    os.makedirs("/root/pi-smoke", exist_ok=True)
    srv = lib.start_server("pi")
    try:
        secs, out = lib.pi('--no-session -p "Reply with exactly: PI-OK"', "/root/pi-smoke", 300)
        res["pi_reply"] = {"s": secs, "out": out[-2000:]}
        secs, out = lib.pi('--no-session -p "Use the bash tool to run: echo TOOL-OK. '
                           'Then report what it printed."', "/root/pi-smoke", 300)
        res["pi_tool"] = {"s": secs, "out": out[-2000:]}
        res["pi_server_metrics"] = lib.metrics()
    finally:
        lib.stop_server(srv)
    lib.log(f"pi reply: {res['pi_reply']['out'][-200:]!r}")
    lib.log(f"pi tool : {res['pi_tool']['out'][-300:]!r}")
    lib.save_json("smoke.json", res)


if __name__ == "__main__":
    main()
