"""Day 2: resume Day 1's session on a fresh machine, warm vs cold.

Both arms fork the SAME saved pi session and send the SAME next message, so
the only difference is whether the KV cache is restored from disk:

  warm : vLLM + ZeroPrefill connector, cache copied from Day 1's output
  cold : plain vLLM, the whole history is prefilled again

Arms alternate (cold, warm, cold, warm, ...), each on a freshly started server
with the repo reset to how pi left it on Day 1.
"""

import glob
import json
import os
import shutil
import time

os.environ.setdefault("ZP_MAX_LEN", "32768")  # must match Day 1: RoPE scaling changes cached K
import lib  # noqa: E402

REPO_PARENT = "/root/work"
REPO = f"{REPO_PARENT}/requests"  # must match Day 1
CACHE = "/root/zp-cache"
REPS = int(os.environ.get("ZP_REPS", "2"))
NEXT_MESSAGE = ("Good morning. Briefly summarize what we did in this session, then read "
                "src/requests/structures.py and explain how CaseInsensitiveDict works.")


def day1_dir():
    hits = glob.glob("/kaggle/input/**/results/day1.json", recursive=True)
    if not hits:
        raise FileNotFoundError("Day 1 output not attached (kernel_sources)")
    return os.path.dirname(os.path.dirname(hits[0]))


def run_arm(arm, rep, day1, session_file):
    tag = f"{arm}{rep}"
    shutil.rmtree(REPO, ignore_errors=True)
    lib.sh(f"tar -C {REPO_PARENT} -xf {day1}/repo.tar")
    srv = lib.start_server(tag, cache_root=CACHE if arm == "warm" else None)
    proxy = lib.start_proxy(tag)
    try:
        e0 = lib.gpu_energy_mj()
        secs, out = lib.pi(f"--session-dir /root/sessions-{tag} --fork {session_file} "
                           f"-p {json.dumps(NEXT_MESSAGE)}", REPO, timeout=1800)
        energy = (lib.gpu_energy_mj() - e0) / 1000.0
        server = lib.metrics()
    finally:
        lib.stop_proxy(proxy)
        lib.stop_server(srv)
    with open(f"{lib.OUT}/requests_{tag}.jsonl") as f:
        reqs = [json.loads(line) for line in f]
    first = reqs[0]
    return {"arm": arm, "rep": rep, "pi_s": secs, "pi_energy_j": energy,
            "requests": len(reqs),
            "first": {k: first[k] for k in ("ttft_s", "total_s", "energy_j", "output")},
            "first_prompt_messages": len(first["request"].get("messages", [])),
            "zp_hit": lib.grep(f"{lib.OUT}/server_{tag}.log", "ZP_HIT")[:1],
            "server": server, "out_tail": out[-1500:]}


def main():
    lib.sh("nvidia-smi")
    day1 = day1_dir()
    lib.install_vllm()
    lib.install_pi()
    os.makedirs(REPO_PARENT, exist_ok=True)

    res = {"day1": json.load(open(f"{day1}/results/day1.json")), "arms": []}
    res["day1"].pop("turns", None)
    t0 = time.time()
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.copytree(f"{day1}/zp-cache", CACHE)
    res["fetch_cache_s"] = time.time() - t0
    session_file = sorted(glob.glob(f"{day1}/pi-sessions/**/*.jsonl", recursive=True))[-1]
    res["session_file"] = session_file

    for rep in range(REPS):
        for arm in ("cold", "warm"):
            r = run_arm(arm, rep, day1, session_file)
            res["arms"].append(r)
            lib.save_json("day2.json", res)
            lib.log(f"{arm} rep {rep}: first-token {r['first']['ttft_s']:.2f}s, "
                    f"first-request energy {r['first']['energy_j']:.0f}J, "
                    f"pi total {r['pi_s']:.0f}s, hit {r['zp_hit']}")

    firsts = [json.dumps(a["first"]["output"], sort_keys=True) for a in res["arms"]]
    res["first_outputs_identical"] = len(set(firsts)) == 1
    lib.save_json("day2.json", res)
    lib.log(f"first responses identical across all arms: {res['first_outputs_identical']}")


if __name__ == "__main__":
    main()
