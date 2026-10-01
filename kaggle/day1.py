"""Day 1: build a real pi session on a real repo, then persist its KV cache.

pi works through TASKS in one session (connector on, so every prompt's KV is
written to disk) until the context reaches TARGET tokens. On a clean exit the
outputs hold everything Day 2 needs: the KV cache, pi's session file and the
repo as pi left it.
"""

import os
import shlex
import shutil

os.environ.setdefault("ZP_MAX_LEN", "32768")  # YaRN failed to start on Day 1 v1
import lib  # noqa: E402

REPO_URL = "https://github.com/psf/requests"
REPO_TAG = "v2.32.3"
REPO = "/root/work/requests"  # identical on Day 2: pi puts cwd in its prompt
SESSIONS = f"{lib.WORK}/pi-sessions"
CACHE = "/root/zp-cache"
TARGET = int(os.environ.get("ZP_TARGET_TOKENS", "36000"))
HEADROOM = 6000  # one more turn can add a large file read

TASKS = [
    "Read README.md and src/requests/api.py. Explain what the public API looks like.",
    "Read src/requests/sessions.py in full. Explain how Session.request builds and "
    "sends a request, including how redirects are resolved.",
    "Read src/requests/adapters.py in full. Explain how HTTPAdapter handles retries, "
    "timeouts and connection pooling.",
    "Read src/requests/models.py in full. Explain how PreparedRequest prepares the "
    "body for file uploads and for JSON.",
    "Add a function redact_url(url) to src/requests/utils.py that replaces any "
    "password in the URL with '***'. Write tests in tests_zp/test_redact.py and run "
    "them with: PYTHONPATH=src python -m pytest tests_zp -q -p no:cacheprovider",
    "Read src/requests/auth.py and src/requests/cookies.py and suggest three "
    "concrete improvements.",
    "Read src/requests/utils.py and list the public functions that have no test in "
    "tests/test_utils.py.",
]


def context_tokens():
    """Prompt size of the latest request, from the connector's lookup log."""
    lines = lib.grep(f"{lib.OUT}/server_day1.log", "ZP_LOOKUP")
    return int(lines[-1].rsplit("tokens=", 1)[1]) if lines else 0


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    lib.install_pi()
    shutil.rmtree(REPO, ignore_errors=True)
    lib.sh(f"git clone -q --depth 1 --branch {REPO_TAG} {REPO_URL} {REPO}")
    shutil.rmtree(CACHE, ignore_errors=True)

    res = {"repo": f"{REPO_URL}@{REPO_TAG}", "turns": []}
    srv = lib.start_server("day1", cache_root=CACHE)
    res["max_len"] = lib.MAX_LEN
    proxy = lib.start_proxy("day1")
    try:
        for i, task in enumerate(TASKS):
            cont = "-c " if i else ""
            e0 = lib.gpu_energy_mj()
            secs, out = lib.pi(f"--session-dir {SESSIONS} {cont}-p {shlex.quote(task)}",
                               REPO, timeout=1800)
            ctx = context_tokens()
            res["turns"].append({"task": task, "s": secs, "context_tokens": ctx,
                                 "energy_j": (lib.gpu_energy_mj() - e0) / 1000.0,
                                 "out_tail": out[-1500:]})
            lib.save_json("day1.json", res)
            lib.log(f"turn {i + 1}: {secs:.0f}s, context {ctx} tokens")
            if ctx >= TARGET or ctx + HEADROOM > lib.MAX_LEN:
                break
    finally:
        lib.stop_proxy(proxy)
        lib.stop_server(srv)

    shutil.copytree(CACHE, f"{lib.WORK}/zp-cache", dirs_exist_ok=True)
    lib.sh(f"tar -C {os.path.dirname(REPO)} -cf {lib.WORK}/repo.tar {os.path.basename(REPO)}")
    res["final_context_tokens"] = context_tokens()
    res["cache_gib"] = sum(os.path.getsize(os.path.join(d, f))
                           for d, _, fs in os.walk(CACHE) for f in fs) / 2**30
    res["zp_saved"] = lib.grep(f"{lib.OUT}/server_day1.log", "ZP_SAVED")[-3:]
    lib.save_json("day1.json", res)
    lib.log(f"done: context {res['final_context_tokens']} tokens, "
            f"cache {res['cache_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
