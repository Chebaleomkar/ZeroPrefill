"""Shared helpers for the Kaggle runs: install, serve, measure.

Runs on Kaggle's 2x T4 image (driver 580 / CUDA 13.0), where vllm==0.30.0 with
AWQ and tensor parallel 2 is known to work (verified in ZeroHost-vLLM-v2).
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("ZP_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
SERVED_NAME = "zp-model"
TP = int(os.environ.get("ZP_TP", "2"))
PORT = 8001        # vLLM
PROXY_PORT = 8000  # pi talks to the logging proxy, which forwards to vLLM
# Qwen2.5 is natively 32k. Longer windows use YaRN rope scaling.
MAX_LEN = int(os.environ.get("ZP_MAX_LEN", "32768"))
NATIVE_LEN = 32768
VLLM_SPEC = "vllm==0.30.0"
NODE_VERSION = "v24.14.0"  # same as the local machine
PI_SPEC = "@earendil-works/pi-coding-agent@0.84.3"

WORK = "/kaggle/working"
CODE_DIR = f"{WORK}/zp_code"   # written by the bundled script header
OUT = f"{WORK}/results"
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[zp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, **kw):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kw)


def save_json(name, obj):
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- install

def install_vllm():
    sh(f"{sys.executable} -m pip install -q {VLLM_SPEC}")


def install_pi():
    node_dir = f"/root/node-{NODE_VERSION}"
    if not os.path.exists(f"{node_dir}/bin/node"):
        sh(f"curl -fsSL https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz"
           f" | tar -xJ -C /root && mv /root/node-{NODE_VERSION}-linux-x64 {node_dir}")
    os.environ["PATH"] = f"{node_dir}/bin:" + os.environ["PATH"]
    sh(f"npm install -g --silent {PI_SPEC}")
    sh("pi --version")

    # pi needs an apiKey field even for a keyless local server.
    models = {"providers": {"zp": {
        "baseUrl": f"http://127.0.0.1:{PROXY_PORT}/v1",
        "api": "openai-completions",
        "apiKey": "zp-local",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False,
                   "supportsStore": False, "maxTokensField": "max_tokens"},
        "models": [{"id": SERVED_NAME, "name": MODEL, "contextWindow": MAX_LEN,
                    "maxTokens": 4096,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }}}
    os.makedirs("/root/.pi/agent", exist_ok=True)
    with open("/root/.pi/agent/models.json", "w") as f:
        json.dump(models, f, indent=2)
    # Compaction rewrites history into a summary, which changes the prompt
    # prefix and defeats any KV reuse. With a 32k window it would fire at ~16k.
    with open("/root/.pi/agent/settings.json", "w") as f:
        json.dump({"compaction": {"enabled": False}}, f, indent=2)


PI_FLAGS = ("--provider zp --model zp-model --no-context-files --no-skills "
            "--no-extensions --no-prompt-templates --offline")


def pi(args, cwd, timeout):
    """Run pi headless; returns (seconds, combined output)."""
    t0 = time.time()
    r = subprocess.run(f"pi {PI_FLAGS} {args}", shell=True, cwd=cwd, timeout=timeout,
                       capture_output=True, text=True)
    return time.time() - t0, r.stdout + r.stderr


# ---------------------------------------------------------------- server

def start_server(tag, cache_root=None, prefix_caching=True):
    """cache_root=None serves plain vLLM, the no-ZeroPrefill baseline."""
    global MAX_LEN
    try:
        return _start_server(tag, cache_root, prefix_caching)
    except RuntimeError:
        if MAX_LEN <= NATIVE_LEN:
            raise
        log(f"server failed at max_len {MAX_LEN} with YaRN; retrying at {NATIVE_LEN}")
        MAX_LEN = NATIVE_LEN
        return _start_server(tag, cache_root, prefix_caching)


def _start_server(tag, cache_root, prefix_caching):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL, "--served-model-name", SERVED_NAME,
           "--host", "127.0.0.1", "--port", str(PORT),
           "--dtype", "half",  # T4 has no bf16
           "--quantization", "awq",
           "--tensor-parallel-size", str(TP),
           "--max-model-len", str(MAX_LEN),
           "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", "4",
           "--enforce-eager",  # same for both arms; drop only after A==B holds
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if MAX_LEN > NATIVE_LEN:
        cmd += ["--hf-overrides", json.dumps({"rope_parameters": {
            "rope_type": "yarn", "factor": MAX_LEN / NATIVE_LEN,
            "original_max_position_embeddings": NATIVE_LEN, "rope_theta": 1000000.0}})]
    if not prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if cache_root:
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "ZeroPrefillConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "zeroprefill.connector",
            "kv_connector_extra_config": {"zeroprefill": {
                "cache_root": cache_root, "quantization": "awq"}}})]
    env = {**os.environ,
           "PYTHONPATH": f"{CODE_DIR}:" + os.environ.get("PYTHONPATH", ""),
           "ZP_SUMMARY": f"{OUT}/summary.jsonl"}
    logpath = f"{OUT}/server_{tag}.log"
    log(f"starting server [{tag}] connector={'on' if cache_root else 'off'}")
    proc = subprocess.Popen(cmd, stdout=open(logpath, "w"), stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            raise RuntimeError(f"server [{tag}] exited:\n{tail(logpath)}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    log(f"server [{tag}] ready after {time.time() - t0:.0f}s")
                    return proc
        except OSError:
            pass
        time.sleep(5)
    stop_server(proc)
    raise TimeoutError(f"server [{tag}] not ready:\n{tail(logpath)}")


def stop_server(proc):
    """Kill the whole process group (TP workers hold the GPUs), wait for VRAM."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    for _ in range(60):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if all(int(m) < 500 for m in used):
            break
        time.sleep(2)
    log("server stopped")


def tail(path, n=60):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def grep(path, needle):
    try:
        with open(path) as f:
            return [line.rstrip() for line in f if needle in line]
    except OSError:
        return []


# ---------------------------------------------------------------- measurement

def gpu_energy_mj():
    """Total energy counters of all GPUs in millijoules (NVML, Volta and newer)."""
    import pynvml
    pynvml.nvmlInit()
    try:
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(
            pynvml.nvmlDeviceGetHandleByIndex(i))
            for i in range(pynvml.nvmlDeviceGetCount()))
    finally:
        pynvml.nvmlShutdown()


def metrics():
    """Selected server counters, so prefill work is measured server side too."""
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as r:
        text = r.read().decode()
    keep = ("vllm:prompt_tokens_total", "vllm:request_prefill_time_seconds_sum",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
    out = {}
    for line in text.splitlines():
        for k in keep:
            if line.startswith(k):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def chat(messages, max_tokens=64):
    """Greedy streaming chat request. Measures time to first token and energy."""
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="zp-local")
    m0, e0, t0 = metrics(), gpu_energy_mj(), time.time()
    ttft, parts, usage = None, [], None
    stream = client.chat.completions.create(
        model=SERVED_NAME, messages=messages, max_tokens=max_tokens,
        temperature=0.0, seed=0, stream=True, stream_options={"include_usage": True})
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.time() - t0
            parts.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage.model_dump()
    total = time.time() - t0
    e1, m1 = gpu_energy_mj(), metrics()
    return {"text": "".join(parts), "ttft_s": ttft, "total_s": total,
            "energy_j": (e1 - e0) / 1000.0, "usage": usage,
            "server": {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}}


# ---------------------------------------------------------------- proxy

def start_proxy(tag):
    """Logging proxy between pi and vLLM, one JSONL record per chat request."""
    logpath = f"{OUT}/requests_{tag}.jsonl"
    code = f"import lib; lib.proxy_main({logpath!r})"
    env = {**os.environ, "PYTHONPATH": f"{CODE_DIR}:" + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=True,
                            stdout=open(f"{OUT}/proxy_{tag}.log", "w"),
                            stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=2):
                return proc
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"proxy did not start:\n{tail(f'{OUT}/proxy_{tag}.log')}")


def stop_proxy(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=30)


def proxy_main(logpath):
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=None)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        body = await request.body()
        t0, e0 = time.time(), gpu_energy_mj()
        upstream = await client.send(client.build_request(
            request.method, f"/{path}", content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")}),
            stream=True)

        async def relay():
            first, chunks = None, []
            async for chunk in upstream.aiter_raw():
                if first is None:
                    first = time.time() - t0
                chunks.append(chunk)
                yield chunk
            await upstream.aclose()
            if path.endswith("chat/completions"):
                raw = b"".join(chunks).decode(errors="replace")
                with open(logpath, "a") as f:
                    f.write(json.dumps({
                        "t_start": t0, "ttft_s": first, "total_s": time.time() - t0,
                        "energy_j": (gpu_energy_mj() - e0) / 1000.0,
                        "request": json.loads(body or b"{}"),
                        "output": sse_output(raw)}) + "\n")

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type"))

    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning")


def sse_output(raw):
    """Text and tool calls from a streamed chat response, minus ids and timestamps."""
    text, calls, usage = [], {}, None
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except ValueError:
            continue
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["arguments"] += fn.get("arguments") or ""
    return {"text": "".join(text), "tool_calls": [calls[k] for k in sorted(calls)],
            "usage": usage}
