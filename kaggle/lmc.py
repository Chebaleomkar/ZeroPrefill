"""LMCache multi-process (MP) mode helpers shared by lmcache-mp.py and h2h.py."""

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import lib

LMCACHE_SPEC = "lmcache==0.5.6"  # wheel built against torch 2.13.0 ...
TORCH_SPEC = "torch==2.13.0"     # ... which vllm 0.30.0 pins
LMC_PORT = 5555
L2_DIR = "/root/lmc-l2"
LMC_ENV = {"PYTHONHASHSEED": "0"}
CONNECTOR = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
             "kv_connector_extra_config": {"lmcache.mp.host": "tcp://127.0.0.1",
                                           "lmcache.mp.port": LMC_PORT}}
LOG_KEYS = ("hit", "retriev", "prefetch", "lookup", "l2", "error", "traceback")


def install_lmcache():
    """Install LMCache without letting pip move vllm or torch off the tested pins."""
    lib.sh(f'{sys.executable} -m pip install -q {LMCACHE_SPEC} "{lib.VLLM_SPEC}" "{TORCH_SPEC}"')
    lib.sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E '^(vllm|torch|lmcache|transformers) '")
    lib.sh(f"{sys.executable} -c 'import lmcache.integration.vllm.lmcache_mp_connector'")


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_lmcache(tag):
    """Start the LMCache MP server; ready once its request port accepts."""
    exe = shutil.which("lmcache")
    cmd = ([exe, "server"] if exe
           else [sys.executable, "-m", "lmcache.v1.multiprocess.server"])
    cmd += ["--host", "127.0.0.1", "--port", str(LMC_PORT), "--chunk-size", "256",
            "--l1-size-gb", "8", "--eviction-policy", "LRU",
            "--l2-adapter", json.dumps({"type": "fs", "base_path": L2_DIR})]
    logpath = f"{lib.OUT}/lmc_{tag}.log"
    lib.log(f"starting LMCache server [{tag}]: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=open(logpath, "w"), stderr=subprocess.STDOUT,
                            env={**os.environ, **LMC_ENV}, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError(f"LMCache server [{tag}] exited:\n{lib.tail(logpath)}")
        if _port_open(LMC_PORT):
            time.sleep(5)  # port binds before the request handlers are registered
            lib.log(f"LMCache server [{tag}] ready after {time.time() - t0:.0f}s")
            return proc
        time.sleep(2)
    stop_lmcache(proc)
    raise TimeoutError(f"LMCache server [{tag}] not ready:\n{lib.tail(logpath)}")


def stop_lmcache(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    lib.log("LMCache server stopped")


def wait_for_l2(settle_s=15, max_s=300):
    """L1 -> L2 stores are asynchronous: wait until the file count stops growing."""
    last, stable_since, t0 = -1, time.time(), time.time()
    while time.time() - t0 < max_s:
        n = len(glob.glob(f"{L2_DIR}/**/*.data", recursive=True))
        if n != last:
            last, stable_since = n, time.time()
        elif time.time() - stable_since >= settle_s:
            break
        time.sleep(1)
    lib.log(f"{L2_DIR}: {last} .data files after {time.time() - t0:.0f}s")
    return last


def lmc_lines(*paths):
    out = []
    for path in paths:
        for line in lib.grep(path, ""):
            low = line.lower()
            if "lmcache" in low and any(k in low for k in LOG_KEYS):
                out.append(line[-300:])
    return out[-40:]


def vllm_server(tag):
    return lib.start_server(tag, kv_transfer=CONNECTOR, extra_env=LMC_ENV)
