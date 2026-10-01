# ZeroPrefill — Engineering Handoff

> **Frozen record of phase 1 (local GTX 1650 work).** Kept for the bug history
> and measurements. The current plan lives in `docs/RESEARCH.md`. The Windows to
> WSL pi scripts mentioned in §5 and §7.1 (`pi_agent_e2e.ps1`,
> `serve_pi_start.sh`, `stop_server.sh`, `check_prompt.py`) were deleted on
> 2026-09-30; the agent experiment moves to Kaggle, where pi runs natively on
> Linux.

**Status:** core mechanism proven (132.91x prefill reduction, verified identical
output). Real-agent integration incomplete.
**Last updated:** 2026-09-30
**Author context:** first working session on this repo, ~2 hours.

This document is written for an engineer with no prior context on the project.
It covers what exists, how it works, what was verified, what failed, and what
to do next. Numbers quoted as "measured" come from runs on the actual machine;
numbers marked "estimate" are arithmetic, not observation.

---

## 1. What this project is trying to achieve

### The problem

Commercial LLM providers evict prompt caches after 5–60 minutes of inactivity.
A developer using a coding agent leaves for the night and returns 8–10 hours
later. The provider has discarded the cached KV blocks, so the entire codebase
context is re-prefilled from scratch on the next request. For a large codebase
this is the dominant cost of the first token, repeated every day.

The goal is a local/private stack that intercepts PagedAttention KV cache
blocks, writes them to local NVMe when a session ends, and restores them into
GPU VRAM on the next session — a "zero-prefill cold start" that survives process
restarts.

### Intended architecture (from the original brief)

1. **Deterministic block-hash registry** — key cache blocks by a chained
   cryptographic hash so sequential token dependencies are preserved:
   `cache_key = hash(model_id, quantization, kv_dtype, block_size, token_ids, parent_block_hash)`
2. **Tensor lifecycle management** — detect session end, dump KV tensors to
   `.bin` on SSD, free host GPU memory
3. **Pre-flight onload engine** — on next launch, match the prefix hash, stream
   blocks back before the first decode step
4. **Cache invalidation driver** — detect workspace file changes, invalidate only
   affected blocks

Plus an OpenAI-compatible routing proxy as the client-facing layer.

### Where the four components stand

| # | Component | State |
|---|---|---|
| 1 | Chained block-hash registry | **Done and unit tested** |
| 2 | Tensor lifecycle (save to disk) | **Done and verified** |
| 3 | Pre-flight onload (restore to VRAM) | **Done and verified** |
| 4 | Invalidation / evaporation | **Not started** — see §7, the spec is wrong |
| — | OpenAI-compatible proxy | **Not started** (vLLM's server *is* one) |

---

## 2. Hardware reality, and why the model changed

The original brief specified Qwen2.5-Coder-3B-Instruct with a 20,000-token
prompt. This machine cannot run that. Measured VRAM: **4 GiB** (GTX 1650,
Turing sm_75).

KV cache size scales with `kv_heads`, and Qwen2.5-0.5B has 2 KV heads against
the 3B model's 16. So the 20k-token acceptance test and the 3B model are
**independent axes**.

Formula used (K and V, fp16):

```
kv_bytes = 2 (K,V) * kv_heads * head_dim * bytes_per_elem * num_layers * tokens
```

| model | layers | kv_heads | head_dim | weights | KV @ 20k | total | fits 4 GiB |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | 36 | 16 | 128 | 5.76 G | 5.49 G | **11.25 G** | no |
| Qwen2.5-1.5B-Instruct | 28 | 2 | 128 | 2.87 G | 0.53 G | 3.40 G | no |
| **Qwen2.5-0.5B-Instruct** | 24 | 2 | 64 | 0.91 G | 0.23 G | **1.14 G** | **yes** |
| Llama-3.2-1B-Instruct | 16 | 8 | 64 | 2.31 G | 0.61 G | 2.92 G | marginal |

**Decision (user-confirmed):** target Qwen2.5-0.5B-Instruct. The connector and
registry code are model-agnostic, so moving to 3B on an 8 GB+ card is a
`--model` flag, not a code change. The full 20k-token acceptance test runs here.

Practical constraint discovered: ~3.08 GiB of the 4 GiB is actually usable, so
`--gpu-memory-utilization 0.70` is the working value. 0.80 fails with
`Free memory on device cuda:0 (3.08/4.0 GiB) ... less than desired`.

---

## 3. Environment: what had to be fixed to get vLLM running

vLLM is **Linux-only**. All inference runs in WSL2 (Ubuntu-24.04, pre-installed).
The Windows Python 3.13 / torch 2.7.1+cu118 install is irrelevant to this work.

**GPU passthrough works** — WSL2 sees the GTX 1650, and an fp16 matmul on device
succeeds. This was verified before committing to the approach, because sm_75 is
old enough that it might not have.

Three environment problems, each of which cost real time:

**a. `sudo` hangs without a TTY.** The first provisioning script called
`sudo apt-get install`. WSL has no password set for the default user, so sudo
blocked on a password prompt and the run sat there for **25 minutes** before I
hit a tool timeout. Fix: run everything as root via
`wsl -d Ubuntu-24.04 -u root -e ...`. `setup_env.sh` now does this and says so.

**b. No pip, no venv, no ensurepip** in the base image. Fixed with
`apt-get install python3.12-venv python3-pip` as root.

**c. `/tmp` is tmpfs in WSL2** and is wiped whenever the distro stops — which is
exactly when a "session ends". Log files vanished between phases during
debugging. All logs and the connector summary now live in `/root/zp-logs/`
(on ext4). Anything that must survive a session boundary cannot go in `/tmp`.

**PowerShell quoting.** Repeatedly corrupted inline `wsl -e bash -c "..."`
commands (nested quotes, `$()` interpolation, `|` treated as a PowerShell
operator). The reliable pattern used throughout: **write a `.sh` file to the
repo, then execute that file in WSL.** Do not inline complex shell in PowerShell.

### Installed and verified

```
vllm 0.30.0        (API differs substantially from older versions — see §6)
torch 2.13.0+cu130
Python 3.12.3      (venv at /root/.venvs/zeroprefill)
GPU  NVIDIA GeForce GTX 1650, 4096 MiB, compute_cap 7.5
```

vLLM logs `Cannot use FA version 2 ... only supported on compute capability >= 8`
on this GPU. **This is benign** — vLLM falls back to `TRITON_ATTN` automatically
and inference works. It is not an error to fix.

---

## 4. Approach and reasoning

### Why fork vLLM's `ExampleConnector` rather than write from scratch

vLLM ships `ExampleConnector` at
`vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`, which
already implements cross-process disk save/load. It is the documented
`KVConnectorBase_V1` contract. Forking it meant the memory-layout handling was
already correct, leaving only the caching *policy* — which is the actual
contribution — to be written.

### Why the registry design is the real contribution

`ExampleConnector` hashes the **entire token sequence** into one folder name.
Change one token at position 4 and all 20,000 tokens miss. That makes it useless
for an agent workflow, where prompts are long and one byte changes.

Chaining a hash per block changes the failure mode from *all-or-nothing* to
*invalidates only the edited block and its descendants*. This is the difference
between a useless cache and a useful one.

### Why storage is segmented, not per-block

The first implementation wrote **one file per block per layer**. For a 20k-token
prompt that is 1,249 blocks x 24 layers = **~30,000 small files**. Restoring it
would cost more I/O than the prefill it replaces — the cache would be a net
loss.

Fixed by grouping 64 consecutive blocks (1,024 tokens) into one file per layer:
20 segments x 24 layers = **480 files**. Same restore result, ~60x fewer
syscalls.

### Why the connector writes to disk itself rather than a separate service

A separate lifecycle service would need the physical block→slot mapping, which
only exists inside the engine process. vLLM's KVConnector V1 API hands the
connector `slot_mapping` in `save_kv_layer` (worker side) — the mapping is only
available there. A save service reading files after the fact would be reading
already-recycled block tables.

---

## 5. What was built

```
zeroprefill/
  hashing.py        chained per-block cache keys, prefix matching      (109 lines)
  connector.py      disk-backed KVConnectorBase_V1                    (503 lines)
  prompt.py         deterministic 20k-token prompt                     (52 lines)
scripts/
  setup_env.sh        WSL2 provisioning (root, no sudo)
  save.py             phase 1: prefill -> persist
  restore.py          phase 2: cold process -> load
  verify.sh           3-stage correctness harness
  verify_correctness.py
  offline_e2e.sh      cold/warm timing comparison
  serve_pi.sh         vLLM server for the pi agent
  serve_pi_start.sh   detached server launch
  stop_server.sh      kill server, wait for GPU/ports to free
  pi_agent_e2e.ps1    two-run agent experiment  (INCOMPLETE — §7)
  check_prompt.py     determinism check
  test_wsl.sh         run pytest inside the WSL venv
tests/
  test_hashing.py   8 tests — chaining, invalidation, dtype isolation
  test_segment.py   4 tests — segmentation (run in WSL where vLLM is present)
```

Early throwaway scripts (`run_baseline.sh`, `serve.sh`, `e2e.sh`,
`client_run.py`, `offline_save.sh`, `diag_lookup.py`) were removed; they were
superseded by `offline_e2e.sh` and `serve_pi.sh`.

### The hashing design (component 1)

`CacheLayout` fingerprints `(model_id, quantization, kv_dtype, block_size)` into
a 16-byte BLAKE2b digest. That fingerprint is the **root parent hash**, so
different models, dtypes, quantizations, or block sizes produce entirely
separate chains — a cache written under one config can never be read as if it
were another.

Block `i` hashes as `BLAKE2b(parent_hash || token_ids[i*block_size:(i+1)*block_size])`,
chained from the root. The final token of a prompt is excluded, because its KV
is not stable until decode advances (this matches vLLM's own convention).

Storage layout on disk:

```
/root/zeroprefill-cache/
  <layout_fingerprint>/
    layers.json                 manifest of layer names
    <segment_hash>/
      layer_<layer_name>.safetensors
```

A segment's hash is the hash of its **last** block, so it commits to both the
content and the position of the whole run.

### Verified properties (unit tested)

```
1249 blocks from 20000 tokens at block_size 16
edit at token 5000  -> blocks 0..311 still hit (25% preserved), 312+ invalidated
append 1 token      -> ZERO invalidation  (this is the real agent-loop case)
different kv_dtype  -> different fingerprint, no collision
hole at block 250   -> prefix match truncates to 250, does not skip the gap
```

The append case is the important one: an agent loop appends to the conversation
every turn, so a cache design that invalidated on append would never hit.

### The connector (components 2 and 3)

Registered with vLLM via `kv_connector_module_path`, so **no vLLM source
patches are needed**. vLLM's factory does `getattr(module, kv_connector_name)`,
so the class name must equal the `kv_connector` value.

Implements the full V1 contract: `get_num_new_matched_tokens`,
`update_state_after_alloc`, `build_connector_meta`, `start_load_kv`,
`wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `request_finished`,
`get_finished`, `register_kv_caches`.

KV is read out of / written into vLLM's paged buffer using the standardized
`[B, H, N, C]` per-layer view via `slot_mapping`:

```python
block_idxs = slot_mapping // block_size
offsets    = slot_mapping % block_size
layer[block_idxs, :, offsets]   # gather
layer[block_idxs, :, offsets] = src   # scatter
```

All device transfers and `.cpu()` calls are wrapped in `gpu_sync_allowed()` from
`vllm.utils.gpu_sync_debug`, as the reference connector does — vLLM raises on
implicit GPU sync in debug mode.

---

## 6. What was verified (measured, not estimated)

### Primary result: 20k-token zero-prefill cold start

Two separate OS processes. Process A prefills and writes KV to disk, then exits
(process kill is a real process death, not a simulated one). Process B starts
with an empty GPU and restores from disk alone.

```
prompt tokens      : 20000
cold prefill       : 110.62 s
restored prefill   : 0.83 s
speedup            : 132.91x
tokens restored    : 19984 / 20000  (99.9%)
cache on disk      : 234.23 MiB  (20 segments x 24 layers)
first token cold   : " v"
first token warm   : " v"
```

### Correctness: restored KV is numerically right

Timing alone proves nothing — a cache can be fast and wrong. Three separate
processes, same prompt, greedy decoding, 24 tokens:

```
stage cold     : 4096-token prompt, empty cache,        ids [14291, 271, 750, ...]
stage populate : same prompt, writes cache,             ids [14291, 271, 750, ...]
stage warm     : fresh process, restores 4080/4096 (99.6%)
RESULT         : IDENTICAL
```

Cold and cache-restored decodes produce **byte-identical token sequences**.

### Test suite

`8 passed, 1 skipped` on Windows (segment tests skip without vLLM);
`12 passed` inside the WSL venv.

---

## 7. What failed

Recorded because the failures are the useful part.

### 7.1 The `pi` agent integration — NOT COMPLETE

This was the explicit request and **it is not done.** No cache-hit measurement
from a real agent session exists.

What works: `pi` (v0.84.3, `@earendil-works/pi-coding-agent`) is a **Windows
Node program and cannot execute inside WSL** — `exec: node: not found`. A
`zeroprefill` provider was registered in `~/.pi/agent/models.json` pointing at
`http://127.0.0.1:8000/v1`, and a plain prompt **does** work end to end through
the local server (returned `PI-OK` in 11.9 s). So the plumbing is sound.

Required fixes along the way:
- `pi` needs an `apiKey` field even for a keyless local server, else
  `No API key found for zeroprefill`
- the vLLM server needs `--enable-auto-tool-choice --tool-call-parser hermes`,
  else `400: "auto" tool choice requires ...`
- vLLM 0.30 deprecates `python -m vllm.entrypoints.openai.api_server` (works, warns)

**The blocker:** in the two-run cold/warm orchestrator, `serve_pi.sh` reports
"launched", exits, and produces neither a log file nor a running process. This
was happening *after* the server ran successfully several times interactively,
so it is specific to the detached launch path. I had just discovered that
`serve_pi.sh` was creating its log dir *after* an `exec >` redirect into it
(which I fixed), and was mid-investigation when the session ended.
**Next agent should start here:** run `serve_pi_start.sh`, then check
`ls /root/zp-logs/` and `pgrep -af api_server`; if still empty, run
`serve_pi.sh` in the foreground to see the failure directly.

Until this works, the 132.91x figure comes from a **synthetic** 20k prompt, not
from real codebase context. That distinction matters when reporting results.

### 7.2 Five connector bugs, all real

Each was diagnosed by reading vLLM 0.30 source, not guessed.

1. **Chunked prefill silently dropped the prompt tail.** `max_num_batched_tokens`
   defaults to 8192, so a 20k prompt runs in 3 chunks. `build_connector_meta`
   only handled `scheduled_new_reqs`, so only the first chunk was saved —
   **8 of 20 segments**, 41% hit rate. Fixed by handling
   `scheduled_cached_reqs` and tracking per-segment coverage.
   *This was masked by a lying counter: the connector reported
   `segments_written: 20` while only 8 existed on disk, because it marked
   segments written at metadata-emit time rather than at actual flush time.*

2. **`new_block_ids` semantics are undocumented.** For a chunked-prefill
   continuation vLLM reports only the *new* blocks; a resumed request reports
   all of them. Guessing wrong produced an empty slot window and silent no-ops.
   Fixed by accepting either convention and verifying actual coverage
   (`_cover()`), so the connector is correct either way.

3. **Scheduler-side and worker-side connectors are different objects.**
   `register_kv_caches` populates the *worker* instance, but
   `get_num_new_matched_tokens` runs on the *scheduler* instance. The layer-name
   list was therefore always empty on the read path, and completeness checks
   always failed. Fixed by publishing `layers.json` to disk and reading it back.
   **Any per-request in-memory state shared across these two roles is a bug.**

4. **Structural mismatch in the lookup path.** `matched_prefix_blocks` tested
   `os.path.isdir()` per *block* hash, but disk is keyed by *segment* hash.
   Returned 0 matches while 8 valid segments sat on disk. Fixed with an
   `is_present` predicate and a block→segment map.

5. **Connector logging went nowhere.** vLLM configures log handlers only under
   the `vllm` namespace, so a connector in an external package emitted nothing.
   Zero `ZP_` lines appeared in logs despite the code running. Fixed with an
   explicit `logging.StreamHandler(sys.stderr)`.

### 7.3 A design correction: component 4 as specified cannot work

The original brief says: on file change, "invalidate only the affected
downstream suffix block hashes while retaining the unchanged system prompt and
early codebase structural blocks."

**With a single global hash chain over a concatenated codebase prompt, this is
not achievable.** If editing one file changes its length, every subsequent token
shifts position, so every subsequent block hash changes. The "unchanged early
blocks" are only preserved if the edit does not change byte offsets.

The correct design is **per-file independent chains** plus a Merkle index over
per-file content hashes, so an edit to `z.py` invalidates only `z.py`'s chunks
and leaves every other file's blocks reusable. Not started. **Do not build the
naive version.**

### 7.4 Smaller corrections

- `align_to_block_size` exists only inside `example_connector.py` in 0.30, not in
  a shared module. It is reimplemented locally with a comment.
- `LLM.generate()` in 0.30 takes `[{"prompt_token_ids": ids}]`, not
  `prompt_token_ids=[ids]`.
- vLLM 0.30's `KVTransferConfig` has **no** `kv_connector_class_name` field
  (a later version added it). The class name must match `kv_connector`.
- `KVTransferConfig` is a pydantic dataclass, so `model_fields` does not exist;
  use `dataclasses.fields`.
- Putting the venv on `/mnt/e` is a 9p mount and slow for the many small
  safetensors reads the restore path does. Use `/root/.venvs`.

### 7.5 Not attempted, deliberately

- **Multi-turn sessions.** Only single-turn save/restore is tested. The real
  agent scenario — cache surviving an append plus a tool-result round trip — is
  untested, though the unit tests confirm appends invalidate nothing.
- **OpenAI-compatible proxy.** Not built. vLLM's own server is already
  OpenAI-compatible, which is what the pi test uses.
- **Async PCIe DMA.** Restore currently uses blocking H2D copies wrapped in
  `gpu_sync_allowed()`. Pipelining per layer is available in the V1 API
  (`wait_for_layer_load`) but not implemented.

---

## 8. Where we are, and where we're going

### Honest position

The **core mechanism is proven**: chained hashing plus disk persistence turns a
110-second cold prefill into 0.83 seconds with bit-identical output, across a
real process restart, on a 4 GiB consumer GPU. That is the hard part and it
works.

What is **not** proven is the product claim. The motivating scenario is an agent
session resuming tomorrow morning on a real codebase, and that has not been
measured. Three things block it: the pi orchestrator (§7.1), multi-turn
behaviour, and the invalidation design.

### Suggested order of work

1. **Fix the pi orchestrator.** Highest value: converts a synthetic result into
   a product result. Start with the diagnosis in §7.1.
2. **Measure a realistic prompt.** A 20k synthetic blob is a best case. A real
   repo read by an agent, including tool-call round trips, is the honest number.
   Expect the hit rate to be lower and the restore slower.
3. **Multi-turn test.** Two sequential pi turns sharing a prefix. This is the
   append case, and it is the actual product loop.
4. **Redesign component 4** as per-file chains + Merkle index (§7.3). Decide
   this before writing any invalidation code.
5. **Segment size tuning.** 64 blocks is a guess. Sweep it against real prompts
   and measure restore I/O.
6. **Only then** consider the routing proxy, DMA pipelining, and multi-GPU.

### Scaling to the original target

Moving to Qwen2.5-Coder-3B needs an 8 GB+ card (11.25 GiB total). No code
change expected — `CacheLayout` already keys on model id, dtype, and
quantization, so caches from different models are already isolated. Worth
re-running `offline_e2e.sh` with `--model` pointed at 3B on suitable hardware
to confirm the speedup holds at larger KV sizes.

---

## 9. How to run it

All commands are run from PowerShell. Anything nontrivial goes in a `.sh` file
in the repo, then executed — do not inline complex shell.

```powershell
# one-time setup (root, no sudo)
wsl -d Ubuntu-24.04 -u root -e bash /mnt/e/inference-engineering/projects/ZeroPrefill/scripts/setup_env.sh

# unit tests (segment tests need the WSL venv)
wsl -d Ubuntu-24.04 -u root -e bash /mnt/e/inference-engineering/projects/ZeroPrefill/scripts/test_wsl.sh

# correctness: 3 processes, cold vs restored must be IDENTICAL
wsl -d Ubuntu-24.04 -u root -e bash /mnt/e/inference-engineering/projects/ZeroPrefill/scripts/verify.sh

# headline timing: 20k prompt, save then restore across a process kill
wsl -d Ubuntu-24.04 -u root -e bash /mnt/e/inference-engineering/projects/ZeroPrefill/scripts/offline_e2e.sh

# pi agent experiment (CURRENTLY BROKEN — see §7.1)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pi_agent_e2e.ps1
```

Useful knobs: `ZP_TOKENS` (default 20000), `ZP_GPU_UTIL` (default 0.70, must be
<= 0.75 on this card), `ZP_CACHE` (default `/root/zeroprefill-cache`).

### Reading the logs

Connector signals go to **stderr** and to `/root/zp-logs/summary.jsonl`:

```
ZP_LOOKUP req=<id> tokens=<n>     lookup attempted
ZP_HIT tokens=19984 of 20000     tokens restored from disk
ZP_MISS no prefix matched        nothing usable on disk
```

`ZP_HIT` is the signal that the cache was actually used. Its absence means the
restore path did not fire, regardless of timing.

### Local machine changes made

`~/.pi/agent/models.json` has a `zeroprefill` provider added; the original is
backed up at `~/.pi/agent/models.json.zeroprefill.bak`. Remove or keep as
appropriate.

---

## 10. Recommendations for the next agent

- **Trust the log signals, not the counters.** The `segments_written: 20` lie in
  §7.2 is the cautionary example: the connector's own in-memory accounting
  disagreed with the filesystem. When in doubt, `find` the cache directory.
- **Read vLLM 0.30 source before changing connector code.** The API differs
  meaningfully from published examples, and `example_connector.py` in
  site-packages is the authoritative reference — not the online docs.
- **Remember the two-instance split.** Scheduler-side and worker-side connector
  objects are separate. Cross-role state must go to disk. This caused a silent
  total failure.
- **The 132.91x number is a synthetic best case.** Quote it with that caveat.
  A real agent workload will be worse, and reporting it unqualified would be
  misleading.
- **Do not build component 4 as originally specified** (§7.3).
- Keep `HF_HOME`, the cache root, and the venv on the WSL ext4 filesystem, not
  `/mnt/e` (9p) and not `/tmp` (tmpfs, wiped on shutdown).
