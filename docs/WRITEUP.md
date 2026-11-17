# Resuming an LLM session without re-reading it

*ZeroPrefill: saving a session's KV cache to disk and restoring it into a fresh
vLLM server. Measured on Kaggle's free 2x T4, November 2026.*

## The problem

A coding agent builds up tens of thousands of tokens of context during a work
session. Close the laptop, come back the next morning, and the provider's
prompt cache (minutes to an hour) is long gone, so the model reads (prefills)
the whole conversation again. That is time, money, GPU power and water spent
recomputing work that was already done.

ZeroPrefill is a vLLM KV connector plugin (vLLM itself is not patched). When a
session goes idle it writes the session's KV cache to disk. When the user
returns, possibly on a different machine, it loads that cache back into GPU
memory and the model continues without re-reading the conversation.

The idea is not new: CachedAttention / AttentionStore, Pensieve, LMCache,
Mooncake and Continuum all store KV outside the GPU, and providers ship
versions of it. What this project adds is measurement on a real setup, with
energy, and the failure modes found along the way.

## Setup

- Qwen2.5-14B-Instruct-AWQ, tensor parallel 2, on Kaggle's free 2x NVIDIA T4
- vLLM 0.30.0, connector loaded through `kv_connector_module_path`
- Cache keyed on chained per-block hashes plus a layout fingerprint (model,
  quantization, KV dtype, block size, TP size, RoPE settings, max context)
- Every number below links to raw outputs in [`results/`](../results/)

## Result 1: one long prompt, 64x faster first token

A 16,031-token prompt is prefilled and saved, the server is killed, and the
same prompt goes to a fresh server.

| | cold | restored from disk |
|---|---|---|
| time to first token | 102.6 s | **1.60 s** |
| GPU energy | 14,349 J | **851 J** |
| tokens restored | 0 | 16,016 of 16,031 |
| output | reference | identical |

Raw: [`results/smoke/`](../results/smoke/).

## Result 2: a real agent session, resumed on another machine

Day 1: the pi coding agent works on psf/requests for 7 tasks and reaches
23,699 tokens of context; the 4.6 GiB cache is saved. Day 2: a new Kaggle
machine restores it and the session continues with a 26,788-token prompt.

| | cold | warm |
|---|---|---|
| tokens restored | 0 | 23,552 (87.9%) |
| time to first token | 260.7 s / 260.7 s | 75.7 s / 73.7 s (3.5x) |
| GPU energy, first reply | 61.8 / 62.1 kJ | 33.1 / 33.3 kJ (46% less) |

This was measured **before** the fix in Result 3: Day 1's last answer was not
saved and had to be recomputed, which is most of the remaining 75 s. Copying
the cache from Kaggle's storage took another 73.7 s, not included above. Raw:
[`results/day1/`](../results/day1/), [`results/day2/`](../results/day2/).

## Result 3: saving the model's own answer

Only tokens the model *read* were being saved. Its answer lives in GPU blocks
that vLLM frees as soon as the request ends, and the next turn's prompt starts
with that answer.

The fix: when a request finishes, the connector tells vLLM to keep its blocks
(`request_finished` returns `True`), writes the answer's KV one engine step
later, and then releases them. Verified on Kaggle: turn 1 answered in 871
tokens; turn 2 (16,936 tokens) restored **16,384** tokens, where a prompt-only
save can restore at most 15,360, and its output matched cold exactly. Raw:
[`results/answer/`](../results/answer/).

LMCache's multi-process connector already has this as an option
(`lmcache.mp.save_decode_cache`, off by default).

## Result 4: compared with LMCache

Same model, GPUs, vLLM version and 16,031-token prompt, LMCache 0.5.6.

**In-process connector: wrong output.** The restore loaded every token without
an error, and the model then answered as if the prompt were garbled. vLLM 0.30
uses a KV layout (LBNHC) that LMCache's in-process path mishandles; this is
LMCache issue #6156, closed without a fix because in-process mode is
deprecated. Its `local_disk` backend restored nothing after a restart (#1175).

**Multi-process mode: correct output.** Head to head, in one Kaggle session,
each tool saved once and restored twice, each restore in fresh processes:

| | restore 1 | restore 2 | GPU energy | output |
|---|---|---|---|---|
| cold prefill | 105.7 s / 100.3 s | | 14,469 / 14,307 J | reference |
| ZeroPrefill | **8.55 s** | **3.04 s** | 1,254 / 940 J | identical |
| LMCache MP | 14.96 s | 8.01 s | 1,775 / 1,392 J | identical |

The bottlenecks are opposite. ZeroPrefill reads its files in 1.0 s / 0.18 s
but spends 6.8 s / 2.15 s copying them into the paged GPU cache. LMCache copies
into the GPU in 0.1 s but spends 12.5 s / 5.5 s fetching its chunk files.

Limits of this comparison: Kaggle does not allow dropping the OS page cache
and about 30 GB of RAM was free, so both tools most likely read from memory
rather than disk. It is two restores per tool in one session, and the first
restore is slower for both. Raw: [`results/h2h/`](../results/h2h/),
[`results/lmcache-mp/`](../results/lmcache-mp/),
[`results/lmcache-inprocess/`](../results/lmcache-inprocess/).

## Bugs found on the way

- **My end-of-session save crashed the engine on its first real run.** vLLM
  0.30's model runner calls the connector's `get_finished` with no forward
  context on steps that have no forward pass, and the connector read the
  layers from the forward context. Unit tests had faked it. The fix reads the
  layers from the config's `static_forward_context` (the same dict), and the
  test now runs without a forward context so it fails on the old code.
- **Forcing a long answer with `min_tokens` broke the experiment.** Unable to
  stop, the model emitted `<tool_call>`, and the server's tool parser
  swallowed the rest of the text. The test now asks a question that needs a
  long answer instead.
- **The cache key was missing RoPE settings and context length.** Keys are
  stored after RoPE is applied, so a cache from a YaRN-scaled server is not
  valid in an unscaled one. Both are in the fingerprint now.
- **LMCache's default 10 s blocking timeout** was shorter than one GPU's read
  of 1.45 GB, and vLLM failed the request with an empty reply. Similar reports
  already exist upstream.

## What it is not

Not production serving: one model, one GPU type, small samples, Kaggle's free
tier. Restores copy all layers before compute starts. A longer follow-up
prompt cannot yet reuse the last partial 1,024-token segment of the previous
save. Code and every raw log: https://github.com/Chebaleomkar/ZeroPrefill

## Next

Speed up the copy into the GPU cache, reuse the partial last segment, re-run
the real agent session with the answer fix, and build a break-even table of
store vs recompute by context length and storage speed.
