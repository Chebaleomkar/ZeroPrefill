# ZeroPrefill

Save an LLM session's KV cache to disk when it goes idle, and load it straight
back into GPU memory when the user returns, so the model does not have to
re-read (prefill) the whole conversation.

**The motivation.** A developer works with a coding agent all day, closes the
laptop at night and comes back 8 to 10 hours later. Providers keep prompt caches
for minutes to an hour, so the entire conversation is prefilled again from
scratch. That costs the user money and costs GPU time, power and water for work
that was already done the day before.

## Results so far

All on Kaggle's free 2x NVIDIA T4, Qwen2.5-14B-Instruct-AWQ, tensor parallel 2,
vLLM 0.30.0. Raw outputs are in [`results/`](results/).

**Smoke test: one 16,031-token prompt, save, kill the server, restore in a fresh one**

| | cold (normal prefill) | warm (KV restored from disk) |
|---|---|---|
| time to first token | 102.6 s | **1.60 s** (64x) |
| GPU energy for the request | 14,349 J | **851 J** (17x less) |
| tokens restored | 0 | 16,016 of 16,031 (99.9%) |
| output | reference | **identical** |

**Real agent session resumed on a different machine (Day 1 / Day 2)**

Day 1: the [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
coding agent works on [psf/requests](https://github.com/psf/requests) v2.32.3
for 7 tasks, reaching 23,699 tokens of context. Its KV cache (4.6 GiB) is saved
as the Kaggle run's output. Day 2: a new Kaggle machine attaches that output,
forks the same pi session twice and sends the same next message, once with the
cache restored (warm) and once without (cold). Each arm ran twice.

| | cold | warm |
|---|---|---|
| resumed prompt | 26,788 tokens | 26,788 tokens |
| tokens restored | 0 | 23,552 (87.9%) |
| time to first token | 260.7 s / 260.7 s | **75.7 s / 73.7 s** (3.5x) |
| GPU energy, first reply | 61.8 / 62.1 kJ | **33.1 / 33.3 kJ** (46% less) |

Read these with the caveats:

- **Why 3.5x and not 64x.** Only tokens the model *read* are saved, not the
  ones it *wrote*. Day 1's last answer (~3k tokens) was recomputed on Day 2,
  and that is most of the remaining 75 s. Saving at session end is the next fix.
- **Fetch cost.** Copying the 4.6 GiB cache from Kaggle's storage to local disk
  took 73.7 s, which is not included in the warm time.
- **Output match.** Each arm reproduces itself exactly. Cold and warm agree for
  the first ~1,200 tokens of the reply, then split at a near-tie word. Day 1
  built the KV over 8 turns while cold builds it in one pass, so floating-point
  rounding differs. Corrupt KV would diverge from the first token. The smoke
  test, where both sides compute the KV the same way, matched exactly.
- **Weak agent behaviour.** Qwen2.5-14B-AWQ only used tools on the first task
  and answered the rest without reading files. The session is a real 27k-token
  multi-turn conversation, but not yet a realistic codebase read.

Full log with dates and reasoning: [`docs/RESEARCH.md`](docs/RESEARCH.md) §6.

## How it works

A vLLM `KVConnectorBase_V1` plugin, loaded with `kv_connector_module_path`, so
vLLM itself is not patched.

1. **Naming the memory.** vLLM stores KV in 16-token blocks. Each block gets a
   hash chained to the previous block, rooted in a fingerprint of model,
   quantization, KV dtype, block size and TP size
   ([`zeroprefill/hashing.py`](zeroprefill/hashing.py)). Appending to a
   conversation invalidates nothing; editing token N invalidates only what
   comes after it.
2. **Saving.** While vLLM prefills, the connector copies KV out of GPU memory
   in segments of 64 blocks (1,024 tokens), one safetensors file per layer per
   GPU, named by the segment's hash. Segments already on disk are skipped, so
   an agent loop does not rewrite its history every turn. Writes are atomic.
3. **Restoring.** On a new request vLLM asks the connector how much of the
   prompt it already has. The connector hashes the prompt, finds the longest
   run of complete segments on disk (all layers, all GPUs) and reports it
   (`ZP_HIT`). vLLM reserves empty blocks; the connector copies the saved
   tensors straight into them. vLLM only computes the rest.

On-disk layout:

```
<cache_root>/<layout_fingerprint>/<segment_hash>/layer_<name>.rank<r>.safetensors
```

## How we ran it

- **Kaggle, only through the CLI.** `kaggle/push.py` bundles the connector and
  a run script into one private script kernel and pushes it to the 2x T4
  machine. Outputs come back with `kaggle kernels output`.
- **pi headless.** pi talks to a small logging proxy in front of vLLM, which
  records every request's time to first token, GPU energy (NVML counters) and
  exact output.
- **Fair comparison.** Day 2 forks one saved pi session (`pi --fork`) for both
  arms, so both start from exactly the same tokens. Each arm gets a freshly
  started server and the repo reset to how pi left it on Day 1.
- **Things that would silently break it, and the fixes:** pi's auto-compaction
  (rewrites history at ~16k tokens with a 32k window) is disabled; pi's system
  prompt has no date and only depends on cwd and pi version, which are pinned;
  Day 2 uses exactly Day 1's context length, because RoPE scaling changes the
  cached K values.

## Reproduce

Needs a Kaggle account with phone verification (GPU + internet) and the Kaggle
CLI logged in (`pip install kaggle`, `python -m kaggle auth login`).

```
python kaggle/push.py smoke                                 # ~25 min
python kaggle/push.py day1                                  # ~40 min
python kaggle/push.py day2 <kaggle-user>/zeroprefill-day1   # ~60 min
python -m kaggle kernels status <kaggle-user>/zeroprefill-<run>
python -m kaggle kernels output <kaggle-user>/zeroprefill-<run> -p runs/<run>
```

Add `--file-pattern '^(results/|pi-sessions/)'` to `kernels output` to skip
downloading the multi-GB KV cache.

Unit tests need vLLM, so run them on Linux (phase 1 used WSL2):

```
python -m pytest tests/ -q     # 19 tests
```

## Repository layout

```
zeroprefill/          the connector
  hashing.py          chained per-block cache keys, layout fingerprint
  connector.py        save / lookup / restore for vLLM 0.30
  prompt.py           deterministic synthetic prompt (phase 1)
kaggle/               the 2x T4 experiments
  lib.py              install vLLM + pi, start/stop server, logging proxy, energy
  smoke.py            run 0: save, restore, compare with cold
  day1.py             build a real pi session, save its KV cache
  day2.py             resume on a fresh machine, warm vs cold
  push.py             bundle + push as a private Kaggle kernel
results/              raw outputs of every run (no KV caches)
  smoke/  day1/  day2/
tests/                hashing, segments, store path (unaligned prompts, chunking)
scripts/              phase 1: local GTX 1650 runs under WSL2
docs/
  RESEARCH.md         plan, metrics, use cases, limits, dated status log
  phase1-handoff.md   phase 1 record: local runs, the five connector bugs
```

## Known issues

- Last answer of a session is not cached (see results).
- The fingerprint does not yet include RoPE scaling or max context length.
  Never restore a cache under a different context-length setting.
- Restore copies all layers before compute starts; no per-layer pipelining.
- Only tested on vLLM 0.30.0. The connector API changes between versions.
- Per-file invalidation (drop only the edited file's part of the cache) is not
  built. Plain resume does not need it.

## Next steps

1. Save at session end, including the last answer.
2. Add RoPE / max length to the fingerprint.
3. Day 1 v2 with Qwen3-14B-AWQ (better tool calling, native 40k) and prompts
   that require tool use.
4. More sessions and a second model size; LMCache as a baseline; a break-even
   table of store vs recompute by model, context and storage speed.
5. Warm-up on open: start loading the cache when the session opens, to hide
   the fetch time.

## Related work

This mechanism is not new. Closest prior work: CachedAttention / AttentionStore
(USENIX ATC 2024), Pensieve, LMCache (2025), Mooncake, Continuum (2025, KV
time-to-live for agents), and on the energy side GreenCache and "Store or
Recompute?". Providers ship versions of it too: DeepSeek's disk cache,
Anthropic's 1-hour cache, OpenAI's extended retention. Our contribution so far
is measurement: real agent sessions resumed across machines, with energy, and
the practical failure modes found along the way.
