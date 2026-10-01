# ZeroPrefill Research Plan

Living document. Update it when a decision changes or a result comes in.
Phase 1 history (local GPU work, bugs found, first numbers) is in
`docs/phase1-handoff.md`.

---

## 1. The idea in plain words

When you chat with a coding agent, the model has to "read" the whole
conversation before it can answer. That reading step is called **prefill**. The
result of that reading is stored on the GPU as the **KV cache**, which is the
model's working memory of everything said so far.

Providers keep that memory for a short time (5 minutes to 1 hour). A developer
stops at night and comes back 8 to 10 hours later. By then the memory is gone,
so the model has to read the entire conversation again from scratch. The user
pays for it, and the GPU burns power and water doing work it already did
yesterday.

**ZeroPrefill saves that memory to disk when a session goes quiet, and puts it
back on the GPU when the user returns.** The model skips the re-reading and
goes straight to answering.

### Why this is worth doing when LMCache and Mooncake exist

Those systems already move KV cache off the GPU, but they are internal plumbing
for the provider. The user cannot see it, control it, or rely on it:

- no guarantee your session survives the night
- no way to say "keep this session for 3 days"
- no report of how much compute was saved on your behalf

Our angle is **user-facing, transparent, long-lived KV persistence**, backed by
honest benchmarks on real agent sessions.

---

## 2. Main experiment: resume a real agent session

### Setup

| item | choice | why |
|---|---|---|
| hardware | Kaggle, 2x T4 (2 x 15 GB), driven only through the Kaggle CLI | free, pi runs natively on Linux |
| model | Qwen2.5-14B-Instruct-AWQ, tensor parallel 2 (fallback: 7B-AWQ) | strong enough to drive pi; ~10 GB weights + ~7.5 GB KV at 40k fits in 30 GB |
| server | vLLM + our connector, `--tool-call-parser hermes` | proven in phase 1 |
| agent | `pi` in headless mode (`pi -p`) | real coding agent, saves sessions to a file |
| target | session grows to 30k to 40k tokens | realistic "one working day" of context |

### Steps

```
Day 1 (Kaggle run 1)
  1. Start vLLM + ZeroPrefill connector.
  2. Run pi headless on a real coding task until context reaches ~30-40k tokens.
  3. Save: pi's session file + the KV cache on disk.
  4. Shut everything down. Download both as the run's output.

Day 2 (Kaggle run 2, fresh machine, nothing in GPU memory)
  5. From the SAME saved session, make two identical copies (pi --fork).
  6. Copy A (warm): restore the KV cache from disk, then send the next message.
  7. Copy B (cold): no KV cache, send the same next message.
  8. Compare A vs B. Repeat twice, alternating cold/warm.
```

### How to run it

```
python kaggle/push.py smoke                               # run 0: connector at TP=2
python kaggle/push.py day1                                # build + save the session
python kaggle/push.py day2 <kaggle-user>/zeroprefill-day1  # resume warm vs cold
python -m kaggle kernels output <kaggle-user>/zeroprefill-<run> -p runs/<run>
```

| file | role |
|---|---|
| `kaggle/lib.py` | install vLLM 0.30.0 + pi 0.84.3, start/stop server, logging proxy, NVML energy |
| `kaggle/smoke.py` | 16k-token prompt: save, restore in a fresh server, compare with cold |
| `kaggle/day1.py` | pi on psf/requests v2.32.3 until ~36k tokens; saves cache, session, repo |
| `kaggle/day2.py` | copies cache from Day 1 output, forks the session, cold vs warm |
| `kaggle/push.py` | bundles the connector into one script and pushes a private kernel |

pi talks to a small proxy in front of vLLM. The proxy logs every request's
time to first token, GPU energy and exact output, which is how A and B are
compared request by request.

Things that would silently break the comparison, and how they are handled:

- **pi compaction.** With a 32k window pi summarizes history at ~16k tokens,
  which changes the prefix. Disabled in `~/.pi/agent/settings.json`.
- **Changing system prompt.** pi's prompt has no date; it contains the cwd
  and pi's install path, so the repo path and pi version are fixed.
- **Context above 32k.** Qwen2.5 is natively 32k; day runs use YaRN (x1.5,
  49k) and fall back to 32k if vLLM rejects it.

### Why fork one session instead of running the task twice

If we run the agent twice from scratch, the two runs will drift apart: a
different tool call here, a slightly different answer there. Then the two
contexts are not the same, and the comparison is unfair. Forking one saved
session guarantees both copies start from **exactly the same tokens**, so the
only difference between A and B is whether the KV cache was restored.

### What we measure

| metric | how | what it tells us |
|---|---|---|
| time to first token | logging proxy, first streamed chunk | what the user feels |
| prefill tokens computed | vLLM metrics + `ZP_HIT` log line | work actually skipped |
| GPU energy (joules) | NVML total-energy counters of both GPUs, before/after | power and water saved |
| fetch time | copy of the cache from Day 1 output to local disk | cost of the storage side |
| output match | compare A and B token ids (greedy decoding) | restored memory is correct |
| cache hit rate | `ZP_HIT tokens=x of y` | how much of the session was reusable |

Rule from phase 1: **trust `ZP_HIT` and the files on disk, not timers alone.**

### Success criteria

- A and B produce identical first responses under greedy decoding.
- A restores more than 95% of the prefix tokens.
- A uses measurably less GPU energy and time before the first token than B.
- Everything is reproducible from one script and one Kaggle notebook.

---

## 3. Other use cases to try later (plain language)

Same mechanism each time: "save the model's working memory, reuse it later."
Ordered roughly by value and ease.

1. **Overnight resume** (main experiment above). Close the laptop, come back
   tomorrow, continue with no re-reading.

2. **Branching ("try three ideas").** From one saved point, start three
   different attempts at a fix. All three reuse the same memory instead of each
   re-reading the whole conversation. Directly supported by `pi --fork`.

3. **Undo / rewind.** Jump back to any earlier turn of the conversation and
   continue from there without recomputing. Like git checkpoints for a chat.

4. **Crash and move recovery.** The GPU machine dies or the session moves to
   another machine. The memory travels with it as files, so the new machine
   continues instantly. Our Day 1 / Day 2 Kaggle setup already tests this,
   since run 2 is a different machine.

5. **Shared team starting point.** Ten developers on the same repo all start
   with the same system prompt, tool list, and core files. Read it once, reuse
   it for everyone. Needs care: sharing across users can leak information
   through timing, so this must be opt-in per team.

6. **Edited files (smart invalidation).** The agent re-reads a file you changed
   overnight. Only that file's part of the memory should be thrown away, not
   everything after it. Needs per-file hash chains (phase 1 handoff §7.3).
   Note: plain resume (use case 1) does not need this, because yesterday's
   conversation text itself never changes.

7. **User dashboard.** Show the user "this session saved 38,000 tokens of
   prefill, 1.2 GPU-minutes, about X Wh". Let them pin, extend, export, or
   delete their cache. This is the transparency piece nobody offers today.

8. **Break-even calculator.** For a given model size, context length, GPU, and
   storage speed: is saving and restoring cheaper than recomputing? This keeps
   the claims honest (see §4).

---

## 4. Honest limits (keep these in every write-up)

- **Fast GPUs shrink the time win.** On an H100, prefilling 30k tokens takes
  about a second or two, and reading a large cache from disk can take as long.
  The 132.91x from phase 1 was on a slow 4 GB GPU with a synthetic prompt. The
  energy and compute saving is the more durable claim.
- **The cache is big and model-specific.** Roughly 56 KB per token for
  Qwen2.5-7B, and around 320 KB per token for a 70B-class model. It only works
  with the exact same model, weights, and settings. Storage lives on the
  provider side; shipping it to the user's laptop is slower than recomputing.
- **Privacy.** A KV cache is derived from the user's code and chat. It must be
  encrypted at rest, deletable on request, and never shared across users by
  default.

---

## 5. Open decisions

| question | options | leaning |
|---|---|---|
| context above 32k | Qwen2.5 and Qwen3 are natively 32k; 40k needs YaRN rope scaling | start at ~30k natively, add YaRN only if needed |
| two T4s | tensor parallel 2 means each GPU holds half the KV heads | **decided:** connector writes one file per GPU rank; verify identical output |
| vLLM version on Kaggle | phase 1 used vLLM 0.30 (CUDA 13 torch); Kaggle's driver may be older | probe the Kaggle driver first, pin a matching vLLM |
| coding task for pi | needs enough file reading to reach 30-40k tokens | pick a mid-size open-source repo and a multi-step task |

---

## 6. Status log

| date | what happened |
|---|---|
| 2026-09-30 | Phase 1 done locally: 20k synthetic prompt, 110.62 s cold to 0.83 s restored, identical output. |
| 2026-09-30 | Plan moved to Kaggle + real pi sessions. Repo cleaned up; this doc created. |
| 2026-09-30 | Decided: vLLM, Qwen2.5-14B-Instruct-AWQ on 2x T4 (TP=2), Kaggle CLI only. Dropped the multi-day use case: once offloaded, elapsed time does not matter. |
| 2026-09-30 | Connector made TP-aware (per-rank files, `tp_size` in fingerprint, all-rank completeness, atomic writes) and skips segments already on disk. |
| 2026-09-30 | Fixed store-path bug hidden in phase 1: prompts whose length is not a multiple of 16 were saved from the wrong slots or not at all. Phase 1 prompts (20000, 4096) were both aligned. Regression tests in `tests/test_store_windows.py`. |
| 2026-09-30 | Kaggle harness written; smoke run pushed. |
| 2026-09-30 | **Smoke run passed** (Qwen2.5-14B-AWQ, 2x T4, TP=2, 16,031-token prompt). Cold time to first token 102.6 s, restored 1.60 s (64x). GPU energy per request 14,349 J cold vs 851 J restored (17x less). `ZP_HIT` 16,016 of 16,031 tokens. Restored output identical to cold; cold identical across two runs. Saving adds ~4.6 s to the first prefill. Cache 3.0 GiB (192 KB/token, 1,536 files). pi reply and bash tool call both work. |
| 2026-09-30 | **Day 1 (v1) done.** pi on psf/requests v2.32.3, 7 tasks, 8 LLM requests, final context 23,699 tokens, cache 4.63 GiB (24 segments), no connector warnings. **Weak agent behaviour:** only task 1 used tools; Qwen2.5-14B-AWQ answered tasks 2-7 without reading files, and task 3 hit the 4,096-token limit in a repetition loop. The context is a real multi-turn conversation but not a realistic codebase read. YaRN (49k) failed to start; ran at 32k. |
| 2026-09-30 | Found: the cache fingerprint does not include RoPE scaling or max length, which change the cached K values. Day 2 is pinned to Day 1's 32k config; the fingerprint must be extended before varying it. |
| 2026-09-30 | Known gap: only prompt tokens are saved, so the last answer of a session (up to ~3k tokens here) is not cached and must be recomputed on resume. |
| 2026-09-30 | **Day 2 (v1) done**, fresh Kaggle machine, Day 1 output attached. Resumed prompt 26,788 tokens. `ZP_HIT` 23,552 (87.9%); the 3,236 uncached tokens are Day 1's last answer plus the new message. Time to first token: cold 260.7 s / 260.7 s, warm 75.7 s / 73.7 s (**3.5x**). First-request GPU energy (incl. ~1.8k decoded tokens): cold 61.8 / 62.1 kJ, warm 33.1 / 33.3 kJ (**46% less**). Copying the 4.6 GiB cache from Kaggle's input mount to local disk took 73.7 s (~64 MB/s), not included in warm time. |
| 2026-09-30 | Output check: each arm reproduces itself exactly (cold0 == cold1, warm0 == warm1). Cold and warm agree for the first 64% of the reply (~1,200 tokens), then split at a near-tie word. Corrupt KV would diverge from the first token; this is floating-point drift because Day 1 computed the KV incrementally over 8 turns while cold computes it in one pass (the same effect exists in vLLM's own prefix cache). Smoke run, where both sides compute the KV the same way, matched exactly. |
| 2026-09-30 | Takeaway: most of the remaining warm time is recomputing Day 1's uncached last answer (3,236 tokens at deep context). Saving the session's final tokens at session end should bring resume time close to the smoke result. Next: Day 1 v2 with Qwen3-14B-AWQ, explicit tool use, session-end save, RoPE/max-len in the fingerprint. |
