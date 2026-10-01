"""Deterministic 20k-token prompt, identical across processes.

The cache key is derived from token ids, so both the save and the restore
process must produce byte-identical tokens. Generating from a seeded RNG
guarantees that without depending on a checked-in fixture file.
"""

from __future__ import annotations

import random

WORDS = [
    "cache", "token", "block", "hash", "prefill", "decode", "tensor", "kernel",
    "paged", "attention", "vram", "offload", "manifest", "evict", "prefix",
    "registry", "worker", "scheduler", "bucket", "resident", "swap", "page",
]


def synthetic_corpus(target_tokens: int, model: str, seed: int = 0) -> str:
    """Build a prompt of at least target_tokens tokens, deterministically."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rng = random.Random(seed)

    chunks: list[str] = []
    total = 0
    i = 0
    while total < target_tokens:
        file_idx = i // 40
        if i % 40 == 0:
            chunks.append(f"\n# ==== src/module_{file_idx:04d}.py ====\n")
        body = " ".join(rng.choice(WORDS) for _ in range(24))
        chunks.append(
            f"def fn_{i:05d}(a, b):\n"
            f'    """{rng.choice(WORDS)} {body}"""\n'
            f"    return a + b  # {body}\n\n"
        )
        i += 1
        if i % 40 == 0:
            total = len(tok("".join(chunks)).input_ids)

    ids = tok("".join(chunks)).input_ids[:target_tokens]
    return tok.decode(ids)


def token_ids_for(target_tokens: int, model: str, seed: int = 0) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(synthetic_corpus(target_tokens, model, seed)).input_ids
    return ids[:target_tokens]
