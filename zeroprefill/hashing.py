"""Chained per-block cache keys.

vLLM's stock ExampleConnector keys a cache folder on a hash of the *entire*
token sequence. Change one token at position 4 and all 20,000 tokens miss.
Here each block gets its own key, chained to its parent, so an edit only
invalidates that block and everything after it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

PERSON = b"zeroprefill-blk"


@dataclass(frozen=True)
class CacheLayout:
    """Everything that changes the bytes of a KV block.

    Two caches are only interchangeable when every field here matches.
    """

    model_id: str
    quantization: str
    kv_dtype: str
    block_size: int
    # Each tensor-parallel rank holds a different slice of the KV heads, so a
    # cache written at one TP size cannot be read back at another.
    tp_size: int = 1

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "quantization": self.quantization,
                "kv_dtype": self.kv_dtype,
                "block_size": self.block_size,
                "tp_size": self.tp_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.blake2b(payload, digest_size=16, person=PERSON).hexdigest()

    @property
    def root(self) -> str:
        """Parent hash for block 0. Severs the chain across models/dtypes."""
        return self.fingerprint


def _block_digest(parent: str, token_ids: list[int]) -> str:
    h = hashlib.blake2b(digest_size=16, person=PERSON)
    h.update(parent.encode())
    h.update(b"|")
    h.update(",".join(str(t) for t in token_ids).encode())
    return h.hexdigest()


def chain_blocks(
    token_ids: list[int],
    layout: CacheLayout,
    drop_last: bool = True,
) -> list[tuple[str, int, int]]:
    """Return (block_hash, token_start, token_end) per full block.

    The final token of a prompt has no stable KV until decode advances, so it
    is excluded by default. That matches vLLM's ExampleConnector behaviour.
    """
    usable = len(token_ids) - 1 if drop_last else len(token_ids)
    n_full = usable // layout.block_size

    out: list[tuple[str, int, int]] = []
    parent = layout.root
    for i in range(n_full):
        start = i * layout.block_size
        end = start + layout.block_size
        parent = _block_digest(parent, token_ids[start:end])
        out.append((parent, start, end))
    return out


def block_dir(cache_root: str, layout: CacheLayout, block_hash: str) -> str:
    return os.path.join(cache_root, layout.fingerprint, block_hash)


def has_block(cache_root: str, layout: CacheLayout, block_hash: str) -> bool:
    return os.path.isdir(block_dir(cache_root, layout, block_hash))


def matched_prefix_blocks(
    token_ids: list[int],
    layout: CacheLayout,
    cache_root: str,
    is_present=None,
) -> list[tuple[str, int, int]]:
    """Longest run of blocks present on disk, starting at block 0.

    `is_present` decides what counts as a hit. Storage is segmented, so the
    connector passes a segment-aware predicate rather than testing each block
    directory: a block only exists on disk once its enclosing segment is written.
    """
    check = is_present or (lambda h: has_block(cache_root, layout, h))
    chain = chain_blocks(token_ids, layout)
    out: list[tuple[str, int, int]] = []
    for block_hash, start, end in chain:
        if not check(block_hash):
            break
        out.append((block_hash, start, end))
    return out
