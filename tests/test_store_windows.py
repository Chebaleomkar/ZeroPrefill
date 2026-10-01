"""Store path: every segment gets saved from the right physical slots.

Chunked prefill hands the scheduler a request a slice at a time, and prompt
lengths are rarely a multiple of block_size. Phase 1 only used aligned prompts
(20000, 4096), which hid a bug where unaligned prompts were saved from the
wrong slots or not at all.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("vllm", reason="vLLM is required to import the connector")

from zeroprefill.connector import (  # noqa: E402
    ZeroPrefillConnector,
    ZeroPrefillMetadata,
    segment_chain,
)
from zeroprefill.hashing import CacheLayout, chain_blocks  # noqa: E402

BS = 16
LAYOUT = CacheLayout("m", "awq", "float16", BS, tp_size=2)


def _connector(tmp_path):
    c = ZeroPrefillConnector.__new__(ZeroPrefillConnector)
    c._block_size = BS
    c.segment_blocks = 64
    c.layout = LAYOUT
    c.cache_root = str(tmp_path)
    c._tp_size = 2
    c._layer_names = ["l0"]
    c._store_active = {}
    return c


def _blocks(n):
    """Physical block ids in a scrambled order, as a real allocator gives."""
    ids = list(range(100, 100 + n))
    random.Random(0).shuffle(ids)
    return ids


def _run(tmp_path, n_tokens, steps):
    """steps: (block_ids, num_computed, num_scheduled) per scheduler step."""
    c = _connector(tmp_path)
    tokens = list(range(n_tokens))
    c._begin_store("r", tokens)
    meta = ZeroPrefillMetadata()
    for block_ids, computed, scheduled in steps:
        c._emit_store(meta, "r", block_ids, computed, scheduled)
    return meta, segment_chain(chain_blocks(tokens, LAYOUT), 64)


def _assert_all_saved_correctly(meta, segs, phys):
    saved = {}
    for req in meta.requests:
        for seg_hash, start, slots in req.seg_slots:
            saved[seg_hash] = (start, slots)
    assert set(saved) == {s.seg_hash for s in segs}
    for s in segs:
        start, slots = saved[s.seg_hash]
        assert start == s.start_tok
        assert slots == [phys[p // BS] * BS + p % BS for p in range(s.start_tok, s.end_tok)]


def test_two_chunks_new_blocks_only(tmp_path):
    phys = _blocks(1003)  # ceil(16043 / 16)
    meta, segs = _run(tmp_path, 16043, [(phys[:512], 0, 8192),
                                        (phys[512:], 8192, 7851)])
    assert len(segs) == 16
    _assert_all_saved_correctly(meta, segs, phys)


def test_two_chunks_all_blocks_every_time(tmp_path):
    phys = _blocks(1003)
    meta, segs = _run(tmp_path, 16043, [(phys[:512], 0, 8192),
                                        (phys, 8192, 7851)])
    _assert_all_saved_correctly(meta, segs, phys)


def test_single_unaligned_chunk(tmp_path):
    phys = _blocks(188)  # ceil(3000 / 16)
    meta, segs = _run(tmp_path, 3000, [(phys, 0, 3000)])
    assert len(segs) == 3
    _assert_all_saved_correctly(meta, segs, phys)


def test_chunk_ending_mid_block(tmp_path):
    """Block 511 is half computed after step 1; its segment must wait."""
    phys = _blocks(1003)
    c_meta, segs = _run(tmp_path, 16043, [(phys[:512], 0, 8190)])
    early = {h for r in c_meta.requests for h, _, _ in r.seg_slots}
    assert all(s.end_tok <= 8190 for s in segs if s.seg_hash in early)
    meta, segs = _run(tmp_path, 16043, [(phys[:512], 0, 8190),
                                        (phys[512:], 8190, 7853)])
    _assert_all_saved_correctly(meta, segs, phys)


def test_local_prefix_cache_hit(tmp_path):
    """Turn 2+ of an agent loop: vLLM already holds the first 2048 tokens."""
    phys = _blocks(313)  # ceil(5000 / 16)
    meta, segs = _run(tmp_path, 5000, [(phys, 2048, 2952)])
    _assert_all_saved_correctly(meta, segs, phys)
