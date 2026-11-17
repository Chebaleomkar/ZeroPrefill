"""Properties the cache key must hold for partial reuse to be possible."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from zeroprefill.hashing import (
    CacheLayout,
    block_dir,
    chain_blocks,
    matched_prefix_blocks,
    rope_json,
)

LAYOUT = CacheLayout(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    quantization="none",
    kv_dtype="float16",
    block_size=16,
)


def _touch(root, hashes):
    for h in hashes:
        os.makedirs(block_dir(root, LAYOUT, h), exist_ok=True)


def test_block_count_and_uniqueness():
    ids = list(range(20000))
    chain = chain_blocks(ids, LAYOUT)
    assert len(chain) == (20000 - 1) // 16
    assert len({h for h, _, _ in chain}) == len(chain)


def test_mid_prompt_edit_invalidates_only_the_suffix():
    ids = list(range(20000))
    before = [h for h, _, _ in chain_blocks(ids, LAYOUT)]

    edited = list(ids)
    edited[5000] = 999_999
    after = [h for h, _, _ in chain_blocks(edited, LAYOUT)]

    first_diff = next(i for i, (a, b) in enumerate(zip(before, after)) if a != b)
    assert first_diff == 5000 // 16
    # The blocks ahead of the edit are still reusable.
    assert before[:first_diff] == after[:first_diff]


def test_append_invalidates_nothing():
    """The agent-loop case: every turn appends tokens and must hit the cache."""
    ids = list(range(20000))
    before = [h for h, _, _ in chain_blocks(ids, LAYOUT)]
    after = [h for h, _, _ in chain_blocks(ids + [12345], LAYOUT)]
    assert after[: len(before)] == before


def test_different_kv_dtype_does_not_collide():
    other = CacheLayout(
        model_id=LAYOUT.model_id,
        quantization=LAYOUT.quantization,
        kv_dtype="bfloat16",
        block_size=LAYOUT.block_size,
    )
    ids = list(range(32))
    assert LAYOUT.fingerprint != other.fingerprint
    assert chain_blocks(ids, LAYOUT)[0][0] != chain_blocks(ids, other)[0][0]


def test_different_tp_size_does_not_collide():
    other = CacheLayout(
        model_id=LAYOUT.model_id,
        quantization=LAYOUT.quantization,
        kv_dtype=LAYOUT.kv_dtype,
        block_size=LAYOUT.block_size,
        tp_size=2,
    )
    ids = list(range(32))
    assert LAYOUT.fingerprint != other.fingerprint
    assert chain_blocks(ids, LAYOUT)[0][0] != chain_blocks(ids, other)[0][0]


def test_different_rope_scaling_does_not_collide():
    """A YaRN-extended model stores different keys for the same tokens."""
    base = SimpleNamespace(rope_theta=1000000.0, rope_scaling=None)
    yarn = SimpleNamespace(rope_theta=1000000.0,
                           rope_scaling={"type": "yarn", "factor": 4.0,
                                         "original_max_position_embeddings": 32768})
    a = replace(LAYOUT, rope=rope_json(base))
    b = replace(LAYOUT, rope=rope_json(yarn))
    ids = list(range(32))
    assert a.fingerprint != b.fingerprint
    assert chain_blocks(ids, a)[0][0] != chain_blocks(ids, b)[0][0]


def test_rope_json_reads_old_and_new_transformers_configs():
    old = SimpleNamespace(rope_theta=10000.0, rope_scaling={"factor": 2.0, "type": "linear"})
    new = SimpleNamespace(rope_parameters={"type": "linear", "factor": 2.0,
                                           "rope_theta": 10000.0})
    plain = SimpleNamespace()
    assert rope_json(old) != rope_json(plain)
    assert rope_json(new) != rope_json(plain)
    # Key order inside the config must not change the cache key.
    reordered = SimpleNamespace(rope_theta=10000.0,
                                rope_scaling={"type": "linear", "factor": 2.0})
    assert rope_json(old) == rope_json(reordered)


def test_different_max_model_len_does_not_collide():
    a = replace(LAYOUT, max_model_len=32768)
    b = replace(LAYOUT, max_model_len=131072)
    assert a.fingerprint != b.fingerprint


def test_partial_tail_is_not_cached():
    """The final token has no stable KV until decode advances."""
    assert chain_blocks(list(range(16)), LAYOUT) == []
    assert len(chain_blocks(list(range(17)), LAYOUT)) == 1
    assert len(chain_blocks(list(range(32)), LAYOUT)) == 1


def test_matched_prefix_stops_at_first_hole(tmp_path):
    root = str(tmp_path)
    chain = chain_blocks(list(range(20000)), LAYOUT)
    _touch(root, [h for h, _, _ in chain[:300]])

    assert len(matched_prefix_blocks(list(range(20000)), LAYOUT, root)) == 300

    import shutil

    shutil.rmtree(block_dir(root, LAYOUT, chain[250][0]))
    assert len(matched_prefix_blocks(list(range(20000)), LAYOUT, root)) == 250


def test_cold_cache_yields_no_match(tmp_path):
    assert matched_prefix_blocks(list(range(20000)), LAYOUT, str(tmp_path)) == []


def test_is_present_predicate_drives_matching(tmp_path):
    """Segmented storage means no per-block dir exists; the predicate decides."""
    ids = list(range(20000))
    assert matched_prefix_blocks(ids, LAYOUT, str(tmp_path)) == []

    allowed = {h for h, _, _ in chain_blocks(ids, LAYOUT)[:320]}
    matched = matched_prefix_blocks(
        ids, LAYOUT, str(tmp_path), is_present=lambda h: h in allowed)
    assert len(matched) == 320

    # A hole inside the allowed set truncates the run at the hole.
    holed = {h for i, (h, _, _) in enumerate(chain_blocks(ids, LAYOUT))
             if i < 320 and i != 250}
    matched2 = matched_prefix_blocks(
        ids, LAYOUT, str(tmp_path), is_present=lambda h: h in holed)
    assert len(matched2) == 250
