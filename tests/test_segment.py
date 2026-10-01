"""Segmented storage: chaining correctness and segment identity.

These import the connector, which needs vLLM present. That is intentional: the
segment logic lives in the connector because it must agree with the layout
written at save time.
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM is required to import the connector")

from zeroprefill.connector import segment_chain  # noqa: E402
from zeroprefill.hashing import CacheLayout, chain_blocks  # noqa: E402

LAYOUT = CacheLayout("Qwen/Qwen2.5-0.5B-Instruct", "none", "float16", 16)
SEG = 64


def _chain(ids=None, n: int = 20000):
    return chain_blocks(list(range(n)) if ids is None else ids, LAYOUT)


def test_segment_count_and_coverage():
    chain = _chain()
    segs = segment_chain(chain, SEG)
    assert len(segs) == 20          # 1249 blocks / 64, last segment partial
    assert segs[0].start_tok == 0
    assert segs[-1].end_tok == chain[-1][2]
    for a, b in zip(segs, segs[1:]):
        assert a.end_tok == b.start_tok


def test_segment_sizes_are_block_aligned():
    for seg in segment_chain(_chain(), SEG):
        assert seg.num_tokens % LAYOUT.block_size == 0
    assert segment_chain(_chain(), SEG)[0].num_tokens == SEG * LAYOUT.block_size


def test_segment_hash_commits_to_position():
    """Identical block content at a different offset must not collide."""
    a = segment_chain(_chain(), SEG)
    b = segment_chain(_chain(ids=[7] * 20000), SEG)
    assert {s.seg_hash for s in a} != {s.seg_hash for s in b}
    assert len({s.seg_hash for s in a}) == len(a)


def test_block_to_segment_mapping_covers_every_block():
    """The lookup path asks per block; disk is keyed per segment."""
    from zeroprefill.connector import _slice_chain

    chain = _chain()
    segs = segment_chain(chain, SEG)
    slices = _slice_chain(chain, SEG)
    assert len(segs) == len(slices)
    mapping = {}
    for seg, blocks in zip(segs, slices):
        for bh, _, _ in blocks:
            mapping[bh] = seg
    assert len(mapping) == len(chain)
    # every block maps to the segment that ends at or after it
    assert mapping[chain[0][0]] is segs[0]
    assert mapping[chain[63][0]] is segs[0]
    assert mapping[chain[64][0]] is segs[1]
    assert mapping[chain[-1][0]] is segs[-1]


def test_edit_changes_only_later_segments():
    before = segment_chain(_chain(), SEG)
    after = segment_chain(_chain(ids=[*range(5000), 123456, *range(5001, 20000)]), SEG)
    first = next(i for i, (x, y) in enumerate(zip(before, after))
                 if x.seg_hash != y.seg_hash)
    assert first == (5000 // LAYOUT.block_size) // SEG
    assert ([s.seg_hash for s in before[:first]]
            == [s.seg_hash for s in after[:first]])
