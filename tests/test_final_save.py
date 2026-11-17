"""End-of-session save: the model's last answer is cached for the next turn.

Before this, only prompt tokens were saved, so on resume the previous answer
(~3k tokens in the Day 2 run) was prefilled again. The next turn's prompt is
prompt + answer + new message; these tests check that its lookup now covers
the answer, using the scheduler-side and worker-side methods vLLM calls.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="vLLM is required to import the connector")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import zeroprefill.connector as zc  # noqa: E402
from zeroprefill.connector import ZeroPrefillConnector  # noqa: E402
from zeroprefill.hashing import CacheLayout  # noqa: E402

BS = 16
SEG = 4  # blocks per segment, small so a short session spans several
LAYOUT = CacheLayout("m", "awq", "float16", BS, tp_size=1)
HEADS, DIM, NUM_BLOCKS = 2, 4, 64
LAYERS = ["l0", "l1"]


def _connector(tmp_path, monkeypatch):
    monkeypatch.setattr(zc, "SUMMARY", str(tmp_path / "summary.jsonl"))
    c = ZeroPrefillConnector.__new__(ZeroPrefillConnector)
    c._block_size = BS
    c.segment_blocks = SEG
    c.layout = LAYOUT
    c.cache_root = str(tmp_path / "kv")
    c._tp_size = 1
    c._tp_rank = 0
    c._layer_names = list(LAYERS)
    c._store_active = {}
    c._final_saves = {}
    c._saved_reqs = set()
    c._requests_need_load = set()
    c._connector_metadata = None
    return c


def _kv_caches():
    """Paged KV per layer in vLLM's [blocks, heads, block_size, dim] view."""
    g = torch.Generator().manual_seed(0)
    return {ln: torch.randn(NUM_BLOCKS, HEADS, BS, DIM, generator=g) for ln in LAYERS}


def _finished_request(prompt, answer, block_ids):
    # vLLM's state when the stop token is sampled: the last sampled token has
    # no KV yet, so it is not counted as computed.
    return SimpleNamespace(request_id="turn1", prompt_token_ids=prompt,
                           all_token_ids=prompt + answer,
                           num_computed_tokens=len(prompt) + len(answer) - 1,
                           num_in_flight_tokens=0)


def _run_worker_step(c, monkeypatch, kv):
    """One engine step with no forward pass, as after the last request ends."""
    sched_out = SimpleNamespace(scheduled_new_reqs=[],
                                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                                num_scheduled_tokens={})
    c.bind_connector_metadata(c.build_connector_meta(sched_out))
    # No forward context is set: vLLM 0.30's model runner calls get_finished()
    # outside one on steps with no forward pass, and reading it crashed the
    # engine on the first real run.
    layers = {ln: SimpleNamespace(kv_cache=t) for ln, t in kv.items()}
    c._vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers))
    try:
        return c.get_finished({"turn1"})
    finally:
        c.clear_connector_metadata()


def _next_turn(tokens):
    return SimpleNamespace(request_id="turn2", prompt_token_ids=tokens)


def test_next_turn_hits_the_previous_answer(tmp_path, monkeypatch):
    c = _connector(tmp_path, monkeypatch)
    prompt = list(range(1000, 1000 + 150))
    answer = list(range(5000, 5000 + 230))
    blocks = list(range(NUM_BLOCKS))
    random.Random(0).shuffle(blocks)
    req = _finished_request(prompt, answer, blocks)

    hold, _ = c.request_finished(req, blocks)
    assert hold, "blocks must stay allocated until the answer is written"

    kv = _kv_caches()
    sending, recving = _run_worker_step(c, monkeypatch, kv)
    assert sending == {"turn1"} and recving is None

    # Turn 2 sends the whole history plus a new message.
    computed = len(prompt) + len(answer) - 1
    turn2 = prompt + answer + list(range(9000, 9040))
    matched, _ = c.get_num_new_matched_tokens(_next_turn(turn2), 0)
    full_segments_tokens = (computed // (SEG * BS)) * SEG * BS
    assert matched == full_segments_tokens
    assert matched > len(prompt), "the answer itself must now be restorable"


def test_saved_kv_is_the_kv_of_those_tokens(tmp_path, monkeypatch):
    """Each saved row comes from the physical slot that held that token."""
    c = _connector(tmp_path, monkeypatch)
    prompt, answer = list(range(70)), list(range(100, 160))
    blocks = list(range(NUM_BLOCKS))
    random.Random(1).shuffle(blocks)
    c.request_finished(_finished_request(prompt, answer, blocks), blocks)
    segs = [s for _, s, _ in c._final_saves.values()][0]

    kv = _kv_caches()
    _run_worker_step(c, monkeypatch, kv)

    for seg in segs:
        for ln in LAYERS:
            saved = load_file(c._seg_file(seg, ln, 0))["kv_cache"]
            for i, pos in enumerate(range(seg.start_tok, seg.end_tok)):
                b, off = blocks[pos // BS], pos % BS
                assert torch.equal(saved[i], kv[ln][b, :, off])


def test_nothing_to_add_releases_blocks_immediately(tmp_path, monkeypatch):
    """A request too short to fill one block has nothing to save."""
    c = _connector(tmp_path, monkeypatch)
    req = _finished_request(list(range(5)), list(range(5, 9)), [0])
    hold, _ = c.request_finished(req, [0])
    assert not hold
    assert c._final_saves == {}


def test_failed_write_still_releases_blocks(tmp_path, monkeypatch):
    """A save error must not leave the scheduler holding blocks forever."""
    c = _connector(tmp_path, monkeypatch)
    blocks = list(range(NUM_BLOCKS))
    c.request_finished(_finished_request(list(range(100)), list(range(100)), blocks),
                       blocks)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(zc, "save_file", boom)
    sending, _ = _run_worker_step(c, monkeypatch, _kv_caches())
    assert sending == {"turn1"}
