"""Disk-backed KV connector keyed on ZeroPrefill chained block hashes.

Loaded by vLLM through KVTransferConfig(kv_connector_module_path=...), so no
vLLM source changes are needed. Follows the KVConnectorBase_V1 contract of
vllm 0.30.x.

Storage is segmented: N consecutive blocks share one file per layer. Storing one
file per block would mean ~30k tiny reads to restore a 20k-token prompt, which
costs more than the prefill it replaces.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file, save_file

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from .hashing import CacheLayout, block_dir, chain_blocks, matched_prefix_blocks

logger = init_logger(__name__)


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """vLLM 0.30 only defines this inside example_connector, so mirror it."""
    return (num_tokens // block_size) * block_size

# Blocks per stored file. 64 blocks = 1024 tokens at block_size 16.
SEGMENT_BLOCKS = 64
SUMMARY = os.environ.get("ZP_SUMMARY", "./cache/connector_summary.jsonl")


@dataclass
class Segment:
    """A run of consecutive blocks written to one file per layer."""

    seg_hash: str
    start_tok: int
    end_tok: int

    @property
    def num_tokens(self) -> int:
        return self.end_tok - self.start_tok


def segment_chain(chain: list[tuple[str, int, int]],
                  seg_blocks: int = SEGMENT_BLOCKS) -> list[Segment]:
    """Group a block chain into segments keyed on the last block of each.

    The final block hash commits to the whole prefix, so it identifies both the
    segment's contents and its position.
    """
    segs: list[Segment] = []
    for i in range(0, len(chain), seg_blocks):
        chunk = chain[i:i + seg_blocks]
        segs.append(Segment(seg_hash=chunk[-1][0],
                            start_tok=chunk[0][1],
                            end_tok=chunk[-1][2]))
    return segs


def _slice_chain(chain: list[tuple[str, int, int]],
                 seg_blocks: int) -> list[list[tuple[str, int, int]]]:
    return [chain[i:i + seg_blocks] for i in range(0, len(chain), seg_blocks)]


@dataclass
class ZPReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    segments: list[Segment] = field(default_factory=list)
    # Explicit per-segment slots: (seg_hash, token_start, slots). Used on the
    # store path, where chunked prefill delivers the prompt in slices and the
    # mapping from token index to physical slot must be assembled by hand.
    seg_slots: list[tuple[str, int, list[int]]] = field(default_factory=list)

    @staticmethod
    def make_meta(token_ids, block_size, is_store, segments,
                  block_ids=None, windows=None):
        if windows:
            # Store path: slots come per segment.
            seg_slots = [(h, start, slots) for h, start, slots in windows]
            flat = [s for _, _, ss in seg_slots for s in ss]
            return ZPReqMeta(token_ids=torch.tensor(token_ids[:len(flat)],
                                                     dtype=torch.long),
                             slot_mapping=torch.tensor(flat, dtype=torch.long),
                             is_store=is_store,
                             segments=segments,
                             seg_slots=seg_slots)
        # Load path: one contiguous mapping built straight from block ids.
        valid = align_to_block_size(len(token_ids), block_size)
        bid = torch.tensor(block_ids, dtype=torch.long)
        n_blocks = bid.shape[0]
        offsets = torch.arange(0, block_size).reshape((1, block_size))
        slots = (offsets + bid.reshape((n_blocks, 1)) * block_size).flatten()[:valid]
        return ZPReqMeta(token_ids=torch.tensor(token_ids[:valid], dtype=torch.long),
                         slot_mapping=slots,
                         is_store=is_store,
                         segments=segments)

    def slots_for(self, seg: Segment) -> torch.Tensor:
        for seg_hash, start, slots in self.seg_slots:
            if seg_hash == seg.seg_hash:
                return torch.tensor(slots, dtype=torch.long)
        lo, hi = seg.start_tok, seg.end_tok
        return self.slot_mapping[lo:hi]


@dataclass
class ZeroPrefillMetadata(KVConnectorMetadata):
    requests: list[ZPReqMeta] = field(default_factory=list)

    def add_request(self, **kwargs) -> None:
        self.requests.append(ZPReqMeta.make_meta(**kwargs))


@dataclass
class _StoreState:
    """Tracks coverage of one in-flight request's blocks.

    Chunked prefill hands the request over a slice at a time, and the block ids
    that come with each slice are not documented as absolute or relative to the
    prompt. Rather than guess, every slice is recorded as a (token_start, slots)
    window; a segment is written only once a window actually covers it. That
    makes the connector correct under either convention.
    """

    token_ids: list[int]
    segments: list[Segment]
    windows: list[tuple[int, list[int]]] = field(default_factory=list)
    written: set[str] = field(default_factory=set)
    emitted: set[str] = field(default_factory=set)


def _gather(layer: torch.Tensor, slot_mapping: torch.Tensor,
            block_size: int) -> torch.Tensor:
    """Pull [num_tokens, H, C] out of the standardized [B, H, N, C] paged view."""
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    return layer[block_idxs, :, offsets]


def _scatter(dst: torch.Tensor, src: torch.Tensor, slot_mapping: torch.Tensor,
             block_size: int) -> None:
    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    dst[block_idxs, :, offsets] = src


class ZeroPrefillConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = self._kv_transfer_config.get_from_extra_config("zeroprefill", {})
        self.cache_root = extra.get("cache_root", "./cache/kv")
        self.segment_blocks = int(extra.get("segment_blocks", SEGMENT_BLOCKS))
        self._block_size = vllm_config.cache_config.block_size
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        # Set in register_kv_caches(), which only runs on workers and only
        # after distributed init. The scheduler never knows a rank.
        self._tp_rank: int | None = None

        self.layout = CacheLayout(
            model_id=vllm_config.model_config.model,
            quantization=str(extra.get("quantization", "none")),
            kv_dtype=str(vllm_config.cache_config.cache_dtype),
            block_size=self._block_size,
            tp_size=self._tp_size,
        )

        self._requests_need_load: set[str] = set()
        self._saved_reqs: set[str] = set()
        self._store_active: dict[str, _StoreState] = {}
        self._layer_names: list[str] | None = None
        self._restored_tokens = 0
        self._restored_segments = 0
        os.makedirs(self.cache_root, exist_ok=True)
        logger.info("ZeroPrefill root=%s fingerprint=%s seg_blocks=%d",
                    self.cache_root, self.layout.fingerprint, self.segment_blocks)

    # ---------------- paths ----------------

    def _seg_dir(self, seg: Segment) -> str:
        return block_dir(self.cache_root, self.layout, seg.seg_hash)

    def _seg_file(self, seg: Segment, layer_name: str, rank: int) -> str:
        return os.path.join(self._seg_dir(seg),
                            f"layer_{layer_name}.rank{rank}.safetensors")

    def _seg_complete(self, seg: Segment, layer_names: list[str]) -> bool:
        """True only when every layer of every TP rank is on disk."""
        if not layer_names:
            return False
        return all(os.path.exists(self._seg_file(seg, ln, r))
                   for r in range(self._tp_size) for ln in layer_names)

    def _known_layer_names(self) -> list[str]:
        """Layer names for completeness checks.

        register_kv_caches() populates the *worker-side* connector instance, but
        get_num_new_matched_tokens() runs on the *scheduler-side* instance, so
        in-memory state is not shared between them. The layer list therefore has
        to be published to disk at save time and read back here. A copy the
        worker can see is written by save_kv_layer, which runs after
        register_kv_caches.
        """
        if self._layer_names:
            return self._layer_names
        path = os.path.join(self.cache_root, "layers.json")
        try:
            with open(path) as f:
                self._layer_names = list(json.load(f))
        except (OSError, ValueError):
            self._layer_names = []
        return self._layer_names

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._layer_names = list(kv_caches.keys())
        self._tp_rank = get_tensor_model_parallel_rank()
        # Every rank writes the same list; write-then-rename so a reader never
        # sees a half-written file while two ranks race.
        path = os.path.join(self.cache_root, "layers.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(self.cache_root, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(self._layer_names, f)
            os.replace(tmp, path)
        except OSError:
            pass
        super().register_kv_caches(kv_caches)

    # ---------------- worker side ----------------

    def start_load_kv(self, forward_context, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        first_layer = next(iter(forward_context.no_compile_layers), None)
        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                for seg in request.segments:
                    path = self._seg_file(seg, layer_name, self._tp_rank)
                    if not os.path.exists(path):
                        continue
                    with gpu_sync_allowed():
                        cpu_t = load_file(path)["kv_cache"]
                        src = cpu_t.to(kv_cache_layer.device, non_blocking=True)
                    slots = request.slots_for(seg)
                    if slots.numel() == 0 or src.shape[0] != slots.numel():
                        logger.warning("segment %s shape mismatch: %s vs %s",
                                       seg.seg_hash[:12], src.shape[0], slots.numel())
                        continue
                    with gpu_sync_allowed():
                        slots = slots.to(kv_cache_layer.device, non_blocking=True)
                    _scatter(kv_cache_layer, src, slots, self._block_size)
                    if layer_name == first_layer:
                        self._restored_segments += 1
                        self._restored_tokens += seg.num_tokens

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        for request in metadata.requests:
            if not request.is_store:
                continue
            for seg in request.segments:
                slots = request.slots_for(seg)
                if slots.numel() != seg.num_tokens:
                    continue
                with gpu_sync_allowed():
                    slots = slots.to(kv_layer.device, non_blocking=True)
                kv = _gather(kv_layer, slots, self._block_size)
                with gpu_sync_allowed():
                    cpu_t = {"kv_cache": kv.detach().cpu()}
                os.makedirs(self._seg_dir(seg), exist_ok=True)
                # Lookups only check that a file exists, so a file cut short
                # by a crash must never appear under its final name.
                path = self._seg_file(seg, layer_name, self._tp_rank)
                save_file(cpu_t, path + ".tmp")
                os.replace(path + ".tmp", path)
                self._mark_written(seg)

    def _mark_written(self, seg: Segment) -> None:
        return

    def wait_for_save(self) -> None:
        return

    # ---------------- scheduler side ----------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        token_ids = list(request.prompt_token_ids or [])
        if not token_ids:
            return 0, False
        logger.info("ZP_LOOKUP req=%s tokens=%d", request.request_id, len(token_ids))
        chain = chain_blocks(token_ids, self.layout)
        if not chain:
            return 0, False
        segs = segment_chain(chain, self.segment_blocks)
        layer_names = self._known_layer_names()
        if not layer_names:
            logger.info("ZP_MISS layer names not registered yet")
            return 0, False

        # matched_prefix_blocks() asks about individual block hashes, but disk
        # holds segments keyed on the last block of each run. A block counts as
        # present when the segment ending at or after it is on disk.
        block_to_seg: dict[str, Segment] = {}
        for seg, seg_blocks in zip(segs, _slice_chain(chain, self.segment_blocks)):
            for bh, _, _ in seg_blocks:
                block_to_seg[bh] = seg

        def present(block_hash: str) -> bool:
            seg = block_to_seg.get(block_hash)
            return seg is not None and os.path.isdir(self._seg_dir(seg))

        matched = matched_prefix_blocks(token_ids, self.layout, self.cache_root,
                                        is_present=present)
        if not matched:
            logger.info("ZP_MISS no prefix matched (segments=%d)", len(segs))
            return 0, False

        complete = 0
        for seg in segment_chain(matched, self.segment_blocks):
            if self._seg_complete(seg, layer_names):
                complete += seg.num_tokens
            else:
                break
        if complete == 0:
            return 0, False
        loadable = min(complete, len(token_ids) - 1)
        loadable = align_to_block_size(loadable, self._block_size)
        delta = loadable - num_computed_tokens
        if delta <= 0:
            return 0, False
        logger.info("ZP_HIT tokens=%d of %d (%.1f%%)", loadable, len(token_ids),
                    100.0 * loadable / len(token_ids))
        return delta, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        if num_external_tokens > 0:
            self._requests_need_load.add(request.request_id)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = ZeroPrefillMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            if not token_ids:
                continue
            if new_req.req_id in self._requests_need_load:
                segs = segment_chain(chain_blocks(token_ids, self.layout),
                                     self.segment_blocks)
                if segs:
                    meta.add_request(token_ids=token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size,
                                     is_store=False,
                                     segments=segs)
            elif new_req.req_id not in self._saved_reqs:
                self._begin_store(new_req.req_id, token_ids)
                # num_computed is nonzero when vLLM's own prefix cache already
                # holds the start of the prompt, e.g. turn 2+ of an agent loop.
                self._emit_store(meta, new_req.req_id, new_req.block_ids[0],
                                 num_computed=new_req.num_computed_tokens,
                                 num_scheduled=scheduler_output.num_scheduled_tokens.get(
                                     new_req.req_id, 0))

        # Chunked prefill: a long prompt arrives over several steps, and each
        # step only fills in the next slice of blocks. Without this the tail of
        # a 20k prompt is never written to disk.
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id not in self._store_active:
                continue
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is None:
                continue
            num_computed = cached.num_computed_tokens[i]
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            self._emit_store(meta, req_id, new_block_ids[0],
                             num_computed=num_computed,
                             num_scheduled=num_scheduled)

        self._requests_need_load.clear()
        return meta

    def _begin_store(self, req_id: str, token_ids: list[int]) -> None:
        segs = segment_chain(chain_blocks(token_ids, self.layout),
                             self.segment_blocks)
        # In an agent loop every turn's prompt repeats the whole history.
        # Segments already on disk are skipped, otherwise each turn would
        # rewrite the entire cache.
        layer_names = self._known_layer_names()
        done = {s.seg_hash for s in segs if self._seg_complete(s, layer_names)}
        self._store_active[req_id] = _StoreState(token_ids=token_ids, segments=segs,
                                                 emitted=done)

    def _emit_store(self, meta: ZeroPrefillMetadata, req_id: str,
                    block_ids: list[int], num_computed: int,
                    num_scheduled: int) -> None:
        """Persist every segment whose blocks are now fully computed."""
        state = self._store_active.get(req_id)
        if state is None:
            return
        bs = self._block_size
        tokens_done = num_computed + num_scheduled

        bid = torch.tensor(list(block_ids), dtype=torch.long)
        offsets = torch.arange(bs, dtype=torch.long).reshape(1, -1)
        slots = (offsets + bid.reshape(-1, 1) * bs).flatten().tolist()

        # Record this slice. The ids are either only the newly allocated blocks
        # (they start at the first block not yet allocated) or all blocks from
        # token 0. vLLM allocates exactly enough blocks for tokens_done, so the
        # block count tells the two readings apart. Prompt lengths are rarely a
        # multiple of block_size, so the last block usually has padding slots.
        needed_blocks = -(-tokens_done // bs)
        for start in (-(-num_computed // bs) * bs, 0):
            if start // bs + len(block_ids) != needed_blocks:
                continue
            if start and self._window_conflicts(state, start, slots):
                continue
            state.windows.append((start, slots))
            break
        else:
            logger.warning("could not place chunk of %d slots for %s",
                           len(slots), req_id)

        # Windows span whole blocks, so a segment can be mapped before all of
        # its tokens are computed. Only save it once they are.
        pending = [s for s in state.segments
                   if s.seg_hash not in state.emitted
                   and s.end_tok <= tokens_done
                   and self._cover(state, s) is not None]
        if not pending:
            return
        state.emitted |= {s.seg_hash for s in pending}
        meta.add_request(token_ids=state.token_ids,
                         block_size=bs,
                         is_store=True,
                         segments=pending,
                         windows=[(s.seg_hash, *self._cover(state, s))
                                  for s in pending])

    @staticmethod
    def _window_conflicts(state: _StoreState, start: int, slots: list[int]) -> bool:
        for other_start, other in state.windows:
            for i, slot in enumerate(slots):
                pos = start + i
                if other_start <= pos < other_start + len(other):
                    if other[pos - other_start] != slot:
                        return True
        return False

    def _cover(self, state: _StoreState, seg: Segment) -> tuple[int, list[int]] | None:
        """Return (token_start, slots) if `seg` is fully covered by windows."""
        merged: dict[int, int] = {}
        for start, slots in state.windows:
            for i, slot in enumerate(slots):
                pos = start + i
                if seg.start_tok <= pos < seg.end_tok:
                    merged[pos] = slot
        if len(merged) != seg.num_tokens:
            return None
        return seg.start_tok, [merged[p] for p in range(seg.start_tok, seg.end_tok)]

    def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
        if block_ids and request.prompt_token_ids:
            self._saved_reqs.add(request.request_id)
            state = self._store_active.pop(request.request_id, None)
            n_files = sum(len(f) for _, _, f in os.walk(self.cache_root))
            # Counted from disk: the worker-side connector writes the tensors,
            # so an in-memory flag on this scheduler-side object would not know.
            on_disk = set()
            for _, dirs, _ in os.walk(self.cache_root):
                on_disk.update(dirs)
            rec = {"event": "saved",
                   "request_id": request.request_id,
                   "prompt_tokens": len(request.prompt_token_ids),
                   "segments_total": len(state.segments) if state else 0,
                   "segments_emitted": len(state.emitted) if state else 0,
                   "segment_dirs_on_disk": len(on_disk),
                   "cache_files": n_files}
            logger.info("ZP_SAVED %s", json.dumps(rec))
            _append_summary(rec)
        return False, None

    def get_finished(self, finished_req_ids):
        return None, None


def _append_summary(rec: dict) -> None:
    try:
        os.makedirs(os.path.dirname(SUMMARY) or ".", exist_ok=True)
        with open(SUMMARY, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# vLLM's logging setup only wires handlers under the "vllm" namespace, so a
# connector living outside the package can emit to nowhere. Send our records
# to stderr directly; the JSONL summary remains the machine-readable channel.
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[ZP] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
