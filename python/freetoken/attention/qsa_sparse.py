"""Qwen4 QSA sparse GQA attention backend.

The correctness-first cache stores raw per-token index keys. Complete logical
4-token blocks are compressed on demand by ``Qwen4ExpQSAIndexer``; selected
token positions are mapped through the normal page table, K/V are compactly
gathered, and a dense GQA attention is evaluated over at most 2051 positions.

Short prefills take an identity path through a wrapped FULL backend. Decode uses
one fixed-shape path on both sides of the identity boundary: top-k covers every
complete block below the boundary, so the 2051-slot selection is exactly dense
there without a host-side branch. Long prefill scores query chunks against one
request-wide compressed-key table.

Decode stages a fixed-width page-table snapshot plus device live lengths. A
fixed-grid score kernel reads raw index keys through that snapshot, selection is
fixed at 2051 slots, and only the selected K/V are gathered. The whole
score -> top-k -> chronological gather -> dense GQA path is CUDA-graph safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.utils import decode_profile_range, init_logger

from .base import AttentionSpec, AttnType, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig
    from freetoken.models.qwen4_exp.indexer import Qwen4ExpQSAIndexer

logger = init_logger(__name__)

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
_PREFILL_SCORE_BYTES = 256 << 20
_PREFILL_WORK_BYTES = 256 << 20
_PREFILL_CHUNK = 512


_FALSE_VALUES = {"0", "false", "no", "off"}


def _direct_attend_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN4_QSA_DIRECT_ATTEND", "1").strip().lower() not in _FALSE_VALUES


def _pick_inner_backend(page_size: int) -> str:
    """Resolve a FULL backend that can address the QSA pool's page size."""
    from freetoken.attention import attention_backend_info

    def _page_ok(name: str) -> bool:
        return all(
            (sizes := attention_backend_info(part).page_sizes) is None
            or page_size in sizes
            for part in name.split(",")
        )

    override = os.getenv("FREETOKEN_QWEN4_INNER_BACKEND")
    if override:
        for part in override.split(","):
            try:
                info = attention_backend_info(part)
            except KeyError:
                raise ValueError(
                    f"FREETOKEN_QWEN4_INNER_BACKEND={override!r}: unknown "
                    f"attention backend {part!r}"
                ) from None
            if AttnType.FULL not in info.supported_types:
                raise ValueError(
                    f"FREETOKEN_QWEN4_INNER_BACKEND={override!r}: {part!r} "
                    "does not serve FULL attention"
                )
        if not _page_ok(override):
            raise ValueError(
                f"FREETOKEN_QWEN4_INNER_BACKEND={override!r} cannot address "
                f"the QSA pool's {page_size}-token pages"
            )
        logger.info(f"qsa_sparse identity backend: {override} (env override)")
        return override

    from freetoken.engine.engine import _resolve_auto_attention_backend

    name = _resolve_auto_attention_backend(frozenset({AttnType.FULL}), False)
    if not _page_ok(name):
        from freetoken.engine.engine import _backend_requirements_met
        from freetoken.utils.arch import is_sm90_family

        for candidate, arch_ok in (
            ("fa,fi", is_sm90_family()),
            ("fi", True),
            ("triton", True),
        ):
            if arch_ok and _page_ok(candidate) and _backend_requirements_met(candidate):
                name = candidate
                break
        else:  # pragma: no cover - triton is unconditional
            name = "triton"
    logger.info(f"qsa_sparse identity backend: {name} (auto)")
    return name


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    is_decode: bool
    last_indices: torch.Tensor
    qo_indptr_cpu: torch.Tensor
    kv_len_cpu: torch.Tensor
    inner: BaseAttnMetadata | None
    # Decode-only logical -> physical row snapshot and device-read live lengths.
    # Graph replay points these at persistent capture buffers; eager decode
    # materializes an equivalent snapshot lazily at its first QSA layer.
    rows: torch.Tensor | None = None
    kvlen: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.attention import create_attention_backend
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None and args.use_sparse, (
            "qsa_sparse backend needs Qwen4 sparse mode; set FREETOKEN_QWEN4_SPARSE=1"
        )
        self.config = config
        self.args = args
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.groups_per_kv = self.num_heads // self.num_kv_heads
        self.sm_scale = config.attn_sm_scale or (self.head_dim**-0.5)
        self.identity_limit = args.indexer_budget + args.indexer_compress_ratio - 1
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        assert isinstance(self.kvcache, QSAKVCache), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )

        qsa_specs = [
            spec
            for spec in config.kv_cache_group_specs()
            if spec.attn_type is AttnType.QSA
        ]
        assert len(qsa_specs) == 1, f"expected one QSA group, got {len(qsa_specs)}"
        self._idx_slot = {
            layer_id: slot for slot, layer_id in enumerate(qsa_specs[0].layer_ids)
        }

        self._inner_name = _pick_inner_backend(self.kvcache._page_size)
        self.inner = create_attention_backend(self._inner_name, config)
        self._rows_buf: torch.Tensor | None = None
        self._kvlen_buf: torch.Tensor | None = None
        self._stage_width = 0
        self.capture_bs: list[int] = []
        self._direct_attend = _direct_attend_enabled()

    def _k_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.k_cache(layer_id)
        return cache.view(-1, cache.shape[-2], cache.shape[-1])

    def _v_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.v_cache(layer_id)
        return cache.view(-1, cache.shape[-2], cache.shape[-1])

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        if is_decode:
            # QSA decode never delegates, including in the identity range. Avoid
            # constructing or capturing the unused FULL backend's metadata.
            inner_md = None
        else:
            self.inner.prepare_metadata(batch)
            inner_md = batch.attn_metadata
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        batch.attn_metadata = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr.to(torch.int32),
            kv_len_cpu=kv_len,
            inner=inner_md,
        )

    def _delegate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert md.inner is not None, "QSA FULL delegation is prefill-only"
        batch.attn_metadata = md.inner
        try:
            return self.inner.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)
        finally:
            batch.attn_metadata = md

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        return self._delegate(q, k, v, layer_id, batch, attn_spec)

    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        raw_index_k: torch.Tensor,
        indexer: Qwen4ExpQSAIndexer,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        # Raw-key correctness baseline. TODO(qsa-cache): emit one compressed key
        # per complete 4-token block and retain only a four-slot tail ring.
        with decode_profile_range("FT.QSA.IndexKeyStore"):
            self.kvcache.store_index_k(raw_index_k, batch.out_loc, slot)

        # Decode is a single fixed-shape code path, including the identity range.
        # For kv_len <= identity_limit, top-k necessarily contains every complete
        # block and the tail contains every remaining token, so this is dense
        # attention without a capture-breaking host decision.
        if md.is_decode:
            with decode_profile_range("FT.QSA.KVStore"):
                self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
            if md.rows is None:
                md.rows = self._decode_rows(batch, int(md.kv_len_cpu.max()))
                md.kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)
            return self._decode(md, layer_id, slot, q, index_q, indexer)

        # Prefill remains eager; its short-context FULL delegation is outside any
        # captured graph and retains the dense kernel's bitwise identity behavior.
        if int(md.kv_len_cpu.max()) <= self.identity_limit:
            return self._delegate(q, k, v, layer_id, batch)

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return self._prefill(md, layer_id, slot, q, index_q, indexer, batch)

    def _logical_to_physical(
        self, selected: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        # Transformers applies a mask over chronological K/V. Sorting the compact
        # gather restores that reduction order instead of top-k score order.
        sentinel = torch.iinfo(selected.dtype).max
        order = torch.where(selected >= 0, selected, sentinel)
        logical = order.sort(dim=-1).values
        valid = logical != sentinel
        safe = torch.where(valid, logical, torch.zeros_like(logical)).long()
        if rows.ndim == 1:
            physical = rows.index_select(0, safe.flatten()).view_as(logical)
        else:
            if rows.ndim != 2 or rows.shape[0] != selected.shape[0]:
                raise ValueError(
                    "QSA batched logical mapping expects [batch, width] rows, "
                    f"got {tuple(rows.shape)} for {tuple(selected.shape)}"
                )
            physical = torch.gather(rows, 1, safe)
        return torch.where(valid, physical, torch.full_like(physical, -1)).to(
            torch.int32
        )

    def _attend_selected(
        self, q: torch.Tensor, layer_id: int, physical_rows: torch.Tensor
    ) -> torch.Tensor:
        count, width = physical_rows.shape
        with decode_profile_range("FT.QSA.KVGather"):
            valid = physical_rows >= 0
            safe_rows = physical_rows.clamp_min(0).long()
            k = self._k_rows(layer_id).index_select(0, safe_rows.flatten())
            v = self._v_rows(layer_id).index_select(0, safe_rows.flatten())
            k = k.view(count, width, self.num_kv_heads, self.head_dim)
            v = v.view(count, width, self.num_kv_heads, self.head_dim)
        with decode_profile_range("FT.QSA.GQAAttend"):
            q_grouped = q.view(count, self.num_kv_heads, self.groups_per_kv, self.head_dim)
            # This is the Transformers eager ``q @ k.T`` / ``p @ v`` order with the
            # repeated GQA heads represented as a view rather than materialized 12x.
            logits = torch.matmul(q_grouped, k.permute(0, 2, 3, 1)) * self.sm_scale
            logits.masked_fill_(~valid[:, None, None, :], float("-inf"))
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v.permute(0, 2, 1, 3))
            return out.reshape(count, self.num_heads, self.head_dim)

    def _attend_selected_direct(
        self, q: torch.Tensor, layer_id: int, physical_rows: torch.Tensor
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa_sparse import qsa_direct_attention

        return qsa_direct_attention(
            q,
            self._k_rows(layer_id),
            self._v_rows(layer_id),
            physical_rows,
            softmax_scale=self.sm_scale,
        )

    def _compressed_keys(
        self,
        slot: int,
        rows: torch.Tensor,
        kv_len: int,
        indexer: Qwen4ExpQSAIndexer,
    ) -> torch.Tensor:
        raw_keys = self.kvcache.index_k_cache(slot).index_select(0, rows.long())
        num_blocks = kv_len // indexer.compress_ratio
        block_tokens = torch.arange(
            num_blocks * indexer.compress_ratio,
            device=self.device,
            dtype=torch.long,
        ).view(num_blocks, indexer.compress_ratio)
        positions = torch.arange(kv_len, device=self.device, dtype=torch.long)
        return indexer.compress_keys(raw_keys, block_tokens, positions)

    def _chunk_size(self, num_blocks: int, dtype: torch.dtype) -> int:
        score_per_query = max(num_blocks * 4, 1)
        gather_per_query = (
            2 * self.identity_limit * self.num_kv_heads * self.head_dim * dtype.itemsize
        )
        logits_per_query = self.num_heads * self.identity_limit * dtype.itemsize
        by_score = _PREFILL_SCORE_BYTES // score_per_query
        by_work = _PREFILL_WORK_BYTES // max(gather_per_query + logits_per_query, 1)
        return max(1, min(_PREFILL_CHUNK, by_score, by_work))

    def _prefill(
        self,
        md: QSASparseMetadata,
        layer_id: int,
        slot: int,
        q: torch.Tensor,
        index_q: torch.Tensor,
        indexer: Qwen4ExpQSAIndexer,
        batch: Batch,
    ) -> torch.Tensor:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        out = torch.empty_like(q)
        for req_idx, req in enumerate(reqs):
            start, end = qo[req_idx], qo[req_idx + 1]
            if start == end:
                continue
            kv_len = req.device_len
            rows = page_table[req.table_idx, :kv_len]
            block_keys = self._compressed_keys(slot, rows, kv_len, indexer)
            chunk = self._chunk_size(block_keys.shape[0], q.dtype)
            for query_start in range(start, end, chunk):
                query_end = min(query_start + chunk, end)
                local_start = query_start - start
                local_end = query_end - start
                scores = indexer.score_blocks(
                    index_q[query_start:query_end], block_keys
                )
                visible = torch.arange(
                    req.cached_len + local_start + 1,
                    req.cached_len + local_end + 1,
                    device=self.device,
                    dtype=torch.long,
                )
                selected = indexer.select_blocks_batched(scores, visible)
                physical = self._logical_to_physical(selected, rows)
                out[query_start:query_end] = self._attend_selected(
                    q[query_start:query_end], layer_id, physical
                )
        return out

    def _decode(
        self,
        md: QSASparseMetadata,
        layer_id: int,
        slot: int,
        q: torch.Tensor,
        index_q: torch.Tensor,
        indexer: Qwen4ExpQSAIndexer,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa_sparse import qsa_decode_scores

        assert md.rows is not None and md.kvlen is not None
        with decode_profile_range("FT.QSA.ScoreKernel"):
            scores = qsa_decode_scores(
                index_q,
                self.kvcache.index_k_cache(slot),
                md.rows,
                md.kvlen,
                indexer.k_layernorm.weight,
                compress_ratio=indexer.compress_ratio,
                rotary_dim=indexer.rotary_dim,
                rms_eps=indexer.k_layernorm.eps,
                rope_theta=indexer.rope_theta,
            )
        with decode_profile_range("FT.QSA.TopK"):
            selected = indexer.select_blocks_batched(scores, md.kvlen)
        with decode_profile_range("FT.QSA.LogicalToPhysical"):
            physical = self._logical_to_physical(selected, md.rows)
        if (
            getattr(self, "_direct_attend", True)
            and q.is_cuda
            and self.head_dim >= 16
            and not (self.head_dim & (self.head_dim - 1))
        ):
            with decode_profile_range("FT.QSA.DirectAttend"):
                return self._attend_selected_direct(q, layer_id, physical)
        return self._attend_selected(q, layer_id, physical)

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        self._stage_width = max_seq_len
        self._rows_buf = torch.full(
            (max_bs, self._stage_width),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._kvlen_buf = torch.zeros(max_bs, dtype=torch.int32, device=self.device)

    def _table_rows(self, batch: Batch) -> torch.Tensor:
        if getattr(batch, "active_table_idx", None) is not None:
            return batch.active_table_idx.to(torch.int64)
        # Unit tests and direct eager callers may not pass through Scheduler's
        # device-side active_table_idx staging. Keep this fallback out of capture.
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        return torch.tensor(
            [req.table_idx for req in reqs], dtype=torch.int64, device=self.device
        )

    def _decode_rows(self, batch: Batch, width: int) -> torch.Tensor:
        """Snapshot the live logical-row prefix once per eager decode step."""
        table = get_global_ctx().page_table[:, :width]
        return table.index_select(0, self._table_rows(batch))

    def _stage_decode(self, batch: Batch, bs: int, table_idx: torch.Tensor) -> None:
        """Restage replay addressing into persistent capture buffers."""
        with decode_profile_range("FT.Graph.QSAInputStaging"):
            assert self._rows_buf is not None and self._kvlen_buf is not None
            md = batch.attn_metadata
            assert isinstance(md, QSASparseMetadata)
            table = get_global_ctx().page_table
            assert table.shape[1] >= self._stage_width
            # Only current live columns can be read by the device-bounded score and
            # selected-row gathers. Copy that prefix into the fixed-width buffer;
            # stale columns are harmless and avoiding max_seq_len bytes per replay is
            # essential at the checkpoint's 262K context ceiling.
            live_width = min(int(md.kv_len_cpu.max()), self._stage_width)
            source = table[:, :live_width].index_select(0, table_idx)
            self._rows_buf[:bs, :live_width].copy_(source[:bs])
            self._kvlen_buf[:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
            md.rows = self._rows_buf[:bs]
            md.kvlen = self._kvlen_buf[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        assert batch.size in self.capture_bs
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        dummy_rows = torch.full(
            (batch.size,),
            batch.padded_reqs[0].table_idx,
            dtype=torch.int64,
            device=self.device,
        )
        self._stage_decode(batch, batch.size, dummy_rows)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, (
            "decode batch is missing its page-table rows"
        )
        self._stage_decode(
            batch, batch.padded_size, batch.active_table_idx.to(torch.int64)
        )

    def reset_capture(self) -> None:
        super().reset_capture()
        self.inner.reset_capture()
        self._rows_buf = None
        self._kvlen_buf = None
        self._stage_width = 0


__all__ = ["QSASparseAttnBackend", "QSASparseMetadata"]
