from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import GemmaPlusOneRMSNorm
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.utils import decode_profile_range, nvtx_annotate

from .indexer import Qwen4ExpQSAIndexer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpAttention(Qwen3_5Attention):
    """Qwen4 gated GQA plus its checkpoint-resident QSA indexer."""

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        # Qwen4 stores centered norm weights: effective scale is ``1 + weight``.
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(config, layer_id)
        self.use_sparse = config.qwen4_args.use_sparse

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        with decode_profile_range("FT.QSA.MainProjection"):
            q, k, v, gate = self._project(x)
        if self.use_sparse:
            with decode_profile_range("FT.QSA.IndexerProjection"):
                index_q, raw_index_k = self.indexer.project_queries(x, ctx.batch.positions)
            with decode_profile_range("FT.QSA.SparseBackend"):
                out = ctx.attn_backend.qsa_forward(
                    q,
                    k,
                    v,
                    index_q,
                    raw_index_k,
                    self.indexer,
                    self.layer_id,
                    ctx.batch,
                )
        else:
            out = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        with decode_profile_range("FT.QSA.OutputProjection"):
            return self._combine(out, gate)


__all__ = ["Qwen4ExpAttention"]
