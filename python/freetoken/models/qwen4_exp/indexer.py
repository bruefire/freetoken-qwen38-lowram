from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpQSAIndexer(BaseOP):
    """Checkpoint-resident Qwen Sparse Attention block indexer."""

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.qwen4_args
        self.layer_id = layer_id
        self.index_n_heads = args.indexer_n_heads
        self.index_kv_heads = args.indexer_kv_heads
        self.index_head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.selection_size = self.token_budget + self.compress_ratio - 1
        self.rotary_dim = config.rotary_config.rotary_dim
        self.rope_theta = config.rotary_config.base
        # Underscore-prefixed runtime cache is intentionally excluded from
        # BaseOP.state_dict.  Values are built once per device with the same
        # expression used previously on every token.
        self._inv_freq_by_device: dict[torch.device, torch.Tensor] = {}

        if self.index_kv_heads != 1:
            raise ValueError(
                f"Qwen4-Exp QSA requires one indexer KV head, got {self.index_kv_heads}"
            )
        if self.rotary_dim > self.index_head_dim:
            raise ValueError(
                f"QSA rotary dim {self.rotary_dim} exceeds head dim {self.index_head_dim}"
            )

        output_size = (self.index_n_heads + self.index_kv_heads) * self.index_head_dim
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, output_size, has_bias=False
        )
        # Qwen4 stores scale - 1 in both norm weights.  Keep the checkpoint
        # values centered and perform the fp32 +1 at runtime.
        self.q_layernorm = GemmaPlusOneRMSNorm(self.index_head_dim, config.rms_norm_eps)
        self.k_layernorm = GemmaPlusOneRMSNorm(self.index_head_dim, config.rms_norm_eps)

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unnormalized query heads and the single raw token-key stream."""
        qk = self.index_qk_proj.forward(hidden_states)
        query_size = self.index_n_heads * self.index_head_dim
        q, raw_keys = torch.split(
            qk,
            [query_size, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q = q.reshape(
            *hidden_states.shape[:-1], self.index_n_heads, self.index_head_dim
        )
        raw_keys = raw_keys.reshape(
            *hidden_states.shape[:-1], self.index_kv_heads, self.index_head_dim
        ).squeeze(-2)
        return q, raw_keys

    def apply_partial_rope(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply the model's NeoX partial RoPE to indexer query or key heads."""
        if positions.ndim != 1 or positions.shape[0] != hidden.shape[0]:
            raise ValueError(
                "QSA positions must be a vector matching the leading hidden dimension, "
                f"got {tuple(positions.shape)} and {tuple(hidden.shape)}"
            )
        inv_freq = self._inv_freq_by_device.get(hidden.device)
        if inv_freq is None:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0,
                        self.rotary_dim,
                        2,
                        dtype=torch.float32,
                        device=hidden.device,
                    )
                    / self.rotary_dim
                )
            )
            self._inv_freq_by_device[hidden.device] = inv_freq
        freqs = torch.outer(
            positions.to(device=hidden.device, dtype=torch.float32), inv_freq
        )
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos().to(hidden.dtype)
        sin = angles.sin().to(hidden.dtype)
        while cos.ndim < hidden.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)

        rotary, passthrough = (
            hidden[..., : self.rotary_dim],
            hidden[..., self.rotary_dim :],
        )
        half = self.rotary_dim // 2
        rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
        rotary = rotary * cos + rotated_half * sin
        return torch.cat((rotary, passthrough), dim=-1)

    def project_queries(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states, then normalize and rotate only the query stream."""
        q, raw_keys = self.project(hidden_states)
        q = self.q_layernorm.forward(q.contiguous())
        return self.apply_partial_rope(q, positions), raw_keys

    def compress_keys(
        self,
        raw_keys: torch.Tensor,
        block_token_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compress complete token blocks and rotate them at each block's first token.

        Averaging intentionally happens in fp32 before casting back for the
        centered RMSNorm, matching Transformers' Qwen4-Exp implementation.
        """
        if (
            block_token_indices.ndim != 2
            or block_token_indices.shape[-1] != self.compress_ratio
        ):
            raise ValueError(
                f"QSA blocks must have shape [N, {self.compress_ratio}], "
                f"got {tuple(block_token_indices.shape)}"
            )
        if block_token_indices.numel() == 0:
            return raw_keys.new_empty((0, self.index_head_dim))
        key_groups = raw_keys.index_select(0, block_token_indices.flatten())
        key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
        pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm.forward(pooled_keys)
        block_positions = positions.index_select(0, block_token_indices[:, 0])
        return self.apply_partial_rope(pooled_keys, block_positions)

    def score_blocks(
        self, query: torch.Tensor, block_keys: torch.Tensor
    ) -> torch.Tensor:
        """Sum ``relu(q @ k)`` over query heads and scale by sqrt(head dim)."""
        head_scores = torch.matmul(query.float(), block_keys.float().transpose(-1, -2))
        return F.relu(head_scores).sum(dim=-2) / math.sqrt(self.index_head_dim)

    def select_blocks(
        self,
        block_token_indices: torch.Tensor,
        scores: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Select top-scoring complete blocks and append the uncompressed tail."""
        if scores.ndim != 1 or scores.shape[0] != block_token_indices.shape[0]:
            raise ValueError(
                "QSA scores must contain one value per complete block, "
                f"got {tuple(scores.shape)} for {tuple(block_token_indices.shape)}"
            )
        if block_token_indices.shape[0]:
            selected_blocks = scores.topk(
                min(self.block_topk, block_token_indices.shape[0]), dim=0
            ).indices
            selected_tokens = block_token_indices.index_select(
                0, selected_blocks
            ).flatten()
        else:
            selected_tokens = tail.new_empty(0)
        selected_tokens = torch.cat((selected_tokens, tail)).to(torch.int32)
        if selected_tokens.numel() > self.selection_size:
            raise RuntimeError(
                f"QSA selected {selected_tokens.numel()} tokens into {self.selection_size} slots"
            )
        output = torch.full(
            (self.selection_size,),
            -1,
            dtype=torch.int32,
            device=block_token_indices.device,
        )
        output[: selected_tokens.numel()] = selected_tokens
        return output

    def select_blocks_batched(
        self,
        scores: torch.Tensor,
        visible_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized causal block selection for a query chunk.

        ``scores`` may include compressed blocks produced from tokens later than
        a query. ``visible_lengths`` masks those blocks, expands each selected
        block back to four token positions, and appends that query's incomplete
        0..3-token tail. The result uses the same fixed -1-padded 2051-slot
        representation as :meth:`select_blocks`.
        """
        if scores.ndim != 2 or visible_lengths.ndim != 1:
            raise ValueError(
                "QSA batched selection expects [queries, blocks] scores and "
                f"[queries] lengths, got {tuple(scores.shape)} and "
                f"{tuple(visible_lengths.shape)}"
            )
        if scores.shape[0] != visible_lengths.shape[0]:
            raise ValueError("QSA score and visible-length query counts differ")

        visible_lengths = visible_lengths.to(device=scores.device, dtype=torch.long)
        complete = torch.div(visible_lengths, self.compress_ratio, rounding_mode="floor")
        num_blocks = scores.shape[1]
        num_picks = min(self.block_topk, num_blocks)
        if num_picks:
            block_ids = torch.arange(num_blocks, device=scores.device)
            masked_scores = scores.masked_fill(
                block_ids.unsqueeze(0) >= complete.unsqueeze(1), float("-inf")
            )
            picks = masked_scores.topk(num_picks, dim=-1).indices
            live_picks = picks < complete.unsqueeze(1)
            offsets = torch.arange(self.compress_ratio, device=scores.device)
            block_tokens = picks.unsqueeze(-1) * self.compress_ratio + offsets
            block_tokens = torch.where(
                live_picks.unsqueeze(-1), block_tokens, torch.full_like(block_tokens, -1)
            ).flatten(1)
        else:
            block_tokens = torch.empty(
                scores.shape[0], 0, dtype=torch.long, device=scores.device
            )

        tail_offsets = torch.arange(self.compress_ratio - 1, device=scores.device)
        tail_lengths = torch.remainder(visible_lengths, self.compress_ratio)
        tail = complete.unsqueeze(1) * self.compress_ratio + tail_offsets
        tail = torch.where(
            tail_offsets.unsqueeze(0) < tail_lengths.unsqueeze(1),
            tail,
            torch.full_like(tail, -1),
        )
        selected = torch.cat((block_tokens, tail), dim=-1).to(torch.int32)
        output = torch.full(
            (scores.shape[0], self.selection_size),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
        output[:, : selected.shape[1]] = selected
        return output

    def select_token_indices(
        self,
        query: torch.Tensor,
        raw_keys: torch.Tensor,
        visible_token_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Select tokens for one query using complete blocks plus a 0..ratio-1 tail."""
        num_complete_blocks = visible_token_indices.numel() // self.compress_ratio
        complete_length = num_complete_blocks * self.compress_ratio
        block_token_indices = visible_token_indices[:complete_length].view(
            num_complete_blocks, self.compress_ratio
        )
        tail = visible_token_indices[complete_length:]
        block_keys = self.compress_keys(raw_keys, block_token_indices, positions)
        scores = self.score_blocks(query, block_keys)
        return self.select_blocks(block_token_indices, scores, tail)


__all__ = ["Qwen4ExpQSAIndexer"]
