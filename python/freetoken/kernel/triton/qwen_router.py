"""Single-launch softmax + top-k router for Qwen's top-10 MoE path.

Unlike the GPT-OSS router, Qwen with ``renormalize=False`` applies softmax over
*all* experts before selecting the routes.  The selected-expert softmax used by
``renormalize=True`` is kept as a constexpr branch in the same kernel.  Qwen's
router logits are fp16/bf16, so their raw 16-bit value plus a 16-bit expert-id
tie break fit in uint32.  Triton's partial bitonic top-16 then selects top-10
without the old uint64 full sort over all 512 experts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qwen_fused_topk_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_token_non_padded_ptr,
    stride_logits_t,
    E: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_NUM_TOKEN_NON_PADDED: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E
    logits_raw = tl.load(
        logits_ptr + token_id * stride_logits_t + offs_e,
        mask=valid,
        other=-float("inf"),
    )
    logits = logits_raw.to(tl.float32)

    # FloatFlip maps fp16/bf16 bit patterns to monotonically ordered uint16
    # keys.  Normalize signed zero so numerical ties are resolved solely by the
    # smaller expert id, matching stable descending argsort.  Canonicalize NaN
    # above +inf as torch.topk does; router logits should not normally contain it.
    logit_bits = logits_raw.to(tl.uint16, bitcast=True)
    logit_bits = tl.where(logits == 0.0, 0, logit_bits)
    sign = logit_bits & 0x8000
    value_key = logit_bits ^ tl.where(sign != 0, 0xFFFF, 0x8000)
    value_key = tl.where(logits != logits, 0xFFFF, value_key)
    # Larger packed values mean better routes; BLOCK_E-id makes exact-value
    # ties prefer the smaller expert id. Invalid padded lanes stay below -inf.
    id_key = BLOCK_E - offs_e
    packed = (value_key.to(tl.uint32) << 16) | id_key.to(tl.uint32)
    packed = tl.where(valid, packed, 0)
    selected_packed = tl.topk(packed, K_PAD)
    selected_ids = (BLOCK_E - (selected_packed & 0xFFFF)).to(tl.int32)
    # When K_PAD exceeds E the tail lanes select padded slots whose id decodes
    # to BLOCK_E; they are dropped at the store, but the load still has to stay
    # inside the row.
    selected_logits = tl.load(
        logits_ptr + token_id * stride_logits_t + selected_ids,
        mask=selected_ids < E,
        other=-float("inf"),
    ).to(tl.float32)

    offs_k = tl.arange(0, K_PAD)
    top_mask = offs_k < K
    if RENORMALIZE:
        # Standard Qwen3.5 path: normalize among selected experts.
        normalizer_logits = tl.where(top_mask, selected_logits, -float("inf"))
        max_logit = tl.max(normalizer_logits, axis=0)
        numerator = tl.where(top_mask, tl.exp(selected_logits - max_logit), 0.0)
        denominator = tl.sum(numerator, axis=0)
        weights = numerator / denominator
    else:
        # Qwen4 path: softmax over every expert, then retain the top-k weights.
        max_logit = tl.max(tl.where(valid, logits, -float("inf")), axis=0)
        all_exp = tl.where(valid, tl.exp(logits - max_logit), 0.0)
        denominator = tl.sum(all_exp, axis=0)
        weights = tl.exp(selected_logits - max_logit) / denominator

    if HAS_NUM_TOKEN_NON_PADDED:
        row_is_valid = token_id < tl.load(num_token_non_padded_ptr)
        selected_ids = tl.where(row_is_valid, selected_ids, -1)

    out_off = token_id * K + offs_k
    tl.store(topk_weights_ptr + out_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + out_off, selected_ids, mask=top_mask)


def qwen_fused_topk(
    logits: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 weights and int32 ids with one Triton launch."""
    assert logits.is_cuda and logits.dim() == 2
    tokens, num_experts = logits.shape
    assert logits.dtype in (torch.float16, torch.bfloat16)
    assert logits.stride(-1) == 1
    assert 0 < topk <= num_experts <= 1024
    if num_token_non_padded is not None:
        assert num_token_non_padded.is_cuda and num_token_non_padded.numel() == 1

    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=logits.device
    )
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device=logits.device)
    if tokens == 0:
        return topk_weights, topk_ids

    block_e = triton.next_power_of_2(num_experts)
    k_pad = triton.next_power_of_2(topk)
    _qwen_fused_topk_kernel[(tokens,)](
        logits,
        topk_weights,
        topk_ids,
        num_token_non_padded if num_token_non_padded is not None else topk_ids,
        logits.stride(0),
        E=num_experts,
        K=topk,
        K_PAD=k_pad,
        BLOCK_E=block_e,
        RENORMALIZE=renormalize,
        HAS_NUM_TOKEN_NON_PADDED=num_token_non_padded is not None,
        num_warps=8 if block_e >= 512 else 4,
    )
    return topk_weights, topk_ids


__all__ = ["qwen_fused_topk"]
