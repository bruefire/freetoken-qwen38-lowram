"""Fused elementwise stages around Qwen HyperConnection GEMVs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_div_kernel(x_ptr, out_ptr, n_elements, divisor: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # Match eager's materialized division as closely as possible: round to the
    # projection dtype before applying SiLU.
    divided = (x / divisor).to(x_ptr.dtype.element_ty).to(tl.float32)
    out = divided * tl.sigmoid(divided)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _mix_reduce_kernel(
    mix_ptr,
    normalized_ptr,
    out_ptr,
    M,
    stride_mix_m,
    stride_norm_m,
    stride_out_m,
    D: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (m < M) & (offsets < D)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h in tl.static_range(HC):
        mix = tl.load(
            mix_ptr + m * stride_mix_m + h * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normalized = tl.load(
            normalized_ptr + m * stride_norm_m + h * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Eager materializes both sigmoid and multiply in the projection dtype.
        gate = tl.sigmoid(mix).to(mix_ptr.dtype.element_ty).to(tl.float32)
        product = (gate * normalized).to(mix_ptr.dtype.element_ty).to(tl.float32)
        acc += product
    tl.store(out_ptr + m * stride_out_m + offsets, acc / HC, mask=mask)


@triton.jit
def _inject_kernel(
    logits_ptr,
    out_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    divided = (logits / divisor).to(logits_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.sigmoid(divided).to(logits_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + offsets, 2.0 * gate, mask=mask)


@triton.jit
def _weighted_residual_kernel(
    residual_ptr,
    mixed_ptr,
    weights_ptr,
    out_ptr,
    M,
    stride_residual_m,
    stride_mixed_m,
    stride_weights_m,
    stride_out_m,
    D: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = (m < M) & (offsets < D)
    residual = tl.load(
        residual_ptr + m * stride_residual_m + h * D + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mixed = tl.load(
        mixed_ptr + m * stride_mixed_m + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    weight = tl.load(weights_ptr + m * stride_weights_m + h).to(tl.float32)
    # Preserve eager's rounded multiply before the residual add.
    contribution = (mixed * weight).to(residual_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + m * stride_out_m + h * D + offsets,
        residual + contribution,
        mask=mask,
    )


def hyper_silu_div(x: torch.Tensor, divisor: int) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    out = torch.empty_like(x)
    if x.numel():
        _silu_div_kernel[(triton.cdiv(x.numel(), 256),)](
            x, out, x.numel(), divisor=divisor, BLOCK=256, num_warps=4
        )
    return out


def hyper_mix_reduce(
    mix_logits: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    assert mix_logits.is_cuda and normalized.is_cuda
    assert mix_logits.shape == normalized.shape
    assert mix_logits.dim() == 2 and mix_logits.stride(-1) == 1
    assert normalized.stride(-1) == 1
    assert mix_logits.shape[1] == hc_count * hidden_size
    tokens = mix_logits.shape[0]
    out = torch.empty(
        (tokens, hidden_size), dtype=normalized.dtype, device=normalized.device
    )
    if tokens:
        _mix_reduce_kernel[(tokens, triton.cdiv(hidden_size, 256))](
            mix_logits,
            normalized,
            out,
            tokens,
            mix_logits.stride(0),
            normalized.stride(0),
            out.stride(0),
            D=hidden_size,
            HC=hc_count,
            BLOCK=256,
            num_warps=4,
        )
    return out


def hyper_inject(logits: torch.Tensor, divisor: int) -> torch.Tensor:
    assert logits.is_cuda and logits.is_contiguous()
    out = torch.empty_like(logits)
    if logits.numel():
        _inject_kernel[(triton.cdiv(logits.numel(), 128),)](
            logits, out, logits.numel(), divisor=divisor, BLOCK=128, num_warps=4
        )
    return out


def hyper_weighted_residual(
    residual: torch.Tensor,
    mixed: torch.Tensor,
    weights: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    assert residual.is_cuda and mixed.is_cuda and weights.is_cuda
    assert residual.dim() == mixed.dim() == weights.dim() == 2
    assert residual.shape == (mixed.shape[0], hc_count * hidden_size)
    assert mixed.shape[1] == hidden_size
    assert weights.shape == (mixed.shape[0], hc_count)
    assert residual.stride(-1) == mixed.stride(-1) == weights.stride(-1) == 1
    out = torch.empty_like(residual)
    tokens = residual.shape[0]
    if tokens:
        _weighted_residual_kernel[
            (tokens, hc_count, triton.cdiv(hidden_size, 256))
        ](
            residual,
            mixed,
            weights,
            out,
            tokens,
            residual.stride(0),
            mixed.stride(0),
            weights.stride(0),
            out.stride(0),
            D=hidden_size,
            HC=hc_count,
            BLOCK=256,
            num_warps=4,
        )
    return out


__all__ = [
    "hyper_inject",
    "hyper_mix_reduce",
    "hyper_silu_div",
    "hyper_weighted_residual",
]
