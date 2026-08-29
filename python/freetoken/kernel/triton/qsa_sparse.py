"""CUDA-graph-safe Qwen4 QSA decode scoring.

The QSA cache stores one raw index key per token.  Decode needs to pool every
complete four-token logical block, apply the indexer's centered RMSNorm and
partial RoPE, then score it against the four query heads.  Materializing the
fixed-width raw-key gather would cost ``[B, max_seq_len, index_dim]``; this
kernel instead follows a staged page-table snapshot and performs that work in
place, emitting only ``[B, max_blocks]`` fp32 scores.

The launch grid depends on staged tensor shapes only.  Each program reads the
live number of complete blocks from ``kv_len`` in device memory, which makes
the same captured graph valid before and after the 2051-token identity/QSA
boundary.  Columns outside the live length stay ``-inf`` in the caller-owned
output and therefore cannot enter top-k.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_chunks"])
def _qsa_decode_score_kernel(
    q_ptr,  # [batch, index_heads, index_dim]
    ik_ptr,  # raw index-key slab [physical_rows, index_dim]
    rows_ptr,  # staged logical -> physical rows [batch, stage_width]
    kv_len_ptr,  # device live lengths [batch]
    weight_ptr,  # centered k RMSNorm weight [index_dim]
    score_ptr,  # pre-filled -inf [batch, max_blocks]
    index_heads: tl.constexpr,
    index_dim: tl.constexpr,
    compress_ratio: tl.constexpr,
    rotary_dim: tl.constexpr,
    rms_eps: tl.constexpr,
    rope_log2: tl.constexpr,
    score_scale: tl.constexpr,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_ik_r,
    stride_ik_d,
    stride_rows_b,
    stride_score_b,
    num_chunks,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    kv_len = tl.maximum(tl.load(kv_len_ptr + pid_b), 0)
    num_blocks = kv_len // compress_ratio
    chunk_blocks = (num_blocks + num_chunks - 1) // num_chunks
    block_start = pid_c * chunk_blocks
    block_end = tl.minimum(block_start + chunk_blocks, num_blocks)
    if block_start >= block_end:
        return

    off_h = tl.arange(0, BLOCK_H)
    off_d = tl.arange(0, BLOCK_D)
    off_r = tl.arange(0, BLOCK_R)
    h_mask = off_h < index_heads
    d_mask = off_d < index_dim
    r_mask = off_r < compress_ratio

    query = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + off_h[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + off_d, mask=d_mask, other=0.0).to(tl.float32)

    half_rotary: tl.constexpr = rotary_dim // 2
    # NeoX rotary pairs the two halves.  Re-load the paired raw-key columns below;
    # this avoids ever materializing the pooled key and keeps the kernel's only
    # global output to one scalar per block.
    paired_d = tl.where(
        off_d < half_rotary,
        off_d + half_rotary,
        tl.where(off_d < rotary_dim, off_d - half_rotary, off_d),
    )
    paired_weight = tl.load(weight_ptr + paired_d, mask=d_mask, other=0.0).to(
        tl.float32
    )

    for block_id in tl.range(block_start, block_end):
        logical = block_id * compress_ratio + off_r
        physical = tl.load(
            rows_ptr + pid_b * stride_rows_b + logical,
            mask=r_mask,
            other=0,
        ).to(tl.int64)
        physical = tl.maximum(physical, 0)

        raw = tl.load(
            ik_ptr + physical[:, None] * stride_ik_r + off_d[None, :] * stride_ik_d,
            mask=r_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        pooled = tl.sum(raw, axis=0) / compress_ratio
        # Transformers casts the fp32 mean back to the raw-key dtype before the
        # centered RMSNorm.  Preserve that rounding seam exactly.
        pooled = pooled.to(ik_ptr.dtype.element_ty).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / index_dim + rms_eps)
        normed = (pooled * inv_rms * (1.0 + weight)).to(ik_ptr.dtype.element_ty)

        paired_raw = tl.load(
            ik_ptr + physical[:, None] * stride_ik_r + paired_d[None, :] * stride_ik_d,
            mask=r_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        paired_pooled = tl.sum(paired_raw, axis=0) / compress_ratio
        paired_pooled = paired_pooled.to(ik_ptr.dtype.element_ty).to(tl.float32)
        paired_normed = (paired_pooled * inv_rms * (1.0 + paired_weight)).to(
            ik_ptr.dtype.element_ty
        )

        freq_index = off_d % half_rotary
        inv_freq = tl.exp2(-(2.0 * freq_index / rotary_dim) * rope_log2)
        angle = block_id * compress_ratio * inv_freq
        cos = tl.cos(angle).to(ik_ptr.dtype.element_ty)
        sin = tl.sin(angle).to(ik_ptr.dtype.element_ty)
        rotated = (
            normed * cos
            + tl.where(off_d < half_rotary, -paired_normed, paired_normed) * sin
        )
        block_key = tl.where(off_d < rotary_dim, rotated, normed)
        # Partial RoPE is also an input-dtype operation before score_blocks casts
        # both operands to fp32.
        block_key = block_key.to(ik_ptr.dtype.element_ty).to(tl.float32)

        per_head = tl.sum(query * block_key[None, :], axis=1)
        score = tl.sum(tl.maximum(per_head, 0.0), axis=0) * score_scale
        tl.store(score_ptr + pid_b * stride_score_b + block_id, score)


def _torch_qsa_decode_scores(
    index_q: torch.Tensor,
    raw_index_keys: torch.Tensor,
    rows: torch.Tensor,
    kv_len: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    compress_ratio: int,
    rotary_dim: int,
    rms_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    """Device-agnostic reference/fallback with the same fixed-shape contract."""
    batch, _, index_dim = index_q.shape
    num_blocks = rows.shape[1] // compress_ratio
    logical_width = num_blocks * compress_ratio
    physical = rows[:, :logical_width].clamp_min(0).long()
    raw = raw_index_keys.index_select(0, physical.flatten()).view(
        batch, num_blocks, compress_ratio, index_dim
    )
    pooled = raw.float().mean(dim=2).to(raw_index_keys.dtype)
    pooled_fp32 = pooled.float()
    inv_rms = torch.rsqrt(pooled_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
    normed = (pooled_fp32 * inv_rms * (1.0 + k_norm_weight.float())).to(
        raw_index_keys.dtype
    )

    positions = torch.arange(
        0,
        logical_width,
        compress_ratio,
        dtype=torch.float32,
        device=index_q.device,
    )
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=index_q.device)
            / rotary_dim
        )
    )
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(normed.dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(normed.dtype)
    rotary = normed[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    block_keys = torch.cat(
        (
            rotary * cos.unsqueeze(0) + rotated_half * sin.unsqueeze(0),
            normed[..., rotary_dim:],
        ),
        dim=-1,
    )
    scores = torch.matmul(
        index_q.float(), block_keys.float().transpose(-1, -2)
    ).relu_().sum(dim=-2) / math.sqrt(index_dim)
    complete = torch.div(kv_len.long(), compress_ratio, rounding_mode="floor")
    block_ids = torch.arange(num_blocks, device=index_q.device)
    return scores.masked_fill(
        block_ids.unsqueeze(0) >= complete.unsqueeze(1), float("-inf")
    )


@torch.no_grad()
def qsa_decode_scores(
    index_q: torch.Tensor,
    raw_index_keys: torch.Tensor,
    rows: torch.Tensor,
    kv_len: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    compress_ratio: int,
    rotary_dim: int,
    rms_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    """Return fixed-width QSA block scores, masked by device-read live lengths."""
    if index_q.device.type != "cuda":
        return _torch_qsa_decode_scores(
            index_q,
            raw_index_keys,
            rows,
            kv_len,
            k_norm_weight,
            compress_ratio=compress_ratio,
            rotary_dim=rotary_dim,
            rms_eps=rms_eps,
            rope_theta=rope_theta,
        )

    batch, index_heads, index_dim = index_q.shape
    num_blocks = rows.shape[1] // compress_ratio
    score = torch.full(
        (batch, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=index_q.device,
    )
    # Four-token pooling is cheap relative to M3's 128-token block score, so use
    # a wider fixed grid to keep long contexts from serializing in a few CTAs.
    target_grid = 8192
    max_chunks = 256
    target = max(1, min(max_chunks, target_grid // max(batch, 1)))
    num_chunks = 1 << (target.bit_length() - 1)
    _qsa_decode_score_kernel[(batch, num_chunks)](
        index_q,
        raw_index_keys,
        rows,
        kv_len,
        k_norm_weight,
        score,
        index_heads,
        index_dim,
        compress_ratio,
        rotary_dim,
        rms_eps,
        math.log2(rope_theta),
        index_dim**-0.5,
        index_q.stride(0),
        index_q.stride(1),
        index_q.stride(2),
        raw_index_keys.stride(0),
        raw_index_keys.stride(1),
        rows.stride(0),
        score.stride(0),
        num_chunks,
        BLOCK_H=triton.next_power_of_2(index_heads),
        BLOCK_D=triton.next_power_of_2(index_dim),
        BLOCK_R=triton.next_power_of_2(compress_ratio),
        num_warps=4,
        num_stages=2,
    )
    return score


@triton.jit
def _qsa_direct_gqa_splitk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rows_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    stride_rows_row,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_rows,
    TOPK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        physical_row = tl.load(
            rows_ptr + row * stride_rows_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        valid = (physical_row >= 0) & (physical_row < num_cache_rows)
        safe_row = tl.maximum(physical_row, 0).to(tl.int64)
        keys = tl.load(
            k_ptr
            + safe_row[None, :] * stride_k_row
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_ptr
            + safe_row[:, None] * stride_v_row
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys) * SOFTMAX_SCALE_LOG2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def qsa_direct_attention(
    q: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    physical_rows: torch.Tensor,
    *,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA without materializing selected K/V rows."""
    if q.ndim != 3 or k_rows.ndim != 3 or v_rows.shape != k_rows.shape:
        raise ValueError("QSA direct attention received invalid Q/K/V shapes")
    if physical_rows.ndim != 2 or physical_rows.shape[0] != q.shape[0]:
        raise ValueError("QSA direct attention needs one row list per query")
    if physical_rows.shape[1] <= 0:
        raise ValueError("QSA direct attention needs a positive selection width")
    if q.shape[2] != k_rows.shape[2] or q.shape[1] % k_rows.shape[1]:
        raise ValueError("QSA direct attention needs valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_rows.dtype == v_rows.dtype
    assert physical_rows.dtype == torch.int32
    assert q.stride(2) == k_rows.stride(2) == v_rows.stride(2) == 1
    assert physical_rows.stride(1) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_rows.shape[1]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_rows.shape[1]
    small_profile_limit = 8 if block_m <= 8 else 4
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(physical_rows.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    _qsa_direct_gqa_splitk_kernel[(q.shape[0], k_rows.shape[1], num_splits)](
        q,
        k_rows,
        v_rows,
        physical_rows,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_rows.stride(0),
        k_rows.stride(1),
        v_rows.stride(0),
        v_rows.stride(1),
        physical_rows.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_rows.shape[0],
        TOPK=physical_rows.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SOFTMAX_SCALE_LOG2=softmax_scale * 1.4426950408889634,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["qsa_decode_scores", "qsa_direct_attention"]
