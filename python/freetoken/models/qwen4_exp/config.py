from __future__ import annotations

import os
from typing import Any

from freetoken.attention import AttnType
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import Qwen4ExpArgs


def _qsa_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN4_SPARSE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _mtp_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN4_MTP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_config(hf_config: Any) -> ModelConfig:
    text = hf_config.text_config
    layer_types = list(text.layer_types)
    mtp_num_hidden_layers = int(getattr(text, "mtp_num_hidden_layers", 0) or 0)
    mtp_enabled = _mtp_enabled() and mtp_num_hidden_layers > 0
    if mtp_enabled:
        if bool(getattr(text, "mtp_use_dedicated_embeddings", False)):
            raise ValueError("Qwen4-Exp MTP requires shared token embeddings")
        mtp_config = getattr(text, "mtp", None)
        mtp_layer_types = list(
            (mtp_config.get("layer_types") if isinstance(mtp_config, dict) else None)
            or ["full_attention"] * mtp_num_hidden_layers
        )
        if len(mtp_layer_types) != mtp_num_hidden_layers or any(
            layer_type not in {"full_attention", "qwen_sparse_attention"}
            for layer_type in mtp_layer_types
        ):
            raise ValueError(
                "Qwen4-Exp MTP currently requires full-attention predictor layers"
            )
    sparse_attention_types = {"full_attention", "qwen_sparse_attention"}
    unsupported = sorted(set(layer_types) - {"linear_attention", *sparse_attention_types})
    if unsupported:
        raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")

    head_dim = int(text.head_dim)
    rope = text.rope_parameters
    rotary_dim = round(head_dim * float(rope.get("partial_rotary_factor", 1.0)))
    indexer_budget = int(text.indexer_budget)
    use_sparse = _qsa_enabled()
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        # The sparse backend serves the checkpoint's full context. The escape
        # hatch retains the pre-QSA dense ceiling and backend behavior.
        max_position=(
            int(text.max_position_embeddings)
            if use_sparse
            else min(int(text.max_position_embeddings), indexer_budget)
        ),
        base=float(rope["rope_theta"]),
        scaling=None,
    )

    full_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type in sparse_attention_types
    )
    if mtp_enabled:
        full_ids += tuple(
            range(len(layer_types), len(layer_types) + mtp_num_hidden_layers)
        )
    linear_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    )
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    # The first experimental checkpoint predates the public Transformers config
    # fields for these three fixed QSA dimensions.  Retain its architectural
    # values as fallbacks while honoring explicit values in newer configs.
    indexer_n_heads_value = getattr(text, "indexer_n_heads", None)
    indexer_kv_heads_value = getattr(text, "indexer_kv_heads", None)
    indexer_head_dim_value = getattr(text, "indexer_head_dim", None)
    indexer_n_heads = int(4 if indexer_n_heads_value is None else indexer_n_heads_value)
    indexer_kv_heads = int(
        1 if indexer_kv_heads_value is None else indexer_kv_heads_value
    )
    indexer_head_dim = int(
        128 if indexer_head_dim_value is None else indexer_head_dim_value
    )
    indexer_compress_ratio = int(text.indexer_compress_ratio)
    if (
        min(indexer_n_heads, indexer_head_dim, indexer_budget, indexer_compress_ratio)
        <= 0
    ):
        raise ValueError(
            "Qwen4-Exp QSA dimensions, budget, and compression ratio must be positive"
        )
    if indexer_kv_heads != 1:
        raise ValueError(
            f"Qwen4-Exp QSA requires one indexer KV head, got {indexer_kv_heads}"
        )
    if indexer_budget % indexer_compress_ratio:
        raise ValueError(
            "Qwen4-Exp indexer_budget must be divisible by indexer_compress_ratio, "
            f"got {indexer_budget} and {indexer_compress_ratio}"
        )
    if rotary_dim > indexer_head_dim:
        raise ValueError(
            "Qwen4-Exp rotary dimensions must fit the indexer head, "
            f"got {rotary_dim} and {indexer_head_dim}"
        )
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(text.linear_num_key_heads),
            num_value_heads=int(text.linear_num_value_heads),
            key_head_dim=int(text.linear_key_head_dim),
            value_head_dim=int(text.linear_value_head_dim),
            conv_kernel_dim=int(text.linear_conv_kernel_dim),
            output_gate=True,
        ),
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
            index_head_dim=indexer_head_dim if use_sparse else 0,
            num_index_layers=len(full_ids) if use_sparse else 0,
            attn_type=AttnType.QSA if use_sparse else AttnType.FULL,
        ),
    )
    qwen4_args = Qwen4ExpArgs(
        use_sparse=use_sparse,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=tuple(int(layer_id) - 1 for layer_id in text.ple_layer_ids),
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        split_ngram_parts=int(text.split_ngram_parts),
        eos_token_id=int(eos_token_id),
        indexer_n_heads=indexer_n_heads,
        indexer_kv_heads=indexer_kv_heads,
        indexer_head_dim=indexer_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=indexer_compress_ratio,
        output_gate_type=str(text.output_gate_type or text.hidden_act),
        mtp_enabled=mtp_enabled,
        mtp_num_hidden_layers=mtp_num_hidden_layers if mtp_enabled else 0,
    )

    quant = hf_config.quantization_config
    if not isinstance(quant, dict):
        quant = quant.to_dict()
    method = str(quant.get("quant_method") or "").lower()
    algo = str(quant.get("quant_algo") or method).lower()
    if method == "fp8":
        block_size = tuple(int(value) for value in quant["weight_block_size"])
        if block_size != (128, 128):
            raise ValueError(f"Qwen4-Exp only supports 128x128 block-FP8, got {block_size}")
        expert_quant = "fp8_block"
    elif "fp4" in algo:
        # ModelOpt NVFP4 checkpoints use packed routed experts while leaving the
        # shared expert and other resident text weights in BF16.
        block_size = None
        expert_quant = "nvfp4"
    else:
        raise ValueError(
            "Qwen4-Exp requires a 128x128 block-FP8 or ModelOpt NVFP4 checkpoint, "
            f"got quant_method={method!r}, quant_algo={algo!r}"
        )

    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=int(text.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=block_size,
        # Only routed experts and PLE are quantized in the supported checkpoints.
        # Attention, hyper-connections, and shared-expert projections stay BF16.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        # PLE keeps per-request dilated-convolution state outside the generic
        # radix cache, so prefix snapshots remain unsupported. Its host embedding
        # lookup is staged into a stable device buffer before CUDA-graph replay.
        requires_naive_cache=True,
        # QSA prefill stays eager; decode stages fixed-shape addressing and keeps
        # score -> top-k -> compact gather inside the captured graph.
        supports_cuda_graph=True,
    )


__all__ = ["_mtp_enabled", "parse_config"]
