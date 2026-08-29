from __future__ import annotations

import mmap
from types import SimpleNamespace

import pytest
import torch
from freetoken.attention import AttnType
from freetoken.core import HostInputIds
from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import (
    _GatedResidual,
    _HostNGramEmbedding,
    _MTPModule,
    _ple_mmap_advice,
    _ple_row_cache_bytes,
    _PLELayer,
    _preload_ple_enabled,
    _tokens_for_ngram_forward,
    build_ngram_ids,
)
from freetoken.models.qwen4_exp.weight import _rename, _try_fuse
from freetoken.models.register import get_model_spec


def _config():
    text = SimpleNamespace(
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        head_dim=256,
        rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10_000_000},
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        max_position_embeddings=262_144,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        eos_token_id=248044,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        split_ngram_parts=128,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        hidden_act="silu",
        num_hidden_layers=4,
        num_attention_heads=24,
        hidden_size=2560,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        norm_topk_prob=None,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(
        text_config=text,
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        image_token_id=248056,
    )


def test_qwen4_config_uses_sparse_qsa_by_default(monkeypatch):
    monkeypatch.delenv("FREETOKEN_QWEN4_SPARSE", raising=False)
    config = parse_config(_config())
    assert config.rotary_config.max_position == 262_144
    assert config.expert_quant == "fp8_block"
    assert config.attn_quant == "none"
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.output_gate_type == "sigmoid"
    assert (
        config.qwen4_args.indexer_n_heads,
        config.qwen4_args.indexer_kv_heads,
        config.qwen4_args.indexer_head_dim,
    ) == (4, 1, 128)
    assert config.requires_naive_cache
    assert config.qwen4_args.use_sparse
    assert config.attn_type_for_layer(3) is AttnType.QSA
    assert config.supports_cuda_graph
    assert config.is_linear_layer(0)
    assert not config.is_linear_layer(3)


def test_qwen4_dense_escape_hatch_restores_2048_limit(monkeypatch):
    monkeypatch.setenv("FREETOKEN_QWEN4_SPARSE", "0")

    config = parse_config(_config())

    assert config.rotary_config.max_position == 2048
    assert not config.qwen4_args.use_sparse
    assert config.attn_type_for_layer(3) is AttnType.FULL
    assert config.supports_cuda_graph


def test_qwen4_attention_routes_qsa_inputs_to_sparse_backend(monkeypatch):
    calls = []

    class Backend:
        def qsa_forward(self, *args):
            calls.append(args)
            return torch.full((1, 2, 64), 7.0)

    attention = object.__new__(Qwen4ExpAttention)
    attention.use_sparse = True
    attention.layer_id = 3
    attention._project = lambda hidden: (
        torch.zeros(1, 2, 64),
        torch.zeros(1, 64),
        torch.zeros(1, 64),
        torch.zeros(1, 128),
    )
    attention._combine = lambda output, gate: output
    attention.indexer = SimpleNamespace(
        project_queries=lambda hidden, positions: (
            torch.zeros(1, 4, 128),
            torch.zeros(1, 128),
        )
    )
    batch = SimpleNamespace(positions=torch.tensor([0]))
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.attention.get_global_ctx",
        lambda: SimpleNamespace(batch=batch, attn_backend=Backend()),
    )

    output = attention.forward(torch.zeros(1, 32))

    assert output.shape == (1, 2, 64)
    assert calls and calls[0][-2] == 3 and calls[0][-1] is batch


def test_qwen4_config_accepts_transformers_sparse_attention_alias():
    hf_config = _config()
    hf_config.text_config.layer_types[-1] = "qwen_sparse_attention"
    config = parse_config(hf_config)
    assert not config.is_linear_layer(3)


def test_qwen4_config_accepts_modelopt_nvfp4_experts():
    hf_config = _config()
    hf_config.quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }

    config = parse_config(hf_config)

    assert config.expert_quant == "nvfp4"
    assert config.weight_block_size is None
    assert config.attn_quant == "none"
    assert config.dense_quant == "none"


def test_qwen4_mtp_is_opt_in_and_adds_one_qsa_cache_layer(monkeypatch):
    hf_config = _config()
    hf_config.text_config.mtp_num_hidden_layers = 1
    hf_config.text_config.mtp_use_dedicated_embeddings = False
    hf_config.text_config.mtp = {"layer_types": ["full_attention"]}

    monkeypatch.delenv("FREETOKEN_QWEN4_MTP", raising=False)
    disabled = parse_config(hf_config)
    assert not disabled.qwen4_args.mtp_enabled

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP", "1")
    enabled = parse_config(hf_config)
    assert enabled.qwen4_args.mtp_enabled
    assert enabled.qwen4_args.mtp_num_hidden_layers == 1
    assert enabled.attn_type_for_layer(enabled.num_layers) is AttnType.QSA


def test_qwen4_mtp_weight_names_are_loaded_only_when_enabled(monkeypatch):
    name = "mtp.fc_hidden.weight"
    monkeypatch.delenv("FREETOKEN_QWEN4_MTP", raising=False)
    assert _rename(name) is None
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP", "1")
    assert _rename(name) == name


def test_qwen4_mtp_input_fusion_preserves_hyper_connection_streams():
    module = object.__new__(_MTPModule)
    module.hc_count = 2
    module.hidden_size = 3
    module.pre_fc_norm_embedding = SimpleNamespace(forward=lambda value: value + 1)
    module.pre_fc_norm_hidden = SimpleNamespace(forward=lambda value: value + 2)
    module.fc_embedding = SimpleNamespace(forward=lambda value: value * 10)
    module.fc_hidden = SimpleNamespace(forward=lambda value: value * 3)
    expanded = torch.arange(12, dtype=torch.float32).view(2, 6)
    embeddings = torch.arange(6, dtype=torch.float32).view(2, 3)

    actual = module.fuse_inputs(expanded, embeddings)

    hidden = ((expanded + 2).view(2, 2, 3) * 3)
    expected = (hidden + ((embeddings + 1) * 10).unsqueeze(1)).flatten(1)
    torch.testing.assert_close(actual, expected)


def test_qwen4_registry_entry():
    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_weight_names():
    assert _rename("model.language_model.layers.1.ple.key_proj.weight") == (
        "model.layers.1.ple.key_proj.weight"
    )
    assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
    assert _rename("model.language_model.layers.3.self_attn.indexer.q_layernorm.weight") == (
        "model.layers.3.self_attn.indexer.q_layernorm.weight"
    )


def test_qwen4_projection_fusion_order():
    buffers = {}
    base = "model.layers.3.self_attn."
    parts = [
        ("q_proj.weight", torch.full((2, 3), 1.0)),
        ("k_proj.weight", torch.full((1, 3), 2.0)),
        ("v_proj.weight", torch.full((1, 3), 3.0)),
    ]
    assert _try_fuse(base + parts[0][0], parts[0][1], buffers) == ()
    assert _try_fuse(base + parts[1][0], parts[1][1], buffers) == ()
    name, fused = _try_fuse(base + parts[2][0], parts[2][1], buffers)
    assert name == base + "qkv_proj.weight"
    assert fused[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0]


def test_qwen4_hyper_connection_projection_fusion_padding():
    buffers = {}
    base = "model.layers.3.attn_hyper_connection."
    down = torch.full((5, 3), 1.0)
    inject = torch.full((2, 3), 2.0)

    assert _try_fuse(base + "input_mix_weight_down.weight", down, buffers) == ()
    name, fused = _try_fuse(base + "block_inject_weight.weight", inject, buffers)

    assert name == base + "input_mix_weight_down_block_inject.weight"
    assert fused.shape == (16, 3)
    assert fused[:5].eq(1).all()
    assert fused[5:7].eq(2).all()
    assert fused[7:].eq(0).all()


def test_qwen4_hyper_connection_merged_weight_reproduces_two_projections():
    config = SimpleNamespace(
        hidden_size=8,
        rms_norm_eps=1e-6,
        qwen4_args=SimpleNamespace(hc_count=4, hc_lowrank=5),
    )
    residual = _GatedResidual(config)
    torch.manual_seed(19)
    residual.input_mix_weight_down_block_inject.weight.normal_()
    normalized = torch.randn(3, 32)

    residual._projection_fused = False
    split_down, split_inject = residual._down(normalized)
    residual._projection_fused = True
    fused_down, fused_inject = residual._down(normalized)

    torch.testing.assert_close(fused_down, split_down)
    torch.testing.assert_close(fused_inject, split_inject)


def test_ngram_hash_resets_at_eos():
    tokens = torch.tensor([4, 5, 99, 6, 7])
    multipliers = torch.tensor([3, 5, 7])
    sizes = torch.tensor([101, 103])
    offsets = torch.tensor([0, 101])
    ids = build_ngram_ids(
        tokens,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=multipliers,
        vocab_sizes=sizes,
        offsets=offsets,
    )
    assert ids.shape == (5, 2)
    expected_bigram_after_eos = (6 * 3) ^ (99 * 5)
    assert ids[3, 0].item() == expected_bigram_after_eos % 101


def test_ngram_history_includes_inflight_overlap_token():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6]), device_len=4)

    tokens = _tokens_for_ngram_forward(req, torch.tensor([7], device="cpu"))

    assert tokens.tolist() == [4, 5, 6, 7]
    assert req.input_ids.tolist() == [4, 5, 6]


def test_ngram_history_does_not_duplicate_drained_token():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6, 7]), device_len=4)

    tokens = _tokens_for_ngram_forward(req, torch.tensor([7], device="cpu"))

    assert tokens.tolist() == [4, 5, 6, 7]


def test_ngram_history_can_select_only_required_suffix():
    req = SimpleNamespace(input_ids=torch.tensor([4, 5, 6]), device_len=5)

    tokens = _tokens_for_ngram_forward(
        req,
        torch.tensor([7, 8], device="cpu"),
        start=2,
    )

    assert tokens.tolist() == [6, 7, 8]
    assert _tokens_for_ngram_forward(req, torch.tensor([7, 8]), start=4).tolist() == [8]


def test_incremental_ngram_hash_matches_full_history():
    tokens = torch.tensor([10, 99, 4, 5, 6, 7])
    kwargs = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "eos_token_id": 99,
        "multipliers": torch.tensor([3, 5, 7]),
        "vocab_sizes": torch.tensor([101, 103]),
        "offsets": torch.tensor([0, 101]),
    }

    full = build_ngram_ids(tokens, **kwargs)
    history_start = 3
    incremental = build_ngram_ids(tokens[history_start:], **kwargs)

    assert torch.equal(incremental[2], full[5])


def test_ple_graph_convolution_matches_eager(monkeypatch):
    def make_layer():
        layer = object.__new__(_PLELayer)
        layer.hidden_size = 2
        layer.hc_count = 2
        layer.state_len = 1
        layer.dilation = 1
        layer.conv1d = SimpleNamespace(weight=torch.randn(4, 1, 2))
        layer._conv_state_pool = None
        layer._mtp_prefix_conv_state = None
        return layer

    req = SimpleNamespace(table_idx=1, extend_len=1, cached_len=0)
    linear_pool = SimpleNamespace(conv_states=torch.empty(1, 3, 1, 1))
    hidden = torch.randn(1, 4)

    eager = make_layer()
    eager_batch = SimpleNamespace(
        is_decode=True,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=False,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    context = SimpleNamespace(batch=eager_batch, linear_state_pool=linear_pool)
    monkeypatch.setattr("freetoken.models.qwen4_exp.model.get_global_ctx", lambda: context)
    eager_output = eager._short_conv(hidden)

    graph = make_layer()
    graph.conv1d.weight.copy_(eager.conv1d.weight)
    context.batch = SimpleNamespace(
        is_decode=True,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=True,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    graph_output = graph._short_conv(hidden)

    torch.testing.assert_close(graph_output, eager_output)
    torch.testing.assert_close(graph._conv_state_pool, eager._conv_state_pool)


def test_ple_mtp_graph_convolution_matches_two_token_eager(monkeypatch):
    def make_layer():
        layer = object.__new__(_PLELayer)
        layer.hidden_size = 2
        layer.hc_count = 2
        layer.state_len = 1
        layer.dilation = 1
        layer.conv1d = SimpleNamespace(weight=torch.randn(4, 1, 2))
        layer._conv_state_pool = None
        layer._mtp_prefix_conv_state = None
        return layer

    req = SimpleNamespace(table_idx=1, extend_len=2, cached_len=0)
    linear_pool = SimpleNamespace(conv_states=torch.empty(1, 3, 1, 1))
    hidden = torch.randn(2, 4)

    eager = make_layer()
    eager_batch = SimpleNamespace(
        is_decode=False,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=False,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    context = SimpleNamespace(batch=eager_batch, linear_state_pool=linear_pool)
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx", lambda: context
    )
    eager_output = eager._short_conv(hidden)

    graph = make_layer()
    graph.conv1d.weight.copy_(eager.conv1d.weight)
    context.batch = SimpleNamespace(
        is_decode=False,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=True,
        mtp_verify=True,
        mtp_checkpoint_capacity=1,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    graph_output = graph._short_conv(hidden)

    torch.testing.assert_close(graph_output, eager_output)
    torch.testing.assert_close(graph._conv_state_pool, eager._conv_state_pool)
    expected_prefix = hidden[0].unsqueeze(-1)
    torch.testing.assert_close(graph._mtp_prefix_conv_state[0], expected_prefix)
    graph._conv_state_pool[1].fill_(-1)
    graph.commit_mtp_prefix_state(1, 0)
    torch.testing.assert_close(graph._conv_state_pool[1], expected_prefix)


def test_ple_mtp_graph_saves_each_three_token_prefix(monkeypatch):
    layer = object.__new__(_PLELayer)
    layer.hidden_size = 2
    layer.hc_count = 1
    layer.state_len = 1
    layer.dilation = 1
    layer.conv1d = SimpleNamespace(weight=torch.randn(2, 1, 2))
    layer._conv_state_pool = None
    layer._mtp_prefix_conv_state = None
    req = SimpleNamespace(table_idx=1, extend_len=3, cached_len=0)
    batch = SimpleNamespace(
        is_decode=False,
        reqs=[req],
        padded_reqs=[req],
        cuda_graph_capture=True,
        mtp_verify=True,
        mtp_checkpoint_capacity=2,
        linear_table_idx=torch.tensor([1], dtype=torch.int32),
    )
    context = SimpleNamespace(
        batch=batch,
        linear_state_pool=SimpleNamespace(conv_states=torch.empty(1, 3, 1, 1)),
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx", lambda: context
    )
    hidden = torch.randn(3, 2)

    layer._short_conv(hidden)

    torch.testing.assert_close(
        layer._mtp_prefix_conv_state[0], hidden[0].unsqueeze(-1)
    )
    torch.testing.assert_close(
        layer._mtp_prefix_conv_state[1], hidden[1].unsqueeze(-1)
    )
    layer._conv_state_pool[1].fill_(-1)
    layer.commit_mtp_prefix_state(1, 1)
    torch.testing.assert_close(layer._conv_state_pool[1], hidden[1].unsqueeze(-1))


def test_mtp_capture_preallocates_largest_verify_buffer(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "2")
    model = object.__new__(Qwen4ExpForCausalLM)
    model.mtp = object()
    captured = []
    model.model = SimpleNamespace(
        prepare_cuda_graph_capture=lambda token_count: captured.append(token_count)
    )
    batch = SimpleNamespace(input_ids=torch.zeros(1, dtype=torch.int32))

    model.prepare_cuda_graph_capture(batch)

    assert captured == [3]


def test_qwen4_ple_preload_is_opt_in(monkeypatch):
    monkeypatch.delenv("FREETOKEN_QWEN4_PLE_PRELOAD", raising=False)
    assert not _preload_ple_enabled()

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_PRELOAD", "true")
    assert _preload_ple_enabled()


def test_ple_row_cache_budget_is_opt_in_and_validated(monkeypatch):
    monkeypatch.delenv("FREETOKEN_QWEN4_PLE_CACHE_GIB", raising=False)
    assert _ple_row_cache_bytes() == 0

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_CACHE_GIB", "0.5")
    assert _ple_row_cache_bytes() == 1 << 29

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_CACHE_GIB", "-1")
    with pytest.raises(ValueError, match="non-negative finite"):
        _ple_row_cache_bytes()


def test_ple_mmap_advice_is_configurable(monkeypatch):
    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_MMAP_ADVICE", "normal")
    assert _ple_mmap_advice() == "normal"

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_MMAP_ADVICE", "random-willneed")
    assert _ple_mmap_advice() == "random-willneed"

    monkeypatch.setenv("FREETOKEN_QWEN4_PLE_MMAP_ADVICE", "sequential")
    with pytest.raises(ValueError, match="must be one of"):
        _ple_mmap_advice()


def _small_host_embedding() -> _HostNGramEmbedding:
    embedding = object.__new__(_HostNGramEmbedding)
    embedding.layer_id = 1
    embedding.head_dim = 4
    embedding._shards = [
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint8),
        torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]], dtype=torch.uint8),
    ]
    embedding._shard_ends_list = (2, 4)
    embedding._shard_starts_list = (0, 2)
    embedding._row_cache = None
    embedding._row_cache_tags = None
    embedding._handles = []
    embedding._mmap_advice = "normal"
    embedding._mmap_prefetch = False
    embedding._cache_hits = 0
    embedding._cache_misses = 0
    embedding._decode_source_rows = 0
    embedding._prefill_source_rows = 0
    embedding._decode_major_faults = 0
    embedding._prefill_major_faults = 0
    embedding._output_buffer = None
    return embedding


def test_ple_row_cache_is_bit_exact_and_reports_hits():
    embedding = _small_host_embedding()
    # Two rows plus their int64 tags. Global rows 0 and 3 map to different slots.
    embedding._init_row_cache(2 * (embedding.head_dim + 8))
    output = torch.empty(3, embedding.head_dim, dtype=torch.uint8)

    embedding._copy_decode_rows(torch.tensor([0, 3, 0]), output)

    expected = torch.stack(
        [embedding._shards[0][0], embedding._shards[1][1], embedding._shards[0][0]]
    )
    assert torch.equal(output, expected)
    stats = embedding.cache_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert stats["cache_hit_rate"] == pytest.approx(1 / 3)
    assert stats["decode_source_rows"] == 2


def test_ple_decode_prefetches_source_rows(monkeypatch):
    embedding = _small_host_embedding()
    embedding._mmap_prefetch = True
    prefetched = []
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model._prefetch_tensor_rows",
        lambda rows: prefetched.extend(rows),
    )
    output = torch.empty(2, embedding.head_dim, dtype=torch.uint8)

    embedding._copy_decode_rows(torch.tensor([0, 3]), output)

    assert len(prefetched) == 2
    assert torch.equal(output[0], embedding._shards[0][0])
    assert torch.equal(output[1], embedding._shards[1][1])


def test_ple_prefill_prefetches_grouped_source_rows(monkeypatch):
    embedding = _small_host_embedding()
    embedding._mmap_prefetch = True
    prefetched = []
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model._prefetch_tensor_indices",
        lambda tensor, indices: prefetched.append((tensor, indices)),
    )
    output = torch.empty(3, embedding.head_dim, dtype=torch.uint8)

    embedding._copy_prefill_rows(torch.tensor([0, 3, 1]), output)

    assert len(prefetched) == 2
    assert prefetched[0][1] == [0, 1]
    assert prefetched[1][1] == [1]
    assert torch.equal(output[0], embedding._shards[0][0])
    assert torch.equal(output[1], embedding._shards[1][1])
    assert torch.equal(output[2], embedding._shards[0][1])


def test_ple_mmap_advice_switch_enables_decode_prefetch(monkeypatch):
    embedding = _small_host_embedding()
    embedding._handles = [object()]
    advised = []
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model._madvise_tensor_ranges",
        lambda tensors, advice: advised.append((tensors, advice)),
    )

    embedding.set_mmap_advice("random-willneed")

    assert advised == [(embedding._shards, mmap.MADV_RANDOM)]
    assert embedding._mmap_advice == "random-willneed"
    assert embedding._mmap_prefetch


def test_ple_pinned_output_buffer_is_reused(monkeypatch):
    embedding = _small_host_embedding()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    first = embedding._ensure_output_buffer(16)
    smaller = embedding._ensure_output_buffer(8)

    assert first.data_ptr() == smaller.data_ptr()
    assert smaller.shape == (8, embedding.head_dim)


def test_ple_decode_uses_sampler_host_token_and_waits_once(monkeypatch):
    class Ready:
        calls = 0

        def synchronize(self):
            self.calls += 1

    embedding = _small_host_embedding()
    embedding.ngram_size = 3
    embedding.heads_per_ngram = 1
    embedding.eos_token_id = 99
    embedding._host_constants = (
        torch.tensor([3, 5, 7]),
        torch.tensor([101, 103]),
        torch.tensor([0, 101]),
    )
    req = SimpleNamespace(
        input_ids=torch.tensor([4, 5, 6], dtype=torch.int32),
        cached_len=3,
        device_len=4,
        extend_len=1,
    )
    ready = Ready()
    batch = SimpleNamespace(
        is_decode=True,
        padded_reqs=[req],
        input_ids=torch.tensor([123], dtype=torch.int32),
        host_input_ids=HostInputIds(
            [torch.tensor([7], dtype=torch.int32)], ready
        ),
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=batch),
    )

    actual = embedding._current_ngram_ids()
    expected = build_ngram_ids(
        torch.tensor([5, 6, 7]),
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=embedding._host_constants[0],
        vocab_sizes=embedding._host_constants[1],
        offsets=embedding._host_constants[2],
    )[-1:]

    assert torch.equal(actual, expected)
    assert ready.calls == 1
