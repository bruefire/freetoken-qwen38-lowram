from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.layers import GemmaPlusOneRMSNorm
from freetoken.models.qwen4_exp.indexer import Qwen4ExpQSAIndexer

INDEX_HEADS = 4
INDEX_KV_HEADS = 1
INDEX_HEAD_DIM = 128
ROTARY_DIM = 64
COMPRESS_RATIO = 4
TOKEN_BUDGET = 2048


def _config(hidden_size: int = 32):
    return SimpleNamespace(
        hidden_size=hidden_size,
        rms_norm_eps=1e-6,
        rotary_config=SimpleNamespace(rotary_dim=ROTARY_DIM, base=10_000_000),
        qwen4_args=SimpleNamespace(
            indexer_n_heads=INDEX_HEADS,
            indexer_kv_heads=INDEX_KV_HEADS,
            indexer_head_dim=INDEX_HEAD_DIM,
            indexer_budget=TOKEN_BUDGET,
            indexer_compress_ratio=COMPRESS_RATIO,
        ),
    )


def _centered_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    input_dtype = hidden.dtype
    hidden_fp32 = hidden.float()
    variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
    result = hidden_fp32 * torch.rsqrt(variance + eps)
    result = (result * (1.0 + weight.float())).to(input_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _partial_rope(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int = ROTARY_DIM,
    theta: float = 10_000_000,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=hidden.device)
            / rotary_dim
        )
    )
    freqs = torch.outer(
        positions.to(device=hidden.device, dtype=torch.float32), inv_freq
    )
    angles = torch.cat((freqs, freqs), dim=-1)
    cos = angles.cos().to(hidden.dtype)
    sin = angles.sin().to(hidden.dtype)
    while cos.ndim < hidden.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotary, passthrough = hidden[..., :rotary_dim], hidden[..., rotary_dim:]
    half = rotary_dim // 2
    rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat((rotary * cos + rotated_half * sin, passthrough), dim=-1)


class _ReferenceCenteredRMSNorm(torch.nn.Module):
    """Qwen4ExpTextRMSNorm formula vendored from Transformers main.

    Reference revision: huggingface/transformers@155b89935a648278dd38c78184cbc40e6a65f14b.
    """

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return _centered_rms_norm(hidden, self.weight, self.eps)


class _ReferenceIndexer(torch.nn.Module):
    """Numerical seam of Transformers' Qwen4ExpTextQSAIndexer."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.index_qk_proj = torch.nn.Linear(
            hidden_size,
            (INDEX_HEADS + INDEX_KV_HEADS) * INDEX_HEAD_DIM,
            bias=False,
        )
        self.q_layernorm = _ReferenceCenteredRMSNorm(INDEX_HEAD_DIM, 1e-6)
        self.k_layernorm = _ReferenceCenteredRMSNorm(INDEX_HEAD_DIM, 1e-6)

    def project_queries(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [INDEX_HEADS * INDEX_HEAD_DIM, INDEX_KV_HEADS * INDEX_HEAD_DIM],
            dim=-1,
        )
        q = q.reshape(*hidden_states.shape[:-1], INDEX_HEADS, INDEX_HEAD_DIM)
        raw_keys = token_k.reshape(
            *hidden_states.shape[:-1], INDEX_KV_HEADS, INDEX_HEAD_DIM
        ).squeeze(-2)
        q = self.q_layernorm(q)
        return _partial_rope(q, positions), raw_keys

    def compress_keys(
        self,
        raw_keys: torch.Tensor,
        block_token_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        key_groups = raw_keys.index_select(0, block_token_indices.flatten())
        key_groups = key_groups.view(*block_token_indices.shape, INDEX_HEAD_DIM)
        pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys)
        block_starts = block_token_indices[:, 0]
        return _partial_rope(pooled_keys, positions.index_select(0, block_starts))

    @staticmethod
    def score_blocks(query: torch.Tensor, block_keys: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(
            query.float(), block_keys.float().transpose(-1, -2)
        ).transpose(-1, -2)
        return torch.relu(scores).sum(dim=-1) / math.sqrt(INDEX_HEAD_DIM)


def _new_pair(hidden_size: int = 32) -> tuple[Qwen4ExpQSAIndexer, _ReferenceIndexer]:
    torch.manual_seed(17)
    indexer = Qwen4ExpQSAIndexer(_config(hidden_size), layer_id=3)
    reference = _ReferenceIndexer(hidden_size)
    state_dict = {
        name: torch.randn_like(tensor) * 0.05
        for name, tensor in indexer.state_dict().items()
    }
    indexer.load_state_dict(dict(state_dict))
    reference.load_state_dict(state_dict, strict=True)

    # The production norm is GPU-kernel backed.  Inject the exact CPU formula so
    # these architecture tests remain runnable in CPU-only CI.
    indexer.q_layernorm.gemma_rmsnorm = _centered_rms_norm
    indexer.k_layernorm.gemma_rmsnorm = _centered_rms_norm
    return indexer, reference


def _reference_select_blocks(
    block_token_indices: torch.Tensor,
    scores: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    selected_blocks = scores.topk(
        min(TOKEN_BUDGET // COMPRESS_RATIO, scores.numel())
    ).indices
    selected_tokens = block_token_indices.index_select(0, selected_blocks).flatten()
    selected_tokens = torch.cat((selected_tokens, tail)).to(torch.int32)
    output = torch.full((TOKEN_BUDGET + COMPRESS_RATIO - 1,), -1, dtype=torch.int32)
    output = output.to(block_token_indices.device)
    output[: selected_tokens.numel()] = selected_tokens
    return output


def test_indexer_checkpoint_names_and_strict_load():
    indexer, _ = _new_pair()
    prefix = "model.layers.3.self_attn.indexer"

    assert set(indexer.state_dict(prefix=prefix)) == {
        f"{prefix}.index_qk_proj.weight",
        f"{prefix}.q_layernorm.weight",
        f"{prefix}.k_layernorm.weight",
    }
    assert indexer.index_qk_proj.weight.shape == (5 * INDEX_HEAD_DIM, 32)
    assert isinstance(indexer.q_layernorm, GemmaPlusOneRMSNorm)
    assert isinstance(indexer.k_layernorm, GemmaPlusOneRMSNorm)

    checkpoint = {
        name: torch.randn_like(tensor)
        for name, tensor in indexer.state_dict(prefix=prefix).items()
    }
    indexer.load_state_dict(dict(checkpoint), prefix=prefix)
    loaded = indexer.state_dict(prefix=prefix)
    for name, expected in checkpoint.items():
        torch.testing.assert_close(loaded[name], expected)


def test_projection_query_norm_and_partial_rope_match_transformers_reference():
    indexer, reference = _new_pair()
    hidden = torch.randn(7, 32)
    positions = torch.tensor([0, 1, 3, 4, 17, 63, 129])

    actual_q, actual_raw_keys = indexer.project_queries(hidden, positions)
    expected_q, expected_raw_keys = reference.project_queries(hidden, positions)

    assert actual_q.shape == (7, 4, 128)
    assert actual_raw_keys.shape == (7, 128)
    torch.testing.assert_close(actual_q, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_raw_keys, expected_raw_keys, rtol=0, atol=0)


def test_partial_rope_reuses_inv_freq_on_the_same_device():
    indexer, _ = _new_pair()
    hidden = torch.randn(2, INDEX_HEADS, INDEX_HEAD_DIM)
    positions = torch.tensor([1, 2])

    first = indexer.apply_partial_rope(hidden, positions)
    inv_freq = indexer._inv_freq_by_device[hidden.device]
    second = indexer.apply_partial_rope(hidden, positions)

    assert indexer._inv_freq_by_device[hidden.device].data_ptr() == inv_freq.data_ptr()
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_fp32_block_compression_and_scoring_match_transformers_reference():
    indexer, reference = _new_pair()
    raw_keys = (torch.randn(20, INDEX_HEAD_DIM) * 30).to(torch.bfloat16)
    block_token_indices = torch.tensor(
        [[0, 2, 4, 6], [7, 8, 10, 13], [14, 15, 17, 19]], dtype=torch.long
    )
    positions = torch.tensor(
        [0, 1, 4, 5, 9, 12, 13, 17, 18, 21, 25, 26, 30, 33, 40, 44, 48, 52, 57, 61]
    )
    query = torch.randn(INDEX_HEADS, INDEX_HEAD_DIM).to(torch.bfloat16)

    actual_keys = indexer.compress_keys(raw_keys, block_token_indices, positions)
    expected_keys = reference.compress_keys(raw_keys, block_token_indices, positions)
    actual_scores = indexer.score_blocks(query, actual_keys)
    expected_scores = reference.score_blocks(query, expected_keys)

    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("sequence_length", "tail_length"),
    [(2047, 3), (2048, 0), (2049, 1), (2050, 2), (2051, 3)],
)
def test_topk_boundary_and_incomplete_tail_match_transformers_reference(
    sequence_length: int, tail_length: int
):
    indexer, _ = _new_pair()
    visible = torch.arange(sequence_length)
    num_blocks = sequence_length // COMPRESS_RATIO
    complete_length = num_blocks * COMPRESS_RATIO
    blocks = visible[:complete_length].view(num_blocks, COMPRESS_RATIO)
    tail = visible[complete_length:]
    scores = torch.linspace(-1.0, 1.0, num_blocks)

    actual = indexer.select_blocks(blocks, scores, tail)
    expected = _reference_select_blocks(blocks, scores, tail)

    assert tail.numel() == tail_length
    assert actual.shape == (2051,)
    assert torch.equal(actual, expected)
    if tail_length:
        assert torch.equal(actual[actual >= 0][-tail_length:], tail.to(torch.int32))


@pytest.mark.parametrize(("current_score", "selected"), [(0.0, False), (2.0, True)])
def test_completed_current_block_competes_for_topk(
    current_score: float, selected: bool
):
    indexer, _ = _new_pair()
    # Query position 2051 completes block 512: unlike positions 2048..2050 it
    # is no longer an unconditional tail and must compete for the 512 slots.
    visible = torch.arange(2052)
    blocks = visible.view(513, COMPRESS_RATIO)
    tail = visible.new_empty(0)
    scores = torch.linspace(0.5, 1.5, 513)
    scores[-1] = current_score

    actual = indexer.select_blocks(blocks, scores, tail)
    expected = _reference_select_blocks(blocks, scores, tail)

    assert torch.equal(actual, expected)
    assert bool((actual == 2051).any()) is selected
    assert (actual >= 0).sum().item() == TOKEN_BUDGET


def test_batched_selection_matches_transformers_loop_at_all_tail_boundaries():
    indexer, _ = _new_pair()
    lengths = torch.tensor([2048, 2049, 2050, 2051, 2052])
    scores = torch.randn(lengths.numel(), 513)
    # At position 2051 the newly complete block participates in top-k selection.
    scores[-1, -1] = scores[-1].max() + 1

    actual = indexer.select_blocks_batched(scores, lengths)

    for row, length in enumerate(lengths.tolist()):
        visible = torch.arange(length)
        complete = length // COMPRESS_RATIO
        blocks = visible[: complete * COMPRESS_RATIO].view(-1, COMPRESS_RATIO)
        expected = _reference_select_blocks(
            blocks,
            scores[row, :complete],
            visible[complete * COMPRESS_RATIO :],
        )
        assert torch.equal(actual[row], expected)
    assert (actual[-1] == 2051).any()


class _TorchQSAKVCache:
    """Small row-flat cache double for backend/reference seam tests."""

    def __init__(
        self,
        capacity: int,
        kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.k = torch.zeros(capacity, kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.index_k = torch.zeros(capacity, INDEX_HEAD_DIM, device=device, dtype=dtype)

    def store_kv(self, k, v, out_loc, layer_id):
        rows = out_loc.long()
        self.k[rows] = k.view(k.shape[0], *self.k.shape[1:])
        self.v[rows] = v.view(v.shape[0], *self.v.shape[1:])

    def store_index_k(self, k, out_loc, slot):
        self.index_k[out_loc.long()] = k

    def k_cache(self, layer_id):
        return self.k

    def v_cache(self, layer_id):
        return self.v

    def index_k_cache(self, slot):
        return self.index_k


def _reference_selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    selected = selected[selected >= 0].long().sort().values
    key = k.index_select(0, selected).repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    value = v.index_select(0, selected).repeat_interleave(
        q.shape[1] // v.shape[1], dim=1
    )
    logits = torch.matmul(q.transpose(0, 1), key.permute(1, 2, 0)).transpose(
        0, 1
    ) / math.sqrt(q.shape[-1])
    probability = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probability.transpose(0, 1), value.transpose(0, 1)).transpose(
        0, 1
    )


def _reference_qsa_outputs(
    reference: _ReferenceIndexer,
    index_q: torch.Tensor,
    raw_index_k: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    visible_lengths: list[int],
) -> torch.Tensor:
    outputs = []
    positions = torch.arange(raw_index_k.shape[0], device=raw_index_k.device)
    for row, visible_length in enumerate(visible_lengths):
        visible = torch.arange(visible_length, device=raw_index_k.device)
        complete = visible_length // COMPRESS_RATIO
        blocks = visible[: complete * COMPRESS_RATIO].view(-1, COMPRESS_RATIO)
        block_keys = reference.compress_keys(raw_index_k, blocks, positions)
        scores = reference.score_blocks(index_q[row], block_keys)
        selected = _reference_select_blocks(
            blocks,
            scores,
            visible[complete * COMPRESS_RATIO :],
        )
        outputs.append(
            _reference_selected_attention(
                q[row : row + 1],
                k[:visible_length],
                v[:visible_length],
                selected,
            )
        )
    return torch.cat(outputs)


class _DenseIdentityBackend:
    def __init__(self, cache: _TorchQSAKVCache):
        self.cache = cache
        self.calls = 0

    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        self.calls += 1
        self.cache.store_kv(k, v, batch.out_loc, layer_id)
        outputs = []
        for row, req in enumerate(batch.reqs):
            outputs.append(
                _reference_selected_attention(
                    q[row : row + 1],
                    self.cache.k[: req.device_len],
                    self.cache.v[: req.device_len],
                    torch.arange(
                        req.device_len,
                        dtype=torch.int32,
                        device=self.cache.device,
                    ),
                )
            )
        return torch.cat(outputs)

    def reset_capture(self):
        pass


def _backend_for_test(cache: _TorchQSAKVCache) -> QSASparseAttnBackend:
    backend = object.__new__(QSASparseAttnBackend)
    backend.num_heads = 2
    backend.num_kv_heads = 1
    backend.head_dim = 8
    backend.groups_per_kv = 2
    backend.sm_scale = 8**-0.5
    backend.identity_limit = TOKEN_BUDGET + COMPRESS_RATIO - 1
    backend.kvcache = cache
    backend.device = cache.device
    backend._idx_slot = {0: 0}
    backend.inner = _DenseIdentityBackend(cache)
    return backend


def _qsa_step(
    monkeypatch,
    *,
    backend: QSASparseAttnBackend,
    indexer: Qwen4ExpQSAIndexer,
    reference: _ReferenceIndexer,
    hidden: torch.Tensor,
    raw_index_k: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    cached_len: int,
    device_len: int,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    extend_len = device_len - cached_len
    positions = torch.arange(cached_len, device_len, device=hidden.device)
    index_q, current_raw_k = indexer.project_queries(
        hidden[cached_len:device_len], positions
    )
    reference_q, reference_raw_k = reference.project_queries(
        hidden[cached_len:device_len], positions
    )
    torch.testing.assert_close(index_q, reference_q)
    torch.testing.assert_close(current_raw_k, reference_raw_k)

    q = torch.randn(extend_len, 2, 8, device=hidden.device, dtype=hidden.dtype)
    req = SimpleNamespace(
        table_idx=0,
        cached_len=cached_len,
        device_len=device_len,
        extend_len=extend_len,
    )
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        phase=phase,
        out_loc=torch.arange(
            cached_len, device_len, dtype=torch.int32, device=hidden.device
        ),
    )
    batch.attn_metadata = QSASparseMetadata(
        is_decode=phase == "decode",
        last_indices=torch.tensor(
            [extend_len - 1], dtype=torch.int32, device=hidden.device
        ),
        qo_indptr_cpu=torch.tensor([0, extend_len], dtype=torch.int32),
        kv_len_cpu=torch.tensor([device_len], dtype=torch.int32),
        inner=SimpleNamespace(),
    )
    context = SimpleNamespace(
        kv_cache=backend.kvcache,
        page_table=torch.arange(
            raw_index_k.shape[0], dtype=torch.int32, device=hidden.device
        ).view(1, -1),
    )
    monkeypatch.setattr(
        "freetoken.attention.qsa_sparse.get_global_ctx", lambda: context
    )

    actual = backend.qsa_forward(
        q,
        all_k[cached_len:device_len].flatten(1),
        all_v[cached_len:device_len].flatten(1),
        index_q,
        current_raw_k,
        indexer,
        0,
        batch,
    )
    expected = _reference_qsa_outputs(
        reference,
        reference_q,
        raw_index_k,
        q,
        all_k,
        all_v,
        list(range(cached_len + 1, device_len + 1)),
    )
    return actual, expected


def _pair_for_backend_reference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    indexer, reference = _new_pair()
    reference.to(device=device, dtype=dtype)
    indexer.index_qk_proj.weight = indexer.index_qk_proj.weight.to(
        device=device, dtype=dtype
    )
    indexer.q_layernorm.weight = indexer.q_layernorm.weight.to(
        device=device, dtype=dtype
    )
    indexer.k_layernorm.weight = indexer.k_layernorm.weight.to(
        device=device, dtype=dtype
    )
    return indexer, reference, device, dtype


@pytest.mark.parametrize("sequence_length", [2047, 2048, 2049, 2051, 2052, 4096])
def test_qsa_backend_boundary_output_matches_transformers_reference(
    monkeypatch, sequence_length: int
):
    torch.manual_seed(41 + sequence_length)
    indexer, reference, device, dtype = _pair_for_backend_reference()
    hidden = torch.randn(sequence_length, 32, device=device, dtype=dtype)
    # Project the cached key stream directly from the shared pinned state dict;
    # query normalization/RoPE still goes through both independent implementations.
    raw_index_k = F.linear(hidden, reference.index_qk_proj.weight[-INDEX_HEAD_DIM:])
    all_k = torch.randn(sequence_length, 1, 8, device=device, dtype=dtype)
    all_v = torch.randn_like(all_k)
    cache = _TorchQSAKVCache(sequence_length, 1, 8, device=device, dtype=dtype)
    cache.index_k[: sequence_length - 1] = raw_index_k[: sequence_length - 1]
    cache.k[: sequence_length - 1] = all_k[: sequence_length - 1]
    cache.v[: sequence_length - 1] = all_v[: sequence_length - 1]
    backend = _backend_for_test(cache)

    actual, expected = _qsa_step(
        monkeypatch,
        backend=backend,
        indexer=indexer,
        reference=reference,
        hidden=hidden,
        raw_index_k=raw_index_k,
        all_k=all_k,
        all_v=all_v,
        cached_len=sequence_length - 1,
        device_len=sequence_length,
        phase="prefill",
    )

    if sequence_length <= 2051:
        # The backend delegates the whole operation to FULL attention here;
        # selection is an identity mask and must not perturb one output bit.
        assert torch.equal(actual, expected)
    else:
        tolerance = 3e-2 if dtype is torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert backend.inner.calls == int(sequence_length <= 2051)


@pytest.mark.parametrize("sequence_length", [2047, 2051, 2052, 4096])
def test_qsa_fixed_decode_matches_transformers_reference(
    monkeypatch, sequence_length: int
):
    """Decode stays numerical across the identity boundary without host branching."""
    torch.manual_seed(73 + sequence_length)
    indexer, reference, device, dtype = _pair_for_backend_reference()
    hidden = torch.randn(sequence_length, 32, device=device, dtype=dtype)
    raw_index_k = F.linear(hidden, reference.index_qk_proj.weight[-INDEX_HEAD_DIM:])
    all_k = torch.randn(sequence_length, 1, 8, device=device, dtype=dtype)
    all_v = torch.randn_like(all_k)
    cache = _TorchQSAKVCache(sequence_length, 1, 8, device=device, dtype=dtype)
    cache.index_k[: sequence_length - 1] = raw_index_k[: sequence_length - 1]
    cache.k[: sequence_length - 1] = all_k[: sequence_length - 1]
    cache.v[: sequence_length - 1] = all_v[: sequence_length - 1]
    backend = _backend_for_test(cache)

    actual, expected = _qsa_step(
        monkeypatch,
        backend=backend,
        indexer=indexer,
        reference=reference,
        hidden=hidden,
        raw_index_k=raw_index_k,
        all_k=all_k,
        all_v=all_v,
        cached_len=sequence_length - 1,
        device_len=sequence_length,
        phase="decode",
    )

    tolerance = 3e-2 if dtype is torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert backend.inner.calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_direct_attention_matches_selected_gqa():
    from freetoken.kernel.triton.qsa_sparse import qsa_direct_attention

    torch.manual_seed(211)
    dtype = torch.bfloat16
    q = torch.randn(1, 24, 256, device="cuda", dtype=dtype)
    k = torch.randn(2049, 2, 256, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    rows = torch.full((1, 2051), -1, device="cuda", dtype=torch.int32)
    rows[:, :2049] = torch.arange(2049, device="cuda", dtype=torch.int32)

    actual = qsa_direct_attention(q, k, v, rows, softmax_scale=256**-0.5)

    gathered_k = k.index_select(0, rows[:, :2049].flatten().long()).view(1, 2049, 2, 256)
    gathered_v = v.index_select(0, rows[:, :2049].flatten().long()).view(1, 2049, 2, 256)
    grouped_q = q.view(1, 2, 12, 256)
    logits = torch.matmul(grouped_q, gathered_k.permute(0, 2, 3, 1)) * (256**-0.5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)
    expected = torch.matmul(probs, gathered_v.permute(0, 2, 1, 3)).reshape_as(q)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_graph_replay_restages_static_decode_buffers(monkeypatch):
    cache = _TorchQSAKVCache(16, 1, 8, device=torch.device("cpu"), dtype=torch.float32)
    backend = _backend_for_test(cache)
    backend._rows_buf = None
    backend._kvlen_buf = None
    backend._stage_width = 0
    backend.capture_bs = []
    page_table = torch.arange(24, dtype=torch.int32).view(3, 8)
    monkeypatch.setattr(
        "freetoken.attention.qsa_sparse.get_global_ctx",
        lambda: SimpleNamespace(page_table=page_table),
    )
    backend.init_capture_graph(max_seq_len=8, bs_list=[1, 2])

    md = QSASparseMetadata(
        is_decode=True,
        last_indices=torch.arange(2, dtype=torch.int32),
        qo_indptr_cpu=torch.arange(3, dtype=torch.int32),
        kv_len_cpu=torch.tensor([5, 7], dtype=torch.int32),
        inner=None,
    )
    batch = SimpleNamespace(
        attn_metadata=md,
        active_table_idx=torch.tensor([2, 1], dtype=torch.int32),
        padded_size=2,
    )
    backend.prepare_for_replay(batch)
    rows_ptr, kvlen_ptr = md.rows.data_ptr(), md.kvlen.data_ptr()
    torch.testing.assert_close(md.rows[:, :7], page_table[[2, 1], :7])
    assert (md.rows[:, 7:] == -1).all()
    torch.testing.assert_close(md.kvlen, torch.tensor([5, 7], dtype=torch.int32))

    page_table[0].add_(100)
    md.kv_len_cpu.copy_(torch.tensor([3, 8], dtype=torch.int32))
    batch.active_table_idx.copy_(torch.tensor([0, 2], dtype=torch.int32))
    backend.prepare_for_replay(batch)
    assert md.rows.data_ptr() == rows_ptr
    assert md.kvlen.data_ptr() == kvlen_ptr
    torch.testing.assert_close(md.rows, page_table[[0, 2]])
    torch.testing.assert_close(md.kvlen, torch.tensor([3, 8], dtype=torch.int32))


def test_qsa_prefill_then_decode_continuation_matches_transformers_reference(
    monkeypatch,
):
    torch.manual_seed(93)
    final_length = 2053
    indexer, reference, device, dtype = _pair_for_backend_reference()
    hidden = torch.randn(final_length, 32, device=device, dtype=dtype)
    raw_index_k = F.linear(hidden, reference.index_qk_proj.weight[-INDEX_HEAD_DIM:])
    all_k = torch.randn(final_length, 1, 8, device=device, dtype=dtype)
    all_v = torch.randn_like(all_k)
    cache = _TorchQSAKVCache(final_length, 1, 8, device=device, dtype=dtype)
    cache.index_k[:2048] = raw_index_k[:2048]
    cache.k[:2048] = all_k[:2048]
    cache.v[:2048] = all_v[:2048]
    backend = _backend_for_test(cache)

    prefill_actual, prefill_expected = _qsa_step(
        monkeypatch,
        backend=backend,
        indexer=indexer,
        reference=reference,
        hidden=hidden,
        raw_index_k=raw_index_k,
        all_k=all_k,
        all_v=all_v,
        cached_len=2048,
        device_len=2052,
        phase="prefill",
    )
    decode_actual, decode_expected = _qsa_step(
        monkeypatch,
        backend=backend,
        indexer=indexer,
        reference=reference,
        hidden=hidden,
        raw_index_k=raw_index_k,
        all_k=all_k,
        all_v=all_v,
        cached_len=2052,
        device_len=2053,
        phase="decode",
    )

    tolerance = 3e-2 if dtype is torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        prefill_actual, prefill_expected, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        decode_actual, decode_expected, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(cache.index_k[:final_length], raw_index_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graph capture")
def test_qsa_decode_cuda_graph_matches_eager_across_identity_boundary():
    """One captured shape must follow device kv_len on both sides of 2051."""
    torch.manual_seed(151)
    batch_size = 2
    stage_width = 4096
    indexer, _, device, dtype = _pair_for_backend_reference()
    cache = _TorchQSAKVCache(stage_width, 1, 8, device=device, dtype=dtype)
    cache.index_k.copy_(torch.randn_like(cache.index_k))
    cache.k.copy_(torch.randn_like(cache.k))
    cache.v.copy_(torch.randn_like(cache.v))
    backend = _backend_for_test(cache)

    q = torch.randn(batch_size, 2, 8, device=device, dtype=dtype)
    index_q = torch.randn(
        batch_size, INDEX_HEADS, INDEX_HEAD_DIM, device=device, dtype=dtype
    )
    rows = torch.arange(stage_width, dtype=torch.int32, device=device).repeat(
        batch_size, 1
    )
    kvlen = torch.tensor([2051, 2047], dtype=torch.int32, device=device)
    md = QSASparseMetadata(
        is_decode=True,
        last_indices=torch.arange(batch_size, dtype=torch.int32, device=device),
        qo_indptr_cpu=torch.arange(batch_size + 1, dtype=torch.int32),
        kv_len_cpu=torch.tensor([2051, 2047], dtype=torch.int32),
        inner=SimpleNamespace(),
        rows=rows,
        kvlen=kvlen,
    )

    def run():
        return backend._decode(md, 0, 0, q, index_q, indexer)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    captured = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.copy_(run())

    for lengths in ((2047, 2051), (2052, stage_width), (stage_width, 2047)):
        kvlen.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize()
        graph_output = captured.clone()
        eager_output = run()
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
