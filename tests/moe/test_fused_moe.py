import sys
from types import SimpleNamespace

import pytest
import torch


def _activation_and_mul(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    raise AssertionError(f"unsupported test activation {activation}")


def _reference_fused_experts_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    for token_idx in range(hidden_states.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            route_weight = topk_weights[token_idx, route_idx].float()
            gate_up = torch.matmul(w1_fp32[expert_id], hidden_fp32[token_idx])
            if apply_router_weight_on_input:
                gate_up = gate_up * route_weight
            activated = _activation_and_mul(gate_up, activation)
            contribution = torch.matmul(w2_fp32[expert_id], activated)
            if not apply_router_weight_on_input:
                contribution = contribution * route_weight
            output[token_idx] += contribution
    return output.to(hidden_states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_accepts_triton_kernel_tuple_output():
    from freetoken.moe.fused import fused_topk

    logits = torch.tensor(
        [[4.0, 1.0, -1.0, 2.0], [0.5, 3.0, 2.0, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states = torch.zeros((2, 8), device="cuda", dtype=torch.bfloat16)

    weights, ids = fused_topk(hidden_states, logits, topk=2, renormalize=True)

    ref_logits, ref_ids = torch.topk(logits.float(), 2, dim=-1)
    ref_weights = torch.softmax(ref_logits, dim=-1)
    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-4)


def test_fused_topk_pads_non_power_of_two_for_triton(monkeypatch):
    import freetoken.moe.fused as fused

    monkeypatch.setattr(
        "freetoken.kernel.backend.is_triton_kernels_installed",
        lambda: True,
    )
    called = {}

    def fake_topk(logits, topk, *, apply_softmax):
        called["topk"] = topk
        values, ids = torch.topk(logits, topk, dim=-1)
        if apply_softmax:
            values = torch.softmax(values, dim=-1)
        # triton_kernels stores a SparseMatrix in ascending expert-id order.
        ids, order = torch.sort(ids, dim=-1)
        values = torch.gather(values, 1, order)
        return SimpleNamespace(vals=values, indx=ids)

    monkeypatch.setitem(sys.modules, "triton_kernels", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "triton_kernels.topk", SimpleNamespace(topk=fake_topk))
    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    hidden_states = torch.zeros((2, 8))

    weights, ids = fused.fused_topk(
        hidden_states,
        logits,
        topk=10,
        renormalize=True,
    )

    assert called["topk"] == 16
    ref_weights, ref_ids = torch.topk(torch.softmax(logits, dim=-1), 10, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, ref_weights)
    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    assert weights.is_contiguous()


def test_qwen_topk_dispatch_is_independently_disableable(monkeypatch):
    import freetoken.moe.fused as fused

    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    hidden_states = torch.zeros((2, 8))
    sentinel = (torch.ones((2, 10)), torch.zeros((2, 10), dtype=torch.int32))
    called = {}

    def fake_qwen_topk(logits_arg, topk, renormalize, num_token_non_padded):
        called["args"] = (logits_arg, topk, renormalize, num_token_non_padded)
        return sentinel

    monkeypatch.setattr(fused, "_should_use_qwen_fused_router", lambda *_: True)
    monkeypatch.setitem(
        sys.modules,
        "freetoken.kernel.triton.qwen_router",
        SimpleNamespace(qwen_fused_topk=fake_qwen_topk),
    )
    actual = fused.fused_topk(hidden_states, logits, topk=10, renormalize=False)
    assert actual is sentinel
    assert called["args"] == (logits, 10, False, None)

    monkeypatch.delenv("FREETOKEN_QWEN_FUSED_ROUTER", raising=False)
    assert fused._qwen_fused_router_enabled()
    monkeypatch.setenv("FREETOKEN_QWEN_FUSED_ROUTER", "0")
    assert not fused._qwen_fused_router_enabled()


def test_qwen_topk_auto_dispatch_includes_batch_one(monkeypatch):
    import freetoken.moe.fused as fused

    class FakeCudaLogits:
        is_cuda = True
        dtype = torch.bfloat16

        def __init__(self, tokens):
            self.shape = (tokens, 512)

        def dim(self):
            return 2

        def stride(self, dim):
            return (512, 1)[dim]

    monkeypatch.delenv("FREETOKEN_QWEN_FUSED_ROUTER", raising=False)
    monkeypatch.delenv("FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS", raising=False)
    assert fused._should_use_qwen_fused_router(FakeCudaLogits(1), 10)
    assert fused._should_use_qwen_fused_router(FakeCudaLogits(2), 10)

    monkeypatch.setenv("FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS", "2")
    assert not fused._should_use_qwen_fused_router(FakeCudaLogits(1), 10)
    monkeypatch.setenv("FREETOKEN_QWEN_FUSED_ROUTER", "0")
    assert not fused._should_use_qwen_fused_router(FakeCudaLogits(2), 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 7, 33])
@pytest.mark.parametrize("renormalize", [False, True])
def test_qwen_one_kernel_topk_matches_torch_and_legacy_router(
    monkeypatch, renormalize, batch_size
):
    import freetoken.moe.fused as fused
    from freetoken.kernel.triton.qwen_router import qwen_fused_topk

    torch.manual_seed(1234)
    logits = torch.randn((batch_size, 512), device="cuda", dtype=torch.bfloat16)
    logits += torch.arange(512, device="cuda", dtype=torch.bfloat16) * 1e-3
    weights, ids = qwen_fused_topk(logits, 10, renormalize)

    ref_ids = torch.argsort(
        logits.float(), dim=-1, descending=True, stable=True
    )[:, :10]
    top_logits = torch.gather(logits.float(), 1, ref_ids)
    if renormalize:
        ref_weights = torch.softmax(top_logits, dim=-1)
    else:
        ref_weights = torch.gather(
            torch.softmax(logits.float(), dim=-1), 1, ref_ids
        )
    assert torch.equal(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-6)

    monkeypatch.setenv("FREETOKEN_QWEN_FUSED_ROUTER", "0")
    hidden_states = torch.zeros((logits.shape[0], 8), device="cuda")
    legacy_weights, legacy_ids = fused.fused_topk(
        hidden_states, logits, topk=10, renormalize=renormalize
    )
    assert torch.equal(ids, legacy_ids)
    torch.testing.assert_close(weights, legacy_weights, rtol=3e-4, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen_one_kernel_topk_exact_ties_prefer_smaller_expert(dtype):
    from freetoken.kernel.triton.qwen_router import qwen_fused_topk

    logits = torch.zeros((2, 512), device="cuda", dtype=dtype)
    logits[0, :32] = 1
    logits[1, 100:132] = -0.0
    _, ids = qwen_fused_topk(logits, 10, False)
    ref_ids = torch.argsort(
        logits.float(), dim=-1, descending=True, stable=True
    )[:, :10]
    assert torch.equal(ids, ref_ids.to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_one_kernel_topk_masks_padded_ids():
    from freetoken.kernel.triton.qwen_router import qwen_fused_topk

    logits = torch.randn((4, 64), device="cuda", dtype=torch.float16)
    num_real = torch.tensor(2, device="cuda", dtype=torch.int32)
    _, ids = qwen_fused_topk(logits, 10, False, num_real)
    assert (ids[:2] >= 0).all()
    assert (ids[2:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 24])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_experts_decode_matches_reference_for_non_contiguous_slots(batch_size, dtype):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    cache_size = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 8
    torch.manual_seed(42 + batch_size)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    slot_ids = torch.tensor([31, 4, 18, 0, 29, 7, 35, 12], device=device, dtype=torch.int32)
    topk_ids = slot_ids.repeat(batch_size, 1).contiguous()

    output = fused_experts_decode_impl(hidden_states, gate_up, down, topk_weights, topk_ids)
    expected = _reference_fused_experts_decode(hidden_states, gate_up, down, topk_weights, topk_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_matches_grouped_impl(dtype, apply_router_weight_on_input):
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    device = torch.device("cuda")
    batch_size = 4
    num_experts = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    torch.manual_seed(91)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.tensor(
        [[31, 4, 18, 0], [29, 7, 35, 12], [6, 21, 1, 33], [16, 3, 28, 9]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = fused_experts_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_experts_grouped_impl_is_cuda_graph_capturable():
    from freetoken.moe.fused import fused_experts_impl

    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_experts = 8
    hidden_size = 32
    intermediate_size = 24
    torch.manual_seed(17)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    topk_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.float32)
    topk_ids = torch.tensor([[6, 1, 4, 7]], device=device, dtype=torch.int32)
    output = torch.empty_like(hidden_states)

    def run():
        output.copy_(
            fused_experts_impl(
                hidden_states,
                gate_up,
                down,
                topk_weights,
                topk_ids,
            )
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_activation_and_router_weight_modes(
    activation,
    apply_router_weight_on_input,
):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    batch_size = 3
    cache_size = 19
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    dtype = torch.float16
    torch.manual_seed(123)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    topk_weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.15, 0.35, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        device=device,
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[17, 2, 11, 5], [3, 17, 0, 14], [8, 6, 12, 1]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = _reference_fused_experts_decode(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)
