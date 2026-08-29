import inspect
import math

import pytest
import torch
import torch.nn.functional as F


class _FakeCudaTensor:
    """Small tensor-shaped object for exercising launch argument binding."""

    def __init__(self, shape, *, dtype=torch.bfloat16, device="cuda"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = True

    def dim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def stride(self, dim=None):
        strides = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        strides = tuple(reversed(strides))
        return strides if dim is None else strides[dim]


class _BindingKernel:
    """Stand-in for a JITFunction which still enforces Python call binding."""

    _launch_options = {"num_warps", "num_stages", "num_ctas"}

    def __init__(self, jit_function):
        self.signature = inspect.signature(jit_function.fn)
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kernel_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in self._launch_options
            }
            # Signature.bind reproduces JITFunction.run's argument validation,
            # including duplicate positional/keyword values such as the D bug.
            bound = self.signature.bind(*args, **kernel_kwargs)
            self.launches.append((grid, bound.arguments))

        return launch


def test_qwen_fused_kernel_launch_call_conventions_without_cuda(monkeypatch):
    """Bind every new Triton wrapper call without requiring a CUDA device."""
    from freetoken.kernel.triton import qwen_hyper_connection as hyper
    from freetoken.kernel.triton import qwen_router as router

    kernels = {}
    for module, name in (
        (hyper, "_silu_div_kernel"),
        (hyper, "_mix_reduce_kernel"),
        (hyper, "_inject_kernel"),
        (hyper, "_weighted_residual_kernel"),
        (router, "_qwen_fused_topk_kernel"),
    ):
        binding_kernel = _BindingKernel(getattr(module, name))
        kernels[name] = binding_kernel
        monkeypatch.setattr(module, name, binding_kernel)

    def empty(shape, *, dtype, device):
        return _FakeCudaTensor(shape, dtype=dtype, device=device)

    def empty_like(tensor):
        return _FakeCudaTensor(
            tensor.shape, dtype=tensor.dtype, device=tensor.device
        )

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "empty_like", empty_like)

    tokens, hc_count, hidden_size = 3, 4, 256
    hc_size = hc_count * hidden_size
    normalized = _FakeCudaTensor((tokens, hc_size))
    lowrank_logits = _FakeCudaTensor((tokens, 80))
    mix_logits = _FakeCudaTensor((tokens, hc_size))
    inject_logits = _FakeCudaTensor((tokens, hc_count))
    mixed = _FakeCudaTensor((tokens, hidden_size))

    hyper.hyper_silu_div(lowrank_logits, hc_count)
    hyper.hyper_mix_reduce(mix_logits, normalized, hc_count, hidden_size)
    hyper.hyper_inject(inject_logits, hc_count)
    hyper.hyper_weighted_residual(
        normalized, mixed, inject_logits, hc_count, hidden_size
    )

    logits = _FakeCudaTensor((tokens, 512))
    router.qwen_fused_topk(logits, topk=10, renormalize=False)
    valid_tokens = _FakeCudaTensor((1,), dtype=torch.int32)
    router.qwen_fused_topk(
        logits,
        topk=10,
        renormalize=True,
        num_token_non_padded=valid_tokens,
    )

    for name in (
        "_silu_div_kernel",
        "_mix_reduce_kernel",
        "_inject_kernel",
        "_weighted_residual_kernel",
    ):
        assert len(kernels[name].launches) == 1
    assert len(kernels["_qwen_fused_topk_kernel"].launches) == 2

    mix_arguments = kernels["_mix_reduce_kernel"].launches[0][1]
    assert mix_arguments["D"] == hidden_size
    assert mix_arguments["HC"] == hc_count
    residual_arguments = kernels["_weighted_residual_kernel"].launches[0][1]
    assert residual_arguments["D"] == hidden_size
    assert residual_arguments["HC"] == hc_count
    router_arguments = kernels["_qwen_fused_topk_kernel"].launches[0][1]
    assert router_arguments["E"] == 512
    assert router_arguments["K"] == 10
    assert router_arguments["K_PAD"] == 16
    assert router_arguments["RENORMALIZE"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("combine", [False, True])
def test_qwen_hyper_connection_fused_elementwise_matches_eager(combine):
    from freetoken.kernel.triton.qwen_hyper_connection import (
        hyper_inject,
        hyper_mix_reduce,
        hyper_silu_div,
        hyper_weighted_residual,
    )

    torch.manual_seed(2026)
    tokens, hc_count, hidden_size, lowrank = 3, 4, 256, 80
    hc_size = hc_count * hidden_size
    hyper_input = torch.randn(
        (tokens, hc_size), device="cuda", dtype=torch.bfloat16
    ) * 0.25
    norm_weight = torch.randn(
        (hc_size,), device="cuda", dtype=torch.bfloat16
    ) * 0.02
    down_weight = torch.randn(
        (lowrank, hc_size), device="cuda", dtype=torch.bfloat16
    ) * 0.02
    up_weight = torch.randn(
        (hc_size, lowrank), device="cuda", dtype=torch.bfloat16
    ) * 0.02
    inject_weight = torch.randn(
        (hc_count, hc_size), device="cuda", dtype=torch.bfloat16
    ) * 0.02

    from freetoken.kernel.fla import rms_norm_gated

    norm_kwargs = dict(
        x=hyper_input,
        weight=norm_weight,
        bias=None,
        eps=1e-6,
        group_size=hidden_size,
        is_rms_norm=True,
        weight_plus_one=True,
    )
    legacy_normalized = rms_norm_gated(**norm_kwargs, save_stats=True)
    normalized = rms_norm_gated(**norm_kwargs, save_stats=False)
    assert torch.equal(normalized, legacy_normalized)

    legacy_down = F.linear(legacy_normalized, down_weight)
    down = F.linear(normalized, down_weight)
    ref_lowrank = F.silu(legacy_down / hc_count)
    actual_lowrank = hyper_silu_div(down, hc_count)
    torch.testing.assert_close(actual_lowrank, ref_lowrank, rtol=2e-2, atol=2e-3)

    legacy_up = F.linear(ref_lowrank, up_weight)
    up = F.linear(actual_lowrank, up_weight)
    ref_mixed = (
        torch.sigmoid(legacy_up).view(tokens, hc_count, hidden_size)
        * legacy_normalized.view(tokens, hc_count, hidden_size)
    ).mean(dim=1)
    actual_mixed = hyper_mix_reduce(up, normalized, hc_count, hidden_size)
    torch.testing.assert_close(actual_mixed, ref_mixed, rtol=2e-2, atol=2e-3)

    if not combine:
        return
    legacy_inject_logits = F.linear(legacy_normalized, inject_weight)
    inject_logits = F.linear(normalized, inject_weight)
    ref_inject = 2 * torch.sigmoid(legacy_inject_logits / hc_count)
    actual_inject = hyper_inject(inject_logits, hc_count)
    torch.testing.assert_close(actual_inject, ref_inject, rtol=2e-2, atol=2e-3)

    residual = torch.randn_like(normalized) * 0.25
    ref_output = residual + (
        ref_mixed.unsqueeze(1) * ref_inject.unsqueeze(-1)
    ).flatten(1)
    actual_output = hyper_weighted_residual(
        residual, actual_mixed, actual_inject, hc_count, hidden_size
    )
    torch.testing.assert_close(actual_output, ref_output, rtol=2e-2, atol=2e-3)


def test_qwen_hyper_connection_env_switch(monkeypatch):
    from freetoken.models.qwen4_exp.model import _hyper_connection_fused_enabled

    monkeypatch.delenv("FREETOKEN_QWEN_HYPER_CONNECTION_FUSED", raising=False)
    assert _hyper_connection_fused_enabled()
    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("FREETOKEN_QWEN_HYPER_CONNECTION_FUSED", value)
        assert not _hyper_connection_fused_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_shared_gate_kernels_match_torch():
    from freetoken.kernel.triton.moe_shared_gate import (
        shared_gate_mul_add,
        shared_gate_sigmoid,
    )

    torch.manual_seed(29)
    hidden = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.05
    routed = torch.randn_like(hidden)
    shared = torch.randn_like(hidden)

    gate = shared_gate_sigmoid(hidden, weight)
    actual = shared_gate_mul_add(routed, shared, gate)
    expected_gate = torch.sigmoid(F.linear(hidden, weight.unsqueeze(0))).flatten()
    expected = routed + expected_gate.unsqueeze(1) * shared
    reference = routed.float() + shared.float() * torch.sigmoid(
        hidden.float() @ weight.float()
    ).unsqueeze(1)

    torch.testing.assert_close(gate, expected_gate.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert (actual.float() - reference).abs().max() <= (
        expected.float() - reference
    ).abs().max() + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_merged_hyper_connection_handles_prefill_rows():
    from types import SimpleNamespace

    from freetoken.models.qwen4_exp.model import _GatedResidual

    config = SimpleNamespace(
        hidden_size=256,
        rms_norm_eps=1e-6,
        qwen4_args=SimpleNamespace(hc_count=4, hc_lowrank=80),
    )
    residual = _GatedResidual(config)
    torch.manual_seed(31)
    residual.hc_norm.weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    residual.input_mix_weight_up.weight = torch.randn(
        1024, 80, device="cuda", dtype=torch.bfloat16
    )
    residual.input_mix_weight_down_block_inject.weight = torch.randn(
        96, 1024, device="cuda", dtype=torch.bfloat16
    )
    hidden = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16)

    residual._projection_fused = True
    mixed, _, inject = residual.forward(hidden)

    assert mixed.shape == (3, 256)
    assert inject.shape == (3, 4)
