from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


class _Op:
    def __init__(self, fn):
        self._fn = fn

    def forward(self, *args):
        return self._fn(*args)


def test_gdn_decode_has_nested_breakdown_ranges(monkeypatch):
    import freetoken.models.qwen3_5_moe.gdn as gdn_module

    op = object.__new__(gdn_module.Qwen3_5GatedDeltaNet)
    op.layer_id = 0
    op._fp8 = False
    op.conv_dim = 6
    op.key_dim = 2
    op.value_dim = 2
    op.num_k_heads = 1
    op.num_v_heads = 1
    op.head_k_dim = 2
    op.head_v_dim = 2
    op._in_proj_split = [6, 2, 1, 1]
    op.scale = 2**-0.5
    op.A_log = torch.zeros(1)
    op.dt_bias = torch.zeros(1)
    op.in_proj = _Op(lambda hidden: torch.arange(10, dtype=hidden.dtype).view(1, 10))
    op._conv_decode = lambda conv_in, *_: conv_in
    op.norm = _Op(lambda core, gate: core + gate)
    op.out_proj = _Op(lambda value: value)

    fla = SimpleNamespace(
        cache_indices=torch.tensor([0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
    )
    pool = SimpleNamespace(
        local_index=lambda _layer_id: 0,
        recurrent_states=torch.zeros((1, 1, 1, 2, 2)),
    )
    context = SimpleNamespace(
        batch=SimpleNamespace(is_decode=True, fla_metadata=fla),
        linear_state_pool=pool,
    )
    monkeypatch.setattr(gdn_module, "get_global_ctx", lambda: context)
    monkeypatch.setattr(
        gdn_module,
        "gdn_decode_fla",
        lambda _q, _k, v, _a, _b, **_kwargs: v[0],
    )

    entered = []
    stack = []

    @contextmanager
    def profile_range(name):
        entered.append(name)
        stack.append(name)
        try:
            yield
        finally:
            assert stack.pop() == name

    monkeypatch.setattr(gdn_module, "decode_profile_range", profile_range)
    output = op.forward(torch.ones((1, 3)))

    assert output.shape == (1, 2)
    assert entered == [
        "FT.GDN",
        "FT.GDN.InProj",
        "FT.GDN.Conv1d",
        "FT.GDN.Recurrence",
        "FT.GDN.Norm",
        "FT.GDN.OutProj",
    ]


def test_gdn_fast_path_env_switch(monkeypatch):
    from freetoken.models.qwen3_5_moe.gdn import _gdn_fast_path_enabled

    monkeypatch.delenv("FREETOKEN_GDN_FAST_PATH", raising=False)
    assert _gdn_fast_path_enabled()
    monkeypatch.setenv("FREETOKEN_GDN_FAST_PATH", "0")
    assert not _gdn_fast_path_enabled()


def test_gdn_mtp_prefix_state_can_replace_verified_suffix():
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

    op = object.__new__(Qwen3_5GatedDeltaNet)
    op.layer_id = 3
    op._mtp_prefix_conv_state = None
    op._mtp_prefix_recurrent_state = None
    pool = SimpleNamespace(
        local_index=lambda _layer_id: 0,
        conv_states=torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4),
        recurrent_states=torch.arange(16, dtype=torch.float32).view(1, 2, 2, 2, 2),
    )

    expected_conv = pool.conv_states[0, 1].clone()
    expected_recurrent = pool.recurrent_states[0, 1].clone()
    op._save_mtp_prefix_state(
        pool, 0, torch.tensor([1], dtype=torch.int32), 0, 2
    )
    pool.conv_states[0, 1].add_(100)
    pool.recurrent_states[0, 1].add_(100)
    second_conv = pool.conv_states[0, 1].clone()
    second_recurrent = pool.recurrent_states[0, 1].clone()
    op._save_mtp_prefix_state(
        pool, 0, torch.tensor([1], dtype=torch.int32), 1, 2
    )
    pool.conv_states[0, 1].fill_(-1)
    pool.recurrent_states[0, 1].fill_(-1)

    op.commit_mtp_prefix_state(pool, 1, 1)

    torch.testing.assert_close(pool.conv_states[0, 1], second_conv)
    torch.testing.assert_close(pool.recurrent_states[0, 1], second_recurrent)
    op.commit_mtp_prefix_state(pool, 1, 0)
    torch.testing.assert_close(pool.conv_states[0, 1], expected_conv)
    torch.testing.assert_close(pool.recurrent_states[0, 1], expected_recurrent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_norm_without_unused_stats_matches_legacy():
    from freetoken.kernel.fla import rms_norm_gated

    torch.manual_seed(99)
    x = torch.randn((48, 128), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn((128,), device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        x=x,
        weight=weight,
        bias=None,
        z=gate,
        eps=1e-6,
        is_rms_norm=True,
        norm_before_gate=True,
        activation="silu",
    )
    legacy = rms_norm_gated(**kwargs, save_stats=True)
    optimized = rms_norm_gated(**kwargs, save_stats=False)
    assert torch.equal(optimized, legacy)
