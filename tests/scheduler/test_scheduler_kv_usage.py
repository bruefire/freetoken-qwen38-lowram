from __future__ import annotations

from types import SimpleNamespace

import torch


def test_kv_usage_pages_excludes_evictable_prefix_cache():
    # page_usage now lives on the CacheManagerLike interface (polymorphic vs DSV4); test the
    # generic formula there. The unbound method works on a duck-typed namespace.
    from freetoken.scheduler.cache import CacheManager

    cache_manager = SimpleNamespace(
        num_pages=100,
        page_size=4,
        free_slots=[0] * 20,
        is_hybrid=False,
        is_swa=False,
        prefix_cache=SimpleNamespace(
            size_info=SimpleNamespace(evictable_size=120, protected_size=40)
        ),
    )

    used_pages, total_pages = CacheManager.page_usage(cache_manager)

    assert used_pages == 50
    assert total_pages == 100


def test_release_allocated_tail_returns_only_uncommitted_pages():
    from freetoken.scheduler.cache import CacheManager

    page_table = torch.zeros((1, 8), dtype=torch.int32)
    manager = CacheManager(8, 1, page_table, "naive")
    manager.free_slots = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32)
    page_table[0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    req = SimpleNamespace(table_idx=0)

    manager.release_allocated_tail(req, keep_end=2, allocated_end=3)

    assert 2 in manager.free_slots.tolist()
    assert page_table[0, :3].tolist() == [0, 1, 0]


def test_multi_token_drain_advances_lengths_and_releases_rejected_tail():
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.scheduler.scheduler import Scheduler

    req = Req(
        input_ids=torch.tensor([10, 20], dtype=torch.int32),
        table_idx=0,
        cached_len=1,
        output_len=5,
        uid=9,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )
    # State immediately after the ordinary target decode that produced the base
    # token. The sampled base has not been appended on the host yet.
    req.cached_len = 2
    req.device_len = 3
    batch = Batch(reqs=[req], phase="decode")
    batch.speculative_allocated_ends = {req: 4}
    released = []
    scheduler = SimpleNamespace(
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        cache_manager=SimpleNamespace(
            release_allocated_tail=lambda request, **kw: released.append(kw)
        ),
        decode_manager=SimpleNamespace(remove_req=lambda request: None),
        _match_stop_str=lambda request: None,
        _free_req_resources=lambda request: None,
    )
    replies = []
    finished = set()

    Scheduler._drain_multi_token(
        scheduler,
        req,
        torch.tensor([21, 22], dtype=torch.int32),
        0,
        2,
        replies,
        finished,
        batch,
    )

    assert req.input_ids.tolist() == [10, 20, 21, 22]
    assert (req.cached_len, req.device_len) == (3, 4)
    assert released == [{"keep_end": 3, "allocated_end": 4}]
    assert [reply.next_token for reply in replies] == [21, 22]
    assert not any(reply.finished for reply in replies)


def test_single_token_drain_helper_releases_confidence_gated_tail():
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.scheduler.scheduler import Scheduler

    req = Req(
        input_ids=torch.tensor([10, 20], dtype=torch.int32),
        table_idx=0,
        cached_len=1,
        output_len=5,
        uid=9,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )
    req.cached_len = 2
    req.device_len = 3
    batch = Batch(reqs=[req], phase="decode")
    batch.speculative_allocated_ends = {req: 4}
    released = []
    scheduler = SimpleNamespace(
        cache_manager=SimpleNamespace(
            release_allocated_tail=lambda request, **kw: released.append(kw)
        )
    )

    Scheduler._release_speculative_tail(scheduler, batch, req)

    assert released == [{"keep_end": 2, "allocated_end": 4}]
    assert not batch.speculative_allocated_ends


def test_swa_token_usage_counts_window_pool():
    from freetoken.scheduler.scheduler import Scheduler

    # swa_paged model: 129-token pool -> 128 allocatable (sentinel excluded from total);
    # 64 available (free + evictable tree) -> 64 used.
    cache_manager = SimpleNamespace(
        swa_paged=True,
        swa_pool=SimpleNamespace(swa_num_tokens=129),
        swa_available_size=64,
    )
    assert Scheduler._swa_token_usage(SimpleNamespace(cache_manager=cache_manager)) == (64, 128)

    # non-SWA models report None (no swa field in logs/stats).
    plain = SimpleNamespace(cache_manager=SimpleNamespace(swa_paged=False))
    assert Scheduler._swa_token_usage(plain) is None


def _fake_engine(swa=True, moe=True, mamba=True):
    import torch

    # Superset fake engine driving the REAL compute_cache_pools/compute_cache_unit_bytes:
    # KV 64 pages x 16 tokens x 1 MiB/token = 1.00 GiB; swa 512 tokens x 2 MiB = 1.00 GiB;
    # mamba 64 slots x 16 MiB = 1.00 GiB; MoE 24 slots of 8x16 experts, 4 KiB/slot.
    return SimpleNamespace(
        num_pages=64,
        config=SimpleNamespace(
            page_size=16,
            cache_type="swa_radix",
            model_config=SimpleNamespace(dsv4_args=None, has_swa_attention=swa),
        ),
        kv_cache=SimpleNamespace(
            swa_num_tokens=513, unit_bytes=lambda: (1 << 20, 1 << 21)
        ),
        moe_offload_cache=SimpleNamespace(
            cache_size=24, num_layers=8, num_experts=16,
            bank_caches={"w": torch.zeros((24, 1024), dtype=torch.float32)},
        ) if moe else None,
        linear_state_pool=SimpleNamespace(num_slots=65, bytes_per_slot=lambda: 1 << 24)
        if mamba else None,
    )


def test_log_cache_geometry_reports_all_pools(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    lines: list[str] = []
    monkeypatch.setattr(sched_mod.logger, "info_rank0", lines.append)
    Scheduler._log_cache_geometry(SimpleNamespace(engine=_fake_engine()), "Cache rebuilt")
    line = lines[-1]
    assert "Cache rebuilt: KV 64 pages (1024 tokens, 1.00 GiB)" in line
    assert "swa 512 pages (512 tokens, 1.00 GiB)" in line
    assert "mamba 64 slots (1.00 GiB)" in line
    assert "MoE cache 24/128 (0.00 GiB)" in line


def test_log_cache_geometry_plain_model_kv_only(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    lines: list[str] = []
    monkeypatch.setattr(sched_mod.logger, "info_rank0", lines.append)
    engine = _fake_engine(swa=False, moe=False, mamba=False)
    Scheduler._log_cache_geometry(SimpleNamespace(engine=engine), "Cache rebuilt")
    line = lines[-1]
    assert "Cache rebuilt: KV 64 pages (1024 tokens, 1.00 GiB)" in line
    assert "swa" not in line and "mamba" not in line and "MoE" not in line
