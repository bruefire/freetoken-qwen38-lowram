"""QSA raw per-token index-key slab and sparse full-layer remapping."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kvcache.qsa_pool import QSAKVCache


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def test_qsa_pool_stores_raw_keys_and_only_qsa_layer_kv():
    pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=4,
        head_dim=64,
        num_pages=8,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        index_head_dim=128,
        num_index_layers=1,
        layer_ids=(3,),
    )
    rows = torch.tensor([16, 17, 31], dtype=torch.int32, device="cuda")
    k = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    raw_index_k = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")

    pool.store_kv(k, v, rows, layer_id=3)
    pool.store_index_k(raw_index_k, rows, slot=0)

    torch.testing.assert_close(pool.k_cache(3).view(-1, 1, 64)[rows.long(), 0], k)
    torch.testing.assert_close(pool.index_k_cache(0)[rows.long()], raw_index_k)
    with pytest.raises(KeyError, match="no paged KV storage"):
        pool.k_cache(0)
