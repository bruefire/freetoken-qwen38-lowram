from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.scheduler import _host_buffer, _stage_host_input_ids


def _req(tokens: list[int], *, output_len: int = 4, uid: int = 1) -> Req:
    return Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=uid,
        cached_len=len(tokens) - 1,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_stage_host_input_ids_reuses_sampler_result_for_inflight_token():
    req = _req([4, 5, 6])
    req.complete_one()
    dummy = _req([0], output_len=1, uid=-1)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req, dummy]
    sampled = torch.tensor([7], dtype=torch.int32)
    ready = object()
    last_data = (
        SimpleNamespace(batch=SimpleNamespace(reqs=[req])),
        SimpleNamespace(
            next_tokens_cpu=sampled,
            copy_done_event=ready,
        ),
    )

    _stage_host_input_ids(batch, last_data)  # type: ignore[arg-type]

    assert batch.host_input_ids is not None
    assert batch.host_input_ids.ready is ready
    assert batch.host_input_ids.parts[0].data_ptr() == sampled.data_ptr()
    assert batch.host_input_ids.parts[0].tolist() == [7]
    assert batch.host_input_ids.parts[1].tolist() == [0]


def test_stage_host_input_ids_uses_drained_host_history_without_event():
    req = _req([4, 5, 6])
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]

    _stage_host_input_ids(batch, None)

    assert batch.host_input_ids is not None
    assert batch.host_input_ids.ready is None
    assert batch.host_input_ids.parts[0].tolist() == [7]


def test_host_buffer_is_grow_only_and_reuses_storage(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    buffers: dict[str, torch.Tensor] = {}

    first = _host_buffer(buffers, "positions", 3, torch.int32)
    smaller = _host_buffer(buffers, "positions", 2, torch.int32)
    grown = _host_buffer(buffers, "positions", 9, torch.int32)

    assert first.data_ptr() == smaller.data_ptr()
    assert grown.numel() == 9
    assert buffers["positions"].numel() == 16
