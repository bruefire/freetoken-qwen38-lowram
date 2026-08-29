from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.mtp_worker import (
    MTPAdaptiveController,
    MTPWorker,
    _mtp_adaptive_cycles,
    _mtp_adaptive_enabled,
    _mtp_adaptive_min_acceptance,
    _mtp_adaptive_probe_interval,
    _mtp_adaptive_samples,
    _mtp_adaptive_window,
    _mtp_draft_p_min,
    _mtp_max_drafts,
    choose_adaptive_draft_width,
)


def _req(tokens: list[int]) -> Req:
    return Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=7,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )


def test_mtp_draft_environment_is_validated(monkeypatch):
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "2")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "0.6")
    assert _mtp_max_drafts() == 2
    assert _mtp_draft_p_min() == 0.6

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "3")
    with pytest.raises(ValueError, match="must be 1 or 2"):
        _mtp_max_drafts()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "1.1")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _mtp_draft_p_min()

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "true")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES", "24")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE", "0.8")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES", "4")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW", "32")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL", "8")
    assert _mtp_adaptive_enabled()
    assert _mtp_adaptive_cycles() == 24
    assert _mtp_adaptive_min_acceptance() == 0.8
    assert _mtp_adaptive_samples() == 4
    assert _mtp_adaptive_window() == 32
    assert _mtp_adaptive_probe_interval() == 8

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "maybe")
    with pytest.raises(ValueError, match="must be a boolean"):
        _mtp_adaptive_enabled()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _mtp_adaptive_cycles()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE", "1.1")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _mtp_adaptive_min_acceptance()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _mtp_adaptive_samples()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _mtp_adaptive_window()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _mtp_adaptive_probe_interval()


def test_mtp_can_speculate_only_for_initialized_single_request():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    worker = object.__new__(MTPWorker)
    worker.uid = req.uid
    worker.predictor_cached_len = req.cached_len
    worker.pending_hidden = None
    worker.pending_draft = torch.tensor([7], dtype=torch.int32)
    worker.adaptive_disabled_uid = None

    assert worker.can_speculate(batch)

    other = _req([10, 11, 12])
    other.uid = req.uid + 1
    assert not worker.can_speculate(Batch(reqs=[other], phase="decode"))
    assert not worker.can_speculate(Batch(reqs=[req, other], phase="decode"))

    worker.predictor_cached_len += 1
    assert not worker.can_speculate(batch)


def test_mtp_prefill_keeps_hidden_and_next_token_alignment_across_chunks():
    worker = object.__new__(MTPWorker)
    worker.engine = SimpleNamespace(device=torch.device("cpu"))
    # Offline generation reuses request UIDs. A new position-zero prefill must
    # still reset all per-request predictor state.
    worker.uid = 7
    worker.predictor_cached_len = 99
    worker.pending_hidden = torch.tensor([[-1.0]])
    worker.pending_draft = torch.tensor([3], dtype=torch.int32)
    worker.pending_predictor_hidden = None
    worker.max_supported_drafts = 1
    worker.max_drafts = 1
    worker.draft_p_min = 0.0
    worker.adaptive_controller = None
    worker._adaptive_pending = []
    calls = []

    def run_predictor(self, batch, req, hidden, token_ids, source_start):
        calls.append((source_start, hidden.clone(), token_ids.clone()))
        self.predictor_cached_len += token_ids.numel()
        output = torch.zeros(token_ids.numel(), 4)
        return output, hidden + 10

    worker._run_predictor = MethodType(run_predictor, worker)

    first = _req([10, 11, 12])
    first_batch = Batch(reqs=[first], phase="prefill")
    worker.update_prefill(
        first_batch,
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([99], dtype=torch.int32),
        start=0,
        end=3,
        final=False,
    )

    second = Req(
        input_ids=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
        table_idx=0,
        cached_len=3,
        output_len=8,
        uid=7,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )
    second_batch = Batch(reqs=[second], phase="prefill")
    worker.update_prefill(
        second_batch,
        torch.tensor([[3.0], [4.0]]),
        torch.tensor([15], dtype=torch.int32),
        start=3,
        end=5,
        final=True,
    )

    assert calls[0][0] == 0
    torch.testing.assert_close(calls[0][1].flatten(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(calls[0][2], torch.tensor([11, 12], dtype=torch.int32))
    assert calls[1][0] == 2
    torch.testing.assert_close(calls[1][1].flatten(), torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(
        calls[1][2], torch.tensor([13, 14, 15], dtype=torch.int32)
    )
    assert worker.predictor_cached_len == 5
    assert worker.pending_hidden is None
    assert worker.pending_draft.tolist() == [0]


def _decode_worker(
    req: Req,
    verify_tokens: list[int],
    *,
    prefix_checkpoint=True,
    max_drafts=1,
    predictor_outputs: list[int] | None = None,
):
    worker = object.__new__(MTPWorker)
    worker.engine = SimpleNamespace(
        device=torch.device("cpu"),
        linear_state_pool=None,
        model=SimpleNamespace(),
    )
    worker.model = worker.engine.model
    worker.uid = req.uid
    worker.predictor_cached_len = req.cached_len
    worker.pending_hidden = None
    worker.pending_draft = torch.tensor([7], dtype=torch.int32)
    worker.pending_predictor_hidden = torch.tensor([[50.0]])
    worker.pending_draft_confidence = 1.0
    worker.max_supported_drafts = max_drafts
    worker.max_drafts = max_drafts
    worker.draft_p_min = 0.0
    worker.adaptive_enabled = False
    worker.adaptive_cycles = 64
    worker.adaptive_min_acceptance = 0.75
    worker.adaptive_disabled_uid = None
    worker.adaptive_decision_cycle = None
    worker.metrics = SimpleNamespace(
        cycles=0,
        proposed_drafts=0,
        accepted_drafts=0,
        emitted_tokens=0,
        acceptance_rate=0.0,
        cycle_trace=[],
    )
    worker.log_interval = 0
    worker.timing_enabled = False
    worker.timing_events = []
    extensions = []
    predictors = []
    prefix_commits = []
    predictor_outputs = list(predictor_outputs or [4])

    class TargetGraph:
        def commit_prefix(self, target_req, checkpoint_index):
            prefix_commits.append((target_req, checkpoint_index))

    worker.target_verify_graphs = (
        {count: TargetGraph() for count in range(2, max_drafts + 2)}
        if prefix_checkpoint
        else {}
    )

    def target_extension(self, batch, target_req, token_ids, start):
        extensions.append((start, token_ids.clone()))
        rows = token_ids.numel()
        logits = torch.full((rows, 16), -1.0)
        for row, token in enumerate(verify_tokens[:rows]):
            logits[row, token] = 1.0
        return logits, token_ids.to(torch.float32).unsqueeze(1)

    def predictor(self, batch, target_req, hidden, token_ids, source_start):
        predictors.append((source_start, hidden.clone(), token_ids.clone()))
        self.predictor_cached_len += token_ids.numel()
        logits = torch.full((token_ids.numel(), 16), -1.0)
        logits[-1, predictor_outputs.pop(0)] = 1.0
        return logits, hidden + 100

    worker._run_target_extension = MethodType(target_extension, worker)
    worker._run_predictor = MethodType(predictor, worker)
    return worker, extensions, predictors, prefix_commits


def test_mtp_decode_accepts_draft_and_carries_next_draft():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [7, 9])

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 9]
    assert len(extensions) == 1
    assert extensions[0][0] == 2
    assert extensions[0][1].tolist() == [12, 7]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [12.0, 7.0]
    assert predictors[0][2].tolist() == [7, 9]
    assert not prefix_commits
    assert worker.predictor_cached_len == 4
    assert worker.pending_draft.tolist() == [4]
    assert (req.cached_len, req.device_len) == (3, 4)


def test_mtp_decode_commits_verified_prefix_after_rejection():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [8, 9])

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert len(extensions) == 1
    assert extensions[0][0] == 2
    assert extensions[0][1].tolist() == [12, 7]
    assert prefix_commits == [(req, 0)]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [12.0]
    assert predictors[0][2].tolist() == [8]
    assert worker.predictor_cached_len == 3
    assert worker.pending_draft.tolist() == [4]
    assert (req.cached_len, req.device_len) == (3, 4)


def test_mtp_decode_replays_rejection_without_target_graph():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req, [8, 9], prefix_checkpoint=False
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert [tokens.tolist() for _, tokens in extensions] == [[12, 7], [12]]
    assert not prefix_commits
    assert predictors[0][1].flatten().tolist() == [12.0]
    assert predictors[0][2].tolist() == [8]


def test_mtp_confidence_gate_can_skip_target_speculation():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [8])
    worker.draft_p_min = 0.6
    worker.pending_draft_confidence = 0.5

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [8]
    assert extensions[0][1].tolist() == [12]
    assert predictors[0][2].tolist() == [8]
    assert not prefix_commits
    assert worker.metrics.proposed_drafts == 0


def test_mtp_two_drafts_commit_three_rows_when_both_are_accepted():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 4, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 4, 9]
    assert extensions[0][1].tolist() == [12, 7, 4]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [50.0]
    assert predictors[0][2].tolist() == [7]
    assert predictors[1][0] == 2
    assert predictors[1][1].flatten().tolist() == [12.0, 7.0, 4.0]
    assert predictors[1][2].tolist() == [7, 4, 9]
    assert not prefix_commits
    assert worker.predictor_cached_len == 5
    assert worker.pending_draft.tolist() == [5]
    assert worker.metrics.proposed_drafts == 2
    assert worker.metrics.accepted_drafts == 2


def test_mtp_two_drafts_commit_second_prefix_after_partial_acceptance():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 8, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [7, 8]
    assert prefix_commits == [(req, 1)]
    assert predictors[1][1].flatten().tolist() == [12.0, 7.0]
    assert predictors[1][2].tolist() == [7, 8]
    assert worker.predictor_cached_len == 4
    assert worker.metrics.accepted_drafts == 1


def test_mtp_two_drafts_commit_first_prefix_when_first_is_rejected():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, predictors, prefix_commits = _decode_worker(
        req,
        [8, 4, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert prefix_commits == [(req, 0)]
    assert predictors[1][1].flatten().tolist() == [12.0]
    assert predictors[1][2].tolist() == [8]
    assert worker.predictor_cached_len == 3
    assert worker.metrics.accepted_drafts == 0


def test_mtp_second_draft_confidence_gate_verifies_only_first_draft():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )
    worker.draft_p_min = 0.6
    candidates = iter(
        [
            (torch.tensor([4], dtype=torch.int32), 0.5),
            (torch.tensor([5], dtype=torch.int32), 1.0),
        ]
    )

    def draft_candidate(self, logits):
        return next(candidates)

    worker._draft_candidate = MethodType(draft_candidate, worker)

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 9]
    assert extensions[0][1].tolist() == [12, 7]
    assert predictors[0][2].tolist() == [7]
    assert predictors[1][2].tolist() == [7, 9]
    assert not prefix_commits
    assert worker.metrics.proposed_drafts == 1


def test_mtp_tail_limit_does_not_change_adaptive_policy_width():
    req = _req([10, 11, 12])
    req.cached_len = 2
    req.max_device_len = req.device_len + 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, _predictors, _prefix_commits = _decode_worker(
        req, [7, 8], max_drafts=2
    )
    policy_widths = []

    def adaptive_width(self, max_width):
        policy_widths.append(max_width)
        return max_width

    worker._adaptive_width = MethodType(adaptive_width, worker)

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 8]
    assert policy_widths == [2]
    assert extensions[0][1].tolist() == [12, 7]


@pytest.mark.parametrize(
    ("survival", "costs", "expected"),
    [
        ([0.5, 0.25], [10.0, 20.0, 30.0], 0),
        ([0.8, 0.6], [10.0, 14.0, 30.0], 1),
        ([0.8, 0.6], [10.0, 15.0, 19.0], 2),
    ],
)
def test_mtp_adaptive_selector_maximizes_expected_throughput(survival, costs, expected):
    assert choose_adaptive_draft_width(survival, costs) == expected


def test_mtp_adaptive_selector_validates_survival_and_costs():
    with pytest.raises(ValueError, match="one cost"):
        choose_adaptive_draft_width([0.8, 0.6], [10.0, 20.0])
    with pytest.raises(ValueError, match="non-increasing"):
        choose_adaptive_draft_width([0.5, 0.6], [10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="positive"):
        choose_adaptive_draft_width([0.8], [10.0, 0.0])


def test_mtp_adaptive_rejects_an_independent_confidence_gate():
    worker = object.__new__(MTPWorker)
    worker.adaptive_enabled = True
    worker.draft_p_min = 0.6

    with pytest.raises(RuntimeError, match="cannot be combined"):
        worker._adaptive_width(2)


def test_mtp_adaptive_controller_warms_max_width_before_calibration():
    controller = MTPAdaptiveController(
        samples=2, window=32, probe_interval=100, disable_after=4
    )
    widths = []
    for _ in range(8):
        width = controller.choose(2)
        widths.append(width)
        controller.record(
            width=width,
            accepted=width,
            cycle_ms=20.0 + width,
            target_ms=10.0,
        )

    assert widths == [2, 2, 2, 2, 1, 1, 2, 2]


def test_mtp_adaptive_controller_delays_ordinary_fallback():
    controller = MTPAdaptiveController(
        samples=1, window=32, probe_interval=100, disable_after=4
    )
    for _ in range(4):
        assert controller.choose(2) == 2
        controller.record(
            width=2,
            accepted=0,
            cycle_ms=50.0,
            target_ms=10.0,
        )

    assert controller.choose(2) == 1
    controller.record(width=1, accepted=0, cycle_ms=30.0, target_ms=10.0)
    assert controller.choose(2) == 0
    controller.record(width=0, accepted=0, cycle_ms=20.0, target_ms=10.0)
    assert controller.choose(2) == 0
    assert controller.selected_width == 0


def test_mtp_adaptive_controller_adds_measured_host_gap_to_target_cost():
    controller = MTPAdaptiveController(
        samples=1, window=8, probe_interval=8, disable_after=1
    )

    controller.record(
        width=0,
        accepted=0,
        cycle_ms=20.0,
        target_ms=10.0,
        wall_ms=23.0,
    )

    assert controller.summary(2)["cost_ms"][0] == pytest.approx(13.0)


def test_mtp_adaptive_controller_keeps_cuda_cost_hidden_by_async_wall_time():
    controller = MTPAdaptiveController(
        samples=1, window=8, probe_interval=8, disable_after=1
    )

    controller.record(
        width=2,
        accepted=2,
        cycle_ms=20.0,
        target_ms=10.0,
        wall_ms=15.0,
    )

    assert controller.summary(2)["cost_ms"][2] == pytest.approx(20.0)
    assert controller.summary(2)["trace"] == [
        {
            "cycle": 1,
            "width": 2,
            "accepted": 2,
            "cycle_ms": 20.0,
            "target_ms": 10.0,
            "wall_ms": 15.0,
            "host_ms": 0.0,
            "cost_ms": 20.0,
        }
    ]


def test_mtp_adaptive_controller_selects_two_drafts_when_profitable():
    controller = MTPAdaptiveController(
        samples=1, window=32, probe_interval=100, disable_after=3
    )
    samples = [
        (2, 2, 19.0, 14.0),
        (2, 2, 19.0, 14.0),
        (2, 2, 19.0, 14.0),
        (1, 1, 15.0, 12.0),
    ]
    for width, accepted, cycle_ms, target_ms in samples:
        assert controller.choose(2) == width
        controller.record(
            width=width,
            accepted=accepted,
            cycle_ms=cycle_ms,
            target_ms=target_ms,
        )

    assert controller.choose(2) == 2
    assert controller.summary(2)["survival"] == [1.0, 1.0]


def test_mtp_worker_collects_completed_adaptive_cuda_samples():
    class Event:
        def __init__(self, milliseconds, ready=True):
            self.milliseconds = milliseconds
            self.ready = ready

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            return other.milliseconds - self.milliseconds

    worker = object.__new__(MTPWorker)
    worker.adaptive_controller = MTPAdaptiveController(
        samples=1, window=8, probe_interval=8, disable_after=8
    )
    worker._adaptive_pending = [
        SimpleNamespace(
            width=2,
            accepted=1,
            wall_started=0.0,
            cycle_started=Event(1.0),
            target_started=Event(3.0),
            target_ended=Event(15.0),
            cycle_ended=Event(21.0),
            wall_ms=22.0,
        ),
        SimpleNamespace(
            width=1,
            accepted=1,
            wall_started=0.0,
            cycle_started=Event(21.0),
            target_started=Event(23.0),
            target_ended=Event(30.0),
            cycle_ended=Event(35.0, ready=False),
            wall_ms=None,
        ),
    ]

    worker._collect_adaptive_samples()

    assert worker.adaptive_controller.summary(2)["cost_ms"] == [None, None, 22.0]
    assert len(worker._adaptive_pending) == 1


def test_mtp_cost_controller_falls_back_without_losing_current_token(monkeypatch):
    class Event:
        clock = 0.0

        def __init__(self, **_kwargs):
            self.milliseconds = 0.0

        def record(self, _stream):
            Event.clock += 1.0
            self.milliseconds = Event.clock

        def query(self):
            return True

        def elapsed_time(self, other):
            return other.milliseconds - self.milliseconds

    monkeypatch.setattr(torch.cuda, "Event", Event)
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, _predictors, _prefix_commits = _decode_worker(
        req, [8], max_drafts=2
    )
    worker.engine.stream = None
    worker.adaptive_enabled = True
    worker.adaptive_controller = MTPAdaptiveController(
        samples=1, window=8, probe_interval=8, disable_after=3
    )
    worker._adaptive_pending = []
    worker.adaptive_disabled_uid = None
    worker.adaptive_decision_cycle = None
    worker.adaptive_controller.record(
        width=0, accepted=0, cycle_ms=20.0, target_ms=10.0
    )
    worker.adaptive_controller.record(
        width=1, accepted=0, cycle_ms=30.0, target_ms=10.0
    )
    worker.adaptive_controller.record(
        width=2, accepted=0, cycle_ms=40.0, target_ms=10.0
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [8]
    assert extensions[0][1].tolist() == [12]
    assert worker.adaptive_disabled_uid == req.uid
    assert worker.adaptive_decision_cycle == 1
    assert worker._adaptive_pending == []
    assert not worker.can_speculate(batch)
