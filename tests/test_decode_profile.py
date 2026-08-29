from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from freetoken.benchmark.decode_profile import (
    _apply_optimization_variant,
    _CudaStepRecorder,
    _ensure_mtp_sweep_warmup,
    _first_mismatch,
    _force_greedy_tokens,
    _graph_execution,
    _mtp_prompt_suite,
    _OptimizationVariant,
    _parser,
    _phase_prompt,
    _profile_summary,
    _read_mtp_stats,
    _read_ple_stats,
    _reset_ple_stats,
    _run_mtp_sweep,
    _run_ple_mmap_sweep,
    _StepEvent,
)
from freetoken.benchmark.ple_replay import _ngram_rows, _trajectory_from_report
from freetoken.engine.mtp_worker import MTPMetrics
from freetoken.models.qwen4_exp.model import build_ngram_ids
from freetoken.scheduler.decode_profiler import ServerDecodeProfiler


class _Stamp:
    def __init__(self, milliseconds: float):
        self.milliseconds = milliseconds

    def elapsed_time(self, other: _Stamp) -> float:
        return other.milliseconds - self.milliseconds


class _ProfileEvent:
    def __init__(
        self,
        key: str,
        *,
        parent: _ProfileEvent | None = None,
        cpu_us: float = 0.0,
        device_us: float = 0.0,
        device_type=None,
    ):
        self.key = key
        self.cpu_parent = parent
        self.cpu_time_total = cpu_us
        self.device_time_total = device_us
        self.device_type = device_type


def test_parser_defaults_match_reference_server() -> None:
    args = _parser("ft bench decode-profile").parse_args([])
    assert args.model == "RadixArk/Qwen3.8-Flash-Next-NVFP4"
    assert args.moe_backend == "offload"
    assert args.moe_cache_size is None
    assert args.moe_cpu_layers is None
    assert args.num_tokens is None
    assert args.max_seq_len_override == 8192
    assert args.max_extend_tokens == 8192
    assert not args.moe_collect_stats
    assert args.profiler == "torch"
    assert args.ple_cache_gib is None
    assert args.ple_mmap_advice is None
    assert not args.optimization_sweep
    assert not args.mtp_ab
    assert not args.mtp_sweep
    assert not args.mtp_real_prompts
    assert not args.mtp_expanded_prompts
    assert args.mtp_long_prompt_tokens == 2048
    assert not args.ple_mmap_sweep


def test_parser_accepts_opt_in_ple_cache_size() -> None:
    args = _parser("ft bench decode-profile").parse_args(["--ple-cache-gib", "4"])
    assert args.ple_cache_gib == 4.0


def test_parser_accepts_ple_mmap_advice() -> None:
    args = _parser("ft bench decode-profile").parse_args(
        ["--ple-mmap-advice", "random-willneed"]
    )
    assert args.ple_mmap_advice == "random-willneed"


def test_mtp_sweep_reuses_one_loaded_worker(monkeypatch) -> None:
    class Worker:
        max_supported_drafts = 2
        max_drafts = 2
        draft_p_min = 0.0
        adaptive_enabled = False
        pld_mode = None

        def reset_metrics(self):
            pass

    worker = Worker()
    llm = SimpleNamespace(engine=SimpleNamespace(mtp_worker=worker))
    calls = []

    def measure(target, _prompt, token_count):
        active = target.engine.mtp_worker
        key = (
            "baseline"
            if active is None
            else (
                active.max_drafts,
                active.draft_p_min,
                active.adaptive_enabled,
                active.pld_mode,
            )
        )
        calls.append((key, token_count))
        rate = {
            "baseline": 10.0,
            (1, 0.0, False, None): 11.0,
            (2, 0.0, False, None): 12.0,
            (2, 0.6, False, None): 13.0,
            (2, 0.0, True, None): 14.0,
            (2, 0.0, False, "hybrid"): 15.0,
            (2, 0.0, False, "only"): 16.0,
        }[key]
        return {
            "timing": {"decode_output_tokens_per_second": rate},
            "output_token_ids": [1, 2],
        }

    monkeypatch.setattr(
        "freetoken.benchmark.decode_profile._measure_generation", measure
    )

    result = _run_mtp_sweep(llm, [1], warmup_tokens=2, benchmark_tokens=3)

    assert len(calls) == 14
    assert result["variants"]["one_draft"]["speedup"] == pytest.approx(1.1)
    assert result["variants"]["two_drafts"]["speedup"] == pytest.approx(1.2)
    assert result["variants"]["two_drafts_p60"]["speedup"] == pytest.approx(1.3)
    assert result["variants"]["adaptive_two_drafts"]["speedup"] == pytest.approx(1.4)
    assert result["variants"]["pld_two_drafts"]["speedup"] == pytest.approx(1.5)
    assert result["variants"]["pld_only"]["speedup"] == pytest.approx(1.6)
    assert (
        worker.max_drafts,
        worker.draft_p_min,
        worker.adaptive_enabled,
        worker.pld_mode,
    ) == (2, 0.0, False, None)


def test_mtp_stats_include_cost_aware_controller_state() -> None:
    controller = {
        "cycles": 24,
        "selected_width": 2,
        "selection_cycle": 24,
        "survival": [0.9, 0.8],
        "cost_ms": [16.0, 29.0, 35.0],
        "samples": [8, 8, 8],
    }
    worker = SimpleNamespace(
        metrics=MTPMetrics(
            cycles=24,
            proposed_drafts=24,
            accepted_drafts=20,
            emitted_tokens=44,
            cycle_trace=[{"cycle": 1, "width": 2, "accepted": 1, "emitted": 2}],
        ),
        max_drafts=2,
        draft_p_min=0.0,
        adaptive_enabled=True,
        adaptive_disabled_uid=None,
        adaptive_decision_cycle=None,
        adaptive_cycles=64,
        adaptive_min_acceptance=0.75,
        adaptive_samples=8,
        adaptive_window=64,
        adaptive_probe_interval=16,
        pld_mode=None,
        uid=7,
        adaptive_summary=lambda: controller,
        timing=dict,
    )
    llm = SimpleNamespace(
        engine=SimpleNamespace(
            mtp_worker=worker,
            model=SimpleNamespace(mtp_offload_cache=None),
        )
    )

    result = _read_mtp_stats(llm)

    assert result["adaptive_controller"] == controller
    assert result["adaptive_min_acceptance"] == 0.75
    assert result["adaptive_samples"] == 8
    assert result["cycle_trace"] == [
        {"cycle": 1, "width": 2, "accepted": 1, "emitted": 2}
    ]
    assert result["cycle_trace_truncated"]
    assert result["tokens_per_cycle"] == pytest.approx(44 / 24)

    worker.metrics.cycle_trace.append(
        {"cycle": 2, "width": 1, "accepted": 0, "emitted": 1}
    )
    assert result["cycle_trace"] == [
        {"cycle": 1, "width": 2, "accepted": 1, "emitted": 2}
    ]


def test_mtp_sweep_warms_full_measured_trajectory(capsys) -> None:
    args = SimpleNamespace(mtp_sweep=True, warmup_tokens=32, benchmark_tokens=256)

    _ensure_mtp_sweep_warmup(args)

    assert args.warmup_tokens == 256
    assert "raising --warmup-tokens from 32 to 256" in capsys.readouterr().out


def test_non_mtp_sweep_keeps_short_warmup(capsys) -> None:
    args = SimpleNamespace(mtp_sweep=False, warmup_tokens=32, benchmark_tokens=256)

    _ensure_mtp_sweep_warmup(args)

    assert args.warmup_tokens == 32
    assert capsys.readouterr().out == ""


def test_mtp_prompt_suite_includes_requested_long_context():
    class LLM:
        @staticmethod
        def _tokenize_one(text):
            return torch.arange(max(1, len(text) // 8), dtype=torch.int32)

    prompts = _mtp_prompt_suite(LLM(), 128)

    assert set(prompts) == {"code", "prose", "tool_use", "long_context"}
    assert len(prompts["long_context"]) == 128
    assert all(prompts[name] for name in prompts)


def test_mtp_expanded_prompt_suite_adds_two_prompts_per_real_category():
    class LLM:
        @staticmethod
        def _tokenize_one(text):
            return torch.arange(max(1, len(text) // 8), dtype=torch.int32)

    prompts = _mtp_prompt_suite(LLM(), 128, expanded=True)

    assert set(prompts) == {
        "code",
        "code_rust",
        "code_cuda",
        "prose",
        "prose_science",
        "prose_decision",
        "tool_use",
        "tool_calendar",
        "tool_database",
        "long_context",
        "long_context_mixed",
        "long_context_structured",
    }
    assert all(len(prompts[name]) == 128 for name in prompts if name.startswith("long"))


def test_optimization_variant_updates_model_and_backend(monkeypatch) -> None:
    layer = SimpleNamespace(
        attn_hyper_connection=SimpleNamespace(_projection_fused=False),
        mlp_hyper_connection=SimpleNamespace(_projection_fused=False),
        mlp=SimpleNamespace(_shared_gate_fused=False),
    )
    backend = SimpleNamespace(_direct_attend=False)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=SimpleNamespace(op_list=[layer]))
    )
    llm = SimpleNamespace(engine=SimpleNamespace(model=model, attn_backend=backend))
    variant = _OptimizationVariant("test", True, True, True, True)
    monkeypatch.setenv("FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS", "2")

    _apply_optimization_variant(llm, variant)

    assert os.environ["FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS"] == "1"
    assert layer.attn_hyper_connection._projection_fused
    assert layer.mlp_hyper_connection._projection_fused
    assert layer.mlp._shared_gate_fused
    assert backend._direct_attend


def test_forced_greedy_tokens_replays_inputs_and_records_natural_choices() -> None:
    class Sampler:
        def sample(self, logits, _args):
            return torch.argmax(logits, dim=-1)

    sampler = Sampler()
    llm = SimpleNamespace(
        device=torch.device("cpu"), engine=SimpleNamespace(sampler=sampler)
    )

    with _force_greedy_tokens(llm, [7, 9], record_predictions=True) as capture:
        first = sampler.sample(torch.tensor([[0.0, 2.0, 1.0]]), None)
        second = sampler.sample(torch.tensor([[3.0, 1.0, 0.0]]), None)
        repeated = sampler.sample(torch.tensor([[0.0, 1.0, 4.0]]), None)

    assert first.tolist() == [7]
    assert second.tolist() == [9]
    assert repeated.tolist() == [9]
    assert capture.prediction_token_ids(3) == [1, 0, 2]
    assert sampler.sample(torch.tensor([[1.0, 4.0]]), None).tolist() == [1]


def test_first_mismatch_includes_length_difference() -> None:
    assert _first_mismatch([1, 2], [1, 3]) == 1
    assert _first_mismatch([1, 2], [1, 2]) is None
    assert _first_mismatch([1, 2], [1, 2, 3]) == 2


def test_phase_prompt_uses_distinct_in_vocab_ids() -> None:
    assert _phase_prompt(8, 3, 0, 10) == [8, 8, 8]
    assert _phase_prompt(8, 3, 2, 10) == [0, 0, 0]


def test_ple_replay_ngram_rows_match_model_hash_across_eos() -> None:
    prompt = [4, 5, 99]
    output = [6, 7, 8]
    multipliers = (3, 5, 7)
    vocab_sizes = (101, 103)
    offsets = (0, 101)

    actual = _ngram_rows(
        prompt,
        output,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        offsets=offsets,
    )
    expected = build_ngram_ids(
        torch.tensor(prompt + output),
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=torch.tensor(multipliers),
        vocab_sizes=torch.tensor(vocab_sizes),
        offsets=torch.tensor(offsets),
    )[len(prompt) :]

    assert actual == [tuple(row) for row in expected.tolist()]


def test_ple_replay_extracts_fixed_sweep_trajectory() -> None:
    report = {
        "phase_prompt_token_ids": {"wall": [1, 2]},
        "optimization_sweep": {"baseline": {"prime": {"output_token_ids": [3, 4]}}},
    }

    assert _trajectory_from_report(
        report, prompt_phase="wall", variant="baseline", window="prime"
    ) == ([1, 2], [3, 4])


def test_ple_mmap_sweep_reuses_one_fixed_trajectory(monkeypatch) -> None:
    prepared = []
    measured = []

    class Model:
        def prepare_ple_cold_replay(self, advice):
            prepared.append(advice)

    def measure(_llm, prompt, tokens):
        measured.append((prompt, tokens))
        return {"output_token_ids": [7, 8]}

    llm = SimpleNamespace(
        device=torch.device("cpu"),
        engine=SimpleNamespace(model=Model(), sampler=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "freetoken.benchmark.decode_profile._measure_generation", measure
    )

    result = _run_ple_mmap_sweep(llm, [1, 2], benchmark_tokens=2)

    assert prepared == ["normal", "random", "random-willneed"]
    assert len(measured) == 7
    assert result["reference_source"] == "local_generation"
    assert all(
        arm["cold_tokens_match_reference"] and arm["warm_tokens_match_reference"]
        for arm in result["results"].values()
    )


def test_cuda_step_timing_separates_engine_work_from_between_step_gap() -> None:
    recorder = object.__new__(_CudaStepRecorder)
    recorder.events = [
        _StepEvent("prefill", _Stamp(0.0), _Stamp(10.0)),
        _StepEvent("decode", _Stamp(20.0), _Stamp(25.0)),
        _StepEvent("decode", _Stamp(30.0), _Stamp(37.0)),
    ]

    timing = recorder.timing()

    assert timing["decode_steps"] == 2
    assert timing["engine_stream_ms"] == 12.0
    assert timing["decode_span_ms"] == 17.0
    assert timing["engine_stream_ms_per_step"] == 6.0
    assert timing["between_step_gap_ms_per_step"] == 2.5


def test_profile_summary_keeps_only_named_decode_descendants() -> None:
    decode = _ProfileEvent("FT.DecodeStep")
    gdn1 = _ProfileEvent("FT.GDN", parent=decode, cpu_us=100.0, device_us=300.0)
    gdn2 = _ProfileEvent("FT.GDN", parent=decode, cpu_us=200.0, device_us=500.0)
    prefill = _ProfileEvent("FT.PrefillStep")
    ignored_phase = _ProfileEvent(
        "FT.GDN", parent=prefill, cpu_us=999.0, device_us=999.0
    )
    ignored_name = _ProfileEvent(
        "aten::empty", parent=decode, cpu_us=999.0, device_us=999.0
    )

    rows = _profile_summary(
        [decode, gdn1, gdn2, prefill, ignored_phase, ignored_name], decode_steps=2
    )
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"FT.DecodeStep", "FT.GDN"}
    assert by_name["FT.GDN"]["calls_per_decode_step"] == 1.0
    assert by_name["FT.GDN"]["cpu_ms_per_decode_step"] == pytest.approx(0.15)
    assert by_name["FT.GDN"]["device_ms_per_decode_step"] == pytest.approx(0.4)


def test_profile_summary_includes_decode_scheduler_children() -> None:
    prepare = _ProfileEvent("FT.Scheduler.DecodePrepare", cpu_us=50.0)
    allocate = _ProfileEvent("FT.Scheduler.PageAllocate", parent=prepare, cpu_us=20.0)

    rows = _profile_summary([prepare, allocate], decode_steps=2)
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"FT.Scheduler.DecodePrepare", "FT.Scheduler.PageAllocate"}
    assert by_name["FT.Scheduler.PageAllocate"][
        "cpu_ms_per_decode_step"
    ] == pytest.approx(0.01)


def test_profile_summary_drops_nvtx_gpu_mirror_ranges() -> None:
    device_type = type("DeviceType", (), {"name": "CUDA"})()
    cpu = _ProfileEvent("FT.Scheduler.DecodePrepare", cpu_us=50.0)
    gpu_mirror = _ProfileEvent(
        "FT.Scheduler.DecodePrepare", device_us=20.0, device_type=device_type
    )

    rows = _profile_summary([cpu, gpu_mirror], decode_steps=1)

    assert len(rows) == 1
    assert rows[0]["calls"] == 1
    assert rows[0]["device_total_us"] == 0.0


def test_graph_execution_temporarily_forces_eager() -> None:
    runner = type("Runner", (), {"max_graph_bs": 8})()
    engine = type("Engine", (), {"graph_runner": runner})()
    llm = type("LLM", (), {"engine": engine})()

    with _graph_execution(llm, enabled=False):
        assert runner.max_graph_bs == 0
        assert engine._decode_profile_stage_graph_inputs is True
    assert runner.max_graph_bs == 8
    assert not hasattr(engine, "_decode_profile_stage_graph_inputs")


def test_ple_stats_are_read_and_reset_through_model_hooks() -> None:
    class Model:
        resets = 0

        def reset_ple_cache_stats(self):
            self.resets += 1

        def ple_cache_stats(self):
            return {"cache_hit_rate": 0.75, "decode_major_faults": 2}

    model = Model()
    llm = type("LLM", (), {"engine": type("Engine", (), {"model": model})()})()

    _reset_ple_stats(llm)

    assert model.resets == 1
    assert _read_ple_stats(llm) == {
        "cache_hit_rate": 0.75,
        "decode_major_faults": 2,
    }


class _FakeProfiler:
    def __init__(self, events: list[_ProfileEvent]):
        self._events = events
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def events(self) -> list[_ProfileEvent]:
        return self._events


def test_server_decode_profiler_logs_bounded_json_window() -> None:
    decode = _ProfileEvent("FT.DecodeStep", cpu_us=1000.0, device_us=2000.0)
    prepare = _ProfileEvent("FT.Scheduler.DecodePrepare", cpu_us=500.0)
    metadata = _ProfileEvent(
        "FT.Scheduler.AttentionMetadata", parent=prepare, cpu_us=300.0
    )
    fake = _FakeProfiler([decode, prepare, metadata])
    logs: list[str] = []
    collector = ServerDecodeProfiler(2, logs.append, profile_factory=lambda: fake)

    collector.start_if_needed()
    collector.complete_decode_step()
    assert collector.active
    collector.complete_decode_step()

    assert fake.started and fake.stopped
    assert not collector.active
    prefix, raw = logs[0].split(" ", 1)
    assert prefix == "FT.DecodeProfile"
    payload = json.loads(raw)
    assert payload["completed_decode_drains"] == 2
    assert payload["observed_decode_forwards"] == 1
    assert payload["normalization_decode_steps"] == 1
    assert payload["partial"] is False
    by_name = {row["name"]: row for row in payload["ranges"]}
    assert by_name["FT.Scheduler.AttentionMetadata"][
        "cpu_ms_per_decode_step"
    ] == pytest.approx(0.3)


def test_disabled_decode_range_reuses_static_noop() -> None:
    from freetoken.utils import torch_utils

    previous = torch_utils.decode_profile_ranges_enabled()
    try:
        torch_utils.configure_decode_profile_ranges(False)
        first = torch_utils.decode_profile_range("one")
        second = torch_utils.decode_profile_range("two")
        assert first is second
        assert first.__enter__() is None
        assert first.__exit__(None, None, None) is None
    finally:
        torch_utils.configure_decode_profile_ranges(previous)
