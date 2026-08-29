r"""Offline decode profiler for Qwen3.8-Flash-Next (one model load, no server).

The default run matches ``ft serve --max-running-requests 1``: one visible GPU,
NVFP4 offload, automatic expert/KV cache sizing, no forced CPU MoE layer split, and
an 8192-token maximum sequence length.
It loads the model once, warms it, records an unprofiled CUDA-graph timing window,
then records short CUDA-graph and eager torch.profiler traces.  The eager trace is
intentional: CUDA-graph replay preserves the real latency but collapses Python/NVTX
layer ranges; eager execution exposes the already-present layer annotations and the
fine-grained QSA/MoE/GDN ranges while running the same kernels.

``--optimization-sweep`` obtains one greedy baseline trajectory, forces that token
trajectory through every measured variant, and separately records each variant's
natural greedy choices.  This keeps PLE and MoE routing inputs comparable even when
small numerical differences would otherwise make generation diverge.

Typical invocation (the environment is shown explicitly for reproducibility)::

    PYTHONPATH=/home/stran/AI/FreeToken/python:/home/stran/AI/freetoken-runtime/.venv/lib/python3.12/site-packages \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    TVM_FFI_CUDA_ARCH_LIST=12.0 TORCH_CUDA_ARCH_LIST=12.0 \
    FREETOKEN_ALLOW_CUDA_MISMATCH=1 \
    ft bench decode-profile \
      --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
      --moe-backend offload --moe-cache-auto --max-seq-len-override 8192 \
      --profiler torch -o profiles/qwen38-decode

For Nsight Systems, replace ``--profiler torch`` with ``--profiler nvtx`` and run::

    nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o profiles/qwen38-decode \
      ft bench decode-profile ... --profiler nvtx

Nsight capture starts only after load/warmup/wall timing, avoiding a multi-minute
model-load trace.  In ``report.json``, ``decode_span_ms_per_step`` is the observed
steady decode clock (including between-step host gaps), while
``engine_stream_ms_per_step`` is time bracketed around Engine.forward_batch.  Their
difference exposes scheduler/host gaps.  Torch trace rows are nested and therefore
must not all be added together: use the top-level GDN/QSA/MoE/PLE/hyper-connection
rows for the model split, then their child rows for the internal split.

PLE locality is reported for every phase under ``<phase>.ple``.  In particular,
``cache_hit_rate`` and ``decode_major_faults`` are scoped to decode PLE row reads;
``process_major_faults_during_generation`` is the wider process cross-check.  The
cache remains disabled unless ``--ple-cache-gib`` or
``FREETOKEN_QWEN4_PLE_CACHE_GIB`` is set.  ``--ple-mmap-advice`` selects the
checkpoint mapping hint independently of that optional row cache.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import resource
import socket
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from freetoken.profiling import summarize_decode_profile_events as _profile_summary

_DEFAULT_MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative finite number, got {value!r}"
        )
    return parsed


def _dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _phase_prompt(
    token_id: int, length: int, phase_offset: int, vocab_size: int
) -> list[int]:
    """Use a nearby prompt per phase so PLE mmap faults are not all hidden by warmup."""
    return [(token_id + phase_offset) % vocab_size] * length


def _mtp_prompt_suite(
    llm, long_prompt_tokens: int, *, expanded: bool = False
) -> dict[str, list[int]]:
    texts = {
        "code": (
            "Implement a Python async task scheduler with cancellation, bounded "
            "concurrency, type annotations, tests, and a short explanation of the "
            "fairness guarantees."
        ),
        "prose": (
            "Explain why a memory-mapped model can have excellent steady-state "
            "throughput but slower first-token batches. Compare page faults, "
            "readahead, pinned memory, and practical latency in clear prose."
        ),
        "tool_use": (
            "You can call get_weather(city, date) and search_hotels(city, check_in, "
            "nights). Plan a three-night trip to Kyoto next week. Return the tool "
            "calls as compact JSON before writing any recommendation."
        ),
    }
    if expanded:
        texts.update(
            {
                "code_rust": (
                    "Write a Rust implementation of a bounded least-recently-used "
                    "cache. Use generics, avoid unnecessary cloning, explain the "
                    "ownership choices, and include concurrent stress tests."
                ),
                "code_cuda": (
                    "Diagnose a CUDA stream race in a Python inference server where "
                    "an asynchronous device-to-host token copy is reused too early. "
                    "Propose the smallest fix and a deterministic regression test."
                ),
                "prose_science": (
                    "Explain to an interested undergraduate how gravitational-wave "
                    "detectors distinguish a real signal from local vibration. Use "
                    "an analogy, then describe the limits of that analogy."
                ),
                "prose_decision": (
                    "Compare buying and leasing an expensive machine for a small "
                    "research lab. Discuss utilization, maintenance risk, financing, "
                    "and uncertainty without pretending there is one universal answer."
                ),
                "tool_calendar": (
                    "You can call list_events(start, end), find_rooms(start, end, "
                    "capacity), and create_event(title, start, end, room). Schedule a "
                    "45-minute review next Tuesday afternoon for eight people. Emit "
                    "compact JSON tool calls before a confirmation message."
                ),
                "tool_database": (
                    "You can call get_customer(email), search_orders(customer_id, "
                    "since), and create_support_ticket(order_id, reason, priority). "
                    "Investigate a duplicate charge reported yesterday and emit only "
                    "the required JSON tool calls before summarizing the result."
                ),
            }
        )
    result = {name: llm._tokenize_one(text).tolist() for name, text in texts.items()}
    long_texts = {
        "long_context": (
            "A systems engineer records one observation at a time: model state, cache "
            "residency, page-fault behavior, generated tokens, and elapsed time. The "
            "record distinguishes cold-start cost from steady-state throughput and "
            "avoids comparing measurements made with different memory policies. "
        )
    }
    if expanded:
        long_texts.update(
            {
                "long_context_mixed": (
                    "Section one describes a coastal wetland survey with salinity, "
                    "bird counts, and seasonal uncertainty. Section two specifies a "
                    "message queue with delivery acknowledgements and retry limits. "
                    "Section three compares two municipal budget proposals and lists "
                    "their assumptions. Preserve distinctions between the sections. "
                ),
                "long_context_structured": (
                    "Incident record: timestamp, component, observation, hypothesis, "
                    "test, and result. A failed hypothesis stays in the record. A new "
                    "test changes one variable. The final summary separates confirmed "
                    "causes, contributing conditions, and unresolved questions. "
                ),
            }
        )
    for name, text in long_texts.items():
        repeated = text
        while True:
            token_ids = llm._tokenize_one(repeated).tolist()
            if len(token_ids) >= long_prompt_tokens:
                result[name] = token_ids[:long_prompt_tokens]
                break
            repeated += text
    return result


@dataclass
class _StepEvent:
    phase: str
    started: Any
    ended: Any


@dataclass(frozen=True)
class _OptimizationVariant:
    name: str
    router_batch_one: bool
    qsa_direct: bool
    hc_projection: bool
    shared_gate: bool


@dataclass(frozen=True)
class _MTPVariant:
    name: str
    max_drafts: int
    draft_p_min: float
    adaptive: bool = False


_OPTIMIZATION_SWEEP = (
    _OptimizationVariant("baseline", False, False, False, False),
    _OptimizationVariant("router", True, False, False, False),
    _OptimizationVariant("router_hc", True, False, True, False),
    _OptimizationVariant("router_hc_gate", True, False, True, True),
    _OptimizationVariant("all", True, True, True, True),
)

_MTP_SWEEP = (
    _MTPVariant("one_draft", 1, 0.0),
    _MTPVariant("two_drafts", 2, 0.0),
    _MTPVariant("two_drafts_p60", 2, 0.6),
    _MTPVariant("adaptive_two_drafts", 2, 0.0, True),
)


class _CudaStepRecorder:
    """Bracket Engine.forward_batch without synchronizing each token."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.events: list[_StepEvent] = []

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        import torch

        engine = self.llm.engine
        original = engine.forward_batch

        def timed(batch, args):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record(engine.stream)
            result = original(batch, args)
            ended.record(engine.stream)
            self.events.append(
                _StepEvent("decode" if batch.is_decode else "prefill", started, ended)
            )
            return result

        engine.forward_batch = timed
        try:
            yield
        finally:
            # The instance normally inherits this method; removing the temporary shadow
            # restores normal descriptor binding without retaining the recorder closure.
            del engine.forward_batch

    def timing(self) -> dict[str, float | int | None]:
        decode = [event for event in self.events if event.phase == "decode"]
        if not decode:
            return {
                "decode_steps": 0,
                "decode_span_ms": None,
                "decode_span_ms_per_step": None,
                "engine_stream_ms": None,
                "engine_stream_ms_per_step": None,
                "between_step_gap_ms_per_step": None,
            }
        engine_ms = sum(event.started.elapsed_time(event.ended) for event in decode)
        span_ms = decode[0].started.elapsed_time(decode[-1].ended)
        count = len(decode)
        return {
            "decode_steps": count,
            "decode_span_ms": span_ms,
            "decode_span_ms_per_step": span_ms / count,
            "engine_stream_ms": engine_ms,
            "engine_stream_ms_per_step": engine_ms / count,
            "between_step_gap_ms_per_step": max(0.0, span_ms - engine_ms) / count,
        }


@contextlib.contextmanager
def _graph_execution(llm, enabled: bool) -> Iterator[None]:
    runner = llm.engine.graph_runner
    previous = runner.max_graph_bs
    had_stage_flag = hasattr(llm.engine, "_decode_profile_stage_graph_inputs")
    previous_stage_flag = getattr(
        llm.engine, "_decode_profile_stage_graph_inputs", False
    )
    runner.max_graph_bs = previous if enabled else 0
    llm.engine._decode_profile_stage_graph_inputs = not enabled
    try:
        yield
    finally:
        runner.max_graph_bs = previous
        if had_stage_flag:
            llm.engine._decode_profile_stage_graph_inputs = previous_stage_flag
        else:
            del llm.engine._decode_profile_stage_graph_inputs


def _set_sweep_startup_environment() -> None:
    os.environ["FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS"] = "2"
    os.environ["FREETOKEN_QWEN4_QSA_DIRECT_ATTEND"] = "0"
    os.environ["FREETOKEN_QWEN_HC_PROJECTION_FUSED"] = "0"
    os.environ["FREETOKEN_QWEN_SHARED_GATE_FUSED"] = "0"


def _qsa_decode_backend(llm):
    backend = llm.engine.attn_backend
    while hasattr(backend, "decode_backend"):
        backend = backend.decode_backend
    if not hasattr(backend, "_direct_attend"):
        raise RuntimeError("optimization sweep needs the Qwen4 QSA sparse backend")
    return backend


def _apply_optimization_variant(llm, variant: _OptimizationVariant) -> None:
    os.environ["FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS"] = (
        "1" if variant.router_batch_one else "2"
    )
    model = getattr(llm.engine.model, "model", None)
    layers = getattr(getattr(model, "layers", None), "op_list", None)
    if layers is None:
        raise RuntimeError("optimization sweep needs Qwen4 decoder layers")
    for layer in layers:
        layer.attn_hyper_connection._projection_fused = variant.hc_projection
        layer.mlp_hyper_connection._projection_fused = variant.hc_projection
        layer.mlp._shared_gate_fused = variant.shared_gate
    _qsa_decode_backend(llm)._direct_attend = variant.qsa_direct


def _recapture_graphs(llm) -> float:
    pool = llm.engine.linear_state_pool
    if pool is None:
        raise RuntimeError("optimization sweep needs the Qwen4 linear-state pool")
    started = time.perf_counter()
    llm.rebuild_cache(num_mamba_slots=pool.num_slots - 1)
    return time.perf_counter() - started


class _ForcedGreedyCapture:
    def __init__(self, token_ids: list[int], device, record_predictions: bool) -> None:
        import torch

        if not token_ids:
            raise ValueError("forced decode needs at least one token")
        self.forced = torch.tensor(token_ids, dtype=torch.int64, device=device)
        self.record_predictions = record_predictions
        self.predictions: list[Any] = []
        self.calls = 0

    def sample(self, logits, _args):
        import torch

        if logits.shape[0] != 1:
            raise RuntimeError("forced decode supports only max_running_req=1")
        if self.record_predictions:
            self.predictions.append(torch.argmax(logits, dim=-1))
        index = min(self.calls, self.forced.numel() - 1)
        self.calls += 1
        return self.forced[index : index + 1]

    def prediction_token_ids(self, count: int) -> list[int]:
        import torch

        if not self.record_predictions:
            return []
        if len(self.predictions) < count:
            raise RuntimeError(
                f"forced decode recorded {len(self.predictions)} predictions for {count} output tokens"
            )
        return torch.cat(self.predictions[:count]).cpu().tolist()


@contextlib.contextmanager
def _force_greedy_tokens(
    llm, token_ids: list[int], *, record_predictions: bool
) -> Iterator[_ForcedGreedyCapture]:
    sampler = llm.engine.sampler
    had_shadow = "sample" in vars(sampler)
    previous = vars(sampler).get("sample")
    capture = _ForcedGreedyCapture(token_ids, llm.device, record_predictions)
    sampler.sample = capture.sample
    try:
        yield capture
    finally:
        if had_shadow:
            sampler.sample = previous
        else:
            del sampler.sample


def _first_mismatch(reference: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, actual)):
        if left != right:
            return index
    return None if len(reference) == len(actual) else min(len(reference), len(actual))


def _run_optimization_sweep(
    llm,
    prompt: list[int],
    *,
    prime_tokens: int,
    benchmark_tokens: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    reference_tokens: list[int] | None = None
    for index, variant in enumerate(_OPTIMIZATION_SWEEP):
        print(f"Optimization sweep: {variant.name} ...", flush=True)
        _apply_optimization_variant(llm, variant)
        recapture_seconds = 0.0 if index == 0 else _recapture_graphs(llm)
        if reference_tokens is None:
            if prime_tokens != benchmark_tokens:
                raise ValueError(
                    "optimization sweep needs equal prime and benchmark token counts"
                )
            prime = _measure_generation(llm, prompt, prime_tokens)
            reference_tokens = prime["output_token_ids"]
        else:
            with _force_greedy_tokens(llm, reference_tokens, record_predictions=False):
                prime = _measure_generation(llm, prompt, prime_tokens)
        with _force_greedy_tokens(
            llm, reference_tokens, record_predictions=True
        ) as capture:
            measured = _measure_generation(llm, prompt, benchmark_tokens)
        output_tokens = measured["output_token_ids"]
        natural_tokens = capture.prediction_token_ids(len(output_tokens))
        matching_predictions = sum(
            left == right for left, right in zip(reference_tokens, natural_tokens)
        )
        results[variant.name] = {
            "settings": {
                "router_batch_one": variant.router_batch_one,
                "qsa_direct": variant.qsa_direct,
                "hc_projection": variant.hc_projection,
                "shared_gate": variant.shared_gate,
            },
            "recapture_seconds": recapture_seconds,
            "prime": prime,
            "measured": measured,
            "forced_tokens_match_reference": output_tokens == reference_tokens,
            "natural_output_token_ids": natural_tokens,
            "matches_baseline_tokens": natural_tokens == reference_tokens,
            "matching_baseline_predictions": matching_predictions,
            "matching_baseline_prediction_rate": matching_predictions
            / len(reference_tokens),
            "first_baseline_prediction_mismatch": _first_mismatch(
                reference_tokens, natural_tokens
            ),
        }
    return results


def _run_ple_mmap_sweep(
    llm,
    prompt: list[int],
    *,
    benchmark_tokens: int,
    reference_tokens: list[int] | None = None,
) -> dict[str, Any]:
    model = llm.engine.model
    prepare_cold = getattr(model, "prepare_ple_cold_replay", None)
    if prepare_cold is None:
        raise RuntimeError("PLE mmap sweep needs a model cold-replay hook")
    reference = None
    if reference_tokens is None:
        print("PLE mmap sweep: reference trajectory ...", flush=True)
        reference = _measure_generation(llm, prompt, benchmark_tokens)
        reference_tokens = reference["output_token_ids"]
    results = {}
    for advice in ("normal", "random", "random-willneed"):
        print(f"PLE mmap sweep: {advice} cold ...", flush=True)
        prepare_cold(advice)
        with _force_greedy_tokens(llm, reference_tokens, record_predictions=False):
            cold = _measure_generation(llm, prompt, benchmark_tokens)
        print(f"PLE mmap sweep: {advice} warm ...", flush=True)
        with _force_greedy_tokens(llm, reference_tokens, record_predictions=False):
            warm = _measure_generation(llm, prompt, benchmark_tokens)
        results[advice] = {
            "cold": cold,
            "warm": warm,
            "cold_tokens_match_reference": cold["output_token_ids"] == reference_tokens,
            "warm_tokens_match_reference": warm["output_token_ids"] == reference_tokens,
        }
    return {
        "reference": reference,
        "reference_source": (
            "local_generation"
            if reference is not None
            else "optimization_sweep_baseline"
        ),
        "results": results,
    }


def _rusage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _proc_io() -> dict[str, int] | None:
    try:
        with open("/proc/self/io", encoding="utf-8") as file:
            return {
                key: int(value) for key, value in (line.split(":", 1) for line in file)
            }
    except OSError:
        return None


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def _reset_moe_stats(llm) -> None:
    caches = (
        llm.engine.moe_offload_cache,
        getattr(llm.engine.model, "mtp_offload_cache", None),
    )
    for cache in caches:
        if cache is not None and cache.collect_stats:
            cache.reset_stats()


def _read_moe_stats(llm) -> dict[str, Any] | None:
    cache = llm.engine.moe_offload_cache
    if cache is None or not cache.collect_stats:
        return None
    return {
        "aggregate": cache.decode_miss_stats(),
        "per_layer": cache.decode_miss_stats_per_layer()["per_layer"],
        "routing": cache.decode_routing_stats(),
        "cpu_layer_ids": sorted(cache.cpu_layer_ids),
    }


def _reset_ple_stats(llm) -> None:
    reset = getattr(llm.engine.model, "reset_ple_cache_stats", None)
    if reset is not None:
        reset()


def _read_ple_stats(llm) -> dict[str, Any] | None:
    read = getattr(llm.engine.model, "ple_cache_stats", None)
    return read() if read is not None else None


def _reset_mtp_stats(llm) -> None:
    worker = llm.engine.mtp_worker
    if worker is not None:
        worker.reset_metrics()


def _read_mtp_stats(llm) -> dict[str, Any] | None:
    worker = llm.engine.mtp_worker
    if worker is None:
        return None
    metrics = worker.metrics
    cache = getattr(llm.engine.model, "mtp_offload_cache", None)
    cache_stats = None
    if cache is not None and cache.collect_stats:
        cache_stats = {
            "aggregate": cache.decode_miss_stats(),
            "per_layer": cache.decode_miss_stats_per_layer()["per_layer"],
            "routing": cache.decode_routing_stats(),
            "cache_size": cache.cache_size,
        }
    return {
        "cycles": metrics.cycles,
        "max_drafts": worker.max_drafts,
        "draft_p_min": worker.draft_p_min,
        "adaptive_enabled": worker.adaptive_enabled,
        "adaptive_disabled": worker.adaptive_disabled_uid == worker.uid,
        "adaptive_decision_cycle": worker.adaptive_decision_cycle,
        "adaptive_cycles": worker.adaptive_cycles,
        "adaptive_min_acceptance": worker.adaptive_min_acceptance,
        "adaptive_samples": worker.adaptive_samples,
        "adaptive_window": worker.adaptive_window,
        "adaptive_probe_interval": worker.adaptive_probe_interval,
        "adaptive_controller": worker.adaptive_summary(),
        "cycle_trace": list(metrics.cycle_trace),
        "cycle_trace_truncated": metrics.cycles > len(metrics.cycle_trace),
        "proposed_drafts": metrics.proposed_drafts,
        "accepted_drafts": metrics.accepted_drafts,
        "acceptance_rate": metrics.acceptance_rate,
        "emitted_tokens": metrics.emitted_tokens,
        "tokens_per_cycle": (
            metrics.emitted_tokens / metrics.cycles if metrics.cycles else None
        ),
        "phase_timing": worker.timing(),
        "moe": cache_stats,
    }


def _measure_generation(llm, prompt: list[int], max_tokens: int) -> dict[str, Any]:
    import torch
    from freetoken.core import SamplingParams

    _reset_moe_stats(llm)
    _reset_ple_stats(llm)
    recorder = _CudaStepRecorder(llm)
    before = _rusage()
    before_io = _proc_io()
    started = time.perf_counter()
    with recorder.installed():
        outputs = llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens),
        )
    torch.cuda.synchronize(llm.device)
    wall = time.perf_counter() - started
    after = _rusage()
    after_io = _proc_io()
    timing = recorder.timing()
    steps = int(timing["decode_steps"])
    decode_output_tokens = max(0, len(outputs[0]["token_ids"]) - 1)
    decode_span_ms = timing["decode_span_ms"]
    timing["decode_output_tokens"] = decode_output_tokens
    timing["decode_ms_per_output_token"] = (
        decode_span_ms / decode_output_tokens
        if decode_span_ms is not None and decode_output_tokens
        else None
    )
    timing["decode_output_tokens_per_second"] = (
        1000.0 * decode_output_tokens / decode_span_ms
        if decode_span_ms is not None and decode_span_ms > 0
        else None
    )
    faults = _delta(after, before)
    ple = _read_ple_stats(llm)
    if ple is not None:
        # The model-local counter brackets only PLE source-row reads.  Keep the
        # process-wide generation delta alongside it for cross-checking.
        ple["process_major_faults_during_generation"] = faults["major_faults"]
        ple["process_major_faults_per_decode_step"] = (
            faults["major_faults"] / steps if steps else None
        )
    return {
        "requested_output_tokens": max_tokens,
        "returned_output_tokens": len(outputs[0]["token_ids"]),
        "wall_seconds_including_prefill": wall,
        "timing": timing,
        "resource_delta": faults,
        "process_io_delta": (
            _delta(after_io, before_io)
            if before_io is not None and after_io is not None
            else None
        ),
        "major_faults_per_decode_step": faults["major_faults"] / steps
        if steps
        else None,
        "ple": ple,
        "moe": _read_moe_stats(llm),
        "mtp": _read_mtp_stats(llm),
        "output_token_ids": outputs[0]["token_ids"],
    }


def _run_mtp_ab(
    llm,
    prompt: list[int],
    *,
    warmup_tokens: int,
    benchmark_tokens: int,
) -> dict[str, Any]:
    worker = llm.engine.mtp_worker
    if worker is None:
        raise RuntimeError("MTP A/B needs FREETOKEN_QWEN4_MTP=1")

    print("MTP A/B: ordinary greedy warmup ...", flush=True)
    llm.engine.mtp_worker = None
    try:
        baseline_warmup = _measure_generation(llm, prompt, warmup_tokens)
        print("MTP A/B: ordinary greedy measurement ...", flush=True)
        baseline = _measure_generation(llm, prompt, benchmark_tokens)
    finally:
        llm.engine.mtp_worker = worker

    print("MTP A/B: speculative warmup ...", flush=True)
    _reset_mtp_stats(llm)
    mtp_warmup = _measure_generation(llm, prompt, warmup_tokens)
    print("MTP A/B: speculative measurement ...", flush=True)
    _reset_mtp_stats(llm)
    mtp = _measure_generation(llm, prompt, benchmark_tokens)
    baseline_rate = baseline["timing"]["decode_output_tokens_per_second"]
    mtp_rate = mtp["timing"]["decode_output_tokens_per_second"]
    return {
        "prompt_token_ids": prompt,
        "baseline_warmup": baseline_warmup,
        "mtp_warmup": mtp_warmup,
        "baseline": baseline,
        "mtp": mtp,
        "outputs_match": baseline["output_token_ids"] == mtp["output_token_ids"],
        "first_output_mismatch": _first_mismatch(
            baseline["output_token_ids"], mtp["output_token_ids"]
        ),
        "speedup": (
            mtp_rate / baseline_rate
            if baseline_rate is not None and mtp_rate is not None and baseline_rate > 0
            else None
        ),
    }


def _run_mtp_sweep(
    llm,
    prompt: list[int],
    *,
    warmup_tokens: int,
    benchmark_tokens: int,
) -> dict[str, Any]:
    worker = llm.engine.mtp_worker
    if worker is None or worker.max_supported_drafts < 2:
        raise RuntimeError(
            "MTP sweep needs FREETOKEN_QWEN4_MTP=1 and FREETOKEN_QWEN4_MTP_MAX_DRAFTS=2"
        )

    print("MTP sweep: ordinary greedy warmup ...", flush=True)
    llm.engine.mtp_worker = None
    try:
        baseline_warmup = _measure_generation(llm, prompt, warmup_tokens)
        print("MTP sweep: ordinary greedy measurement ...", flush=True)
        baseline = _measure_generation(llm, prompt, benchmark_tokens)
    finally:
        llm.engine.mtp_worker = worker

    baseline_rate = baseline["timing"]["decode_output_tokens_per_second"]
    original = (
        worker.max_drafts,
        worker.draft_p_min,
        worker.adaptive_enabled,
    )
    variants = {}
    try:
        for variant in _MTP_SWEEP:
            worker.max_drafts = variant.max_drafts
            worker.draft_p_min = variant.draft_p_min
            worker.adaptive_enabled = variant.adaptive
            print(f"MTP sweep: {variant.name} warmup ...", flush=True)
            _reset_mtp_stats(llm)
            warmup = _measure_generation(llm, prompt, warmup_tokens)
            print(f"MTP sweep: {variant.name} measurement ...", flush=True)
            _reset_mtp_stats(llm)
            measured = _measure_generation(llm, prompt, benchmark_tokens)
            rate = measured["timing"]["decode_output_tokens_per_second"]
            variants[variant.name] = {
                "max_drafts": variant.max_drafts,
                "draft_p_min": variant.draft_p_min,
                "adaptive": variant.adaptive,
                "warmup": warmup,
                "measured": measured,
                "outputs_match_baseline": (
                    baseline["output_token_ids"] == measured["output_token_ids"]
                ),
                "first_output_mismatch": _first_mismatch(
                    baseline["output_token_ids"], measured["output_token_ids"]
                ),
                "speedup": (
                    rate / baseline_rate
                    if rate is not None
                    and baseline_rate is not None
                    and baseline_rate > 0
                    else None
                ),
            }
    finally:
        (
            worker.max_drafts,
            worker.draft_p_min,
            worker.adaptive_enabled,
        ) = original

    return {
        "prompt_token_ids": prompt,
        "baseline_warmup": baseline_warmup,
        "baseline": baseline,
        "variants": variants,
    }


def _run_torch_trace(
    llm,
    prompt: list[int],
    max_tokens: int,
    *,
    graph: bool,
    trace_path: Path,
) -> dict[str, Any]:
    import torch
    from freetoken.utils import decode_profile_range, enable_record_function_ranges

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    enable_record_function_ranges(True)
    try:
        with (
            torch.profiler.profile(activities=activities) as profiler,
            _graph_execution(llm, graph),
            decode_profile_range(f"FT.Bench.{'Graph' if graph else 'Eager'}Trace"),
        ):
            measured = _measure_generation(llm, prompt, max_tokens)
    finally:
        enable_record_function_ranges(False)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    table_path = trace_path.with_suffix(".table.txt")
    table_path.write_text(
        profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=250)
        + "\n"
    )
    measured["trace"] = str(trace_path)
    measured["operator_table"] = str(table_path)
    measured["ranges"] = _profile_summary(
        profiler.events(), int(measured["timing"]["decode_steps"])
    )
    return measured


def _run_nvtx_capture(
    llm, prompt: list[int], max_tokens: int, eager: bool
) -> dict[str, Any]:
    import torch
    from freetoken.utils import decode_profile_range

    torch.cuda.profiler.start()
    try:
        result: dict[str, Any] = {}
        with _graph_execution(llm, True), decode_profile_range("FT.Bench.GraphTrace"):
            result["graph"] = _measure_generation(llm, prompt, max_tokens)
        if eager:
            with (
                _graph_execution(llm, False),
                decode_profile_range("FT.Bench.EagerTrace"),
            ):
                result["eager"] = _measure_generation(llm, prompt, max_tokens)
        torch.cuda.synchronize(llm.device)
        return result
    finally:
        torch.cuda.profiler.stop()


def _model_geometry(llm) -> dict[str, Any]:
    config = llm.config
    model = config.model_config
    linear_ids = [i for i in range(model.num_layers) if model.is_linear_layer(i)]
    full_ids = [i for i in range(model.num_layers) if i not in linear_ids]
    qwen = model.qwen4_args
    return {
        "num_layers": model.num_layers,
        "linear_attention_layers": linear_ids,
        "full_attention_layers": full_ids,
        "ple_layers": list(qwen.ple_layer_ids) if qwen is not None else [],
        "qsa_selection_size": (
            qwen.indexer_budget + qwen.indexer_compress_ratio - 1
            if qwen is not None
            else None
        ),
        "qsa_score_width_blocks": (
            llm.engine.max_seq_len // qwen.indexer_compress_ratio
            if qwen is not None
            else None
        ),
        "ple_rows_per_token": (
            (qwen.ngram_size - 1) * qwen.heads_per_ngram if qwen is not None else None
        ),
        "ple_fp8_h2d_bytes_per_token": qwen.ple_embed_dim if qwen is not None else None,
        "resolved_cache_type": config.cache_type,
        "resolved_attention_backend": config.attention_backend,
        "resolved_moe_backend": config.moe_backend,
        "resolved_moe_cache_size": (
            llm.engine.moe_offload_cache.cache_size
            if llm.engine.moe_offload_cache is not None
            else 0
        ),
        "resolved_cpu_moe_layer_ids": (
            sorted(llm.engine.moe_offload_cache.cpu_layer_ids)
            if llm.engine.moe_offload_cache is not None
            else []
        ),
        "resolved_num_pages": llm.engine.num_pages,
        "resolved_kv_tokens": llm.engine.num_pages * config.page_size,
        "configured_max_running_req": config.max_running_req,
        "configured_max_seq_len": config.max_seq_len,
        "configured_num_token_override": config.num_token_override,
        "engine_max_seq_len": llm.engine.max_seq_len,
        "cuda_graph_batch_sizes": list(llm.engine.graph_runner.graph_bs_list),
        "mtp_enabled": llm.engine.mtp_worker is not None,
        "mtp_moe_cache_size": (
            llm.engine.model.mtp_offload_cache.cache_size
            if llm.engine.mtp_worker is not None
            else 0
        ),
        "mtp_max_supported_drafts": (
            llm.engine.mtp_worker.max_supported_drafts
            if llm.engine.mtp_worker is not None
            else 0
        ),
        "mtp_max_drafts": (
            llm.engine.mtp_worker.max_drafts if llm.engine.mtp_worker is not None else 0
        ),
        "mtp_draft_p_min": (
            llm.engine.mtp_worker.draft_p_min
            if llm.engine.mtp_worker is not None
            else None
        ),
    }


def _print_timing(label: str, result: dict[str, Any]) -> None:
    timing = result["timing"]
    rate = timing.get("decode_output_tokens_per_second")
    rate_text = f"{rate:.2f} tok/s" if rate is not None else "n/a"
    print(
        f"  {label:<12} steps={timing['decode_steps']:>3}  "
        f"cycle={timing['decode_span_ms_per_step']:.3f} ms  "
        f"engine={timing['engine_stream_ms_per_step']:.3f}  "
        f"host-gap={timing['between_step_gap_ms_per_step']:.3f}  rate={rate_text}  "
        f"majflt={result['major_faults_per_decode_step']:.3f}/token"
    )
    ple = result.get("ple")
    if ple is not None:
        hit_rate = ple.get("cache_hit_rate")
        hit_text = f"{hit_rate:.3%}" if hit_rate is not None else "n/a"
        print(
            f"  {'PLE cache':<12} hit={hit_text}  hits={ple['cache_hits']}  "
            f"misses={ple['cache_misses']}  "
            f"ple_decode_majflt={ple['decode_major_faults']}"
        )


def _print_ranges(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n  {label} named ranges (nested; do not sum parent + child rows)")
    print(f"    {'range':<38} {'calls/tok':>9} {'CPU ms/tok':>11} {'CUDA ms/tok':>12}")
    for row in rows:
        if (
            row["name"] == "FT.DecodeStep"
            or row["name"].startswith(
                (
                    "FT.GDN",
                    "FT.QSA",
                    "FT.MoE",
                    "FT.PLE",
                    "FT.Hyper",
                    "FT.MTP",
                    "FT.Graph",
                    "FT.Eager",
                    "FT.Sampler",
                    "FT.Scheduler",
                    "FT.AttentionMetadata",
                )
            )
            or row["name"] in {"QSA", "LMHead", "Sampler"}
        ):
            print(
                f"    {row['name']:<38} {row['calls_per_decode_step']:>9.2f} "
                f"{row['cpu_ms_per_decode_step']:>11.3f} "
                f"{row['device_ms_per_decode_step']:>12.3f}"
            )


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "--model-path", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        help="use fabricated weights to debug initialization without checkpoint I/O",
    )
    parser.add_argument(
        "--moe-backend", choices=("offload", "cpu", "hybrid"), default="offload"
    )
    moe_cache = parser.add_mutually_exclusive_group()
    moe_cache.add_argument(
        "--moe-cache-size",
        type=_positive_int,
        default=None,
        help="fixed expert-cache slots (default: use the server's automatic sizing)",
    )
    moe_cache.add_argument(
        "--moe-cache-auto",
        action="store_true",
        help="size expert and KV caches from available VRAM (the default when no size is given)",
    )
    parser.add_argument(
        "--moe-cpu-layers",
        default=None,
        help="force a CPU MoE layer spec; unset uses the same automatic policy as ft serve",
    )
    parser.add_argument("--moe-cpu-threads", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--moe-collect-stats",
        action="store_true",
        help="collect MoE routing/cache counters (off by default for serve parity)",
    )
    parser.add_argument(
        "--ple-cache-gib",
        type=_nonnegative_float,
        default=None,
        help=(
            "opt-in model-wide PLE hot-row cache GiB; overrides "
            "FREETOKEN_QWEN4_PLE_CACHE_GIB (default: mmap only)"
        ),
    )
    parser.add_argument(
        "--ple-mmap-advice",
        choices=("normal", "random", "random-willneed"),
        default=None,
        help=(
            "PLE checkpoint mmap policy; overrides FREETOKEN_QWEN4_PLE_MMAP_ADVICE "
            "(default: random-willneed on supported platforms)"
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=_positive_int,
        default=None,
        help="fixed KV-cache capacity; unset lets --moe-cache-auto choose it like ft serve",
    )
    parser.add_argument("--max-seq-len-override", type=_positive_int, default=8192)
    parser.add_argument("--max-extend-tokens", type=_positive_int, default=8192)
    parser.add_argument("--prompt-tokens", type=_positive_int, default=32)
    parser.add_argument("--prompt-token-id", type=_nonnegative_int, default=1)
    parser.add_argument("--warmup-tokens", type=_positive_int, default=8)
    parser.add_argument("--benchmark-tokens", type=_positive_int, default=64)
    parser.add_argument(
        "--optimization-sweep",
        action="store_true",
        help="measure cumulative Qwen4 decode optimizations with one weight load",
    )
    parser.add_argument(
        "--mtp-ab",
        action="store_true",
        help="enable MTP and compare ordinary greedy against speculative decode with one load",
    )
    parser.add_argument(
        "--mtp-sweep",
        action="store_true",
        help="compare ordinary, one-draft, two-draft, and confidence-gated MTP with one load",
    )
    parser.add_argument(
        "--mtp-real-prompts",
        action="store_true",
        help="run the MTP sweep on code, prose, tool-use, and long-context prompts",
    )
    parser.add_argument(
        "--mtp-expanded-prompts",
        action="store_true",
        help="add two more prompts to every real-prompt MTP category",
    )
    parser.add_argument(
        "--mtp-long-prompt-tokens",
        type=_positive_int,
        default=2048,
        help="token length of the MTP long-context prompt",
    )
    parser.add_argument(
        "--ple-mmap-sweep",
        action="store_true",
        help="measure cold and warm PLE mmap policies with one weight load",
    )
    parser.add_argument("--trace-tokens", type=_positive_int, default=8)
    parser.add_argument(
        "--profiler",
        choices=("torch", "nvtx", "none"),
        default="torch",
        help="torch exports Chrome traces; nvtx brackets capture for an outer nsys command",
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help="skip the eager detail trace (graph timing still runs)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="output directory (default profiles/decode-<timestamp>)",
    )
    return parser


def _ensure_mtp_sweep_warmup(args: argparse.Namespace) -> None:
    if args.mtp_sweep and args.warmup_tokens < args.benchmark_tokens:
        print(
            "MTP sweep: raising --warmup-tokens from "
            f"{args.warmup_tokens} to {args.benchmark_tokens} so every variant "
            "warms the full measured trajectory.",
            flush=True,
        )
        args.warmup_tokens = args.benchmark_tokens


def main(argv: list[str] | None = None, prog: str = "ft bench decode-profile") -> int:
    parser = _parser(prog)
    args = parser.parse_args(argv)
    if args.moe_cache_size is None:
        args.moe_cache_size = 0
        args.moe_cache_auto = True
    if args.prompt_token_id < 0:
        parser.error("--prompt-token-id must be non-negative")
    if args.benchmark_tokens < 2:
        parser.error(
            "--benchmark-tokens must be at least 2 (prefill plus one decode step)"
        )
    if args.profiler != "none" and args.trace_tokens < 2:
        parser.error("--trace-tokens must be at least 2 when a profiler is enabled")
    selected_sweeps = sum((args.mtp_ab, args.mtp_sweep, args.optimization_sweep))
    if selected_sweeps > 1:
        parser.error(
            "--mtp-ab, --mtp-sweep, and --optimization-sweep cannot be combined"
        )
    if args.mtp_real_prompts and not args.mtp_sweep:
        parser.error("--mtp-real-prompts requires --mtp-sweep")
    if args.mtp_expanded_prompts and not args.mtp_real_prompts:
        parser.error("--mtp-expanded-prompts requires --mtp-real-prompts")
    _ensure_mtp_sweep_warmup(args)

    # torch_utils and every instrumented model module read this at import time.  Keep
    # heavyweight FreeToken/torch imports below so ``ft bench`` remains lazy.
    os.environ["FREETOKEN_DECODE_PROFILE_RANGES"] = "1"
    if args.mtp_ab or args.mtp_sweep:
        os.environ["FREETOKEN_QWEN4_MTP"] = "1"
        os.environ["FREETOKEN_QWEN4_MTP_TIMING"] = "1"
    if args.mtp_sweep:
        os.environ["FREETOKEN_QWEN4_MTP_MAX_DRAFTS"] = "2"
    if args.optimization_sweep:
        _set_sweep_startup_environment()
    from freetoken.utils import configure_decode_profile_ranges

    configure_decode_profile_ranges(True)
    if args.ple_cache_gib is not None:
        os.environ["FREETOKEN_QWEN4_PLE_CACHE_GIB"] = str(args.ple_cache_gib)
    if args.ple_mmap_advice is not None:
        os.environ["FREETOKEN_QWEN4_PLE_MMAP_ADVICE"] = args.ple_mmap_advice
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "decode-profile needs one CUDA device; set CUDA_VISIBLE_DEVICES to the target GPU"
            )

        from freetoken.llm import LLM

        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            Path(args.output_dir or f"profiles/decode-{stamp}").expanduser().resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading {args.model} once; this may take 5-13 minutes ...", flush=True)
        load_started = time.perf_counter()
        llm = LLM(
            model_path=args.model,
            dtype=_dtype(args.dtype),
            max_running_req=1,
            max_extend_tokens=args.max_extend_tokens,
            max_seq_len_override=args.max_seq_len_override,
            moe_backend=args.moe_backend,
            moe_cache_size=args.moe_cache_size,
            moe_cache_auto=args.moe_cache_auto,
            moe_cpu_layers=args.moe_cpu_layers,
            moe_cpu_threads=args.moe_cpu_threads,
            moe_collect_stats=args.moe_collect_stats,
            num_token_override=args.num_tokens,
            use_dummy_weight=args.dummy_weight,
            cuda_graph_bs=[1],
            cuda_graph_max_bs=1,
            decode_log_interval=1_000_000_000,
        )
        model_load_seconds = time.perf_counter() - load_started
        try:
            vocab_size = llm.config.model_config.vocab_size
            if args.prompt_token_id >= vocab_size:
                parser.error(
                    f"--prompt-token-id {args.prompt_token_id} is outside vocab size {vocab_size}"
                )
            prompts = {
                "warmup": _phase_prompt(
                    args.prompt_token_id, args.prompt_tokens, 0, vocab_size
                ),
                "wall": _phase_prompt(
                    args.prompt_token_id, args.prompt_tokens, 1, vocab_size
                ),
                # Graph and eager traces use the same route/content. The graph trace sees
                # a fresh PLE n-gram stream; eager then shows the resident-page CPU cost.
                "trace": _phase_prompt(
                    args.prompt_token_id, args.prompt_tokens, 2, vocab_size
                ),
            }
            print("Warmup ...", flush=True)
            warmup = _measure_generation(llm, prompts["warmup"], args.warmup_tokens)

            optimization_sweep = None
            if args.optimization_sweep:
                optimization_sweep = _run_optimization_sweep(
                    llm,
                    prompts["wall"],
                    prime_tokens=args.benchmark_tokens,
                    benchmark_tokens=args.benchmark_tokens,
                )

            mtp_ab = None
            if args.mtp_ab:
                mtp_ab = _run_mtp_ab(
                    llm,
                    prompts["wall"],
                    warmup_tokens=args.warmup_tokens,
                    benchmark_tokens=args.benchmark_tokens,
                )

            mtp_sweep = None
            if args.mtp_sweep:
                mtp_prompts = {"synthetic": prompts["wall"]}
                if args.mtp_real_prompts:
                    mtp_prompts.update(
                        _mtp_prompt_suite(
                            llm,
                            args.mtp_long_prompt_tokens,
                            expanded=args.mtp_expanded_prompts,
                        )
                    )
                mtp_sweep = {
                    "prompts": {
                        name: _run_mtp_sweep(
                            llm,
                            prompt,
                            warmup_tokens=args.warmup_tokens,
                            benchmark_tokens=args.benchmark_tokens,
                        )
                        for name, prompt in mtp_prompts.items()
                    }
                }

            ple_mmap_sweep = None
            if args.ple_mmap_sweep:
                reference_tokens = (
                    optimization_sweep["baseline"]["prime"]["output_token_ids"]
                    if optimization_sweep is not None
                    else None
                )
                ple_mmap_sweep = _run_ple_mmap_sweep(
                    llm,
                    prompts["wall"],
                    benchmark_tokens=args.benchmark_tokens,
                    reference_tokens=reference_tokens,
                )

            print("Unprofiled CUDA-graph timing ...", flush=True)
            with _graph_execution(llm, True):
                wall_graph = _measure_generation(
                    llm, prompts["wall"], args.benchmark_tokens
                )

            traces: dict[str, Any] = {}
            if args.profiler == "torch":
                print("torch.profiler CUDA-graph trace ...", flush=True)
                traces["graph"] = _run_torch_trace(
                    llm,
                    prompts["trace"],
                    args.trace_tokens,
                    graph=True,
                    trace_path=output_dir / "graph-trace.json",
                )
                if not args.skip_eager:
                    print("torch.profiler eager detail trace ...", flush=True)
                    traces["eager"] = _run_torch_trace(
                        llm,
                        prompts["trace"],
                        args.trace_tokens,
                        graph=False,
                        trace_path=output_dir / "eager-trace.json",
                    )
            elif args.profiler == "nvtx":
                print("Starting cudaProfilerApi-delimited NVTX capture ...", flush=True)
                traces = _run_nvtx_capture(
                    llm, prompts["trace"], args.trace_tokens, eager=not args.skip_eager
                )

            report = {
                "version": 10,
                "timestamp": datetime.now(timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
                "host": socket.gethostname(),
                "model": args.model,
                "model_load_seconds": model_load_seconds,
                "arguments": vars(args),
                "phase_prompt_token_ids": prompts,
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_DEVICE_ORDER",
                        "CUDA_VISIBLE_DEVICES",
                        "TVM_FFI_CUDA_ARCH_LIST",
                        "TORCH_CUDA_ARCH_LIST",
                        "FREETOKEN_ALLOW_CUDA_MISMATCH",
                        "FREETOKEN_PIN_BUDGET_GB",
                        "FREETOKEN_QWEN4_PLE_CACHE_GIB",
                        "FREETOKEN_QWEN4_PLE_MMAP_ADVICE",
                        "FREETOKEN_QWEN_FUSED_ROUTER_MIN_TOKENS",
                        "FREETOKEN_QWEN4_QSA_DIRECT_ATTEND",
                        "FREETOKEN_QWEN_HC_PROJECTION_FUSED",
                        "FREETOKEN_QWEN_SHARED_GATE_FUSED",
                        "FREETOKEN_QWEN4_MTP",
                        "FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE",
                        "FREETOKEN_QWEN4_MTP_MAX_DRAFTS",
                        "FREETOKEN_QWEN4_MTP_DRAFT_P_MIN",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW",
                        "FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL",
                        "FREETOKEN_QWEN4_MTP_TIMING",
                    )
                },
                "gpu": {
                    "name": torch.cuda.get_device_name(llm.device),
                    "capability": list(torch.cuda.get_device_capability(llm.device)),
                },
                "model_geometry": _model_geometry(llm),
                "warmup": warmup,
                "optimization_sweep": optimization_sweep,
                "mtp_ab": mtp_ab,
                "mtp_sweep": mtp_sweep,
                "ple_mmap_sweep": ple_mmap_sweep,
                "wall_graph": wall_graph,
                "traces": traces,
            }
            report_path = output_dir / "report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")

            print("\nDecode profile summary")
            _print_timing("warmup", warmup)
            if optimization_sweep is not None:
                for label, result in optimization_sweep.items():
                    _print_timing(f"sweep-{label}", result["measured"])
                    print(
                        f"  {'trajectory':<12} forced={result['forced_tokens_match_reference']}  "
                        f"natural-match={result['matching_baseline_prediction_rate']:.3%}  "
                        f"first-diff={result['first_baseline_prediction_mismatch']}"
                    )
            if mtp_ab is not None:
                _print_timing("mtp-baseline", mtp_ab["baseline"])
                _print_timing("mtp", mtp_ab["mtp"])
                print(
                    f"  {'MTP result':<12} outputs-match={mtp_ab['outputs_match']}  "
                    f"first-diff={mtp_ab['first_output_mismatch']}  "
                    f"speedup={mtp_ab['speedup']:.3f}x"
                )
                mtp_stats = mtp_ab["mtp"].get("mtp")
                if mtp_stats is not None:
                    rate = mtp_stats["acceptance_rate"]
                    rate_text = f"{rate:.3%}" if rate is not None else "n/a"
                    print(
                        f"  {'MTP accept':<12} rate={rate_text}  "
                        f"tokens/cycle={mtp_stats['tokens_per_cycle']:.3f}"
                    )
            if mtp_sweep is not None:
                for prompt_name, prompt_result in mtp_sweep["prompts"].items():
                    _print_timing(
                        f"mtp-{prompt_name}-baseline", prompt_result["baseline"]
                    )
                    for name, result in prompt_result["variants"].items():
                        _print_timing(f"mtp-{prompt_name}-{name}", result["measured"])
                        stats = result["measured"].get("mtp")
                        rate = stats["acceptance_rate"] if stats is not None else None
                        rate_text = f"{rate:.3%}" if rate is not None else "n/a"
                        speedup = result["speedup"]
                        speedup_text = (
                            f"{speedup:.3f}x" if speedup is not None else "n/a"
                        )
                        tokens_per_cycle = (
                            stats["tokens_per_cycle"] if stats is not None else None
                        )
                        tokens_per_cycle_text = (
                            f"{tokens_per_cycle:.3f}"
                            if tokens_per_cycle is not None
                            else "n/a"
                        )
                        print(
                            f"  {prompt_name + '/' + name:<32} "
                            f"speedup={speedup_text}  "
                            f"accept={rate_text}  "
                            f"tokens/cycle={tokens_per_cycle_text}"
                        )
            if ple_mmap_sweep is not None:
                for advice, result in ple_mmap_sweep["results"].items():
                    _print_timing(f"ple-{advice}-cold", result["cold"])
                    _print_timing(f"ple-{advice}-warm", result["warm"])
            _print_timing("graph-wall", wall_graph)
            for label, result in traces.items():
                _print_timing(label, result)
                if result.get("ranges"):
                    _print_ranges(label, result["ranges"])
            print(f"\n  report: {report_path}")
            if args.profiler == "torch":
                print(
                    "  open graph-trace.json and eager-trace.json in chrome://tracing or Perfetto"
                )
            elif args.profiler == "nvtx":
                print(
                    "  nsys output is written by the outer `nsys profile -o ...` command"
                )
            return 0
        finally:
            llm.shutdown()
    except (RuntimeError, OSError, ValueError) as error:
        if os.getenv("FREETOKEN_BENCH_TRACEBACK", "0") == "1":
            traceback.print_exc()
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
