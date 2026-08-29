from __future__ import annotations

import contextlib
import math
import os
import statistics
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.core import Batch, Req
from freetoken.pld_index import PLDIndex
from freetoken.utils import decode_profile_range, init_logger

from .graph import project_lm_head_all_positions

if TYPE_CHECKING:
    from .engine import Engine
    from .sample import BatchSamplingArgs


logger = init_logger(__name__)


@dataclass
class MTPMetrics:
    cycles: int = 0
    proposed_drafts: int = 0
    accepted_drafts: int = 0
    emitted_tokens: int = 0
    pld_cycles: int = 0
    pld_proposed_drafts: int = 0
    pld_accepted_drafts: int = 0
    cycle_trace: list[dict[str, int]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float | None:
        if not self.proposed_drafts:
            return None
        return self.accepted_drafts / self.proposed_drafts


def _mtp_max_drafts() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1 or 2") from error
    if value not in (1, 2):
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1 or 2")
    return value


def _mtp_draft_p_min() -> float:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "0").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_DRAFT_P_MIN must be between 0 and 1"
        ) from error
    if not 0.0 <= value <= 1.0:
        raise ValueError("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN must be between 0 and 1")
    return value


def _mtp_pld_mode() -> str | None:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_PLD", "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return None
    if raw in {"1", "true", "yes", "on", "hybrid"}:
        return "hybrid"
    if raw == "only":
        return "only"
    raise ValueError("FREETOKEN_QWEN4_MTP_PLD must be a boolean or 'only'")


def _mtp_pld_ngram_range() -> tuple[int, int]:
    raw_min = os.getenv("FREETOKEN_QWEN4_MTP_PLD_MIN_NGRAM", "6").strip()
    raw_max = os.getenv("FREETOKEN_QWEN4_MTP_PLD_MAX_NGRAM", "12").strip()
    try:
        minimum = int(raw_min)
        maximum = int(raw_max)
    except ValueError as error:
        raise ValueError("MTP PLD n-gram bounds must be integers") from error
    if not 1 <= minimum <= maximum:
        raise ValueError("MTP PLD needs 1 <= MIN_NGRAM <= MAX_NGRAM")
    return minimum, maximum


def _mtp_adaptive_enabled() -> bool:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE must be a boolean")


def _mtp_adaptive_cycles() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES", "64").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES must be positive"
        ) from error
    if value < 1:
        raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES must be positive")
    return value


def _mtp_adaptive_min_acceptance() -> float:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE", "0.75").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE must be between 0 and 1"
        ) from error
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE must be between 0 and 1"
        )
    return value


def _mtp_adaptive_samples() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES", "4").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES must be positive"
        ) from error
    if value < 1:
        raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE_SAMPLES must be positive")
    return value


def _mtp_adaptive_window() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW", "64").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW must be positive"
        ) from error
    if value < 1:
        raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE_WINDOW must be positive")
    return value


def _mtp_adaptive_probe_interval() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL", "16").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL must be positive"
        ) from error
    if value < 1:
        raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE_PROBE_INTERVAL must be positive")
    return value


def choose_adaptive_draft_width(
    survival: Sequence[float], costs_ms: Sequence[float]
) -> int:
    """Choose 0, 1, or 2 drafts by measured expected tokens per millisecond."""
    if len(costs_ms) != len(survival) + 1:
        raise ValueError("adaptive MTP needs one cost for every draft width")
    previous = 1.0
    for probability in survival:
        if (
            not math.isfinite(probability)
            or probability < 0.0
            or probability > previous
        ):
            raise ValueError("adaptive MTP survival must be finite and non-increasing")
        previous = probability
    for cost in costs_ms:
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("adaptive MTP costs must be finite and positive")

    expected_tokens = 1.0
    best_width = 0
    best_throughput = expected_tokens / costs_ms[0]
    for width, probability in enumerate(survival, start=1):
        expected_tokens += probability
        throughput = expected_tokens / costs_ms[width]
        if throughput > best_throughput:
            best_width = width
            best_throughput = throughput
    return best_width


@dataclass(frozen=True)
class _AdaptiveOutcome:
    width: int
    accepted: int


class MTPAdaptiveController:
    """Request-local measured-cost controller for zero to two drafts."""

    def __init__(
        self,
        *,
        samples: int,
        window: int,
        probe_interval: int,
        disable_after: int,
        min_acceptance: float = 0.75,
    ) -> None:
        if samples < 1 or window < 1 or probe_interval < 1 or disable_after < 1:
            raise ValueError("adaptive MTP controller limits must be positive")
        if window < samples:
            raise ValueError("adaptive MTP window must cover calibration samples")
        if not 0.0 <= min_acceptance <= 1.0:
            raise ValueError("adaptive MTP minimum acceptance must be between 0 and 1")
        self.samples = samples
        self.window = window
        self.probe_interval = probe_interval
        self.disable_after = disable_after
        self.min_acceptance = min_acceptance
        self.reset()

    def reset(self) -> None:
        self.cycles = 0
        self.selected_width: int | None = None
        self.selection_cycle: int | None = None
        self._costs = [deque(maxlen=self.window) for _ in range(3)]
        self._outcomes: deque[_AdaptiveOutcome] = deque(maxlen=self.window)
        self._last_sample = [-1, -1, -1]
        self._trace: deque[dict[str, float | int]] = deque(maxlen=4096)

    def record(
        self,
        *,
        width: int,
        accepted: int,
        cycle_ms: float,
        target_ms: float,
        wall_ms: float | None = None,
    ) -> None:
        if width not in (0, 1, 2) or accepted < 0 or accepted > width:
            raise ValueError("invalid adaptive MTP outcome")
        if wall_ms is None:
            wall_ms = cycle_ms
        if not math.isfinite(wall_ms) or wall_ms <= 0.0:
            return
        host_ms = max(wall_ms - cycle_ms, 0.0)
        cost = target_ms + host_ms if width == 0 else cycle_ms + host_ms
        if not math.isfinite(cost) or cost <= 0.0:
            return
        self._trace.append(
            {
                "cycle": self.cycles + 1,
                "width": width,
                "accepted": accepted,
                "cycle_ms": float(cycle_ms),
                "target_ms": float(target_ms),
                "wall_ms": float(wall_ms),
                "host_ms": float(host_ms),
                "cost_ms": float(cost),
            }
        )
        self._costs[width].append(float(cost))
        if width:
            self._outcomes.append(_AdaptiveOutcome(width, accepted))
        self._last_sample[width] = self.cycles
        self.cycles += 1

    def _calibration_width(self, max_width: int) -> int | None:
        for offset in range(max_width):
            width = 1 + (self.cycles + offset) % max_width
            if len(self._costs[width]) < self.samples:
                return width
        return None

    def _survival(self, max_width: int) -> list[float]:
        first = [outcome.accepted >= 1 for outcome in self._outcomes]
        if not first:
            raise RuntimeError("adaptive MTP has no first-draft observations")
        result = [sum(first) / len(first)]
        if max_width >= 2:
            second = [
                outcome.accepted >= 2
                for outcome in self._outcomes
                if outcome.width >= 2
            ]
            if not second:
                raise RuntimeError("adaptive MTP has no second-draft observations")
            result.append(min(result[0], sum(second) / len(second)))
        return result

    def choose(self, max_width: int) -> int:
        if max_width not in (1, 2):
            raise ValueError("adaptive MTP supports one or two drafts")
        if self.cycles < self.disable_after:
            return max_width
        calibration = self._calibration_width(max_width)
        if calibration is not None:
            return calibration

        survival = self._survival(max_width)
        if sum(survival) / len(survival) < self.min_acceptance:
            selected = 0
        else:
            expected_tokens = 1.0 + survival[0]
            selected = 1
            best_throughput = expected_tokens / statistics.median(self._costs[1])
            for width in range(2, max_width + 1):
                expected_tokens += survival[width - 1]
                throughput = expected_tokens / statistics.median(self._costs[width])
                if throughput > best_throughput:
                    selected = width
                    best_throughput = throughput
        if selected != self.selected_width:
            self.selected_width = selected
            self.selection_cycle = self.cycles
        if (
            selected
            and self.cycles - min(self._last_sample[1 : max_width + 1])
            >= self.probe_interval
        ):
            return min(range(1, max_width + 1), key=self._last_sample.__getitem__)
        return selected

    def summary(self, max_width: int) -> dict[str, Any]:
        first = [outcome.accepted >= 1 for outcome in self._outcomes]
        survival: list[float | None] = [sum(first) / len(first) if first else None]
        if max_width >= 2:
            second = [
                outcome.accepted >= 2
                for outcome in self._outcomes
                if outcome.width >= 2
            ]
            survival.append(sum(second) / len(second) if second else None)
        return {
            "cycles": self.cycles,
            "selected_width": self.selected_width,
            "selection_cycle": self.selection_cycle,
            "survival": survival,
            "cost_ms": [
                statistics.median(costs) if costs else None
                for costs in self._costs[: max_width + 1]
            ],
            "samples": [len(costs) for costs in self._costs[: max_width + 1]],
            "trace": list(self._trace),
        }


@dataclass
class _TimingEvent:
    phase: str
    started: torch.cuda.Event
    ended: torch.cuda.Event


@dataclass
class _AdaptiveTimingSample:
    width: int
    accepted: int
    wall_started: float
    cycle_started: torch.cuda.Event
    target_started: torch.cuda.Event
    target_ended: torch.cuda.Event
    cycle_ended: torch.cuda.Event
    wall_ms: float | None = None


@dataclass
class _MutableForwardState:
    req_input_ids: torch.Tensor
    req_cached_len: int
    req_device_len: int
    batch_phase: str
    batch_input_ids: torch.Tensor
    batch_host_input_ids: Any
    batch_positions: torch.Tensor
    batch_out_loc: torch.Tensor | None
    batch_padded_reqs: list[Req]
    batch_fla_metadata: Any
    batch_attn_metadata: Any
    batch_moe_decode_cache: bool


@dataclass
class _TargetRuntimeState:
    linear_slot: int | None
    linear_state: tuple[torch.Tensor, torch.Tensor] | None
    model_state: Any


class _TargetVerifyGraph:
    """Captured fixed-row target extension for one Qwen4-Exp request."""

    def __init__(
        self,
        engine: Engine,
        token_count: int,
        checkpoint_capacity: int,
        graph_pool=None,
    ):
        if not isinstance(engine.attn_backend, QSASparseAttnBackend):
            raise NotImplementedError(
                "Qwen4-Exp MTP target graph requires the QSA sparse backend"
            )
        if engine.graph_runner.max_graph_bs < 1:
            raise NotImplementedError("Qwen4-Exp MTP target graph requires CUDA graphs")
        if token_count not in (2, 3):
            raise ValueError("MTP target graph supports two or three tokens")
        if checkpoint_capacity < token_count - 1:
            raise ValueError("MTP target checkpoint capacity is too small")

        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.token_count = token_count
        self.graph_pool = graph_pool
        self.input_ids = torch.zeros(
            self.token_count, dtype=torch.int32, device=self.device
        )
        self.positions = torch.zeros_like(self.input_ids)
        self.out_loc = torch.full_like(self.input_ids, engine.num_pages)
        self.linear_table_idx = torch.full(
            (1,),
            engine.linear_state_pool.padding_slot,
            dtype=torch.int32,
            device=self.device,
        )
        self.rows = torch.full(
            (self.token_count, engine.page_table.shape[1]),
            engine.num_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self.kvlen = torch.arange(
            1, self.token_count + 1, dtype=torch.int32, device=self.device
        )
        vocab_size = engine.config.model_config.vocab_size
        hidden_width = self.model.mtp.hc_count * self.model.mtp.hidden_size
        hidden_dtype = self.model.model.embed_tokens.weight.dtype
        self.logits = torch.empty(
            self.token_count, vocab_size, dtype=torch.float32, device=self.device
        )
        self.expanded = torch.empty(
            self.token_count,
            hidden_width,
            dtype=hidden_dtype,
            device=self.device,
        )

        dummy_req = Req(
            input_ids=torch.zeros(self.token_count, dtype=torch.int32),
            table_idx=engine.dummy_req.table_idx,
            cached_len=0,
            output_len=1,
            uid=-2,
            sampling_params=engine.dummy_req.sampling_params,
            cache_handle=engine.dummy_req.cache_handle,
        )
        dummy_req.linear_slot_idx = engine.linear_state_pool.padding_slot
        self.batch = Batch(reqs=[dummy_req], phase="prefill")
        self.batch.padded_reqs = self.batch.reqs
        self.batch.input_ids = self.input_ids
        self.batch.positions = self.positions
        self.batch.out_loc = self.out_loc
        self.batch.linear_table_idx = self.linear_table_idx
        self.batch.moe_decode_cache = True
        self.batch.mtp_verify = True
        self.batch.mtp_checkpoint_capacity = checkpoint_capacity
        self.batch.cuda_graph_capture = True
        self.batch.fla_metadata = self._fla_metadata()
        self.batch.attn_metadata = self._attn_metadata()
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _fla_metadata(self):
        from freetoken.attention.linear import FLAMetadata

        return FLAMetadata(
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            cache_indices=self.linear_table_idx,
            has_initial_state=torch.ones(1, dtype=torch.bool, device=self.device),
        )

    def _attn_metadata(self) -> QSASparseMetadata:
        return QSASparseMetadata(
            is_decode=True,
            last_indices=torch.tensor(
                [self.token_count - 1], dtype=torch.int32, device=self.device
            ),
            qo_indptr_cpu=torch.tensor(
                [0, self.token_count], dtype=torch.int32, pin_memory=True
            ),
            kv_len_cpu=torch.tensor(
                [self.token_count], dtype=torch.int32, pin_memory=True
            ),
            inner=None,
            rows=self.rows,
            kvlen=self.kvlen,
        )

    def _forward(self) -> None:
        hidden, expanded = self.model.model.forward(
            self.batch.input_ids, return_expanded=True
        )
        self.logits.copy_(project_lm_head_all_positions(self.model.lm_head, hidden))
        self.expanded.copy_(expanded)

    def _capture(self) -> None:
        cache = self.engine.moe_offload_cache
        if cache is not None:
            cache.reset()
        self.model.prepare_cuda_graph_capture(self.batch)
        with self.engine.ctx.forward_batch(self.batch):
            self._forward()
            with torch.cuda.graph(
                self.graph, pool=self.graph_pool, stream=self.engine.stream
            ):
                self._forward()
        if cache is not None:
            cache.reset()

    def replay(
        self, req: Req, token_ids: torch.Tensor, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.numel() != self.token_count:
            raise ValueError(
                f"MTP target graph requires exactly {self.token_count} tokens"
            )
        end = start + self.token_count
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        self.input_ids.copy_(token_ids)
        torch.arange(
            start, end, dtype=torch.int32, device=self.device, out=self.positions
        )
        self.out_loc.copy_(self.engine.page_table[req.table_idx, start:end])
        self.linear_table_idx.fill_(slot)
        rows = self.engine.page_table[req.table_idx, :end]
        self.rows[:, :end].copy_(rows.unsqueeze(0).expand(self.token_count, -1))
        torch.arange(
            start + 1,
            end + 1,
            dtype=torch.int32,
            device=self.device,
            out=self.kvlen,
        )
        self.batch.reqs = [req]
        self.batch.padded_reqs = self.batch.reqs
        with self.engine.ctx.forward_batch(self.batch):
            self.model.prepare_cuda_graph_replay(self.batch)
            self.graph.replay()
        return self.logits, self.expanded

    def commit_prefix(self, req: Req, checkpoint_index: int) -> None:
        self.model.commit_mtp_prefix_state(
            req, self.engine.linear_state_pool, checkpoint_index
        )


class _PredictorGraph:
    """Captured one- to three-token MTP predictor extension."""

    def __init__(self, engine: Engine, token_count: int, graph_pool=None):
        if not isinstance(engine.attn_backend, QSASparseAttnBackend):
            raise NotImplementedError(
                "Qwen4-Exp MTP predictor graph requires the QSA sparse backend"
            )
        if token_count not in (1, 2, 3):
            raise ValueError("MTP predictor graph supports one to three tokens")

        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.token_count = token_count
        self.graph_pool = graph_pool
        hidden_width = self.model.mtp.hc_count * self.model.mtp.hidden_size
        hidden_dtype = self.model.model.embed_tokens.weight.dtype
        self.hidden = torch.zeros(
            token_count, hidden_width, dtype=hidden_dtype, device=self.device
        )
        self.input_ids = torch.zeros(token_count, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros_like(self.input_ids)
        self.out_loc = torch.full_like(self.input_ids, engine.num_pages)
        self.rows = torch.full(
            (token_count, engine.page_table.shape[1]),
            engine.num_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self.kvlen = torch.arange(
            1, token_count + 1, dtype=torch.int32, device=self.device
        )
        self.logits = torch.empty(
            token_count,
            engine.config.model_config.vocab_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.expanded = torch.empty(
            token_count,
            hidden_width,
            dtype=hidden_dtype,
            device=self.device,
        )

        dummy_req = Req(
            input_ids=torch.zeros(token_count, dtype=torch.int32),
            table_idx=engine.dummy_req.table_idx,
            cached_len=0,
            output_len=1,
            uid=-3 - token_count,
            sampling_params=engine.dummy_req.sampling_params,
            cache_handle=engine.dummy_req.cache_handle,
        )
        self.batch = Batch(reqs=[dummy_req], phase="prefill")
        self.batch.padded_reqs = self.batch.reqs
        self.batch.input_ids = self.input_ids
        self.batch.positions = self.positions
        self.batch.out_loc = self.out_loc
        self.batch.moe_decode_cache = True
        self.batch.cuda_graph_capture = True
        self.batch.attn_metadata = self._attn_metadata()
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _attn_metadata(self) -> QSASparseMetadata:
        qo_indptr = torch.tensor(
            [0, self.token_count], dtype=torch.int32, pin_memory=True
        )
        kv_len_cpu = torch.tensor(
            [self.token_count], dtype=torch.int32, pin_memory=True
        )
        return QSASparseMetadata(
            is_decode=True,
            last_indices=torch.tensor(
                [self.token_count - 1], dtype=torch.int32, device=self.device
            ),
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len_cpu,
            inner=None,
            rows=self.rows,
            kvlen=self.kvlen,
        )

    def _forward(self) -> None:
        logits, expanded = self.model.mtp_forward(self.hidden, self.input_ids)
        self.logits.copy_(logits)
        self.expanded.copy_(expanded)

    def _capture(self) -> None:
        cache = self.model.mtp_offload_cache
        cache.reset()
        with self.engine.ctx.forward_batch(self.batch):
            self._forward()
            with torch.cuda.graph(
                self.graph, pool=self.graph_pool, stream=self.engine.stream
            ):
                self._forward()
        cache.reset()

    def replay(
        self,
        req: Req,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[0] != self.token_count or token_ids.numel() != self.token_count:
            raise ValueError(
                f"MTP predictor graph requires exactly {self.token_count} tokens"
            )
        end = start + self.token_count
        self.batch.reqs = [req]
        self.batch.padded_reqs = self.batch.reqs
        self.hidden.copy_(hidden)
        self.input_ids.copy_(token_ids)
        torch.arange(
            start, end, dtype=torch.int32, device=self.device, out=self.positions
        )
        self.out_loc.copy_(self.engine.page_table[req.table_idx, start:end])
        rows = self.engine.page_table[req.table_idx, :end]
        self.rows[:, :end].copy_(rows.unsqueeze(0).expand(self.token_count, -1))
        torch.arange(
            start + 1,
            end + 1,
            dtype=torch.int32,
            device=self.device,
            out=self.kvlen,
        )
        with self.engine.ctx.forward_batch(self.batch):
            self.graph.replay()
        return self.logits, self.expanded


def _snapshot_mutable_forward_state(batch: Batch, req: Req) -> _MutableForwardState:
    return _MutableForwardState(
        req_input_ids=req.input_ids,
        req_cached_len=req.cached_len,
        req_device_len=req.device_len,
        batch_phase=batch.phase,
        batch_input_ids=batch.input_ids,
        batch_host_input_ids=batch.host_input_ids,
        batch_positions=batch.positions,
        batch_out_loc=batch.out_loc,
        batch_padded_reqs=batch.padded_reqs,
        batch_fla_metadata=batch.fla_metadata,
        batch_attn_metadata=getattr(batch, "attn_metadata", None),
        batch_moe_decode_cache=batch.moe_decode_cache,
    )


def _restore_mutable_forward_state(
    batch: Batch, req: Req, state: _MutableForwardState
) -> None:
    req.input_ids = state.req_input_ids
    req.cached_len = state.req_cached_len
    req.device_len = state.req_device_len
    batch.phase = state.batch_phase
    batch.input_ids = state.batch_input_ids
    batch.host_input_ids = state.batch_host_input_ids
    batch.positions = state.batch_positions
    batch.out_loc = state.batch_out_loc
    batch.padded_reqs = state.batch_padded_reqs
    batch.fla_metadata = state.batch_fla_metadata
    batch.attn_metadata = state.batch_attn_metadata
    batch.moe_decode_cache = state.batch_moe_decode_cache


def _snapshot_target_runtime(engine: Engine, req: Req) -> _TargetRuntimeState:
    pool = engine.linear_state_pool
    linear_slot = None
    linear_state = None
    if pool is not None:
        linear_slot = (
            req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        )
        linear_state = (
            pool.conv_states[:, linear_slot].clone(),
            pool.recurrent_states[:, linear_slot].clone(),
        )
    snapshot_model = getattr(engine.model, "snapshot_runtime_state", None)
    model_state = snapshot_model(req) if snapshot_model is not None else None
    return _TargetRuntimeState(linear_slot, linear_state, model_state)


def _restore_target_runtime(
    engine: Engine, req: Req, state: _TargetRuntimeState
) -> None:
    if state.linear_state is not None:
        assert state.linear_slot is not None
        conv_state, recurrent_state = state.linear_state
        engine.linear_state_pool.conv_states[:, state.linear_slot].copy_(conv_state)
        engine.linear_state_pool.recurrent_states[:, state.linear_slot].copy_(
            recurrent_state
        )
    restore_model = getattr(engine.model, "restore_runtime_state", None)
    if restore_model is not None and state.model_state is not None:
        restore_model(req, state.model_state)


class MTPWorker:
    """Single-request Qwen4-Exp MTP predictor and greedy verifier."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.model = engine.model
        cache = getattr(self.model, "mtp_offload_cache", None)
        if cache is None:
            raise RuntimeError("Qwen4-Exp MTP expert cache is not initialized")
        top_k = engine.config.model_config.num_experts_per_tok
        self.max_predictor_chunk = cache.cache_size // top_k
        if self.max_predictor_chunk < 1:
            raise ValueError(
                f"MTP expert cache has {cache.cache_size} slots but top_k={top_k}"
            )
        self.uid: int | None = None
        self.predictor_cached_len = 0
        self.pending_hidden: torch.Tensor | None = None
        self.pending_draft: torch.Tensor | None = None
        self.pending_predictor_hidden: torch.Tensor | None = None
        self.pending_draft_confidence = 1.0
        self.max_supported_drafts = _mtp_max_drafts()
        self.max_drafts = self.max_supported_drafts
        self.draft_p_min = _mtp_draft_p_min()
        self.adaptive_enabled = _mtp_adaptive_enabled()
        self.adaptive_cycles = _mtp_adaptive_cycles()
        self.adaptive_min_acceptance = _mtp_adaptive_min_acceptance()
        self.adaptive_samples = _mtp_adaptive_samples()
        self.adaptive_window = _mtp_adaptive_window()
        self.adaptive_probe_interval = _mtp_adaptive_probe_interval()
        self.adaptive_controller = MTPAdaptiveController(
            samples=self.adaptive_samples,
            window=self.adaptive_window,
            probe_interval=self.adaptive_probe_interval,
            disable_after=self.adaptive_cycles,
            min_acceptance=self.adaptive_min_acceptance,
        )
        self.adaptive_disabled_uid: int | None = None
        self.adaptive_decision_cycle: int | None = None
        self._adaptive_pending: list[_AdaptiveTimingSample] = []
        self.pld_mode = _mtp_pld_mode()
        self.pld_min_ngram, self.pld_max_ngram = _mtp_pld_ngram_range()
        self.pld_index: PLDIndex | None = None
        if self.pld_mode is not None and self.adaptive_enabled:
            raise RuntimeError(
                "adaptive MTP and FREETOKEN_QWEN4_MTP_PLD cannot be combined"
            )
        # Small persistent staging so a lookup draft is a pinned H2D copy, not
        # a pageable blocking transfer. Allocated unconditionally: the sweep
        # can switch pld_mode after construction.
        self._pld_host_tokens = torch.empty(
            self.max_supported_drafts,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        self._pld_device_tokens = torch.empty(
            self.max_supported_drafts, dtype=torch.int32, device=engine.device
        )
        self.metrics = MTPMetrics()
        self.log_interval = max(
            0, int(os.getenv("FREETOKEN_QWEN4_MTP_LOG_INTERVAL", "40"))
        )
        self.timing_enabled = os.getenv(
            "FREETOKEN_QWEN4_MTP_TIMING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.timing_events: list[_TimingEvent] = []
        self.target_verify_graphs: dict[int, _TargetVerifyGraph] = {}
        try:
            graph_pool = None
            for token_count in range(2, self.max_supported_drafts + 2):
                graph = _TargetVerifyGraph(
                    engine,
                    token_count,
                    self.max_supported_drafts,
                    graph_pool=graph_pool,
                )
                self.target_verify_graphs[token_count] = graph
                graph_pool = graph.graph.pool()
        except NotImplementedError as error:
            logger.warning_rank0(f"Qwen4-Exp MTP target graph disabled: {error}")
        self.predictor_graphs: dict[int, _PredictorGraph] = {}
        if self.target_verify_graphs:
            for token_count in range(1, self.max_supported_drafts + 2):
                graph = _PredictorGraph(engine, token_count, graph_pool=graph_pool)
                self.predictor_graphs[token_count] = graph
                graph_pool = graph.graph.pool()

    def reset(self, uid: int | None = None) -> None:
        self.uid = uid
        self.predictor_cached_len = 0
        self.pending_hidden = None
        self.pending_draft = None
        self.pending_predictor_hidden = None
        self.pending_draft_confidence = 1.0
        self.adaptive_disabled_uid = None
        self.adaptive_decision_cycle = None
        self._adaptive_pending.clear()
        if self.adaptive_controller is not None:
            self.adaptive_controller.reset()
        self.pld_index = (
            PLDIndex(self.pld_min_ngram, self.pld_max_ngram)
            if self.pld_mode is not None
            else None
        )

    def reset_metrics(self) -> None:
        self.metrics = MTPMetrics()
        self.timing_events.clear()

    def timing(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = {}
        for event in self.timing_events:
            grouped.setdefault(event.phase, []).append(
                event.started.elapsed_time(event.ended)
            )
        return {
            phase: {
                "calls": len(milliseconds),
                "total_ms": sum(milliseconds),
                "mean_ms": sum(milliseconds) / len(milliseconds),
            }
            for phase, milliseconds in grouped.items()
        }

    def adaptive_summary(self) -> dict[str, Any] | None:
        if not self.adaptive_enabled:
            return None
        self._finish_adaptive_wall(time.perf_counter())
        self._collect_adaptive_samples()
        return self.adaptive_controller.summary(self.max_supported_drafts)

    def _finish_adaptive_wall(self, ended: float) -> None:
        for sample in reversed(self._adaptive_pending):
            if sample.wall_ms is None:
                sample.wall_ms = max((ended - sample.wall_started) * 1000.0, 1e-6)
                return

    def _collect_adaptive_samples(self) -> None:
        controller = self.adaptive_controller
        pending = []
        for sample in self._adaptive_pending:
            if sample.wall_ms is None or not sample.cycle_ended.query():
                pending.append(sample)
                continue
            controller.record(
                width=sample.width,
                accepted=sample.accepted,
                cycle_ms=sample.cycle_started.elapsed_time(sample.cycle_ended),
                target_ms=sample.target_started.elapsed_time(sample.target_ended),
                wall_ms=sample.wall_ms,
            )
        self._adaptive_pending = pending

    def _adaptive_width(self, max_width: int) -> int:
        if not self.adaptive_enabled:
            return max_width
        if self.draft_p_min > 0.0:
            raise RuntimeError(
                "adaptive MTP and FREETOKEN_QWEN4_MTP_DRAFT_P_MIN cannot be combined"
            )
        if self.pld_mode is not None:
            raise RuntimeError(
                "adaptive MTP and FREETOKEN_QWEN4_MTP_PLD cannot be combined"
            )
        controller = self.adaptive_controller
        previous = controller.selected_width
        width = controller.choose(max_width)
        if controller.selected_width != previous:
            costs = controller.summary(max_width)["cost_ms"]
            logger.info_rank0(
                "Qwen4-Exp MTP adaptive width: "
                f"cycle={controller.cycles}, width={controller.selected_width}, "
                f"cost_ms={costs}"
            )
        return width

    @contextlib.contextmanager
    def _profile_phase(self, phase: str):
        with decode_profile_range(f"FT.MTP.{phase}"):
            if not self.timing_enabled:
                yield
                return
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record(self.engine.stream)
            try:
                yield
            finally:
                ended.record(self.engine.stream)
                self.timing_events.append(_TimingEvent(phase, started, ended))

    def _activate(self, uid: int) -> None:
        if self.uid != uid:
            self.reset(uid)

    def can_speculate(self, batch: Batch) -> bool:
        if not batch.is_decode or batch.size != 1:
            return False
        req = batch.reqs[0]
        if not (
            req.sampling_params.is_greedy
            and req.remain_len >= 2
            and self.adaptive_disabled_uid != req.uid
            and self.uid == req.uid
            and self.pending_hidden is None
        ):
            return False
        if self.pld_mode == "only":
            # Lookup-only drafting never advances the predictor after prefill,
            # so predictor-state currency is not required.
            return self.pld_index is not None
        return (
            self.pending_draft is not None
            and self.predictor_cached_len == req.cached_len
        )

    def update_prefill(
        self,
        batch: Batch,
        expanded_hidden: torch.Tensor,
        base_token: torch.Tensor,
        *,
        start: int,
        end: int,
        final: bool,
    ) -> None:
        req = batch.reqs[0]
        if start == 0:
            self.reset(req.uid)
        else:
            self._activate(req.uid)
        if expanded_hidden.shape[0] != end - start:
            raise ValueError(
                f"MTP target hidden rows {expanded_hidden.shape[0]} do not match "
                f"prefill range [{start}, {end})"
            )

        pair_end = end - 1 if final else end - 2
        source_start = self.predictor_cached_len
        hidden_parts = []
        if self.pending_hidden is not None:
            if source_start != start - 1:
                raise RuntimeError(
                    f"MTP pending source starts at {source_start}, expected {start - 1}"
                )
            hidden_parts.append(self.pending_hidden)
        elif source_start != start:
            raise RuntimeError(
                f"MTP predictor cache ends at {source_start}, target chunk starts at {start}"
            )

        current_first = max(source_start, start)
        if pair_end >= current_first:
            hidden_parts.append(
                expanded_hidden[current_first - start : pair_end - start + 1]
            )

        if pair_end >= source_start:
            hidden = torch.cat(hidden_parts, dim=0)
            token_start = source_start + 1
            prompt_token_end = min(pair_end + 2, end)
            token_ids = req.input_ids[token_start:prompt_token_end].to(
                device=self.engine.device, dtype=torch.int32
            )
            if final:
                token_ids = torch.cat([token_ids, base_token[:1].to(dtype=torch.int32)])
            expected = pair_end - source_start + 1
            if hidden.shape[0] != expected or token_ids.numel() != expected:
                raise RuntimeError(
                    f"MTP prefill alignment produced hidden={hidden.shape[0]}, "
                    f"tokens={token_ids.numel()}, expected={expected}"
                )
            logits, predictor_expanded = self._run_predictor(
                batch, req, hidden, token_ids, source_start
            )
            if final:
                (
                    self.pending_draft,
                    self.pending_draft_confidence,
                ) = self._draft_candidate(logits)
                if self.max_supported_drafts > 1:
                    self.pending_predictor_hidden = predictor_expanded[-1:].clone()

        if final and self.pld_index is not None:
            # One host sync at the end of prefill: the lookup context is the
            # prompt plus the first generated token.
            prompt_tokens = req.input_ids[:end].tolist()
            prompt_tokens.append(int(base_token[0].item()))
            self.pld_index.extend(prompt_tokens)

        self.pending_hidden = None if final else expanded_hidden[-1:].clone()
        expected_cached = end if final else max(0, end - 1)
        if self.predictor_cached_len != expected_cached:
            raise RuntimeError(
                f"MTP predictor cache ends at {self.predictor_cached_len}, "
                f"expected {expected_cached}"
            )

    def forward_decode(
        self, batch: Batch, _args: BatchSamplingArgs
    ) -> tuple[torch.Tensor, bool]:
        req = batch.reqs[0]
        self._activate(req.uid)
        adaptive_wall_started = time.perf_counter()
        if self.adaptive_enabled:
            self._finish_adaptive_wall(adaptive_wall_started)
            self._collect_adaptive_samples()
        source_position = req.cached_len
        pld_only = self.pld_mode == "only"
        if self.pending_hidden is not None:
            raise RuntimeError("MTP prompt initialization is incomplete")
        if not pld_only:
            if self.pending_draft is None:
                raise RuntimeError("MTP draft is not initialized")
            if self.predictor_cached_len != source_position:
                raise RuntimeError(
                    f"MTP predictor cache ends at {self.predictor_cached_len}, "
                    f"target decode starts at {source_position}"
                )

        max_drafts = min(
            self.max_drafts,
            self.max_supported_drafts,
        )
        max_drafts = self._adaptive_width(max_drafts)
        disable_after_cycle = (
            self.adaptive_enabled
            and self.adaptive_controller.selected_width == 0
            and max_drafts == 0
        )
        max_drafts = min(max_drafts, max(req.remain_len - 1, 0))
        adaptive_cycle_started = None
        if self.adaptive_enabled:
            adaptive_cycle_started = torch.cuda.Event(enable_timing=True)
            adaptive_cycle_started.record(self.engine.stream)
        draft_parts = []
        predictor_speculated = False
        pld_draft_count = 0
        if max_drafts and self.pld_index is not None:
            with self._profile_phase("PLDLookup"):
                pld_tokens = self.pld_index.draft(max_drafts)
            if pld_tokens:
                pld_draft_count = len(pld_tokens)
                draft_parts.append(self._pld_draft_tensor(pld_tokens))
        if (
            not draft_parts
            and not pld_only
            and max_drafts
            and self.pending_draft_confidence >= self.draft_p_min
        ):
            draft_parts.append(self.pending_draft[:1])
            if max_drafts >= 2:
                if self.pending_predictor_hidden is None:
                    raise RuntimeError("MTP recursive draft hidden is not initialized")
                with self._profile_phase("PredictorDraft2"):
                    second_logits, _ = self._run_predictor(
                        batch,
                        req,
                        self.pending_predictor_hidden,
                        self.pending_draft[:1],
                        source_position,
                    )
                predictor_speculated = True
                second_draft, second_confidence = self._draft_candidate(second_logits)
                if second_confidence >= self.draft_p_min:
                    draft_parts.append(second_draft)

        draft_tokens = (
            torch.cat(draft_parts)
            if draft_parts
            else torch.empty(0, dtype=torch.int32, device=self.engine.device)
        )
        current_token = batch.input_ids[:1]
        verify_input = torch.cat([current_token, draft_tokens])
        target_graph = self.target_verify_graphs.get(verify_input.numel())
        target_state = None
        if draft_tokens.numel() and target_graph is None:
            with self._profile_phase("Snapshot"):
                target_state = _snapshot_target_runtime(self.engine, req)
        adaptive_target_started = None
        adaptive_target_ended = None
        if self.adaptive_enabled:
            adaptive_target_started = torch.cuda.Event(enable_timing=True)
            adaptive_target_ended = torch.cuda.Event(enable_timing=True)
            adaptive_target_started.record(self.engine.stream)
        with self._profile_phase(f"TargetVerify{verify_input.numel()}"):
            verify_logits, verify_expanded = self._run_target_extension(
                batch, req, verify_input, source_position
            )
            verify_tokens = torch.argmax(verify_logits, dim=-1).to(torch.int32)
        if adaptive_target_ended is not None:
            adaptive_target_ended.record(self.engine.stream)
        with self._profile_phase("DecisionSync"):
            matches = (
                (verify_tokens[: draft_tokens.numel()] == draft_tokens)
                .to(device="cpu")
                .tolist()
            )
        accepted_count = 0
        for matches_draft in matches:
            if not matches_draft:
                break
            accepted_count += 1

        output = torch.cat(
            [
                draft_tokens[:accepted_count],
                verify_tokens[accepted_count : accepted_count + 1],
            ]
        )
        committed_expanded = verify_expanded[: output.numel()]
        if accepted_count < draft_tokens.numel():
            if target_graph is not None:
                with self._profile_phase("PrefixCommit"):
                    target_graph.commit_prefix(req, accepted_count)
            else:
                assert target_state is not None
                with self._profile_phase("Restore"):
                    _restore_target_runtime(self.engine, req, target_state)
                replay_count = accepted_count + 1
                with self._profile_phase(f"TargetReplay{replay_count}"):
                    _, committed_expanded = self._run_target_extension(
                        batch,
                        req,
                        verify_input[:replay_count],
                        source_position,
                    )

        if not pld_only:
            if predictor_speculated:
                self.predictor_cached_len = source_position
            with self._profile_phase(f"PredictorCommit{output.numel()}"):
                draft_logits, predictor_expanded = self._run_predictor(
                    batch,
                    req,
                    committed_expanded,
                    output,
                    source_position,
                )

            self.pending_draft, self.pending_draft_confidence = self._draft_candidate(
                draft_logits
            )
            if self.max_supported_drafts > 1:
                self.pending_predictor_hidden = predictor_expanded[-1:].clone()
        if adaptive_cycle_started is not None and not disable_after_cycle:
            assert adaptive_target_started is not None
            assert adaptive_target_ended is not None
            adaptive_cycle_ended = torch.cuda.Event(enable_timing=True)
            adaptive_cycle_ended.record(self.engine.stream)
            self._adaptive_pending.append(
                _AdaptiveTimingSample(
                    width=draft_tokens.numel(),
                    accepted=accepted_count,
                    wall_started=adaptive_wall_started,
                    cycle_started=adaptive_cycle_started,
                    target_started=adaptive_target_started,
                    target_ended=adaptive_target_ended,
                    cycle_ended=adaptive_cycle_ended,
                )
            )
        if disable_after_cycle:
            self.adaptive_disabled_uid = req.uid
            self.adaptive_decision_cycle = self.metrics.cycles + 1
            logger.info_rank0(
                "Qwen4-Exp MTP adaptive fallback: "
                f"cycle={self.adaptive_decision_cycle}, ordinary decode selected"
            )
        if self.pld_index is not None:
            # The decision sync above already drained past verification, so
            # this short D2H copy of the emitted tokens does not stall.
            with self._profile_phase("PLDCommit"):
                self.pld_index.extend(output.to(device="cpu").tolist())
        req.complete_one()

        self.metrics.cycles += 1
        self.metrics.proposed_drafts += draft_tokens.numel()
        self.metrics.accepted_drafts += accepted_count
        self.metrics.emitted_tokens += output.numel()
        if pld_draft_count:
            self.metrics.pld_cycles += 1
            self.metrics.pld_proposed_drafts += pld_draft_count
            self.metrics.pld_accepted_drafts += accepted_count
        if self.timing_enabled and len(self.metrics.cycle_trace) < 4096:
            self.metrics.cycle_trace.append(
                {
                    "cycle": self.metrics.cycles,
                    "width": draft_tokens.numel(),
                    "accepted": accepted_count,
                    "emitted": output.numel(),
                    "pld": pld_draft_count,
                }
            )
        if self.log_interval and self.metrics.cycles % self.log_interval == 0:
            acceptance = self.metrics.acceptance_rate
            acceptance_text = f"{acceptance:.3f}" if acceptance is not None else "n/a"
            logger.info_rank0(
                "Qwen4-Exp MTP: "
                f"cycles={self.metrics.cycles}, "
                f"acceptance={acceptance_text}, "
                f"tokens/cycle={self.metrics.emitted_tokens / self.metrics.cycles:.3f}"
            )
        return output, accepted_count == draft_tokens.numel()

    def _pld_draft_tensor(self, tokens: list[int]) -> torch.Tensor:
        count = len(tokens)
        for index, token in enumerate(tokens):
            self._pld_host_tokens[index] = token
        self._pld_device_tokens[:count].copy_(
            self._pld_host_tokens[:count], non_blocking=True
        )
        return self._pld_device_tokens[:count]

    def _draft_candidate(self, logits: torch.Tensor) -> tuple[torch.Tensor, float]:
        last = logits[-1]
        top_logit, token = torch.max(last, dim=-1, keepdim=True)
        token = token.to(torch.int32)
        if self.draft_p_min <= 0.0:
            return token, 1.0
        confidence = torch.exp(top_logit - torch.logsumexp(last, dim=-1))
        return token, float(confidence.item())

    def _run_predictor(
        self,
        batch: Batch,
        req: Req,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        source_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[0] != token_ids.numel():
            raise ValueError("MTP hidden/token row counts differ")
        if source_start != self.predictor_cached_len:
            raise RuntimeError(
                f"MTP predictor starts at {source_start}, "
                f"but cache ends at {self.predictor_cached_len}"
            )
        graph = (
            self.predictor_graphs.get(token_ids.numel()) if batch.is_decode else None
        )
        if graph is not None:
            logits, expanded = graph.replay(req, hidden, token_ids, source_start)
            self.predictor_cached_len += token_ids.numel()
            return logits, expanded
        state = _snapshot_mutable_forward_state(batch, req)
        last_logits = None
        last_expanded = None
        try:
            for offset in range(0, token_ids.numel(), self.max_predictor_chunk):
                count = min(self.max_predictor_chunk, token_ids.numel() - offset)
                start = source_start + offset
                end = start + count
                batch.phase = "prefill"
                batch.padded_reqs = batch.reqs
                batch.input_ids = token_ids[offset : offset + count]
                batch.host_input_ids = None
                batch.positions = torch.arange(
                    start, end, dtype=torch.int32, device=self.engine.device
                )
                batch.out_loc = self.engine.page_table[req.table_idx, start:end]
                batch.fla_metadata = None
                batch.moe_decode_cache = True
                req.cached_len = start
                req.device_len = end
                self.engine.attn_backend.prepare_metadata(batch)
                with self.engine.ctx.forward_batch(batch):
                    last_logits, last_expanded = self.model.mtp_forward(
                        hidden[offset : offset + count],
                        token_ids[offset : offset + count],
                    )
                self.predictor_cached_len = end
        finally:
            _restore_mutable_forward_state(batch, req, state)
        if last_logits is None or last_expanded is None:
            raise ValueError("MTP predictor received no tokens")
        return last_logits, last_expanded

    def _run_target_extension(
        self,
        batch: Batch,
        req: Req,
        token_ids: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = _snapshot_mutable_forward_state(batch, req)
        try:
            end = start + token_ids.numel()
            host_end = req.input_ids.numel()
            if host_end < start or host_end > end:
                raise RuntimeError(
                    f"MTP target extension host length {host_end} is outside "
                    f"forward range [{start}, {end}]"
                )
            if host_end < end:
                req.append_host(
                    token_ids[host_end - start :].to(
                        device="cpu", dtype=req.input_ids.dtype
                    )
                )
            req.cached_len = start
            req.device_len = end
            target_graph = self.target_verify_graphs.get(token_ids.numel())
            if target_graph is not None:
                return target_graph.replay(req, token_ids, start)
            if token_ids.numel() == 1 and self.engine.graph_runner.can_use_cuda_graph(
                batch
            ):
                batch.phase = "decode"
                batch.input_ids = token_ids
                batch.positions = torch.arange(
                    start, end, dtype=torch.int32, device=self.engine.device
                )
                batch.out_loc = self.engine.page_table[req.table_idx, start:end]
                with self.engine.ctx.forward_batch(batch):
                    return self.engine.graph_runner.replay(batch, return_expanded=True)
            batch.phase = "prefill"
            batch.padded_reqs = batch.reqs
            batch.input_ids = token_ids
            batch.host_input_ids = None
            batch.positions = torch.arange(
                start, end, dtype=torch.int32, device=self.engine.device
            )
            batch.out_loc = self.engine.page_table[req.table_idx, start:end]
            batch.moe_decode_cache = True
            batch.fla_metadata = build_fla_metadata(batch, self.engine.device)
            self.engine.attn_backend.prepare_metadata(batch)
            with self.engine.ctx.forward_batch(batch):
                hidden, expanded = self.model.model.forward(
                    batch.input_ids, return_expanded=True
                )
                logits = project_lm_head_all_positions(self.model.lm_head, hidden)
            return logits, expanded
        finally:
            _restore_mutable_forward_state(batch, req, state)


__all__ = [
    "MTPAdaptiveController",
    "MTPMetrics",
    "MTPWorker",
    "choose_adaptive_draft_width",
]
