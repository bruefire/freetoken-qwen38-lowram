"""Offline MTP policy replay over recorded decode-profile sweep traces.

Replays the per-cycle acceptance streams recorded by ``bench decode-profile
--mtp-sweep`` (report version >= 10 with cycle traces) through alternative
draft-width policies, without a model load or a GPU. Costs come from the same
report: per-phase CUDA means plus a per-prompt host residual calibrated so the
recorded fixed-width arms reproduce their own measured cycle time.

The simulation is an approximation, not a measurement:

- A policy cycle at output position ``t`` uses the recorded outcome of the
  same-width arm's cycle covering ``t``. Cycle boundaries of the simulated
  policy and the recorded arm drift apart after rejections.
- Extrapolating past the recorded output wraps the acceptance stream, which
  assumes the trajectory's acceptance statistics are stationary.
- Confidence-threshold policies cannot be simulated because per-cycle draft
  confidence is not recorded.

Rank policies with it; verify winners with a real single-load sweep.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from freetoken.pld_index import PLDIndex

_FIXED_ARMS = ("one_draft", "two_drafts", "two_drafts_p60")
_STREAM_ARM = {1: "one_draft", 2: "two_drafts"}


@dataclass(frozen=True)
class CycleOutcome:
    width: int
    accepted: int

    @property
    def emitted(self) -> int:
        return self.accepted + 1


@dataclass
class ArmData:
    name: str
    outcomes: list[CycleOutcome]
    measured_ms_per_cycle: float
    measured_tps: float


class AcceptanceStream:
    """Recorded outcomes of one arm, addressable by output position."""

    def __init__(self, outcomes: list[CycleOutcome]):
        if not outcomes:
            raise ValueError("acceptance stream needs at least one cycle")
        self.outcomes = outcomes
        self.starts: list[int] = []
        position = 0
        for outcome in outcomes:
            self.starts.append(position)
            position += outcome.emitted
        self.total_emitted = position

    def outcome_at(self, position: int) -> CycleOutcome:
        position %= self.total_emitted
        index = bisect.bisect_right(self.starts, position) - 1
        return self.outcomes[index]


@dataclass
class PromptData:
    name: str
    baseline_ms: float
    baseline_tps: float
    recorded_tokens: int
    arms: dict[str, ArmData]
    measured_tps: dict[str, float]
    phase_ms: dict[str, float]
    host_ms: float = 0.0
    streams: dict[int, AcceptanceStream] = field(default_factory=dict)
    prompt_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    def cycle_cost_ms(self, width: int, accepted: int) -> float:
        """Estimated full MTP cycle cost for one draft width and outcome."""
        if width not in (0, 1, 2) or not 0 <= accepted <= width:
            raise ValueError(f"invalid cycle width={width} accepted={accepted}")
        if width == 0:
            # Recorded zero-draft MTP cycle (output-tail clamp): one ordinary
            # target token plus the predictor commit that keeps state fresh.
            return self.shadow_cost_ms() + self.host_ms
        cost = (
            self.phase_ms[f"TargetVerify{width + 1}"]
            + self._predictor_commit_ms(accepted + 1)
            + self.phase_ms.get("DecisionSync", 0.0)
            + self.host_ms
        )
        if width == 2:
            cost += self.phase_ms.get("PredictorDraft2", 0.0)
        if accepted < width:
            cost += self.phase_ms.get("PrefixCommit", 0.0)
        return cost

    def shadow_cost_ms(self) -> float:
        """Ordinary-speed token that still commits the predictor (width 0)."""
        return (
            self.baseline_ms
            + self._predictor_commit_ms(1)
            + self.phase_ms.get("DecisionSync", 0.0)
        )

    def pld_cycle_cost_ms(self, width: int, accepted: int) -> float:
        """Cycle cost for prompt-lookup drafts: verification only.

        Lookup drafts need no recursive predictor forward and no predictor
        commit. Verification rows beyond the captured three-token graph are
        extrapolated linearly from the two- and three-row graph costs.
        """
        if width < 1 or not 0 <= accepted <= width:
            raise ValueError(f"invalid PLD cycle width={width} accepted={accepted}")
        two_rows = self.phase_ms["TargetVerify2"]
        three_rows = self.phase_ms["TargetVerify3"]
        if width == 1:
            verify = two_rows
        else:
            verify = three_rows + (three_rows - two_rows) * (width - 2)
        cost = verify + self.phase_ms.get("DecisionSync", 0.0) + self.host_ms
        if accepted < width:
            cost += self.phase_ms.get("PrefixCommit", 0.0)
        return cost

    def _predictor_commit_ms(self, tokens: int) -> float:
        # A prompt with no fully rejected (or fully accepted) cycles has no
        # sample for that commit size; the sizes measured within 1 ms of each
        # other, so the nearest recorded mean is an adequate stand-in.
        exact = self.phase_ms.get(f"PredictorCommit{tokens}")
        if exact is not None:
            return exact
        available = [
            value
            for phase, value in self.phase_ms.items()
            if phase.startswith("PredictorCommit")
        ]
        if not available:
            raise ValueError(f"prompt {self.name} recorded no predictor commits")
        return statistics.fmean(available)


def _clipped_outcomes(mtp: dict[str, Any]) -> list[CycleOutcome]:
    # Reports written before the _read_mtp_stats snapshot fix can carry
    # later-arm entries appended to the final adaptive trace; only the first
    # `cycles` entries belong to the arm.
    trace = mtp["cycle_trace"][: mtp["cycles"]]
    return [CycleOutcome(entry["width"], entry["accepted"]) for entry in trace]


def _pool_phase_means(
    arms: dict[str, ArmData], raw: dict[str, Any]
) -> dict[str, float]:
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    for arm_name in arms:
        phase_timing = raw[arm_name]["measured"]["mtp"]["phase_timing"]
        for phase, stats in phase_timing.items():
            totals[phase] = totals.get(phase, 0.0) + stats["total_ms"]
            calls[phase] = calls.get(phase, 0) + stats["calls"]
    return {phase: totals[phase] / calls[phase] for phase in totals if calls[phase]}


def _calibrate_host_ms(prompt: PromptData) -> float:
    residuals: list[tuple[float, int]] = []
    for arm_name in _FIXED_ARMS:
        arm = prompt.arms.get(arm_name)
        if arm is None or not arm.outcomes:
            continue
        prompt.host_ms = 0.0
        modeled = statistics.fmean(
            prompt.cycle_cost_ms(outcome.width, outcome.accepted)
            for outcome in arm.outcomes
        )
        residuals.append((arm.measured_ms_per_cycle - modeled, len(arm.outcomes)))
    if not residuals:
        return 0.0
    weight = sum(count for _, count in residuals)
    return max(0.0, sum(value * count for value, count in residuals) / weight)


def load_prompts(report: dict[str, Any]) -> list[PromptData]:
    sweep = report.get("mtp_sweep")
    if not sweep or "prompts" not in sweep:
        raise ValueError("report has no mtp_sweep prompt data")
    prompts = []
    for name, entry in sweep["prompts"].items():
        baseline = entry["baseline"]["timing"]
        variants = entry["variants"]
        arms: dict[str, ArmData] = {}
        for arm_name, variant in variants.items():
            measured = variant["measured"]
            mtp = measured["mtp"]
            outcomes = _clipped_outcomes(mtp)
            timing = measured["timing"]
            cycles = mtp["cycles"] or 1
            arms[arm_name] = ArmData(
                name=arm_name,
                outcomes=outcomes,
                measured_ms_per_cycle=timing["decode_span_ms"] / cycles,
                measured_tps=timing["decode_output_tokens_per_second"],
            )
        missing = [arm for arm in _STREAM_ARM.values() if arm not in arms]
        if missing:
            raise ValueError(f"prompt {name} lacks required arms: {missing}")
        prompt = PromptData(
            name=name,
            baseline_ms=baseline["decode_ms_per_output_token"],
            baseline_tps=baseline["decode_output_tokens_per_second"],
            recorded_tokens=baseline["decode_output_tokens"],
            arms=arms,
            measured_tps={arm_name: arm.measured_tps for arm_name, arm in arms.items()},
            phase_ms=_pool_phase_means(arms, variants),
            prompt_ids=list(entry.get("prompt_token_ids", [])),
            output_ids=list(entry["baseline"].get("output_token_ids", [])),
        )
        for width, arm_name in _STREAM_ARM.items():
            prompt.streams[width] = AcceptanceStream(arms[arm_name].outcomes)
        prompt.host_ms = _calibrate_host_ms(prompt)
        prompts.append(prompt)
    return prompts


class Policy:
    """One simulated draft-width policy. Subclasses choose 0, 1 or 2."""

    name = "policy"

    def reset(self, prompt: PromptData) -> None:
        pass

    def choose(self) -> int:
        raise NotImplementedError

    def observe(self, width: int, accepted: int | None) -> None:
        pass

    def extra_cost_ms(self) -> float:
        """One-off cost charged this cycle (e.g. predictor resync)."""
        return 0.0

    def uses_shadow_fallback(self) -> bool:
        return False

    def forced_outcome(self) -> CycleOutcome | None:
        """Outcome the policy itself determines; None uses the arm streams."""
        return None

    def cycle_cost_ms(self, prompt: PromptData, width: int, accepted: int) -> float:
        return prompt.cycle_cost_ms(width, accepted)


class FixedPolicy(Policy):
    def __init__(self, width: int):
        self.width = width
        self.name = f"fixed{width}"

    def choose(self) -> int:
        return self.width


class ReplayArmPolicy(Policy):
    """Replays one recorded arm's own width/outcome sequence (validation)."""

    def __init__(self, arm_name: str):
        self.arm_name = arm_name
        self.name = f"replay:{arm_name}"

    def reset(self, prompt: PromptData) -> None:
        self.outcomes = prompt.arms[self.arm_name].outcomes
        self.index = 0

    def choose(self) -> int:
        outcome = self.outcomes[self.index % len(self.outcomes)]
        return outcome.width

    def forced_outcome(self) -> CycleOutcome:
        outcome = self.outcomes[self.index % len(self.outcomes)]
        self.index += 1
        return outcome


class PLDHybridPolicy(Policy):
    """Prompt-lookup (n-gram) drafts verified by the existing target graphs.

    Uses the same ``PLDIndex`` the engine executes: when the context's longest
    recent n-gram match (at least `min_ngram` tokens) has a recorded
    continuation, up to `draft_len` tokens of it become the draft; otherwise
    the cycle is ordinary decode. Drafting is free (a host-side table lookup),
    so speculative cycles pay only target verification. Draft correctness is
    checked against the recorded baseline trajectory, so this policy needs
    `output_token_ids` in the report.
    """

    def __init__(self, min_ngram: int = 6, max_ngram: int = 12, draft_len: int = 3):
        if draft_len < 1:
            raise ValueError("PLD draft length must be positive")
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.draft_len = draft_len
        self.name = f"pld(n{min_ngram},k{draft_len})"

    def reset(self, prompt: PromptData) -> None:
        if not prompt.output_ids:
            raise ValueError(f"prompt {prompt.name} has no recorded output ids")
        self.prompt = prompt
        self.index = PLDIndex(self.min_ngram, self.max_ngram)
        self.index.extend(prompt.prompt_ids)
        self.position = 0
        self.pending: CycleOutcome | None = None

    def choose(self) -> int:
        self.pending = None
        draft = self.index.draft(self.draft_len)
        if not draft:
            return 0
        out = self.prompt.output_ids
        cursor = self.position % len(out)
        accepted = 0
        for proposed, actual in zip(draft, out[cursor : cursor + len(draft)]):
            if proposed != actual:
                break
            accepted += 1
        self.pending = CycleOutcome(len(draft), accepted)
        return len(draft)

    def forced_outcome(self) -> CycleOutcome | None:
        return self.pending

    def cycle_cost_ms(self, prompt: PromptData, width: int, accepted: int) -> float:
        return prompt.pld_cycle_cost_ms(width, accepted)

    def observe(self, width: int, accepted: int | None) -> None:
        emitted = 1 if accepted is None else accepted + 1
        out = self.prompt.output_ids
        for offset in range(emitted):
            self.index.extend([out[(self.position + offset) % len(out)]])
        self.position += emitted


class V3Policy(Policy):
    """The shipped request-local controller: observe 64, gate, then cost."""

    name = "v3"

    def __init__(
        self,
        observe_cycles: int = 64,
        min_acceptance: float = 0.75,
        samples: int = 4,
        window: int = 64,
        probe_interval: int = 16,
    ):
        self.observe_cycles = observe_cycles
        self.min_acceptance = min_acceptance
        self.samples = samples
        self.window = window
        self.probe_interval = probe_interval

    def reset(self, prompt: PromptData) -> None:
        self.prompt = prompt
        self.cycles = 0
        self.disabled = False
        self.outcomes: deque[CycleOutcome] = deque(maxlen=self.window)
        self.costs = {1: deque(maxlen=self.window), 2: deque(maxlen=self.window)}
        self.last_sample = {1: -1, 2: -1}

    def _survival(self) -> tuple[float, float]:
        first = [outcome.accepted >= 1 for outcome in self.outcomes]
        second = [
            outcome.accepted >= 2 for outcome in self.outcomes if outcome.width >= 2
        ]
        s1 = sum(first) / len(first) if first else 0.0
        s2 = sum(second) / len(second) if second else 0.0
        return s1, min(s1, s2)

    def choose(self) -> int:
        if self.disabled:
            return 0
        if self.cycles < self.observe_cycles:
            return 2
        for width in (1, 2):
            if len(self.costs[width]) < self.samples:
                return width
        s1, s2 = self._survival()
        if (s1 + s2) / 2.0 < self.min_acceptance:
            self.disabled = True
            return 0
        throughput1 = (1.0 + s1) / statistics.median(self.costs[1])
        throughput2 = (1.0 + s1 + s2) / statistics.median(self.costs[2])
        selected = 2 if throughput2 > throughput1 else 1
        stalest = min((1, 2), key=self.last_sample.__getitem__)
        if self.cycles - self.last_sample[stalest] >= self.probe_interval:
            return stalest
        return selected

    def observe(self, width: int, accepted: int | None) -> None:
        self.cycles += 1
        if width and accepted is not None:
            self.outcomes.append(CycleOutcome(width, accepted))
            self.costs[width].append(self.prompt.cycle_cost_ms(width, accepted))
            self.last_sample[width] = self.cycles


class EmaCostPolicy(Policy):
    """Continuous cost-aware control from EMA survival with an optimistic prior.

    No fixed observation window: survival estimates start from a prior and are
    updated every cycle, and the width is re-chosen every cycle by expected
    tokens per millisecond, including the ordinary path as width 0. Width-0
    cycles run in shadow mode (ordinary speed plus a predictor commit), so the
    policy can resume speculation instantly and re-probes with backoff.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        prior_survival: float = 0.85,
        prior_weight: int = 8,
        probe_interval: int = 16,
        probe_backoff: float = 2.0,
        probe_interval_max: int = 256,
    ):
        self.alpha = alpha
        self.prior_survival = prior_survival
        self.prior_weight = prior_weight
        self.probe_interval = probe_interval
        self.probe_backoff = probe_backoff
        self.probe_interval_max = probe_interval_max
        self.name = f"ema-cost(a={alpha})"

    def reset(self, prompt: PromptData) -> None:
        self.prompt = prompt
        self.s1 = self.prior_survival
        self.s2 = self.prior_survival**2
        self.seen = 0
        self.cycles = 0
        self.idle = 0
        self.current_probe_interval = self.probe_interval

    def uses_shadow_fallback(self) -> bool:
        return True

    def _blended(self) -> tuple[float, float]:
        if self.seen >= self.prior_weight:
            return self.s1, self.s2
        mix = self.seen / self.prior_weight
        s1 = self.prior_survival * (1.0 - mix) + self.s1 * mix
        s2 = self.prior_survival**2 * (1.0 - mix) + self.s2 * mix
        return s1, min(s1, s2)

    def choose(self) -> int:
        s1, s2 = self._blended()
        # Expected cost of a width-w cycle marginalizes accepted counts.
        cost1 = s1 * self.prompt.cycle_cost_ms(1, 1) + (1 - s1) * (
            self.prompt.cycle_cost_ms(1, 0)
        )
        cost2 = (
            s2 * self.prompt.cycle_cost_ms(2, 2)
            + (s1 - s2) * self.prompt.cycle_cost_ms(2, 1)
            + (1 - s1) * self.prompt.cycle_cost_ms(2, 0)
        )
        best_width = 0
        best = 1.0 / self.prompt.shadow_cost_ms()
        for width, throughput in ((1, (1 + s1) / cost1), (2, (1 + s1 + s2) / cost2)):
            if throughput > best:
                best_width = width
                best = throughput
        if best_width == 0:
            if self.idle >= self.current_probe_interval:
                self.idle = 0
                self.current_probe_interval = min(
                    self.probe_interval_max,
                    int(self.current_probe_interval * self.probe_backoff),
                )
                return 2
            return 0
        self.current_probe_interval = self.probe_interval
        return best_width

    def observe(self, width: int, accepted: int | None) -> None:
        self.cycles += 1
        if not width or accepted is None:
            self.idle += 1
            return
        self.idle = 0
        self.seen += 1
        self.s1 += self.alpha * (float(accepted >= 1) - self.s1)
        if width >= 2:
            self.s2 += self.alpha * (float(accepted >= 2) - self.s2)
        self.s2 = min(self.s1, self.s2)


class EarlyResumePolicy(Policy):
    """Short observation window plus a resumable ordinary fallback.

    Observes at width 2 for `observe_cycles`, falls back to true ordinary
    decode when yield is low, and periodically pays a predictor resync
    (`resync_ms_per_token` times the ordinary gap) to re-probe. The re-probe
    gap doubles after every failed probe.
    """

    def __init__(
        self,
        observe_cycles: int = 16,
        min_acceptance: float = 0.75,
        probe_cycles: int = 8,
        reprobe_gap_tokens: int = 64,
        reprobe_gap_max: int = 1024,
        resync_ms_per_token: float = 0.5,
    ):
        self.observe_cycles = observe_cycles
        self.min_acceptance = min_acceptance
        self.probe_cycles = probe_cycles
        self.reprobe_gap_tokens = reprobe_gap_tokens
        self.reprobe_gap_max = reprobe_gap_max
        self.resync_ms_per_token = resync_ms_per_token
        self.name = f"early-resume(r={resync_ms_per_token})"

    def reset(self, prompt: PromptData) -> None:
        self.prompt = prompt
        self.state = "observe"
        self.window: deque[CycleOutcome] = deque(maxlen=64)
        self.state_cycles = 0
        self.gap_tokens = 0
        self.current_gap = self.reprobe_gap_tokens
        self.pending_resync = 0.0

    def _yield(self) -> float:
        proposed = sum(outcome.width for outcome in self.window)
        accepted = sum(outcome.accepted for outcome in self.window)
        return accepted / proposed if proposed else 0.0

    def _speculative_width(self) -> int:
        first = [outcome.accepted >= 1 for outcome in self.window]
        second = [
            outcome.accepted >= 2 for outcome in self.window if outcome.width >= 2
        ]
        s1 = sum(first) / len(first) if first else 0.0
        s2 = min(s1, sum(second) / len(second) if second else 0.0)
        throughput1 = (1 + s1) / self.prompt.cycle_cost_ms(1, 1)
        throughput2 = (1 + s1 + s2) / self.prompt.cycle_cost_ms(2, 2)
        return 2 if throughput2 > throughput1 else 1

    def choose(self) -> int:
        if self.state in ("observe", "probe"):
            return 2
        if self.state == "on":
            return self._speculative_width()
        if self.gap_tokens >= self.current_gap:
            self.pending_resync = self.gap_tokens * self.resync_ms_per_token
            self.gap_tokens = 0
            self.state = "probe"
            self.state_cycles = 0
            self.window.clear()
            return 2
        return 0

    def extra_cost_ms(self) -> float:
        cost = self.pending_resync
        self.pending_resync = 0.0
        return cost

    def observe(self, width: int, accepted: int | None) -> None:
        if width and accepted is not None:
            self.window.append(CycleOutcome(width, accepted))
        self.state_cycles += 1
        if self.state == "observe" and self.state_cycles >= self.observe_cycles:
            self._decide(after_probe=False)
        elif self.state == "probe" and self.state_cycles >= self.probe_cycles:
            self._decide(after_probe=True)
        elif self.state == "off":
            self.gap_tokens += 1
        elif self.state == "on" and self._yield() < self.min_acceptance:
            self.state = "off"
            self.state_cycles = 0
            self.gap_tokens = 0

    def _decide(self, after_probe: bool) -> None:
        if self._yield() >= self.min_acceptance:
            self.state = "on"
            self.current_gap = self.reprobe_gap_tokens
        else:
            self.state = "off"
            self.gap_tokens = 0
            if after_probe:
                self.current_gap = min(self.reprobe_gap_max, self.current_gap * 2)
        self.state_cycles = 0


class OracleCyclePolicy(Policy):
    """Hindsight upper bound: per cycle, the best width given known outcomes."""

    name = "oracle-cycle"

    def reset(self, prompt: PromptData) -> None:
        self.prompt = prompt
        self.position = 0

    def choose(self) -> int:
        best_width = 0
        best = 1.0 / self.prompt.baseline_ms
        for width in (1, 2):
            outcome = self.prompt.streams[width].outcome_at(self.position)
            accepted = min(outcome.accepted, width)
            rate = (accepted + 1) / self.prompt.cycle_cost_ms(width, accepted)
            if rate > best:
                best_width = width
                best = rate
        return best_width

    def observe(self, width: int, accepted: int | None) -> None:
        self.position += 1 if accepted is None else accepted + 1


@dataclass
class SimulationResult:
    tokens: int
    total_ms: float
    cycles: int

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / (self.total_ms / 1000.0)


def simulate(prompt: PromptData, policy: Policy, total_tokens: int) -> SimulationResult:
    policy.reset(prompt)
    position = 0
    total_ms = 0.0
    cycles = 0
    while position < total_tokens:
        width = policy.choose()
        total_ms += policy.extra_cost_ms()
        if width == 0:
            if policy.uses_shadow_fallback():
                total_ms += prompt.shadow_cost_ms()
            else:
                total_ms += prompt.baseline_ms
            position += 1
            policy.observe(0, None)
        else:
            outcome = policy.forced_outcome()
            if outcome is None:
                outcome = prompt.streams[width].outcome_at(position)
            accepted = min(outcome.accepted, width)
            total_ms += policy.cycle_cost_ms(prompt, width, accepted)
            position += accepted + 1
            policy.observe(width, accepted)
        cycles += 1
    return SimulationResult(tokens=position, total_ms=total_ms, cycles=cycles)


def validate(prompts: list[PromptData]) -> list[dict[str, Any]]:
    """Replay each recorded arm through the cost model; report the error."""
    rows = []
    for prompt in prompts:
        for arm_name, arm in prompt.arms.items():
            if arm_name == "adaptive_two_drafts" or not arm.outcomes:
                continue
            policy = ReplayArmPolicy(arm_name)
            policy.reset(prompt)
            modeled = statistics.fmean(
                prompt.cycle_cost_ms(outcome.width, outcome.accepted)
                for outcome in arm.outcomes
            )
            rows.append(
                {
                    "prompt": prompt.name,
                    "arm": arm_name,
                    "measured_ms_per_cycle": arm.measured_ms_per_cycle,
                    "modeled_ms_per_cycle": modeled,
                    "error_pct": 100.0
                    * (modeled - arm.measured_ms_per_cycle)
                    / arm.measured_ms_per_cycle,
                }
            )
    return rows


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def build_policies(args: argparse.Namespace) -> list[Policy]:
    policies: list[Policy] = [
        FixedPolicy(1),
        FixedPolicy(2),
        V3Policy(),
        EmaCostPolicy(alpha=args.ema_alpha),
    ]
    for resync in args.resync_ms_per_token:
        policies.append(EarlyResumePolicy(resync_ms_per_token=resync))
    policies.append(
        PLDHybridPolicy(min_ngram=args.pld_min_ngram, draft_len=args.pld_draft_len)
    )
    policies.append(OracleCyclePolicy())
    return policies


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.report) as handle:
        report = json.load(handle)
    prompts = load_prompts(report)
    policies = build_policies(args)
    result: dict[str, Any] = {
        "report": args.report,
        "output_tokens": args.output_tokens,
        "validation": validate(prompts),
        "prompts": {},
    }
    speedups: dict[str, list[float]] = {}
    for prompt in prompts:
        tokens = args.output_tokens or prompt.recorded_tokens
        row: dict[str, Any] = {
            "baseline_tps": prompt.baseline_tps,
            "measured_tps": prompt.measured_tps,
            "host_ms": prompt.host_ms,
            "policies": {},
        }
        for policy in policies:
            sim = simulate(prompt, policy, tokens)
            # tokens simulated can overshoot by up to two accepted drafts;
            # compare times at equal token counts.
            speedup = (prompt.baseline_ms * sim.tokens) / sim.total_ms
            row["policies"][policy.name] = {
                "est_tps": prompt.baseline_tps * speedup,
                "speedup": speedup,
                "cycles": sim.cycles,
            }
            speedups.setdefault(policy.name, []).append(speedup)
        # A perfect cross-request prior would pick the request's best fixed
        # policy (including ordinary decode) with no observation cost.
        oracle_fixed = max(
            1.0,
            row["policies"]["fixed1"]["speedup"],
            row["policies"]["fixed2"]["speedup"],
        )
        row["policies"]["oracle-fixed"] = {
            "est_tps": prompt.baseline_tps * oracle_fixed,
            "speedup": oracle_fixed,
            "cycles": 0,
        }
        speedups.setdefault("oracle-fixed", []).append(oracle_fixed)
        result["prompts"][prompt.name] = row
    result["geomean_speedup"] = {
        name: geometric_mean(values) for name, values in speedups.items()
    }
    return result


def _print_report(result: dict[str, Any]) -> None:
    errors = [abs(row["error_pct"]) for row in result["validation"]]
    print(
        f"cost-model validation over {len(errors)} recorded arms: "
        f"mean |error| {statistics.fmean(errors):.2f}%, "
        f"max |error| {max(errors):.2f}%"
    )
    worst = max(result["validation"], key=lambda row: abs(row["error_pct"]))
    print(
        f"  worst: {worst['prompt']}/{worst['arm']} "
        f"modeled {worst['modeled_ms_per_cycle']:.2f} vs "
        f"measured {worst['measured_ms_per_cycle']:.2f} ms/cycle"
    )
    policy_names = list(result["geomean_speedup"])
    header = f"{'prompt':26s}" + "".join(f"{name:>22s}" for name in policy_names)
    print()
    print(header)
    for prompt_name, row in result["prompts"].items():
        cells = "".join(
            f"{row['policies'][name]['speedup']:>21.3f}x" for name in policy_names
        )
        print(f"{prompt_name:26s}{cells}")
    cells = "".join(
        f"{result['geomean_speedup'][name]:>21.3f}x" for name in policy_names
    )
    print(f"{'geomean':26s}{cells}")


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Simulate MTP draft policies over recorded sweep traces",
    )
    parser.add_argument("report", help="decode-profile report.json with --mtp-sweep")
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=0,
        help="simulate this many output tokens (0 = recorded length; longer "
        "runs wrap the recorded acceptance stream)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for the ema-cost policy",
    )
    parser.add_argument(
        "--resync-ms-per-token",
        type=float,
        nargs="+",
        default=[0.5, 2.0],
        help="assumed predictor catch-up cost per ordinary token for the "
        "early-resume policy (one policy per value)",
    )
    parser.add_argument(
        "--pld-min-ngram",
        type=int,
        default=6,
        help="minimum suffix match length for the prompt-lookup draft policy",
    )
    parser.add_argument(
        "--pld-draft-len",
        type=int,
        default=3,
        help="maximum draft tokens per prompt-lookup cycle",
    )
    parser.add_argument("-o", "--json-output", help="also write the result as JSON")
    return parser


def main(argv: list[str] | None = None, prog: str = "mtp-policy-replay") -> int:
    args = _parser(prog).parse_args(argv)
    result = run(args)
    _print_report(result)
    if args.json_output:
        with open(args.json_output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
