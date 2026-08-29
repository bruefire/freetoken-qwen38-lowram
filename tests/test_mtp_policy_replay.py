from __future__ import annotations

import json

import pytest
from freetoken.benchmark.mtp_policy_replay import (
    AcceptanceStream,
    CycleOutcome,
    EarlyResumePolicy,
    EmaCostPolicy,
    FixedPolicy,
    PLDHybridPolicy,
    V3Policy,
    _clipped_outcomes,
    load_prompts,
    main,
    simulate,
)

_PHASES = {
    "TargetVerify2": 18.0,
    "TargetVerify3": 30.0,
    "PredictorDraft2": 2.0,
    "PredictorCommit1": 2.0,
    "PredictorCommit2": 2.0,
    "PredictorCommit3": 2.5,
    "DecisionSync": 0.1,
    "PrefixCommit": 1.4,
}
_HOST_MS = 0.5


def _phase_timing(counts: dict[str, int]) -> dict[str, dict[str, float]]:
    return {
        phase: {
            "calls": calls,
            "total_ms": _PHASES[phase] * calls,
            "mean_ms": _PHASES[phase],
        }
        for phase, calls in counts.items()
        if calls
    }


def _cycle_cost(width: int, accepted: int) -> float:
    cost = (
        _PHASES[f"TargetVerify{width + 1}"]
        + _PHASES[f"PredictorCommit{accepted + 1}"]
        + _PHASES["DecisionSync"]
        + _HOST_MS
    )
    if width == 2:
        cost += _PHASES["PredictorDraft2"]
    if accepted < width:
        cost += _PHASES["PrefixCommit"]
    return cost


def _arm(trace: list[tuple[int, int]], extra_trace: int = 0) -> dict:
    outcomes = [CycleOutcome(width, accepted) for width, accepted in trace]
    span_ms = sum(_cycle_cost(o.width, o.accepted) for o in outcomes)
    emitted = sum(o.emitted for o in outcomes)
    counts = {
        "DecisionSync": len(outcomes),
        "TargetVerify2": sum(1 for o in outcomes if o.width == 1),
        "TargetVerify3": sum(1 for o in outcomes if o.width == 2),
        "PredictorDraft2": sum(1 for o in outcomes if o.width == 2),
        "PredictorCommit1": sum(1 for o in outcomes if o.accepted == 0),
        "PredictorCommit2": sum(1 for o in outcomes if o.accepted == 1),
        "PredictorCommit3": sum(1 for o in outcomes if o.accepted == 2),
        "PrefixCommit": sum(1 for o in outcomes if o.accepted < o.width),
    }
    cycle_trace = [
        {"cycle": index + 1, "width": o.width, "accepted": o.accepted}
        for index, o in enumerate(outcomes)
    ]
    cycle_trace += [{"cycle": 0, "width": 2, "accepted": 2}] * extra_trace
    return {
        "measured": {
            "timing": {
                "decode_span_ms": span_ms,
                "decode_output_tokens_per_second": emitted / (span_ms / 1000.0),
            },
            "mtp": {
                "cycles": len(outcomes),
                "cycle_trace": cycle_trace,
                "phase_timing": _phase_timing(counts),
            },
        }
    }


def _report(
    one_draft: list[tuple[int, int]],
    two_drafts: list[tuple[int, int]],
    baseline_ms: float = 15.0,
    output_ids: list[int] | None = None,
) -> dict:
    emitted = 40
    if output_ids is None:
        output_ids = list(range(1000, 1000 + emitted))
    return {
        "mtp_sweep": {
            "prompts": {
                "fixture": {
                    "prompt_token_ids": [1, 2, 3, 4],
                    "baseline": {
                        "timing": {
                            "decode_ms_per_output_token": baseline_ms,
                            "decode_output_tokens_per_second": 1000.0 / baseline_ms,
                            "decode_output_tokens": emitted,
                        },
                        "output_token_ids": output_ids,
                    },
                    "variants": {
                        "one_draft": _arm(one_draft),
                        "two_drafts": _arm(two_drafts),
                    },
                }
            }
        }
    }


_GOOD_ONE = [(1, 1)] * 20
_GOOD_TWO = [(2, 2)] * 14
_BAD_TWO = [(2, 0)] * 40


def test_clipped_outcomes_drops_contaminated_tail():
    arm = _arm([(2, 2)] * 5, extra_trace=94)
    mtp = arm["measured"]["mtp"]
    assert len(mtp["cycle_trace"]) == 99
    assert len(_clipped_outcomes(mtp)) == 5


def test_acceptance_stream_position_lookup_and_wrap():
    stream = AcceptanceStream(
        [CycleOutcome(2, 2), CycleOutcome(2, 0), CycleOutcome(1, 1)]
    )
    assert stream.total_emitted == 6
    assert stream.outcome_at(0).accepted == 2
    assert stream.outcome_at(2).accepted == 2
    assert stream.outcome_at(3).accepted == 0
    assert stream.outcome_at(4).accepted == 1
    assert stream.outcome_at(6).accepted == 2  # wraps


def test_cost_model_and_host_calibration():
    # Mixed outcomes so every phase has recorded samples.
    one = [(1, 1)] * 16 + [(1, 0)] * 4
    two = [(2, 2)] * 10 + [(2, 1)] * 2 + [(2, 0)] * 2
    prompts = load_prompts(_report(one, two))
    prompt = prompts[0]
    assert prompt.host_ms == pytest.approx(_HOST_MS, abs=1e-6)
    for width, accepted in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
        assert prompt.cycle_cost_ms(width, accepted) == pytest.approx(
            _cycle_cost(width, accepted)
        )
    # No recorded PredictorCommit1 is fine: the nearest recorded commit mean
    # stands in.
    del prompt.phase_ms["PredictorCommit1"]
    assert prompt.cycle_cost_ms(1, 0) == pytest.approx(_cycle_cost(1, 0), abs=0.5)


def test_fixed_policy_reproduces_arm_arithmetic():
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO))
    prompt = prompts[0]
    sim = simulate(prompt, FixedPolicy(2), 30)
    assert sim.tokens == 30
    assert sim.cycles == 10
    assert sim.total_ms == pytest.approx(10 * _cycle_cost(2, 2))
    ordinary = simulate(prompt, FixedPolicy(2), 30).total_ms
    assert ordinary < 30 * prompt.baseline_ms


def test_v3_policy_disables_on_rejection_and_stays_ordinary():
    prompts = load_prompts(_report([(1, 0)] * 40, _BAD_TWO))
    prompt = prompts[0]
    policy = V3Policy(observe_cycles=8, samples=2, probe_interval=16)
    sim = simulate(prompt, policy, 200)
    assert policy.disabled
    # Observation plus calibration cycles, then ordinary decode to the end.
    speculative_cycles = 8 + 2
    ordinary_tokens = sim.tokens - speculative_cycles
    expected = (
        8 * _cycle_cost(2, 0)
        + 2 * _cycle_cost(1, 0)
        + ordinary_tokens * prompt.baseline_ms
    )
    assert sim.total_ms == pytest.approx(expected)


def test_v3_policy_keeps_speculating_on_acceptance():
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO))
    policy = V3Policy(observe_cycles=8, samples=2)
    sim = simulate(prompts[0], policy, 300)
    assert not policy.disabled
    assert sim.total_ms < 300 * prompts[0].baseline_ms


def test_early_resume_falls_back_then_probes_with_backoff():
    prompts = load_prompts(_report([(1, 0)] * 40, _BAD_TWO))
    prompt = prompts[0]
    policy = EarlyResumePolicy(
        observe_cycles=4,
        probe_cycles=2,
        reprobe_gap_tokens=8,
        resync_ms_per_token=0.5,
    )
    sim = simulate(prompt, policy, 400)
    assert policy.state == "off"
    assert policy.current_gap > 8  # backoff grew after failed probes
    # Fallback tokens run at true ordinary cost, so the total stays close to
    # ordinary decode despite observation, probes and resync.
    assert sim.total_ms < 400 * prompt.baseline_ms * 1.35


def test_early_resume_stays_on_for_good_streams():
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO))
    policy = EarlyResumePolicy(observe_cycles=4, probe_cycles=2)
    sim = simulate(prompts[0], policy, 300)
    assert policy.state == "on"
    assert sim.total_ms < 300 * prompts[0].baseline_ms


def test_ema_cost_policy_tracks_streams():
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO))
    policy = EmaCostPolicy()
    sim = simulate(prompts[0], policy, 300)
    assert sim.total_ms < 300 * prompts[0].baseline_ms
    bad = load_prompts(_report([(1, 0)] * 40, _BAD_TWO))[0]
    policy = EmaCostPolicy(alpha=0.3, probe_interval=64)
    sim = simulate(bad, policy, 300)
    # Shadow fallback costs more than ordinary decode but must stay bounded.
    assert sim.total_ms < 300 * bad.shadow_cost_ms() * 1.6


def test_pld_policy_speculates_on_repetitive_output():
    output_ids = [10, 11, 12, 13, 14] * 8  # strongly periodic
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO, output_ids=output_ids))
    prompt = prompts[0]
    policy = PLDHybridPolicy(min_ngram=3, draft_len=3)
    sim = simulate(prompt, policy, 40)
    assert sim.cycles < 40  # lookup drafts emitted multiple tokens per cycle
    assert sim.total_ms < 40 * prompt.baseline_ms
    # Verification-only cycle cost: cheaper than the MTP cycle at same shape.
    assert prompt.pld_cycle_cost_ms(2, 2) < prompt.cycle_cost_ms(2, 2)
    # Four-row verification extrapolates beyond the recorded graphs.
    tv2 = prompt.phase_ms["TargetVerify2"]
    tv3 = prompt.phase_ms["TargetVerify3"]
    assert prompt.pld_cycle_cost_ms(3, 3) == pytest.approx(
        2 * tv3 - tv2 + prompt.phase_ms["DecisionSync"] + prompt.host_ms
    )


def test_pld_policy_stays_ordinary_without_repetition():
    prompts = load_prompts(_report(_GOOD_ONE, _GOOD_TWO))  # unique token ids
    prompt = prompts[0]
    policy = PLDHybridPolicy(min_ngram=3, draft_len=3)
    sim = simulate(prompt, policy, 40)
    assert sim.cycles == 40
    assert sim.total_ms == pytest.approx(40 * prompt.baseline_ms)


def test_main_end_to_end(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(_GOOD_ONE, _GOOD_TWO)))
    json_path = tmp_path / "out.json"
    assert main([str(report_path), "-o", str(json_path)]) == 0
    result = json.loads(json_path.read_text())
    assert "fixture" in result["prompts"]
    policies = result["prompts"]["fixture"]["policies"]
    oracle_fixed = policies["oracle-fixed"]["speedup"]
    assert oracle_fixed >= policies["fixed1"]["speedup"]
    assert oracle_fixed >= policies["fixed2"]["speedup"]
    assert oracle_fixed >= 1.0
    assert policies["oracle-cycle"]["speedup"] >= oracle_fixed - 1e-9
    for row in result["validation"]:
        assert abs(row["error_pct"]) < 0.1
    out = capsys.readouterr().out
    assert "cost-model validation" in out
    assert "geomean" in out
