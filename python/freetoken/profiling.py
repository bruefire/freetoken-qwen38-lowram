from __future__ import annotations

from collections.abc import Iterable
from typing import Any

PROFILE_PREFIXES = ("FT.", "Layer_", "QSA", "Embedding", "LMHead", "Sampler")


def _is_cpu_event(event: Any) -> bool:
    """Drop NVTX's GPU mirror of a CPU record_function range."""
    device_type = getattr(event, "device_type", None)
    return device_type is None or getattr(device_type, "name", "CPU") == "CPU"


def _is_decode_descendant(event: Any) -> bool:
    current = event
    while current is not None:
        key = getattr(current, "key", None)
        if key == "FT.DecodeStep" or (
            isinstance(key, str) and key.startswith("FT.Scheduler.Decode")
        ):
            return True
        current = getattr(current, "cpu_parent", None)
    return False


def count_decode_forwards(events: Iterable[Any]) -> int:
    return sum(
        1
        for event in events
        if _is_cpu_event(event) and getattr(event, "key", None) == "FT.DecodeStep"
    )


def summarize_decode_profile_events(
    events: Iterable[Any], decode_steps: int
) -> list[dict[str, float | int | str]]:
    """Aggregate named ranges whose CPU ancestry is a decode scheduler/engine step."""
    aggregated: dict[str, dict[str, float | int | str]] = {}
    for event in events:
        if not _is_cpu_event(event):
            continue
        key = str(getattr(event, "key", ""))
        if not key.startswith(PROFILE_PREFIXES) or not _is_decode_descendant(event):
            continue
        row = aggregated.setdefault(
            key,
            {"name": key, "calls": 0, "cpu_total_us": 0.0, "device_total_us": 0.0},
        )
        row["calls"] = int(row["calls"]) + 1
        row["cpu_total_us"] = float(row["cpu_total_us"]) + float(
            getattr(event, "cpu_time_total", 0.0)
        )
        row["device_total_us"] = float(row["device_total_us"]) + float(
            getattr(event, "device_time_total", 0.0)
        )

    denom = max(1, decode_steps)
    result = []
    for row in aggregated.values():
        calls = int(row["calls"])
        row["calls_per_decode_step"] = calls / denom
        row["cpu_ms_per_decode_step"] = float(row["cpu_total_us"]) / 1000.0 / denom
        row["device_ms_per_decode_step"] = (
            float(row["device_total_us"]) / 1000.0 / denom
        )
        row["cpu_ms_per_call"] = float(row["cpu_total_us"]) / 1000.0 / max(1, calls)
        row["device_ms_per_call"] = (
            float(row["device_total_us"]) / 1000.0 / max(1, calls)
        )
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            -float(row["device_ms_per_decode_step"]),
            -float(row["cpu_ms_per_decode_step"]),
            str(row["name"]),
        ),
    )
