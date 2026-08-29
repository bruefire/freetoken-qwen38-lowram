from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from freetoken.profiling import count_decode_forwards, summarize_decode_profile_events
from freetoken.utils import enable_record_function_ranges


class ServerDecodeProfiler:
    """Collect bounded torch.profiler windows from the live scheduler process."""

    def __init__(
        self,
        interval: int,
        log: Callable[[str], None],
        profile_factory: Callable[[], Any] | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("decode profile interval must be positive")
        self.interval = interval
        self.log = log
        self._profile_factory = profile_factory or self._torch_profile
        self._profiler: Any | None = None
        self._completed_decode_steps = 0
        self._window = 0

    @staticmethod
    def _torch_profile():
        import torch

        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )

    @property
    def active(self) -> bool:
        return self._profiler is not None

    def start_if_needed(self) -> None:
        if self._profiler is not None:
            return
        enable_record_function_ranges(True)
        try:
            self._profiler = self._profile_factory()
            self._profiler.start()
        except Exception:
            self._profiler = None
            enable_record_function_ranges(False)
            raise

    def complete_decode_step(self) -> None:
        if self._profiler is None:
            return
        self._completed_decode_steps += 1
        if self._completed_decode_steps >= self.interval:
            self.flush(partial=False)

    def flush(self, *, partial: bool) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        self._profiler = None
        try:
            profiler.stop()
        finally:
            enable_record_function_ranges(False)

        events = list(profiler.events())
        observed = count_decode_forwards(events)
        denominator = observed or self._completed_decode_steps
        self._window += 1
        payload = {
            "version": 1,
            "window": self._window,
            "configured_interval": self.interval,
            "partial": partial,
            "completed_decode_drains": self._completed_decode_steps,
            "observed_decode_forwards": observed,
            "normalization_decode_steps": denominator,
            "ranges_are_nested": True,
            "ranges": summarize_decode_profile_events(events, denominator),
        }
        self.log("FT.DecodeProfile " + json.dumps(payload, separators=(",", ":")))
        self._completed_decode_steps = 0

    def close(self) -> None:
        self.flush(partial=True)
