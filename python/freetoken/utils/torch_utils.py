from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Detailed decode ranges are opt-in because a Qwen4-Exp token crosses hundreds of
# small regions. ``@nvtx_annotate`` has its own FREETOKEN_NVTX gate; this switch gates
# the finer-grained ranges added for decode profiling.
_DECODE_PROFILE_RANGES = os.getenv("FREETOKEN_DECODE_PROFILE_RANGES", "") == "1"
_RECORD_FUNCTION_RANGES = False


class _NoopRange:
    """Reusable disabled range: no generator creation and no runtime flag check."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info) -> None:
        return None


_NOOP_RANGE = _NoopRange()


def _noop_decode_profile_range(name: str) -> _NoopRange:
    return _NOOP_RANGE


def enable_record_function_ranges(enabled: bool = True) -> None:
    """Mirror NVTX ranges into torch.profiler while a decode trace is active.

    ``record_function`` is deliberately toggled only around the short torch-profiler
    windows.  Leaving it on during the wall-time pass would make the measurement pay
    profiler annotation overhead even when no profiler is collecting it.
    """
    global _RECORD_FUNCTION_RANGES
    _RECORD_FUNCTION_RANGES = enabled


@contextmanager
def _active_decode_profile_range(name: str):
    """NVTX/torch.profiler range used only in an explicitly enabled process."""
    import torch.cuda.nvtx as nvtx

    with nvtx.range(name):
        if _RECORD_FUNCTION_RANGES:
            import torch

            with torch.profiler.record_function(name):
                yield
        else:
            yield


# Instrumented modules import this symbol once.  Select the disabled implementation at
# import/configuration time so an ordinary server never performs a per-range flag lookup.
decode_profile_range = (
    _active_decode_profile_range if _DECODE_PROFILE_RANGES else _noop_decode_profile_range
)


def configure_decode_profile_ranges(enabled: bool = True) -> None:
    """Select the range implementation before instrumented hot-path modules import it."""
    global _DECODE_PROFILE_RANGES, decode_profile_range
    _DECODE_PROFILE_RANGES = enabled
    decode_profile_range = (
        _active_decode_profile_range if enabled else _noop_decode_profile_range
    )


def decode_profile_ranges_enabled() -> bool:
    return _DECODE_PROFILE_RANGES


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    # Emitting an NVTX range per decorated call costs a Python call plus a
    # string format on the decode hot path; with every layer annotated that is
    # hundreds of ranges per token. Off by default outside a profiler run.
    if os.environ.get("FREETOKEN_NVTX", "").strip().lower() not in {"1", "true", "yes", "on"}:
        def passthrough(fn):
            return fn

        return passthrough

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            if _DECODE_PROFILE_RANGES and _RECORD_FUNCTION_RANGES:
                import torch

                with nvtx.range(display_name), torch.profiler.record_function(display_name):
                    return fn(self, *args, **kwargs)
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
