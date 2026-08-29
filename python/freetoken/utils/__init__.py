from . import torch_utils as _torch_utils
from .arch import (
    is_arch_supported,
    is_sm90_family,
    is_sm90_supported,
    is_sm100_family,
    is_sm100_supported,
)
from .hf import (
    cached_load_hf_config,
    download_hf_weight,
    load_eos_token_ids,
    load_generation_sampling,
    load_tokenizer,
    load_toolcall_anchor_id,
)
from .logger import init_logger
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even, mem_GB
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import (
    decode_profile_range,
    decode_profile_ranges_enabled,
    enable_record_function_ranges,
    nvtx_annotate,
    torch_dtype,
)


def configure_decode_profile_ranges(enabled: bool = True) -> None:
    """Configure ranges, then refresh this package's export for later imports."""
    global decode_profile_range
    _torch_utils.configure_decode_profile_ranges(enabled)
    decode_profile_range = _torch_utils.decode_profile_range

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_eos_token_ids",
    "load_generation_sampling",
    "load_tokenizer",
    "load_toolcall_anchor_id",
    "init_logger",
    "is_arch_supported",
    "is_sm90_family",
    "is_sm90_supported",
    "is_sm100_family",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "mem_GB",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "decode_profile_range",
    "decode_profile_ranges_enabled",
    "configure_decode_profile_ranges",
    "enable_record_function_ranges",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
