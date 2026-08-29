from __future__ import annotations

import bisect
import ctypes
import json
import math
import mmap
import os
import resource
from dataclasses import replace
from typing import TYPE_CHECKING

import safetensors
import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.utils import (
    decode_profile_range,
    download_hf_weight,
    init_logger,
    nvtx_annotate,
)

from .attention import Qwen4ExpAttention

logger = init_logger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off"}
_PLE_MMAP_ADVICES = {"normal", "random", "random-willneed"}
_PAGE_SIZE = mmap.PAGESIZE
_LIBC_MADVISE = None


def _mtp_moe_cache_size() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE", "64").strip()
    try:
        size = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE must be a positive integer"
        ) from error
    if size <= 0:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE must be a positive integer"
        )
    return size


def _mtp_max_drafts() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1 or 2"
        ) from error
    if value not in (1, 2):
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1 or 2")
    return value


def _hyper_connection_fused_enabled() -> bool:
    return (
        os.getenv("FREETOKEN_QWEN_HYPER_CONNECTION_FUSED", "1").strip().lower()
        not in _FALSE_VALUES
    )

def _hyper_connection_projection_fused_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN_HC_PROJECTION_FUSED", "1").strip().lower() not in _FALSE_VALUES


def _shared_expert_gate_fused_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN_SHARED_GATE_FUSED", "1").strip().lower() not in _FALSE_VALUES


def _ple_mmap_advice() -> str:
    raw = os.getenv("FREETOKEN_QWEN4_PLE_MMAP_ADVICE", "").strip().lower()
    if not raw:
        return (
            "random-willneed"
            if hasattr(mmap, "MADV_RANDOM") and hasattr(mmap, "MADV_WILLNEED")
            else "normal"
        )
    if raw not in _PLE_MMAP_ADVICES:
        choices = ", ".join(sorted(_PLE_MMAP_ADVICES))
        raise ValueError(
            f"FREETOKEN_QWEN4_PLE_MMAP_ADVICE must be one of {choices}, got {raw!r}"
        )
    if raw != "normal" and (
        not hasattr(mmap, "MADV_RANDOM") or not hasattr(mmap, "MADV_WILLNEED")
    ):
        raise ValueError(f"PLE mmap advice {raw!r} is unavailable on this platform")
    return raw


def _madvise(address: int, length: int, advice: int) -> None:
    global _LIBC_MADVISE
    if _LIBC_MADVISE is None:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.madvise
        function.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        function.restype = ctypes.c_int
        _LIBC_MADVISE = function
    if _LIBC_MADVISE(address, length, advice):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _madvise_tensor_ranges(tensors: list[torch.Tensor], advice: int) -> None:
    for tensor in tensors:
        address = tensor.data_ptr()
        start = address - address % _PAGE_SIZE
        end_address = address + tensor.numel() * tensor.element_size()
        end = (end_address + _PAGE_SIZE - 1) // _PAGE_SIZE * _PAGE_SIZE
        _madvise(start, end - start, advice)


def _prefetch_tensor_rows(rows: list[torch.Tensor]) -> None:
    pages = set()
    for row in rows:
        address = row.data_ptr()
        first_page = address - address % _PAGE_SIZE
        last_address = address + row.numel() * row.element_size() - 1
        last_page = last_address - last_address % _PAGE_SIZE
        pages.add(first_page)
        pages.add(last_page)
    for page in pages:
        _madvise(page, _PAGE_SIZE, mmap.MADV_WILLNEED)


def _prefetch_tensor_indices(tensor: torch.Tensor, indices: list[int]) -> None:
    base_address = tensor.data_ptr()
    row_bytes = tensor.shape[1] * tensor.element_size()
    pages = set()
    for index in indices:
        address = base_address + index * row_bytes
        first_page = address - address % _PAGE_SIZE
        last_address = address + row_bytes - 1
        last_page = last_address - last_address % _PAGE_SIZE
        pages.add(first_page)
        pages.add(last_page)
    for page in pages:
        _madvise(page, _PAGE_SIZE, mmap.MADV_WILLNEED)


if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

    from .args import Qwen4ExpArgs


class _GroupedRMSNorm(BaseOP):
    def __init__(self, size: int, group_size: int, eps: float, *, save_stats: bool = True):
        if size % group_size:
            raise ValueError(f"RMSNorm size {size} is not divisible by group size {group_size}")
        self.weight = torch.empty(size)
        self.group_size = group_size
        self.eps = eps
        self.save_stats = save_stats

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            eps=self.eps,
            group_size=self.group_size,
            is_rms_norm=True,
            weight_plus_one=True,
            save_stats=self.save_stats,
        )


class _GatedRMSNorm(BaseOP):
    def __init__(
        self, size: int, eps: float, activation: str, *, save_stats: bool = True
    ):
        self.weight = torch.empty(size)
        self.eps = eps
        self.activation = activation
        self.save_stats = save_stats

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.eps,
            is_rms_norm=True,
            norm_before_gate=True,
            activation=self.activation,
            save_stats=self.save_stats,
        )


class _GatedResidual(BaseOP):
    def __init__(self, config: ModelConfig, combine: bool = True):
        args: Qwen4ExpArgs = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.lowrank = args.hc_lowrank
        self._fused = _hyper_connection_fused_enabled()
        self._projection_fused = _hyper_connection_projection_fused_enabled()
        hc_size = self.hc_count * self.hidden_size
        self.hc_norm = _GroupedRMSNorm(
            hc_size,
            self.hidden_size,
            config.rms_norm_eps,
            save_stats=not self._fused,
        )
        self.input_mix_weight_up = LinearReplicated(self.lowrank, hc_size, has_bias=False)
        if combine:
            self._inject_padding = (-(self.lowrank + self.hc_count)) % 16
            self.input_mix_weight_down_block_inject = LinearReplicated(
                hc_size,
                self.lowrank + self.hc_count + self._inject_padding,
                has_bias=False,
            )
        else:
            self._inject_padding = 0
            self.input_mix_weight_down = LinearReplicated(hc_size, self.lowrank, has_bias=False)

    def _down(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not hasattr(self, "input_mix_weight_down_block_inject"):
            return self.input_mix_weight_down.forward(normalized), None
        weight = self.input_mix_weight_down_block_inject.weight
        if self._projection_fused:
            down = self.input_mix_weight_down_block_inject.forward(normalized)
            lowrank = down[:, : self.lowrank]
            inject = down[:, self.lowrank : self.lowrank + self.hc_count]
            if not lowrank.is_contiguous():
                lowrank = lowrank.contiguous()
            if not inject.is_contiguous():
                inject = inject.contiguous()
            return lowrank, inject
        return (
            F.linear(normalized, weight[: self.lowrank]),
            F.linear(normalized, weight[self.lowrank : self.lowrank + self.hc_count]),
        )

    def forward(self, hyper_input: torch.Tensor):
        normalized = self.hc_norm.forward(hyper_input)
        down, injection_logits = self._down(normalized)
        if self._fused and hyper_input.is_cuda:
            from freetoken.kernel.triton.qwen_hyper_connection import (
                hyper_inject,
                hyper_mix_reduce,
                hyper_silu_div,
            )

            mix = hyper_silu_div(down, self.hc_count)
            mixed = hyper_mix_reduce(
                self.input_mix_weight_up.forward(mix),
                normalized,
                self.hc_count,
                self.hidden_size,
            )
            if injection_logits is None:
                return mixed
            inject = hyper_inject(injection_logits, self.hc_count)
            return mixed, hyper_input, inject

        mix = F.silu(down / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up.forward(mix))
        mix = mix.view(-1, self.hc_count, self.hidden_size)
        mixed = (mix * normalized.view(-1, self.hc_count, self.hidden_size)).mean(dim=1)
        if injection_logits is None:
            return mixed
        inject = 2 * torch.sigmoid(injection_logits / self.hc_count)
        return mixed, hyper_input, inject


def _weighted_residual_combine(
    residual: torch.Tensor,
    mixed: torch.Tensor,
    weights: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    fused: bool | None = None,
) -> torch.Tensor:
    if fused is None:
        fused = _hyper_connection_fused_enabled()
    if fused and residual.is_cuda:
        from freetoken.kernel.triton.qwen_hyper_connection import (
            hyper_weighted_residual,
        )

        return hyper_weighted_residual(
            residual, mixed, weights, hc_count, hidden_size
        )
    return residual + (mixed.unsqueeze(1) * weights.unsqueeze(-1)).flatten(1)


class _SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig):
        width = config.shared_expert_intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [width, width], has_bias=False
        )
        self.down_proj = LinearRowParallel(width, config.hidden_size, has_bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class _SparseMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=bool(config.norm_topk_prob),
            weight_format="fp8_block",
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self._shared_gate_fused = _shared_expert_gate_fused_enabled()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with decode_profile_range("FT.MoE"):
            return self._forward_impl(hidden)

    def _forward_impl(self, hidden: torch.Tensor) -> torch.Tensor:
        with decode_profile_range("FT.MoE.RouterProjection"):
            router_logits = self.gate.forward(hidden)
        with decode_profile_range("FT.MoE.SharedExpert"):
            shared = self.shared_expert.forward(hidden)
            if self._shared_gate_fused and hidden.is_cuda:
                from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

                shared_gate = shared_gate_sigmoid(hidden, self.shared_expert_gate.weight.view(-1))
            else:
                shared_gate = torch.sigmoid(self.shared_expert_gate.forward(hidden))
                shared *= shared_gate
        with decode_profile_range("FT.MoE.RoutedExperts"):
            routed = self.experts.forward(hidden_states=hidden, router_logits=router_logits)
        if self._shared_gate_fused and hidden.is_cuda:
            from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add

            return shared_gate_mul_add(routed, shared, shared_gate)
        return routed + shared


def _shift_right_ignore_eos(tokens: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    if shift == 0:
        return tokens
    positions = torch.arange(tokens.numel(), dtype=torch.long)
    eos_positions = torch.where(tokens == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]])
    segment_start = previous_eos + 1
    source_positions = positions - shift
    shifted = tokens[source_positions.clamp_min(0)]
    valid = (positions - segment_start >= shift) & (source_positions >= 0)
    return torch.where(valid, shifted, tokens.new_full((), eos_token_id))


def build_ngram_ids(
    tokens: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    tokens = tokens.to(dtype=torch.long, device="cpu")
    shifted = [
        _shift_right_ignore_eos(tokens, shift, eos_token_id) for shift in range(ngram_size)
    ]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        stop = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        sizes = vocab_sizes[start:stop]
        heads = torch.remainder(mixed.unsqueeze(-1), sizes)
        blocks.append(heads + offsets[start:stop])
    return torch.cat(blocks, dim=-1)


def _tokens_for_ngram_forward(
    req,
    current_ids: torch.Tensor,
    *,
    start: int = 0,
) -> torch.Tensor:
    """Return request history through ``device_len`` without mutating the request.

    Under overlap scheduling the next decode starts before the previous sampled token is
    appended to ``req.input_ids``.  That token is already present in ``batch.input_ids``;
    splice only the missing suffix from there so PLE sees the same history in overlap and
    non-overlap modes. ``start`` lets decode retain only the short suffix needed by the
    n-gram hash instead of copying and re-hashing the complete request on every token.
    """
    if not 0 <= start <= req.device_len:
        raise ValueError(f"Qwen4-Exp PLE history start {start} is outside [0, {req.device_len}]")
    host_len = min(req.input_ids.numel(), req.device_len)
    host_start = min(start, host_len)
    tokens = req.input_ids[host_start:host_len].to(dtype=torch.long, device="cpu")
    missing = req.device_len - host_len
    if missing:
        if missing > current_ids.numel():
            raise RuntimeError(
                f"Qwen4-Exp PLE history is missing {missing} tokens, "
                f"but the forward only carries {current_ids.numel()}"
            )
        inflight = current_ids[-missing:]
        if start > host_len:
            inflight = inflight[start - host_len :]
        tokens = torch.cat([tokens, inflight.to(dtype=torch.long, device="cpu")])
    return tokens


def _preload_ple_enabled() -> bool:
    return os.getenv("FREETOKEN_QWEN4_PLE_PRELOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ple_row_cache_bytes() -> int:
    """Configured total direct-mapped PLE row-cache budget."""
    raw = os.getenv("FREETOKEN_QWEN4_PLE_CACHE_GIB", "").strip()
    if not raw:
        return 0
    try:
        gib = float(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_PLE_CACHE_GIB must be a non-negative number"
        ) from error
    if not math.isfinite(gib) or gib < 0:
        raise ValueError(
            "FREETOKEN_QWEN4_PLE_CACHE_GIB must be a non-negative finite number"
        )
    return int(gib * (1 << 30))


class _HostNGramEmbedding(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.eos_token_id = args.eos_token_id
        self.embedding_dim = args.ple_embed_dim
        self.split_ngram_parts = args.split_ngram_parts
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.head_dim = self.embedding_dim // self.ngram_heads
        self.layer_multipliers = torch.empty(args.ngram_size, dtype=torch.long)
        self.ngram_heads_vocab_sizes = torch.empty(self.ngram_heads, dtype=torch.long)
        self.ngram_heads_offsets = torch.empty(self.ngram_heads, dtype=torch.long)
        self._handles = []
        self._mmap_paths: tuple[str, ...] = ()
        self._shards: list[torch.Tensor] = []
        self._shard_ends = torch.empty(0, dtype=torch.long)
        self._shard_starts = torch.empty(0, dtype=torch.long)
        self._shard_ends_list: tuple[int, ...] = ()
        self._shard_starts_list: tuple[int, ...] = ()
        self._scale = torch.tensor(1.0, dtype=torch.bfloat16)
        self._host_constants: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._dummy = False
        self._graph_output: torch.Tensor | None = None
        self._device_scale: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self._row_cache: torch.Tensor | None = None
        self._row_cache_tags: torch.Tensor | None = None
        self._mmap_advice = "normal"
        self._mmap_prefetch = False
        # The environment value is a model-wide hard ceiling.  Split it evenly
        # if a checkpoint ever places PLE on more than one layer.
        self._row_cache_budget_bytes = _ple_row_cache_bytes() // max(
            1, len(args.ple_layer_ids)
        )
        self._cache_hits = 0
        self._cache_misses = 0
        self._decode_source_rows = 0
        self._prefill_source_rows = 0
        self._decode_major_faults = 0
        self._prefill_major_faults = 0
        self._collect_ple_stats = bool(
            self._row_cache_budget_bytes
            or os.getenv("FREETOKEN_DECODE_PROFILE_RANGES", "").strip()
        )

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        if dummy:
            self._dummy = True
            return
        folder = download_hf_weight(model_path)
        index_path = os.path.join(folder, "model.safetensors.index.json")
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        prefix = (
            f"model.language_model.layers.{self.layer_id}.ple.ple_embedding."
            "ngram_embedding"
        )
        shard_count = len([key for key in weight_map if key.startswith(prefix + ".shard_")])
        if shard_count != self.split_ngram_parts:
            raise RuntimeError(
                f"Qwen4-Exp PLE has {shard_count} shards, expected {self.split_ngram_parts}"
            )
        shard_keys = [f"{prefix}.shard_{shard_id}.weight" for shard_id in range(shard_count)]
        if not shard_keys or any(key not in weight_map for key in shard_keys):
            raise RuntimeError(f"Incomplete Qwen4-Exp PLE shards under {prefix}")

        handles = {}
        shards = []
        for key in shard_keys:
            filename = weight_map[key]
            handle = handles.get(filename)
            if handle is None:
                handle = safetensors.safe_open(
                    os.path.join(folder, filename), framework="pt", device="cpu"
                ).__enter__()
                handles[filename] = handle
            shard = handle.get_tensor(key)
            if shard.dtype != torch.float8_e4m3fn or shard.shape[1] != self.head_dim:
                raise RuntimeError(f"Unexpected PLE shard {key}: {shard.dtype} {tuple(shard.shape)}")
            shards.append(shard.view(torch.uint8))
        scale_key = prefix + ".weight_scale"
        scale_handle = handles.get(weight_map[scale_key])
        if scale_handle is None:
            scale_handle = safetensors.safe_open(
                os.path.join(folder, weight_map[scale_key]), framework="pt", device="cpu"
            ).__enter__()
            handles[weight_map[scale_key]] = scale_handle

        preload = _preload_ple_enabled()
        if preload:
            total_gib = sum(shard.numel() * shard.element_size() for shard in shards) / (1 << 30)
            logger.info_rank0(
                f"Preloading {total_gib:.2f} GiB of Qwen4-Exp PLE into host RAM "
                "(FREETOKEN_QWEN4_PLE_PRELOAD=1)"
            )
            shards = [shard.clone() for shard in shards]
            logger.info_rank0("Qwen4-Exp PLE preload complete")
        self._handles = [] if preload else list(handles.values())
        self._mmap_paths = (
            ()
            if preload
            else tuple(os.path.join(folder, filename) for filename in handles)
        )
        self._shards = shards
        self._shard_ends = torch.tensor([shard.shape[0] for shard in shards]).cumsum(0)
        self._shard_starts = torch.cat(
            [self._shard_ends.new_zeros(1), self._shard_ends[:-1]]
        )
        self._shard_ends_list = tuple(int(value) for value in self._shard_ends)
        self._shard_starts_list = tuple(int(value) for value in self._shard_starts)
        self._scale = scale_handle.get_tensor(scale_key).reshape(()).clone()
        if preload:
            for handle in handles.values():
                handle.__exit__(None, None, None)
        self._host_constants = (
            self.layer_multipliers.cpu(),
            self.ngram_heads_vocab_sizes.cpu(),
            self.ngram_heads_offsets.cpu(),
        )
        expected_rows = int(self._host_constants[1][-1] + self._host_constants[2][-1])
        if int(self._shard_ends[-1]) < expected_rows:
            raise RuntimeError(
                f"PLE table has {int(self._shard_ends[-1])} rows, needs {expected_rows}"
            )
        if not preload:
            self.set_mmap_advice(_ple_mmap_advice())
        if self._row_cache_budget_bytes and not preload:
            self._init_row_cache(self._row_cache_budget_bytes)
        elif self._row_cache_budget_bytes:
            logger.info_rank0(
                "Ignoring FREETOKEN_QWEN4_PLE_CACHE_GIB because the full PLE "
                "table is already preloaded"
            )

    def set_mmap_advice(self, advice: str) -> None:
        if advice not in _PLE_MMAP_ADVICES:
            raise ValueError(f"unknown PLE mmap advice {advice!r}")
        if not self._shards or not self._handles:
            self._mmap_advice = "preloaded" if self._shards else advice
            self._mmap_prefetch = False
            return
        if advice == "normal" and not hasattr(mmap, "MADV_NORMAL"):
            self._mmap_advice = advice
            self._mmap_prefetch = False
            return
        if advice != "normal" and (
            not hasattr(mmap, "MADV_RANDOM") or not hasattr(mmap, "MADV_WILLNEED")
        ):
            raise ValueError(f"PLE mmap advice {advice!r} is unavailable")
        madvise_value = mmap.MADV_RANDOM if advice != "normal" else mmap.MADV_NORMAL
        _madvise_tensor_ranges(self._shards, madvise_value)
        self._mmap_advice = advice
        self._mmap_prefetch = advice == "random-willneed"
        logger.info_rank0(
            f"Qwen4-Exp PLE layer {self.layer_id} mmap advice: {advice}"
        )

    def prepare_cold_mmap_replay(self, advice: str) -> None:
        if self._handles:
            if not hasattr(mmap, "MADV_DONTNEED"):
                raise RuntimeError("cold PLE replay needs MADV_DONTNEED")
            if not hasattr(os, "posix_fadvise") or not hasattr(
                os, "POSIX_FADV_DONTNEED"
            ):
                raise RuntimeError("cold PLE replay needs POSIX_FADV_DONTNEED")
            _madvise_tensor_ranges(self._shards, mmap.MADV_DONTNEED)
            for path in self._mmap_paths:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        self.set_mmap_advice(advice)

    def _init_row_cache(self, budget_bytes: int) -> None:
        # One int64 tag accompanies each exact uint8/FP8 row.  Direct mapping has
        # constant-time metadata and a predictable hard memory ceiling; at GiB-scale
        # capacities, collisions within a decode working set are rare.
        row_bytes = self.head_dim + 8
        capacity = min(
            budget_bytes // row_bytes,
            self._shard_ends_list[-1] if self._shard_ends_list else 0,
        )
        if capacity < 1:
            raise ValueError(
                "FREETOKEN_QWEN4_PLE_CACHE_GIB is too small for one PLE row and tag"
            )
        self._row_cache = torch.empty(capacity, self.head_dim, dtype=torch.uint8)
        # Zero is the untouched sentinel; stored tags use global_row_id + 1.
        self._row_cache_tags = torch.zeros(capacity, dtype=torch.int64)
        allocated = capacity * row_bytes
        logger.info_rank0(
            f"Qwen4-Exp PLE layer {self.layer_id} row cache: "
            f"{capacity:,} rows, {allocated / (1 << 30):.2f} GiB "
            "(direct-mapped, opt-in)"
        )

    def _ensure_output_buffer(self, rows: int) -> torch.Tensor:
        buffer = self._output_buffer
        if buffer is None or buffer.shape[0] < rows:
            buffer = torch.empty(
                rows,
                self.head_dim,
                dtype=torch.uint8,
                pin_memory=torch.cuda.is_available(),
            )
            self._output_buffer = buffer
        return buffer[:rows]

    def reset_cache_stats(self) -> None:
        self._cache_hits = 0
        self._cache_misses = 0
        self._decode_source_rows = 0
        self._prefill_source_rows = 0
        self._decode_major_faults = 0
        self._prefill_major_faults = 0

    def cache_stats(self) -> dict[str, int | float | bool | None]:
        cache_rows = self._row_cache.shape[0] if self._row_cache is not None else 0
        attempts = self._cache_hits + self._cache_misses
        return {
            "layer_id": self.layer_id,
            "mmap_advice": self._mmap_advice,
            "cache_enabled": self._row_cache is not None,
            "cache_capacity_rows": cache_rows,
            "cache_capacity_bytes": cache_rows * (self.head_dim + 8),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / attempts if attempts else None,
            "decode_source_rows": self._decode_source_rows,
            "prefill_source_rows": self._prefill_source_rows,
            "decode_major_faults": self._decode_major_faults,
            "prefill_major_faults": self._prefill_major_faults,
        }

    def _current_ngram_ids(self) -> torch.Tensor:
        if self._host_constants is None:
            raise RuntimeError("Qwen4-Exp PLE host weights are not loaded")
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        multipliers, vocab_sizes, offsets = self._host_constants
        current_parts: list[torch.Tensor] | None = None
        with decode_profile_range("FT.PLE.InputIdsHostReady"):
            if batch.is_decode:
                host_input_ids = getattr(batch, "host_input_ids", None)
                if host_input_ids is None:
                    raise RuntimeError(
                        "Qwen4-Exp decode requires scheduler-staged host input IDs"
                    )
                # The event belongs to the sampler D2H from the preceding
                # forward.  It is the sole autoregressive dependency; no new
                # device copy is issued here.
                if host_input_ids.ready is not None:
                    host_input_ids.ready.synchronize()
                current_parts = host_input_ids.parts
        if current_parts is not None and len(current_parts) != len(reqs):
            raise RuntimeError(
                f"Qwen4-Exp PLE received {len(current_parts)} host token parts "
                f"for {len(reqs)} padded requests"
            )
        with decode_profile_range("FT.PLE.NGramBuildCPU"):
            pieces = []
            token_offset = 0
            for req_index, req in enumerate(reqs):
                length = req.extend_len
                history_start = max(0, req.cached_len - (self.ngram_size - 1))
                if current_parts is not None:
                    current_ids = current_parts[req_index]
                    tokens = _tokens_for_ngram_forward(
                        req,
                        current_ids,
                        start=history_start,
                    )
                    token_offset += length
                else:
                    tokens = req.input_ids[history_start : req.device_len]
                all_ids = build_ngram_ids(
                    tokens,
                    ngram_size=self.ngram_size,
                    heads_per_ngram=self.heads_per_ngram,
                    eos_token_id=self.eos_token_id,
                    multipliers=multipliers,
                    vocab_sizes=vocab_sizes,
                    offsets=offsets,
                )
                pieces.append(
                    all_ids[
                        req.cached_len - history_start : req.device_len - history_start
                    ]
                )
            result = torch.cat(pieces, dim=0)
        current_count = (
            sum(part.numel() for part in current_parts)
            if current_parts is not None
            else 0
        )
        if current_parts is not None and token_offset != current_count:
            raise RuntimeError(
                f"Qwen4-Exp PLE consumed {token_offset} decode tokens, "
                f"but the host batch carries {current_count}"
            )
        if result.shape[0] != batch.input_ids.numel():
            raise RuntimeError(
                f"PLE token count {result.shape[0]} does not match batch {batch.input_ids.numel()}"
            )
        return result

    def _copy_decode_rows(
        self, ngram_ids: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Copy a small decode gather with no per-shard tensor temporaries."""
        cache = self._row_cache
        tags = self._row_cache_tags
        if cache is not None and tags is None:
            raise RuntimeError("Qwen4-Exp PLE row cache tags are not initialized")
        capacity = cache.shape[0] if cache is not None else 0
        global_ids = ngram_ids.tolist()
        if self._mmap_prefetch:
            source_rows = []
            for global_id in global_ids:
                shard_id = bisect.bisect_right(self._shard_ends_list, global_id)
                if shard_id >= len(self._shards):
                    raise IndexError(f"PLE row {global_id} is outside the loaded table")
                if cache is not None:
                    slot = global_id % capacity
                    if int(tags[slot]) == global_id + 1:
                        continue
                local_id = global_id - self._shard_starts_list[shard_id]
                source_rows.append(self._shards[shard_id][local_id])
            if source_rows:
                _prefetch_tensor_rows(source_rows)

        for position, global_id in enumerate(global_ids):
            shard_id = bisect.bisect_right(self._shard_ends_list, global_id)
            if shard_id >= len(self._shards):
                raise IndexError(f"PLE row {global_id} is outside the loaded table")
            local_id = global_id - self._shard_starts_list[shard_id]
            if cache is None:
                output[position].copy_(self._shards[shard_id][local_id])
                self._decode_source_rows += 1
                continue

            slot = global_id % capacity
            encoded_tag = global_id + 1
            if int(tags[slot]) == encoded_tag:
                output[position].copy_(cache[slot])
                self._cache_hits += 1
            else:
                # Publish the tag only after the exact checkpoint bytes are cached.
                cache[slot].copy_(self._shards[shard_id][local_id])
                tags[slot] = encoded_tag
                output[position].copy_(cache[slot])
                self._cache_misses += 1
                self._decode_source_rows += 1

    def _copy_prefill_rows(
        self, ngram_ids: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Retain grouped tensor gathers for large prefill batches."""
        grouped: dict[int, tuple[list[int], list[int]]] = {}
        for position, global_id in enumerate(ngram_ids.tolist()):
            shard_id = bisect.bisect_right(self._shard_ends_list, global_id)
            if shard_id >= len(self._shards):
                raise IndexError(f"PLE row {global_id} is outside the loaded table")
            positions, local_ids = grouped.setdefault(shard_id, ([], []))
            positions.append(position)
            local_ids.append(global_id - self._shard_starts_list[shard_id])
        if self._mmap_prefetch:
            for shard_id, (_, local_ids_list) in grouped.items():
                _prefetch_tensor_indices(self._shards[shard_id], local_ids_list)
        for shard_id, (positions_list, local_ids_list) in grouped.items():
            positions = torch.tensor(positions_list, dtype=torch.long)
            local_ids = torch.tensor(local_ids_list, dtype=torch.long)
            rows = self._shards[shard_id].index_select(0, local_ids)
            output.index_copy_(0, positions, rows)
        self._prefill_source_rows += ngram_ids.numel()

    def _copy_rows(self, ngram_ids: torch.Tensor, output: torch.Tensor) -> None:
        before_faults = (
            resource.getrusage(resource.RUSAGE_SELF).ru_majflt
            if self._collect_ple_stats
            else 0
        )
        batch = get_global_ctx().batch
        if batch.is_decode:
            self._copy_decode_rows(ngram_ids, output)
        else:
            self._copy_prefill_rows(ngram_ids, output)
        if self._collect_ple_stats:
            faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - before_faults
            if batch.is_decode:
                self._decode_major_faults += faults
            else:
                self._prefill_major_faults += faults

    def _lookup(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._dummy:
            token_count = get_global_ctx().batch.input_ids.numel()
            return torch.zeros(token_count, self.embedding_dim, device=device, dtype=dtype)
        with decode_profile_range("FT.PLE.Lookup"):
            ngram_ids = self._current_ngram_ids().reshape(-1)
            with decode_profile_range("FT.PLE.ShardPlanCPU"):
                # Shard boundaries and starts are immutable host metadata cached at
                # load time.  The 16-row decode path uses bisect directly below.
                if not self._shard_starts_list:
                    raise RuntimeError("Qwen4-Exp PLE shards are not loaded")
            with decode_profile_range("FT.PLE.PinnedBufferCPU"):
                output = self._ensure_output_buffer(ngram_ids.numel())
            with decode_profile_range("FT.PLE.ShardIndexSelectCPU"):
                self._copy_rows(ngram_ids, output)
            with decode_profile_range("FT.PLE.H2D"):
                fp8 = output.to(device=device, non_blocking=True).view(torch.float8_e4m3fn)
            with decode_profile_range("FT.PLE.DeviceDequant"):
                if (
                    self._device_scale is None
                    or self._device_scale.device != device
                    or self._device_scale.dtype != dtype
                ):
                    self._device_scale = self._scale.to(device=device, dtype=dtype)
                embedded = fp8.to(dtype) * self._device_scale
            return embedded.view(-1, self.embedding_dim)

    def prepare_cuda_graph_capture(
        self, token_count: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self._ensure_output_buffer(token_count * self.ngram_heads)
        if self._graph_output is None or self._graph_output.shape[0] < token_count:
            self._graph_output = torch.zeros(
                token_count, self.embedding_dim, device=device, dtype=dtype
            )

    def prepare_cuda_graph_replay(self, device: torch.device, dtype: torch.dtype) -> None:
        assert self._graph_output is not None
        embedded = self._lookup(device, dtype)
        with decode_profile_range("FT.PLE.GraphOutputCopy"):
            self._graph_output[: embedded.shape[0]].copy_(embedded)

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.cuda_graph_capture:
            assert self._graph_output is not None
            return self._graph_output[: batch.input_ids.numel()]
        return self._lookup(device, dtype)


class _DepthwiseConv(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)


class _PLELayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        hc_size = self.hidden_size * self.hc_count
        self.ple_embedding = _HostNGramEmbedding(config, layer_id)
        self.key_proj = LinearReplicated(args.ple_embed_dim, hc_size, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, self.hidden_size, has_bias=False)
        self.norm_key = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_query = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_conv = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.conv1d = _DepthwiseConv(hc_size, args.ple_conv_kernel_size)
        self.dilation = args.ngram_size
        self.state_len = (args.ple_conv_kernel_size - 1) * self.dilation
        self._conv_state_pool: torch.Tensor | None = None
        self._mtp_prefix_conv_state: torch.Tensor | None = None

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        self.ple_embedding.load_host_weights(model_path, dummy=dummy)

    def _ensure_conv_state_pool(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        linear_pool = get_global_ctx().linear_state_pool
        if linear_pool is None:
            raise RuntimeError("Qwen4-Exp PLE requires a linear-state pool")
        num_slots = linear_pool.conv_states.shape[1]
        expected = (num_slots, self.hc_count * self.hidden_size, self.state_len)
        if (
            self._conv_state_pool is None
            or self._conv_state_pool.shape != expected
            or self._conv_state_pool.device != device
            or self._conv_state_pool.dtype != dtype
        ):
            self._conv_state_pool = torch.zeros(expected, device=device, dtype=dtype)
        return self._conv_state_pool

    def prepare_cuda_graph_capture(self, token_count: int) -> None:
        device = self.conv1d.weight.device
        dtype = self.conv1d.weight.dtype
        self.ple_embedding.prepare_cuda_graph_capture(token_count, device, dtype)
        self._ensure_conv_state_pool(device, dtype)

    def prepare_cuda_graph_replay(self) -> None:
        self.ple_embedding.prepare_cuda_graph_replay(
            self.conv1d.weight.device, self.conv1d.weight.dtype
        )

    def _save_mtp_prefix_state(
        self, state: torch.Tensor, checkpoint_index: int, checkpoint_capacity: int
    ) -> None:
        if not 0 <= checkpoint_index < checkpoint_capacity:
            raise RuntimeError("MTP PLE checkpoint index is outside its capacity")
        expected = (checkpoint_capacity, *state.shape)
        if (
            self._mtp_prefix_conv_state is None
            or self._mtp_prefix_conv_state.shape != expected
            or self._mtp_prefix_conv_state.device != state.device
            or self._mtp_prefix_conv_state.dtype != state.dtype
        ):
            self._mtp_prefix_conv_state = torch.empty(
                expected, dtype=state.dtype, device=state.device
            )
        self._mtp_prefix_conv_state[checkpoint_index].copy_(state)

    def commit_mtp_prefix_state(
        self, table_idx: int, checkpoint_index: int
    ) -> None:
        if self._conv_state_pool is None or self._mtp_prefix_conv_state is None:
            raise RuntimeError("Qwen4-Exp MTP PLE prefix state is not initialized")
        if not 0 <= checkpoint_index < self._mtp_prefix_conv_state.shape[0]:
            raise RuntimeError("MTP PLE checkpoint index is outside its capacity")
        self._conv_state_pool[table_idx].copy_(
            self._mtp_prefix_conv_state[checkpoint_index]
        )

    def _short_conv(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        state_pool = self._ensure_conv_state_pool(hidden.device, hidden.dtype)
        if batch.cuda_graph_capture:
            assert batch.linear_table_idx is not None
            slots = batch.linear_table_idx.long()
            if getattr(batch, "mtp_verify", False):
                if len(reqs) != 1 or slots.numel() != 1:
                    raise RuntimeError("Qwen4-Exp MTP PLE expects one request")
                state = state_pool.index_select(0, slots)
                current = hidden.transpose(0, 1).unsqueeze(0)
                combined = torch.cat([state, current], dim=-1)
                convolved = F.conv1d(
                    combined,
                    self.conv1d.weight,
                    groups=self.conv1d.weight.shape[0],
                    dilation=self.dilation,
                )
                for checkpoint_index in range(hidden.shape[0] - 1):
                    start = checkpoint_index + 1
                    self._save_mtp_prefix_state(
                        combined[0, :, start : start + self.state_len],
                        checkpoint_index,
                        batch.mtp_checkpoint_capacity,
                    )
                state_pool.index_copy_(0, slots, combined[..., -self.state_len :])
                return F.silu(convolved).squeeze(0).transpose(0, 1)
            state = state_pool.index_select(0, slots)
            combined = torch.cat([state, hidden.unsqueeze(-1)], dim=-1)
            convolved = F.conv1d(
                combined,
                self.conv1d.weight,
                groups=self.conv1d.weight.shape[0],
                dilation=self.dilation,
            )
            state_pool.index_copy_(0, slots, combined[..., -self.state_len :])
            return F.silu(convolved).squeeze(-1)

        outputs = []
        offset = 0
        weight = self.conv1d.weight
        for req in reqs:
            length = req.extend_len
            current = hidden[offset : offset + length].transpose(0, 1).unsqueeze(0)
            state = state_pool[req.table_idx].unsqueeze(0)
            if req.cached_len == 0:
                state.zero_()
            combined = torch.cat([state, current], dim=-1)
            convolved = F.conv1d(
                combined,
                weight,
                groups=weight.shape[0],
                dilation=self.dilation,
            )
            outputs.append(F.silu(convolved).squeeze(0).transpose(0, 1))
            state_pool[req.table_idx].copy_(combined[0, :, -self.state_len :])
            offset += length
        return torch.cat(outputs, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with decode_profile_range("FT.PLE.Layer"):
            embeddings = self.ple_embedding.forward(hidden.device, hidden.dtype)
            key = self.norm_key.forward(self.key_proj.forward(embeddings))
            key = key.view(-1, self.hc_count, self.hidden_size)
            value = self.value_proj.forward(embeddings)
            query = self.norm_query.forward(hidden).view(-1, self.hc_count, self.hidden_size)
            gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gated = (torch.sigmoid(gate) * value.unsqueeze(1)).flatten(1)
            normalized = self.norm_conv.forward(gated)
            return gated + self._short_conv(normalized)


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        self.hc_count = config.qwen4_args.hc_count
        self.hidden_size = config.hidden_size
        dense_config = replace(config, expert_quant="none", attn_quant="none")
        if self._is_linear:
            group = config.linear_attention_group()
            assert group is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant="none",
                attn_quant="none",
            )
            self.linear_attn.norm = _GatedRMSNorm(
                group.value_head_dim,
                config.rms_norm_eps,
                config.qwen4_args.output_gate_type,
                save_stats=self.linear_attn.norm.save_stats,
            )
        else:
            self.self_attn = Qwen4ExpAttention(dense_config, layer_id)
        self.mlp = _SparseMoE(config, layer_id)
        self.ple = (
            _PLELayer(config, layer_id)
            if layer_id in config.qwen4_args.ple_layer_ids
            else None
        )
        self.attn_hyper_connection = _GatedResidual(config)
        self.mlp_hyper_connection = _GatedResidual(config)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden)
        with decode_profile_range("FT.HyperConnection.Attention"):
            mixed, residual, weights = self.attn_hyper_connection.forward(hidden)
        mixed = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        with decode_profile_range("FT.HyperConnection.AttentionCombine"):
            hidden = _weighted_residual_combine(
                residual,
                mixed,
                weights,
                self.hc_count,
                self.hidden_size,
                self.attn_hyper_connection._fused,
            )
        with decode_profile_range("FT.HyperConnection.MoE"):
            mixed, residual, weights = self.mlp_hyper_connection.forward(hidden)
        mixed = self.mlp.forward(mixed)
        with decode_profile_range("FT.HyperConnection.MoECombine"):
            return _weighted_residual_combine(
                residual,
                mixed,
                weights,
                self.hc_count,
                self.hidden_size,
                self.mlp_hyper_connection._fused,
            )


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = _GatedResidual(config, combine=False)
        self.hc_count = config.qwen4_args.hc_count

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.load_host_weights(model_path, dummy=dummy)

    def prepare_cuda_graph_capture(self, token_count: int) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.prepare_cuda_graph_capture(token_count)

    def prepare_cuda_graph_replay(self) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.prepare_cuda_graph_replay()

    def set_ple_mmap_advice(self, advice: str) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.ple_embedding.set_mmap_advice(advice)

    def prepare_ple_cold_replay(self, advice: str) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.ple_embedding.prepare_cold_mmap_replay(advice)

    def reset_ple_cache_stats(self) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.ple_embedding.reset_cache_stats()

    def ple_cache_stats(self) -> dict:
        layers = [
            layer.ple.ple_embedding.cache_stats()
            for layer in self.layers.op_list
            if layer.ple is not None
        ]
        hits = sum(int(layer["cache_hits"]) for layer in layers)
        misses = sum(int(layer["cache_misses"]) for layer in layers)
        attempts = hits + misses
        return {
            "mmap_advice": (
                layers[0]["mmap_advice"]
                if layers
                and all(
                    layer["mmap_advice"] == layers[0]["mmap_advice"]
                    for layer in layers
                )
                else None
            ),
            "cache_enabled": any(bool(layer["cache_enabled"]) for layer in layers),
            "cache_capacity_rows": sum(
                int(layer["cache_capacity_rows"]) for layer in layers
            ),
            "cache_capacity_bytes": sum(
                int(layer["cache_capacity_bytes"]) for layer in layers
            ),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": hits / attempts if attempts else None,
            "decode_source_rows": sum(
                int(layer["decode_source_rows"]) for layer in layers
            ),
            "prefill_source_rows": sum(
                int(layer["prefill_source_rows"]) for layer in layers
            ),
            "decode_major_faults": sum(
                int(layer["decode_major_faults"]) for layer in layers
            ),
            "prefill_major_faults": sum(
                int(layer["prefill_major_faults"]) for layer in layers
            ),
            "layers": layers,
        }

    def snapshot_ple_state(self, table_idx: int) -> tuple[torch.Tensor, ...]:
        snapshots = []
        for layer in self.layers.op_list:
            if layer.ple is None:
                continue
            state_pool = layer.ple._conv_state_pool
            if state_pool is None:
                raise RuntimeError("Qwen4-Exp PLE state pool is not initialized")
            snapshots.append(state_pool[table_idx].clone())
        return tuple(snapshots)

    def restore_ple_state(
        self, table_idx: int, snapshots: tuple[torch.Tensor, ...]
    ) -> None:
        state_layers = [
            layer.ple for layer in self.layers.op_list if layer.ple is not None
        ]
        if len(state_layers) != len(snapshots):
            raise ValueError(
                f"Qwen4-Exp PLE snapshot has {len(snapshots)} layers, "
                f"expected {len(state_layers)}"
            )
        for ple, snapshot in zip(state_layers, snapshots):
            state_pool = ple._conv_state_pool
            if state_pool is None:
                raise RuntimeError("Qwen4-Exp PLE state pool is not initialized")
            state_pool[table_idx].copy_(snapshot)

    def commit_mtp_prefix_state(
        self,
        linear_pool,
        linear_slot: int,
        table_idx: int,
        checkpoint_index: int,
    ) -> None:
        for layer in self.layers.op_list:
            if layer._is_linear:
                layer.linear_attn.commit_mtp_prefix_state(
                    linear_pool, linear_slot, checkpoint_index
                )
            if layer.ple is not None:
                layer.ple.commit_mtp_prefix_state(table_idx, checkpoint_index)

    def forward(
        self, input_ids: torch.Tensor, *, return_expanded: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        mixed = self.hyper_connection_mixer.forward(hidden)
        return (mixed, hidden) if return_expanded else mixed


class _MTPModule(BaseOP):
    def __init__(self, config: ModelConfig):
        args: Qwen4ExpArgs = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.pre_fc_norm_embedding = _GroupedRMSNorm(
            config.hidden_size,
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.pre_fc_norm_hidden = _GroupedRMSNorm(
            args.hc_count * config.hidden_size,
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        first_layer_id = config.num_layers
        layers = [
            Qwen4ExpDecoderLayer(config, first_layer_id + index)
            for index in range(args.mtp_num_hidden_layers)
        ]
        for index, layer in enumerate(layers):
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and hasattr(experts, "layer_id"):
                experts.layer_id = index
        self.layers = OPList(layers)
        self.hyper_connection_mixer = _GatedResidual(config, combine=False)

    def fuse_inputs(
        self, expanded_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if expanded_hidden.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                "Qwen4-Exp MTP hidden width does not match the hyper-connection stream"
            )
        embedded = self.fc_embedding.forward(
            self.pre_fc_norm_embedding.forward(token_embeddings)
        )
        hidden = self.pre_fc_norm_hidden.forward(expanded_hidden)
        hidden = hidden.view(-1, self.hc_count, self.hidden_size)
        hidden = self.fc_hidden.forward(hidden)
        return (hidden + embedded.unsqueeze(1)).flatten(1)

    def forward(
        self, expanded_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.fuse_inputs(expanded_hidden, token_embeddings)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        return self.hyper_connection_mixer.forward(hidden), hidden


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
        self.mtp = _MTPModule(config) if config.qwen4_args.mtp_enabled else None
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def _iter_offload_moe_layers(self):
        for layer in self.model.layers.op_list:
            yield layer.mlp.experts

    def auxiliary_runtime_memory_bytes(self, config) -> int:
        if self.mtp is None:
            return 0
        from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req

        cache_size = _mtp_moe_cache_size()
        expert_bytes = (
            3
            * config.model_config.hidden_size
            * config.model_config.moe_intermediate_size
            * config.dtype.itemsize
        )
        linear_group = config.model_config.linear_attention_group()
        checkpoint_capacity = _mtp_max_drafts()
        linear_checkpoint_bytes = (
            checkpoint_capacity
            * linear_state_bytes_per_req(
                linear_group, config.tp_info.size, config.dtype
            )
            if linear_group is not None
            else 0
        )
        args = config.model_config.qwen4_args
        ple_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        ple_checkpoint_bytes = (
            checkpoint_capacity
            * len(args.ple_layer_ids)
            * args.hc_count
            * config.model_config.hidden_size
            * ple_state_len
            * config.dtype.itemsize
        )
        return (
            cache_size * expert_bytes
            + linear_checkpoint_bytes
            + ple_checkpoint_bytes
            + (32 << 20)
        )

    def setup_auxiliary_runtime(self, config, device: torch.device):
        if self.mtp is None:
            return None
        from freetoken.moe import is_offload_moe_backend

        if not is_offload_moe_backend(config.moe_backend):
            raise NotImplementedError(
                "Qwen4-Exp MTP currently requires an offload-family MoE backend"
            )
        if config.tp_info.size != 1:
            raise NotImplementedError(
                "Qwen4-Exp MTP currently requires tensor parallel size 1"
            )
        from freetoken.models.qwen4_exp.weight import load_mtp_expert_banks
        from freetoken.moe.offload_cache import OffloadMoeCache

        banks = load_mtp_expert_banks(
            config.model_path,
            config.model_config,
            dummy=config.use_dummy_weight,
        )
        cache = OffloadMoeCache(
            num_layers=config.model_config.qwen4_args.mtp_num_hidden_layers,
            num_experts=config.model_config.num_experts,
            cache_size=_mtp_moe_cache_size(),
            device=device,
            cache_policy=config.moe_cache_policy,
            prefill_overlap=False,
            quant_format="bf16",
            decode_target="gpu",
            allow_partial_cache=True,
        )
        cache.collect_stats = config.moe_collect_stats
        cache.collect_decode_freq = config.moe_collect_stats
        cache.set_bank_sources(banks.sources)
        for layer in self.mtp.layers.op_list:
            layer.mlp.experts.offload_cache = cache
        self.mtp_offload_cache = cache
        return cache

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        self.model.load_host_weights(model_path, dummy=dummy)

    def prepare_cuda_graph_capture(self, batch: Batch) -> None:
        token_count = batch.input_ids.numel()
        if self.mtp is not None:
            token_count = max(token_count, _mtp_max_drafts() + 1)
        self.model.prepare_cuda_graph_capture(token_count)

    def prepare_cuda_graph_replay(self, batch: Batch) -> None:
        self.model.prepare_cuda_graph_replay()

    def set_ple_mmap_advice(self, advice: str) -> None:
        self.model.set_ple_mmap_advice(advice)

    def prepare_ple_cold_replay(self, advice: str) -> None:
        self.model.prepare_ple_cold_replay(advice)

    def reset_ple_cache_stats(self) -> None:
        self.model.reset_ple_cache_stats()

    def ple_cache_stats(self) -> dict:
        return self.model.ple_cache_stats()

    def snapshot_runtime_state(self, req) -> tuple[torch.Tensor, ...]:
        return self.model.snapshot_ple_state(req.table_idx)

    def restore_runtime_state(
        self, req, snapshot: tuple[torch.Tensor, ...]
    ) -> None:
        self.model.restore_ple_state(req.table_idx, snapshot)

    def commit_mtp_prefix_state(
        self, req, linear_pool, checkpoint_index: int
    ) -> None:
        linear_slot = (
            req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        )
        self.model.commit_mtp_prefix_state(
            linear_pool, linear_slot, req.table_idx, checkpoint_index
        )

    def forward(
        self, *, return_expanded: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output = self.model.forward(
            get_global_ctx().batch.input_ids,
            return_expanded=return_expanded,
        )
        if return_expanded:
            hidden, expanded = output
            return self.lm_head.forward(hidden), expanded
        return self.lm_head.forward(output)

    def mtp_forward(
        self, expanded_hidden: torch.Tensor, next_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mtp is None:
            raise RuntimeError("Qwen4-Exp MTP is disabled")
        token_embeddings = self.model.embed_tokens.forward(next_token_ids)
        hidden, expanded = self.mtp.forward(expanded_hidden, token_embeddings)
        return self.lm_head.forward(hidden), expanded


__all__ = ["Qwen4ExpForCausalLM", "build_ngram_ids"]
