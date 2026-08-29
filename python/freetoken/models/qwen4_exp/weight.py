from __future__ import annotations

import json
import os
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.utils import download_hf_weight
from tqdm import tqdm

from freetoken.models.qwen3_5_moe.weight import (
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    setup_offload_expert_banks,
)

from .config import _mtp_enabled


_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    ".self_attn.qkv_proj.weight": (
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
        ),
        0,
    ),
    ".linear_attn.in_proj.weight": (
        (
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.in_proj_a.weight",
        ),
        0,
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        (
            ".mlp.shared_expert.gate_proj.weight",
            ".mlp.shared_expert.up_proj.weight",
        ),
        0,
    ),
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": (
        (
            ".attn_hyper_connection.input_mix_weight_down.weight",
            ".attn_hyper_connection.block_inject_weight.weight",
        ),
        16,
    ),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": (
        (
            ".mlp_hyper_connection.input_mix_weight_down.weight",
            ".mlp_hyper_connection.block_inject_weight.weight",
        ),
        16,
    ),
}


def _rename(raw_name: str) -> str | None:
    if raw_name.startswith("mtp.") and not _mtp_enabled():
        return None
    if raw_name.startswith(("model.visual.", "visual.")):
        return None
    if ".ple.ple_embedding.ngram_embedding." in raw_name:
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(name: str, tensor: torch.Tensor, buffers: dict):
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for index, part in enumerate(parts):
            if name.endswith(part):
                fused_name = name[: -len(part)] + fused_suffix
                slots = buffers.setdefault(fused_name, {})
                slots[index] = tensor
                if len(slots) == len(parts):
                    del buffers[fused_name]
                    rows = [slots[i] for i in range(len(parts))]
                    padding = (-sum(row.shape[0] for row in rows)) % pad_to if pad_to else 0
                    if padding:
                        rows.append(
                            torch.zeros(
                                padding,
                                *rows[0].shape[1:],
                                dtype=rows[0].dtype,
                                device=rows[0].device,
                            )
                        )
                    return fused_name, torch.cat(rows, dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen4-Exp currently supports TP=1 only")
    if include_moe_experts:
        raise ValueError("Qwen4-Exp requires --moe-backend offload, cpu, or hybrid")
    if not include_non_moe:
        return

    buffers = {}
    for filename in tqdm(
        iter_weight_files(model_path),
        desc="Loading Qwen4-Exp resident weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(filename, framework="pt", device=str(device)) as handle:
            for raw_name in handle.keys():
                name = _rename(raw_name)
                if (
                    name is None
                    or ".mlp.experts." in name
                    or raw_name.endswith(".weight_scale_inv")
                ):
                    continue
                tensor = handle.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, buffers)
                if fused is not None:
                    if fused:
                        yield fused
                    continue
                yield name, tensor
    if buffers:
        raise RuntimeError(f"Incomplete Qwen4-Exp projection fusions: {sorted(buffers)}")


def load_mtp_expert_banks(model_path: str, config, *, dummy: bool = False):
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import PinPipeline, alloc_layer_banks

    if not config.qwen4_args.mtp_enabled:
        raise ValueError("Qwen4-Exp MTP expert loading requested while MTP is disabled")
    if config.qwen4_args.mtp_num_hidden_layers != 1:
        raise NotImplementedError("Qwen4-Exp currently supports one MTP layer")

    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.moe_intermediate_size
    host_banks = alloc_layer_banks(
        {
            "gate_up": (
                (experts, 2 * intermediate, hidden),
                torch.bfloat16,
            ),
            "down": (
                (experts, hidden, intermediate),
                torch.bfloat16,
            ),
        },
        1,
    )
    if dummy:
        for per_layer in host_banks.values():
            per_layer[0].tensor.zero_()
        with PinPipeline() as pins:
            pins(0, {name: per_layer[0] for name, per_layer in host_banks.items()})
    else:
        folder = download_hf_weight(model_path)
        with open(
            os.path.join(folder, "model.safetensors.index.json"),
            encoding="utf-8",
        ) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        keys = {
            "gate_up": "mtp.layers.0.mlp.experts.gate_up_proj",
            "down": "mtp.layers.0.mlp.experts.down_proj",
        }
        paths = {os.path.join(folder, weight_map[key]) for key in keys.values()}
        with PinPipeline() as pins:
            for bank_name, key in keys.items():
                path = os.path.join(folder, weight_map[key])
                with safetensors.safe_open(
                    path, framework="pt", device="cpu"
                ) as handle:
                    tensor = handle.get_tensor(key)
                    destination = host_banks[bank_name][0].tensor
                    if tensor.dtype != destination.dtype or tensor.shape != destination.shape:
                        raise RuntimeError(
                            f"Unexpected Qwen4-Exp MTP expert tensor {key}: "
                            f"{tensor.dtype} {tuple(tensor.shape)}"
                        )
                    destination.copy_(tensor)
                pins.submit(host_banks[bank_name][0])
        for path in paths:
            drop_page_cache(path)

    return ExpertBanks(
        "bf16",
        {
            name: [per_layer[0].tensor]
            for name, per_layer in host_banks.items()
        },
    )


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "load_mtp_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "setup_offload_expert_banks",
]
