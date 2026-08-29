"""Replay Qwen4-Exp PLE row reads without loading the model or expert banks."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import mmap
import os
import resource
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
_DEFAULT_REPORT = "profiles/qwen38-opt-sweep-forced-20260829/report.json"
_PAGE_SIZE = mmap.PAGESIZE


@dataclass(frozen=True)
class _TensorMeta:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _TableShard:
    path: Path
    tensor_name: str
    global_start: int
    rows: int
    row_bytes: int
    file_offset: int

    @property
    def global_end(self) -> int:
        return self.global_start + self.rows


@dataclass(frozen=True)
class _ReplayTable:
    shards: tuple[_TableShard, ...]
    row_bytes: int

    @property
    def shard_ends(self) -> tuple[int, ...]:
        return tuple(shard.global_end for shard in self.shards)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(shard.path for shard in self.shards))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _resolve_model(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )


def _load_header(path: Path) -> dict[str, _TensorMeta]:
    with path.open("rb") as file:
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = file.read(header_length)
    header = json.loads(raw_header)
    data_start = 8 + header_length
    tensors = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        start, end = (int(value) for value in raw["data_offsets"])
        tensors[name] = _TensorMeta(
            path=path,
            dtype=str(raw["dtype"]),
            shape=tuple(int(value) for value in raw["shape"]),
            offset=data_start + start,
            nbytes=end - start,
        )
    return tensors


def _tensor_meta(
    folder: Path,
    weight_map: dict[str, str],
    headers: dict[Path, dict[str, _TensorMeta]],
    name: str,
) -> _TensorMeta:
    try:
        path = folder / weight_map[name]
    except KeyError as error:
        raise ValueError(f"checkpoint is missing {name}") from error
    header = headers.get(path)
    if header is None:
        header = _load_header(path)
        headers[path] = header
    try:
        return header[name]
    except KeyError as error:
        raise ValueError(f"{path} is missing indexed tensor {name}") from error


def _read_i64(meta: _TensorMeta) -> tuple[int, ...]:
    if meta.dtype != "I64" or meta.nbytes % 8:
        raise ValueError(
            f"expected an I64 tensor for {meta.path}:{meta.offset}, got {meta.dtype}"
        )
    count = meta.nbytes // 8
    with meta.path.open("rb") as file:
        file.seek(meta.offset)
        raw = file.read(meta.nbytes)
    if len(raw) != meta.nbytes:
        raise ValueError(f"short tensor read from {meta.path}")
    return struct.unpack(f"<{count}q", raw)


def _load_ple_table(
    folder: Path,
) -> tuple[
    _ReplayTable, dict[str, Any], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    with (folder / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    text = config.get("text_config", config)
    ple_layer_ids = text.get("ple_layer_ids")
    if not ple_layer_ids:
        raise ValueError("checkpoint config has no PLE layer")
    if len(ple_layer_ids) != 1:
        raise ValueError(
            f"PLE replay currently needs one PLE layer, got {ple_layer_ids}"
        )
    layer_id = int(ple_layer_ids[0]) - 1
    split_parts = int(text["split_ngram_parts"])
    ngram_size = int(text["ngram_size"])
    heads_per_ngram = int(text["heads_per_ngram"])
    embedding_dim = int(text["ple_embed_dim"])
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if embedding_dim % ngram_heads:
        raise ValueError(
            f"PLE embedding dimension {embedding_dim} is not divisible by {ngram_heads} heads"
        )
    expected_row_bytes = embedding_dim // ngram_heads

    with (folder / "model.safetensors.index.json").open(encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]
    headers: dict[Path, dict[str, _TensorMeta]] = {}
    prefix = f"model.language_model.layers.{layer_id}.ple.ple_embedding"
    multipliers = _read_i64(
        _tensor_meta(folder, weight_map, headers, prefix + ".layer_multipliers")
    )
    vocab_sizes = _read_i64(
        _tensor_meta(folder, weight_map, headers, prefix + ".ngram_heads_vocab_sizes")
    )
    offsets = _read_i64(
        _tensor_meta(folder, weight_map, headers, prefix + ".ngram_heads_offsets")
    )
    if len(multipliers) != ngram_size:
        raise ValueError(
            f"PLE has {len(multipliers)} multipliers, expected {ngram_size}"
        )
    if len(vocab_sizes) != ngram_heads or len(offsets) != ngram_heads:
        raise ValueError(
            f"PLE has {len(vocab_sizes)} vocab sizes and {len(offsets)} offsets, "
            f"expected {ngram_heads}"
        )

    shards = []
    global_start = 0
    for shard_id in range(split_parts):
        name = prefix + f".ngram_embedding.shard_{shard_id}.weight"
        meta = _tensor_meta(folder, weight_map, headers, name)
        if meta.dtype != "F8_E4M3" or len(meta.shape) != 2:
            raise ValueError(f"unexpected PLE shard {name}: {meta.dtype} {meta.shape}")
        rows, row_bytes = meta.shape
        if row_bytes != expected_row_bytes or meta.nbytes != rows * row_bytes:
            raise ValueError(
                f"unexpected PLE shard row layout for {name}: {meta.shape}, {meta.nbytes} bytes"
            )
        shards.append(
            _TableShard(
                path=meta.path,
                tensor_name=name,
                global_start=global_start,
                rows=rows,
                row_bytes=row_bytes,
                file_offset=meta.offset,
            )
        )
        global_start += rows
    table = _ReplayTable(tuple(shards), expected_row_bytes)
    geometry = {
        "layer_id": layer_id,
        "ngram_size": ngram_size,
        "heads_per_ngram": heads_per_ngram,
        "ngram_heads": ngram_heads,
        "embedding_dim": embedding_dim,
        "row_bytes": expected_row_bytes,
        "logical_shards": len(shards),
        "physical_files": len(table.paths),
        "table_rows": global_start,
        "table_bytes": sum(shard.rows * shard.row_bytes for shard in shards),
        "eos_token_id": int(text["eos_token_id"]),
    }
    return table, geometry, multipliers, vocab_sizes, offsets


def _trajectory_from_report(
    report: dict[str, Any],
    *,
    prompt_phase: str,
    variant: str,
    window: str,
) -> tuple[list[int], list[int]]:
    try:
        prompt = report["phase_prompt_token_ids"][prompt_phase]
        output = report["optimization_sweep"][variant][window]["output_token_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"report has no {prompt_phase!r} prompt and {variant!r}/{window!r} trajectory"
        ) from error
    if not prompt or not output:
        raise ValueError("PLE replay needs non-empty prompt and output token lists")
    return [int(token) for token in prompt], [int(token) for token in output]


def _ngram_rows(
    prompt: list[int],
    output: list[int],
    *,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    multipliers: tuple[int, ...],
    vocab_sizes: tuple[int, ...],
    offsets: tuple[int, ...],
) -> list[tuple[int, ...]]:
    tokens = [*prompt, *output]
    last_eos = -1
    for position, token in enumerate(prompt):
        if token == eos_token_id:
            last_eos = position
    steps = []
    for position in range(len(prompt), len(tokens)):
        mixed_by_ngram = []
        for ngram in range(2, ngram_size + 1):
            mixed = tokens[position] * multipliers[0]
            for shift in range(1, ngram):
                source = position - shift
                shifted = (
                    tokens[source]
                    if source >= 0 and source > last_eos
                    else eos_token_id
                )
                mixed ^= shifted * multipliers[shift]
            mixed_by_ngram.append(mixed)
        rows = []
        for ngram_index, mixed in enumerate(mixed_by_ngram):
            start = ngram_index * heads_per_ngram
            for head in range(start, start + heads_per_ngram):
                rows.append(mixed % vocab_sizes[head] + offsets[head])
        steps.append(tuple(rows))
        if tokens[position] == eos_token_id:
            last_eos = position
    return steps


def _count_unique_pages(
    table: _ReplayTable, rows_by_step: list[tuple[int, ...]]
) -> int:
    shard_ends = table.shard_ends
    unique_pages: set[tuple[Path, int]] = set()
    for rows in rows_by_step:
        for global_id in rows:
            shard_id = bisect.bisect_right(shard_ends, global_id)
            if shard_id >= len(table.shards):
                raise ValueError(f"PLE row {global_id} is outside the table")
            shard = table.shards[shard_id]
            offset = (
                shard.file_offset + (global_id - shard.global_start) * table.row_bytes
            )
            first_page = offset // _PAGE_SIZE
            last_page = (offset + table.row_bytes - 1) // _PAGE_SIZE
            for page in range(first_page, last_page + 1):
                unique_pages.add((shard.path, page))
    return len(unique_pages)


def _proc_io() -> dict[str, int]:
    values = {}
    with open("/proc/self/io", encoding="utf-8") as file:
        for line in file:
            key, value = line.split(":", 1)
            values[key] = int(value)
    return values


def _rusage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - value for key, value in before.items()}


def _quantile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index] / 1e6


def _drop_file_cache(paths: tuple[Path, ...]) -> float:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("cold PLE replay needs POSIX_FADV_DONTNEED")
    started = time.perf_counter()
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return time.perf_counter() - started


def _open_embedding(table: _ReplayTable, layer_id: int):
    import safetensors
    import torch
    from freetoken.models.qwen4_exp.model import _HostNGramEmbedding

    handles = {}
    shards = []
    try:
        for shard in table.shards:
            handle = handles.get(shard.path)
            if handle is None:
                handle = safetensors.safe_open(
                    shard.path, framework="pt", device="cpu"
                ).__enter__()
                handles[shard.path] = handle
            tensor = handle.get_tensor(shard.tensor_name).view(torch.uint8)
            if tuple(tensor.shape) != (shard.rows, shard.row_bytes):
                raise ValueError(
                    f"safetensors returned {tuple(tensor.shape)} for {shard.tensor_name}"
                )
            shards.append(tensor)
    except BaseException:
        for handle in handles.values():
            handle.__exit__(None, None, None)
        raise

    embedding = object.__new__(_HostNGramEmbedding)
    embedding.layer_id = layer_id
    embedding.head_dim = table.row_bytes
    embedding._handles = list(handles.values())
    embedding._shards = shards
    embedding._shard_ends_list = table.shard_ends
    embedding._shard_starts_list = tuple(shard.global_start for shard in table.shards)
    embedding._row_cache = None
    embedding._row_cache_tags = None
    embedding._mmap_advice = "normal"
    embedding._mmap_prefetch = False
    embedding._cache_hits = 0
    embedding._cache_misses = 0
    embedding._decode_source_rows = 0
    embedding._prefill_source_rows = 0
    embedding._decode_major_faults = 0
    embedding._prefill_major_faults = 0
    return embedding, handles


def _run_arm(
    table: _ReplayTable,
    embedding,
    rows_by_step: list[tuple[int, ...]],
    *,
    advice: str,
    cold: bool,
    unique_pages: int,
) -> dict[str, Any]:
    import torch
    from freetoken.models.qwen4_exp.model import _madvise_tensor_ranges

    if cold:
        _madvise_tensor_ranges(embedding._shards, mmap.MADV_DONTNEED)
    drop_seconds = _drop_file_cache(table.paths) if cold else 0.0
    embedding.set_mmap_advice(advice)
    token_rows = [torch.tensor(rows, dtype=torch.long) for rows in rows_by_step]
    output = torch.empty(len(rows_by_step[0]), table.row_bytes, dtype=torch.uint8)
    latencies = []
    checksum = 0
    before_io = _proc_io()
    before_usage = _rusage()
    started = time.perf_counter_ns()
    for rows in token_rows:
        step_started = time.perf_counter_ns()
        embedding._copy_decode_rows(rows, output)
        latencies.append(time.perf_counter_ns() - step_started)
        checksum = (checksum + int(output[0, 0]) + int(output[-1, -1])) & 0xFFFFFFFF
    elapsed_ns = time.perf_counter_ns() - started
    after_usage = _rusage()
    after_io = _proc_io()
    latencies.sort()
    requested_bytes = len(rows_by_step) * len(rows_by_step[0]) * table.row_bytes
    io_delta = _delta(after_io, before_io)
    read_bytes = io_delta.get("read_bytes", 0)
    elapsed_seconds = elapsed_ns / 1e9
    return {
        "advice": advice,
        "mapping_advice": "random" if advice == "random-willneed" else advice,
        "prefetch": "madvise_willneed" if advice == "random-willneed" else None,
        "cold_requested": cold,
        "cache_drop_seconds": drop_seconds,
        "steps": len(rows_by_step),
        "rows": len(rows_by_step) * len(rows_by_step[0]),
        "requested_row_bytes": requested_bytes,
        "unique_page_bytes": unique_pages * _PAGE_SIZE,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": len(rows_by_step) / elapsed_seconds,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) / 1e6,
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "p99": _quantile(latencies, 0.99),
            "max": latencies[-1] / 1e6,
        },
        "process_io_delta": io_delta,
        "resource_delta": _delta(after_usage, before_usage),
        "read_amplification_vs_rows": (
            read_bytes / requested_bytes if requested_bytes else None
        ),
        "read_amplification_vs_unique_pages": (
            read_bytes / (unique_pages * _PAGE_SIZE) if unique_pages else None
        ),
        "checksum": checksum,
    }


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Replay Qwen4-Exp PLE mmap reads without loading model weights",
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=Path(_DEFAULT_REPORT))
    parser.add_argument("--prompt-phase", default="wall")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--window", choices=("prime", "measured"), default="prime")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("normal", "random", "random-willneed"),
        default=["normal", "random", "random-willneed"],
    )
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("-o", "--output", type=Path)
    return parser


def _print_arm(result: dict[str, Any]) -> None:
    latency = result["latency_ms"]
    resources = result["resource_delta"]
    read_mib = result["process_io_delta"]["read_bytes"] / (1 << 20)
    print(
        f"  {result['advice']:<6} {result['steps_per_second']:>8.1f} step/s  "
        f"p50={latency['p50']:.3f} ms  p99={latency['p99']:.3f} ms  "
        f"majflt={resources['major_faults']:,}  read={read_mib:.1f} MiB  "
        f"amp={result['read_amplification_vs_rows']:.1f}x"
    )


def main(argv: list[str] | None = None, prog: str = "ft bench ple-replay") -> int:
    parser = _parser(prog)
    args = parser.parse_args(argv)
    try:
        with args.report.open(encoding="utf-8") as file:
            source_report = json.load(file)
        prompt, output = _trajectory_from_report(
            source_report,
            prompt_phase=args.prompt_phase,
            variant=args.variant,
            window=args.window,
        )
        if args.max_steps is not None:
            output = output[: args.max_steps]
        folder = _resolve_model(args.model)
        table, geometry, multipliers, vocab_sizes, offsets = _load_ple_table(folder)
        rows_by_step = _ngram_rows(
            prompt,
            output,
            ngram_size=geometry["ngram_size"],
            heads_per_ngram=geometry["heads_per_ngram"],
            eos_token_id=geometry["eos_token_id"],
            multipliers=multipliers,
            vocab_sizes=vocab_sizes,
            offsets=offsets,
        )
        unique_pages = _count_unique_pages(table, rows_by_step)
        embedding, handles = _open_embedding(table, geometry["layer_id"])
        arms = []
        try:
            for advice in args.arms:
                print(f"PLE replay: {advice} ...", flush=True)
                result = _run_arm(
                    table,
                    embedding,
                    rows_by_step,
                    advice=advice,
                    cold=args.cold,
                    unique_pages=unique_pages,
                )
                arms.append(result)
                _print_arm(result)
        finally:
            for handle in handles.values():
                handle.__exit__(None, None, None)
        report = {
            "version": 1,
            "timestamp": datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds"),
            "model": args.model,
            "model_folder": str(folder),
            "source_report": str(args.report),
            "trajectory": {
                "prompt_phase": args.prompt_phase,
                "variant": args.variant,
                "window": args.window,
                "prompt_tokens": len(prompt),
                "decode_input_tokens": len(output),
                "semantics": "Each saved output token is replayed as the next decode input.",
            },
            "geometry": geometry,
            "backend": "production_safetensors_gather",
            "page_size": _PAGE_SIZE,
            "unique_pages": unique_pages,
            "arms": arms,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"  report: {args.output}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
