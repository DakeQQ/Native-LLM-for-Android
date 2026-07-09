"""Benchmark the Qwen v3 KV-cache quantization implementations.

This script imports Export_Qwen.KVQuantizer and exercises the same quantize,
packing, rotation, Hadamard, shuffle, clipping, and scale/bias paths used by the
exporter. It compares each tested storage mode against a float32 KV cache,
reports round-trip accuracy, encode/decode elapsed time, and extrapolates cache
storage for a 2B-element KV cache. Use --suite stacked to run a broader
combinatorial matrix of overlapping quantization options. Running this file
directly from VS Code uses the defaults below and starts the full stacked suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import Export_Qwen as exporter


ROTARY_DTYPES = {"ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
Q4_DTYPES = {"ROTARY_Q4", "ROTARY_Q4_CUDA"}
CUDA_PACKED_DTYPES = {"Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"}
QUANTIZED_DTYPES = {"Q8", "Q8_CUDA", *ROTARY_DTYPES}
SCRIPT_DIR = Path(__file__).resolve().parent


# VS Code Run defaults. Edit these values when you want a different no-CLI run.
DEFAULT_SUITE = "stacked"
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_ATTENTION_HEADS = 16
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_SEQ_LEN = 1024
DEFAULT_GROUP_SIZE = 128
DEFAULT_GROUPED_SIZE = 64
DEFAULT_GROUP_SIZES = (128, 64, 32, 16)
DEFAULT_KV_ELEMENTS = 2_000_000_000
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1
DEFAULT_SEED = 20260708
DEFAULT_STD = 1.0
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_DEVICE = "cpu"
DEFAULT_THREADS = 0
DEFAULT_SAVE_RESULTS = True
DEFAULT_RESULTS_STEM = "KV_Quantization_stacked_results"


@dataclass(frozen=True)
class QuantCase:
    name: str
    kv_dtype: str
    group_size: int
    compute_in_f32: bool = False
    use_hadamard: bool = False
    use_clip: bool = False
    clip_sigma: float = 3.0
    use_shuffle: bool = False
    use_sym: bool = True
    use_float16_scale_bias: bool = True
    use_qdq_friendly_asym: bool = False


@dataclass
class BenchmarkResult:
    case: dict
    effective_group_size: int
    notes: list[str]
    sample_elements: int
    sample_storage_bytes: int
    sample_fp32_bytes: int
    compression_ratio: float
    assumed_fp32_bytes: int
    assumed_storage_bytes: int
    encode_ms_mean: float
    encode_ms_stdev: float
    decode_ms_mean: float
    decode_ms_stdev: float
    total_ms_mean: float
    key_mae: float
    key_rmse: float
    key_max_abs: float
    key_sqnr_db: float
    value_mae: float
    value_rmse: float
    value_max_abs: float
    value_sqnr_db: float
    combined_mae: float
    combined_rmse: float
    combined_max_abs: float
    combined_sqnr_db: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen v3 KV-cache quantization settings against float32. "
            "Default storage extrapolation assumes a 2B-element KV cache."
        )
    )
    parser.add_argument("--suite", choices=("quick", "full", "stacked"), default=DEFAULT_SUITE)
    parser.add_argument("--list-cases", action="store_true", help="Print cases for the chosen suite and exit.")
    parser.add_argument("--cases", default="", help="Comma-separated case names to run after suite expansion.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_NUM_ATTENTION_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--grouped-size", type=int, default=DEFAULT_GROUPED_SIZE)
    parser.add_argument(
        "--group-sizes",
        default=",".join(str(group_size) for group_size in DEFAULT_GROUP_SIZES),
        help="Comma-separated group-size sweep for --suite stacked; values are adjusted to valid head_dim divisors.",
    )
    parser.add_argument("--kv-elements", type=int, default=DEFAULT_KV_ELEMENTS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--std", type=float, default=DEFAULT_STD, help="Standard deviation for synthetic KV tensors.")
    parser.add_argument("--clip-sigma", type=float, default=DEFAULT_CLIP_SIGMA, help="Sigma bound used by cases with USE_CLIP.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=DEFAULT_DEVICE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Set torch CPU threads when > 0.")
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_RESULTS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()
    if args.save_results:
        if args.json_out is None:
            args.json_out = SCRIPT_DIR / f"{DEFAULT_RESULTS_STEM}.json"
        if args.csv_out is None:
            args.csv_out = SCRIPT_DIR / f"{DEFAULT_RESULTS_STEM}.csv"
    return args


def best_divisor_at_or_below(value: int, limit: int) -> int:
    limit = max(1, min(value, limit))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def group_size_sweep(args: argparse.Namespace, base_group: int, grouped_size: int) -> list[int]:
    raw_values = [part.strip() for part in args.group_sizes.split(",") if part.strip()]
    if not raw_values:
        raw_sizes = [base_group, grouped_size]
    else:
        raw_sizes = [int(value) for value in raw_values]
    raw_sizes.extend([base_group, grouped_size])
    group_sizes = {best_divisor_at_or_below(args.head_dim, size) for size in raw_sizes}
    return sorted(group_sizes, reverse=True)


def modifier_variants(group_size: int, head_dim: int) -> tuple[tuple[bool, bool], ...]:
    if group_size >= head_dim:
        return ((False, False),)
    return (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    )


def clip_label(clip_sigma: float) -> str:
    return f"clip{clip_sigma:g}".replace(".", "p")


def quant_case_name(
    kv_dtype: str,
    group_size: int,
    use_sym: bool,
    use_hadamard: bool,
    use_shuffle: bool,
    use_clip: bool,
    clip_sigma: float,
    use_float16_scale_bias: bool,
    use_qdq_friendly_asym: bool,
) -> str:
    name_parts = [kv_dtype, "sym" if use_sym else "asym", f"g{group_size}"]
    if use_hadamard:
        name_parts.append("hadamard")
    if use_shuffle:
        name_parts.append("shuffle")
    if use_clip:
        name_parts.append(clip_label(clip_sigma))
    name_parts.append("scale16" if use_float16_scale_bias else "scale32")
    if use_qdq_friendly_asym:
        name_parts.append("qdq")
    return "_".join(name_parts)


def build_stacked_cases(args: argparse.Namespace, base_group: int, grouped_size: int) -> list[QuantCase]:
    cases = [
        QuantCase("F32_reference", "F32", base_group),
        QuantCase("F16_compute_f16", "F16", base_group, compute_in_f32=False),
        QuantCase("F16_compute_f32", "F16", base_group, compute_in_f32=True),
    ]
    group_sizes = group_size_sweep(args, base_group, grouped_size)

    for kv_dtype in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"):
        for group_size in group_sizes:
            for use_hadamard, use_shuffle in modifier_variants(group_size, args.head_dim):
                for use_sym in (True, False):
                    qdq_options = (False,) if use_sym else (False, True)
                    for use_float16_scale_bias in (True, False):
                        for use_clip in (False, True):
                            for use_qdq_friendly_asym in qdq_options:
                                cases.append(QuantCase(
                                    name=quant_case_name(
                                        kv_dtype=kv_dtype,
                                        group_size=group_size,
                                        use_sym=use_sym,
                                        use_hadamard=use_hadamard,
                                        use_shuffle=use_shuffle,
                                        use_clip=use_clip,
                                        clip_sigma=args.clip_sigma,
                                        use_float16_scale_bias=use_float16_scale_bias,
                                        use_qdq_friendly_asym=use_qdq_friendly_asym,
                                    ),
                                    kv_dtype=kv_dtype,
                                    group_size=group_size,
                                    use_hadamard=use_hadamard,
                                    use_clip=use_clip,
                                    clip_sigma=args.clip_sigma,
                                    use_shuffle=use_shuffle,
                                    use_sym=use_sym,
                                    use_float16_scale_bias=use_float16_scale_bias,
                                    use_qdq_friendly_asym=use_qdq_friendly_asym,
                                ))
    return cases


def build_cases(args: argparse.Namespace) -> list[QuantCase]:
    base_group = best_divisor_at_or_below(args.head_dim, args.group_size)
    grouped_size = best_divisor_at_or_below(args.head_dim, min(args.grouped_size, max(1, args.head_dim // 2)))

    if args.suite == "stacked":
        cases = build_stacked_cases(args, base_group, grouped_size)
    else:
        cases: list[QuantCase] = [
            QuantCase("F32_reference", "F32", base_group),
            QuantCase("F16_compute_f16", "F16", base_group, compute_in_f32=False),
            QuantCase("F16_compute_f32", "F16", base_group, compute_in_f32=True),
            QuantCase(f"Q8_sym_g{base_group}", "Q8", base_group, use_sym=True),
            QuantCase(f"Q8_asym_g{base_group}", "Q8", base_group, use_sym=False),
            QuantCase(f"Q8_sym_g{grouped_size}_hadamard", "Q8", grouped_size, use_sym=True, use_hadamard=True),
            QuantCase(f"Q8_sym_g{grouped_size}_shuffle", "Q8", grouped_size, use_sym=True, use_shuffle=True),
            QuantCase(f"ROTARY_Q8_sym_g{base_group}", "ROTARY_Q8", base_group, use_sym=True),
            QuantCase(f"ROTARY_Q4_sym_g{base_group}", "ROTARY_Q4", base_group, use_sym=True),
        ]

        if args.suite == "full":
            cases.extend([
                QuantCase(f"Q8_sym_g{base_group}_scale32", "Q8", base_group, use_sym=True, use_float16_scale_bias=False),
                QuantCase(f"Q8_asym_g{base_group}_scale32", "Q8", base_group, use_sym=False, use_float16_scale_bias=False),
                QuantCase(f"Q8_sym_g{base_group}_clip", "Q8", base_group, use_sym=True, use_clip=True, clip_sigma=args.clip_sigma),
                QuantCase(f"Q8_sym_g{grouped_size}_hadamard_shuffle", "Q8", grouped_size, use_sym=True, use_hadamard=True, use_shuffle=True),
                QuantCase(f"Q8_CUDA_sym_g{base_group}", "Q8_CUDA", base_group, use_sym=True),
                QuantCase(f"Q8_CUDA_asym_g{base_group}", "Q8_CUDA", base_group, use_sym=False),
                QuantCase(f"ROTARY_Q8_asym_g{base_group}", "ROTARY_Q8", base_group, use_sym=False),
                QuantCase(f"ROTARY_Q8_sym_g{grouped_size}_hadamard", "ROTARY_Q8", grouped_size, use_sym=True, use_hadamard=True),
                QuantCase(f"ROTARY_Q8_sym_g{grouped_size}_shuffle", "ROTARY_Q8", grouped_size, use_sym=True, use_shuffle=True),
                QuantCase(f"ROTARY_Q8_CUDA_sym_g{base_group}", "ROTARY_Q8_CUDA", base_group, use_sym=True),
                QuantCase(f"ROTARY_Q8_CUDA_asym_g{base_group}", "ROTARY_Q8_CUDA", base_group, use_sym=False),
                QuantCase(f"ROTARY_Q4_asym_g{base_group}", "ROTARY_Q4", base_group, use_sym=False),
                QuantCase(f"ROTARY_Q4_sym_g{grouped_size}_hadamard", "ROTARY_Q4", grouped_size, use_sym=True, use_hadamard=True),
                QuantCase(f"ROTARY_Q4_sym_g{grouped_size}_shuffle", "ROTARY_Q4", grouped_size, use_sym=True, use_shuffle=True),
                QuantCase(f"ROTARY_Q4_CUDA_sym_g{base_group}", "ROTARY_Q4_CUDA", base_group, use_sym=True),
                QuantCase(f"ROTARY_Q4_CUDA_asym_g{base_group}", "ROTARY_Q4_CUDA", base_group, use_sym=False),
            ])

    if args.cases:
        wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
        missing = wanted.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"Unknown case name(s): {', '.join(sorted(missing))}")
        cases = [case for case in cases if case.name in wanted]

    return cases


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(name)


def configure_exporter(case: QuantCase, head_dim: int) -> tuple[int, list[str]]:
    exporter.KV_QUANT_DTYPE = case.kv_dtype
    exporter.KV_QUANT_GROUP_SIZE = case.group_size
    exporter.COMPUTE_IN_F32 = case.compute_in_f32
    exporter.USE_HADAMARD = case.use_hadamard
    exporter.USE_CLIP = case.use_clip
    exporter.CLIP_SIGMA = case.clip_sigma
    exporter.USE_SHUFFLE = case.use_shuffle
    exporter.USE_SYM = case.use_sym
    exporter.USE_FLOAT16_SCALE_BIAS = case.use_float16_scale_bias
    exporter.USE_QDQ_FRIENDLY_ASYM = case.use_qdq_friendly_asym
    notes = exporter.normalize_kv_quant_settings(head_dim)
    return exporter.KV_QUANT_GROUP_SIZE, notes


def build_quantizer(case: QuantCase, args: argparse.Namespace, device: torch.device) -> exporter.KVQuantizer | None:
    if case.kv_dtype not in QUANTIZED_DTYPES:
        return None
    num_kv_groups = args.num_attention_heads // args.num_kv_heads
    quantizer = exporter.KVQuantizer(
        head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads,
        num_kv_groups=num_kv_groups,
        is_q4=case.kv_dtype in Q4_DTYPES,
        is_rotary=case.kv_dtype in ROTARY_DTYPES,
        is_q8_cuda=case.kv_dtype in CUDA_PACKED_DTYPES,
        use_sym=case.use_sym,
        use_hadamard=case.use_hadamard,
        use_clip=case.use_clip,
        clip_sigma=case.clip_sigma,
        use_shuffle=case.use_shuffle,
    )
    return quantizer.to(device).eval()


def make_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed)
    key_shape = (args.batch_size, args.num_kv_heads, 1, args.head_dim, args.seq_len)
    value_shape = (args.batch_size, args.num_kv_heads, 1, args.seq_len, args.head_dim)
    key_cache = torch.randn(key_shape, generator=generator, device=device, dtype=torch.float32) * args.std
    value_cache = torch.randn(value_shape, generator=generator, device=device, dtype=torch.float32) * args.std
    return key_cache, value_cache


def pack_width_for(case: QuantCase, head_dim: int) -> int:
    if case.kv_dtype == "ROTARY_Q4_CUDA":
        return head_dim // 8
    return head_dim // 4


def cuda_unpack_head_dim_for(case: QuantCase, head_dim: int) -> int:
    if case.kv_dtype == "ROTARY_Q4_CUDA":
        return head_dim // 2
    return head_dim


def encode_case(
    case: QuantCase,
    quantizer: exporter.KVQuantizer | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, ...]:
    if case.kv_dtype == "F32":
        return key_cache.clone(), value_cache.clone()
    if case.kv_dtype == "F16":
        return key_cache.half(), value_cache.half()
    if quantizer is None:
        raise RuntimeError(f"No quantizer created for {case.name}.")
    return quantizer(
        key_cache,
        value_cache,
        args.batch_size,
        args.num_kv_heads,
        pack_width_for(case, args.head_dim),
    )


def dequantize_key(
    case: QuantCase,
    quantizer: exporter.KVQuantizer,
    packed_key: torch.Tensor,
    key_scale: torch.Tensor,
    key_bias: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    if exporter.USE_FLOAT16_SCALE_BIAS:
        key_scale = key_scale.float()
        if key_bias is not None:
            key_bias = key_bias.float()
    if quantizer.is_q8_cuda:
        packed_key = quantizer.unpack_cuda(
            packed_key,
            -2,
            args.batch_size,
            args.num_kv_heads,
            cuda_unpack_head_dim_for(case, args.head_dim),
        )
    if quantizer.is_q4:
        key_int = quantizer.unpack_q4_k(packed_key, args.batch_size)
        if quantizer.use_sym:
            key_int = quantizer._decode_signed_q4_storage(key_int)
    else:
        key_int = quantizer._decode_signed_q8_storage(packed_key) if quantizer.use_sym else packed_key
    key_float = key_int.float()
    if quantizer.is_grouped:
        grouped_key = key_float.view(
            args.batch_size,
            args.num_kv_heads,
            1,
            quantizer.kv_quant_num_groups,
            quantizer.kv_quant_group_size,
            -1,
        )
        key_cache = grouped_key * key_scale if quantizer.use_sym else grouped_key * key_scale + key_bias
        key_cache = key_cache.reshape(args.batch_size, args.num_kv_heads, 1, args.head_dim, -1)
    else:
        key_cache = key_float * key_scale if quantizer.use_sym else key_float * key_scale + key_bias
    if quantizer.use_hadamard:
        key_cache = quantizer.inverse_hadamard_k(key_cache, args.batch_size)
    if quantizer.use_shuffle:
        key_cache = key_cache.index_select(3, quantizer.unshuffle_idx)
    if quantizer.is_rotary:
        key_cache = quantizer.inverse_rotate_k(key_cache, args.batch_size)
    return key_cache


def inverse_hadamard_value(quantizer: exporter.KVQuantizer, value_cache: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    value_cache = value_cache.reshape(
        args.batch_size,
        args.num_kv_heads,
        1,
        -1,
        quantizer.kv_quant_num_groups,
        quantizer.kv_quant_group_size,
    )
    value_cache = quantizer._apply_hadamard_last_dim(value_cache, inverse=True)
    return value_cache.reshape(args.batch_size, args.num_kv_heads, 1, -1, args.head_dim)


def inverse_rotate_value(quantizer: exporter.KVQuantizer, value_cache: torch.Tensor, batch_size: int) -> torch.Tensor:
    return value_cache * quantizer.rot_cos - quantizer._flip_v(value_cache, batch_size) * quantizer.rot_sin_v


def dequantize_value(
    case: QuantCase,
    quantizer: exporter.KVQuantizer,
    packed_value: torch.Tensor,
    value_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    if exporter.USE_FLOAT16_SCALE_BIAS:
        value_scale = value_scale.float()
        if value_bias is not None:
            value_bias = value_bias.float()
    if quantizer.is_q8_cuda:
        packed_value = quantizer.unpack_cuda(
            packed_value,
            -1,
            args.batch_size,
            args.num_kv_heads,
            cuda_unpack_head_dim_for(case, args.head_dim),
        )
    if quantizer.is_q4:
        value_int = quantizer.unpack_q4_v(packed_value, args.batch_size)
        if quantizer.use_sym:
            value_int = quantizer._decode_signed_q4_storage(value_int)
    else:
        value_int = quantizer._decode_signed_q8_storage(packed_value) if quantizer.use_sym else packed_value
    value_float = value_int.float()
    if quantizer.is_grouped:
        grouped_value = value_float.view(
            args.batch_size,
            args.num_kv_heads,
            1,
            -1,
            quantizer.kv_quant_num_groups,
            quantizer.kv_quant_group_size,
        )
        value_cache = grouped_value * value_scale if quantizer.use_sym else grouped_value * value_scale + value_bias
        value_cache = value_cache.reshape(args.batch_size, args.num_kv_heads, 1, -1, args.head_dim)
    else:
        value_cache = value_float * value_scale if quantizer.use_sym else value_float * value_scale + value_bias
    if quantizer.use_hadamard:
        value_cache = inverse_hadamard_value(quantizer, value_cache, args)
    if quantizer.use_shuffle:
        value_cache = value_cache.index_select(-1, quantizer.unshuffle_idx)
    if quantizer.is_rotary:
        value_cache = inverse_rotate_value(quantizer, value_cache, args.batch_size)
    return value_cache


def decode_case(
    case: QuantCase,
    quantizer: exporter.KVQuantizer | None,
    encoded: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    if case.kv_dtype in {"F32", "F16"}:
        return encoded[0].float(), encoded[1].float()
    if quantizer is None:
        raise RuntimeError(f"No quantizer created for {case.name}.")
    if case.use_sym:
        packed_key, key_scale, packed_value, value_scale = encoded
        key_bias = None
        value_bias = None
    else:
        packed_key, key_scale, key_bias, packed_value, value_scale, value_bias = encoded
    key_cache = dequantize_key(case, quantizer, packed_key, key_scale, key_bias, args)
    value_cache = dequantize_value(case, quantizer, packed_value, value_scale, value_bias, args)
    return key_cache, value_cache


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def accuracy(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (actual.float() - reference.float()).reshape(-1)
    ref = reference.float().reshape(-1)
    count = diff.numel()
    abs_diff = diff.abs()
    mae = abs_diff.mean().item()
    rmse = math.sqrt(diff.square().sum().item() / count)
    max_abs = abs_diff.max().item()
    ref_rms = math.sqrt(ref.square().sum().item() / count)
    sqnr_db = float("inf") if rmse == 0.0 else 20.0 * math.log10(max(ref_rms, 1.0e-30) / rmse)
    return mae, rmse, max_abs, sqnr_db


def combined_accuracy(
    key_ref: torch.Tensor,
    key_actual: torch.Tensor,
    value_ref: torch.Tensor,
    value_actual: torch.Tensor,
) -> tuple[float, float, float, float]:
    diff_key = (key_actual.float() - key_ref.float()).reshape(-1)
    diff_value = (value_actual.float() - value_ref.float()).reshape(-1)
    ref_key = key_ref.float().reshape(-1)
    ref_value = value_ref.float().reshape(-1)
    count = diff_key.numel() + diff_value.numel()
    abs_sum = diff_key.abs().sum().item() + diff_value.abs().sum().item()
    square_sum = diff_key.square().sum().item() + diff_value.square().sum().item()
    ref_square_sum = ref_key.square().sum().item() + ref_value.square().sum().item()
    max_abs = max(diff_key.abs().max().item(), diff_value.abs().max().item())
    mae = abs_sum / count
    rmse = math.sqrt(square_sum / count)
    ref_rms = math.sqrt(ref_square_sum / count)
    sqnr_db = float("inf") if rmse == 0.0 else 20.0 * math.log10(max(ref_rms, 1.0e-30) / rmse)
    return mae, rmse, max_abs, sqnr_db


def mean_ms(values: list[float]) -> float:
    return statistics.fmean(values) * 1000.0


def stdev_ms(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) * 1000.0


def benchmark_case(
    case: QuantCase,
    args: argparse.Namespace,
    device: torch.device,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> BenchmarkResult:
    effective_group_size, notes = configure_exporter(case, args.head_dim)
    quantizer = build_quantizer(case, args, device)
    encode_times: list[float] = []
    decode_times: list[float] = []

    encoded: tuple[torch.Tensor, ...] | None = None
    decoded_key: torch.Tensor | None = None
    decoded_value: torch.Tensor | None = None

    with torch.inference_mode():
        for _ in range(args.warmup):
            encoded = encode_case(case, quantizer, key_cache, value_cache, args)
            decoded_key, decoded_value = decode_case(case, quantizer, encoded, args)
        synchronize(device)

        for _ in range(args.repeats):
            start = time.perf_counter()
            encoded = encode_case(case, quantizer, key_cache, value_cache, args)
            synchronize(device)
            encode_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            decoded_key, decoded_value = decode_case(case, quantizer, encoded, args)
            synchronize(device)
            decode_times.append(time.perf_counter() - start)

    if encoded is None or decoded_key is None or decoded_value is None:
        raise RuntimeError(f"No benchmark iteration ran for {case.name}; increase --repeats or --warmup.")

    sample_elements = key_cache.numel() + value_cache.numel()
    sample_fp32_bytes = sample_elements * 4
    sample_storage_bytes = tensor_bytes(encoded)
    compression_ratio = sample_fp32_bytes / sample_storage_bytes
    assumed_fp32_bytes = args.kv_elements * 4
    assumed_storage_bytes = int(round(assumed_fp32_bytes / compression_ratio))

    key_mae, key_rmse, key_max_abs, key_sqnr_db = accuracy(key_cache, decoded_key)
    value_mae, value_rmse, value_max_abs, value_sqnr_db = accuracy(value_cache, decoded_value)
    combined_mae, combined_rmse, combined_max_abs, combined_sqnr_db = combined_accuracy(
        key_cache,
        decoded_key,
        value_cache,
        decoded_value,
    )

    return BenchmarkResult(
        case=asdict(case),
        effective_group_size=effective_group_size,
        notes=notes,
        sample_elements=sample_elements,
        sample_storage_bytes=sample_storage_bytes,
        sample_fp32_bytes=sample_fp32_bytes,
        compression_ratio=compression_ratio,
        assumed_fp32_bytes=assumed_fp32_bytes,
        assumed_storage_bytes=assumed_storage_bytes,
        encode_ms_mean=mean_ms(encode_times),
        encode_ms_stdev=stdev_ms(encode_times),
        decode_ms_mean=mean_ms(decode_times),
        decode_ms_stdev=stdev_ms(decode_times),
        total_ms_mean=mean_ms([enc + dec for enc, dec in zip(encode_times, decode_times)]),
        key_mae=key_mae,
        key_rmse=key_rmse,
        key_max_abs=key_max_abs,
        key_sqnr_db=key_sqnr_db,
        value_mae=value_mae,
        value_rmse=value_rmse,
        value_max_abs=value_max_abs,
        value_sqnr_db=value_sqnr_db,
        combined_mae=combined_mae,
        combined_rmse=combined_rmse,
        combined_max_abs=combined_max_abs,
        combined_sqnr_db=combined_sqnr_db,
    )


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} TiB"


def fmt_float(value: float, digits: int = 4) -> str:
    if math.isinf(value):
        return "inf"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "case",
        "dtype",
        "g",
        "sym",
        "settings",
        "SQNR dB",
        "RMSE",
        "max abs",
        "enc ms",
        "dec ms",
        "total ms",
        "compress",
        "2B storage",
    ]
    rows: list[list[str]] = []
    for result in results:
        case = result.case
        settings = []
        if case["compute_in_f32"]:
            settings.append("compute_f32")
        if case["use_hadamard"]:
            settings.append("hadamard")
        if case["use_shuffle"]:
            settings.append("shuffle")
        if case["use_clip"]:
            settings.append(f"clip{case['clip_sigma']:g}")
        if not case["use_float16_scale_bias"]:
            settings.append("scale32")
        if case["use_qdq_friendly_asym"]:
            settings.append("qdq_asym")
        rows.append([
            case["name"],
            case["kv_dtype"],
            str(result.effective_group_size),
            "Y" if case["use_sym"] else "N",
            "+".join(settings) if settings else "-",
            fmt_float(result.combined_sqnr_db, 2),
            fmt_float(result.combined_rmse),
            fmt_float(result.combined_max_abs),
            fmt_float(result.encode_ms_mean, 3),
            fmt_float(result.decode_ms_mean, 3),
            fmt_float(result.total_ms_mean, 3),
            f"{result.compression_ratio:.2f}x",
            format_bytes(result.assumed_storage_bytes),
        ])
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def write_json(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for result in results:
        row = asdict(result)
        case = row.pop("case")
        row.update({f"case_{key}": value for key, value in case.items()})
        row["notes"] = " | ".join(row["notes"])
        flat_rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.batch_size < 1 or args.num_attention_heads < 1 or args.num_kv_heads < 1:
        raise ValueError("Batch and head counts must be positive.")
    if args.num_attention_heads % args.num_kv_heads != 0:
        raise ValueError("--num-attention-heads must be divisible by --num-kv-heads.")
    if args.head_dim < 1 or args.seq_len < 1:
        raise ValueError("--head-dim and --seq-len must be positive.")
    if args.head_dim % 8 != 0:
        raise ValueError("--head-dim must be divisible by 8 so every supported packed format can run.")
    try:
        parsed_group_sizes = [int(part.strip()) for part in args.group_sizes.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--group-sizes must be a comma-separated list of positive integers.") from exc
    if not parsed_group_sizes:
        raise ValueError("--group-sizes must contain at least one positive integer.")
    if any(group_size < 1 for group_size in parsed_group_sizes):
        raise ValueError("--group-sizes values must be positive.")
    if args.kv_elements < 1:
        raise ValueError("--kv-elements must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)
    cases = build_cases(args)
    if args.list_cases:
        for case in cases:
            print(case.name)
        return

    key_cache, value_cache = make_inputs(args, device)
    print(
        f"Sample KV: batch={args.batch_size}, kv_heads={args.num_kv_heads}, "
        f"head_dim={args.head_dim}, seq_len={args.seq_len}, device={device}"
    )
    print(
        f"Assumed KV cache: {args.kv_elements:,} float32 elements = "
        f"{format_bytes(args.kv_elements * 4)} baseline"
    )
    print(f"Running {len(cases)} case(s), repeats={args.repeats}, warmup={args.warmup}\n")

    results = []
    active_group_sizes = sorted({case.group_size for case in cases if case.kv_dtype in QUANTIZED_DTYPES}, reverse=True)
    if active_group_sizes:
        print(f"KV group sizes: {', '.join(str(group_size) for group_size in active_group_sizes)}")
    for case in cases:
        result = benchmark_case(case, args, device, key_cache, value_cache)
        results.append(result)
        print(f"done: {case.name}")

    print()
    print_table(results)

    if args.json_out is not None:
        write_json(args.json_out, results)
        print(f"\nWrote JSON: {args.json_out}")
    if args.csv_out is not None:
        write_csv(args.csv_out, results)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()