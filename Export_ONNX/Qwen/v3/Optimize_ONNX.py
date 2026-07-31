"""Optimize Qwen v3 ONNX modules and rebuild the shared merged bundle."""

import os
import gc
import sys
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx
import onnx.version_converter
from onnx.external_data_helper import load_external_data_for_model
from onnx import TensorProto, helper, numpy_helper
from onnxslim import slim
from transformers import AutoConfig
from onnxruntime.quantization import (
    QuantType,
    matmul_nbits_quantizer,
    quant_utils,
    quantize_dynamic,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import Shared_Merged  # noqa: E402 - local sibling import follows the script-directory path setup


# User config
ORIGINAL_FOLDER_PATH           = str(_SCRIPT_DIR / "Qwen_ONNX")         # Folder holding the exported *.onnx modules.
QUANTED_FOLDER_PATH            = str(_SCRIPT_DIR / "Qwen_Optimized")    # Destination folder.
DOWNLOAD_PATH                  = str(Path.home() / "Downloads" / "Qwen3-0.6B")  # Model dir (attention fusion); "NONE" to skip.

QUANT_METHOD                   = "Q4"                                # "Q2" | "Q4" | "Q8" | "F16" | "F32" | "DYNAMIC".
WEIGHT_ONLY_ALGORITHM          = "AFFINE_REFINE_V2"                  # "AFFINE_REFINE_V2" | "DEFAULT" | "RTN" | "HQQ" | "k_quant". AFFINE_REFINE_V2 supports Q4/Q8 and dynamic INT8/UINT8; k_quant/RTN are Q4-only.
BLOCK_SIZE                     = 32                                  # Power of two, [16..256].
ACCURACY_LEVEL                 = 4                                   # MatMulNBits compute: 1=FP32 (best accuracy), 2=FP16, 3=BF16, 4=INT8 (fastest), 0=Default.
QUANT_SYMMETRIC                = False                               # False = asymmetric (accuracy), True = symmetric (speed).
QUANT_FORMAT                   = "QOperator"                         # "QOperator" (MatMulNBits op) | "QDQ" (DEFAULT algo + 4-bit only).

AFFINE_V2_SEED_ITERATIONS      = 4                                   # Magnitude-weighted seed coordinate-descent passes.
AFFINE_V2_SEED_ZP_RADIUS       = 2                                   # Search the ORT k-quant zero point +/- this radius.
AFFINE_V2_SEED_CHUNK_BLOCKS    = 65536                               # Peak-memory bound for seed construction.
AFFINE_V2_SEED_BLOCKS_PER_JOB  = 1024                                # Keep small seeds single-threaded to avoid dispatch overhead.
AFFINE_V2_SEED_WORKERS         = 4
AFFINE_V2_NUMBA_THREADS        = 4

# AFFINE_REFINE_V2 minimizes plain (unweighted) block MSE. For the data-free
# maximum-entropy prior of white/diagonal-covariance activations, the expected
# projection error equals the Frobenius weight error. A per-block Pareto guard
# keeps magnitude-weighted error within a bounded fraction of the internal seed
# so no block's largest weights degrade badly.
AFFINE_V2_ITERATIONS           = 6                                   # Plain-MSE scale/code alternating passes per start.
AFFINE_V2_CLIP_RATIOS          = (1.0, 0.94, 0.82, 0.70, 0.55)       # Deterministic range starts for every integer zero point.
AFFINE_V2_CHUNK_BLOCKS         = 8192                                # Bounds temporary fitting arrays independently of model size.
AFFINE_V2_WEIGHTED_TOLERANCE   = 0.15                                # Pareto guard: max fractional magnitude-weighted regression vs the internal seed.
# The asymmetric main-refine sweep evaluates every integer zero point. That is 16
# candidates at Q4 but 256 at Q8, which is intractable across a full model. When
# the candidate count exceeds this limit (Q8 only), sweep a window of this many
# zero points centered on each block's near-optimal k-quant seed instead of the
# whole range. Must be >= 16 so Q2/Q4 always sweep fully and stay byte-identical.
AFFINE_V2_ASYM_ZP_SWEEP_LIMIT  = 32

DYNAMIC_WEIGHT_TYPE            = "QInt8"                             # "QUInt8" | "QInt8".
DYNAMIC_PER_CHANNEL            = True                                # Per-channel weights (accuracy, slower).
DYNAMIC_REDUCE_RANGE           = False                               # 7-bit weights; can help on non-VNNI CPUs.

NODES_TO_EXCLUDE               = None                                # Node names to keep unquantized, or None.
NODES_TO_INCLUDE               = None                                # Node names to quantize exclusively, or None.
FAIR_Q4_MATMUL_ONLY            = False                               # Force identical MatMul-only coverage for quantizer comparisons.

FORCE_EXTERNAL_DATA            = False                               # Two-part storage (*.onnx.data); auto-forced when >2GB.
UPGRADE_OPSET                  = 0                                   # Target ONNX opset (0 = keep current).

OPTIMIZER_LEVEL                = 2                                   # ORT graph optimization level: 0 | 1 | 2 | 99.
OPTIMIZER_MODEL_TYPE           = "bert"                              # Fusion template; "bert" is a safe generic choice.
OPTIMIZER_ONLY_ONNXRUNTIME     = False                               # True = only ORT's built-in optimizer (skip Python fusions).
OPTIMIZER_FUSION_OPTIONS       = None                                # Optional dict of FusionOptions overrides, e.g. {"enable_gelu": False}.
SHAPE_INFER                    = True                                # Run shape inference before the optimizer (needed for some fusions).

# EliminationReshape is unsafe when consecutive reshapes use zero-copy dimensions
# across a rank change: it applies the second shape directly to the first input.
SLIM_SKIP_FUSION_PATTERNS      = ["EliminationReshape"]              # Additional fusion patterns to skip, or None.
SLIM_SKIP_OPTIMIZATIONS        = None                                # Optimizations to skip, or None.
SLIM_SIZE_THRESHOLD            = None                                # Max constant size (bytes) to fold; None = fold all.

F16_KEEP_IO_TYPES              = None                                # None = auto (keep fp32 I/O only when precisions are mixed).
F16_FORCE_INITIALIZERS         = True                                # Cast float initializers to float16.
F16_MIN_POSITIVE_VAL           = 1e-7                                # Clamp floor for tiny positive values.
F16_MAX_FINITE_VAL             = 32767.0                             # Clamp ceiling for large finite values.
F16_NODE_BLOCK_LIST            = None                                # Node names forced to stay float32, or None.
F16_OP_BLOCK_LIST              = [                                   # Op types kept out of the float16 conversion.
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
]

KV_ATTENTION_SURGERY           = "auto"                              # "auto" (enable when quantized KV detected) | True | False.
# ORT 1.27's CUDA EP rejects ONNX blocked QuantizeLinear/DequantizeLinear at
# execution time ("Unsupported quantization type"). Keep the original
# Div -> Round -> Clip -> Cast KV write/rope-shift tails for portable CPU/CUDA
# graphs. Enable only for a CPU-only runtime known to support blocked Q/DQ.
KV_BLOCKED_QDQ_SURGERY         = True


@dataclass
class Plan:
    """Per-module recipe. None inherits the USER CONFIG default."""
    method:              str                    = "Q4"     # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
    # weight-only (Q2/Q4/Q8)
    algo:                str  | None            = None     # AFFINE_REFINE_V2 | DEFAULT | RTN | HQQ | k_quant
    op_types:            tuple[str, ...] | None = None     # e.g. ("MatMul",) or ("Gather",)
    axes:                tuple[int, ...] | None = None     # quant axis per op type
    block_size:          int  | None            = None
    accuracy_level:      int  | None            = None     # MatMulNBits accuracy level (DEFAULT algo)
    symmetric:           bool | None            = None
    quant_format:        str  | None            = None     # QOperator | QDQ
    # dynamic INT8
    dynamic_weight_type: str  | None            = None     # QUInt8 | QInt8
    per_channel:         bool | None            = None
    reduce_range:        bool | None            = None
    # node selection
    nodes_to_exclude:    list[str] | None       = None
    nodes_to_include:    list[str] | None       = None
    # optimize / precision (used as-is, no global)
    optimize:            bool                   = True
    fp16:                bool                   = False
    # surgery (None inherits global; "auto" | True | False)
    kv_surgery:          bool | str | None      = None     # quantized int KV
    # storage
    external:            bool | None            = None     # None inherits; auto-forced when >2GB


_PRIMARY_MERGED_MODEL = Path(Shared_Merged.PREFILL_GREEDY_MODEL_NAME).stem
_MERGED_MODEL_NAMES = tuple(Path(name).stem for name, _, _ in Shared_Merged.MERGED_BUILD_PLAN)


def _main_quantization_ops(
    method: str,
    algorithm: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if FAIR_Q4_MATMUL_ONLY:
        return ("MatMul",), (0,)
    # ORT's built-in Gather quantization is 4-bit only, but AFFINE_REFINE_V2 emits
    # its own GatherBlockQuantized ops and supports both 4- and 8-bit. The k_quant,
    # RTN, and HQQ backends quantize MatMul only. Bundle finalization shares the
    # embedding with lm_head only after exact source-weight equality is verified.
    if algorithm == "AFFINE_REFINE_V2":
        if method in ("Q4", "Q8"):
            return ("MatMul", "Gather"), (0, 1)
        return ("MatMul",), (0,)
    if method != "Q4" or algorithm in ("k_quant", "RTN", "HQQ"):
        return ("MatMul",), (0,)
    return ("MatMul", "Gather"), (0, 1)


_MAIN_QUANT_OP_TYPES, _MAIN_QUANT_AXES = _main_quantization_ops(
    QUANT_METHOD,
    WEIGHT_ONLY_ALGORITHM,
)


MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata":        Plan(method="F32", optimize=False),
    # Quantize Main once, then transplant that Main block into the other merged shells.
    _PRIMARY_MERGED_MODEL: Plan(
        method=QUANT_METHOD, external=True, optimize=True,
        op_types=_MAIN_QUANT_OP_TYPES, axes=_MAIN_QUANT_AXES,
    ),
    "LLM_KV_Slice":        Plan(method="F32"),
    "LLM_KV_Split2":       Plan(method="F32"),
    "LLM_KV_Concat":       Plan(method="F32"),
    "LLM_RopeShift":       Plan(method="F32"),
}


_WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}     # method -> weight-only bit width
_QUANT_FORMATS = {
    "QOPERATOR": quant_utils.QuantFormat.QOperator,
    "QDQ": quant_utils.QuantFormat.QDQ,
}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_WEIGHT_ONLY_ALGO_BITS = {
    "DEFAULT": frozenset(_WEIGHT_ONLY_BITS.values()),
    "HQQ": frozenset(_WEIGHT_ONLY_BITS.values()),
    # ORT routes RTN and k_quant through _generate_q4_node_config(), which hard-codes bits=4.
    "AFFINE_REFINE_V2": frozenset({4, 8}),
    "RTN": frozenset({4}),
    "k_quant": frozenset({4}),
}
_VALID_ALGOS = set(_WEIGHT_ONLY_ALGO_BITS)


@dataclass
class ResolvedPlan:
    method:              str
    algo:                str
    op_types:            tuple[str, ...]
    axes:                tuple[int, ...]
    block_size:          int
    accuracy_level:      int
    symmetric:           bool
    quant_format:        str
    dynamic_weight_type: str
    per_channel:         bool
    reduce_range:        bool
    nodes_to_exclude:    list[str] | None
    nodes_to_include:    list[str] | None
    optimize:            bool
    fp16:                bool
    kv_surgery:          bool | str
    external:            bool


def _pick(value, default):
    return default if value is None else value


def resolve_plan(plan: Plan) -> ResolvedPlan:
    return ResolvedPlan(
        method=plan.method.upper(),
        algo=_pick(plan.algo, WEIGHT_ONLY_ALGORITHM),
        op_types=_pick(plan.op_types, ("MatMul",)),
        axes=_pick(plan.axes, (0,)),
        block_size=_pick(plan.block_size, BLOCK_SIZE),
        accuracy_level=_pick(plan.accuracy_level, ACCURACY_LEVEL),
        symmetric=_pick(plan.symmetric, QUANT_SYMMETRIC),
        quant_format=_pick(plan.quant_format, QUANT_FORMAT).upper(),
        dynamic_weight_type=_pick(plan.dynamic_weight_type, DYNAMIC_WEIGHT_TYPE).upper(),
        per_channel=_pick(plan.per_channel, DYNAMIC_PER_CHANNEL),
        reduce_range=_pick(plan.reduce_range, DYNAMIC_REDUCE_RANGE),
        nodes_to_exclude=_pick(plan.nodes_to_exclude, NODES_TO_EXCLUDE),
        nodes_to_include=_pick(plan.nodes_to_include, NODES_TO_INCLUDE),
        optimize=plan.optimize,
        fp16=plan.fp16,
        kv_surgery=_pick(plan.kv_surgery, KV_ATTENTION_SURGERY),
        external=_pick(plan.external, FORCE_EXTERNAL_DATA),
    )


def validate_plan(name: str, rp: ResolvedPlan) -> None:
    valid_methods = set(_WEIGHT_ONLY_BITS) | {"DYNAMIC", "F16", "F32"}
    if rp.method not in valid_methods:
        raise ValueError(f"[{name}] unknown method {rp.method!r}; choose one of {sorted(valid_methods)}.")

    if rp.kv_surgery not in ("auto", True, False):
        raise ValueError(f"[{name}] kv_surgery must be 'auto', True, or False (got {rp.kv_surgery!r}).")

    if rp.method in _WEIGHT_ONLY_BITS:
        bits = _WEIGHT_ONLY_BITS[rp.method]
        if rp.algo not in _VALID_ALGOS:
            raise ValueError(f"[{name}] unknown algo {rp.algo!r}; choose one of {sorted(_VALID_ALGOS)}.")
        if bits not in _WEIGHT_ONLY_ALGO_BITS[rp.algo]:
            compatible = sorted(
                algo for algo, supported_bits in _WEIGHT_ONLY_ALGO_BITS.items()
                if bits in supported_bits
            )
            raise ValueError(
                f"[{name}] algo={rp.algo!r} cannot produce {bits}-bit weights; its ORT backend "
                f"emits 4-bit only. Use one of {compatible} for method={rp.method!r}."
            )
        if rp.quant_format not in _QUANT_FORMATS:
            raise ValueError(f"[{name}] unknown quant_format; choose 'QOperator' or 'QDQ'.")
        if len(rp.op_types) != len(rp.axes):
            raise ValueError(f"[{name}] op_types {rp.op_types} and axes {rp.axes} must have equal length.")
        if "Gather" in rp.op_types and rp.algo not in ("DEFAULT", "AFFINE_REFINE_V2"):
            raise ValueError(
                f"[{name}] Gather quantization requires algo='DEFAULT' or "
                f"'AFFINE_REFINE_V2' (got {rp.algo!r})."
            )
        if rp.quant_format == "QDQ" and (rp.algo != "DEFAULT" or bits != 4):
            raise ValueError(
                f"[{name}] QDQ format supports only algo='DEFAULT' with 4-bit (got {rp.algo!r}, {bits}-bit)."
            )
        # AFFINE_REFINE_V2 supports both asymmetric (learned per-block zero point) and
        # symmetric (zero point fixed at the 4-bit midpoint 8) Q4.

    if rp.method == "DYNAMIC":
        if rp.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
            raise ValueError(f"[{name}] unknown dynamic_weight_type; choose 'QUInt8' or 'QInt8'.")
        if rp.algo == "AFFINE_REFINE_V2" and any(op_type != "MatMul" for op_type in rp.op_types):
            raise ValueError(
                f"[{name}] AFFINE_REFINE_V2 dynamic quantization supports MatMul only "
                f"(got {rp.op_types})."
            )


@dataclass
class Q4RefineStats:
    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "Q4RefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


_K_QUANT_SEARCH_OFFSETS = np.asarray(
    tuple(-1.0 + 0.1 * index for index in range(20)), dtype=np.float32
)
_K_QUANT_FINAL_CHUNK_VALUES = 262144


def quant_tensor_k_quant_cpu(
    data: np.ndarray,
    num_bits: int = 4,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize rows with ORT's k-quant objective using reusable float32 buffers."""
    if num_bits < 1:
        raise ValueError(f"num_bits must be positive, got {num_bits}.")
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    values = np.asarray(data)
    values = np.ascontiguousarray(values.reshape(-1, group_size), dtype=np.float32)
    block_count = values.shape[0]
    maxq = (1 << num_bits) - 1
    maxq_float = np.float32(maxq)

    quantized = np.empty_like(values)
    scratch = np.empty_like(values)
    weighted_quantized = np.empty_like(values)

    np.multiply(values, values, out=scratch)
    rms = np.sqrt(
        np.sum(scratch, axis=1, dtype=np.float32) / np.float32(group_size)
    )
    weights = np.empty_like(values)
    np.abs(values, out=weights)
    np.add(weights, rms[:, None], out=weights)

    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    span = maximum - minimum
    sum_weights = np.sum(weights, axis=1, dtype=np.float32)
    np.multiply(weights, values, out=scratch)
    sum_weighted_values = np.sum(scratch, axis=1, dtype=np.float32)

    inverse_scale = np.ones(block_count, dtype=np.float32)
    varying = span != 0.0
    np.divide(maxq_float, span, out=inverse_scale, where=varying)
    best_scale = np.reciprocal(inverse_scale)
    best_minimum = minimum.copy()

    np.subtract(values, best_minimum[:, None], out=scratch)
    np.multiply(scratch, inverse_scale[:, None], out=scratch)
    np.rint(scratch, out=quantized)
    np.clip(quantized, 0.0, maxq_float, out=quantized)
    np.multiply(quantized, best_scale[:, None], out=scratch)
    np.add(scratch, best_minimum[:, None], out=scratch)
    np.subtract(scratch, values, out=scratch)
    np.square(scratch, out=scratch)
    np.multiply(scratch, weights, out=scratch)
    best_error = np.sum(scratch, axis=1, dtype=np.float32)

    candidate_inverse_scale = np.empty(block_count, dtype=np.float32)
    sum_l = np.empty(block_count, dtype=np.float32)
    sum_l2 = np.empty(block_count, dtype=np.float32)
    sum_xl = np.empty(block_count, dtype=np.float32)
    determinant = np.empty(block_count, dtype=np.float32)
    numerator = np.empty(block_count, dtype=np.float32)
    row_scratch = np.empty(block_count, dtype=np.float32)
    candidate_scale = np.empty(block_count, dtype=np.float32)
    candidate_minimum = np.empty(block_count, dtype=np.float32)
    candidate_error = np.empty(block_count, dtype=np.float32)
    valid = np.empty(block_count, dtype=bool)
    improved = np.empty(block_count, dtype=bool)

    # The upstream helper stores winning codes but discards them during its final
    # requantization. Track only the winning affine parameters here.
    for offset in _K_QUANT_SEARCH_OFFSETS:
        np.subtract(maximum, best_minimum, out=span)
        np.not_equal(span, 0.0, out=valid)
        candidate_inverse_scale.fill(1.0)
        np.divide(
            maxq_float + offset,
            span,
            out=candidate_inverse_scale,
            where=valid,
        )

        np.subtract(values, best_minimum[:, None], out=scratch)
        np.multiply(scratch, candidate_inverse_scale[:, None], out=scratch)
        np.rint(scratch, out=quantized)
        np.clip(quantized, 0.0, maxq_float, out=quantized)

        np.multiply(weights, quantized, out=weighted_quantized)
        np.sum(weighted_quantized, axis=1, dtype=np.float32, out=sum_l)
        np.multiply(weighted_quantized, quantized, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=sum_l2)
        np.multiply(weighted_quantized, values, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=sum_xl)

        np.multiply(sum_weights, sum_l2, out=determinant)
        np.multiply(sum_l, sum_l, out=row_scratch)
        np.subtract(determinant, row_scratch, out=determinant)
        np.not_equal(determinant, 0.0, out=valid)
        np.logical_and(valid, np.isfinite(determinant), out=valid)

        np.multiply(sum_weights, sum_xl, out=numerator)
        np.multiply(sum_weighted_values, sum_l, out=row_scratch)
        np.subtract(numerator, row_scratch, out=numerator)
        candidate_scale.fill(0.0)
        np.divide(numerator, determinant, out=candidate_scale, where=valid)

        np.multiply(sum_l2, sum_weighted_values, out=numerator)
        np.multiply(sum_l, sum_xl, out=row_scratch)
        np.subtract(numerator, row_scratch, out=numerator)
        candidate_minimum.fill(0.0)
        np.divide(numerator, determinant, out=candidate_minimum, where=valid)
        np.logical_and(valid, np.isfinite(candidate_scale), out=valid)
        np.logical_and(valid, candidate_scale > 0.0, out=valid)
        np.logical_and(valid, np.isfinite(candidate_minimum), out=valid)

        np.multiply(quantized, candidate_scale[:, None], out=scratch)
        np.add(scratch, candidate_minimum[:, None], out=scratch)
        np.subtract(scratch, values, out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, weights, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=candidate_error)
        np.less(candidate_error, best_error, out=improved)
        np.logical_and(improved, valid, out=improved)
        np.copyto(best_error, candidate_error, where=improved)
        np.copyto(best_scale, candidate_scale, where=improved)
        np.copyto(best_minimum, candidate_minimum, where=improved)

    zero_point_float = np.empty(block_count, dtype=np.float32)
    np.negative(best_minimum, out=zero_point_float)
    np.divide(zero_point_float, best_scale, out=zero_point_float)
    np.rint(zero_point_float, out=zero_point_float)
    np.clip(zero_point_float, 0.0, maxq_float, out=zero_point_float)
    zero_point = zero_point_float.astype(np.uint8).reshape(-1, 1)

    del scratch, weighted_quantized, weights
    rows_per_chunk = max(1, _K_QUANT_FINAL_CHUNK_VALUES // group_size)
    final_buffer = np.empty(
        (min(block_count, rows_per_chunk), group_size), dtype=np.float64
    )
    for start in range(0, block_count, rows_per_chunk):
        end = min(start + rows_per_chunk, block_count)
        final = final_buffer[: end - start]
        np.divide(values[start:end], best_scale[start:end, None], out=final)
        np.add(final, zero_point[start:end], out=final)
        np.rint(final, out=final)
        np.clip(final, 0.0, float(maxq), out=final)
        quantized[start:end] = final
    return quantized, best_scale.reshape(-1, 1), zero_point


@lru_cache(maxsize=1)
def _affine_v2_seed_quantizer():
    # V2 is deliberately CPU-only so quantization is reproducible across hosts
    # and never changes implementation based on accelerator availability.
    return quant_tensor_k_quant_cpu


@lru_cache(maxsize=1)
def _affine_v2_seed_executor():
    return ThreadPoolExecutor(
        max_workers=AFFINE_V2_SEED_WORKERS,
        thread_name_prefix="q4-seed",
    )


@lru_cache(maxsize=1)
def _affine_v2_seed_pipeline_executor():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="q4-v2-pipeline")


def _quantize_affine_v2_seed_partition(weight: np.ndarray, block_size: int, bits: int = 4):
    with np.errstate(divide="ignore", invalid="ignore"):
        return quant_tensor_k_quant_cpu(weight, bits, block_size)


def _quantize_affine_v2_seed_blocks(
    weight: np.ndarray,
    block_size: int,
    bits: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if AFFINE_V2_SEED_WORKERS < 1:
        raise ValueError("AFFINE_REFINE_V2 seed worker count must be positive.")
    maxq = float((1 << bits) - 1)
    quantizer = _affine_v2_seed_quantizer()
    max_workers = max(1, min(AFFINE_V2_SEED_WORKERS, os.cpu_count() or 1))
    worker_count = min(max_workers, max(1, weight.shape[0] // AFFINE_V2_SEED_BLOCKS_PER_JOB))
    if quantizer is not quant_tensor_k_quant_cpu or worker_count == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            varying_q, varying_scale, varying_zp = quantizer(weight, bits, block_size)
        return (
            np.clip(np.asarray(varying_q, dtype=np.float32), 0.0, maxq),
            np.asarray(varying_scale, dtype=np.float32).reshape(-1, 1),
            np.clip(np.asarray(varying_zp, dtype=np.int16).reshape(-1, 1), 0, int(maxq)).astype(np.uint8),
        )

    partitions = np.array_split(weight, worker_count, axis=0)
    futures = [
        _affine_v2_seed_executor().submit(
            _quantize_affine_v2_seed_partition, partition, block_size, bits
        )
        for partition in partitions
    ]
    seed_q = np.empty(weight.shape, dtype=np.float32)
    seed_scale = np.empty((weight.shape[0], 1), dtype=np.float32)
    seed_zp = np.empty((weight.shape[0], 1), dtype=np.uint8)
    offset = 0
    for partition, future in zip(partitions, futures):
        varying_q, varying_scale, varying_zp = future.result()
        end = offset + partition.shape[0]
        seed_q[offset:end] = np.clip(np.asarray(varying_q, dtype=np.float32), 0.0, maxq)
        seed_scale[offset:end] = np.asarray(varying_scale, dtype=np.float32).reshape(-1, 1)
        seed_zp[offset:end] = np.clip(
            np.asarray(varying_zp, dtype=np.int16).reshape(-1, 1), 0, int(maxq)
        ).astype(np.uint8)
        offset = end
    return seed_q, seed_scale, seed_zp


def _iter_q4_row_chunks(values: np.ndarray, block_size: int, max_blocks: int):
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    padded_columns = block_count * block_size
    rows_per_chunk = max(1, max_blocks // block_count)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = np.ascontiguousarray(values[row_start:row_end], dtype=np.float32)
        if padded_columns != columns:
            chunk = np.pad(chunk, ((0, 0), (0, padded_columns - columns)), mode="constant")
        block_start = row_start * block_count
        block_end = row_end * block_count
        yield block_start, block_end, chunk.reshape(-1, block_size)


def _affine_v2_seed_blocks(
    weight: np.ndarray,
    block_size: int,
    symmetric: bool = False,
    bits: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the raw k-quant seed, including deterministic constant-block handling."""
    maxq = float((1 << bits) - 1)
    midpoint = float(1 << (bits - 1))
    if symmetric:
        # Symmetric seed: pin the zero point at the integer midpoint so codes span
        # [-midpoint, maxq - midpoint]; cover the positive range with /(maxq - midpoint)
        # and the negative range with /midpoint. The refinement then only fits the
        # scale -- the zero point never moves.
        tiny = np.finfo(np.float32).tiny
        positive_max = np.maximum(weight.max(axis=1, keepdims=True), np.float32(0.0))
        negative_max = np.maximum(-weight.min(axis=1, keepdims=True), np.float32(0.0))
        seed_scale = np.maximum(
            positive_max / np.float32(maxq - midpoint), negative_max / np.float32(midpoint)
        )
        seed_scale = np.where(seed_scale > tiny, seed_scale, np.float32(1.0)).astype(np.float32)
        seed_q = np.clip(
            np.rint(weight / seed_scale + np.float32(midpoint)), 0.0, maxq
        ).astype(np.float32)
        seed_zp = np.full((weight.shape[0], 1), int(midpoint), dtype=np.uint8)
        return seed_q, seed_scale, seed_zp

    constant = np.ptp(weight, axis=1) == 0.0
    has_constant = np.any(constant)
    if not has_constant:
        seed_q, seed_scale, seed_zp = _quantize_affine_v2_seed_blocks(weight, block_size, bits)
    else:
        seed_q = np.empty(weight.shape, dtype=np.float32)
        seed_scale = np.empty((weight.shape[0], 1), dtype=np.float32)
        seed_zp = np.empty((weight.shape[0], 1), dtype=np.uint8)
        varying = ~constant
        if np.any(varying):
            varying_q, varying_scale, varying_zp = _quantize_affine_v2_seed_blocks(
                weight[varying], block_size, bits
            )
            seed_q[varying] = varying_q
            seed_scale[varying] = varying_scale
            seed_zp[varying] = varying_zp
        constant_value = weight[constant, :1]
        positive = constant_value > 0.0
        negative = constant_value < 0.0
        seed_q[constant] = np.where(positive, np.float32(maxq), np.float32(0.0))
        seed_scale[constant] = np.where(
            positive,
            constant_value / np.float32(maxq),
            np.where(negative, -constant_value / np.float32(maxq), np.float32(1.0)),
        )
        seed_zp[constant] = np.where(negative, np.uint8(int(maxq)), np.uint8(0))

    tiny = np.finfo(np.float32).tiny
    valid_scale = np.isfinite(seed_scale) & (seed_scale > tiny)
    if not np.all(valid_scale):
        fallback_scale = (weight.max(axis=1, keepdims=True) - weight.min(axis=1, keepdims=True)) / np.float32(maxq)
        fallback_scale = np.where(fallback_scale > tiny, fallback_scale, np.float32(1.0))
        seed_scale = np.where(valid_scale, seed_scale, fallback_scale)
    return seed_q, seed_scale, seed_zp


def _affine_v2_seed_refine_q4_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool = False,
    bits: int = 4,
    allow_arbitrary_block_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Q4RefineStats]:
    """Build AFFINE_REFINE_V2's magnitude-weighted integer Q4/Q8 seed."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"AFFINE_REFINE_V2 seed expects a 2-D row matrix, got shape {values.shape}.")
    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError(
            f"AFFINE_REFINE_V2 seed block_size must be a power of two in [16, 256], got {block_size}."
        )
    if AFFINE_V2_SEED_ITERATIONS < 1 or AFFINE_V2_SEED_ZP_RADIUS < 0 or AFFINE_V2_SEED_CHUNK_BLOCKS < 1:
        raise ValueError(
            "AFFINE_REFINE_V2 seed iterations/chunk size must be positive and "
            "zero-point radius nonnegative."
        )
    if not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 seed refuses weights containing NaN or Inf.")

    maxq = float((1 << bits) - 1)
    midpoint = int(1 << (bits - 1))
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    quantized = np.empty((total_blocks, block_size), dtype=np.uint8)
    scales = np.empty((total_blocks, 1), dtype=np.float32)
    zero_points = np.empty((total_blocks, 1), dtype=np.uint8)
    stats = Q4RefineStats(blocks=total_blocks)
    tiny = np.finfo(np.float32).tiny

    for start, end, weight in _iter_q4_row_chunks(
        values, block_size, AFFINE_V2_SEED_CHUNK_BLOCKS
    ):
        seed_q, seed_scale, seed_zp = _affine_v2_seed_blocks(weight, block_size, symmetric, bits)

        importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        importance = np.where(importance.sum(axis=1, keepdims=True) > 0.0, importance, np.float32(1.0))
        seed_dequant = seed_scale * (seed_q - seed_zp.astype(np.float32))
        seed_error = np.sum(importance * (weight - seed_dequant) ** 2, axis=1, keepdims=True)

        best_q = seed_q.copy()
        best_scale = seed_scale.copy()
        best_zp = seed_zp.copy()
        best_error = seed_error.copy()

        for delta in ((0,) if symmetric else range(-AFFINE_V2_SEED_ZP_RADIUS, AFFINE_V2_SEED_ZP_RADIUS + 1)):
            candidate_zp = np.clip(seed_zp.astype(np.int16) + delta, 0, int(maxq)).astype(np.float32)
            candidate_scale = seed_scale.copy()
            for _ in range(AFFINE_V2_SEED_ITERATIONS):
                candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0.0, maxq)
                centered_q = candidate_q - candidate_zp
                denominator = np.sum(importance * centered_q * centered_q, axis=1, keepdims=True)
                numerator = np.sum(importance * centered_q * weight, axis=1, keepdims=True)
                fitted_scale = np.divide(
                    numerator,
                    denominator,
                    out=candidate_scale.copy(),
                    where=denominator > tiny,
                )
                candidate_scale = np.where(
                    np.isfinite(fitted_scale) & (fitted_scale > tiny), fitted_scale, candidate_scale
                )

            candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0.0, maxq)
            candidate_error = np.sum(
                importance * (weight - candidate_scale * (candidate_q - candidate_zp)) ** 2,
                axis=1,
                keepdims=True,
            )
            take = candidate_error[:, 0] < best_error[:, 0]
            best_q[take] = candidate_q[take]
            best_scale[take] = candidate_scale[take]
            best_zp[take] = candidate_zp[take].astype(np.uint8)
            best_error[take] = candidate_error[take]

        quantized[start:end] = best_q.astype(np.uint8)
        scales[start:end] = best_scale
        zero_points[start:end] = best_zp
        stats.improved_blocks += int(np.count_nonzero(best_error[:, 0] < seed_error[:, 0]))
        stats.seed_error += float(seed_error.sum(dtype=np.float64))
        stats.refined_error += float(best_error.sum(dtype=np.float64))

    return (
        quantized.reshape(rows, block_count, block_size),
        scales.reshape(rows, block_count),
        zero_points.reshape(rows, block_count),
        stats,
    )


@lru_cache(maxsize=1)
def _affine_v2_numba_kernel():
    """Build the optional fused CPU kernel lazily so ORT paths pay no import cost."""
    try:
        from numba import njit, prange, set_num_threads
    except ImportError:
        return None
    set_num_threads(AFFINE_V2_NUMBA_THREADS)

    @njit(parallel=True, nogil=True, cache=True)
    def refine_blocks(
        weight,
        quantized,
        scales,
        zero_points,
        clip_ratios,
        seed_iterations,
        seed_zp_radius,
        affine_iterations,
        tolerance,
        tiny,
        symmetric,
        maxq,
        midpoint,
        zp_sweep_limit,
    ):
        max_code = np.float32(maxq)
        max_code_int = int(maxq)
        block_count, width = weight.shape
        baseline_errors = np.empty(block_count, dtype=np.float32)
        refined_errors = np.empty(block_count, dtype=np.float32)
        improved = np.zeros(block_count, dtype=np.bool_)
        candidate_codes = np.empty((block_count, width), dtype=np.uint8)

        for block_index in prange(block_count):
            sum_squares = np.float32(0.0)
            positive_max = np.float32(0.0)
            negative_max = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                sum_squares += value * value
                if value > positive_max:
                    positive_max = value
                if -value > negative_max:
                    negative_max = -value
            rms = np.float32(np.sqrt(sum_squares / np.float32(width)))

            raw_scale = np.float32(scales[block_index])
            raw_zero_point_int = int(zero_points[block_index])
            raw_zero_point = np.float32(raw_zero_point_int)
            best_seed_error = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                centered = np.float32(quantized[block_index, column]) - raw_zero_point
                residual = value - raw_scale * centered
                importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                best_seed_error += importance * residual * residual

            for delta in range(0 if symmetric else -seed_zp_radius,
                               1 if symmetric else seed_zp_radius + 1):
                candidate_zero_point_int = min(max_code_int, max(0, raw_zero_point_int + delta))
                candidate_zero_point = np.float32(candidate_zero_point_int)
                candidate_scale = raw_scale
                for _ in range(seed_iterations):
                    denominator = np.float32(0.0)
                    numerator = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block_index, column])
                        candidate_q = np.rint(value / candidate_scale + candidate_zero_point)
                        candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                        centered = candidate_q - candidate_zero_point
                        importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                        denominator += importance * centered * centered
                        numerator += importance * centered * value
                    if denominator <= tiny:
                        break
                    fitted_scale = numerator / denominator
                    if not np.isfinite(fitted_scale) or fitted_scale <= tiny:
                        break
                    if fitted_scale == candidate_scale:
                        break
                    candidate_scale = fitted_scale

                candidate_seed_error = np.float32(0.0)
                for column in range(width):
                    value = np.float32(weight[block_index, column])
                    candidate_q = np.rint(value / candidate_scale + candidate_zero_point)
                    candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                    candidate_codes[block_index, column] = np.uint8(candidate_q)
                    centered = candidate_q - candidate_zero_point
                    residual = value - candidate_scale * centered
                    importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                    candidate_seed_error += importance * residual * residual
                if candidate_seed_error < best_seed_error:
                    best_seed_error = candidate_seed_error
                    scales[block_index] = candidate_scale
                    zero_points[block_index] = np.uint8(candidate_zero_point_int)
                    for column in range(width):
                        quantized[block_index, column] = candidate_codes[block_index, column]

            seed_scale = np.float32(scales[block_index])
            seed_zero_point = np.float32(zero_points[block_index])
            seed_zero_point_int = int(zero_points[block_index])
            baseline_plain = np.float32(0.0)
            baseline_weighted = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                centered = np.float32(quantized[block_index, column]) - seed_zero_point
                residual = value - seed_scale * centered
                squared = residual * residual
                baseline_plain += squared
                baseline_weighted += (rms + np.abs(value)) * residual * residual

            local_plain = baseline_plain
            weighted_bound = tolerance * baseline_weighted

            # Q4/Q2 (<= zp_sweep_limit candidates) sweep every zero point; Q8 sweeps
            # a window of zp_sweep_limit points centered on the block's k-quant seed.
            if symmetric:
                zp_lo = midpoint
                zp_hi = midpoint
            elif max_code_int + 1 <= zp_sweep_limit:
                zp_lo = 0
                zp_hi = max_code_int
            else:
                zp_lo = seed_zero_point_int - zp_sweep_limit // 2
                if zp_lo < 0:
                    zp_lo = 0
                zp_hi = zp_lo + zp_sweep_limit - 1
                if zp_hi > max_code_int:
                    zp_hi = max_code_int
                    zp_lo = zp_hi - zp_sweep_limit + 1
                    if zp_lo < 0:
                        zp_lo = 0

            for zero_point_int in range(zp_lo, zp_hi + 1):
                zero_point = np.float32(zero_point_int)
                positive_scale = np.float32(0.0)
                negative_scale = np.float32(0.0)
                if zero_point_int < max_code_int:
                    positive_scale = positive_max / np.float32(max_code_int - zero_point_int)
                if zero_point_int > 0:
                    negative_scale = negative_max / np.float32(zero_point_int)
                coverage_scale = max(positive_scale, negative_scale)
                if coverage_scale <= tiny:
                    coverage_scale = np.float32(1.0)

                for start_index in range(clip_ratios.size + 1):
                    if start_index == 0:
                        candidate_scale = seed_scale
                    else:
                        candidate_scale = coverage_scale * clip_ratios[start_index - 1]

                    for _ in range(affine_iterations):
                        denominator = np.float32(0.0)
                        numerator = np.float32(0.0)
                        for column in range(width):
                            value = np.float32(weight[block_index, column])
                            candidate_q = np.rint(value / candidate_scale + zero_point)
                            candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                            centered = candidate_q - zero_point
                            denominator += centered * centered
                            numerator += centered * value
                        if denominator <= tiny:
                            break
                        fitted_scale = numerator / denominator
                        if not np.isfinite(fitted_scale) or fitted_scale <= tiny:
                            break
                        if fitted_scale == candidate_scale:
                            break
                        candidate_scale = fitted_scale

                    candidate_plain = np.float32(0.0)
                    candidate_weighted = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block_index, column])
                        candidate_q = np.rint(value / candidate_scale + zero_point)
                        candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                        candidate_codes[block_index, column] = np.uint8(candidate_q)
                        centered = candidate_q - zero_point
                        residual = value - candidate_scale * centered
                        squared = residual * residual
                        candidate_plain += squared
                        candidate_weighted += (rms + np.abs(value)) * residual * residual

                    if candidate_plain < local_plain and candidate_weighted <= weighted_bound:
                        local_plain = candidate_plain
                        scales[block_index] = candidate_scale
                        zero_points[block_index] = np.uint8(zero_point_int)
                        for column in range(width):
                            quantized[block_index, column] = candidate_codes[block_index, column]

            baseline_errors[block_index] = baseline_plain
            refined_errors[block_index] = local_plain
            improved[block_index] = local_plain < baseline_plain

        return baseline_errors, refined_errors, improved

    return refine_blocks


def _affine_refine_v2_q4_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool = False,
    bits: int = 4,
    allow_arbitrary_block_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Q4RefineStats]:
    """Minimize plain block MSE for 4/8-bit NBits or 7/8-bit dynamic weights.

    Plain Frobenius error is the exact data-free projection-error proxy for
    white or diagonal-covariance activations. A candidate replaces the running
    best only when it strictly lowers plain MSE and keeps magnitude-weighted
    error within ``(1 + AFFINE_V2_WEIGHTED_TOLERANCE)`` of the internal seed.
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"AFFINE_REFINE_V2 expects a 2-D row matrix, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 refuses weights containing NaN or Inf.")
    if AFFINE_V2_ITERATIONS < 1 or AFFINE_V2_CHUNK_BLOCKS < 1:
        raise ValueError("AFFINE_REFINE_V2 iterations and chunk size must be positive.")
    if AFFINE_V2_WEIGHTED_TOLERANCE < 0.0:
        raise ValueError("AFFINE_REFINE_V2 weighted tolerance must be nonnegative.")
    if AFFINE_V2_NUMBA_THREADS < 1:
        raise ValueError("AFFINE_REFINE_V2 Numba thread count must be positive.")
    if AFFINE_V2_ASYM_ZP_SWEEP_LIMIT < 16:
        raise ValueError(
            "AFFINE_REFINE_V2 asymmetric zero-point sweep limit must be >= 16 so Q2/Q4 "
            f"sweep every zero point; got {AFFINE_V2_ASYM_ZP_SWEEP_LIMIT}."
        )
    clip_ratios = np.asarray(AFFINE_V2_CLIP_RATIOS, dtype=np.float32)
    if clip_ratios.ndim != 1 or not clip_ratios.size or np.any((clip_ratios <= 0.0) | (clip_ratios > 1.0)):
        raise ValueError("AFFINE_REFINE_V2 clip ratios must be a non-empty sequence in (0, 1].")

    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError(f"AFFINE_REFINE_V2 block_size must be a power of two in [16, 256], got {block_size}.")
    if bits not in (4, 7, 8):
        raise ValueError(f"AFFINE_REFINE_V2 supports 4-, 7-, or 8-bit weights, got {bits}-bit.")
    maxq = float((1 << bits) - 1)
    midpoint = int(1 << (bits - 1))
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    tiny = np.finfo(np.float32).tiny
    tolerance = np.float32(1.0 + AFFINE_V2_WEIGHTED_TOLERANCE)
    stats = Q4RefineStats(blocks=total_blocks)
    numba_kernel = _affine_v2_numba_kernel()

    if numba_kernel is not None:
        best_q = np.empty((total_blocks, block_size), dtype=np.uint8)
        best_scales = np.empty(total_blocks, dtype=np.float32)
        best_zero_points = np.empty(total_blocks, dtype=np.uint8)
        chunks = iter(_iter_q4_row_chunks(values, block_size, AFFINE_V2_CHUNK_BLOCKS))
        try:
            current_chunk = next(chunks)
        except StopIteration:
            current_chunk = None
        seed_future = (
            _affine_v2_seed_pipeline_executor().submit(
                _affine_v2_seed_blocks, current_chunk[2], block_size, symmetric, bits
            )
            if current_chunk is not None
            else None
        )
        while current_chunk is not None:
            start, end, weight = current_chunk
            seed_q, seed_scales, seed_zero_points = seed_future.result()
            try:
                next_chunk = next(chunks)
            except StopIteration:
                next_chunk = None
            next_seed_future = (
                _affine_v2_seed_pipeline_executor().submit(
                    _affine_v2_seed_blocks, next_chunk[2], block_size, symmetric, bits
                )
                if next_chunk is not None
                else None
            )
            local_q = best_q[start:end]
            local_scales = best_scales[start:end]
            local_zero_points = best_zero_points[start:end]
            local_q[:] = seed_q
            local_scales[:] = seed_scales[:, 0]
            local_zero_points[:] = seed_zero_points[:, 0]
            baseline_plain, local_plain, local_improved = numba_kernel(
                weight,
                local_q,
                local_scales,
                local_zero_points,
                clip_ratios,
                AFFINE_V2_SEED_ITERATIONS,
                AFFINE_V2_SEED_ZP_RADIUS,
                AFFINE_V2_ITERATIONS,
                tolerance,
                tiny,
                symmetric,
                np.float32(maxq),
                np.int64(midpoint),
                np.int64(AFFINE_V2_ASYM_ZP_SWEEP_LIMIT),
            )
            stats.improved_blocks += int(np.count_nonzero(local_improved))
            stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
            stats.refined_error += float(local_plain.sum(dtype=np.float64))
            current_chunk = next_chunk
            seed_future = next_seed_future
        return (
            best_q.reshape(rows, block_count, block_size),
            best_scales.reshape(rows, block_count),
            best_zero_points.reshape(rows, block_count),
            stats,
        )

    best_q, best_scales, best_zero_points, _ = _affine_v2_seed_refine_q4_rows(
        values, block_size, symmetric, bits, allow_arbitrary_block_size
    )
    best_q = best_q.reshape(-1, block_size)
    best_scales = best_scales.reshape(-1)
    best_zero_points = best_zero_points.reshape(-1)

    for start, end, weight in _iter_q4_row_chunks(
        values, block_size, AFFINE_V2_CHUNK_BLOCKS
    ):
        local_q = best_q[start:end]
        local_scales = best_scales[start:end]
        local_zero_points = best_zero_points[start:end]
        # Magnitude weighting is used only for the Pareto safety bound, never for
        # the scale fit, so the optimization target stays plain MSE.
        weight_importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        weight_importance = np.where(
            weight_importance.sum(axis=1, keepdims=True) > 0.0,
            weight_importance,
            np.float32(1.0),
        )
        seed_scales = local_scales.copy().reshape(-1, 1)
        seed_q = local_q.astype(np.float32)
        seed_zero_points = local_zero_points.astype(np.float32).reshape(-1, 1)
        seed_residual = weight - seed_scales * (seed_q - seed_zero_points)
        baseline_plain = np.sum(seed_residual * seed_residual, axis=1)
        baseline_weighted = np.sum(weight_importance * seed_residual * seed_residual, axis=1)
        weighted_bound = tolerance * baseline_weighted
        local_plain = baseline_plain.copy()
        local_improved = np.zeros(end - start, dtype=bool)
        positive_max = np.maximum(weight.max(axis=1, keepdims=True), np.float32(0.0))
        negative_max = np.maximum(-weight.min(axis=1, keepdims=True), np.float32(0.0))

        maxq_int = int(maxq)
        windowing = (not symmetric) and (maxq_int + 1 > AFFINE_V2_ASYM_ZP_SWEEP_LIMIT)
        if not windowing:
            for zero_point_int in ((midpoint,) if symmetric else range(int(maxq) + 1)):
                zero_point = np.float32(zero_point_int)
                positive_scale = (
                    positive_max / np.float32(int(maxq) - zero_point_int)
                    if zero_point_int < int(maxq)
                    else np.zeros_like(positive_max)
                )
                negative_scale = (
                    negative_max / np.float32(zero_point_int)
                    if zero_point_int > 0
                    else np.zeros_like(negative_max)
                )
                coverage_scale = np.maximum(positive_scale, negative_scale)
                coverage_scale = np.where(coverage_scale > tiny, coverage_scale, np.float32(1.0))
                initial_scales = [seed_scales]
                initial_scales.extend(coverage_scale * ratio for ratio in clip_ratios)

                for initial_scale in initial_scales:
                    candidate_scale = initial_scale.copy()
                    for _ in range(AFFINE_V2_ITERATIONS):
                        candidate_q = np.clip(
                            np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                        )
                        centered_q = candidate_q - zero_point
                        # Unweighted least-squares scale keeps the objective plain MSE.
                        denominator = np.sum(centered_q * centered_q, axis=1, keepdims=True)
                        numerator = np.sum(centered_q * weight, axis=1, keepdims=True)
                        fitted_scale = np.divide(
                            numerator,
                            denominator,
                            out=candidate_scale.copy(),
                            where=denominator > tiny,
                        )
                        candidate_scale = np.where(
                            np.isfinite(fitted_scale) & (fitted_scale > tiny),
                            fitted_scale,
                            candidate_scale,
                        )

                    candidate_q = np.clip(
                        np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                    )
                    residual = weight - candidate_scale * (candidate_q - zero_point)
                    candidate_plain = np.sum(residual * residual, axis=1)
                    candidate_weighted = np.sum(weight_importance * residual * residual, axis=1)
                    take = (candidate_plain < local_plain) & (candidate_weighted <= weighted_bound)
                    local_q[take] = candidate_q[take].astype(np.uint8)
                    local_scales[take] = candidate_scale[take, 0]
                    local_zero_points[take] = np.uint8(zero_point_int)
                    local_plain[take] = candidate_plain[take]
                    local_improved[take] = True
        else:
            # Q8 asymmetric: sweep a window of AFFINE_V2_ASYM_ZP_SWEEP_LIMIT zero
            # points centered on each block's k-quant seed (mirrors the numba path).
            half = AFFINE_V2_ASYM_ZP_SWEEP_LIMIT // 2
            seed_zp_int = local_zero_points.astype(np.int64).reshape(-1, 1)
            window_lo = np.clip(seed_zp_int - half, 0, maxq_int)
            window_lo = np.clip(
                window_lo - np.maximum(
                    window_lo + AFFINE_V2_ASYM_ZP_SWEEP_LIMIT - 1 - maxq_int, 0
                ),
                0,
                maxq_int,
            )
            for offset in range(AFFINE_V2_ASYM_ZP_SWEEP_LIMIT):
                zp_int = np.clip(window_lo + offset, 0, maxq_int)
                zero_point = zp_int.astype(np.float32)
                denom_pos = np.float32(maxq_int) - zero_point
                positive_scale = np.where(
                    denom_pos > 0.0,
                    positive_max / np.where(denom_pos > 0.0, denom_pos, np.float32(1.0)),
                    np.float32(0.0),
                )
                negative_scale = np.where(
                    zero_point > 0.0,
                    negative_max / np.where(zero_point > 0.0, zero_point, np.float32(1.0)),
                    np.float32(0.0),
                )
                coverage_scale = np.maximum(positive_scale, negative_scale)
                coverage_scale = np.where(coverage_scale > tiny, coverage_scale, np.float32(1.0))
                initial_scales = [seed_scales]
                initial_scales.extend(coverage_scale * ratio for ratio in clip_ratios)

                for initial_scale in initial_scales:
                    candidate_scale = initial_scale.copy()
                    for _ in range(AFFINE_V2_ITERATIONS):
                        candidate_q = np.clip(
                            np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                        )
                        centered_q = candidate_q - zero_point
                        denominator = np.sum(centered_q * centered_q, axis=1, keepdims=True)
                        numerator = np.sum(centered_q * weight, axis=1, keepdims=True)
                        fitted_scale = np.divide(
                            numerator,
                            denominator,
                            out=candidate_scale.copy(),
                            where=denominator > tiny,
                        )
                        candidate_scale = np.where(
                            np.isfinite(fitted_scale) & (fitted_scale > tiny),
                            fitted_scale,
                            candidate_scale,
                        )

                    candidate_q = np.clip(
                        np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                    )
                    residual = weight - candidate_scale * (candidate_q - zero_point)
                    candidate_plain = np.sum(residual * residual, axis=1)
                    candidate_weighted = np.sum(weight_importance * residual * residual, axis=1)
                    take = (candidate_plain < local_plain) & (candidate_weighted <= weighted_bound)
                    local_q[take] = candidate_q[take].astype(np.uint8)
                    local_scales[take] = candidate_scale[take, 0]
                    local_zero_points[take] = zp_int[take, 0].astype(np.uint8)
                    local_plain[take] = candidate_plain[take]
                    local_improved[take] = True

        stats.improved_blocks += int(np.count_nonzero(local_improved))
        stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
        stats.refined_error += float(local_plain.sum(dtype=np.float64))

    return (
        best_q.reshape(rows, block_count, block_size),
        best_scales.reshape(rows, block_count),
        best_zero_points.reshape(rows, block_count),
        stats,
    )


def _pack_q4_last_axis(values: np.ndarray, pad_value: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.shape[-1] & 1:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)], constant_values=pad_value)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def _pack_codes_last_axis(values: np.ndarray, bits: int, pad_value: int = 0) -> np.ndarray:
    """Pack integer codes for MatMulNBits: nibble-pack at 4 bits, raw uint8 at 8 bits."""
    if bits == 8:
        return np.ascontiguousarray(np.asarray(values, dtype=np.uint8))
    if bits == 4:
        return _pack_q4_last_axis(values, pad_value=pad_value)
    raise ValueError(f"unsupported MatMulNBits bit width {bits}; expected 4 or 8.")


def _make_uint4_initializer(name: str, values: np.ndarray) -> TensorProto:
    values = np.asarray(values, dtype=np.uint8)
    flat = values.reshape(-1)
    if flat.size & 1:
        flat = np.pad(flat, (0, 1))
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return helper.make_tensor(name, TensorProto.UINT4, values.shape, packed.tobytes(), raw=True)


def _make_uintn_initializer(name: str, values: np.ndarray, bits: int) -> TensorProto:
    """Build a Gather data/zero-point initializer: logical UINT4 at 4 bits, UINT8 at 8 bits."""
    if bits == 8:
        return numpy_helper.from_array(
            np.ascontiguousarray(np.asarray(values, dtype=np.uint8)), name=name
        )
    if bits == 4:
        return _make_uint4_initializer(name, values)
    raise ValueError(f"unsupported GatherBlockQuantized bit width {bits}; expected 4 or 8.")


def _make_quant_initializer(name: str, values: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(values), name=name)


def _k_quant_q4_rows(
    data: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ORT's k-quant objective on CPU with bounded temporary memory."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"k_quant expects a 2-D row matrix, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("k_quant refuses weights containing NaN or Inf.")
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    quantized = np.empty((rows * block_count, block_size), dtype=np.uint8)
    scales = np.empty(rows * block_count, dtype=np.float32)
    zero_points = np.empty(rows * block_count, dtype=np.uint8)
    for start, end, weight in _iter_q4_row_chunks(
        values, block_size, AFFINE_V2_SEED_CHUNK_BLOCKS
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            chunk_q, chunk_scales, chunk_zero_points = quant_tensor_k_quant_cpu(
                weight, 4, block_size
            )
        quantized[start:end] = np.clip(chunk_q, 0.0, 15.0).astype(np.uint8)
        scales[start:end] = np.asarray(chunk_scales, dtype=np.float32).reshape(-1)
        zero_points[start:end] = np.clip(
            np.asarray(chunk_zero_points, dtype=np.int16).reshape(-1), 0, 15
        ).astype(np.uint8)
    return (
        quantized.reshape(rows, block_count, block_size),
        scales.reshape(rows, block_count),
        zero_points.reshape(rows, block_count),
    )


def _quantize_k_quant_matmul(graph, node, weight: TensorProto, rp: ResolvedPlan, make_name):
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2 or weight_array.dtype.kind != "f":
        print(
            f"  k_quant: skipping {node.name or weight.name!r}; "
            "MatMul weight must be a floating-point matrix."
        )
        return None
    input_features, output_features = weight_array.shape
    quantized, scales, zero_points = _k_quant_q4_rows(
        weight_array.T, rp.block_size
    )
    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_quant_initializer(weight_name, _pack_q4_last_axis(quantized)),
        _make_quant_initializer(
            scale_name, scales.astype(weight_array.dtype, copy=False)
        ),
        _make_quant_initializer(
            zero_point_name,
            _pack_q4_last_axis(zero_points, pad_value=8),
        ),
    ])
    attributes = {
        "K": input_features,
        "N": output_features,
        "bits": 4,
        "block_size": rp.block_size,
    }
    if rp.accuracy_level:
        attributes["accuracy_level"] = rp.accuracy_level
    return helper.make_node(
        "MatMulNBits",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=f"{node.name}_K_QUANT_Q4" if node.name else make_name("matmul"),
        domain="com.microsoft",
        **attributes,
    )


def quantize_k_quant_model(model: onnx.ModelProto, rp: ResolvedPlan) -> int:
    """Replace selected constant MatMuls with chunked CPU k-quant Q4 ops."""
    quantized_matmuls = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, "k_quant_q4_")
        replaced_initializers: set[str] = set()
        new_nodes = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)

            selected = node.op_type == "MatMul" and "MatMul" in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False
            replacement = None
            if selected and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    replacement = _quantize_k_quant_matmul(
                        graph, node, weight, rp, make_name
                    )
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            new_nodes.append(replacement or node)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        remaining_initializers = {initializer.name for initializer in graph.initializer}
        obsolete_inputs = replaced_initializers - remaining_initializers
        if obsolete_inputs:
            graph_inputs = [
                value for value in graph.input if value.name not in obsolete_inputs
            ]
            graph.ClearField("input")
            graph.input.extend(graph_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls:
        _ensure_ms_domain_opset(model)
        _deduplicate_node_names(model.graph)
    print(f"  k_quant CPU surgery: {quantized_matmuls} MatMul -> MatMulNBits.")
    return quantized_matmuls


def _quantize_affine_v2_matmul(
    graph,
    node,
    weight: TensorProto,
    rp: ResolvedPlan,
    bits: int,
    make_name,
):
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2:
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; "
            "MatMul weight rank is not 2."
        )
        return None, None
    if weight_array.dtype.kind != "f":
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; "
            "MatMul weight is not floating point."
        )
        return None, None

    input_features, output_features = weight_array.shape
    quantized, scales, zero_points, stats = _affine_refine_v2_q4_rows(
        weight_array.T, rp.block_size, rp.symmetric, bits
    )
    packed_weight = _pack_codes_last_axis(quantized, bits)
    packed_zero_points = _pack_codes_last_axis(zero_points, bits, pad_value=1 << (bits - 1))
    scales = scales.astype(weight_array.dtype, copy=False)

    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_quant_initializer(weight_name, packed_weight),
        _make_quant_initializer(scale_name, scales),
        _make_quant_initializer(zero_point_name, packed_zero_points),
    ])
    attributes = {
        "K": input_features,
        "N": output_features,
        "bits": bits,
        "block_size": rp.block_size,
    }
    if rp.accuracy_level:
        attributes["accuracy_level"] = rp.accuracy_level
    replacement = helper.make_node(
        "MatMulNBits",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=(
            f"{node.name}_AFFINE_REFINE_V2_Q{bits}"
            if node.name
            else make_name("matmul")
        ),
        domain="com.microsoft",
        **attributes,
    )
    return replacement, stats


def _quantize_affine_v2_gather(
    graph,
    node,
    weight: TensorProto,
    rp: ResolvedPlan,
    bits: int,
    quantize_axis: int,
    make_name,
):
    weight_array = numpy_helper.to_array(weight)
    rank = weight_array.ndim
    quantize_axis = (quantize_axis + rank) % rank
    try:
        gather_axis = int(helper.get_node_attr_value(node, "axis"))
    except ValueError:
        gather_axis = 0
    gather_axis = (gather_axis + rank) % rank
    if weight_array.dtype.kind != "f":
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; "
            "Gather data is not floating point."
        )
        return None, None
    if gather_axis != 0 or quantize_axis != rank - 1:
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; GatherBlockQuantized "
            "requires gather_axis=0 and quantize_axis=last for CPU/CUDA portability."
        )
        return None, None
    logical_width = weight_array.shape[-1]
    if logical_width % rp.block_size:
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; "
            f"Gather width {logical_width} is not "
            f"divisible by block_size={rp.block_size}, which CUDA does not handle portably."
        )
        return None, None

    outer_shape = weight_array.shape[:-1]
    rows = int(np.prod(outer_shape, dtype=np.int64))
    quantized, scales, zero_points, stats = _affine_refine_v2_q4_rows(
        weight_array.reshape(rows, logical_width), rp.block_size, rp.symmetric, bits
    )
    logical_quantized = quantized.reshape(rows, -1)[:, :logical_width].reshape(weight_array.shape)
    block_count = scales.shape[-1]
    scales = scales.reshape(*outer_shape, block_count).astype(weight_array.dtype, copy=False)
    zero_points = zero_points.reshape(*outer_shape, block_count)

    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_uintn_initializer(weight_name, logical_quantized, bits),
        _make_quant_initializer(scale_name, scales),
        _make_uintn_initializer(zero_point_name, zero_points, bits),
    ])
    replacement = helper.make_node(
        "GatherBlockQuantized",
        [weight_name, node.input[1], scale_name, zero_point_name],
        list(node.output),
        name=f"{node.name}_AFFINE_REFINE_V2_Q{bits}" if node.name else make_name("gather"),
        domain="com.microsoft",
        gather_axis=gather_axis,
        quantize_axis=quantize_axis,
        block_size=rp.block_size,
        bits=bits,
    )
    return replacement, stats


def _ensure_ms_domain_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            opset.version = max(opset.version, 1)
            return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def quantize_affine_v2_model(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    bits: int,
) -> Q4RefineStats:
    """Replace selected constant MatMul/Gather nodes with AFFINE_REFINE_V2 Q4/Q8 ops."""
    if rp.quant_format != "QOPERATOR":
        raise ValueError("AFFINE_REFINE_V2 supports QOperator format only.")
    if bits not in (4, 8):
        raise ValueError(f"AFFINE_REFINE_V2 supports 4- or 8-bit weights, got {bits}-bit.")
    quant_axes = dict(zip(rp.op_types, rp.axes))
    total = Q4RefineStats()
    quantized_matmuls = 0
    quantized_gathers = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls, quantized_gathers
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, f"affine_refine_v2_q{bits}_")
        replaced_initializers: set[str] = set()
        new_nodes = []

        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)

            selected = node.op_type in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False

            replacement = None
            stats = None
            if selected and node.op_type == "MatMul" and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    replacement, stats = _quantize_affine_v2_matmul(
                        graph, node, weight, rp, bits, make_name
                    )
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            elif selected and node.op_type == "Gather" and len(node.input) >= 2:
                weight = init_map.get(node.input[0])
                if weight is not None:
                    replacement, stats = _quantize_affine_v2_gather(
                        graph,
                        node,
                        weight,
                        rp,
                        bits,
                        quant_axes.get("Gather", 1),
                        make_name,
                    )
                    if replacement is not None:
                        quantized_gathers += 1
                        replaced_initializers.add(weight.name)

            new_nodes.append(replacement or node)
            if stats is not None:
                total.add(stats)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        remaining_initializers = {initializer.name for initializer in graph.initializer}
        obsolete_inputs = replaced_initializers - remaining_initializers
        if obsolete_inputs:
            graph_inputs = [value for value in graph.input if value.name not in obsolete_inputs]
            graph.ClearField("input")
            graph.input.extend(graph_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls or quantized_gathers:
        _ensure_ms_domain_opset(model)
        _deduplicate_node_names(model.graph)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 surgery: {quantized_matmuls} MatMul -> MatMulNBits, "
        f"{quantized_gathers} Gather -> GatherBlockQuantized; improved "
        f"{total.improved_blocks}/{total.blocks} blocks over its internal seed, "
        f"plain MSE ratio={ratio:.6f}."
    )
    return total


def _uses_fp16(plan: Plan) -> bool:
    return plan.fp16 or plan.method.upper() == "F16"


_PRECISION_MODEL_PLANS = [plan for name, plan in MODEL_PLANS.items() if name != "LLM_Metadata"]
MIXED_PRECISION = (
    any(_uses_fp16(p) for p in _PRECISION_MODEL_PLANS)
    and not all(_uses_fp16(p) for p in _PRECISION_MODEL_PLANS)
)


def get_model_paths(name: str) -> tuple[str, str]:
    return (
        os.path.join(ORIGINAL_FOLDER_PATH, f"{name}.onnx"),
        os.path.join(QUANTED_FOLDER_PATH, f"{name}.onnx"),
    )


def model_exceeds_2gb(model_path: str) -> bool:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total > 2 * 1024 ** 3


def _remove_external_files(model_path: str) -> None:
    for path in (model_path, model_path + ".data"):
        if os.path.exists(path):
            os.remove(path)


def _save_model(model, model_path: str, external: bool) -> None:
    # Delete first: ONNX appends to existing external-data sidecars.
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
        )
    else:
        onnx.save(model, model_path)


def read_onnx_metadata(model_path: str) -> dict:
    # Metadata lives in the graph proto; skip external data for speed.
    try:
        model = onnx.load(model_path, load_external_data=False)
        meta = {prop.key: prop.value for prop in model.metadata_props}
        del model
        gc.collect()
        return meta
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort; never abort the pipeline over it
        print(f"  Warning: could not read metadata from {os.path.basename(model_path)} ({exc}).")
        return {}


def _metadata_model_file_names(source_folder: Path) -> dict[str, str]:
    metadata = read_onnx_metadata(str(source_folder / "LLM_Metadata.onnx"))
    return {
        key[len("model_file_name_"):]: value
        for key, value in metadata.items()
        if key.startswith("model_file_name_") and value
    }


def _update_onnx_metadata(model: onnx.ModelProto, metadata: dict) -> None:
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)


def write_onnx_metadata(model_path: str, metadata: dict) -> None:
    # Rewrite only the graph proto so external weight sidecars stay untouched.
    model = onnx.load(model_path, load_external_data=False)
    _update_onnx_metadata(model, metadata)
    onnx.save(model, model_path)
    del model
    gc.collect()


def _iter_all_data_tensors(graph):
    # External data can live in initializers and Constant-node attribute tensors.
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_all_data_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _retarget_external_location(model_path: str, new_location: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = new_location
    onnx.save(model, model_path)
    del model
    gc.collect()


def _multiply_shape_terms(left, right):
    coefficient = left[0] * right[0]
    powers = dict(left[1])
    for symbol, exponent in right[1].items():
        powers[symbol] = powers.get(symbol, 0) + exponent
        if powers[symbol] == 0:
            del powers[symbol]
    return coefficient, powers


def _resolve_reshape_shape(shape: tuple[int, ...], input_terms: list):
    result, inferred_index = [], None
    known_product = (Fraction(1), {})
    for index, dimension in enumerate(shape):
        if dimension == -1:
            if inferred_index is not None:
                return None
            inferred_index = index
            result.append(None)
            continue
        if dimension == 0:
            if index >= len(input_terms):
                return None
            term = input_terms[index]
        elif dimension > 0:
            term = (Fraction(dimension), {})
        else:
            return None
        result.append(term)
        known_product = _multiply_shape_terms(known_product, term)

    if inferred_index is not None:
        inverse = (1 / known_product[0], {
            symbol: -exponent for symbol, exponent in known_product[1].items()
        })
        result[inferred_index] = _multiply_shape_terms((Fraction(1), {"size": 1}), inverse)
    return result


def _compose_reshape_shapes(first_shape: tuple[int, ...], second_shape: tuple[int, ...]):
    input_terms = [
        (Fraction(1), {f"dim_{index}": 1})
        for index in range(max(len(first_shape), len(second_shape)))
    ]
    middle_terms = _resolve_reshape_shape(first_shape, input_terms)
    final_terms = _resolve_reshape_shape(second_shape, middle_terms) if middle_terms is not None else None
    if final_terms is None:
        return None

    composed, unresolved = [], []
    for index, (coefficient, powers) in enumerate(final_terms):
        if not powers and coefficient.denominator == 1 and coefficient > 0:
            composed.append(coefficient.numerator)
        elif coefficient == 1 and powers == {f"dim_{index}": 1}:
            composed.append(0)
        else:
            unresolved.append(index)
            composed.append(None)
    if len(unresolved) > 1:
        return None
    if unresolved:
        composed[unresolved[0]] = -1

    candidate = tuple(composed)
    return candidate if _resolve_reshape_shape(candidate, input_terms) == final_terms else None


def _constant_int_values(name: str, producer: dict, init_map: dict) -> tuple[int, ...] | None:
    tensor = init_map.get(name)
    if tensor is None:
        node = producer.get(name)
        if node is None or node.op_type != "Constant":
            return None
        tensor = next((attr.t for attr in node.attribute if attr.name == "value"), None)
    if tensor is None:
        return None
    try:
        values = numpy_helper.to_array(tensor)
    except Exception:
        return None
    if values.dtype.kind not in "iu":
        return None
    return tuple(int(value) for value in values.reshape(-1))


def fuse_consecutive_reshapes_graph(graph) -> int:
    """Fuse constant-shape Reshape pairs only when their composed semantics are provable."""
    graph_outputs = {value.name for value in graph.output}
    make_name = _make_name_factory(graph, "reshape_fusion_")
    removed_values, fused = set(), 0

    while True:
        producer = {output: node for node in graph.node for output in node.output}
        consumers: dict[str, list] = {}
        for node in graph.node:
            for name in node.input:
                consumers.setdefault(name, []).append(node)
        init_map = _init_map(graph)
        replacement = None

        for second in graph.node:
            if second.op_type != "Reshape" or len(second.input) < 2:
                continue
            first = producer.get(second.input[0])
            if first is None or first.op_type != "Reshape" or len(first.input) < 2:
                continue
            middle = first.output[0]
            if middle in graph_outputs or len(consumers.get(middle, [])) != 1:
                continue
            if any(
                next((attr.i for attr in node.attribute if attr.name == "allowzero"), 0)
                for node in (first, second)
            ):
                continue
            first_shape = _constant_int_values(first.input[1], producer, init_map)
            second_shape = _constant_int_values(second.input[1], producer, init_map)
            if first_shape is None or second_shape is None:
                continue
            composed_shape = _compose_reshape_shapes(first_shape, second_shape)
            if composed_shape is None:
                continue
            replacement = first, second, composed_shape, second_shape
            break

        if replacement is None:
            break
        first, second, composed_shape, second_shape = replacement
        second.input[0] = first.input[0]
        if composed_shape != second_shape:
            shape_name = make_name(f"shape_{fused}")
            graph.initializer.append(numpy_helper.from_array(
                np.asarray(composed_shape, dtype=np.int64), name=shape_name
            ))
            second.input[1] = shape_name
        removed_values.update(first.output)
        keep = [node for node in graph.node if id(node) != id(first)]
        graph.ClearField("node")
        graph.node.extend(keep)
        fused += 1

    if fused:
        _dead_code_elimination(graph)
        _drop_unused_initializers(graph)
        keep_info = [value for value in graph.value_info if value.name not in removed_values]
        graph.ClearField("value_info")
        graph.value_info.extend(keep_info)
    return fused


def fuse_consecutive_reshapes(model_path: str) -> int:
    model = onnx.load(model_path, load_external_data=False)
    fused = fuse_consecutive_reshapes_graph(model.graph)
    if fused:
        onnx.save(model, model_path)
    del model
    gc.collect()
    return fused


def resave(src_path: str, dst_path: str, external: bool, do_surgery: bool = False) -> None:
    model = onnx.load(src_path)
    if do_surgery:
        apply_kv_surgery(model)
    _save_model(model, dst_path, external)
    del model
    gc.collect()


def run_onnxslim(model_path: str, external: bool, no_shape_infer: bool = False) -> None:
    # Stash external weights so onnxslim writes a fresh sidecar instead of appending.
    def _slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
            skip_optimizations=SLIM_SKIP_OPTIMIZATIONS,
            size_threshold=SLIM_SIZE_THRESHOLD,
            save_as_external_data=external,
            verbose=False,
        )
        fused = fuse_consecutive_reshapes(model_path)
        if fused:
            print(f"  Fused {fused} semantics-safe consecutive Reshape pairs.")

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        _slim()
        return

    # Repoint to the stash while slimming; restore on failure.
    stash_path = model_path + ".stash.data"
    if os.path.exists(stash_path):
        os.remove(stash_path)
    os.replace(data_path, stash_path)
    _retarget_external_location(model_path, os.path.basename(stash_path))
    try:
        _slim()
    except BaseException:
        if not os.path.exists(data_path):
            os.replace(stash_path, data_path)
            _retarget_external_location(model_path, os.path.basename(data_path))
        raise
    finally:
        if os.path.exists(stash_path):
            os.remove(stash_path)


def build_fusion_options(model_type: str):
    if not OPTIMIZER_FUSION_OPTIONS:
        return None
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions(model_type)
    for key, value in OPTIMIZER_FUSION_OPTIONS.items():
        setattr(options, key, value)
    return options


def optimize_onnx_model(model_path: str, num_heads: int, hidden_size: int,
                        use_fp16: bool, external: bool, keep_io_types: bool,
                        preserve_fp16_compute: bool) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    # A CPU-targeted ORT optimization pass rewrites exported F16 compute islands to
    # F32 and names the boundary nodes InsertedPrecisionFreeCast_*. Level 0 keeps
    # the Python transformer fusions while bypassing that provider-aware rewrite.
    ort_opt_level = 0 if preserve_fp16_compute else OPTIMIZER_LEVEL
    model = optimize_model(
        model_path,
        use_gpu=False,
        opt_level=ort_opt_level,
        num_heads=num_heads,
        hidden_size=hidden_size,
        optimization_options=build_fusion_options(OPTIMIZER_MODEL_TYPE),
        model_type=OPTIMIZER_MODEL_TYPE,
        only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
        verbose=False,
    )
    if use_fp16:
        model.convert_float_to_float16(
            keep_io_types=keep_io_types,
            force_fp16_initializers=F16_FORCE_INITIALIZERS,
            use_symbolic_shape_infer=SHAPE_INFER,
            max_finite_val=F16_MAX_FINITE_VAL,
            min_positive_val=F16_MIN_POSITIVE_VAL,
            op_block_list=F16_OP_BLOCK_LIST,
            node_block_list=F16_NODE_BLOCK_LIST,
        )
        renamed = _deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    if preserve_fp16_compute:
        inserted_casts = [
            node.name
            for node in model.model.graph.node
            if node.name.startswith("InsertedPrecisionFreeCast_")
        ]
        if inserted_casts:
            raise RuntimeError(
                "COMPUTE_IN_F32=0 requires a cast-free F16 compute graph, but ORT "
                f"inserted {len(inserted_casts)} precision-free Cast node(s)."
            )
    model.save_model_to_file(model_path, use_external_data_format=external)
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    try:
        model = onnx.version_converter.convert_version(onnx.load(model_path), version)
        _save_model(model, model_path, external)
        del model
        gc.collect()
    except Exception as exc:
        print(f"  Opset upgrade failed: {exc}. Keeping current version.")


@lru_cache(maxsize=1)
def fetch_transformer_config(download_path: str) -> tuple[int, int]:
    if not download_path or download_path.upper() == "NONE":
        return 0, 0
    try:
        cfg = AutoConfig.from_pretrained(download_path, trust_remote_code=True)
        cfg = getattr(cfg, "llm_config", None) or getattr(cfg, "text_config", None) or cfg
        return getattr(cfg, "num_attention_heads", 0), getattr(cfg, "hidden_size", 0)
    except Exception as e:
        print(f"  Warning: could not read config ({e}); using defaults.")
        return 0, 0


def build_weight_only_config(rp: ResolvedPlan, bits: int):
    supported_bits = _WEIGHT_ONLY_ALGO_BITS.get(rp.algo)
    if supported_bits is None or bits not in supported_bits:
        raise ValueError(
            f"algo={rp.algo!r} cannot produce {bits}-bit weights; validate the plan before quantization."
        )
    op_types, axes = list(rp.op_types), list(rp.axes)
    quant_axes = tuple(zip(op_types, axes))
    quant_format = _QUANT_FORMATS[rp.quant_format]
    common = {
        "quant_format": quant_format,
        "op_types_to_quantize": tuple(op_types),
    }
    if rp.algo == "RTN":
        cfg = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common)
    elif rp.algo == "HQQ":
        cfg = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits, block_size=rp.block_size, axis=axes[0], quant_axes=quant_axes, **common,
        )
    elif rp.algo == "k_quant":
        cfg = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(**common)
    else:  # DEFAULT
        cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=rp.block_size, is_symmetric=rp.symmetric,
            accuracy_level=rp.accuracy_level, quant_axes=quant_axes, **common,
        )
    cfg.bits = bits
    return cfg, quant_axes


def quantize_weight_only(src_path: str, dst_path: str, rp: ResolvedPlan, bits: int, external: bool,
                         do_surgery: bool = False) -> None:
    if rp.algo == "AFFINE_REFINE_V2":
        if bits not in (4, 8):
            raise ValueError(f"{rp.algo} supports Q4/Q8 only, got {bits}-bit.")
        print(
            f"  Quantizing weights ({rp.algo}, {bits}-bit, block={rp.block_size}, "
            f"symmetric={rp.symmetric}, format={rp.quant_format}, ops={list(rp.op_types)})..."
        )
        model = quant_utils.load_model_with_shape_infer(Path(src_path))
        if do_surgery:
            apply_kv_surgery(model)
        quantize_affine_v2_model(model, rp, bits)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return

    if rp.algo == "k_quant":
        if bits != 4:
            raise ValueError(f"{rp.algo} supports Q4 only, got {bits}-bit.")
        print(
            f"  Quantizing weights ({rp.algo}, 4-bit, block={rp.block_size}, "
            f"format={rp.quant_format}, ops={list(rp.op_types)}, CPU-only)..."
        )
        model = quant_utils.load_model_with_shape_infer(Path(src_path))
        if do_surgery:
            apply_kv_surgery(model)
        quantize_k_quant_model(model, rp)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return

    cfg, quant_axes = build_weight_only_config(rp, bits)
    print(f"  Quantizing weights ({rp.algo}, {bits}-bit, block={rp.block_size}, "
          f"format={rp.quant_format}, ops={list(rp.op_types)})...")

    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    if do_surgery:
        apply_kv_surgery(model)
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=rp.block_size,
        is_symmetric=rp.symmetric,
        accuracy_level=rp.accuracy_level,
        quant_format=_QUANT_FORMATS[rp.quant_format],
        op_types_to_quantize=tuple(rp.op_types),
        quant_axes=quant_axes,
        algo_config=cfg,
        nodes_to_exclude=rp.nodes_to_exclude,
        nodes_to_include=rp.nodes_to_include,
    )
    quant.process()
    emitted_bits = {
        int(attribute.i)
        for node in quant.model.model.graph.node
        if node.op_type == "MatMulNBits"
        for attribute in node.attribute
        if attribute.name == "bits"
    }
    if emitted_bits and emitted_bits != {bits}:
        raise RuntimeError(
            f"Weight-only quantizer requested {bits}-bit but emitted MatMulNBits widths "
            f"{sorted(emitted_bits)}; refusing to save a mislabeled model."
        )
    quant.model.save_model_to_file(dst_path, external)
    del model, quant
    gc.collect()


def _quantize_affine_v2_dynamic_matmul(
    graph,
    node,
    weight: TensorProto,
    rp: ResolvedPlan,
    make_name,
):
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2 or weight_array.dtype.kind != "f":
        print(
            f"  AFFINE_REFINE_V2 dynamic: skipping {node.name or weight.name!r}; "
            "MatMul weight must be a floating-point matrix."
        )
        return None, None

    bits = 7 if rp.reduce_range else 8
    if rp.per_channel:
        rows = np.ascontiguousarray(weight_array.T, dtype=np.float32)
        block_size = weight_array.shape[0]
    else:
        rows = np.ascontiguousarray(weight_array.reshape(1, -1), dtype=np.float32)
        block_size = weight_array.size
    quantized, scales, zero_points, stats = _affine_refine_v2_q4_rows(
        rows,
        block_size,
        rp.symmetric,
        bits,
        allow_arbitrary_block_size=True,
    )
    quantized = quantized.reshape(rows.shape)
    scales = scales.reshape(-1)
    zero_points = zero_points.reshape(-1)
    if rp.per_channel:
        quantized = quantized.T
    else:
        quantized = quantized.reshape(weight_array.shape)

    if rp.dynamic_weight_type == "QINT8":
        offset = 1 << (bits - 1)
        quantized = (quantized.astype(np.int16) - offset).astype(np.int8)
        zero_points = (zero_points.astype(np.int16) - offset).astype(np.int8)
    else:
        quantized = quantized.astype(np.uint8)
        zero_points = zero_points.astype(np.uint8)

    weight_name = make_name(f"{weight.name}_quantized")
    scale_name = make_name(f"{weight.name}_scale")
    zero_point_name = make_name(f"{weight.name}_zero_point")
    if not rp.per_channel:
        scales = scales[0]
        zero_points = zero_points[0]
    graph.initializer.extend([
        _make_quant_initializer(weight_name, quantized),
        _make_quant_initializer(scale_name, scales.astype(np.float32, copy=False)),
        _make_quant_initializer(zero_point_name, zero_points),
    ])

    replacement = [helper.make_node(
        "DynamicQuantizeMatMul",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=make_name(f"{node.name or 'matmul'}_dynamic_quantize_matmul"),
        domain="com.microsoft",
    )]
    return replacement, stats


def quantize_affine_v2_dynamic_model(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
) -> Q4RefineStats:
    """Replace selected constant MatMuls with V2-refined dynamic INT8/UINT8 ops."""
    total = Q4RefineStats()
    quantized_matmuls = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, "affine_refine_v2_dynamic_")
        replaced_initializers: set[str] = set()
        new_nodes = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)

            selected = node.op_type == "MatMul" and "MatMul" in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False

            replacement = None
            stats = None
            if selected and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    replacement, stats = _quantize_affine_v2_dynamic_matmul(
                        graph, node, weight, rp, make_name
                    )
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            new_nodes.extend(replacement or [node])
            if stats is not None:
                total.add(stats)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        remaining_initializers = {initializer.name for initializer in graph.initializer}
        obsolete_inputs = replaced_initializers - remaining_initializers
        if obsolete_inputs:
            graph_inputs = [value for value in graph.input if value.name not in obsolete_inputs]
            graph.ClearField("input")
            graph.input.extend(graph_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls:
        _ensure_ms_domain_opset(model)
    _deduplicate_node_names(model.graph)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 dynamic surgery: {quantized_matmuls} MatMul -> "
        f"DynamicQuantizeMatMul; improved {total.improved_blocks}/{total.blocks} channels/tensors "
        f"over its internal seed, plain MSE ratio={ratio:.6f}."
    )
    return total


def quantize_dynamic_int8(src_path: str, dst_path: str, rp: ResolvedPlan, external: bool,
                          do_surgery: bool = False) -> None:
    if rp.algo == "AFFINE_REFINE_V2":
        print(
            f"  Quantizing weights ({rp.algo}, dynamic {rp.dynamic_weight_type}, "
            f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range}, "
            f"symmetric={rp.symmetric})..."
        )
        model = quant_utils.load_model_with_shape_infer(Path(src_path))
        if do_surgery:
            apply_kv_surgery(model)
        quantize_affine_v2_dynamic_model(model, rp)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return

    weight_type = _DYNAMIC_WEIGHT_TYPES[rp.dynamic_weight_type]
    print(f"  Quantizing weights (dynamic INT8, {rp.dynamic_weight_type}, "
          f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range})...")
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    if do_surgery:
        apply_kv_surgery(model)
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=rp.per_channel,
        reduce_range=rp.reduce_range,
        weight_type=weight_type,
        extra_options={
            "ActivationSymmetric": rp.symmetric,
            "WeightSymmetric": rp.symmetric,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": False,
            "MatMulConstBOnly": True,
            # Fallback for default-domain fused ops; only const-B MatMuls are quantized.
            "DefaultTensorType": TensorProto.FLOAT,
        },
        nodes_to_quantize=rp.nodes_to_include,
        nodes_to_exclude=rp.nodes_to_exclude,
        use_external_data_format=external,
    )


def _src_through_casts(name: str, producer: dict) -> str:
    while name in producer and producer[name].op_type == "Cast":
        name = producer[name].input[0]
    return name


def _dead_code_elimination(graph) -> None:
    graph_outputs = {o.name for o in graph.output}
    changed = True
    while changed:
        changed = False
        used = set(graph_outputs)
        for node in graph.node:
            used.update(node.input)
        keep = [n for n in graph.node if (not n.output) or any(o in used for o in n.output)]
        if len(keep) != len(graph.node):
            graph.ClearField("node")
            graph.node.extend(keep)
            changed = True


def _deduplicate_node_names(graph) -> int:
    used_names, next_name_suffix, used_values, next_value_suffix, remap, renamed = set(), {}, set(), {}, {}, 0
    used_values.update(i.name for i in graph.input)
    used_values.update(i.name for i in graph.initializer)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]

        name = node.name
        if name:
            if name not in used_names:
                used_names.add(name)
            else:
                suffix = next_name_suffix.get(name, 1)
                while f"{name}_{suffix}" in used_names:
                    suffix += 1
                node.name = f"{name}_{suffix}"
                used_names.add(node.name)
                next_name_suffix[name] = suffix + 1
                renamed += 1

        for i, output in enumerate(node.output):
            if not output:
                continue
            if output not in used_values:
                used_values.add(output)
                continue
            suffix = next_value_suffix.get(output, 1)
            while f"{output}_{suffix}" in used_values:
                suffix += 1
            new_output = f"{output}_{suffix}"
            node.output[i] = new_output
            used_values.add(new_output)
            next_value_suffix[output] = suffix + 1
            remap[output] = new_output
            renamed += 1
    return renamed


def _ensure_default_opset21(model) -> None:
    has_default = False
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            has_default = True
            if op.version < 21:
                op.version = 21
    if not has_default:
        model.opset_import.append(helper.make_opsetid("", 21))


def _read_int_list(name: str, producer: dict, init_map: dict):
    init = init_map.get(name)
    if init is not None:
        try:
            return numpy_helper.to_array(init).reshape(-1).tolist()
        except Exception:
            return None
    node = producer.get(name)
    if node is not None and node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                try:
                    return numpy_helper.to_array(attr.t).reshape(-1).tolist()
                except Exception:
                    return None
    return None


def _reduce_single_axis(reduce_node, producer: dict, init_map: dict):
    axes = None
    if len(reduce_node.input) > 1 and reduce_node.input[1]:
        axes = _read_int_list(reduce_node.input[1], producer, init_map)
    else:
        for attr in reduce_node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)
    if axes is None or len(axes) != 1:
        return None
    return int(axes[0])


def _is_value_scale_tensor(name: str, producer: dict) -> bool:
    source = _src_through_casts(name, producer)
    for prefix in ("in_value_scale_", "out_value_scale_"):
        if source.startswith(prefix) and source[len(prefix):].isdigit():
            return True
    return False


def _split_value_scale_mul(mul, producer: dict) -> tuple[str, str] | None:
    if mul.op_type != "Mul" or len(mul.input) != 2:
        return None
    left, right = mul.input[0], mul.input[1]
    left_is_scale = _is_value_scale_tensor(left, producer)
    right_is_scale = _is_value_scale_tensor(right, producer)
    if left_is_scale == right_is_scale:
        return None
    return (right, left) if left_is_scale else (left, right)


def _init_map(graph) -> dict[str, TensorProto]:
    return {init.name: init for init in graph.initializer}


def _tensor_dims(tensor: TensorProto) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.dims)


def _node_attrs(node) -> dict:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _graph_used_names(graph) -> set[str]:
    used = {value.name for value in graph.input}
    used.update(value.name for value in graph.output)
    used.update(value.name for value in graph.value_info)
    used.update(init.name for init in graph.initializer)
    for node in graph.node:
        if node.name:
            used.add(node.name)
        used.update(name for name in node.input if name)
        used.update(name for name in node.output if name)
    return used


def _make_name_factory(graph, prefix: str):
    used = _graph_used_names(graph)

    def make(suffix: str) -> str:
        base = f"{prefix}{suffix}"
        if base not in used:
            used.add(base)
            return base
        index = 1
        while f"{base}_{index}" in used:
            index += 1
        name = f"{base}_{index}"
        used.add(name)
        return name

    return make


def _replace_graph_node(graph, target, replacement_nodes: list) -> None:
    nodes, replaced = [], False
    target_id = id(target)
    for node in graph.node:
        if id(node) == target_id:
            nodes.extend(replacement_nodes)
            replaced = True
        else:
            nodes.append(node)
    if not replaced:
        raise RuntimeError(f"node {target.name or target.op_type!r} was not found in the graph")
    graph.ClearField("node")
    graph.node.extend(nodes)


def _drop_initializers(graph, names: set[str]) -> None:
    if names:
        keep = [init for init in graph.initializer if init.name not in names]
        graph.ClearField("initializer")
        graph.initializer.extend(keep)


def _drop_unused_initializers(graph) -> int:
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    unused = {init.name for init in graph.initializer if init.name not in used}
    _drop_initializers(graph, unused)
    return len(unused)


def _drop_shadowed_constant_nodes(graph) -> int:
    """Drop Constant producers only when an identical initializer defines the same value."""
    initializer_map = _init_map(graph)
    kept_nodes = []
    dropped = 0
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                dropped += _drop_shadowed_constant_nodes(attribute.g)
            for subgraph in attribute.graphs:
                dropped += _drop_shadowed_constant_nodes(subgraph)

        shadowed_outputs = [name for name in node.output if name in initializer_map]
        if not shadowed_outputs:
            kept_nodes.append(node)
            continue
        if node.op_type != "Constant" or len(node.output) != 1:
            raise RuntimeError(
                f"node {node.name or node.op_type!r} output collides with initializer(s) "
                f"{shadowed_outputs}"
            )
        value_attribute = next(
            (attribute for attribute in node.attribute if attribute.name == "value"),
            None,
        )
        if value_attribute is None:
            raise RuntimeError(f"shadowed Constant {node.name!r} has no tensor value")
        constant = helper.get_attribute_value(value_attribute)
        initializer = initializer_map[node.output[0]]
        if not isinstance(constant, TensorProto):
            raise RuntimeError(f"shadowed Constant {node.name!r} does not contain a TensorProto")
        same_metadata = (
            constant.data_type == initializer.data_type
            and tuple(constant.dims) == tuple(initializer.dims)
        )
        same_values = same_metadata and np.array_equal(
            numpy_helper.to_array(constant),
            numpy_helper.to_array(initializer),
        )
        if not same_values:
            raise RuntimeError(
                f"Constant {node.name!r} and initializer {initializer.name!r} define "
                "different values"
            )
        dropped += 1

    if dropped:
        graph.ClearField("node")
        graph.node.extend(kept_nodes)
    return dropped


def _validate_graph_references(model: onnx.ModelProto, model_name: str) -> None:
    graph = model.graph
    defined = {value.name for value in graph.input}
    defined.update(init.name for init in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output)
    missing = sorted({
        name
        for node in graph.node
        for name in node.input
        if name and name not in defined
    })
    if missing:
        sample = ", ".join(repr(name) for name in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"Merged graph {model_name!r} has {len(missing)} undefined node input(s): "
            f"{sample}{suffix}"
        )
    missing_outputs = sorted(
        value.name for value in graph.output if value.name not in defined
    )
    if missing_outputs:
        sample = ", ".join(repr(name) for name in missing_outputs[:8])
        suffix = " ..." if len(missing_outputs) > 8 else ""
        raise RuntimeError(
            f"Merged graph {model_name!r} has {len(missing_outputs)} undefined graph output(s): "
            f"{sample}{suffix}"
        )
    declared_types: dict[str, set[int]] = {}
    for collection in (graph.input, graph.output, graph.value_info):
        for value in collection:
            declared_types.setdefault(value.name, set()).add(
                value.type.tensor_type.elem_type
            )
    cast_mismatches = []
    for node in graph.node:
        if node.op_type != "Cast" or len(node.output) != 1:
            continue
        target_type = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        bad_types = declared_types.get(node.output[0], set()) - {
            TensorProto.UNDEFINED,
            target_type,
        }
        if bad_types:
            cast_mismatches.append((node.output[0], target_type, sorted(bad_types)))
    if cast_mismatches:
        raise RuntimeError(
            f"Merged graph {model_name!r} has Cast/value_info type mismatch(es): "
            f"{cast_mismatches[:4]}"
        )


def _restore_kv_scale_outputs(model: onnx.ModelProto) -> int:
    """Repair FP16 cast elimination that strands KV scale graph-output names."""
    graph = model.graph
    defined = {value.name for value in graph.input}
    defined.update(initializer.name for initializer in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output)
    restored = 0
    for graph_output in graph.output:
        output_name = graph_output.name
        if output_name in defined:
            continue
        input_name = None
        for kind in ("key", "value"):
            prefix = f"out_{kind}_scale_"
            suffix = output_name[len(prefix):] if output_name.startswith(prefix) else ""
            if suffix.isdigit():
                input_name = f"in_{kind}_scale_{suffix}"
                break
        if input_name is None:
            continue
        candidates = [
            node
            for node in graph.node
            if node.op_type == "Concat" and input_name in node.input and len(node.output) == 1
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot restore {output_name!r}: expected one Concat consuming "
                f"{input_name!r}, found {len(candidates)}."
            )
        producer = candidates[0]
        old_name = producer.output[0]
        producer.output[0] = output_name
        for node in graph.node:
            if id(node) == id(producer):
                continue
            for index, name in enumerate(node.input):
                if name == old_name:
                    node.input[index] = output_name
        for value in graph.value_info:
            if value.name == old_name:
                value.name = output_name
        defined.add(output_name)
        restored += 1
    return restored


def _align_fp16_shell_boundaries(model: onnx.ModelProto) -> int:
    """Convert transplanted F32 rotary/mask shell boundaries to the FP16 Main contract."""
    graph = model.graph
    boundary_names = {
        "prefill_rotary_cos",
        "prefill_rotary_sin",
        "prefill_attention_mask",
        "decode_rotary_cos",
        "decode_rotary_sin",
        "decode_zero_attention_mask",
    }
    producers = {
        output: node for node in graph.node for output in node.output if output
    }
    initializer_map = _init_map(graph)
    replacements: dict[str, TensorProto] = {}
    aligned = 0
    for name in boundary_names:
        producer = producers.get(name)
        initializer = initializer_map.get(name)
        if producer is not None:
            if producer.op_type != "Cast":
                continue
            to_attribute = next(
                (attribute for attribute in producer.attribute if attribute.name == "to"),
                None,
            )
            if to_attribute is None:
                raise RuntimeError(f"Boundary Cast {producer.name!r} has no 'to' attribute.")
            if to_attribute.i != TensorProto.FLOAT16:
                to_attribute.i = TensorProto.FLOAT16
                aligned += 1
        elif initializer is not None and initializer.data_type == TensorProto.FLOAT:
            replacements[name] = numpy_helper.from_array(
                numpy_helper.to_array(initializer).astype(np.float16),
                name=name,
            )
            aligned += 1

        for collection in (graph.input, graph.output, graph.value_info):
            for value in collection:
                if value.name == name:
                    value.type.tensor_type.elem_type = TensorProto.FLOAT16

    if replacements:
        initializers = [
            replacements.get(initializer.name, initializer)
            for initializer in graph.initializer
        ]
        graph.ClearField("initializer")
        graph.initializer.extend(initializers)
    return aligned


def _convert_transplanted_model_to_fp16(
    model: onnx.ModelProto,
    keep_io_types: bool | None = None,
) -> onnx.ModelProto:
    """Convert F32 strategy-shell arithmetic after transplanting the FP16 Main."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    if keep_io_types is None:
        keep_io_types = MIXED_PRECISION if F16_KEEP_IO_TYPES is None else F16_KEEP_IO_TYPES
    converted = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=True,
        force_fp16_initializers=F16_FORCE_INITIALIZERS,
        min_positive_val=F16_MIN_POSITIVE_VAL,
        max_finite_val=F16_MAX_FINITE_VAL,
        op_block_list=F16_OP_BLOCK_LIST,
        node_block_list=F16_NODE_BLOCK_LIST,
    )
    _restore_kv_scale_outputs(converted)
    from onnxruntime.transformers.onnx_model import OnnxModel

    sorted_model = OnnxModel(converted)
    sorted_model.topological_sort()
    return sorted_model.model


def _find_embed_gather(graph):
    """Return (Gather node, initializer name, vocab, hidden) for Main's token embedding."""
    inits = _init_map(graph)
    candidates = []
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        if node.name.startswith("share_embed_lmhead_"):
            continue
        init = inits.get(node.input[0])
        if init is None or len(init.dims) != 2:
            continue
        rows, cols = _tensor_dims(init)
        if rows <= 0 or cols <= 0:
            continue
        score = rows * cols
        if rows > cols:
            score += rows
        if "input_ids" in node.input[1]:
            score += rows * cols
        candidates.append((score, node, init.name, rows, cols))
    if not candidates:
        raise RuntimeError("embedding Gather with a 2-D initializer was not found")
    _, node, name, vocab, hidden = max(candidates, key=lambda item: item[0])
    return node, name, vocab, hidden


def _restore_prefill_mask_shell_boundary(model: onnx.ModelProto) -> bool:
    """Restore the float16 prefill-mask edge when ORT folds its boundary Casts."""
    graph = model.graph
    canonical_name = "prefill_attention_mask"
    defined = {value.name for value in graph.input}
    defined.update(init.name for init in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output)
    if canonical_name in defined:
        return False

    consumers = _graph_consumers(graph)
    candidates = []
    for producer in graph.node:
        if producer.op_type != "Reshape" or len(producer.output) != 1:
            continue
        old_name = producer.output[0]
        if not old_name.startswith("prefill_/"):
            continue
        users = consumers.get(old_name, [])
        if len(users) != 1 or users[0].op_type != "Cast":
            continue
        cast_users = [
            user
            for output in users[0].output
            for user in consumers.get(output, [])
        ]
        if cast_users and all(user.op_type == "Add" for user in cast_users):
            candidates.append((producer, users[0], old_name))
    if len(candidates) != 1:
        raise RuntimeError(
            "Cannot restore prefill attention-mask boundary: expected one "
            f"Reshape -> Cast -> Add chain, found {len(candidates)}."
        )

    producer, cast, old_name = candidates[0]
    producer.output[0] = canonical_name
    for index, name in enumerate(cast.input):
        if name == old_name:
            cast.input[index] = canonical_name
    for value in graph.value_info:
        if value.name == old_name:
            value.name = canonical_name
    return True


def _find_lmhead(graph, vocab: int, hidden: int):
    """Return (op_type, node) for the lm_head op matching [hidden, vocab] / [vocab, hidden]."""
    inits = _init_map(graph)
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = _node_attrs(node)
        if int(attrs.get("K", -1)) == hidden and int(attrs.get("N", -1)) == vocab:
            return node.op_type, node

    expected = {(hidden, vocab), (vocab, hidden)}
    for op_type in ("MatMul", "Gemm", "MatMulInteger", "DynamicQuantizeMatMul"):
        for node in graph.node:
            if node.op_type != op_type or len(node.input) < 2:
                continue
            init = inits.get(node.input[1])
            if init is not None and _tensor_dims(init) in expected:
                return node.op_type, node
    raise RuntimeError(f"lm_head op with vocab={vocab}, hidden={hidden} was not found")


def _source_embed_lmhead_equal(model_path: Path, chunk_rows: int = 256) -> bool:
    """Compare exported float embedding and LM-head values before quantization."""
    model = onnx.load(str(model_path), load_external_data=False)
    try:
        _, embed_name, vocab, hidden = _find_embed_gather(model.graph)
        lmhead_type, lmhead = _find_lmhead(model.graph, vocab, hidden)
        if lmhead_type not in ("MatMul", "Gemm") or len(lmhead.input) < 2:
            return False
        inits = _init_map(model.graph)
        embed = inits.get(embed_name)
        lmhead_weight = inits.get(lmhead.input[1])
        if embed is None or lmhead_weight is None:
            return False
        def tensor_values(tensor: TensorProto):
            external = {entry.key: entry.value for entry in tensor.external_data}
            if tensor.data_location == TensorProto.EXTERNAL and "location" in external:
                return np.memmap(
                    model_path.parent / external["location"],
                    mode="r",
                    dtype=helper.tensor_dtype_to_np_dtype(tensor.data_type),
                    offset=int(external.get("offset", 0)),
                    shape=_tensor_dims(tensor),
                )
            return numpy_helper.to_array(tensor)

        embed_values = tensor_values(embed)
        lmhead_values = tensor_values(lmhead_weight)
        if lmhead_values.shape == (hidden, vocab):
            lmhead_values = lmhead_values.T
        elif lmhead_values.shape != (vocab, hidden):
            return False
        if embed_values.shape != lmhead_values.shape or embed_values.dtype != lmhead_values.dtype:
            return False
        return all(
            np.array_equal(embed_values[start:start + chunk_rows], lmhead_values[start:start + chunk_rows])
            for start in range(0, vocab, chunk_rows)
        )
    finally:
        del model
        gc.collect()


def _configure_embedding_quantization(rp: ResolvedPlan, share_embed_lmhead: bool) -> None:
    if rp.algo != "AFFINE_REFINE_V2" or rp.method not in ("Q4", "Q8"):
        return
    pairs = [(op_type, axis) for op_type, axis in zip(rp.op_types, rp.axes) if op_type != "Gather"]
    if not share_embed_lmhead:
        pairs.append(("Gather", 1))
    rp.op_types = tuple(op_type for op_type, _ in pairs)
    rp.axes = tuple(axis for _, axis in pairs)


def _make_scalar_initializer(graph, name: str, array: np.ndarray) -> str:
    inits = _init_map(graph)
    if name not in inits:
        graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def _make_axes_initializer(graph, name: str, axes: list[int]) -> str:
    inits = _init_map(graph)
    if name not in inits:
        graph.initializer.append(numpy_helper.from_array(np.array(axes, dtype=np.int64), name=name))
    return name


def _share_float_embed_lmhead(graph, gather, embed_init: str, lmhead, vocab: int, hidden: int) -> dict:
    inits = _init_map(graph)
    shared_weight = lmhead.input[1]
    weight = inits.get(shared_weight)
    if weight is None:
        raise RuntimeError("float lm_head weight initializer was not found")
    dims = _tensor_dims(weight)
    make = _make_name_factory(graph, "share_embed_lmhead_")
    ids, out = gather.input[1], gather.output[0]

    if dims == (vocab, hidden):
        shared_gather = helper.make_node(
            "Gather", [shared_weight, ids], [out], axis=0, name=make("gather")
        )
        replacement = [shared_gather]
    elif dims == (hidden, vocab):
        gathered = make("gathered_hbs")
        shared_gather = helper.make_node(
            "Gather", [shared_weight, ids], [gathered], axis=1, name=make("gather")
        )
        to_bsh = helper.make_node("Transpose", [gathered], [out], perm=[1, 2, 0], name=make("transpose"))
        replacement = [shared_gather, to_bsh]
    else:
        raise RuntimeError(f"unsupported lm_head weight shape {dims}; expected {(hidden, vocab)} or {(vocab, hidden)}")

    _replace_graph_node(graph, gather, replacement)
    if embed_init != shared_weight:
        _drop_initializers(graph, {embed_init})
    return {"lmhead_op": lmhead.op_type, "dropped": embed_init, "shared_weight": shared_weight}


def _share_nbits_embed_lmhead(graph, gather, embed_init: str, lmhead, vocab: int, hidden: int,
                              fallback_block_size: int) -> dict:
    attrs = _node_attrs(lmhead)
    bits = int(attrs.get("bits", 4))
    block_size = int(attrs.get("block_size", fallback_block_size))
    if bits not in (2, 4, 8):
        raise RuntimeError(f"unsupported lm_head MatMulNBits width bits={bits}")
    if block_size <= 0 or hidden % block_size != 0:
        raise RuntimeError(f"lm_head MatMulNBits block_size={block_size} is incompatible with hidden={hidden}")
    # Symmetric MatMulNBits (e.g. ORT DEFAULT) omits the zero-point input; the
    # implied per-code zero point is the integer midpoint 1 << (bits - 1).
    # Asymmetric nodes carry an explicit per-block zero point as input[3].
    has_zero_point = len(lmhead.input) >= 4 and bool(lmhead.input[3])
    bq, bs = lmhead.input[1], lmhead.input[2]
    bz = lmhead.input[3] if has_zero_point else None
    kb = hidden // block_size
    make = _make_name_factory(graph, "share_embed_lmhead_nbits_")
    ids, out = gather.input[1], gather.output[0]

    axm1 = _make_axes_initializer(graph, make("axis_m1"), [-1])
    rs_qint = _make_axes_initializer(graph, make("reshape_qint"), [0, 0, 0, -1])
    rs_flat = _make_axes_initializer(graph, make("reshape_flat"), [0, 0, -1])

    gq, gs = make("gather_q"), make("gather_s")
    qf, gs1, sub, deq = make("qf"), make("gs1"), make("sub"), make("deq")

    replacement = [
        helper.make_node("Gather", [bq, ids], [gq], axis=0, name=make("gather_q_node")),
        helper.make_node("Gather", [bs, ids], [gs], axis=0, name=make("gather_s_node")),
    ]

    def unpack(packed: str, reshape: str, prefix: str) -> str:
        if bits == 8:
            return packed
        modulus = _make_scalar_initializer(
            graph, make(f"{prefix}_modulus"), np.array(1 << bits, dtype=np.uint8)
        )
        unpacked = []
        for group in range(8 // bits):
            shifted = packed
            if group:
                divisor = _make_scalar_initializer(
                    graph,
                    make(f"{prefix}_divisor_{group}"),
                    np.array(1 << (group * bits), dtype=np.uint8),
                )
                shifted = make(f"{prefix}_shifted_{group}")
                replacement.append(helper.make_node(
                    "Div", [packed, divisor], [shifted], name=make(f"{prefix}_div_{group}")
                ))
            digit = make(f"{prefix}_digit_{group}")
            expanded = make(f"{prefix}_digit_{group}_expanded")
            replacement.extend([
                helper.make_node("Mod", [shifted, modulus], [digit], name=make(f"{prefix}_mod_{group}")),
                helper.make_node("Unsqueeze", [digit, axm1], [expanded], name=make(f"{prefix}_unsq_{group}")),
            ])
            unpacked.append(expanded)
        joined = make(f"{prefix}_joined")
        result = make(f"{prefix}_unpacked")
        replacement.extend([
            helper.make_node("Concat", unpacked, [joined], axis=-1, name=make(f"{prefix}_concat")),
            helper.make_node("Reshape", [joined, reshape], [result], name=make(f"{prefix}_reshape")),
        ])
        return result

    qint = unpack(gq, rs_qint, "q")
    replacement.append(
        helper.make_node("Cast", [qint], [qf], to=TensorProto.FLOAT, name=make("q_cast"))
    )

    if has_zero_point:
        gz = make("gather_z")
        zf, zf1 = make("zf"), make("zf1")
        replacement.append(
            helper.make_node("Gather", [bz, ids], [gz], axis=0, name=make("gather_z_node"))
        )
        zflat = unpack(gz, rs_flat, "z")
        if bits == 8:
            zint = zflat
        else:
            z_start = _make_axes_initializer(graph, make("z_start"), [0])
            z_end = _make_axes_initializer(graph, make("z_end"), [kb])
            z_axis = _make_axes_initializer(graph, make("z_axis"), [2])
            zint = make("zint")
            replacement.append(
                helper.make_node("Slice", [zflat, z_start, z_end, z_axis], [zint], name=make("zslice_node"))
            )
        replacement.extend([
            helper.make_node("Cast", [zint], [zf], to=TensorProto.FLOAT, name=make("z_cast")),
            helper.make_node("Unsqueeze", [zf, axm1], [zf1], name=make("z_unsq")),
        ])
        zero_operand = zf1
    else:
        # Symmetric: dequantize against the implied integer midpoint zero point.
        midpoint = float(1 << (bits - 1))
        zero_operand = _make_scalar_initializer(
            graph, make("z_midpoint"), np.array(midpoint, dtype=np.float32)
        )

    replacement.extend([
        helper.make_node("Unsqueeze", [gs, axm1], [gs1], name=make("s_unsq")),
        helper.make_node("Sub", [qf, zero_operand], [sub], name=make("sub_node")),
        helper.make_node("Mul", [sub, gs1], [deq], name=make("mul_node")),
        helper.make_node("Reshape", [deq, rs_flat], [out], name=make("output_reshape")),
    ])

    _replace_graph_node(graph, gather, replacement)
    _drop_initializers(graph, {embed_init})
    return {"lmhead_op": lmhead.op_type, "bits": bits, "dropped": embed_init, "shared_weight": bq}


def _graph_consumers(graph) -> dict[str, list]:
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _is_dynamic_weight_scale_init(init: TensorProto, vocab: int) -> bool:
    return _tensor_dims(init) in ((), (1,), (vocab,)) and init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)


def _find_dynamic_weight_scale(graph, lmhead, vocab: int) -> str | None:
    inits = _init_map(graph)
    consumers = _graph_consumers(graph)
    producer = {out: node for node in graph.node for out in node.output}

    if lmhead.op_type == "DynamicQuantizeMatMul" and len(lmhead.input) > 2:
        scale = inits.get(lmhead.input[2])
        if scale is not None and _is_dynamic_weight_scale_init(scale, vocab):
            return scale.name

    bq = lmhead.input[1]
    candidates = []
    if bq.endswith("_quantized"):
        candidates.append(bq[:-len("_quantized")] + "_scale")
    if len(lmhead.input) > 3 and lmhead.input[3].endswith("_zero_point"):
        candidates.append(lmhead.input[3][:-len("_zero_point")] + "_scale")
    for name in candidates:
        init = inits.get(name)
        if init is not None and _is_dynamic_weight_scale_init(init, vocab):
            return name

    queue = list(lmhead.output)
    seen: set[str] = set(queue)
    for _ in range(8):
        next_queue = []
        for value in queue:
            for node in consumers.get(value, []):
                if node.op_type == "Mul":
                    for inp in node.input:
                        init = inits.get(inp)
                        if init is not None and _is_dynamic_weight_scale_init(init, vocab):
                            return inp
                        scale_mul = producer.get(inp)
                        if scale_mul is not None and scale_mul.op_type == "Mul":
                            for scale_inp in scale_mul.input:
                                scale_init = inits.get(scale_inp)
                                if scale_init is not None and _is_dynamic_weight_scale_init(scale_init, vocab):
                                    return scale_inp
                for out in node.output:
                    if out not in seen:
                        seen.add(out)
                        next_queue.append(out)
        queue = next_queue
        if not queue:
            break
    return None


def _append_vector_or_scalar_dequant_input(graph, nodes: list, name: str, ids: str, vocab: int,
                                           make, suffix: str) -> str:
    init = _init_map(graph).get(name)
    if init is None:
        raise RuntimeError(f"initializer {name!r} was not found")
    dims = _tensor_dims(init)
    if dims == (vocab,):
        gathered = make(f"{suffix}_gathered")
        expanded = make(f"{suffix}_expanded")
        axis = _make_axes_initializer(graph, make(f"{suffix}_axis"), [-1])
        nodes.extend([
            helper.make_node("Gather", [name, ids], [gathered], axis=0, name=make(f"{suffix}_gather")),
            helper.make_node("Unsqueeze", [gathered, axis], [expanded], name=make(f"{suffix}_unsq")),
        ])
        return expanded
    if dims in ((), (1,)):
        return name
    raise RuntimeError(f"initializer {name!r} has unsupported dynamic lm_head scale/zp shape {dims}")


def _share_dynamic_embed_lmhead(graph, gather, embed_init: str, lmhead, vocab: int, hidden: int) -> dict:
    inits = _init_map(graph)
    bq = lmhead.input[1]
    bq_init = inits.get(bq)
    if bq_init is None:
        raise RuntimeError("dynamic lm_head quantized weight initializer was not found")
    bq_dims = _tensor_dims(bq_init)
    b_scale = _find_dynamic_weight_scale(graph, lmhead, vocab)
    if b_scale is None:
        raise RuntimeError("dynamic lm_head weight scale initializer was not found")

    make = _make_name_factory(graph, "share_embed_lmhead_dyn_")
    ids = gather.input[1]
    consumers = _graph_consumers(graph)
    dq_node = None
    gather_consumers = consumers.get(gather.output[0], [])
    if len(gather_consumers) == 1 and gather_consumers[0].op_type == "DequantizeLinear":
        dq_node = gather_consumers[0]
        out = dq_node.output[0]
    else:
        out = gather.output[0]
    replacement: list = []

    if bq_dims == (vocab, hidden):
        q_bsh = make("q_bsh")
        replacement.append(helper.make_node("Gather", [bq, ids], [q_bsh], axis=0, name=make("gather_q")))
    elif bq_dims == (hidden, vocab):
        q_hbs = make("q_hbs")
        q_bsh = make("q_bsh")
        replacement.extend([
            helper.make_node("Gather", [bq, ids], [q_hbs], axis=1, name=make("gather_q")),
            helper.make_node("Transpose", [q_hbs], [q_bsh], perm=[1, 2, 0], name=make("transpose_q")),
        ])
    else:
        raise RuntimeError(f"dynamic lm_head weight has unsupported shape {bq_dims}")

    if len(lmhead.input) > 3 and lmhead.input[3]:
        b_zp = lmhead.input[3]
    else:
        dtype = TensorProto.UINT8 if bq_init.data_type == TensorProto.UINT8 else TensorProto.INT8
        zero = np.array(0, dtype=np.uint8 if dtype == TensorProto.UINT8 else np.int8)
        b_zp = _make_scalar_initializer(graph, make("zero_point"), zero)

    scale = _append_vector_or_scalar_dequant_input(graph, replacement, b_scale, ids, vocab, make, "scale")
    zp = _append_vector_or_scalar_dequant_input(graph, replacement, b_zp, ids, vocab, make, "zp")
    qf, zf, sub = make("qf"), make("zf"), make("sub")
    replacement.extend([
        helper.make_node("Cast", [q_bsh], [qf], to=TensorProto.FLOAT, name=make("q_cast")),
        helper.make_node("Cast", [zp], [zf], to=TensorProto.FLOAT, name=make("zp_cast")),
        helper.make_node("Sub", [qf, zf], [sub], name=make("sub_node")),
        helper.make_node("Mul", [sub, scale], [out], name=make("mul_node")),
    ])

    if dq_node is None:
        _replace_graph_node(graph, gather, replacement)
    else:
        nodes = []
        gather_id, dq_id = id(gather), id(dq_node)
        for node in graph.node:
            if id(node) == gather_id:
                nodes.extend(replacement)
            elif id(node) != dq_id:
                nodes.append(node)
        graph.ClearField("node")
        graph.node.extend(nodes)
    _drop_initializers(graph, {embed_init})
    return {"lmhead_op": lmhead.op_type, "dropped": embed_init, "shared_weight": bq}


def unify_embed_lmhead_graph(model: onnx.ModelProto, method: str, block_size: int = 32,
                             quiet: bool = False) -> dict | None:
    """Share Main's tied embedding with the lm_head weight in place (no disk I/O)."""
    graph = model.graph
    try:
        gather, embed_init, vocab, hidden = _find_embed_gather(graph)
        _, lmhead = _find_lmhead(graph, vocab, hidden)
    except RuntimeError as exc:
        if not quiet:
            print(f"  share_embed_lmhead: skipped ({exc}).")
        return None

    method = method.upper()
    if method in ("F32", "F16"):
        info = _share_float_embed_lmhead(graph, gather, embed_init, lmhead, vocab, hidden)
    elif method == "NBITS":
        if lmhead.op_type != "MatMulNBits":
            raise RuntimeError(f"NBITS share_embed_lmhead expected MatMulNBits lm_head, got {lmhead.op_type}")
        info = _share_nbits_embed_lmhead(graph, gather, embed_init, lmhead, vocab, hidden, block_size)
    elif method == "DYNAMIC":
        if lmhead.op_type not in ("MatMulInteger", "DynamicQuantizeMatMul"):
            raise RuntimeError(
                "DYNAMIC share_embed_lmhead expected MatMulInteger or "
                f"DynamicQuantizeMatMul lm_head, got {lmhead.op_type}"
            )
        info = _share_dynamic_embed_lmhead(graph, gather, embed_init, lmhead, vocab, hidden)
    else:
        raise ValueError(f"unknown share_embed_lmhead method {method!r}")

    _dead_code_elimination(graph)
    _drop_unused_initializers(graph)
    _deduplicate_node_names(graph)
    return info


def _unify_method_kind(rp: ResolvedPlan) -> str:
    if rp.method in _WEIGHT_ONLY_BITS:
        return "NBITS"
    if rp.method == "DYNAMIC":
        return "DYNAMIC"
    return "F16" if (rp.fp16 or rp.method == "F16") else "F32"


def inspect_kv_surgery(graph) -> tuple[bool, str]:
    inputs = {i.name: i for i in graph.input}
    keys = [n for n in inputs if n.startswith("in_key_")
            and not n.startswith("in_key_scale") and not n.startswith("in_key_bias")]
    if not keys:
        return False, "no KV cache inputs (not an attention module) — skipped"
    elem = inputs[keys[0]].type.tensor_type.elem_type
    if elem not in (TensorProto.INT8, TensorProto.UINT8, TensorProto.INT32):
        return False, f"KV is not int8/uint8/int32 (elem_type={elem}); surgery targets Q8/ROTARY_Q8/Q8_CUDA — skipped"
    scale = next((inputs[n] for n in inputs if n.startswith("in_key_scale_")), None)
    grouped = False
    if scale is not None:
        rank = len(scale.type.tensor_type.shape.dim)
        if rank == 6:
            grouped = True
        elif rank != 5:
            return False, f"unexpected key_scale rank {rank} (per-head=5, grouped=6) — skipped"
    if grouped and elem == TensorProto.INT32:
        return False, "grouped Q8_CUDA (int32-packed) KV — grouped surgery is non-CUDA (Q8/ROTARY_Q8) only — skipped"
    inits = {i.name for i in graph.initializer}
    if not any(n.op_type == "MatMul" and n.input[1] not in inits for n in graph.node):
        return False, "no activation@activation matmuls to rewrite — skipped"
    asym = any(n.startswith("in_key_bias_") for n in inputs)
    scheme = "asymmetric" if asym else "symmetric"
    layout = "grouped" if grouped else "per-head"
    kind = f"Q8_CUDA int32-packed ({scheme})" if elem == TensorProto.INT32 else \
           (f"{scheme} " + ("uint8" if elem == TensorProto.UINT8 else "int8"))
    family = "Q8/ROTARY_Q8/ROTARY_Q4" if grouped else "Q8/ROTARY_Q8"
    return True, f"{kind} KV ({family}), {layout}"


def inspect_rope_shift_surgery(graph) -> tuple[bool, str]:
    inputs = {i.name: i for i in graph.input}
    keys = [n for n in inputs if n.startswith("in_key_")
            and not n.startswith("in_key_scale") and not n.startswith("in_key_bias")]
    if not keys:
        return False, "no in_key_* inputs (not a rope-shift module) — skipped"
    if any(n.op_type == "MatMul" for n in graph.node):
        return False, "has MatMul (attention module, not rope-shift) — skipped"
    elem = inputs[keys[0]].type.tensor_type.elem_type
    if elem in (TensorProto.FLOAT, TensorProto.FLOAT16):
        return False, "float (F16/F32) rope-shift has no quant/dequant to convert — skipped"
    if elem not in (TensorProto.INT8, TensorProto.UINT8):
        return False, f"non-int8/uint8 KV (elem_type={elem}); rope-shift Q/DQ surgery targets Q8/ROTARY_Q8 — skipped"
    is_asym = any(n.startswith("in_key_bias_") for n in inputs)
    if (elem == TensorProto.UINT8) != is_asym:
        return False, "KV dtype/bias mismatch (int8 must be symmetric, uint8 must carry bias) — skipped"
    dims = inputs[keys[0]].type.tensor_type.shape.dim
    if len(dims) != 5 or dims[3].dim_value <= 0:
        return False, "unexpected key layout (need static per-head axis-3 head_dim) — skipped"
    scale = next((inputs[n] for n in inputs if n.startswith("in_key_scale_")), None)
    if scale is None or len(scale.type.tensor_type.shape.dim) != 5:
        return False, "grouped/absent key_scale (rank != 5) — rope-shift Q/DQ supports per-head layout only — skipped"
    ops = {n.op_type for n in graph.node}
    if not ({"Div", "Round", "ReduceMax"} <= ops):
        return False, "no quantize tail (Div/Round) — not a quantized rope-shift — skipped"
    scheme = "asymmetric uint8+bias" if is_asym else "symmetric int8"
    return True, f"{scheme} rope-shift (Q8/ROTARY_Q8), per-head"


def inspect_kv_quantize_surgery(graph) -> tuple[bool, str]:
    inputs = {i.name: i for i in graph.input}
    keys = [n for n in inputs if n.startswith("in_key_") and "scale" not in n and "bias" not in n]
    if not keys:
        return False, "no KV cache inputs (not an attention module) — skipped"
    elem = inputs[keys[0]].type.tensor_type.elem_type
    is_asym = any(n.startswith("in_key_bias_") for n in inputs)
    if is_asym and elem != TensorProto.UINT8:
        return False, "asymmetric KV is not uint8 (Q8_CUDA int32 write tail unsupported) — skipped"
    if not is_asym and elem != TensorProto.INT8:
        return False, "symmetric KV is not int8 (Q8_CUDA int32 / f16 write tail unsupported) — skipped"
    scale = next((inputs[n] for n in inputs if n.startswith("in_key_scale_")), None)
    if scale is None or len(scale.type.tensor_type.shape.dim) != 5:
        return False, "grouped/absent key_scale (rank != 5) — per-head write tail only — skipped"
    kdims = inputs[keys[0]].type.tensor_type.shape.dim
    if len(kdims) != 5 or not kdims[3].HasField("dim_value") or kdims[3].dim_value <= 0:
        return False, "no static per-head head_dim on the key cache — skipped"
    return True, f"per-head {'asymmetric uint8+bias' if is_asym else 'symmetric int8'} write tail"


def rewire_attention_to_dynamic_quantize_matmul(model) -> tuple[int, int]:
    graph = model.graph
    inits = {i.name for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    # Needed for grouped KV source dtype checks.
    elem_of: dict[str, int] = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            elem_of[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        elem_of[init.name] = init.data_type

    key_ins = [i for i in graph.input
               if i.name.startswith("in_key_") and "scale" not in i.name and "bias" not in i.name]
    kv_elem = key_ins[0].type.tensor_type.elem_type if key_ins else TensorProto.INT8
    is_cuda = (kv_elem == TensorProto.INT32)
    is_asym = any(i.name.startswith("in_key_bias_") for i in graph.input)
    target_dtype = TensorProto.UINT8 if is_asym else TensorProto.INT8

    bzp_i8, bzp_u8 = "kvsurg_bzp_i8", "kvsurg_bzp_u8"
    for name, arr in ((bzp_i8, np.array(0, np.int8)), (bzp_u8, np.array(0, np.uint8))):
        if name not in inits:
            graph.initializer.append(numpy_helper.from_array(arr, name=name))
            inits.add(name)
    target_bzp = bzp_u8 if is_asym else bzp_i8
    if not any(op.domain == "com.microsoft" for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    def one_f32(name: str) -> str:
        if name not in inits:
            graph.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), name=name))
            inits.add(name)
        return name

    def prep_b(traced: str, pfx: str, tag: str) -> tuple[str, list]:
        if not is_cuda:
            return traced, []
        cast_out = f"{pfx}_{tag}_bcast"
        return cast_out, [helper.make_node("Cast", [traced], [cast_out], to=target_dtype, name=cast_out)]

    new_nodes, n_qk, n_pv = [], 0, 0
    for idx, node in enumerate(graph.node):
        if node.op_type != "MatMul" or node.input[1] in inits:
            new_nodes.append(node)
            continue
        a, b, out = node.input[0], node.input[1], node.output[0]
        pfx = (node.name.replace("/", "_") or "kvsurg") + f"_{idx}"
        is_pv = a in producer and producer[a].op_type == "Softmax"
        if not is_pv:
            b_prod = producer.get(b)
            if b_prod is not None and b_prod.op_type == "Reshape":
                k_src = _src_through_casts(b_prod.input[0], producer)
                k_src_elem = elem_of.get(k_src)
                if is_cuda:
                    new_nodes.append(node)
                    continue
                k_in = f"{pfx}_qk_kre"
                if k_src_elem == TensorProto.INT16:
                    k_i8 = f"{pfx}_qk_ki8"
                    casts = [
                        helper.make_node("Cast", [k_src], [k_i8], to=TensorProto.INT8, name=f"{pfx}_qk_kcast"),
                        helper.make_node("Reshape", [k_i8, b_prod.input[1]], [k_in], name=f"{pfx}_qk_reshape"),
                    ]
                    qk_bzp = bzp_i8
                elif k_src_elem in (TensorProto.INT8, TensorProto.UINT8):
                    casts = [helper.make_node("Reshape", [k_src, b_prod.input[1]], [k_in],
                                              name=f"{pfx}_qk_reshape")]
                    qk_bzp = bzp_u8 if k_src_elem == TensorProto.UINT8 else bzp_i8
                else:
                    new_nodes.append(node)
                    continue
            else:
                k_in, casts = prep_b(_src_through_casts(b, producer), pfx, "qk")
                qk_bzp = target_bzp
            new_nodes.extend(casts)
            new_nodes.append(helper.make_node(
                "DynamicQuantizeMatMul",
                [a, k_in, one_f32(f"{pfx}_qk_one_f32"), qk_bzp],
                [out],
                name=f"{pfx}_qk_dqmm",
                domain="com.microsoft",
            ))
            n_qk += 1
        else:
            bp = producer.get(b)
            if bp is None:
                new_nodes.append(node)
                continue
            if bp.op_type == "Add":
                left_mul, right_mul = producer.get(bp.input[0]), producer.get(bp.input[1])
                left_split = _split_value_scale_mul(left_mul, producer) if left_mul is not None else None
                right_split = _split_value_scale_mul(right_mul, producer) if right_mul is not None else None
                if left_split is not None and right_split is None:
                    v_traced, v_scale_f = left_split
                    v_bias = bp.input[1]
                elif right_split is not None and left_split is None:
                    v_traced, v_scale_f = right_split
                    v_bias = bp.input[0]
                else:
                    new_nodes.append(node)
                    continue
            else:
                split = _split_value_scale_mul(bp, producer)
                if split is None:
                    new_nodes.append(node)
                    continue
                v_traced, v_scale_f = split
                v_bias = None
            v_in, casts = prep_b(_src_through_casts(v_traced, producer), pfx, "pv")
            vst, ps = f"{pfx}_pv_vst", f"{pfx}_pv_ps"
            main = out if v_bias is None else f"{pfx}_pv_main"
            new_nodes.extend(casts)
            new_nodes.extend([
                helper.make_node("Transpose", [v_scale_f], [vst], perm=[0, 1, 2, 4, 3], name=f"{pfx}_pv_tr"),
                helper.make_node("Mul", [a, vst], [ps], name=f"{pfx}_pv_mul"),
                helper.make_node(
                    "DynamicQuantizeMatMul",
                    [ps, v_in, one_f32(f"{pfx}_pv_one_f32"), target_bzp],
                    [main],
                    name=f"{pfx}_pv_dqmm",
                    domain="com.microsoft",
                ),
            ])
            if v_bias is not None:
                biasmm = f"{pfx}_pv_biasmm"
                new_nodes.extend([
                    helper.make_node("MatMul", [a, v_bias], [biasmm], name=f"{pfx}_pv_biasmm"),
                    helper.make_node("Add", [main, biasmm], [out], name=f"{pfx}_pv_biasadd"),
                ])
            n_pv += 1

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)
    return n_qk, n_pv


def rewire_rope_shift_to_qdq(model) -> int:
    graph = model.graph
    inputs = {i.name: i for i in graph.input}
    producer = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)

    key_inputs = [i.name for i in graph.input
                  if i.name.startswith("in_key_")
                  and not i.name.startswith("in_key_scale")
                  and not i.name.startswith("in_key_bias")]
    is_asym = any(i.name.startswith("in_key_bias_") for i in graph.input)
    zp_dtype = TensorProto.UINT8 if is_asym else TensorProto.INT8
    kv_axis = 3
    head_dim = inputs[key_inputs[0]].type.tensor_type.shape.dim[kv_axis].dim_value

    def single_consumer(t):
        return len(consumers.get(t, [])) == 1

    to_delete, replace, count = set(), {}, 0
    for kin in key_inputs:
        idx = kin.rsplit("_", 1)[1]
        sin, kout = f"in_key_scale_{idx}", f"out_key_{idx}"

        cast_chain, cur = [], kin
        while True:
            nxt = [n for n in consumers.get(cur, []) if n.op_type == "Cast"]
            if len(nxt) != 1:
                break
            cast_chain.append(nxt[0])
            cur = nxt[0].output[0]
        if not cast_chain:
            continue
        muls = [n for n in consumers.get(cast_chain[-1].output[0], []) if n.op_type == "Mul"]
        if len(muls) != 1:
            continue
        mul = muls[0]
        scale_operand = [x for x in mul.input if x != cast_chain[-1].output[0]]
        if len(scale_operand) != 1:
            continue
        sc32 = scale_operand[0]
        sc_prod = producer.get(sc32)
        if not (sc32 == sin or (sc_prod is not None and sc_prod.op_type == "Cast"
                                and sc_prod.input and sc_prod.input[0] == sin)):
            continue

        node, tail = producer.get(kout), []
        while node is not None and node.op_type in ("Cast", "Clip", "Round"):
            tail.append(node)
            node = producer.get(node.input[0])
        if not tail or node is None or node.op_type != "Div":
            continue
        div = node
        # Skip residual-corrected asymmetric tails that share Round output.
        if not all(single_consumer(n.output[0]) for n in tail[1:]) or not single_consumer(div.output[0]):
            continue
        x_q, scale_new = div.input[0], div.input[1]

        dql = helper.make_node(
            "DequantizeLinear", [kin, sc32], [mul.output[0]],
            axis=kv_axis, block_size=head_dim, name=f"ropeq_dql_{idx}")
        sshape, zp5 = f"ropeq_scale_shape_{idx}", f"ropeq_zero_point_{idx}"
        shape_node = helper.make_node("Shape", [scale_new], [sshape], name=f"ropeq_shape_{idx}")
        zp_node = helper.make_node(
            "ConstantOfShape", [sshape], [zp5],
            value=helper.make_tensor(f"ropeq_zero_val_{idx}", zp_dtype, [1], [0]),
            name=f"ropeq_zero_{idx}")
        ql = helper.make_node(
            "QuantizeLinear", [x_q, scale_new, zp5], [kout],
            axis=kv_axis, block_size=head_dim, name=f"ropeq_ql_{idx}")

        to_delete.update(id(c) for c in cast_chain)
        to_delete.update(id(n) for n in tail[1:])
        to_delete.add(id(div))
        replace[id(mul)] = [dql]
        replace[id(tail[0])] = [shape_node, zp_node, ql]
        count += 1

    if count == 0:
        return 0

    new_nodes = []
    for n in graph.node:
        if id(n) in to_delete:
            continue
        new_nodes.extend(replace.get(id(n), [n]))
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)

    _ensure_default_opset21(model)
    return count


def rewire_kv_quantize_to_quantizelinear(model) -> int:
    graph = model.graph
    ok, _ = inspect_kv_quantize_surgery(graph)
    if not ok:
        return 0
    producer = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    init_map = {i.name: i for i in graph.initializer}

    key_in = next(i for i in graph.input
                  if i.name.startswith("in_key_") and "scale" not in i.name and "bias" not in i.name)
    head_dim = key_in.type.tensor_type.shape.dim[3].dim_value  # block_size (per-head head_dim, static)
    is_asym = any(i.name.startswith("in_key_bias_") for i in graph.input)
    zp_dtype = TensorProto.UINT8 if is_asym else TensorProto.INT8

    def find_reduce(name):
        p = producer.get(name)
        if p is None:
            return None
        if p.op_type in ("ReduceMax", "ReduceMin"):
            return p
        if p.op_type == "Sub":
            for si in p.input:
                r = find_reduce(si)
                if r is not None:
                    return r
        return None

    to_delete, replace, count = set(), {}, 0
    for div in graph.node:
        if div.op_type != "Div":
            continue
        # Skip residual-corrected asymmetric tails that share Round output.
        rs = consumers.get(div.output[0], [])
        if len(rs) != 1 or rs[0].op_type != "Round":
            continue
        rnd = rs[0]
        cur, clip_nodes = rnd.output[0], []
        cs = consumers.get(cur, [])
        if len(cs) == 1 and cs[0].op_type == "Clip":
            clip_nodes = [cs[0]]
            cur = cs[0].output[0]
        cast_chain = []
        while True:
            nxt = consumers.get(cur, [])
            if len(nxt) == 1 and nxt[0].op_type == "Cast":
                cast_chain.append(nxt[0])
                cur = nxt[0].output[0]
            else:
                break
        if not cast_chain:
            continue
        packed = cur
        concat = next((c for c in consumers.get(packed, []) if c.op_type == "Concat"), None)
        if concat is None or not (concat.output[0].startswith("out_key_")
                                  or concat.output[0].startswith("out_value_")):
            continue
        a, scale = div.input[0], div.input[1]
        smul = producer.get(scale)
        if smul is None or smul.op_type != "Mul":
            continue
        rmax = None
        for s in smul.input:
            rmax = find_reduce(s)
            if rmax is not None:
                break
        if rmax is None:
            continue
        axis = _reduce_single_axis(rmax, producer, init_map)
        if axis is None:
            continue
        if axis < 0:
            axis += 5  # KV tensors are rank 5: (B, KVH, 1, head_dim, S) key / (B, KVH, 1, S, head_dim) value

        sshape, zp = f"kvq_scale_shape_{count}", f"kvq_zero_point_{count}"
        shape_node = helper.make_node("Shape", [scale], [sshape], name=f"kvq_shape_{count}")
        zp_node = helper.make_node(
            "ConstantOfShape", [sshape], [zp],
            value=helper.make_tensor(f"kvq_zero_val_{count}", zp_dtype, [1], [0]),
            name=f"kvq_zero_{count}")
        ql = helper.make_node(
            "QuantizeLinear", [a, scale, zp], [packed],
            axis=axis, block_size=head_dim, name=f"kvq_ql_{count}")

        replace[id(div)] = [shape_node, zp_node, ql]     # Div -> Shape + ConstantOfShape + QuantizeLinear
        to_delete.add(id(rnd))
        to_delete.update(id(c) for c in clip_nodes)      # Round + optional Clip + int Casts removed
        to_delete.update(id(c) for c in cast_chain)      # (Sub(x,min) + scale calc + f16 caches kept)
        count += 1

    if count == 0:
        return 0
    new_nodes = []
    for n in graph.node:
        if id(n) in to_delete:
            continue
        new_nodes.extend(replace.get(id(n), [n]))
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)
    _ensure_default_opset21(model)
    return count


def apply_kv_surgery(model) -> None:
    applicable, _ = inspect_kv_surgery(model.graph)
    if applicable:
        n_qk, n_pv = rewire_attention_to_dynamic_quantize_matmul(model)
        n_q = rewire_kv_quantize_to_quantizelinear(model) if KV_BLOCKED_QDQ_SURGERY else 0
        message = f"    surgery: {n_qk} Q@K + {n_pv} attn@V -> DynamicQuantizeMatMul"
        if n_q:
            message += f"; {n_q} KV write tails -> QuantizeLinear (blocked int8)"
        elif not KV_BLOCKED_QDQ_SURGERY:
            message += "; preserved arithmetic KV write tails (CUDA-compatible)"
        print(message)
        return
    if not KV_BLOCKED_QDQ_SURGERY:
        return
    applicable, _ = inspect_rope_shift_surgery(model.graph)
    if applicable:
        n = rewire_rope_shift_to_qdq(model)
        print(f"    surgery: {n} rope-shift layers -> DequantizeLinear/QuantizeLinear (blocked int8)")


def plan_kv_surgery(src_path: str) -> tuple[bool, str]:
    meta = onnx.load(src_path, load_external_data=False)
    try:
        applicable, reason = inspect_kv_surgery(meta.graph)
        if applicable:
            tail_note = (
                " + blocked Q/DQ write tails"
                if KV_BLOCKED_QDQ_SURGERY
                else "; arithmetic write tails retained for CUDA"
            )
            return True, f"applying ({reason}) -> DynamicQuantizeMatMul{tail_note}, in-memory"
        rope_ok, rope_reason = inspect_rope_shift_surgery(meta.graph)
        if rope_ok:
            if not KV_BLOCKED_QDQ_SURGERY:
                return False, f"{rope_reason}; blocked Q/DQ disabled for CUDA compatibility"
            return True, f"applying ({rope_reason}) -> DequantizeLinear/QuantizeLinear, in-memory"
        for r in (reason, rope_reason):
            if "not an attention module" not in r and "not a rope-shift module" not in r:
                return False, r
        return False, reason
    finally:
        del meta


def process_model(name: str, rp: ResolvedPlan) -> None:
    src_path, dst_path = get_model_paths(name)
    if not os.path.exists(src_path):
        print(f"  Skipping — file not found: {src_path}")
        return

    source_metadata = read_onnx_metadata(src_path)
    preserve_fp16_compute = source_metadata.get("compute_in_f32", "1").lower() in ("0", "false")

    _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = MIXED_PRECISION if F16_KEEP_IO_TYPES is None else F16_KEEP_IO_TYPES

    do_surgery = False
    if rp.kv_surgery is not False:
        do_surgery, message = plan_kv_surgery(src_path)
        print(f"  KV/rope-shift surgery: {message}")
    if do_surgery and use_fp16:
        print(
            "  Surgery: disabled for float16 conversion; ORT's fp16 converter can otherwise leave the "
            "quantized island with invalid mixed f32/f16 types."
        )
        do_surgery = False

    if rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(src_path, dst_path, rp, _WEIGHT_ONLY_BITS[rp.method], external, do_surgery)
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external, do_surgery)
    else:  # F16 / F32 — no integer quant; F16 is applied during the optimize stage.
        resave(src_path, dst_path, external, do_surgery)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        if preserve_fp16_compute and OPTIMIZER_LEVEL > 0:
            print("  Preserving COMPUTE_IN_F32=0: skipping ORT's CPU precision rewrite.")
        run_onnxslim(dst_path, external, no_shape_infer=True)
        heads, hidden = fetch_transformer_config(DOWNLOAD_PATH) if "Main" in name else (0, 0)
        optimize_onnx_model(
            dst_path,
            heads,
            hidden,
            use_fp16,
            external,
            keep_io_types,
            preserve_fp16_compute,
        )
        run_onnxslim(dst_path, external, no_shape_infer=not SHAPE_INFER)

    if UPGRADE_OPSET > 0:
        upgrade_opset_version(dst_path, UPGRADE_OPSET, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")

    validated = onnx.load(dst_path, load_external_data=False)
    if use_fp16:
        restored_outputs = _restore_kv_scale_outputs(validated)
        if restored_outputs:
            onnx.save(validated, dst_path)
            print(f"  Restored {restored_outputs} FP16 KV scale graph-output boundaries.")
    _validate_graph_references(validated, os.path.basename(dst_path))
    del validated

    # activations_fp16 is the only metadata value changed by optimization.
    if source_metadata:
        if use_fp16:
            source_metadata["activations_fp16"] = "1"
        write_onnx_metadata(dst_path, source_metadata)
        fp16_note = " (activations_fp16=1)" if use_fp16 else ""
        print(f"  Metadata: restamped {len(source_metadata)} keys onto the optimized model{fp16_note}.")


def _print_process_header(name: str, rp: ResolvedPlan) -> None:
    print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}]\n{'=' * 60}")


def _cleanup_merged_outputs(out_folder: Path, model_file_names: dict[str, str]) -> None:
    for file_name, _, _ in Shared_Merged.make_merged_build_plan(model_file_names):
        _remove_external_files(str(out_folder / file_name))
    shared_name = model_file_names.get("shared_initializers", Shared_Merged.SHARED_MODEL_NAME)
    _remove_external_files(str(out_folder / shared_name))


def _available_merged_files(source_folder: Path, model_file_names: dict[str, str]) -> list[str]:
    return [
        file_name
        for file_name, _, _ in Shared_Merged.make_merged_build_plan(model_file_names)
        if (source_folder / file_name).exists()
    ]


def _merged_metadata(primary_path: Path, model_file_names: dict[str, str]) -> dict:
    metadata = read_onnx_metadata(str(primary_path))
    if metadata and model_file_names:
        metadata.update({f"model_file_name_{key}": value for key, value in model_file_names.items()})
    return metadata


def _save_merged_model(
    out_folder: Path,
    file_name: str,
    model: onnx.ModelProto,
    metadata: dict,
) -> None:
    out_path = out_folder / file_name
    _validate_graph_references(model, file_name)
    if metadata:
        _update_onnx_metadata(model, metadata)
    Shared_Merged.save_model(model, out_path)
    print(f"  {file_name} ({out_path.stat().st_size} bytes)")


def _load_transplant_donor(primary_path: Path) -> onnx.ModelProto:
    # The donor stays un-unified so embedding reconstruction nodes cannot leak
    # into the Main block copied across strategies.
    donor = onnx.load(str(primary_path), load_external_data=False)
    if _restore_prefill_mask_shell_boundary(donor):
        print("  Restored optimized donor prefill-mask/Main boundary before transplantation.")
    return donor


def _validate_quantized_embedding(primary_path: Path, primary_plan: ResolvedPlan) -> None:
    if primary_plan.algo != "AFFINE_REFINE_V2" or "Gather" not in primary_plan.op_types:
        return
    model = onnx.load(str(primary_path), load_external_data=False)
    count = sum(node.op_type == "GatherBlockQuantized" for node in model.graph.node)
    del model
    if count == 0:
        raise RuntimeError(
            f"{primary_path.name} requested embedding Gather quantization but the final "
            "graph has no GatherBlockQuantized node."
        )
    print(f"  Verified {count} quantized embedding GatherBlockQuantized node(s).")


def _load_unified_primary(
    primary_path: Path,
    method_kind: str,
    block_size: int,
    share_embed_lmhead: bool,
) -> tuple[onnx.ModelProto, dict | None]:
    # Drop the duplicate fp32 embedding before loading external data to bound
    # peak memory to the surviving quantized Main weights.
    model = onnx.load(str(primary_path), load_external_data=False)
    info = (
        unify_embed_lmhead_graph(model, method_kind, block_size=block_size, quiet=True)
        if share_embed_lmhead
        else None
    )
    if info is not None:
        print(
            f"  Shared embed/lm_head: dropped {info['dropped']!r}; "
            f"embedding now reuses {info['shared_weight']!r} ({info['lmhead_op']})."
        )
    _drop_unused_initializers(model.graph)
    load_external_data_for_model(model, str(primary_path.parent))
    return model, info


def _materialize_transplant_donor(
    donor: onnx.ModelProto,
    primary_path: Path,
    unify_info: dict | None,
    external_by_name: dict[str, dict[str, str]],
) -> None:
    # The optimized primary is about to replace its private sidecar. Materialize
    # the donor first, then redirect its shared weights to the new shared blob.
    if unify_info is not None and unify_info["dropped"] != unify_info["shared_weight"]:
        _drop_initializers(donor.graph, {unify_info["dropped"]})
    load_external_data_for_model(donor, str(primary_path.parent))
    Shared_Merged.redirect_shared_initializers_to_external(donor, external_by_name)


def _transplant_merged_strategies(
    source_folder: Path,
    out_folder: Path,
    available_files: list[str],
    primary_file: str,
    donor: onnx.ModelProto,
    method_kind: str,
    block_size: int,
    share_embed_lmhead: bool,
    external_by_name: dict[str, dict[str, str]],
    metadata: dict,
) -> None:
    for file_name in available_files:
        if file_name == primary_file:
            continue
        # Load structure only: transplantation replaces the target's multi-GB
        # Main, and shared initializers are redirected before the graph is saved.
        target = onnx.load(str(source_folder / file_name), load_external_data=False)
        model = Shared_Merged.transplant_quantized_main(target, donor)
        del target
        if method_kind == "F16":
            aligned_boundaries = _align_fp16_shell_boundaries(model)
            print(f"  {file_name}: aligned {aligned_boundaries} rotary/mask boundaries to FP16.")
            if share_embed_lmhead:
                unify_embed_lmhead_graph(
                    model, method_kind, block_size=block_size, quiet=True
                )
            model = _convert_transplanted_model_to_fp16(model)
        dropped_constants = _drop_shadowed_constant_nodes(model.graph)
        if dropped_constants:
            print(f"  {file_name}: dropped {dropped_constants} shadowed Constant node(s).")
        if share_embed_lmhead:
            unify_embed_lmhead_graph(model, method_kind, block_size=block_size, quiet=True)
        fused_reshapes = fuse_consecutive_reshapes_graph(model.graph)
        if fused_reshapes:
            print(f"  {file_name}: fused {fused_reshapes} semantics-safe consecutive Reshape pairs.")
        _drop_unused_initializers(model.graph)
        Shared_Merged.redirect_shared_initializers_to_external(model, external_by_name)
        _save_merged_model(out_folder, file_name, model, metadata)
        del model
        gc.collect()


def build_quantized_merged_bundle(resolved: dict[str, ResolvedPlan]) -> None:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    out_folder = Path(QUANTED_FOLDER_PATH)
    model_file_names = _metadata_model_file_names(source_folder)
    available_files = _available_merged_files(source_folder, model_file_names)
    if not available_files:
        print("\nMerged decode-strategy graphs not found; skipping merged bundle optimization.")
        return

    _cleanup_merged_outputs(out_folder, model_file_names)

    primary_file = model_file_names.get("prefill_greedy", Shared_Merged.PREFILL_GREEDY_MODEL_NAME)
    if not (source_folder / primary_file).exists():
        primary_file = available_files[0]
    primary_stem = Path(primary_file).stem
    primary_plan = resolved.get(primary_stem, resolved[_PRIMARY_MERGED_MODEL])
    share_embed_lmhead = _source_embed_lmhead_equal(source_folder / primary_file)
    _configure_embedding_quantization(primary_plan, share_embed_lmhead)
    print(
        "  Embed/lm_head check: "
        + ("identical; sharing enabled." if share_embed_lmhead else
            "different; sharing disabled and embedding Gather quantization enabled."
            if "Gather" in primary_plan.op_types else
            "different; sharing disabled.")
    )

    _print_process_header(primary_stem, primary_plan)
    process_model(primary_stem, primary_plan)

    primary_path = out_folder / primary_file
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    _validate_quantized_embedding(primary_path, primary_plan)

    method_kind = _unify_method_kind(primary_plan)
    shared_model_name = model_file_names.get("shared_initializers", Shared_Merged.SHARED_MODEL_NAME)
    shared_data_name = model_file_names.get("shared_initializers_data", shared_model_name + ".data")
    source_metadata = _merged_metadata(primary_path, model_file_names)
    metadata_name = model_file_names.get("metadata", "LLM_Metadata.onnx")
    metadata_path = out_folder / metadata_name
    if source_metadata and metadata_path.exists():
        write_onnx_metadata(str(metadata_path), source_metadata)
        print(f"  Metadata carrier: synchronized {len(source_metadata)} keys in {metadata_name}.")

    print(f"\n{'=' * 60}\nTransplanting quantized Main into merged strategy graphs\n{'=' * 60}")

    donor = _load_transplant_donor(primary_path)
    primary_model, unify_info = _load_unified_primary(
        primary_path, method_kind, primary_plan.block_size, share_embed_lmhead
    )
    external_by_name = Shared_Merged.extract_and_write_shared(
        [primary_model],
        out_folder / shared_model_name,
        primary_model=primary_model,
    )

    _materialize_transplant_donor(donor, primary_path, unify_info, external_by_name)

    _save_merged_model(out_folder, primary_file, primary_model, source_metadata)
    del primary_model
    gc.collect()

    _transplant_merged_strategies(
        source_folder,
        out_folder,
        available_files,
        primary_file,
        donor,
        method_kind,
        primary_plan.block_size,
        share_embed_lmhead,
        external_by_name,
        source_metadata,
    )
    del donor
    gc.collect()

    shared_data = out_folder / shared_data_name
    if shared_data.exists():
        print(f"  {shared_data_name} ({shared_data.stat().st_size} bytes)")

    for removed in Shared_Merged.delete_merged_constituents(
        out_folder, protected_names=(shared_model_name, shared_data_name)
    ):
        print(f"  Deleted absorbed split constituent: {removed}")


def main() -> None:
    os.makedirs(QUANTED_FOLDER_PATH, exist_ok=True)

    resolved = {name: resolve_plan(plan) for name, plan in MODEL_PLANS.items()}
    for name, rp in resolved.items():
        validate_plan(name, rp)

    if MIXED_PRECISION and F16_KEEP_IO_TYPES is None:
        print(
            "TIP: mixed float16/float32 modules detected — forcing keep_io_types=True on "
            "float16 conversions so shared graph I/O (KV cache, hidden states) stays "
            "float32-compatible across the split graphs."
        )
    for name, rp in resolved.items():
        if name in _MERGED_MODEL_NAMES:
            continue
        _print_process_header(name, rp)
        process_model(name, rp)
    build_quantized_merged_bundle(resolved)
    print("\n--- All models processed successfully! ---")


if __name__ == "__main__":
    main()
