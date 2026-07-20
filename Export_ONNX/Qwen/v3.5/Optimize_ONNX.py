"""Optimize exported split ONNX modules.

Edit USER CONFIG for defaults and MODEL_PLANS for per-module overrides. Methods:
Q2/Q4/Q8 = MatMulNBits, DYNAMIC = INT8 dynamic, F16 = fp16, F32 = optimize only.
"""

import os
import gc
import sys
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass

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

import Shared_Merged


# ============================== USER CONFIG ==============================

# --- Folders / source model (edit these three to target a different model) ----
# The LLM_* module names in MODEL_PLANS are shared across the exported model family.
ORIGINAL_FOLDER_PATH           = str(_SCRIPT_DIR / "Qwen_ONNX")             # Folder holding the exported *.onnx modules.
QUANTED_FOLDER_PATH            = str(_SCRIPT_DIR / "Qwen_Optimized")        # Destination folder for the results.
DOWNLOAD_PATH                  = str(Path.home() / "Downloads" / "Qwen3.5-0.8B")  # Model dir (attention fusion); "NONE" to skip.

# --- Weight-only quantization defaults (Q2 / Q4 / Q8 -> MatMulNBits) -----------
WEIGHT_ONLY_ALGORITHM          = "k_quant"                           # "k_quant" | "DEFAULT" | "RTN" | "HQQ". k_quant/RTN are Q4-only; use DEFAULT/HQQ for Q2/Q8.
BLOCK_SIZE                     = 32                                  # Power of two, [16..256].
ACCURACY_LEVEL                 = 4                                   # MatMulNBits accuracy level 0-4 (DEFAULT algo only).
QUANT_SYMMETRIC                = False                               # False = asymmetric (accuracy), True = symmetric (speed).
QUANT_FORMAT                   = "QOperator"                         # "QOperator" (MatMulNBits op) | "QDQ" (DEFAULT algo + 4-bit only).

# --- Dynamic INT8 quantization defaults (DYNAMIC -> quantize_dynamic) ----------
DYNAMIC_WEIGHT_TYPE            = "QInt8"                             # "QUInt8" | "QInt8".
DYNAMIC_PER_CHANNEL            = True                                # Per-channel weights (accuracy, slower).
DYNAMIC_REDUCE_RANGE           = False                               # 7-bit weights; can help on non-VNNI CPUs.

# --- Node selection (global defaults; override per module via Plan) ------------
NODES_TO_EXCLUDE               = None                                # Node names to keep unquantized, or None.
NODES_TO_INCLUDE               = None                                # Node names to quantize exclusively, or None.

# --- Storage / opset ----------------------------------------------------------
FORCE_EXTERNAL_DATA            = False                               # Two-part storage (*.onnx.data); auto-forced when >2GB.
UPGRADE_OPSET                  = 0                                   # Target ONNX opset (0 = keep current).

# --- Graph optimizer (onnxruntime.transformers) -------------------------------
OPTIMIZER_LEVEL                = 2                                   # ORT graph optimization level: 0 | 1 | 2 | 99.
OPTIMIZER_MODEL_TYPE           = "bert"                              # Fusion template; "bert" is a safe generic choice.
OPTIMIZER_ONLY_ONNXRUNTIME     = False                               # True = only ORT's built-in optimizer (skip Python fusions).
OPTIMIZER_FUSION_OPTIONS       = None                                # Optional dict of FusionOptions overrides, e.g. {"enable_gelu": False}.
SHAPE_INFER                    = True                                # Run shape inference before the optimizer (needed for some fusions).

# --- onnxslim -----------------------------------------------------------------
# EliminationReshape is unsafe when consecutive reshapes use zero-copy dimensions
# across a rank change: it applies the second shape directly to the first input.
SLIM_SKIP_FUSION_PATTERNS      = ["EliminationReshape"]              # Additional fusion patterns to skip, or None.
SLIM_SKIP_OPTIMIZATIONS        = None                                # Optimizations to skip, or None.
SLIM_SIZE_THRESHOLD            = None                                # Max constant size (bytes) to fold; None = fold all.

# --- float16 conversion (F16 method / Plan.fp16) ------------------------------
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
    # "Pow",
    # "ReduceMean",
    # "ReduceSum",
    # "Sqrt",
    # "Softmax",
]

# --- Q8-KV surgery: auto-rewrite integer KV attention/write/rope-shift chains ---
KV_ATTENTION_SURGERY           = "auto"                              # "auto" (enable when quantized KV detected) | True | False.


@dataclass
class Plan:
    """Per-module recipe. None inherits the USER CONFIG default."""
    method:              str                    = "Q4"     # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
    # weight-only (Q2/Q4/Q8)
    algo:                str  | None            = None     # DEFAULT | RTN | HQQ | k_quant (k_quant/RTN: Q4 only)
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


# Only standalone graphs and the primary merged donor are listed in MODEL_PLANS; the other
# merged strategy graphs inherit the primary's quantized Main via build_quantized_merged_bundle.
_PRIMARY_MERGED_KEY = "text_prefill_greedy"
_PRIMARY_MERGED_MODEL = Path(Shared_Merged.default_model_file_names()[_PRIMARY_MERGED_KEY]).stem
_MERGED_MODEL_NAMES = tuple(Path(name).stem for name, _, _ in Shared_Merged.MERGED_BUILD_PLAN)


# Per-module plan. Comment out a line to skip that module; edit "method" freely.
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata":             Plan(method="F32", optimize=False),
    # # Primary merged donor: quantized once, then transplanted into every strategy graph.
    _PRIMARY_MERGED_MODEL:      Plan(method="Q4", external=True, optimize=True),
    # Standalone vision front-end (kept out of the fused language path).
    "LLM_Vision":               Plan(method="Q4", external=True, optimize=True),
    "LLM_Image_Preprocess":     Plan(method="F32", optimize=True),
    "LLM_Video_Preprocess":     Plan(method="F32", optimize=True),
    # KV-cache maintenance / rope-shift (no learnable weights).
    "LLM_KV_Slice":             Plan(method="F32", optimize=True),
    "LLM_KV_Split2":            Plan(method="F32", optimize=True),
    "LLM_KV_Concat":            Plan(method="F32", optimize=True),
    "LLM_RopeShift":            Plan(method="F32", optimize=True),
}


# ============================== RESOLUTION ==============================

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
        if "Gather" in rp.op_types and rp.algo != "DEFAULT":
            raise ValueError(f"[{name}] Gather quantization requires algo='DEFAULT' (got {rp.algo!r}).")
        if rp.quant_format == "QDQ" and (rp.algo != "DEFAULT" or bits != 4):
            raise ValueError(
                f"[{name}] QDQ format supports only algo='DEFAULT' with 4-bit (got {rp.algo!r}, {bits}-bit)."
            )

    if rp.method == "DYNAMIC" and rp.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
        raise ValueError(f"[{name}] unknown dynamic_weight_type; choose 'QUInt8' or 'QInt8'.")


def _uses_fp16(plan: Plan) -> bool:
    return plan.fp16 or plan.method.upper() == "F16"


# Mixed precision needs fp32 graph I/O at shared boundaries.
_PRECISION_MODEL_PLANS = [plan for name, plan in MODEL_PLANS.items() if name != "LLM_Metadata"]
MIXED_PRECISION = (
    any(_uses_fp16(p) for p in _PRECISION_MODEL_PLANS)
    and not all(_uses_fp16(p) for p in _PRECISION_MODEL_PLANS)
)


# ============================== FILE / STORAGE HELPERS ==============================


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


def write_onnx_metadata(model_path: str, metadata: dict) -> None:
    # Rewrite only the graph proto so external weight sidecars stay untouched.
    model = onnx.load(model_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
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


def fuse_consecutive_reshapes(model_path: str) -> int:
    """Fuse constant-shape Reshape pairs only when their composed semantics are provable."""
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph
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


# ============================== OPTIMIZE / QUANTIZE ==============================


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
    except Exception as e:
        print(f"  Opset upgrade failed: {e}. Keeping current version.")
        resave(model_path, model_path, external)


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


def quantize_dynamic_int8(src_path: str, dst_path: str, rp: ResolvedPlan, external: bool,
                          do_surgery: bool = False) -> None:
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


# ============================== GRAPH SURGERY UTILITIES ==============================


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


def _find_embed_gather(graph):
    inits = _init_map(graph)
    candidates = []
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
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


def _restore_embed_shell_boundary(model: onnx.ModelProto) -> bool:
    """Restore the canonical Embed-to-Main edge after whole-graph optimization."""
    graph = model.graph
    gather, _, _, _ = _find_embed_gather(graph)
    old_name = gather.output[0]
    new_name = "embed_text_hidden_states"
    if old_name == new_name:
        return False

    defined = {value.name for value in graph.input}
    defined.update(init.name for init in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output != old_name)
    if new_name in defined:
        raise RuntimeError(
            f"Cannot restore Embed shell boundary: value {new_name!r} is already defined."
        )

    gather.output[0] = new_name
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
    for collection in (graph.output, graph.value_info):
        for value in collection:
            if value.name == old_name:
                value.name = new_name
    return True


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
    inits = _init_map(graph)
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = _node_attrs(node)
        if int(attrs.get("K", -1)) == hidden and int(attrs.get("N", -1)) == vocab:
            return node.op_type, node

    expected = {(hidden, vocab), (vocab, hidden)}
    for op_type in ("MatMul", "Gemm", "MatMulInteger"):
        for node in graph.node:
            if node.op_type != op_type or len(node.input) < 2:
                continue
            init = inits.get(node.input[1])
            if init is not None and _tensor_dims(init) in expected:
                return node.op_type, node
    raise RuntimeError(f"lm_head op with vocab={vocab}, hidden={hidden} was not found")


def _make_scalar_initializer(graph, name: str, array: np.ndarray) -> str:
    if name not in _init_map(graph):
        graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def _make_axes_initializer(graph, name: str, axes: list[int]) -> str:
    if name not in _init_map(graph):
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
        replacement = [helper.make_node("Gather", [shared_weight, ids], [out], axis=0, name=make("gather"))]
    elif dims == (hidden, vocab):
        gathered = make("gathered_hbs")
        replacement = [
            helper.make_node("Gather", [shared_weight, ids], [gathered], axis=1, name=make("gather")),
            helper.make_node("Transpose", [gathered], [out], perm=[1, 2, 0], name=make("transpose")),
        ]
    else:
        raise RuntimeError(f"unsupported lm_head weight shape {dims}; expected {(hidden, vocab)} or {(vocab, hidden)}")
    _replace_graph_node(graph, gather, replacement)
    if embed_init != shared_weight:
        _drop_initializers(graph, {embed_init})
    return {"lmhead_op": lmhead.op_type, "dropped": embed_init, "shared_weight": shared_weight}


def _share_q4_embed_lmhead(graph, gather, embed_init: str, lmhead, vocab: int, hidden: int, fallback_block_size: int) -> dict:
    attrs = _node_attrs(lmhead)
    block_size = int(attrs.get("block_size", fallback_block_size))
    if block_size <= 0 or hidden % block_size != 0:
        raise RuntimeError(f"lm_head MatMulNBits block_size={block_size} is incompatible with hidden={hidden}")
    if len(lmhead.input) < 4:
        raise RuntimeError("share_embed_lmhead currently requires asymmetric MatMulNBits with a zero-point input")

    bq, bs, bz = lmhead.input[1], lmhead.input[2], lmhead.input[3]
    kb = hidden // block_size
    make = _make_name_factory(graph, "share_embed_lmhead_q4_")
    ids, out = gather.input[1], gather.output[0]

    axm1 = _make_axes_initializer(graph, make("axis_m1"), [-1])
    rs_qint = _make_axes_initializer(graph, make("reshape_qint"), [0, 0, 0, -1])
    rs_flat = _make_axes_initializer(graph, make("reshape_flat"), [0, 0, -1])
    c16 = _make_scalar_initializer(graph, make("c16"), np.array(16, dtype=np.uint8))
    z_start = _make_axes_initializer(graph, make("z_start"), [0])
    z_end = _make_axes_initializer(graph, make("z_end"), [kb])
    z_axis = _make_axes_initializer(graph, make("z_axis"), [2])

    gq, gs, gz = make("gather_q"), make("gather_s"), make("gather_z")
    qlo, qhi, qlo1, qhi1, qcat, qint = make("qlo"), make("qhi"), make("qlo1"), make("qhi1"), make("qcat"), make("qint")
    zlo, zhi, zlo1, zhi1, zcat, zflat, zint = make("zlo"), make("zhi"), make("zlo1"), make("zhi1"), make("zcat"), make("zflat"), make("zint")
    qf, zf, zf1, gs1, sub, deq = make("qf"), make("zf"), make("zf1"), make("gs1"), make("sub"), make("deq")

    replacement = [
        helper.make_node("Gather", [bq, ids], [gq], axis=0, name=make("gather_q_node")),
        helper.make_node("Gather", [bs, ids], [gs], axis=0, name=make("gather_s_node")),
        helper.make_node("Gather", [bz, ids], [gz], axis=0, name=make("gather_z_node")),
        helper.make_node("Mod", [gq, c16], [qlo], name=make("qlo_node")),
        helper.make_node("Div", [gq, c16], [qhi], name=make("qhi_node")),
        helper.make_node("Unsqueeze", [qlo, axm1], [qlo1], name=make("qlo_unsq")),
        helper.make_node("Unsqueeze", [qhi, axm1], [qhi1], name=make("qhi_unsq")),
        helper.make_node("Concat", [qlo1, qhi1], [qcat], axis=-1, name=make("qcat_node")),
        helper.make_node("Reshape", [qcat, rs_qint], [qint], name=make("qreshape_node")),
        helper.make_node("Mod", [gz, c16], [zlo], name=make("zlo_node")),
        helper.make_node("Div", [gz, c16], [zhi], name=make("zhi_node")),
        helper.make_node("Unsqueeze", [zlo, axm1], [zlo1], name=make("zlo_unsq")),
        helper.make_node("Unsqueeze", [zhi, axm1], [zhi1], name=make("zhi_unsq")),
        helper.make_node("Concat", [zlo1, zhi1], [zcat], axis=-1, name=make("zcat_node")),
        helper.make_node("Reshape", [zcat, rs_flat], [zflat], name=make("zreshape_node")),
        helper.make_node("Slice", [zflat, z_start, z_end, z_axis], [zint], name=make("zslice_node")),
        helper.make_node("Cast", [qint], [qf], to=TensorProto.FLOAT, name=make("q_cast")),
        helper.make_node("Cast", [zint], [zf], to=TensorProto.FLOAT, name=make("z_cast")),
        helper.make_node("Unsqueeze", [zf, axm1], [zf1], name=make("z_unsq")),
        helper.make_node("Unsqueeze", [gs, axm1], [gs1], name=make("s_unsq")),
        helper.make_node("Sub", [qf, zf1], [sub], name=make("sub_node")),
        helper.make_node("Mul", [sub, gs1], [deq], name=make("mul_node")),
        helper.make_node("Reshape", [deq, rs_flat], [out], name=make("output_reshape")),
    ]

    _replace_graph_node(graph, gather, replacement)
    _drop_initializers(graph, {embed_init})
    return {"lmhead_op": lmhead.op_type, "dropped": embed_init, "shared_weight": bq}


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


def _append_vector_or_scalar_dequant_input(graph, nodes: list, name: str, ids: str, vocab: int, make, suffix: str) -> str:
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


def unify_embed_lmhead_graph(model: onnx.ModelProto, method: str, block_size: int = 32, quiet: bool = False) -> dict | None:
    """Share Main's tied embedding with the lm_head weight in place (no disk I/O).

    Used by both unify_embed_lmhead (file-based) and the merged-bundle builder, which
    unifies every transplanted strategy graph on the fly before redirecting its weights
    into the shared-initializer blob.
    """
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
    elif method == "Q4":
        if lmhead.op_type != "MatMulNBits":
            raise RuntimeError(f"Q4 share_embed_lmhead expected MatMulNBits lm_head, got {lmhead.op_type}")
        info = _share_q4_embed_lmhead(graph, gather, embed_init, lmhead, vocab, hidden, block_size)
    elif method == "DYNAMIC":
        if lmhead.op_type != "MatMulInteger":
            raise RuntimeError(f"DYNAMIC share_embed_lmhead expected MatMulInteger lm_head, got {lmhead.op_type}")
        info = _share_dynamic_embed_lmhead(graph, gather, embed_init, lmhead, vocab, hidden)
    else:
        raise ValueError(f"unknown share_embed_lmhead method {method!r}")

    _dead_code_elimination(graph)
    _deduplicate_node_names(graph)
    return info


def unify_embed_lmhead(model_path: str, method: str, block_size: int = 32, external: bool | None = None, quiet: bool = False) -> dict | None:
    if external is None:
        external = os.path.exists(model_path + ".data")
    model = onnx.load(model_path)
    info = unify_embed_lmhead_graph(model, method, block_size=block_size, quiet=quiet)
    if info is not None:
        _save_model(model, model_path, external)
    del model
    gc.collect()
    return info


def _unify_method_kind(rp: ResolvedPlan) -> str:
    if rp.method in _WEIGHT_ONLY_BITS:
        return "Q4"
    if rp.method == "DYNAMIC":
        return "DYNAMIC"
    return "F16" if (rp.fp16 or rp.method == "F16") else "F32"


# ============================== Q8-KV SURGERY ==============================
# Keeps quantized KV caches in integer form while using ORT fused quant ops.


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


def rewire_attention_to_matmulintegertofloat(model) -> tuple[int, int]:
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
            qu8, qs, qzp = f"{pfx}_qk_qu8", f"{pfx}_qk_qs", f"{pfx}_qk_qzp"
            new_nodes.extend(casts)
            new_nodes.extend([
                helper.make_node("DynamicQuantizeLinear", [a], [qu8, qs, qzp], name=f"{pfx}_qk_dql"),
                helper.make_node("MatMulIntegerToFloat", [qu8, k_in, qs, one_f32(f"{pfx}_qk_one_f32"), qzp, qk_bzp], [out],
                                 name=f"{pfx}_qk_mmitf", domain="com.microsoft"),
            ])
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
            pu8, psc, pzp = f"{pfx}_pv_pu8", f"{pfx}_pv_psc", f"{pfx}_pv_pzp"
            main = out if v_bias is None else f"{pfx}_pv_main"
            new_nodes.extend(casts)
            new_nodes.extend([
                helper.make_node("Transpose", [v_scale_f], [vst], perm=[0, 1, 2, 4, 3], name=f"{pfx}_pv_tr"),
                helper.make_node("Mul", [a, vst], [ps], name=f"{pfx}_pv_mul"),
                helper.make_node("DynamicQuantizeLinear", [ps], [pu8, psc, pzp], name=f"{pfx}_pv_dql"),
                helper.make_node("MatMulIntegerToFloat", [pu8, v_in, psc, one_f32(f"{pfx}_pv_one_f32"), pzp, target_bzp], [main],
                                 name=f"{pfx}_pv_mmitf", domain="com.microsoft"),
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
        n_qk, n_pv = rewire_attention_to_matmulintegertofloat(model)
        n_q = rewire_kv_quantize_to_quantizelinear(model)
        message = f"    surgery: {n_qk} Q@K + {n_pv} attn@V -> MatMulIntegerToFloat"
        if n_q:
            message += f"; {n_q} KV write tails -> QuantizeLinear (blocked int8)"
        print(message)
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
            return True, f"applying ({reason}) -> MatMulIntegerToFloat, in-memory"
        rope_ok, rope_reason = inspect_rope_shift_surgery(meta.graph)
        if rope_ok:
            return True, f"applying ({rope_reason}) -> DequantizeLinear/QuantizeLinear, in-memory"
        for r in (reason, rope_reason):
            if "not an attention module" not in r and "not a rope-shift module" not in r:
                return False, r
        return False, reason
    finally:
        del meta


# ============================== PIPELINE ==============================


def process_model(name: str, rp: ResolvedPlan) -> None:
    src_path, dst_path = get_model_paths(name)
    if not os.path.exists(src_path):
        print(f"  Skipping — file not found: {src_path}")
        return

    # Restamp exporter metadata after stages that rebuild ModelProto.
    source_metadata = read_onnx_metadata(src_path)
    preserve_fp16_compute = source_metadata.get("compute_in_f32", "1").lower() in ("0", "false")

    _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = MIXED_PRECISION if F16_KEEP_IO_TYPES is None else F16_KEEP_IO_TYPES

    # Quantized (Q8 / rope-shift) KV surgery runs in-memory before optimize. Float16 KV needs no
    # surgery: the exporter's COMPUTE_IN_F32 switch already emits F16- or F32-compute attention.
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

    # activations_fp16 is the only metadata value changed by optimization.
    if source_metadata:
        if use_fp16:
            source_metadata["activations_fp16"] = "1"
        write_onnx_metadata(dst_path, source_metadata)
        fp16_note = " (activations_fp16=1)" if use_fp16 else ""
        print(f"  Metadata: restamped {len(source_metadata)} keys onto the optimized model{fp16_note}.")


def _metadata_model_file_names(source_folder: Path) -> dict[str, str]:
    metadata = read_onnx_metadata(str(source_folder / "LLM_Metadata.onnx"))
    return {
        key[len("model_file_name_"):]: value
        for key, value in metadata.items()
        if key.startswith("model_file_name_") and value
    }


def _print_process_header(name: str, rp: ResolvedPlan) -> None:
    print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}]\n{'=' * 60}")


def _cleanup_merged_outputs(out_folder: Path, model_file_names: dict[str, str]) -> None:
    for file_name, _, _ in Shared_Merged.make_merged_build_plan(model_file_names):
        _remove_external_files(str(out_folder / file_name))
    shared_name = model_file_names.get("shared_initializers", Shared_Merged.SHARED_MODEL_NAME)
    _remove_external_files(str(out_folder / shared_name))


def _available_merged_plan(source_folder: Path, model_file_names: dict[str, str]):
    # Availability is keyed on the merged file itself: the exporter emits merged graphs
    # directly and deletes the split constituents.
    return [
        (file_name, recipe, deps)
        for file_name, recipe, deps in Shared_Merged.make_merged_build_plan(model_file_names)
        if (source_folder / file_name).exists()
    ]


def build_quantized_merged_bundle(resolved: dict[str, ResolvedPlan]) -> None:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    out_folder = Path(QUANTED_FOLDER_PATH)
    model_file_names = _metadata_model_file_names(source_folder) or Shared_Merged.default_model_file_names()
    available = _available_merged_plan(source_folder, model_file_names)
    if not available:
        print("\nExported merged strategy graphs not found; skipping merged bundle optimization.")
        return

    _cleanup_merged_outputs(out_folder, model_file_names)

    # Quantize the canonical primary once, then transplant its Main into every other
    # strategy graph so the whole family keeps sharing one weight bundle.
    primary_file = model_file_names.get(_PRIMARY_MERGED_KEY, available[0][0])
    if not (source_folder / primary_file).exists():
        primary_file = available[0][0]
    primary_stem = Path(primary_file).stem
    primary_plan = resolved.get(primary_stem) or resolved.get(_PRIMARY_MERGED_MODEL)
    if primary_plan is None:
        raise RuntimeError(f"No plan is configured for the primary merged graph {primary_stem!r}.")

    # Whole-graph optimization stays off (Plan.optimize=False) so the shell/Main tensor
    # boundary survives verbatim for transplanting.
    _print_process_header(primary_stem, primary_plan)
    process_model(primary_stem, primary_plan)

    primary_path = out_folder / primary_file
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)

    method_kind = _unify_method_kind(primary_plan)
    shared_model_name = model_file_names.get("shared_initializers", Shared_Merged.SHARED_MODEL_NAME)
    shared_data_name = model_file_names.get("shared_initializers_data", shared_model_name + ".data")

    source_metadata = read_onnx_metadata(str(primary_path))
    if source_metadata and model_file_names:
        source_metadata.update({f"model_file_name_{key}": value for key, value in model_file_names.items()})

    def _persist(file_name: str, model: onnx.ModelProto) -> None:
        out_path = out_folder / file_name
        _validate_graph_references(model, file_name)
        Shared_Merged.save_model(model, out_path)
        if source_metadata:
            write_onnx_metadata(str(out_path), source_metadata)
        print(f"  {file_name} ({out_path.stat().st_size} bytes)")

    print(f"\n{'=' * 60}\nTransplanting quantized Main into multimodal merged graphs\n{'=' * 60}")

    # Un-unified donor for the transplant loop: transplant copies its Main node block into each
    # shell and unification runs per graph AFTERWARDS (so the donor must stay un-unified, or its
    # reconstruction nodes would leak into every graph). It is loaded structure-only and made
    # self-contained below, so its multi-GB Main/embedding weights never enter memory.
    # Captured before _persist overwrites primary_path with the unified primary graph.
    clean_primary = onnx.load(str(primary_path), load_external_data=False)

    # Primary's final form: unify the tied embedding with the quantized lm_head, then seed the
    # shared-initializer blob. The graph is unified structure-only first, so the large fp32
    # embedding that unification discards is never materialized; only the surviving Main tensors
    # that actually feed the shared blob are loaded from disk.
    if _restore_embed_shell_boundary(clean_primary):
        print("  Restored optimized donor Embed/Main boundary before transplantation.")
    if _restore_prefill_mask_shell_boundary(clean_primary):
        print("  Restored optimized donor prefill-mask/Main boundary before transplantation.")
    primary_model = onnx.load(str(primary_path), load_external_data=False)
    info = unify_embed_lmhead_graph(primary_model, method_kind, block_size=primary_plan.block_size, quiet=True)
    if info is not None:
        print(
            f"  Shared embed/lm_head: dropped {info['dropped']!r}; "
            f"embedding now reuses {info['shared_weight']!r} ({info['lmhead_op']})."
        )
    _drop_unused_initializers(primary_model.graph)
    load_external_data_for_model(primary_model, str(primary_path.parent))
    external_by_name = Shared_Merged.write_shared_initializers(primary_model, out_folder / shared_model_name)

    # Make the donor self-contained BEFORE the primary's own .data sidecar is deleted by the
    # persist below. Inlining its remaining initializers mirrors a full onnx.load, so the
    # transplanted graphs stay byte-for-byte identical to the un-optimized pipeline; the large
    # shared weights are then repointed at the blob (their inlined bytes are released), leaving
    # only the few small non-shared Main initializers (e.g. qk_norm_scale) inline in each graph.
    if info is not None and info["dropped"] != info["shared_weight"]:
        _drop_initializers(clean_primary.graph, {info["dropped"]})
    load_external_data_for_model(clean_primary, str(primary_path.parent))
    Shared_Merged.redirect_shared_initializers_to_external(clean_primary, external_by_name)

    # Finalize the primary output (this deletes its .data sidecar; the donor no longer needs it).
    Shared_Merged.redirect_shared_initializers_to_external(primary_model, external_by_name)
    _persist(primary_file, primary_model)
    del primary_model
    gc.collect()

    for file_name, _, _ in available:
        if file_name == primary_file:
            continue
        # Structure-only load: the target's Main/embedding weights live in the multi-GB exporter
        # blob and are never needed here — transplant swaps in the donor's quantized Main, unify
        # drops the tied embedding, and redirect repoints every shared initializer at the blob.
        target = onnx.load(str(source_folder / file_name), load_external_data=False)
        model = Shared_Merged.transplant_quantized_main(target, clean_primary)
        del target
        unify_embed_lmhead_graph(model, method_kind, block_size=primary_plan.block_size, quiet=True)
        _drop_unused_initializers(model.graph)
        Shared_Merged.redirect_shared_initializers_to_external(model, external_by_name)
        _persist(file_name, model)
        del model
        gc.collect()

    shared_data = out_folder / shared_data_name
    if shared_data.exists():
        print(f"  {shared_data_name} ({shared_data.stat().st_size} bytes)")

    for removed in Shared_Merged.delete_merged_constituents(
        out_folder,
        protected_names=(shared_model_name, shared_data_name),
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
