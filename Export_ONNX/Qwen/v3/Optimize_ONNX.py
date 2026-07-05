"""Optimize exported split ONNX modules.

Edit USER CONFIG for defaults and MODEL_PLANS for per-module overrides. Methods:
Q2/Q4/Q8 = MatMulNBits, DYNAMIC = INT8 dynamic, F16 = fp16, F32 = optimize only.
"""

import os
import gc
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import onnx
import onnx.version_converter
from onnx import TensorProto, helper, numpy_helper
from onnxslim import slim
from transformers import AutoConfig
from onnxruntime.quantization import (
    QuantType,
    matmul_nbits_quantizer,
    quant_utils,
    quantize_dynamic,
)


# ============================== USER CONFIG ==============================

# --- Folders / source model (edit these three to target a different model) ----
# The LLM_* module names in MODEL_PLANS are shared across the exported model family.
_SCRIPT_DIR                    = Path(__file__).resolve().parent
ORIGINAL_FOLDER_PATH           = str(_SCRIPT_DIR / "Qwen_ONNX")         # Folder holding the exported *.onnx modules.
QUANTED_FOLDER_PATH            = str(_SCRIPT_DIR / "Qwen_Optimized")    # Destination folder for the results.
DOWNLOAD_PATH                  = r"/home/DakeQQ/Downloads/Qwen3-1.7B"   # Model dir (attention fusion); "NONE" to skip.

# --- Weight-only quantization defaults (Q2 / Q4 / Q8 -> MatMulNBits) -----------
WEIGHT_ONLY_ALGORITHM          = "k_quant"                           # "k_quant" | "DEFAULT" | "RTN" | "HQQ".
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
SLIM_SKIP_FUSION_PATTERNS      = None                                # Fusion patterns to skip, or None.
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

# --- F16-KV surgery: auto-rewrite float16 KV attention + matching prefill mask --
F16_KV_SURGERY                 = "auto"                              # "auto" (enable when float16 KV detected) | True | False.


@dataclass
class Plan:
    """Per-module recipe. None inherits the USER CONFIG default."""
    method:              str                    = "Q4"     # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
    # weight-only (Q2/Q4/Q8)
    algo:                str  | None            = None     # DEFAULT | RTN | HQQ | k_quant
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
    f16_surgery:         bool | str | None      = None     # float16 KV
    # storage
    external:            bool | None            = None     # None inherits; auto-forced when >2GB


# Per-module plan. Comment out a line to skip that module; edit "method" freely.
MODEL_PLANS: dict[str, Plan] = {
    "LLM_Metadata":      Plan(method="F32", optimize=False),
    "LLM_Embed":         Plan(method="Q4", external=True, algo="DEFAULT", block_size=16, op_types=("Gather",), axes=(1,)),
    "LLM_Main":          Plan(method="Q4", external=True),
    "LLM_Greedy":        Plan(method="F32"),
    "LLM_FirstBeam":     Plan(method="F32"),
    "LLM_SecondBeam":    Plan(method="F32"),
    "LLM_Penalty":       Plan(method="F32"),
    "LLM_Argmax":        Plan(method="F32"),
    "LLM_KV_Slice":      Plan(method="F32"),
    "LLM_KV_Split2":     Plan(method="F32"),
    "LLM_KV_Concat":     Plan(method="F32"),
    "LLM_RotaryPrefill": Plan(method="F32"),
    "LLM_RotaryDecode":  Plan(method="F32"),
    "LLM_RopeShift":     Plan(method="F32"),
}


# ============================== RESOLUTION ==============================

_WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}     # method -> weight-only bit width
_QUANT_FORMATS = {
    "QOPERATOR": quant_utils.QuantFormat.QOperator,
    "QDQ": quant_utils.QuantFormat.QDQ,
}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_VALID_ALGOS = {"DEFAULT", "RTN", "HQQ", "k_quant"}


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
    f16_surgery:         bool | str
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
        f16_surgery=_pick(plan.f16_surgery, F16_KV_SURGERY),
        external=_pick(plan.external, FORCE_EXTERNAL_DATA),
    )


def validate_plan(name: str, rp: ResolvedPlan) -> None:
    valid_methods = set(_WEIGHT_ONLY_BITS) | {"DYNAMIC", "F16", "F32"}
    if rp.method not in valid_methods:
        raise ValueError(f"[{name}] unknown method {rp.method!r}; choose one of {sorted(valid_methods)}.")

    if rp.kv_surgery not in ("auto", True, False):
        raise ValueError(f"[{name}] kv_surgery must be 'auto', True, or False (got {rp.kv_surgery!r}).")

    if rp.f16_surgery not in ("auto", True, False):
        raise ValueError(f"[{name}] f16_surgery must be 'auto', True, or False (got {rp.f16_surgery!r}).")

    if rp.method in _WEIGHT_ONLY_BITS:
        bits = _WEIGHT_ONLY_BITS[rp.method]
        if rp.algo not in _VALID_ALGOS:
            raise ValueError(f"[{name}] unknown algo {rp.algo!r}; choose one of {sorted(_VALID_ALGOS)}.")
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
                        use_fp16: bool, external: bool, keep_io_types: bool) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    model = optimize_model(
        model_path,
        use_gpu=False,
        opt_level=OPTIMIZER_LEVEL,
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


# ============================== F16-KV SURGERY ==============================
# Turns KV_QUANT_DTYPE="F16" exports into fully-float16 attention.


def inspect_f16_attention_surgery(graph) -> tuple[bool, str]:
    inputs = {i.name: i for i in graph.input}
    keys = [n for n in inputs if n.startswith("in_key_")
            and not n.startswith("in_key_scale") and not n.startswith("in_key_bias")]
    if not keys:
        return False, "no KV cache inputs (not an attention module) — skipped"
    elem = inputs[keys[0]].type.tensor_type.elem_type
    if elem != TensorProto.FLOAT16:
        return False, f"KV cache is not float16 (elem_type={elem}); F16 surgery targets KV_QUANT_DTYPE=F16 — skipped"
    inits = {i.name for i in graph.initializer}
    if not any(n.op_type == "MatMul" and n.input[1] not in inits for n in graph.node):
        return False, "no activation@activation matmuls to rewrite — skipped"
    return True, "float16 KV attention (f32-compute baseline) -> fully-float16"


def inspect_f16_prefill_surgery(graph) -> tuple[bool, str]:
    if any(n.op_type == "MatMul" for n in graph.node):
        return False, "has MatMul (attention module, not the rotary/mask prefill) — skipped"
    out = next((o for o in graph.output if o.name == "attention_mask"), None)
    if out is None:
        return False, "no attention_mask output (not the rotary/mask prefill) — skipped"
    elem = out.type.tensor_type.elem_type
    if elem == TensorProto.FLOAT16:
        return False, "attention_mask output already float16 — skipped"
    if elem != TensorProto.FLOAT:
        return False, f"attention_mask output is not float32 (elem_type={elem}) — skipped"
    return True, "float32 attention_mask output -> float16"


def rewire_attention_to_f16(model) -> tuple[int, int]:
    graph = model.graph
    inits = {i.name for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}

    elem_of: dict[str, int] = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            elem_of[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        elem_of[init.name] = init.data_type
    kv_key_names = {o.name for o in graph.output if o.name.startswith("out_key_")}
    kv_val_names = {o.name for o in graph.output if o.name.startswith("out_value_")}
    kv_f16_names = kv_key_names | kv_val_names

    def is_kv_upcast(node) -> bool:
        if node.op_type != "Cast":
            return False
        to = next((a.i for a in node.attribute if a.name == "to"), None)
        if to != TensorProto.FLOAT:
            return False
        src = node.input[0]
        return src in kv_f16_names or elem_of.get(src) == TensorProto.FLOAT16

    new_nodes, n_qk, n_pv, pv_outs = [], 0, 0, []
    for idx, node in enumerate(graph.node):
        if node.op_type != "MatMul" or node.input[1] in inits:
            new_nodes.append(node)                          # keep weight (const-B) + non-matmul nodes
            continue
        a, b, out = node.input[0], node.input[1], node.output[0]
        a_prod, b_prod = producer.get(a), producer.get(b)
        a_kv = a_prod is not None and is_kv_upcast(a_prod)
        b_kv = b_prod is not None and is_kv_upcast(b_prod)
        if not (a_kv or b_kv):
            new_nodes.append(node)                          # not a float16-KV attention matmul
            continue
        pfx = (node.name.replace("/", "_") or "f16surg") + f"_{idx}"
        if b_kv:
            kv_cast, other, kv_is_b = b_prod, a, True
        else:
            kv_cast, other, kv_is_b = a_prod, b, False
        kv_f16 = kv_cast.input[0]                            # the float16 KV cache tensor
        if kv_f16 in kv_val_names:
            is_pv = True
        elif kv_f16 in kv_key_names:
            is_pv = False
        else:
            other_prod = producer.get(other)
            is_pv = other_prod is not None and other_prod.op_type == "Softmax"
        if is_pv:
            mm_inputs = [other, kv_f16] if kv_is_b else [kv_f16, other]
            new_nodes.append(helper.make_node("MatMul", mm_inputs, [out], name=f"{pfx}_pv_mm"))
            pv_outs.append(out)
            n_pv += 1
        else:
            q_f16 = f"{pfx}_qk_qf16"
            new_nodes.append(helper.make_node("Cast", [other], [q_f16], to=TensorProto.FLOAT16, name=f"{pfx}_qk_qcast"))
            mm_inputs = [q_f16, kv_f16] if kv_is_b else [kv_f16, q_f16]
            new_nodes.append(helper.make_node("MatMul", mm_inputs, [out], name=f"{pfx}_qk_mm"))
            n_qk += 1

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # Keep reshuffle ops in f16; upcast only before o_proj.
    if pv_outs:
        shape_ops = ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity")
        consumers: dict[str, list] = {}
        for n in graph.node:
            for x in n.input:
                consumers.setdefault(x, []).append(n)
        producer = {o: n for n in graph.node for o in n.output}
        cast_after = {}  # id(producer node) -> the tensor on its output to upcast to float32
        for out_name in pv_outs:
            cur, prod = out_name, producer.get(out_name)
            while True:
                cs = consumers.get(cur, [])
                if len(cs) == 1 and cs[0].op_type in shape_ops and len(cs[0].output) == 1:
                    prod, cur = cs[0], cs[0].output[0]
                else:
                    break
            if prod is not None:
                cast_after[id(prod)] = cur
        if cast_after:
            spliced = []
            for n in graph.node:
                spliced.append(n)
                tgt = cast_after.get(id(n))
                if tgt is not None:
                    tgt_f16 = f"{tgt}_f16"
                    for i, o in enumerate(n.output):
                        if o == tgt:
                            n.output[i] = tgt_f16
                    spliced.append(helper.make_node("Cast", [tgt_f16], [tgt], to=TensorProto.FLOAT,
                                                    name=f"{tgt_f16}_to_f32"))
            graph.ClearField("node")
            graph.node.extend(spliced)

    for vi in graph.input:
        if vi.name == "attention_mask":
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Drop stale inferred types after changing attention dtype.
    graph.ClearField("value_info")
    _dead_code_elimination(graph)
    return n_qk, n_pv


def retype_prefill_mask_to_f16(model) -> int:
    graph = model.graph
    out_vi = next((o for o in graph.output if o.name == "attention_mask"), None)
    if out_vi is None:
        return 0
    prod = next((n for n in graph.node if "attention_mask" in n.output), None)
    if prod is not None and prod.op_type == "Cast":
        for attr in prod.attribute:
            if attr.name == "to":
                attr.i = TensorProto.FLOAT16
        out_vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        return 1
    tmp = "attention_mask_pre_f16"
    if prod is not None:
        for i, o in enumerate(prod.output):
            if o == "attention_mask":
                prod.output[i] = tmp
    graph.node.append(helper.make_node("Cast", [tmp], ["attention_mask"], to=TensorProto.FLOAT16,
                                       name="attention_mask_to_f16"))
    out_vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    return 1


def apply_f16_surgery(model) -> None:
    applicable, _ = inspect_f16_attention_surgery(model.graph)
    if applicable:
        n_qk, n_pv = rewire_attention_to_f16(model)
        print(f"    F16 surgery: {n_qk} Q@K + {n_pv} attn@V -> float16 attention; attention_mask input -> float16")
        return
    applicable, _ = inspect_f16_prefill_surgery(model.graph)
    if applicable:
        n = retype_prefill_mask_to_f16(model)
        print(f"    F16 surgery: {n} attention_mask output -> float16")


def _companion_main_uses_f16_kv() -> bool:
    main_src, _ = get_model_paths("LLM_Main")
    if not os.path.exists(main_src):
        return False
    meta = onnx.load(main_src, load_external_data=False)
    try:
        applicable, _ = inspect_f16_attention_surgery(meta.graph)
        return applicable
    finally:
        del meta


def plan_f16_surgery(src_path: str) -> tuple[bool, str]:
    meta = onnx.load(src_path, load_external_data=False)
    try:
        applicable, reason = inspect_f16_attention_surgery(meta.graph)
        if applicable:
            return True, f"applying ({reason}), post-optimize"
        pre_ok, pre_reason = inspect_f16_prefill_surgery(meta.graph)
        if pre_ok:
            # Only retype prefill mask when Main consumes f16 masks.
            if not _companion_main_uses_f16_kv():
                return False, "companion LLM_Main is not float16-KV; prefill mask stays float32 — skipped"
            return True, f"applying ({pre_reason}), post-optimize"
        for r in (reason, pre_reason):
            if "not an attention module" not in r and "not the rotary/mask prefill" not in r:
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

    _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = MIXED_PRECISION if F16_KEEP_IO_TYPES is None else F16_KEEP_IO_TYPES

    # Detect surgery first: Q8 runs before optimize; F16 runs after.
    do_surgery = False
    if rp.kv_surgery is not False:
        do_surgery, message = plan_kv_surgery(src_path)
        print(f"  KV/rope-shift surgery: {message}")
    do_f16_surgery = False
    if not do_surgery and rp.f16_surgery is not False:
        do_f16_surgery, f16_message = plan_f16_surgery(src_path)
        print(f"  F16 surgery: {f16_message}")
    if (do_surgery or do_f16_surgery) and use_fp16:
        print(
            "  Surgery: disabled for float16 conversion; ORT's fp16 converter can otherwise leave the "
            "quantized/float16 island with invalid mixed f32/f16 types."
        )
        do_surgery = do_f16_surgery = False

    if rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(src_path, dst_path, rp, _WEIGHT_ONLY_BITS[rp.method], external, do_surgery)
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external, do_surgery)
    else:  # F16 / F32 — no integer quant; F16 is applied during the optimize stage.
        resave(src_path, dst_path, external, do_surgery)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        run_onnxslim(dst_path, external, no_shape_infer=True)
        heads, hidden = fetch_transformer_config(DOWNLOAD_PATH) if "Main" in name else (0, 0)
        optimize_onnx_model(dst_path, heads, hidden, use_fp16, external, keep_io_types)
        run_onnxslim(dst_path, external, no_shape_infer=not SHAPE_INFER)

    # F16-KV surgery must run after the CPU optimizer inserts its casts.
    if do_f16_surgery:
        print("  Applying F16-KV surgery (post-optimize)...")
        model = onnx.load(dst_path)
        apply_f16_surgery(model)
        _save_model(model, dst_path, external)
        del model
        gc.collect()

    if UPGRADE_OPSET > 0:
        upgrade_opset_version(dst_path, UPGRADE_OPSET, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")

    # activations_fp16 is the only metadata value changed by optimization.
    if source_metadata:
        if use_fp16 or do_f16_surgery:
            source_metadata["activations_fp16"] = "1"
        write_onnx_metadata(dst_path, source_metadata)
        fp16_note = " (activations_fp16=1)" if (use_fp16 or do_f16_surgery) else ""
        print(f"  Metadata: restamped {len(source_metadata)} keys onto the optimized model{fp16_note}.")


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
        print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}]\n{'=' * 60}")
        process_model(name, rp)
    print("\n--- All models processed successfully! ---")


if __name__ == "__main__":
    main()
