import os
import gc
import sys
import site
import glob
import importlib
from pathlib import Path

import onnx
import onnx.version_converter
import torch
from onnxslim import slim
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from onnxruntime.quantization import (
    QuantType,
    matmul_nbits_quantizer,
    quant_utils,
    quantize_dynamic,
)


# --- Configuration ---

# File Paths
ORIGINAL_FOLDER_PATH = r"/home/DakeQQ/Downloads/Qwen_ONNX"
QUANTED_FOLDER_PATH = r"/home/DakeQQ/Downloads/Qwen_Optimized"
DOWNLOAD_PATH = r"/home/iamj/Downloads/Qwen3-VL-2B-Instruct"

# Model List
MODEL_NAMES = [
    "LLM_Embed",
    "LLM_Main",
    "Greedy_Search",
    "First_Beam_Search",
    "Second_Beam_Search",
    "Apply_Penalty",
    "Argmax",
    "KV_Slice",
    "Rotary_Mask_Text_Prefill",
    "Rotary_Mask_Text_Decode",

    # -------------
    # Vision Models
    # -------------
    "LLM_Vision",
    "LLM_Concat",
    "Rotary_Mask_Vision_Prefill",
    "Rotary_Mask_Vision_Decode"
]

# Quantization Settings
ALGORITHM = "k_quant"       # Strategies: "DEFAULT", "RTN", "HQQ", "k_quant"
BITS = 4                    # Target bit precision (e.g., 4 or 8)
BLOCK_SIZE = 32             # [16, 32, 64, 128, 256]; Smaller => more accuracy, more time/size
ACCURACY_LEVEL = 4          # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
QUANT_SYMMETRIC = False     # False = Asymmetric (Accuracy, Slower), True = Symmetric (Faster)
NODES_TO_EXCLUDE = None     # List of specific ONNX node names to exclude from quantization
USE_Q8_VISION = True        # Enable INT8 dynamic quantization specifically for Vision models
USE_F16 = False             # Convert Q4F32/Q8F32 to Q4F16/Q8F16. BLOCK_SIZE <= 32 recommended
TWO_PARTS_SAVE = False      # Force saving models with external data (.data) regardless of size
UPGRADE_OPSET = 0           # Target ONNX Opset version (0 = keep current version)


# Operator block list for float16 conversion (ops that must stay in higher precision)
_F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
    # "Pow",
    # "ReduceMean",
    # "ReduceSum",
    # "Softmax",
]

# Key model names that require fetching transformer config (num_heads, hidden_size)
_CONFIG_REQUIRED_KEYS = ("Main", "Vision", "Decoder", "Encoder")

# Key model name that requires large model (>2GB) file size check
_SIZE_CHECK_KEY = "Main"


# --- Helper Functions ---


def _is_vision_model(name: str) -> bool:
    return "LLM_Vision" in name


def _should_disable_split(name: str) -> bool:
    return "Concat" in name or "Rotary" in name


def _is_embed_model(name: str) -> bool:
    return "Embed" in name


def _needs_transformer_config(name: str) -> bool:
    """Check if the model name contains any of the keys that require transformer config."""
    return any(key in name for key in _CONFIG_REQUIRED_KEYS)


def _needs_size_check(name: str) -> bool:
    """Check if the model name contains the key that requires file size checking."""
    return _SIZE_CHECK_KEY in name


def get_model_paths(model_name: str) -> tuple[str, str]:
    return (
        os.path.join(ORIGINAL_FOLDER_PATH, f"{model_name}.onnx"),
        os.path.join(QUANTED_FOLDER_PATH, f"{model_name}.onnx"),
    )


def _patch_optimizer_file(disable_transpose: bool) -> None:
    """Toggle the TransposeOptimizer patch in onnxruntime's optimizer.py."""
    pkg_path = site.getsitepackages()[-1]
    optimizer_path = os.path.join(pkg_path, "onnxruntime/transformers/optimizer.py")

    if disable_transpose:
        old = "disabled_optimizers=disabled_optimizers"
        new = 'disabled_optimizers=["TransposeOptimizer", "TransposeOptimizer_CPUExecutionProvider"]'
    else:
        old = 'disabled_optimizers=["TransposeOptimizer", "TransposeOptimizer_CPUExecutionProvider"]'
        new = "disabled_optimizers=disabled_optimizers"

    with open(optimizer_path, "r", encoding="utf-8") as f:
        content = f.read()

    if old in content:
        content = content.replace(old, new)
        with open(optimizer_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Patched {optimizer_path}")


def optimize_onnx_model(
    model_path: str,
    num_heads: int = 0,
    hidden_size: int = 0,
    use_f16: bool = False,
    save_external: bool = False,
) -> None:
    """Run onnxruntime.transformers.optimizer with optional float16 conversion."""
    is_vision = _is_vision_model(model_path)

    if is_vision:
        _patch_optimizer_file(disable_transpose=True)

    from onnxruntime.transformers.optimizer import optimize_model

    model = optimize_model(
        model_path,
        use_gpu=False,
        opt_level=2,
        num_heads=num_heads,
        hidden_size=hidden_size,
        verbose=False,
        model_type="bert",
        only_onnxruntime=False,
    )

    if use_f16:
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            min_positive_val=1e-7,
            op_block_list=_F16_OP_BLOCK_LIST,
        )

    model.save_model_to_file(model_path, use_external_data_format=save_external)
    del model
    gc.collect()

    if is_vision:
        _patch_optimizer_file(disable_transpose=False)


def run_onnxslim(
        model_path: str,
        save_external: bool,
        no_shape_infer: bool = False,
        input_model: str | None = None,
) -> None:
    """Wrapper for onnxslim optimization.

    Args:
        model_path: Output model path (also used as input if input_model is None).
        save_external: Whether to save as external data.
        no_shape_infer: Skip shape inference if True.
        input_model: Optional separate input model path. Defaults to model_path.
    """
    slim(
        model=input_model if input_model is not None else model_path,
        output_model=model_path,
        no_shape_infer=no_shape_infer,
        save_as_external_data=save_external,
        verbose=False,
    )


def upgrade_opset_version(model_path: str, version: int, save_external: bool) -> None:
    """Upgrade ONNX model opset version."""
    print(f"Upgrading Opset to {version}...")
    try:
        model = onnx.load(model_path)
        converted = onnx.version_converter.convert_version(model, version)
        onnx.save(converted, model_path, save_as_external_data=save_external)
        del model, converted
    except Exception as e:
        print(f"Error upgrading opset: {e}. Re-saving original.")
        save_onnx_model(model_path, save_external)
    gc.collect()


def save_onnx_model(model_path: str, save_external: bool) -> None:
    """Load and re-save an ONNX model (ensures consistent serialization)."""
    model = onnx.load(model_path)
    onnx.save(model, model_path, save_as_external_data=save_external)
    del model
    gc.collect()


def fetch_transformer_config(download_path: str, model_name: str) -> tuple[int, int]:
    """Load model configuration to retrieve Attention Heads and Hidden Size."""
    if not download_path or download_path.upper() == "NONE":
        return 0, 0

    path_lower = download_path.lower()

    try:
        cls = _resolve_model_class(path_lower)
        config_obj = cls.from_pretrained(
            download_path,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

        try:
            if _is_vision_model(model_name):
                vc = config_obj.config.vision_config
                return vc.num_heads, vc.hidden_size

            cfg = getattr(config_obj.config, "llm_config", config_obj.config)
            return cfg.num_attention_heads, cfg.hidden_size
        except AttributeError:
            return 0, 0
        finally:
            del config_obj
            gc.collect()

    except Exception as e:
        print(f"Warning: Could not load config: {e}. Using defaults.")
        return 0, 0


def _resolve_model_class(path_lower: str):
    """Determine the HuggingFace model class from the download path."""
    if "vl" in path_lower and "qwen" in path_lower:
        if "2.5" in path_lower:
            return Qwen2_5_VLForConditionalGeneration
        if "3" in path_lower:
            return Qwen3VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    if any(tag in path_lower for tag in ("gemma3", "gemma 3", "gemma-3")):
        return Gemma3ForCausalLM
    return AutoModelForCausalLM


def check_is_large_model(model_path: str, force_split: bool) -> bool:
    """Determine if the model is large (>2 GB) or forced to split."""
    if force_split:
        return True
    try:
        size_bytes = sys.getsizeof(onnx.load(model_path).SerializeToString())
        return size_bytes * 9.31322575e-10 > 2.0
    except Exception:
        return False


def process_vision_quantization(src_path: str, dst_path: str) -> None:
    """Apply dynamic INT8 quantization for Vision models."""
    print("Applying Dynamic Quantization for Vision model...")
    quantize_dynamic(
        model_input=quant_utils.load_model_with_shape_infer(Path(src_path)),
        model_output=dst_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": False,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": False,
            "MatMulConstBOnly": True,
        },
        nodes_to_exclude=None,
        use_external_data_format=TWO_PARTS_SAVE,
    )


def _build_quant_config(algo: str, op_types: list[str], axes: list[int], bits: int):
    """Build the appropriate quantization config based on algorithm name."""
    common = {
        "quant_format": quant_utils.QuantFormat.QOperator,
        "op_types_to_quantize": tuple(op_types),
    }
    quant_axes = tuple((op_types[i], axes[i]) for i in range(len(op_types)))

    if algo == "RTN":
        cfg = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common)
    elif algo == "HQQ":
        cfg = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits, block_size=BLOCK_SIZE, axis=axes[0],
            quant_axes=quant_axes, **common,
        )
    elif algo == "k_quant":
        cfg = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(**common)
    else:  # DEFAULT
        cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=BLOCK_SIZE,
            is_symmetric=QUANT_SYMMETRIC,
            accuracy_level=ACCURACY_LEVEL,
            quant_axes=quant_axes,
            **common,
        )

    cfg.bits = bits
    return cfg, quant_axes


def process_weight_quantization(
    src_path: str, dst_path: str, algo: str, op_types: list[str], axes: list[int], bits: int = BITS,
) -> None:
    """Apply weight-only quantization for standard models."""
    print("Loading model with shape inference...")
    model = quant_utils.load_model_with_shape_infer(Path(src_path))

    # Gather ops need at least 4-bit precision
    if "Gather" in op_types and bits == 2:
        bits = 4

    q_config, quant_axes = _build_quant_config(algo, op_types, axes, bits)

    print(f"Starting quantization ({algo}, {bits}-bit)...")
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=BLOCK_SIZE,
        is_symmetric=QUANT_SYMMETRIC,
        accuracy_level=ACCURACY_LEVEL,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=quant_axes,
        algo_config=q_config,
        nodes_to_exclude=NODES_TO_EXCLUDE,
    )
    quant.process()
    quant.model.save_model_to_file(dst_path, True)

    del model, quant
    gc.collect()


def cleanup_data_files() -> None:
    """Remove temporary *.onnx.data files from the output folder."""
    print("Cleaning up temporary *.onnx.data files...")
    for file_path in glob.glob(os.path.join(QUANTED_FOLDER_PATH, "*.onnx.data")):
        try:
            os.remove(file_path)
            print(f"  Deleted {file_path}")
        except Exception as e:
            print(f"  Error deleting {file_path}: {e}")


# --- Main Logic ---


def main() -> None:
    os.makedirs(QUANTED_FOLDER_PATH, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 50}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 50}")

        src_path, dst_path = get_model_paths(model_name)
        if not os.path.exists(src_path):
            print(f"  Skipping — file not found: {src_path}")
            continue

        use_split = False if _should_disable_split(model_name) else TWO_PARTS_SAVE
        is_vision = _is_vision_model(model_name)

        # --- Step 1: Quantization ---
        if is_vision:
            if USE_Q8_VISION:
                process_vision_quantization(src_path, dst_path)
        elif _is_embed_model(model_name):
            process_weight_quantization(src_path, dst_path, "DEFAULT", ["Gather"], [1])
        else:
            process_weight_quantization(src_path, dst_path, ALGORITHM, ["MatMul"], [0])

        # --- Step 2: Post-Quantization Optimization ---

        # Only 'Main' models need the >2GB file size check; others use use_split directly
        if _needs_size_check(model_name):
            is_large = check_is_large_model(dst_path, use_split)
        else:
            is_large = use_split

        # First onnxslim pass (vision without Q8 uses src_path as input)
        print("  [1/3] onnxslim (first pass)...")
        slim_input = src_path if (is_vision and not USE_Q8_VISION) else None
        run_onnxslim(dst_path, save_external=is_large, no_shape_infer=True, input_model=slim_input)

        # Transformers optimizer — only fetch config for Main/Vision/Decoder/Encoder
        print("  [2/3] onnxruntime transformers optimizer...")
        if _needs_transformer_config(model_name):
            num_heads, hidden_size = fetch_transformer_config(DOWNLOAD_PATH, model_name)
        else:
            num_heads, hidden_size = 0, 0
        optimize_onnx_model(dst_path, num_heads, hidden_size, USE_F16, save_external=is_large)

        # Second onnxslim pass
        print("  [3/3] onnxslim (second pass)...")
        run_onnxslim(dst_path, save_external=is_large)

        # Opset upgrade or consistent re-save
        if UPGRADE_OPSET > 0:
            upgrade_opset_version(dst_path, UPGRADE_OPSET, save_external=is_large)
        else:
            save_onnx_model(dst_path, is_large)

    cleanup_data_files()
    print("\n--- All models processed successfully! ---")


if __name__ == "__main__":
    main()



