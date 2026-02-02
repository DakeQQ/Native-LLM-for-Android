import os
import gc
import sys
import glob
import onnx
import torch
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from transformers import (
    AutoModelForCausalLM, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Gemma3ForCausalLM
)
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quantize_dynamic,
    QuantType,
    quant_utils
)

# --- Configuration ---
# File Paths
ORIGINAL_FOLDER_PATH = r"/home/DakeQQ/Downloads/Qwen_ONNX"
QUANTED_FOLDER_PATH = r"/home/DakeQQ/Downloads/Qwen_Optimized"
DOWNLOAD_PATH = r'/home/DakeQQ/Downloads/Qwen3-VL-2B-Instruct'

# Model List
MODEL_NAMES = [
    "LLM_Embed",
    "LLM_Main",
    "Greedy_Search",
    "First_Beam_Search",
    "Second_Beam_Search",
    "Reset_Penality",
    "Argmax",

    # Vision Models
    "LLM_Vision",
    "LLM_Concat",
    "LLM_Rotary_Vision",
    "LLM_Rotary_Text"
]

# Quantization Settings
ALGORITHM = "k_quant"           # Strategies: "DEFAULT", "RTN", "HQQ", "k_quant"
BITS = 4                        # Target bit precision (e.g., 4 or 8)
BLOCK_SIZE = 32                 # [16, 32, 64, 128, 256]; Smaller block_size => more accuracy, more time and size.
ACCURACY_LEVEL = 4              # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
QUANT_SYMMETRIC = False         # False = Asymmetric (includes ZeroPoint), True = Symmetric
NODES_TO_EXCLUDE = None         # List of specific ONNX node names to exclude from quantization
USE_Q8_VISION = False           # Enable INT8 dynamic quantization specifically for Vision models.
USE_F16 = False                 # Convert Q4F32/Q8F32 to Q4F16/Q8F16. The BLOCK_SIZE <= 32 is recommended.
TWO_PARTS_SAVE = False          # Force saving models with external data (.data) regardless of size
UPGRADE_OPSET = 0               # Target ONNX Opset version (0 = keep current version)


# --- Helper Functions ---
def get_model_paths(model_name):
    """Determines source and destination paths for a given model."""
    # Special handling for LLM_Main path location
    if ('LLM_Main' in model_name) and ('vl' in DOWNLOAD_PATH.lower() or 'LLM_Vision' in MODEL_NAMES):
        src_dir = ORIGINAL_FOLDER_PATH.replace('Qwen_ONNX', 'Qwen_ONNX_2')
    else:
        src_dir = ORIGINAL_FOLDER_PATH
    
    src_path = os.path.join(src_dir, f"{model_name}.onnx")
    dst_path = os.path.join(QUANTED_FOLDER_PATH, f"{model_name}.onnx")
    return src_path, dst_path


def optimize_onnx_model(model_path, num_heads=0, hidden_size=0, use_f16=False, save_external=False):
    """wrapper for onnxruntime.transformers.optimizer"""
    model = optimize_model(
        model_path,
        use_gpu=False,
        opt_level=2,
        num_heads=num_heads,
        hidden_size=hidden_size,
        verbose=False,
        model_type='bert',
        only_onnxruntime=False
    )

    if use_f16:
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            min_positive_val=1e-7,
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'MatMulIntegerToFloat']
        )
    
    model.save_model_to_file(model_path, use_external_data_format=save_external)
    del model
    gc.collect()


def run_onnxslim(model_path, save_external):
    """Wrapper for onnxslim optimization."""
    slim(
        model=model_path,
        output_model=model_path,
        no_shape_infer=True,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=save_external,
        verbose=False
    )


def upgrade_opset_version(model_path, version, save_external):
    """Upgrades ONNX model opset version."""
    print(f"Upgrading Opset to {version}...")
    try:
        model = onnx.load(model_path)
        converted = onnx.version_converter.convert_version(model, version)
        onnx.save(converted, model_path, save_as_external_data=save_external)
        del model, converted
    except Exception as e:
        print(f"Error upgrading opset: {e}. Saving originally.")
        model = onnx.load(model_path)
        onnx.save(model, model_path, save_as_external_data=save_external)
        del model
    gc.collect()


def fetch_transformer_config(download_path, model_name):
    """Loads model configuration to retrieve Attention Heads and Hidden Size."""
    if not download_path or download_path.upper() == "NONE":
        return 0, 0

    path_lower = download_path.lower()
    try:
        # Determine model class
        if 'vl' in path_lower and 'qwen' in path_lower:
            if "2.5" in path_lower:
                cls = Qwen2_5_VLForConditionalGeneration
            elif "3" in path_lower:
                cls = Qwen3VLForConditionalGeneration
            else:
                cls = Qwen2VLForConditionalGeneration
        elif any(x in path_lower for x in ["gemma3", "gemma 3", "gemma-3"]):
            cls = Gemma3ForCausalLM
        else:
            cls = AutoModelForCausalLM

        config_obj = cls.from_pretrained(
            download_path, 
            dtype=torch.float16, 
            device_map='cpu', 
            trust_remote_code=True, 
            low_cpu_mem_usage=True
        ).eval()

        # Extract parameters
        try:
            if "LLM_Vision" in model_name:
                return config_obj.config.vision_config.num_heads, config_obj.config.vision_config.hidden_size
            
            # Check standard config or nested llm_config
            cfg = getattr(config_obj.config, 'llm_config', config_obj.config)
            return cfg.num_attention_heads, cfg.hidden_size
        except AttributeError:
            return 0, 0
        finally:
            del config_obj
            gc.collect()

    except Exception as e:
        print(f"Warning: Could not load config: {e}. Using defaults.")
        return 0, 0


def check_is_large_model(model_path, use_two_parts):
    """Determines if the model is large (>2GB) or forced to split."""
    try:
        size_bytes = sys.getsizeof(onnx.load(model_path).SerializeToString())
        size_gb = size_bytes * 9.31322575e-10
        return size_gb > 2.0 or use_two_parts
    except:
        return use_two_parts


def process_vision_quantization(src_path, dst_path):
    """Handles dynamic quantization for Vision models."""
    print(f"Applying Dynamic Quantization for Vision model...")
    quantize_dynamic(
        model_input=quant_utils.load_model_with_shape_infer(Path(src_path)),
        model_output=dst_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': False,
            'EnableSubgraph': True,
            'ForceQuantizeNoInputCheck': False,
            'MatMulConstBOnly': True
        },
        nodes_to_exclude=None,
        use_external_data_format=TWO_PARTS_SAVE
    )


def process_weight_quantization(src_path, dst_path, algo, op_types, axes):
    """Handles weight-only quantization for standard models."""
    print("Loading model with shape inference...")
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    
    # Configure Algorithm
    common_args = {
        'quant_format': quant_utils.QuantFormat.QOperator,
        'op_types_to_quantize': tuple(op_types)
    }
    
    if algo == "RTN":
        q_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common_args)
    elif algo == "HQQ":
        q_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=BITS, block_size=BLOCK_SIZE, axis=axes[0],
            quant_axes=tuple((op_types[i], axes[i]) for i in range(len(op_types))),
            **common_args
        )
    elif algo == "k_quant":
        q_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(**common_args)
    else: # DEFAULT
        q_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=BLOCK_SIZE, is_symmetric=QUANT_SYMMETRIC, 
            accuracy_level=ACCURACY_LEVEL,
            quant_axes=tuple((op_types[i], axes[i]) for i in range(len(op_types))),
            **common_args
        )
    
    q_config.bits = BITS

    print(f"Starting quantization process ({algo})...")
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=BLOCK_SIZE,
        is_symmetric=QUANT_SYMMETRIC,
        accuracy_level=ACCURACY_LEVEL,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=tuple((op_types[i], axes[i]) for i in range(len(op_types))),
        algo_config=q_config,
        nodes_to_exclude=NODES_TO_EXCLUDE
    )
    quant.process()
    quant.model.save_model_to_file(dst_path, True)
    del model, quant
    gc.collect()


# --- Main Logic ---
def main():
    os.makedirs(QUANTED_FOLDER_PATH, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"--- Processing model: {model_name} ---")
        
        src_path, dst_path = get_model_paths(model_name)
        if not os.path.exists(src_path):
            print(f"Warning: Model file not found at {src_path}. Skipping.")
            continue

        # Validating two_parts setting for this specific model
        is_split_disabled = ('Concat' in model_name) or ('Rotary' in model_name)
        current_two_parts = False if is_split_disabled else TWO_PARTS_SAVE

        # --- Case 1: Reset_Penality (Optimization only) ---
        if 'Reset_Penality' in model_name:
            print("Optimizing Reset_Penality...")
            optimize_onnx_model(src_path, use_f16=USE_F16, save_external=False)
            if UPGRADE_OPSET > 0:
                upgrade_opset_version(src_path, UPGRADE_OPSET, False)
            continue

        # --- Case 2: Quantization ---
        # Determine algorithm and types
        current_algo = "DEFAULT" if 'Embed' in model_name else ALGORITHM
        op_types = ["Gather"] if 'Embed' in model_name else ["MatMul"]
        quant_axes = [1] if 'Embed' in model_name else [0]

        if "LLM_Vision" in model_name:
            if USE_Q8_VISION:
                process_vision_quantization(src_path, dst_path)
        else:
            process_weight_quantization(src_path, dst_path, current_algo, op_types, quant_axes)

        # --- Case 3: Post-Quantization Optimization Pipeline ---
        
        # Check size to determine storage format
        is_large = check_is_large_model(dst_path, current_two_parts)
        
        # 1. First onnxslim pass
        print("Applying first onnxslim pass...")
        # Note: Vision model uses src_path as input in original logic for slim, others use dst
        slim_input = src_path if ("LLM_Vision" in model_name and not USE_Q8_VISION) else dst_path
        slim(
            model=slim_input,
            output_model=dst_path,
            no_shape_infer=True,
            save_as_external_data=is_large,
            verbose=False
        )
        
        # 2. Transformers Optimization
        print("Applying transformers.optimizer...")
        num_heads, hidden_size = fetch_transformer_config(DOWNLOAD_PATH, model_name)
        optimize_onnx_model(dst_path, num_heads, hidden_size, USE_F16, save_external=is_large)

        # 3. Second onnxslim pass
        print("Applying second onnxslim pass...")
        run_onnxslim(dst_path, save_external=is_large)

        # 4. Opset Upgrade
        if UPGRADE_OPSET > 0:
            upgrade_opset_version(dst_path, UPGRADE_OPSET, save_external=is_large)
        else:
            # Re-save to ensure consistency if no upgrade requested (as per original logic)
            m = onnx.load(dst_path)
            onnx.save(m, dst_path, save_as_external_data=is_large)
            del m; gc.collect()

    # --- Cleanup ---
    print("Cleaning up temporary *.onnx.data files...")
    pattern = os.path.join(QUANTED_FOLDER_PATH, '*.onnx.data')
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print("--- All models processed successfully! ---")


if __name__ == "__main__":
    main()


