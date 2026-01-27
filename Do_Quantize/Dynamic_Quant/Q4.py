import os
import gc
import sys
import glob
import onnx
import torch
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from transformers import AutoModelForCausalLM
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)

# Path Setting
original_folder_path = r"/home/DakeQQ/Downloads/Qwen_ONNX"           # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/Qwen_Optimized"        # The optimized folder.
download_path = r'/home/DakeQQ/Downloads/Qwen3-1.7B'                 # Set the folder path where the LLM whole project downloaded, otherwise set "NONE".

# Create the output directory if it doesn't exist
os.makedirs(quanted_folder_path, exist_ok=True)

# List of models to process
model_names = [
    "LLM_Embed",
    "LLM_Main",
    "Greedy_Search",
    "First_Beam_Search",
    "Second_Beam_Search",
    "Reset_Penality",
    "Argmax",
    # "QwenVL_A",
    # "QwenVL_B",
    # "QwenVL_C",
    # "QwenVL_D",
    # "QwenVL_E",
    # "QwenVL_F"
]

# Global Settings
algorithm = "DEFAULT"                              # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits = 4                                           # [4, 8]
block_size = 32                                    # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
accuracy_level = 4                                 # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
use_f16 = False                                    # Convert the F32 operators to F16.
quant_symmetric = False                            # False may get more accuracy.
nodes_to_exclude = None                            # Set the node names here. Such as: ["/layers.0/mlp/down_proj/MatMul"]
two_parts_save = False                             # If you need to use low memory mode on Android, please set it to True.
upgrade_opset = 0                                  # Optional process. Set 0 for close.


# --- Main Processing Loop ---
algorithm_copy = algorithm
for model_name in model_names:
    print(f"--- Processing model: {model_name} ---")

    # Dynamically set model paths for the current iteration
    if 'QwenVL_F' in model_name:
        model_path = os.path.join(original_folder_path.replace('Qwen_ONNX', 'Qwen_ONNX_2'), f"{model_name}.onnx")
    else:
        model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")
    
    # Check if the original model file exists before processing
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    if 'Reset_Penality' in model_name:
        model = optimize_model(model_path,
                               use_gpu=False,
                               opt_level=2,
                               num_heads=0,
                               hidden_size=0,
                               verbose=False,
                               model_type='bert',
                               only_onnxruntime=False)
        model.save_model_to_file(quanted_model_path, use_external_data_format=False)
        if upgrade_opset > 0:
            print(f"Upgrading Opset to {upgrade_opset}...")
            try:
                model = onnx.load(quanted_model_path)
                converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
                onnx.save(converted_model, quanted_model_path, save_as_external_data=False)
                del model, converted_model
                gc.collect()
            except Exception as e:
                print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
                model = onnx.load(quanted_model_path)
                onnx.save(model, quanted_model_path, save_as_external_data=False)
                del model
                gc.collect()
        else:
            model = onnx.load(quanted_model_path)
            onnx.save(model, quanted_model_path, save_as_external_data=False)
            del model
            gc.collect()
        continue

    # Model-specific configuration based on model name
    op_types = ["MatMul"]                                                        # ["MatMul", "Gather"]; Adding Gather may get errors.
    quant_axes = [0]                                                             # Target axes to quant the quant data.
    algorithm = algorithm_copy  # Reset to default
    
    # Special handling for specific model types
    if 'QwenVL_A' in model_name or 'Embed' in model_name:  # Embedding part
        op_types = ["Gather"]
        quant_axes = [1]
        algorithm = "DEFAULT"  # Fallback to DEFAULT for Gather operations
    
    if ('QwenVL_C' in model_name) or ('QwenVL_D' in model_name) or ('QwenVL_E' in model_name):
        _two_parts_save = False
    else:
        _two_parts_save = two_parts_save

    # Start Weight-Only Quantize
    print("Loading model with shape inference...")
    model = quant_utils.load_model_with_shape_infer(Path(model_path))

    if algorithm == "RTN":
        quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types)
        )
    elif algorithm == "HQQ":
        quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits,
            block_size=block_size,
            axis=quant_axes[0],
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
        )
    elif algorithm == "k_quant":
        quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types)
        )
    else:
        quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
        )
    
    quant_config.bits = bits
    print("Starting quantization process...")
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=quant_symmetric,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types))),
        algo_config=quant_config,
        nodes_to_exclude=nodes_to_exclude
    )
    quant.process()
    quant.model.save_model_to_file(
        quanted_model_path,
        True                                         # save_as_external_data
    )
    del model, quant
    gc.collect()

    # Determine if model is large
    model_size_bytes = sys.getsizeof(onnx.load(quanted_model_path).SerializeToString())
    model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
    if model_size_gb > 2.0:
        is_large_model = True
    else:
        is_large_model = True if _two_parts_save else False

    # ONNX Model Optimizer (onnxslim 1st pass)
    print("Applying first onnxslim pass...")
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=True,                          # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=is_large_model,
        verbose=False
    )

    # Get model configuration for transformers.optimizer
    if download_path.upper() == "NONE" or download_path is None:
        num_heads = 0    # default
        hidden_size = 0  # default
    else:
        download_path_lower = download_path.lower()
        try:
            if ('vl' in download_path_lower) & ('qwen' in download_path_lower):
                if "2.5" in download_path:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model_config = Qwen2_5_VLForConditionalGeneration.from_pretrained(download_path, dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
                elif "3" in download_path:
                    from transformers import Qwen3VLForConditionalGeneration
                    model_config = Qwen3VLForConditionalGeneration.from_pretrained(download_path, dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
                else:
                    from transformers import Qwen2VLForConditionalGeneration
                    model_config = Qwen2VLForConditionalGeneration.from_pretrained(download_path, dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
            else:
                if ("gemma3" in download_path_lower) or ("gemma 3" in download_path_lower) or ("gemma-3" in download_path_lower):
                    from transformers import Gemma3ForCausalLM
                    model_config = Gemma3ForCausalLM.from_pretrained(download_path, dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
                else:
                    model_config = AutoModelForCausalLM.from_pretrained(download_path, dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
            
            try:
                if "QwenVL_B" in model_name:  # Vision Part
                    num_heads = model_config.config.vision_config.num_heads
                    hidden_size = model_config.config.vision_config.hidden_size
                else:
                    num_heads = model_config.config.num_attention_heads
                    hidden_size = model_config.config.hidden_size
            except:
                try:
                    num_heads = model_config.config.llm_config.num_attention_heads
                    hidden_size = model_config.config.llm_config.hidden_size
                except:
                    num_heads = 0
                    hidden_size = 0
            
            del model_config
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not load model config: {e}. Using default values.")
            num_heads = 0
            hidden_size = 0

    # transformers.optimizer
    print("Applying transformers.optimizer...")
    model = optimize_model(quanted_model_path,
                           use_gpu=False,
                           opt_level=2,
                           num_heads=num_heads,
                           hidden_size=hidden_size,
                           verbose=False,
                           model_type='bert',
                           only_onnxruntime=False)
    if use_f16:
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
            max_finite_val=32767.0,
            min_positive_val=1e-7,
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'MatMulIntegerToFloat', 'Pow', 'ReduceMean', 'ReduceSum']
            # Common fp16 overflow operators: 'Pow', 'ReduceMean', 'ReduceSum', 'Softmax', 'Sigmoid', 'Erf'
        )
    model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
    del model
    gc.collect()

    # onnxslim 2nd pass
    print("Applying second onnxslim pass...")
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=True,                         # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=is_large_model,
        verbose=False
    )

    # Upgrade the Opset version. (optional process)
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            model = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=is_large_model)
            del model, converted_model
            gc.collect()
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            model = onnx.load(quanted_model_path)
            onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
            del model
            gc.collect()
    else:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
        del model
        gc.collect()


# Clean up external data files at the very end
print("Cleaning up temporary *.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")


