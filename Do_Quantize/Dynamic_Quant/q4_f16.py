import os
import gc
import glob
import sys
import onnx
import torch
import subprocess
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
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Optimized"                 # The optimized folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_Optimized.onnx")   # The optimized model stored path.
download_path = r'C:\Users\Downloads\Qwen2-1.5B-Instruct'                        # Set the folder path where the LLM whole project downloaded, otherwise set "NONE".

target_platform = "arm"                                                          # ['arm', 'amd64']
use_gpu = True                                                                   # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CUDAExecutionProvider'                                               # ['CPUExecutionProvider', 'CUDAExecutionProvider']
use_low_memory_mode_in_Android = True                                            # If you need to use low memory mode on Android, please set it to True.
algorithm = "DEFAULT"                                                            # ["DEFAULT", "RTN", "HQQ",], HQQ will very slow both in quant and inference.
bits = 4                                                                         # [4, 8]
op_types = ["MatMul"]                                                            # ["MatMul", "Gather"]; Adding Gather may get errors.
quant_axes = [0]                                                                 # Target axes to quant the quant data.
block_size = 128                                                                 # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
accuracy_level = 2                                                               # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False                                                          # False may get more accuracy.
nodes_to_exclude = None                                                          # Set the node names here. Such as: ["/layers.0/mlp/down_proj/MatMul"]
upgrade_opset = 20                                                               # Optional process. Set 0 for close.


if 'QwenVL_A' in model_path:  # Embedding part
    op_types = ["Gather"]
    quant_axes = [1]

if ('QwenVL_C' in model_path) or ('QwenVL_D' in model_path) or ('QwenVL_E' in model_path):
    use_low_memory_mode_in_Android = False

# Start Weight-Only Quantize
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
    True                                           # save_as_external_data
)


model_size_bytes = sys.getsizeof(onnx.load(quanted_model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = True if use_low_memory_mode_in_Android else False


# ONNX Model Optimizer
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=True,                           # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


if download_path == "NONE":
    num_heads = 0    # default
    hidden_size = 0  # default
else:
    download_path_lower = download_path.lower()
    if ('vl' in download_path_lower) & ('qwen' in download_path_lower):
        if ("2.5" in download_path) or ("3b" in download_path_lower):
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    else:
        if ("gemma3" in download_path_lower) or ("gemma 3" in download_path_lower) or ("gemma-3" in download_path_lower):
            from transformers import Gemma3ForCausalLM
            model = Gemma3ForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    try:
        if "QwenVL_B" in model_path:  # Vision Part
            num_heads = model.config.vision_config.num_heads
            hidden_size = model.config.vision_config.hidden_size
        else:
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
    except:
        num_heads = model.config.llm_config.num_attention_heads
        hidden_size = model.config.llm_config.hidden_size
    del model
    gc.collect()
  

# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=2,
                       num_heads=num_heads,
                       hidden_size=hidden_size,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
model.convert_float_to_float16(
    keep_io_types=False,
    force_fp16_initializers=True,
    use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
    max_finite_val=65504.0,
    op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']  # Common fp16 overflow operators: 'Pow', 'ReduceMean', 'ReduceSum', 'Softmax'
)
model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,           # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
if upgrade_opset > 0:
    try:
        model = onnx.version_converter.convert_version(model, upgrade_opset)
        onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
        del model
        gc.collect()
    except:
        onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
        del model
        gc.collect()
else:
    onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
    del model
    gc.collect()


pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# It is not recommended to convert an FP16 ONNX model to the ORT format because this process adds a Cast operation to convert the FP16 process back to FP32.
