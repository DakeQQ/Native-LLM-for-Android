import os
import gc
import glob
import sys
import onnx
import torch
import onnx.version_converter
from onnxslim import slim
from onnxconverter_common import float16
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Optimized"                 # The optimized folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_Optimized.onnx")   # The optimized model stored path.
download_path = r'C:\Users\Downloads\Qwen2-1.5B-Instruct'                        # Set the folder path where the LLM whole project downloaded, otherwise set "NONE".
use_gpu = True                                                                   # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CUDAExecutionProvider'                                               # ['CPUExecutionProvider', 'CUDAExecutionProvider'']
use_low_memory_mode_in_Android = True                                            # If you need to use low memory mode on Android, please set it to True.
upgrade_opset = 22                                                               # Optional process. Set 0 for close.


# Convert the fp32 to fp16
model = onnx.load(model_path)
model = float16.convert_float_to_float16(model,
                                         min_positive_val=1e-7,
                                         max_finite_val=65504,
                                         keep_io_types=False,        # True for keep original input format.
                                         disable_shape_infer=False,  # False for more optimize.
                                         op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'Resize', 'Range'],  # The op type list for skip the conversion. These are known unsupported op type for fp16.
                                         node_block_list=None)       # The node name list for skip the conversion.
onnx.save(model, quanted_model_path)
model_size_bytes = sys.getsizeof(model.SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = True if use_low_memory_mode_in_Android else False
del model
gc.collect()


# ONNX Model Optimizer
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                           # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


# transformers.optimizer, Optional, it may cause errors due to fusion operators.
# if download_path == "NONE":
#     num_heads = 0    # default
#     hidden_size = 0  # default
# else:
#     download_path_lower = download_path.lower()
#     if ('vl' in download_path_lower) & ('qwen' in download_path_lower):
#         if "2.5" in download_path or "3b" in download_path_lower:
#             from transformers import Qwen2_5_VLForConditionalGeneration
#             model = Qwen2_5_VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
#         else:
#             from transformers import Qwen2VLForConditionalGeneration
#             model = Qwen2VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
#     else:
#         if ("gemma3" in download_path_lower) or ("gemma 3" in download_path_lower) or ("gemma-3" in download_path_lower):
#             from transformers import Gemma3ForCausalLM
#             model = Gemma3ForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
#         else:
#             model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
#     try:
#         num_heads = model.config.num_attention_heads
#         hidden_size = model.config.hidden_size
#     except:
#         num_heads = model.config.llm_config.num_attention_heads
#         hidden_size = model.config.llm_config.hidden_size
#     del model
#     gc.collect()


# model = optimize_model(quanted_model_path,
#                        use_gpu=use_gpu,
#                        opt_level=2,
#                        num_heads=num_heads,
#                        hidden_size=hidden_size,
#                        provider=provider,
#                        verbose=False,
#                        model_type='bert')
# model.convert_float_to_float16(
#     keep_io_types=False,
#     force_fp16_initializers=True,
#     use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
#     max_finite_val=65504.0,
#     op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
# )
# model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
# del model
# gc.collect()


# onnxslim
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,                                           # False for more optimize but may get errors.
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
