import os
import gc
import sys
import onnx
import torch
import subprocess
import onnx.version_converter
from onnxsim import simplify
from onnxslim import slim
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Optimized"                 # The optimized folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_Optimized.onnx")   # The optimized model stored path.
download_path = r'C:\Users\Downloads\Qwen2-1.5B-Instruct'                        # Set the folder path where the LLM whole project downloaded, otherwise set "NONE".
use_gpu = True                                                                   # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider']
use_low_memory_mode_in_Android = False                                           # If you need to use low memory mode on Android, please set it to True.


# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
# (optional process)
# subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {model_path} --output {quanted_folder_path}'], shell=True)


# Start Quantize
quantize_dynamic(
    model_input=model_path,
    model_output=quanted_model_path,
    per_channel=True,                                        # True for model accuracy but cost a lot of time during quanting process.
    reduce_range=False,                                      # True for some x86_64 platform.
    weight_type=QuantType.QUInt8,                            # It is recommended using uint8 + Symmetric False
    extra_options={'ActivationSymmetric': False,             # True for inference speed. False may keep more accuracy.
                   'WeightSymmetric': False,                 # True for inference speed. False may keep more accuracy.
                   'EnableSubgraph': True,                   # True for more quant.
                   'ForceQuantizeNoInputCheck': False,       # True for more quant.
                   'MatMulConstBOnly': True                  # False for more quant. Sometime, the inference speed may get worse.
                   },
    nodes_to_exclude=None,                                   # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
    use_external_data_format=True                            # Save the model into two parts.
)

model_size_bytes = sys.getsizeof(onnx.load(model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10            # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = True if use_low_memory_mode_in_Android else False


# ONNX Model Optimizer
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,   # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


if download_path == "NONE":
    num_heads = 0    # default
    hidden_size = 0  # default
else:
    if ('vl' in download_path.lower()) & ('qwen' in download_path.lower()):
        if "2.5" in download_path or "3b" in download_path.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
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
    keep_io_types=True,
    force_fp16_initializers=True,
    use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
    op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
)
model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=False,   # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)

# It is not recommended to convert an FP16 ONNX model to the ORT format because this process adds a Cast operation to convert the FP16 process back to FP32.

