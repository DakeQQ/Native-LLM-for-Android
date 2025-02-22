import os
import gc
import sys
import onnx
import torch
import subprocess
import onnx.version_converter
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
use_gpu = False                                                                  # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider']
use_low_memory_mode_in_Android = False                                           # If you need to use low memory mode on Android, please set it to True.


# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
# (optional process)
# subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {model_path} --output {quanted_folder_path}'], shell=True)


# Start Weight-Only Quantize
block_size = 256            # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
symmetric = False           # False may get more accuracy.
accuracy_level = 4          # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
bits = 4                    # [2, 4, 8]
quant_method = 'default'    # ["default", "hqq", "rtn", "gptq"];  default is recommended, or you will get errors.
quant_format = 'QOperator'  # ["QOperator", "QDQ"]; QOperator format quantizes the model with quantized operators directly.  QDQ format quantize the model by inserting DeQuantizeLinear before the MatMul.,
nodes_to_exclude = None     # Specify the unsupported op type, for example: ReduceMean
# Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.quantization.matmul_4bits_quantizer --input_model {model_path} --output_model {quanted_model_path} --block_size {block_size} --symmetric {symmetric} --accuracy_level {accuracy_level} --bits {bits} --quant_method {quant_method} --quant_format {quant_format} --nodes_to_exclude {nodes_to_exclude}'], shell=True)


# Start Quantize
def find_nodes_of_type(model_path, node_type):
    model = onnx.load(model_path)
    nodes_to_exclude = set()
    for node in model.graph.node:
        if node.op_type == node_type:
            nodes_to_exclude.add(node.name)
    return nodes_to_exclude


model_size_bytes = sys.getsizeof(onnx.load(quanted_model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = True if use_low_memory_mode_in_Android else False


nodes_to_exclude = find_nodes_of_type(quanted_model_path, "MatMulNBits")  # "To avoid duplicate quantization."
quantize_dynamic(
    model_input=quanted_model_path,
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
    nodes_to_exclude=nodes_to_exclude,                       # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
    use_external_data_format=is_large_model                  # Save the model into two parts.
)


model_size_bytes = sys.getsizeof(onnx.load(quanted_model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = False


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
del model
gc.collect()


if not is_large_model:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"  # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    target_platform = "arm"                 # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)
