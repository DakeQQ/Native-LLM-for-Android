import os
import gc
import sys
import onnx
import torch
import subprocess
import onnx.version_converter
from onnxsim import simplify
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Quanted"                   # The quanted folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_quanted.onnx")     # The quanted model stored path.
download_path = r'C:\Users\Downloads\Qwen2-1.5B-Instruct'                        # Set the folder path where the LLM whole project downloaded, otherwise set "NONE".
use_gpu = True                                                                   # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider']


# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
# (optional process)
# subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {model_path} --output {quanted_folder_path}'], shell=True)


# Start Quantize
model_size_bytes = sys.getsizeof(onnx.load(model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 8.0:
    is_large_model = True
else:
    is_large_model = False


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
                   'MatMulConstBOnly': False                 # False for more quant. Sometime, the inference speed may get worse.
                   },
    nodes_to_exclude=None,                                   # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
    use_external_data_format=is_large_model                  # Save the model into two parts.
)


# Convert the fp32 to fp16
model = onnx.load(quanted_model_path)
model = float16.convert_float_to_float16(model,
                                         min_positive_val=0,
                                         max_finite_val=65504,
                                         keep_io_types=True,        # True for keep original input format.
                                         disable_shape_infer=False, # False for more optimize.
                                         op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'Resize'],  # The op type list for skip the conversion. These are known unsupported op type for fp16.
                                         node_block_list=None)      # The node name list for skip the conversion.
model_size_bytes = sys.getsizeof(model.SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = False
onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
del model
gc.collect()


# ONNX Model Optimizer
if not is_large_model:
    # onnxsim 1st
    model, _ = simplify(
        model=onnx.load(quanted_model_path),
        include_subgraph=True,
        dynamic_input_shape=False,      # True for dynamic input.
        tensor_size_threshold="1.99GB", # Must less than 2GB.
        perform_optimization=True,      # True for more optimize.
        skip_fuse_bn=False,             # False for more optimize.
        skip_constant_folding=False,    # False for more optimize.
        skip_shape_inference=False,     # False for more optimize but may get errors.
        mutable_initializer=False       # False for static initializer.
    )
    onnx.save(model, quanted_model_path)
    del model
    gc.collect()


if download_path == "NONE":
    num_heads = 0    # default
    hidden_size = 0  # default
else:
    model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).eval()
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    del model
    gc.collect()


# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=99,
                       num_heads=num_heads,
                       hidden_size=hidden_size,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
del model
gc.collect()


# onnxsim 2nd
if not is_large_model:
    model, _ = simplify(
        model=onnx.load(quanted_model_path),
        include_subgraph=True,
        dynamic_input_shape=False,      # True for dynamic input.
        tensor_size_threshold="1.99GB", # Must less than 2GB.
        perform_optimization=True,      # True for more optimize.
        skip_fuse_bn=False,             # False for more optimize.
        skip_constant_folding=False,    # False for more optimize.
        skip_shape_inference=False,     # False for more optimize but may get errors.
        mutable_initializer=False       # False for static initializer.
    )
    onnx.save(model, quanted_model_path)
    del model
    gc.collect()


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)

# It is not recommended to convert an FP16 ONNX model to the ORT format because this process adds a Cast operation to convert the FP16 process back to FP32.

