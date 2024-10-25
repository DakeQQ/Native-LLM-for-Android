import os
import gc
import sys
import onnx
import torch
import onnx.version_converter
from onnxsim import simplify
from onnxconverter_common import float16
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


# Convert the fp32 to fp16
model = onnx.load(model_path)
model = float16.convert_float_to_float16(model,
                                         min_positive_val=0,
                                         max_finite_val=65504,
                                         keep_io_types=True,        # True for keep original input format.
                                         disable_shape_infer=False, # False for more optimize.
                                         op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'Resize'],  # The op type list for skip the conversion. These are known unsupported op type for fp16.
                                         node_block_list=None)      # The node name list for skip the conversion.
onnx.save(model, quanted_model_path)
model_size_bytes = sys.getsizeof(model.SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = False
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
        skip_shape_inference=True,      # False for more optimize but may get errors.
        mutable_initializer=False       # False for static initializer.
    )
    onnx.save(model, quanted_model_path)
    del model
    gc.collect()


if download_path == "NONE":
    num_heads = 0    # default
    hidden_size = 0  # default
else:
    model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True).eval()
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
model.convert_float_to_float16(
    keep_io_types=True,
    force_fp16_initializers=True,
    use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
    op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range']
)
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
        skip_shape_inference=True,      # False for more optimize but may get errors.
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
