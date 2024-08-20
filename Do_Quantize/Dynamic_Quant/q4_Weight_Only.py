import os
import onnx
import onnx.version_converter
from onnxsim import simplify
from onnxruntime.quantization import QuantType, quantize_dynamic


# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Quanted"                   # The quanted folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_quanted.onnx")     # The quanted model stored path.


# Upgrade the Opset version. (optional process)
model = onnx.load(model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, quanted_model_path)
del model

# Preprocess, it also cost alot of memory during preprocess, you can close this command and keep quanting. Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.quantization.preprocess --auto_merge --all_tensors_to_one_file --input {quanted_model_path} --output {quant_model_path}'], shell=True)

# Start Weight-Only Quantize
block_size = 256            # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
symmetric = False           # False may get more accuracy.
accuracy_level = 2          # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
bits = 4                    # [2, 4, 8]
quant_method = 'gptq'       # ["default", "hqq", "rtn", "gptq"];
quant_format = 'QOperator'  # ["QOperator", "QDQ"]; QOperator format quantizes the model with quantized operators directly.  QDQ format quantize the model by inserting DeQuantizeLinear before the MatMul.,
nodes_to_exclude = ''       # Specify the unsupported op type, for example: ReduceMean
# Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.quantization.matmul_4bits_quantizer --input_model {quanted_model_path} --output_model {quanted_model_path} --block_size {block_size} --symmetric {symmetric} --accuracy_level {accuracy_level} --bits {bits} --quant_method {quant_method} --quant_format {quant_format} --nodes_to_exclude {nodes_to_exclude}'], shell=True)

# ONNX Model Optimizer
model, _ = simplify(
    model=onnx.load(quanted_model_path),
    include_subgraph=True,
    dynamic_input_shape=False,          # True for dynamic input.
    tensor_size_threshold="2GB",        # Must less than 2GB.
    perform_optimization=True,          # True for more optimize.
    skip_fuse_bn=False,                 # False for more optimize.
    skip_constant_folding=False,        # False for more optimize.
    skip_shape_inference=False,         # False for more optimize.
    mutable_initializer=False           # False for static initializer.
)
onnx.save(model, quanted_model_path)

# Convert the simplified model to ORT format.
optimization_style = "Runtime"          # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
target_platform = "arm"                 # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
# Call subprocess may get permission failed on Windows system.
subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)