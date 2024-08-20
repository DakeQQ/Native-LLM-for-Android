import os
import onnx
import onnx.version_converter
from onnxsim import simplify
from onnxconverter_common import float16

# Path Setting
original_folder_path = r"C:\Users\Downloads\Model_ONNX"                          # The original folder.
quanted_folder_path = r"C:\Users\Downloads\Model_ONNX_Quanted"                   # The quanted folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_quanted.onnx")     # The quanted model stored path.


# Upgrade the Opset version. (optional process)
# model = onnx.load(model_path)
# model = onnx.version_converter.convert_version(model, 21)
# onnx.save(model, quanted_model_path)
# del model

# Convert the fp32 to fp16
model = onnx.load(model_path, load_external_data=False)  # True for two-parts model.
model = float16.convert_float_to_float16(model,
                                         min_positive_val=0,
                                         max_finite_val=65504,
                                         keep_io_types=True,        # True for keep original input format.
                                         disable_shape_infer=False, # False for more optimize.
                                         op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'Resize'],  # The op type list for skip the conversion. These are known unsupported op type for fp16.
                                         node_block_list=None)      # The node name list for skip the conversion.

# ONNX Model Optimizer
model, _ = simplify(
    model=model,
    include_subgraph=True,
    dynamic_input_shape=False,      # True for dynamic input.
    tensor_size_threshold="1.99GB",    # Must less than 2GB.
    perform_optimization=True,      # True for more optimize.
    skip_fuse_bn=False,             # False for more optimize.
    skip_constant_folding=False,    # False for more optimize.
    skip_shape_inference=False,     # False for more optimize.
    mutable_initializer=False       # False for static initializer.
)
onnx.save(model, quanted_model_path)

# It is not recommended to convert an FP16 ONNX model to the ORT format because this process adds a Cast operation to convert the FP16 process back to FP32.
