import time
import numpy as np
from transformers import AutoTokenizer
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import get_ort_device_type


path = '/home/DakeQQ/Downloads/gemma-3-270m-it'                       # Set the folder path where the Gemma whole project downloaded.
onnx_model_A = '/home/DakeQQ/Downloads/Gemma_Optimized/Gemma.onnx'    # Assign a path where the exported Gemma model stored.
STOP_TOKEN = [106, 1]                                                 # The stop_id in Gemma is "106" & "1"
MAX_SEQ_LEN = 4096                                                    # The max context length.
max_single_chat_length = MAX_SEQ_LEN                                  # It an adjustable value, but must less than max_seq_len.
test_query = "Hello"                                                  # The test query after the export process.
ORT_Accelerate_Providers = ['CUDAExecutionProvider']                  # Changed to CPU for this example
DEVICE_ID = 0

# Run the exported model by ONNX Runtime
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
run_options_A = onnxruntime.RunOptions()
run_options_A.log_severity_level = 4               # Fatal level = 4, it an adjustable value.
run_options_A.log_verbosity_level = 4              # Fatal level = 4, it an adjustable value.
session_opts.log_severity_level = 4                # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4               # Fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0              # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0              # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True           # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")

if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    # GPU configuration
    session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    run_options_A.add_run_config_entry("disable_synchronize_execution_providers", "1")
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,    # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',      # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                          # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',                       # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                    # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
    use_sync_operations = True
else:
    # CPU configuration
    device_type = 'cpu'
    use_sync_operations = False
    provider_options = None
    DEVICE_ID = 0  # CPU always uses device_id = 0


# Create ONNX Runtime session
ort_session_A = onnxruntime.InferenceSession(
    onnx_model_A,
    sess_options=session_opts,
    providers=ORT_Accelerate_Providers,
    provider_options=provider_options
)
print(f'\nUsable Device: {ort_session_A.get_providers()}')

# Get input and output metadata
in_name_A = ort_session_A.get_inputs()
output_metas = ort_session_A.get_outputs()
amount_of_inputs = len(in_name_A)
amount_of_outputs = len(output_metas)
in_name_A = [in_name_A[i].name for i in range(amount_of_inputs)]
out_name_A = [output_metas[i].name for i in range(amount_of_outputs)]
num_layers = (amount_of_outputs - 2) // 2
num_keys_values = num_layers + num_layers
model_dtype = output_metas[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32


# Pre-process inputs
prompt = f'<bos><start_of_turn>user\nYou are a helpful assistant.\n\n{test_query}<end_of_turn>\n<start_of_turn>model\n'
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)

# Create OrtValues on specified device (CPU or GPU) for initial inputs
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len_count = tokens.shape[-1]
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)

# Get output metadata for past key/value shapes

past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas[0].shape[0], 1, output_metas[0].shape[2], 0), dtype=model_dtype), device_type, DEVICE_ID)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas[num_layers].shape[0], 1, 0, output_metas[num_layers].shape[-1]), dtype=model_dtype), device_type, DEVICE_ID)

# Create IO Binding for optimized inference
io_binding_A = ort_session_A.io_binding()._iobinding

# Initial input binding setup
init_input_feed_A = [None] * amount_of_inputs
init_input_feed_A[-4] = input_ids._ortvalue
init_input_feed_A[-3] = history_len._ortvalue
init_input_feed_A[-2] = ids_len._ortvalue
init_input_feed_A[-1] = attention_mask_1._ortvalue

ids_len_1 = ids_len_1._ortvalue
attention_mask_0 = attention_mask_0._ortvalue

# Add past keys and values to inputs
for i in range(num_layers):
    init_input_feed_A[i] = past_keys_A._ortvalue
for i in range(num_layers, num_keys_values):
    init_input_feed_A[i] = past_values_A._ortvalue


def get_ort_device(device_type, device_id):
    return C.OrtDevice(get_ort_device_type(device_type, device_id), C.OrtDevice.default_memory(), device_id)


# Function to bind inputs using IO Binding
def bind_inputs_to_device(io_binding, input_names, ortvalue, num_inputs):
    for i in range(num_inputs):
        io_binding.bind_ortvalue_input(input_names[i], ortvalue[i])


# Function to bind outputs using IO Binding
def bind_outputs_to_device(io_binding, output_names, bind_device_type, num_outputs):
    for i in range(num_outputs):
        io_binding.bind_output(output_names[i], bind_device_type)


device_type_A = get_ort_device(device_type, DEVICE_ID)

# Start optimized LLM inference loop
print(f'\n\nTest Question: {test_query}\nGemma Answering:\n')
num_decode = 0
bind_inputs_to_device(io_binding_A, in_name_A, init_input_feed_A, amount_of_inputs)
start_time = time.time()
while num_decode < max_single_chat_length:
    bind_outputs_to_device(io_binding_A, out_name_A, device_type_A, amount_of_outputs)
    if use_sync_operations:
        io_binding_A.synchronize_inputs()
        ort_session_A._sess.run_with_iobinding(io_binding_A, run_options_A)
        io_binding_A.synchronize_outputs()
    else:
        ort_session_A._sess.run_with_iobinding(io_binding_A, run_options_A)
    all_outputs = io_binding_A.get_outputs()
    max_logit_ids = all_outputs[num_keys_values].numpy()[0, 0]
    num_decode += 1
    if max_logit_ids in STOP_TOKEN:
        break
    if num_decode < 2:
        io_binding_A.bind_ortvalue_input(in_name_A[-1], attention_mask_0)
        io_binding_A.bind_ortvalue_input(in_name_A[-2], ids_len_1)
    bind_inputs_to_device(io_binding_A, in_name_A, all_outputs, amount_of_outputs)
    print(tokenizer.decode(max_logit_ids), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
