import time
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

path = '/home/DakeQQ/Downloads/gemma-3-1b-it'                       # Set the folder path where the Gemma whole project downloaded.
onnx_model_A = '/home/DakeQQ/Downloads/Gemma_Optimized/Gemma.onnx'  # Assign a path where the exported Gemma model stored.
STOP_TOKEN = [106, 1]                                               # The stop_id in Gemma is "106" & "1"
MAX_SEQ_LEN = 4096                                                  # The max context length.
test_query = "地球最高的山是哪座山？"                                   # The test query after the export process.
ORT_Accelerate_Providers = ['CUDAExecutionProvider']                # Changed to CPU for this example
DEVICE_ID = 0

# Run the exported model by ONNX Runtime
max_single_chat_length = MAX_SEQ_LEN                                # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4               # Fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0              # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0              # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True           # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")

run_options = onnxruntime.RunOptions()

if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    # GPU configuration
    session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

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
            'do_copy_in_default_stream': '1',
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
    # For CPU, we don't need device allocator for initializers
    # and synchronization settings are different
    provider_options = None
    device_type = 'cpu'
    use_sync_operations = False
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
amount_of_outputs = len(output_metas)
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
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
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)

# Get output metadata for past key/value shapes

past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas[0].shape[0], 1, output_metas[0].shape[2], 0), dtype=model_dtype), device_type, DEVICE_ID)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas[num_layers].shape[0], 1, 0, output_metas[num_layers].shape[-1]), dtype=model_dtype), device_type, DEVICE_ID)

# Create IO Binding for optimized inference
io_binding = ort_session_A.io_binding()

# Initial input binding setup
input_feed_A = {
    in_name_A[-4]: input_ids,
    in_name_A[-3]: history_len,
    in_name_A[-2]: ids_len,
    in_name_A[-1]: attention_mask_1
}

# Add past keys and values to inputs
for i in range(num_layers):
    input_feed_A[in_name_A[i]] = past_keys_A
for i in range(num_layers, num_keys_values):
    input_feed_A[in_name_A[i]] = past_values_A


# Function to bind inputs using IO Binding
def bind_inputs_to_device(io_binding, input_dict):
    io_binding.clear_binding_inputs()
    for name, ortvalue in input_dict.items():
        io_binding.bind_ortvalue_input(name, ortvalue)


# Function to bind outputs using IO Binding
def bind_outputs_to_device(io_binding, output_names, device_type, device_id):
    io_binding.clear_binding_outputs()
    for name in output_names:
        io_binding.bind_output(name, device_type, device_id)


# Start optimized LLM inference loop
num_decode = 0
print(f'\n\nTest Question: {test_query}\nGemma Answering:\n')
start_time = time.time()
while num_decode < max_single_chat_length:
    bind_inputs_to_device(io_binding, input_feed_A)
    bind_outputs_to_device(io_binding, out_name_A, device_type, DEVICE_ID)
    if use_sync_operations:
        io_binding.synchronize_inputs()
        ort_session_A.run_with_iobinding(io_binding, run_options)
        io_binding.synchronize_outputs()
    else:
        ort_session_A.run_with_iobinding(io_binding, run_options)
    all_outputs = io_binding.get_outputs()
    max_logit_ids = all_outputs[-2].numpy()[0]
    num_decode += 1
    if max_logit_ids in STOP_TOKEN:
        break
    input_feed_A.update(zip(in_name_A[:amount_of_outputs], all_outputs))
    if num_decode < 2:
        input_feed_A[in_name_A[-1]] = attention_mask_0
        input_feed_A[in_name_A[-2]] = ids_len_1
    print(tokenizer.decode(max_logit_ids), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
