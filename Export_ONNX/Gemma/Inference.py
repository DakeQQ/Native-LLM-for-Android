import time
import numpy as np
import onnxruntime
from transformers import AutoTokenizer


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
run_options_A.log_severity_level = 4                  # Fatal level = 4, it an adjustable value.
run_options_A.log_verbosity_level = 4                 # Fatal level = 4, it an adjustable value.
session_opts.log_severity_level = 4                   # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0                 # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0                 # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
run_options_A.add_run_config_entry("disable_synchronize_execution_providers", "1")

if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '1',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
else:
    device_type = 'cpu'
    provider_options = None
    device_id = 0

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options_A)
print(ort_session_A.get_providers())
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
output_metas = ort_session_A.get_outputs()
amount_of_outputs_A = len(out_name_A)
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs_A)]
num_layers = (amount_of_outputs_A - 2) // 2
model_dtype = output_metas[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32

# Pre-process inputs
prompt = f'<bos><start_of_turn>user\nYou are a helpful assistant.\n\n{test_query}<end_of_turn>\n<start_of_turn>model\n'
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._outputs_meta[0].shape[0], 1, ort_session_A._outputs_meta[0].shape[2], 0), dtype=model_dtype), device_type, DEVICE_ID)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._outputs_meta[num_layers].shape[0], 1, 0, ort_session_A._outputs_meta[num_layers].shape[-1]), dtype=model_dtype), device_type, DEVICE_ID)
num_keys_values = num_layers + num_layers
num_decode = 0

input_ids = input_ids._ortvalue
history_len = history_len._ortvalue
ids_len = ids_len._ortvalue
attention_mask_1 = attention_mask_1._ortvalue
past_keys_A = past_keys_A._ortvalue
past_values_A = past_values_A._ortvalue
attention_mask_0 = attention_mask_0._ortvalue
ids_len_1 = ids_len_1._ortvalue

print(f'\n\nTest Question: {test_query}\nGemma Answering:\n')

input_feed_A = {
    in_name_A[-4]: input_ids,
    in_name_A[-3]: history_len,
    in_name_A[-2]: ids_len,
    in_name_A[-1]: attention_mask_1,
}

for i in range(num_layers):
    input_feed_A[in_name_A[i]] = past_keys_A
for i in range(num_layers, num_keys_values):
    input_feed_A[in_name_A[i]] = past_values_A

# Start to run LLM
start_time = time.time()
while num_decode < max_single_chat_length:
    all_outputs_A = ort_session_A._sess.run_with_ort_values(
        input_feed_A,
        out_name_A,
        run_options_A
    )
    max_logit_ids = all_outputs_A[num_keys_values].numpy()[0, 0]
    num_decode += 1
    if max_logit_ids in STOP_TOKEN:
        break
    if num_decode < 2:
        input_feed_A[in_name_A[-1]] = attention_mask_0
        input_feed_A[in_name_A[-2]] = ids_len_1
    input_feed_A.update(zip(in_name_A[:amount_of_outputs_A], all_outputs_A))
    print(tokenizer.decode(max_logit_ids), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
