import os
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import gc


model_folder_path = 'C:/Users/Downloads/Gemma2-2B-it'  # Set the folder path where the Gemma whole project downloaded.
# Replace the original "modeling_gemma2.py" with the modified "modeling_gemma2.py", which stored at the folder "modeling_modified".
modified_path_C = './modeling_modified_C/modeling_gemma2.py'  # The path where the modified modeling_gemma2.py stored.
onnx_model_A = 'C:/Users/Downloads/Gemma_ONNX/part_A/Gemma_A.onnx'  # Assign a path where the exported Gemma model stored.
onnx_model_B = 'C:/Users/Downloads/Gemma_ONNX/part_B/Gemma_B.onnx'  # Assign a path where the exported Gemma model stored.
onnx_model_C = 'C:/Users/Downloads/Gemma_ONNX/part_C/Gemma_C.onnx'  # Assign a path where the exported Gemma model stored.
transformers_gemma2_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma2/modeling_gemma2.py'  # The original modeling_gemma2.py path which was stored in the transformers python package.

# Load the model
shutil.copyfile(modified_path_C, transformers_gemma2_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma2.py on line 875, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
last_hidden_state = torch.zeros(hidden_size, dtype=torch.float32)

print('Export Part_C start...')
with torch.inference_mode():
    torch.onnx.export(
        model,
        (last_hidden_state,),
        onnx_model_C,
        input_names=[
            'last_hidden_state'
        ],
        output_names=['max_logit_ids'],
        do_constant_folding=True,
        opset_version=17)
del model
del last_hidden_state
gc.collect()
print('Export Part_C done!\nStart running the Gemma by ONNXRuntime.\nNow loading . . . it could cost minutes.\n')

# Run the exported model by ONNX Runtime
query = "What is the tallest mountain in America, and how does its height compare to the tallest mountain on Earth in meters?"
max_single_chat_length = 341  # It a adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(model_folder_path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
in_name_B2 = in_name_B[2].name
in_name_B3 = in_name_B[3].name
in_name_B4 = in_name_B[4].name
in_name_B5 = in_name_B[5].name
out_name_B0 = out_name_B[0].name
out_name_B1 = out_name_B[1].name
out_name_B2 = out_name_B[2].name

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
out_name_C0 = out_name_C[0].name

# Pre-process inputs
prompt = f'<bos><start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
token = tokenizer(prompt, return_tensors='pt')['input_ids']
ids_len = token.shape[1] + np.zeros(1, dtype=np.int64)
input_ids = np.zeros(max_seq_len, dtype=np.int32)
input_ids[:ids_len[0]] = token[0, :]
attention_mask = np.array([-65504.0], dtype=np.float32)
history_len = np.zeros(1, dtype=np.int64)
past_key_states_B = np.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=np.float16)
past_values_states_B = past_key_states_B
num_decode = 0
print('\nTest Question: ' + query + "\n\nGemma Answering:\n")


# Start to run LLM
start_time = time.time()
while history_len < max_single_chat_length:
    hidden_state = ort_session_A.run(
        [out_name_A0],
        {in_name_A0: input_ids,
         in_name_A1: ids_len})[0]
    last_hidden_state, past_key_states_B, past_values_states_B = ort_session_B.run([out_name_B0, out_name_B1, out_name_B2],
                                                                          {in_name_B0: hidden_state,
                                                                           in_name_B1: attention_mask,
                                                                           in_name_B2: past_key_states_B,
                                                                           in_name_B3: past_values_states_B,
                                                                           in_name_B4: history_len,
                                                                           in_name_B5: ids_len})
    token_id = ort_session_C.run(
        [out_name_C0],
        {in_name_C0: last_hidden_state})[0]
    if (token_id == 1) | (token_id == 107):  # the stop_id in Gemma2 is "1" & "107"
        break
    else:
        history_len[0] += ids_len[0]
        ids_len[0] = 1
        num_decode += 1
        attention_mask[0] = 0.0
        input_ids[0] = token_id
        print(tokenizer.decode(token_id), end="", flush=True)
end_time = time.time()
print(f"\n\nDecode: {(num_decode / (end_time - start_time)):.3f} token/s")
