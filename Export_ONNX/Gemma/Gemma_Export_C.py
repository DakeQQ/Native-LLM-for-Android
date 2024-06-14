import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil


model_folder_path = 'C:/Users/Downloads/Gemma1.1-2B-it'  # set the folder path where the Gemma whole project downloaded.
modified_path_C = 'C:./modeling_modified_C/modeling_gemma.py'  # The path where store the modified part_C modeling_gemma.py
transformers_gemma_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma/modeling_gemma.py'  # The original modeling_gemma.py path which was stored in the transformers python package.
onnx_model_A = 'C:/Users/Downloads/Gemma_ONNX/Gemma_part_A.onnx'  # The path where the exported Gemma_part_A model stored.
onnx_model_B = 'C:/Users/Downloads/Gemma_ONNX/Gemma_part_B.onnx'  # The path where the exported Gemma_part_B model stored.
onnx_model_C = 'C:/Users/Downloads/Gemma_ONNX/Gemma_part_C.onnx'  # Assign a path where the exported Gemma_part_C model stored.

# Load the model
shutil.copyfile(modified_path_C, transformers_gemma_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma.py on line 931, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
hidden_states_C = torch.ones(hidden_size, dtype=torch.float32)

print('Export Part_C start ...')
torch.onnx.export(
    model, hidden_states_C,
    onnx_model_C,
    input_names=[
        'hidden_states_C'
    ],
    output_names=['max_logit_id'],
    do_constant_folding=True,
    opset_version=17)
del model
print('Export Part_C done!')


print('\nStart running the Gemma1.1 by ONNX Runtime.')
print('Now loading . . . it could cost minutes.\n')

# Run the exported model by ONNX Runtime
query = "Write me a poem about Machine Learning."
max_single_chat_length = 341  # It a adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(model_folder_path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 4  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

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
in_name_C0 = ort_session_C.get_inputs()[0].name
out_name_C0 = ort_session_C.get_outputs()[0].name

# Pre-process inputs
prompt = f"\n\nQuery:{query}\n\nResponse:"
token = tokenizer(prompt, return_tensors='pt')['input_ids']
ids_len = token.shape[1] + np.zeros(1, dtype=np.int64)
input_ids = np.zeros(max_seq_len, dtype=np.int32)
input_ids[0, :ids_len[0]] = token[0, :]
attention_mask = np.array([-65504.0], dtype=np.float16)
history_len = np.zeros(1, dtype=np.int64)
past_key_states_B = np.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=np.float16)
past_values_states_B = past_key_states_B
num_decode = 0
print('Test Question: ' + query + "\n")
print('Gemma1.1 Answering:\n')

# Start to run LLM
start_time = time.time()
while history_len < max_single_chat_length:
    hidden_states_B = ort_session_A.run(
        [out_name_A0],
        {in_name_A0: input_ids, in_name_A1: ids_len})[0]

    hidden_states_C, past_key_states_B, past_values_states_B = ort_session_B.run(
        [out_name_B0, out_name_B1, out_name_B2],
        {in_name_B0: hidden_states_B,
         in_name_B1: attention_mask,
         in_name_B2: past_key_states_B,
         in_name_B3: past_values_states_B,
         in_name_B4: history_len,
         in_name_B5: ids_len})

    token_id = ort_session_C.run(
        [out_name_C0],
        {in_name_C0: hidden_states_C})[0]

    if token_id == 1:  # the stop_id in Gemma1.1 is "1"
        break
    else:
        history_len[0] += ids_len[0]
        ids_len[0] = 1
        num_decode += 1
        attention_mask[0] = 0.0
        input_ids[0] = token_id
        print(tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(token_id)]), end="", flush=True)
end_time = time.time()
print("\n")
print(num_decode / (end_time - start_time))
print("token/s")
