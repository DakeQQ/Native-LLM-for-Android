
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

model_folder_path = 'C:/Users/Downloads/StableLM2-1.6B-Chat'  # set the folder path where the StableLM whole project downloaded.
modified_path_A = 'C:./modeling_modified/modeling_stablelm.py'  # The path where store the modified modeling_stablelm.py
transformers_stablelm_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/stablelm/modeling_stablelm.py'  # The original modeling_stablelm.py path which was stored in the transformers python package.
onnx_model_A = 'C:/Users/Downloads/StableLM_ONNX/StableLM.onnx'  # Assign a path where the exported StableLM model stored.

# Load the model
shutil.copyfile(modified_path_A, transformers_stablelm_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_stablelm.py on line 862, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
partial_head_dim = int(model.config.partial_rotary_factor * head_dim)

# Generate dummies for torch.onnx.export()
input_ids = torch.ones(max_seq_len, dtype=torch.int32)
attention_mask = torch.zeros(1, dtype=torch.float32) - 999999999.0
ids_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
history_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
past_key_states = torch.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=torch.float16)
past_values_states = past_key_states
position_ids = torch.zeros((max_seq_len, 1), dtype=torch.float32)
for i in range(max_seq_len):
    position_ids[i, 0] = float(i)
theta = 10000.0 ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
idx_theta = position_ids * theta
cos_rotary_pos_emb = torch.cos(idx_theta)
sin_rotary_pos_emb = torch.sin(idx_theta)
cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()
model.register_buffer('cos_rotary_pos_emb', cos_rotary_pos_emb)
model.register_buffer('sin_rotary_pos_emb', sin_rotary_pos_emb)

print('Export start ...')
torch.onnx.export(
    model, (
        input_ids, attention_mask, past_key_states, past_values_states, history_len, ids_len),
    onnx_model_A,
    input_names=[
        'input_ids',
        'attention_mask',
        'past_key_states',
        'past_values_states',
        'history_len',
        'ids_len'
    ],
    output_names=['max_logit_id', 'past_key_states', 'past_values_states'],
    do_constant_folding=True,
    opset_version=17)
del model
del past_key_states
del past_values_states
del position_ids
del theta
del idx_theta
del cos_rotary_pos_emb
del sin_rotary_pos_emb
print('Export done!')

print('\nStart running the StableLM by ONNXRuntime.')
print('Now loading . . . it could cost minutes.\n')

# Run the exported model by ONNX Runtime
query = "Hello, how are you?"
max_single_chat_length = 341  # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(model_folder_path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 4  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts,  providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
in_name_A3 = in_name_A[3].name
in_name_A4 = in_name_A[4].name
in_name_A5 = in_name_A[5].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name

# Pre-process inputs
prompt = f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
token = tokenizer(prompt, return_tensors='pt')['input_ids']
ids_len = token.shape[1] + np.zeros(1, dtype=np.int64)
input_ids = np.zeros(max_seq_len, dtype=np.int32)
input_ids[:ids_len[0]] = token[0, :]
attention_mask = np.zeros(1, dtype=np.float32) - 999999999.0
history_len = np.zeros(1, dtype=np.int64)
past_key_states_A = np.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=np.float16)
past_values_states_A = past_key_states_A
num_decode = 0
print('Test Question: ' + query + "\n")
print('StableLM Answering:\n')

# Start to run LLM
start_time = time.time()
while history_len < max_single_chat_length:
    token_id, past_key_states_A, past_values_states_A = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2],
        {in_name_A0: input_ids,
         in_name_A1: attention_mask,
         in_name_A2: past_key_states_A,
         in_name_A3: past_values_states_A,
         in_name_A4: history_len,
         in_name_A5: ids_len})
    if (token_id == 100257) | (token_id == 100278):  # the stop_id in StableLM is "100257" & "100278"
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
