import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer

path_A = 'C:/Users/MiniCPM-2B-dpo-fp32-A'  # set the folder path where the MiniCPM downloaded.
# Replace the original "modeling_minicpm.py" with the modified "modeling_minicpm.py", which stored at the folder "modeling_modified_A".

path_B = 'C:/Users/MiniCPM-2B-dpo-fp32-B'  # Copy the previous downloaded folder and rename it to folder-B.
# Also replace another "modeling_minicpm.py" with the modified "modeling_minicpm.py", which stored at the folder "modeling_modified_B".

onnx_model_A = 'C:/Users/MiniCPM_ONNX_A/MiniCPM_part_A.onnx'  # Assign a path where the MiniCPM_part_A stored.
onnx_model_B = 'C:/Users/MiniCPM_ONNX_B/MiniCPM_part_B.onnx'  # Assign a path where the MiniCPM_part_B stored.

# Load the model
model = AutoModelForCausalLM.from_pretrained(path_A, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
max_seq_len = 768  # Please modify the same variable, which declared in the modified modeling_minicpm.py at line 1038, at the same time.
head_dim = 64  # from the model configs
block_nums = 20
num_heads = 36
num_block = 20  # The original is 40, but we split it into half.
hidden_size = 2304  # from the model configs

# Generate dummies for torch.onnx.export()
input_ids = torch.ones((1, max_seq_len), dtype=torch.int32)
attention_mask = torch.zeros(1, dtype=torch.float32) + -99999999999999.0
ids_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
history_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
offset = torch.zeros(1, dtype=torch.long)
position_ids = torch.zeros((max_seq_len, 1), dtype=torch.float32)
for i in range(max_seq_len):
    position_ids[i, 0] = float(i)
theta = torch.arange(0, head_dim, 2, dtype=torch.float32)
idx_theta = position_ids * theta
past_key_states = torch.zeros((block_nums, num_heads, max_seq_len, head_dim), dtype=torch.float32)
past_values_states = past_key_states
cos_rotary_pos_emb = torch.ones_like(idx_theta)
sin_rotary_pos_emb = cos_rotary_pos_emb
cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0)
sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0)
last_hidden_state = torch.ones((1, max_seq_len, hidden_size), dtype=torch.float32)

print('Part_A export start ...')
torch.onnx.export(
    model, (
        input_ids, attention_mask, cos_rotary_pos_emb, sin_rotary_pos_emb, past_key_states, past_values_states,
        history_len, ids_len),
    onnx_model_A,
    input_names=[
        'input_ids', 'attention_mask', 'cos_rotary_pos_emb', 'sin_rotary_pos_emb', 'past_key_states',
        'past_values_states', 'history_len',
        'ids_len'
    ],
    output_names=['last_hidden_state', 'past_key_states', 'past_values_states'],
    do_constant_folding=True,
    opset_version=17)
del model
print('Part_A export done!')

# Reload for part B
model = AutoModelForCausalLM.from_pretrained(path_B, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
print('Part_B export start ...')
torch.onnx.export(
    model, (
        last_hidden_state, attention_mask, cos_rotary_pos_emb, sin_rotary_pos_emb, past_key_states, past_values_states,
        history_len, ids_len),
    onnx_model_B,
    input_names=[
        'last_hidden_state', 'attention_mask', 'cos_rotary_pos_emb', 'sin_rotary_pos_emb', 'past_key_states',
        'past_values_states', 'history_len',
        'ids_len'
    ],
    output_names=['max_logit_id', 'past_key_states', 'past_values_states'],
    do_constant_folding=True,
    opset_version=17)
del model
print('Part_B export done!')

print('\nStart running the MiniCPM by ONNX Runtime\n')
# Run the exported model by ONNX Runtime
query = "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？"
max_single_chat_length = 341
tokenizer = AutoTokenizer.from_pretrained(path_A, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level
session_opts.intra_op_num_threads = 4  # CPU threads
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A0 = ort_session_A.get_inputs()[0].name
in_name_A1 = ort_session_A.get_inputs()[1].name
in_name_A2 = ort_session_A.get_inputs()[2].name
in_name_A3 = ort_session_A.get_inputs()[3].name
in_name_A4 = ort_session_A.get_inputs()[4].name
in_name_A5 = ort_session_A.get_inputs()[5].name
in_name_A6 = ort_session_A.get_inputs()[6].name
in_name_A7 = ort_session_A.get_inputs()[7].name
out_name_A0 = ort_session_A.get_outputs()[0].name
out_name_A1 = ort_session_A.get_outputs()[1].name
out_name_A2 = ort_session_A.get_outputs()[2].name

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B0 = ort_session_B.get_inputs()[0].name
in_name_B1 = ort_session_B.get_inputs()[1].name
in_name_B2 = ort_session_B.get_inputs()[2].name
in_name_B3 = ort_session_B.get_inputs()[3].name
in_name_B4 = ort_session_B.get_inputs()[4].name
in_name_B5 = ort_session_B.get_inputs()[5].name
in_name_B6 = ort_session_B.get_inputs()[6].name
in_name_B7 = ort_session_B.get_inputs()[7].name
out_name_B0 = ort_session_B.get_outputs()[0].name
out_name_B1 = ort_session_B.get_outputs()[1].name
out_name_B2 = ort_session_B.get_outputs()[2].name

# Pre-process inputs
history_str = tokenizer.apply_chat_template([{"role": 'user', "content": query}], tokenize=False, add_generation_prompt=False)
token = tokenizer(history_str, return_tensors='pt')['input_ids']
ids_len = token.shape[1] + np.zeros(1, dtype=np.int64)
input_ids = np.zeros((1, max_seq_len), dtype=np.int32)
input_ids[0, :ids_len[0]] = token[0, :]
attention_mask = -99999999999999.0 + np.zeros(1, dtype=np.float32)
position_ids = np.zeros((max_seq_len, 1), dtype=np.float32)
for i in range(max_seq_len):
    position_ids[i, 0] = float(i)
theta = 10000.0 ** -(np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
idx_theta = position_ids * theta
cos_rotary_pos_emb = np.cos(idx_theta)
sin_rotary_pos_emb = np.sin(idx_theta)
cos_rotary_pos_emb = np.expand_dims(np.concatenate((cos_rotary_pos_emb, cos_rotary_pos_emb), axis=-1), axis=0)
sin_rotary_pos_emb = np.expand_dims(np.concatenate((sin_rotary_pos_emb, sin_rotary_pos_emb), axis=-1), axis=0)
history_len = np.zeros(1, dtype=np.int64)
past_key_states_A = np.zeros((num_block, num_heads, max_seq_len, head_dim), dtype=np.float32)
past_values_states_A = past_key_states_A
past_key_states_B = past_key_states_A
past_values_states_B = past_key_states_A
num_decode = 0
print('Test Question: ' + query + "\n")
print('MiniCPM Answering:\n')

# Start to run LLM
start_time = time.time()
while history_len < max_single_chat_length:
    last_hidden_state, past_key_states_A, past_values_states_A = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2],
        {in_name_A0: input_ids, in_name_A1: attention_mask,
         in_name_A2: cos_rotary_pos_emb,
         in_name_A3: sin_rotary_pos_emb,
         in_name_A4: past_key_states_A,
         in_name_A5: past_values_states_A,
         in_name_A6: history_len,
         in_name_A7: ids_len})
    token_id, past_key_states_B, past_values_states_B = ort_session_B.run([out_name_B0, out_name_B1, out_name_B2],
                                                                          {in_name_B0: last_hidden_state,
                                                                           in_name_B1: attention_mask,
                                                                           in_name_B2: cos_rotary_pos_emb,
                                                                           in_name_B3: sin_rotary_pos_emb,
                                                                           in_name_B4: past_key_states_B,
                                                                           in_name_B5: past_values_states_B,
                                                                           in_name_B6: history_len,
                                                                           in_name_B7: ids_len})
    if token_id == 2:  # the stop_id in MiniCPM is "2"
        break
    else:
        history_len[0] += ids_len[0]
        ids_len[0] = 1
        num_decode += 1
        attention_mask[0] = 0.0
        input_ids[0, 0] = token_id
        print(tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(token_id)]), end="", flush=True)
end_time = time.time()
print("\n")
print(num_decode / (end_time - start_time))
print("token/s")
