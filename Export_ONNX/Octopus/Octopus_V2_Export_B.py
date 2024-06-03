import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

model_folder_path = 'C:/Users/Downloads/Octopus_V2-2B-fp32'  # set the folder path where the Octopus_V2 whole project downloaded.
modified_path_B = 'C:./modeling_modified_B/modeling_gemma.py'  # The path where store the modified part_B modeling_gemma.py
transformers_gemma_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma/modeling_gemma.py'  # The original modeling_gemma.py path which was stored in the transformers python package.
onnx_model_B = 'C:/Users/Downloads/Octopus_ONNX/Octopus_part_B.onnx'  # Assign a path where the exported Octopus_part_B model stored.

# Load the model
shutil.copyfile(modified_path_B, transformers_gemma_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma.py on line 931, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
attention_mask = torch.zeros(1, dtype=torch.float32) - 999999999.0
ids_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
history_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.
hidden_states_B = torch.ones((max_seq_len, hidden_size), dtype=torch.float32)
past_key_states = torch.zeros((num_layers, num_key_value_heads, max_seq_len, head_dim), dtype=torch.float16)
past_values_states = past_key_states
position_ids = torch.zeros((max_seq_len, 1), dtype=torch.float32)
for i in range(max_seq_len):
    position_ids[i, 0] = float(i)
theta = 10000.0 ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
idx_theta = position_ids * theta
cos_rotary_pos_emb = torch.cos(idx_theta)
sin_rotary_pos_emb = torch.sin(idx_theta)
cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0)
sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0)
model.register_buffer('cos_rotary_pos_emb', cos_rotary_pos_emb)
model.register_buffer('sin_rotary_pos_emb', sin_rotary_pos_emb)


print('Export Part_B start ...')
torch.onnx.export(
    model, (
        hidden_states_B, attention_mask, past_key_states, past_values_states,
        history_len, ids_len),
    onnx_model_B,
    input_names=[
        'hidden_states_B',
        'attention_mask',
        'past_key_states',
        'past_values_states',
        'history_len',
        'ids_len'
    ],
    output_names=['hidden_states_C', 'past_key_states', 'past_values_states'],
    do_constant_folding=True,
    opset_version=17)
print('Export Part_B done!')
