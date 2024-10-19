import torch
from transformers import AutoModelForCausalLM
import shutil


model_folder_path = 'C:/Users/Downloads/Gemma2-2B-it'  # Set the folder path where the Gemma whole project downloaded.
# Replace the original "modeling_gemma2.py" with the modified "modeling_gemma2.py", which stored at the folder "modeling_modified".
modified_path_B = './modeling_modified_B/modeling_gemma2.py'  # The path where the modified modeling_gemma2.py stored.
onnx_model_B = 'C:/Users/Downloads/Gemma_ONNX/part_B/Gemma_B.onnx'  # Assign a path where the exported Gemma model stored.
transformers_gemma2_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma2/modeling_gemma2.py'  # The original modeling_gemma2.py path which was stored in the transformers python package.

# Load the model
shutil.copyfile(modified_path_B, transformers_gemma2_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma2.py on line 874, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
hidden_state = torch.zeros((max_seq_len , hidden_size), dtype=torch.float16)
attention_mask = torch.tensor([-65504.0], dtype=torch.float32)
ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
history_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
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
model.model.norm.weight.data += 1.0
sqrt_hidden_size = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
model.model.norm.weight.data *= sqrt_hidden_size
for i in range(num_layers):
    model.model.layers._modules[f'{i}'].input_layernorm.weight.data += 1.0
    model.model.layers._modules[f'{i}'].post_attention_layernorm.weight.data += 1.0
    model.model.layers._modules[f'{i}'].pre_feedforward_layernorm.weight.data += 1.0
    model.model.layers._modules[f'{i}'].post_feedforward_layernorm.weight.data += 1.0

    model.model.layers._modules[f'{i}'].input_layernorm.weight.data *= sqrt_hidden_size
    model.model.layers._modules[f'{i}'].post_attention_layernorm.weight.data *= sqrt_hidden_size
    model.model.layers._modules[f'{i}'].pre_feedforward_layernorm.weight.data *= sqrt_hidden_size
    model.model.layers._modules[f'{i}'].post_feedforward_layernorm.weight.data *= sqrt_hidden_size

    layer_attn = model.model.layers._modules[f'{i}'].self_attn
    qkv_weight = torch.cat([layer_attn.q_proj.weight.data, layer_attn.k_proj.weight.data, layer_attn.v_proj.weight.data], dim=0)
    qkv_bias = torch.cat([layer_attn.q_proj.bias.data, layer_attn.k_proj.bias.data, layer_attn.v_proj.bias.data], dim=0)
    layer_attn.qkv_proj = torch.nn.Linear(qkv_weight.shape[1], qkv_weight.shape[0], bias=True)
    layer_attn.qkv_proj.weight.data.copy_(qkv_weight)
    layer_attn.qkv_proj.bias.data.copy_(qkv_bias)
    del layer_attn.q_proj
    del layer_attn.k_proj
    del layer_attn.v_proj

print('Export Part_B start ...')
with torch.no_grad():
    torch.onnx.export(
        model, (
            hidden_state, attention_mask, past_key_states, past_values_states, history_len, ids_len),
        onnx_model_B,
        input_names=[
            'hidden_state',
            'attention_mask',
            'past_key_states',
            'past_values_states',
            'history_len',
            'ids_len'
        ],
        output_names=['last_hidden_state', 'past_key_states', 'past_values_states'],
        do_constant_folding=True,
        opset_version=17)
print('Part_B Done!')
