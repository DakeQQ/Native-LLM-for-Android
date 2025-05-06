import gc
import math
import time
import torch
import numpy as np
import onnxruntime
from transformers import Gemma3ForCausalLM, AutoTokenizer


path = '/home/iamj/Downloads/gemma-3-1b-it'                       # Set the folder path where the Gemma whole project downloaded.
onnx_model_A = '/home/iamj/Downloads/Gemma_ONNX/Gemma.onnx'       # Assign a path where the exported Gemma model stored.
STOP_TOKEN = [106, 1]                                             # The stop_id in Gemma is "106" & "1"
MAX_SEQ_LEN = 4096                                                # The max context length.
test_query = "地球最高的山是哪座山？"                                 # The test query after the export process.


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, kv_seq_len):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(-1, head_dim, kv_seq_len)


def repeat_v(kv_states, num_key_value_groups, head_dim, kv_seq_len):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(-1, kv_seq_len, head_dim)


class GEMMA(torch.nn.Module):
    def __init__(self, gemma, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(GEMMA, self).__init__()
        self.gemma = gemma
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = math.pow(head_dim, -0.25)
        for i in range(num_layers):
            self.gemma.model.layers._modules[f'{i}'].input_layernorm.weight.data += 1.0
            self.gemma.model.layers._modules[f'{i}'].self_attn.q_norm.weight.data += 1.0
            self.gemma.model.layers._modules[f'{i}'].self_attn.k_norm.weight.data += 1.0
            self.gemma.model.layers._modules[f'{i}'].self_attn.q_norm.weight.data *= scale_factor
            self.gemma.model.layers._modules[f'{i}'].self_attn.k_norm.weight.data *= scale_factor
            self.gemma.model.layers._modules[f'{i}'].post_attention_layernorm.weight.data += 1.0
            self.gemma.model.layers._modules[f'{i}'].pre_feedforward_layernorm.weight.data += 1.0
            self.gemma.model.layers._modules[f'{i}'].post_feedforward_layernorm.weight.data += 1.0
        self.gemma.model.norm.weight.data += 1.0

        data = self.gemma.model.embed_tokens.weight.data * self.gemma.model.embed_tokens.embed_scale
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        head_dim_range = -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        theta = 10000.0 ** head_dim_range
        idx_theta = position_ids * theta
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb_global = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
        self.sin_rotary_pos_emb_global = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()

        theta = 1000000.0 ** head_dim_range
        idx_theta = position_ids * theta
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb_local = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
        self.sin_rotary_pos_emb_local = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, *all_inputs):
        input_ids = all_inputs[-2]
        ids_len = input_ids.shape[1].unsqueeze(0)
        history_len = all_inputs[0].shape[-1].unsqueeze(0)
        kv_seq_len = ids_len + history_len
        rotary_pos_emb_cos_q_global = self.cos_rotary_pos_emb_global[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_q_global = self.sin_rotary_pos_emb_global[:, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_q_local = self.cos_rotary_pos_emb_local[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_q_local = self.sin_rotary_pos_emb_local[:, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_k_global = rotary_pos_emb_cos_q_global.transpose(-1, -2)
        rotary_pos_emb_sin_k_global = rotary_pos_emb_sin_q_global.transpose(-1, -2)
        rotary_pos_emb_cos_k_local = rotary_pos_emb_cos_q_local.transpose(-1, -2)
        rotary_pos_emb_sin_k_local = rotary_pos_emb_sin_q_local.transpose(-1, -2)
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.gemma.model.layers):
            if layer.self_attn.is_sliding:
                rotary_pos_emb_cos_q = rotary_pos_emb_cos_q_local
                rotary_pos_emb_sin_q = rotary_pos_emb_sin_q_local
                rotary_pos_emb_cos_k = rotary_pos_emb_cos_k_local
                rotary_pos_emb_sin_k = rotary_pos_emb_sin_k_local
            else:
                rotary_pos_emb_cos_q = rotary_pos_emb_cos_q_global
                rotary_pos_emb_sin_q = rotary_pos_emb_sin_q_global
                rotary_pos_emb_cos_k = rotary_pos_emb_cos_k_global
                rotary_pos_emb_sin_k = rotary_pos_emb_sin_k_global
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm.weight * hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = (layer.self_attn.q_norm.weight * q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).transpose(0, 1)
            k = (layer.self_attn.k_norm.weight * k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).permute(2, 1, 3, 0)
            q = q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, 2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, kv_seq_len)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, kv_seq_len)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, ids_len, -1))
            hidden_states = residual + layer.post_attention_layernorm.weight * attn_out / torch.sqrt(attn_out.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            residual = hidden_states
            hidden_states = layer.pre_feedforward_layernorm.weight * hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states = residual + layer.post_feedforward_layernorm.weight * hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        hidden_states = hidden_states[:, -1]
        hidden_states = self.gemma.model.norm.weight * hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return *self.save_key, *self.save_value, torch.argmax(self.gemma.lm_head(hidden_states), dim=-1, keepdim=True).int()


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = Gemma3ForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    # Build an optimized model
    model = GEMMA(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

    # Generate dummies for torch.onnx.export()
    attention_mask = torch.tensor([0], dtype=torch.int8)
    input_ids = torch.ones((1, 10), dtype=torch.int32)  # "10" is just a dummy value.
    past_keys = torch.zeros((num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)

    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {3: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'history_len_plus_ids_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('max_logit_id')

    # torch.onnx.export(
    #     model,
    #     tuple(all_inputs),
    #     onnx_model_A,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=dynamic_axes,
    #     do_constant_folding=True,
    #     opset_version=17
    # )
del model
del input_ids
del attention_mask
del past_keys
del past_values
del input_names
del output_names
del dynamic_axes
del all_inputs
gc.collect()
print('\nExport done!\n\nStart running the Gemma by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
max_single_chat_length = 512   # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0                 # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0                 # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()

# Pre-process inputs
prompt = f'<bos><start_of_turn>user\nYou are a helpful assistant.\n\n{test_query}<end_of_turn>\n<start_of_turn>model\n'
tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens.int().numpy(), 'cpu', 0)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, head_dim, 0), dtype=np.float32), 'cpu', 0)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, 0, head_dim), dtype=np.float32), 'cpu', 0)
num_keys_values = num_layers + num_layers
amount_of_outputs = len(out_name_A)
num_decode = 0
print(f'\n\nTest Question: {test_query}\nGemma Answering:\n')

output_names = []
input_feed = {
    in_name_A[-2].name: input_ids,
    in_name_A[-1].name: attention_mask
}
for i in range(num_layers):
    input_feed[in_name_A[i].name] = past_keys_A
    output_names.append(out_name_A[i].name)
for i in range(num_layers, num_keys_values):
    input_feed[in_name_A[i].name] = past_values_A
    output_names.append(out_name_A[i].name)
output_names.append(out_name_A[num_keys_values].name)

# Start to run LLM
start_time = time.time()
while num_decode < max_single_chat_length:
    all_outputs = ort_session_A.run_with_ort_values(
        output_names,
        input_feed
    )
    max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs[-1])
    num_decode += 1
    if max_logit_ids in STOP_TOKEN:  
        break
    for i in range(amount_of_outputs):
        input_feed[in_name_A[i].name] = all_outputs[i]
    if num_decode < 2:
        input_feed[in_name_A[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
    print(tokenizer.decode(max_logit_ids[0]), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
