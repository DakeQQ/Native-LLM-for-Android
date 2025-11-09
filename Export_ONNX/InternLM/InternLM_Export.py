import gc
import math
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = r'/home/DakeQQ/Downloads/internlm3-8b-instruct'                      # Set the folder path where the InternLM whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/InternLM_ONNX/InternLM.onnx'        # Assign a path where the exported InternLM model stored.
STOP_TOKEN = [2, 128131]                                                    # The stop_id in InternLM is "2" & "128131"
MAX_SEQ_LEN = 4096                                                          # The max context length.
test_query = "地球最高的山是哪座山？"                                           # The test query after the export process.


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


class INTERNLM(torch.nn.Module):
    def __init__(self, internlm, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(INTERNLM, self).__init__()
        self.internlm = internlm
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)
        self.hidden_size = self.internlm.model.layers._modules['0'].self_attn.o_proj.in_features

        scale_factor = math.pow(head_dim, -0.25)
        for i in range(num_layers):
            self.internlm.model.layers._modules[f'{i}'].self_attn.q_proj.weight.data *= scale_factor
            self.internlm.model.layers._modules[f'{i}'].self_attn.k_proj.weight.data *= scale_factor

        data = self.internlm.model.embed_tokens.weight.data
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        theta = self.internlm.config.rope_theta ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        idx_theta = position_ids * theta
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
        self.sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, *all_inputs):
        input_ids = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2)
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.internlm.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).permute(2, 1, 3, 0)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            k = torch.cat((all_inputs[i], k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, 2) * rotary_pos_emb_sin_k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads)
            attn = torch.nn.functional.softmax(torch.matmul(q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, -1, self.hidden_size))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.internlm.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        return *self.save_key, *self.save_value, torch.argmax(self.internlm.lm_head(hidden_states), dim=-1, keepdim=True).int(), kv_seq_len


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    # Build an optimized model
    model = INTERNLM(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

    # Generate dummies for torch.onnx.export()
    attention_mask = torch.tensor([0], dtype=torch.int8)
    ids_len = torch.tensor([10], dtype=torch.int64)  # "10" is just a dummy value.
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    history_len = torch.zeros(1, dtype=torch.int64)
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
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('max_logit_id')
    output_names.append('kv_seq_len')

    torch.onnx.export(
        model,
        tuple(all_inputs),
        onnx_model_A,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
del model
del input_ids
del ids_len
del history_len
del attention_mask
del past_keys
del past_values
del input_names
del output_names
del dynamic_axes
del all_inputs
gc.collect()
print('\nExport done!\n\nStart running the InternLM by ONNX Runtime.\nNow loading . . . it could cost minutes.')

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
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
amount_of_outputs = len(out_name_A)
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs)]

# Pre-process inputs
prompt = f'<|im_start|>user\n{test_query}<|im_end|>\n<|im_start|>assistant\n'
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, 'cpu', 0)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), 'cpu', 0)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, head_dim, 0), dtype=np.float32), 'cpu', 0)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, 0, head_dim), dtype=np.float32), 'cpu', 0)
num_keys_values = num_layers + num_layers
num_decode = 0
print('\n\nTest Question: ' + test_query + "\nInternLM Answering:\n")

input_feed = {
    in_name_A[-4]: input_ids,
    in_name_A[-3]: history_len,
    in_name_A[-2]: ids_len,
    in_name_A[-1]: attention_mask
}

for i in range(num_layers):
    input_feed[in_name_A[i]] = past_keys_A
for i in range(num_layers, num_keys_values):
    input_feed[in_name_A[i]] = past_values_A

# Start to run LLM
start_time = time.time()
while num_decode < max_single_chat_length:
    all_outputs = ort_session_A.run_with_ort_values(
        out_name_A,
        input_feed
    )
    max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs[-2])
    num_decode += 1
    if max_logit_ids in STOP_TOKEN:
        break
    for i in range(amount_of_outputs):
        input_feed[in_name_A[i]] = all_outputs[i]
    if num_decode < 2:
        input_feed[in_name_A[-1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
        input_feed[in_name_A[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
    print(tokenizer.decode(max_logit_ids[0]), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")



