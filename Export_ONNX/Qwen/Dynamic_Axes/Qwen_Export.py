import gc
import math
import time
import site
import shutil
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = '/home/DakeQQ/Downloads/DeepSeek-R1-Distill-Qwen-1.5B'  # Set the folder path where the Qwen whole project downloaded.
onnx_model_A = '/home/DakeQQ/Downloads/Qwen_ONNX/Qwen.onnx'    # Assign a path where the exported Qwen model stored.

# Load the model
shutil.copyfile("./modeling_modified/modeling_qwen2.py", site.getsitepackages()[-1] + "/transformers/models/qwen2/modeling_qwen2.py")
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
max_seq_len = 4096  # Please modify the same variable, which declared in the modified modeling_qwen2.py on line 692, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
attention_mask = torch.tensor([0], dtype=torch.int8)
input_ids = torch.ones((1, 10), dtype=torch.int32)  # "10" is just a dummy value.
past_keys = torch.zeros((num_key_value_heads, 1, head_dim, 0), dtype=torch.float16)
past_values = torch.zeros((num_key_value_heads, 1, 0, head_dim), dtype=torch.float16)
position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
theta = 10000.0 ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
idx_theta = position_ids * theta
cos_rotary_pos_emb = torch.cos(idx_theta)
sin_rotary_pos_emb = torch.sin(idx_theta)
cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()
model.register_buffer('cos_rotary_pos_emb', cos_rotary_pos_emb)
model.register_buffer('sin_rotary_pos_emb', sin_rotary_pos_emb)


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


data = model.model.embed_tokens.weight.data
zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
model.register_buffer("scale", scale)
model.register_buffer("zero_point", zero_point)
model.register_buffer("embed_data", embed_data)

scale_factor = math.pow(head_dim, -0.25)
for i in range(num_layers):
    model.model.layers._modules[f'{i}'].self_attn.q_proj.weight.data *= scale_factor
    model.model.layers._modules[f'{i}'].self_attn.q_proj.bias.data *= scale_factor
    model.model.layers._modules[f'{i}'].self_attn.k_proj.weight.data *= scale_factor
    model.model.layers._modules[f'{i}'].self_attn.k_proj.bias.data *= scale_factor

del position_ids
del theta
del idx_theta
del cos_rotary_pos_emb
del sin_rotary_pos_emb
del embed_data
del data
del scale
del zero_point
gc.collect()

# Prepare input and output names
input_names = []
keys_values = []
output_names = []
dynamic_axes = {'input_ids': {1: 'ids_len'}}
for i in range(num_layers):
    name = f'in_key_{i}'
    input_names.append(name)
    keys_values.append(past_keys)
    dynamic_axes[name] = {3: 'history_len'}
    name = f'out_key_{i}'
    output_names.append(name)
    dynamic_axes[name] = {3: 'history_len_plus_ids_len'}

for i in range(num_layers):
    name = f'in_value_{i}'
    input_names.append(name)
    keys_values.append(past_values)
    dynamic_axes[name] = {2: 'history_len'}
    name = f'out_value_{i}'
    output_names.append(name)
    dynamic_axes[name] = {2: 'history_len_plus_ids_len'}

input_names.append('input_ids')
input_names.append('attention_mask')
output_names.append('max_logit_id')

print('Export start ...')
with torch.inference_mode():
    torch.onnx.export(
        model,
        tuple(keys_values + [input_ids, attention_mask]),
        onnx_model_A,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
del model
del input_ids
del attention_mask
del past_keys
del past_values
del input_names
del output_names
del dynamic_axes
del keys_values
gc.collect()
print('\nExport done!\n\nStart running the Qwen by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
query = "地球最高的山是哪座山？"
max_single_chat_length = 512  # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()

# Pre-process inputs
if "Deep" in path or "deep" in path or "Distill" in path or "distill" in path:
    #  prompt = f'<|begin▁of▁sentence|><｜User｜>{query}<｜Assistant｜>'
    head = torch.tensor([[151646, 151644]], dtype=torch.int32)
    tail = torch.tensor([[151645]], dtype=torch.int32)
    tokens = tokenizer(query, return_tensors='pt')['input_ids']
    tokens = torch.cat((head, tokens, tail), dim=-1)
else:
    prompt = f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens.int().numpy(), 'cpu', 0)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, head_dim, 0), dtype=np.float16), 'cpu', 0)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, 0, head_dim), dtype=np.float16), 'cpu', 0)
num_keys_values = num_layers + num_layers
amount_of_outputs = len(out_name_A)
num_decode = 0
print('\n\nTest Question: ' + query + "\nQwen Answering:\n")

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
    if max_logit_ids in [151643, 151645]:  # the stop_id in Qwen is "151643" & "151645"
        break
    else:
        for i in range(amount_of_outputs):
            input_feed[in_name_A[i].name] = all_outputs[i]
        if num_decode < 1:
            input_feed[in_name_A[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
        num_decode += 1
        print(tokenizer.decode(max_logit_ids[0]), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")

