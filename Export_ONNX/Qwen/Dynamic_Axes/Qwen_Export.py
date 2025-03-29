import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import gc
import site

path = '/home/DakeQQ/Downloads/Qwen2.5-1.5B-Instruct'         # Set the folder path where the Qwen whole project downloaded.
onnx_model_A = '/home/DakeQQ/Downloads/Qwen_ONNX/Qwen.onnx'   # Assign a path where the exported Qwen model stored.

# Load the model
shutil.copyfile("./modeling_modified/modeling_qwen2.py", site.getsitepackages()[-1] + "/transformers/models/qwen2/modeling_qwen2.py")
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_qwen2.py on line 683, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
attention_mask = torch.tensor([-65504.0], dtype=torch.float32)
input_ids = torch.ones((1, 10), dtype=torch.int32)  # "10" is just a dummy value.
past_key_states = torch.zeros((num_key_value_heads, 0, head_dim), dtype=torch.float16)
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


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


model.model.embed_tokens.weight.requires_grad = False
data = model.model.embed_tokens.weight.data
zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
model.register_buffer("scale", scale)
model.register_buffer("zero_point", zero_point)
model.register_buffer("embed_data", embed_data)
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
input_names = ['input_ids', 'attention_mask']
output_names = ['max_logit_id']
dynamic_axes = {'input_ids': {1: 'ids_len'}}
keys_values = []
for i in range(num_layers):
    key_name = f'in_key_{i}'
    input_names.append(key_name)
    keys_values.append(past_key_states)
    dynamic_axes[key_name] = {1: 'history_len'}
    key_name = f'out_key_{i}'
    output_names.append(key_name)
    dynamic_axes[key_name] = {1: 'history_len_plus_ids_len'}

for i in range(num_layers):
    value_name = f'in_values_{i}'
    input_names.append(value_name)
    keys_values.append(past_values_states)
    dynamic_axes[value_name] = {1: 'history_len'}
    value_name = f'out_values_{i}'
    output_names.append(value_name)
    dynamic_axes[value_name] = {1: 'history_len_plus_ids_len'}


print('Export start ...')
with torch.inference_mode():
    input_feed = [input_ids, attention_mask] + keys_values
    torch.onnx.export(
        model,
        tuple(input_feed),
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
del past_key_states
del past_values_states
gc.collect()
print('\nExport done!\n\nStart running the Qwen by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
query = "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？"
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
    #  prompt = f'<|begin▁of▁sentence|><｜User｜>\n{query}<|end▁of▁sentence|>\n<|begin▁of▁sentence|><｜Assistant｜>\n'
    head = torch.tensor([[151646, 151644, 198]], dtype=torch.int32)
    tail = torch.tensor([[151643, 198, 151646, 151645, 198]], dtype=torch.int32)
    token = tokenizer(query, return_tensors='pt')['input_ids']
    token = torch.cat((head, token, tail), dim=-1)
else:
    prompt = f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
    token = tokenizer(prompt, return_tensors='pt')['input_ids']
input_ids = token.int().numpy()
attention_mask = np.array([-65504.0], dtype=np.float32)
past_key_states_A = np.zeros((num_key_value_heads, 0, head_dim), dtype=np.float16)
past_values_states_A = past_key_states_A
num_decode = 0
print('\n\nTest Question: ' + query + "\nQwen Answering:\n")

output_names = []
input_feed = {
    in_name_A[0].name: input_ids,
    in_name_A[1].name: attention_mask,
}
for i in range(2, len(in_name_A)):
    input_feed[in_name_A[i].name] = past_key_states_A
for i in range(len(out_name_A)):
    output_names.append(out_name_A[i].name)

# Start to run LLM
start_time = time.time()
while num_decode < max_single_chat_length:
    max_ids, *keys_values = ort_session_A.run(
        output_names,
        input_feed
    )
    if max_ids in [151643, 151645]:  # the stop_id in Qwen is "151643" & "151645"
        break
    else:
        input_feed[in_name_A[0].name] = max_ids
        for i in range(2, len(in_name_A)):
            input_feed[in_name_A[i].name] = keys_values[i - 2]
        if num_decode < 1:
            input_feed[in_name_A[1].name] = np.array([0.0], dtype=np.float32)
        num_decode += 1
        print(tokenizer.decode(max_ids[0]), end="", flush=True)
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")

