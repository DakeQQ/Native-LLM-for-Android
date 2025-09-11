import gc
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = r'D:\LLM\Seed-X-PPO-7B'                                      # Set the folder path where the Seed-X-PRO or Seed-X-Instruct whole project downloaded.
onnx_model_A = r'D:\LLM\Seed_X_ONNX\Seed_X.onnx'                    # Assign a path where the exported Seed-X model stored.
STOP_TOKEN = [2]                                                    # The stop_id in Seed-X is "2"
MAX_SEQ_LEN = 4096                                                  # The max context length.
sentence = "May the force be with you"                              # The test sentence after the export process.
original_language = "English"                                       # Source language of the text to translate. Accepts: English/Chinese (case-insensitive). See get_language() for all supported languages.
target_language = "Chinese"                                         # Target language for translation. Accepts: English/Chinese (case-insensitive). See get_language() for all supported languages.


def get_language(language_name):
    language_data = {
        # Chinese entries in the same corresponding order
        "阿拉伯语":        {"abbr": "ar", "english_name": "Arabic"},
        "中文":           {"abbr": "zh", "english_name": "Chinese"},
        "捷克语":         {"abbr": "cs", "english_name": "Czech"},
        "丹麦语":         {"abbr": "da", "english_name": "Danish"},
        "荷兰语":         {"abbr": "nl", "english_name": "Dutch"},
        "英语":           {"abbr": "en", "english_name": "English"},
        "芬兰语":         {"abbr": "fi", "english_name": "Finnish"},
        "法语":           {"abbr": "fr", "english_name": "French"},
        "德语":           {"abbr": "de", "english_name": "German"},
        "匈牙利语":        {"abbr": "hu", "english_name": "Hungarian"},
        "印度尼西亚语":     {"abbr": "id", "english_name": "Indonesian"},
        "意大利语":        {"abbr": "it", "english_name": "Italian"},
        "日语":           {"abbr": "ja", "english_name": "Japanese"},
        "韩语":           {"abbr": "ko", "english_name": "Korean"},
        "马来语":          {"abbr": "ms", "english_name": "Malay"},
        "挪威语":          {"abbr": "no", "english_name": "Norwegian"},
        "挪威博克马尔语":   {"abbr": "nb", "english_name": "Norwegian Bokmål"},
        "波兰语":          {"abbr": "pl", "english_name": "Polish"},
        "葡萄牙语":        {"abbr": "pt", "english_name": "Portuguese"},
        "罗马尼亚语":      {"abbr": "ro", "english_name": "Romanian"},
        "俄语":           {"abbr": "ru", "english_name": "Russian"},
        "西班牙语":        {"abbr": "es", "english_name": "Spanish"},
        "瑞典语":         {"abbr": "sv", "english_name": "Swedish"},
        "泰语":           {"abbr": "th", "english_name": "Thai"},
        "土耳其语":        {"abbr": "tr", "english_name": "Turkish"},
        "乌克兰语":        {"abbr": "uk", "english_name": "Ukrainian"},
        "越南语":         {"abbr": "vi", "english_name": "Vietnamese"},

        # English entries sorted A-Z
        "arabic":          {"abbr": "ar", "english_name": "Arabic"},
        "chinese":         {"abbr": "zh", "english_name": "Chinese"},
        "croatian":        {"abbr": "cs", "english_name": "Czech"},
        "danish":          {"abbr": "da", "english_name": "Danish"},
        "dutch":           {"abbr": "nl", "english_name": "Dutch"},
        "english":         {"abbr": "en", "english_name": "English"},
        "finnish":         {"abbr": "fi", "english_name": "Finnish"},
        "french":          {"abbr": "fr", "english_name": "French"},
        "german":          {"abbr": "de", "english_name": "German"},
        "hungarian":       {"abbr": "hu", "english_name": "Hungarian"},
        "indonesian":      {"abbr": "id", "english_name": "Indonesian"},
        "italian":         {"abbr": "it", "english_name": "Italian"},
        "japanese":        {"abbr": "ja", "english_name": "Japanese"},
        "korean":          {"abbr": "ko", "english_name": "Korean"},
        "malay":           {"abbr": "ms", "english_name": "Malay"},
        "norwegian":       {"abbr": "no", "english_name": "Norwegian"},
        "norwegian Bokmål": {"abbr": "nb", "english_name": "Norwegian Bokmål"},
        "polish":          {"abbr": "pl", "english_name": "Polish"},
        "portuguese":      {"abbr": "pt", "english_name": "Portuguese"},
        "romanian":        {"abbr": "ro", "english_name": "Romanian"},
        "russian":         {"abbr": "ru", "english_name": "Russian"},
        "spanish":         {"abbr": "es", "english_name": "Spanish"},
        "swedish":         {"abbr": "sv", "english_name": "Swedish"},
        "thai":            {"abbr": "th", "english_name": "Thai"},
        "turkish":         {"abbr": "tr", "english_name": "Turkish"},
        "ukrainian":       {"abbr": "uk", "english_name": "Ukrainian"},
        "vietnamese":      {"abbr": "vi", "english_name": "Vietnamese"}
    }

    # Check if the language name exists in the dictionary
    language_name = language_name.lower()
    if language_name in language_data:
        abbr = f"<{language_data[language_name]['abbr']}>"
        # Return abbreviation and capitalized English name
        return abbr, language_data[language_name]['english_name']
    else:
        return None, None  # Return None for unknown languages


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


class SEED_X(torch.nn.Module):
    def __init__(self, seed_x, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(SEED_X, self).__init__()
        self.seed_x = seed_x
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.seed_x.model.layers._modules[f'{i}'].self_attn.q_proj.weight.data *= scale_factor
            self.seed_x.model.layers._modules[f'{i}'].self_attn.k_proj.weight.data *= scale_factor

        data = self.seed_x.model.embed_tokens.weight.data
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = position_ids * self.seed_x.model.rotary_emb.inv_freq
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
        for i, layer in enumerate(self.seed_x.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).permute(2, 1, 3, 0)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.seed_x.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        logit = self.seed_x.lm_head(hidden_states)
        max_logit_ids = torch.argmax(logit, dim=-1, keepdim=True).int()  # Greedy Search
        return *self.save_key, *self.save_value, max_logit_ids, kv_seq_len


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.model.layers._modules['0'].self_attn.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    # Build an optimized model
    model = SEED_X(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

    # Generate dummies for torch.onnx.export()
    attention_mask = torch.tensor([0], dtype=torch.int8)
    ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
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
        opset_version=17
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
print('\nExport done!\n\nStart running the Seed-X by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
max_single_chat_length = MAX_SEQ_LEN                  # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)  # Use the current folder tokenizer.json and tokenizer_config.json

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # Fatal level = 4, it an adjustable value.
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
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
amount_of_outputs = len(out_name_A)
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs)]

# Pre-process inputs
_, original_language = get_language(original_language)
abbr, target_language = get_language(target_language)
if original_language and target_language:
    prompt = f"Translate the following {original_language} sentence into {target_language}:\n{sentence} {abbr}"
    tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, 'cpu', 0)
    ids_len = tokens.shape[-1]
    max_single_chat_length -= ids_len
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([ids_len], dtype=np.int64), 'cpu', 0)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
    past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, head_dim, 0), dtype=np.float32), 'cpu', 0)
    past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 1, 0, head_dim), dtype=np.float32), 'cpu', 0)
    num_keys_values = num_layers + num_layers
    num_decode = 0
    print(f'\n\nTest Query: Translate the following {original_language} sentence into {target_language}:\n\n{sentence}\n\nSeed-X Answering:\n')

    input_feed_A = {
        in_name_A[-4]: input_ids,
        in_name_A[-3]: history_len,
        in_name_A[-2]: ids_len,
        in_name_A[-1]: attention_mask_1
    }

    for i in range(num_layers):
        input_feed_A[in_name_A[i]] = past_keys_A
    for i in range(num_layers, num_keys_values):
        input_feed_A[in_name_A[i]] = past_values_A

    # Start to run LLM
    start_time = time.time()
    while num_decode < max_single_chat_length:
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_A[-2])[0]
        num_decode += 1
        if max_logit_ids in STOP_TOKEN:
            break
        for i in range(amount_of_outputs):
            input_feed_A[in_name_A[i]] = all_outputs_A[i]
        if num_decode < 2:
            input_feed_A[in_name_A[-1]] = attention_mask_0
            input_feed_A[in_name_A[-2]] = ids_len_1
        print(tokenizer.decode(max_logit_ids, skip_special_tokens=True), end="", flush=True)
    print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
else:

    print("\nError: The specified translation language is not supported.")



