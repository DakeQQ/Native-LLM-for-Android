import gc
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = r'D:\LLM\Hunyuan-MT-7B'                                      # Set the folder path where the Hunyuan-MT whole project downloaded.
onnx_model_A = r'D:\LLM\Hunyuan_ONNX\Hunyuan_MT.onnx'               # Assign a path where the exported Hunyuan-MT model stored.
STOP_TOKEN = [127960]                                               # The stop_id in Hunyuan-MT is "127960"
MAX_SEQ_LEN = 4096                                                  # The max context length.
sentence = "May the force be with you"                              # The test sentence after the export process.
original_language = "English"                                       # Source language of the text to translate. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.
target_language = "Chinese"                                         # Target language for translation. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.


def get_language(language_input):
    """
    Accepts a language identifier (full Chinese name, abbreviation, or full English name)
    and returns the standardized full English and full Chinese names.

    The function is case-insensitive for English names and abbreviations.

    Args:
        language_input (str): The language identifier to look up.
                               e.g., "中文", "zh", "chinese", "French"

    Returns:
        tuple[str, str] or tuple[None, None]: A tuple containing the
        standard full English name and the full Chinese name.
        Returns (None, None) if the language is not found.
    """

    # 1. CANONICAL DATA STORE
    # The primary source of truth, keyed by the unique language abbreviation.
    # This structure prevents data duplication.
    # Columns are aligned for clear reading.
    LANGUAGE_DATA = {
        # Abbr:     { English Name,              Chinese Name }
        "ar":      {"english_name": "Arabic",                  "chinese_name": "阿拉伯语"},
        "bn":      {"english_name": "Bengali",                 "chinese_name": "孟加拉语"},
        "my":      {"english_name": "Burmese",                 "chinese_name": "缅甸语"},
        "yue":     {"english_name": "Cantonese",               "chinese_name": "粤语"},
        "zh":      {"english_name": "Chinese",                 "chinese_name": "中文"},
        "zh-Hant": {"english_name": "Chinese (Traditional)",   "chinese_name": "繁体中文"},
        "cs":      {"english_name": "Czech",                   "chinese_name": "捷克语"},
        "nl":      {"english_name": "Dutch",                   "chinese_name": "荷兰语"},
        "en":      {"english_name": "English",                 "chinese_name": "英语"},
        "tl":      {"english_name": "Filipino",                "chinese_name": "菲律宾语"},
        "fr":      {"english_name": "French",                  "chinese_name": "法语"},
        "de":      {"english_name": "German",                  "chinese_name": "德语"},
        "gu":      {"english_name": "Gujarati",                "chinese_name": "古吉拉特语"},
        "he":      {"english_name": "Hebrew",                  "chinese_name": "希伯来语"},
        "hi":      {"english_name": "Hindi",                   "chinese_name": "印地语"},
        "id":      {"english_name": "Indonesian",              "chinese_name": "印尼语"},
        "it":      {"english_name": "Italian",                 "chinese_name": "意大利语"},
        "ja":      {"english_name": "Japanese",                "chinese_name": "日语"},
        "kk":      {"english_name": "Kazakh",                  "chinese_name": "哈萨克语"},
        "km":      {"english_name": "Khmer",                   "chinese_name": "高棉语"},
        "ko":      {"english_name": "Korean",                  "chinese_name": "韩语"},
        "ms":      {"english_name": "Malay",                   "chinese_name": "马来语"},
        "mr":      {"english_name": "Marathi",                 "chinese_name": "马拉地语"},
        "mn":      {"english_name": "Mongolian",               "chinese_name": "蒙古语"},
        "fa":      {"english_name": "Persian",                 "chinese_name": "波斯语"},
        "pl":      {"english_name": "Polish",                  "chinese_name": "波兰语"},
        "pt":      {"english_name": "Portuguese",              "chinese_name": "葡萄牙语"},
        "ru":      {"english_name": "Russian",                 "chinese_name": "俄语"},
        "es":      {"english_name": "Spanish",                 "chinese_name": "西班牙语"},
        "ta":      {"english_name": "Tamil",                   "chinese_name": "泰米尔语"},
        "te":      {"english_name": "Telugu",                  "chinese_name": "泰卢固语"},
        "th":      {"english_name": "Thai",                    "chinese_name": "泰语"},
        "bo":      {"english_name": "Tibetan",                 "chinese_name": "藏语"},
        "tr":      {"english_name": "Turkish",                 "chinese_name": "土耳其语"},
        "uk":      {"english_name": "Ukrainian",               "chinese_name": "乌克兰语"},
        "ur":      {"english_name": "Urdu",                    "chinese_name": "乌尔都语"},
        "ug":      {"english_name": "Uyghur",                  "chinese_name": "维吾尔语"},
        "vi":      {"english_name": "Vietnamese",              "chinese_name": "越南语"},
    }

    # 2. ALIAS MAP GENERATION
    # Create a comprehensive lookup map from all possible inputs to the canonical abbreviation.
    # This map is built dynamically from LANGUAGE_DATA to ensure consistency.
    LANGUAGE_ALIAS_MAP = {}
    for abbr, data in LANGUAGE_DATA.items():
        # Map from abbreviation (e.g., "zh")
        LANGUAGE_ALIAS_MAP[abbr.lower()] = abbr
        # Map from English name (e.g., "chinese")
        LANGUAGE_ALIAS_MAP[data["english_name"].lower()] = abbr
        # Map from Chinese name (e.g., "中文")
        LANGUAGE_ALIAS_MAP[data["chinese_name"]] = abbr

    # Add other common aliases.
    EXTRA_ALIASES = {
        "tagalog": "tl",
        "farsi": "fa",
        "myanmar": "my",
        "uighur": "ug",
        "traditional chinese": "zh-Hant",
        "cambodian": "km",
    }
    LANGUAGE_ALIAS_MAP.update(EXTRA_ALIASES)

    # 3. LOOKUP LOGIC
    # Normalize the input to handle whitespace and case variations.
    lookup_key = str(language_input).strip().lower()

    # Find the canonical abbreviation using the alias map.
    canonical_abbr = LANGUAGE_ALIAS_MAP.get(lookup_key)

    if canonical_abbr:
        # Retrieve the definitive language data.
        lang_data = LANGUAGE_DATA.get(canonical_abbr)
        return lang_data["english_name"], lang_data["chinese_name"]
    else:
        # Return None if no match is found.
        return None, None


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


class HUNYUAN(torch.nn.Module):
    def __init__(self, hunyuan, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(HUNYUAN, self).__init__()
        self.hunyuan = hunyuan
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.hunyuan.model.layers._modules[f'{i}'].self_attn.query_layernorm.weight.data *= scale_factor
            self.hunyuan.model.layers._modules[f'{i}'].self_attn.key_layernorm.weight.data *= scale_factor

        data = self.hunyuan.model.embed_tokens.weight.data
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        theta = self.hunyuan.model.rotary_emb.inv_freq
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
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.hunyuan.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = q * rotary_pos_emb_cos + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin
            k = k * rotary_pos_emb_cos + rotate_half(k, self.head_dim_half, -1) * rotary_pos_emb_sin
            q = layer.self_attn.query_layernorm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            k = (layer.self_attn.key_layernorm.weight * (k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(-1, -2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=2)
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
        hidden_states = self.hunyuan.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        return *self.save_key, *self.save_value, torch.argmax(self.hunyuan.lm_head(hidden_states), dim=-1, keepdim=True).int(), kv_seq_len


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    # Build an optimized model
    model = HUNYUAN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

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
print('\nExport done!\n\nStart running the Qwen by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
max_single_chat_length = MAX_SEQ_LEN                  # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

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
en_target_language, zh_target_language = get_language(target_language)
if original_language and target_language:
    if (target_language == 'Chinese') or (original_language == 'Chinese') or ('中文' in target_language) or ('中文' in original_language):
        prompt = f"<|startoftext|>将下面的文本翻译成{zh_target_language}，不要额外解释。\n\n{sentence}<|extra_0|>"
    else:
        prompt = f"<|startoftext|>Translate the following segment into {en_target_language}, without additional explanation.\n\n{sentence}<|extra_0|>"
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
    print(f'\n\nTest Query: Translate the following {original_language} sentence into {target_language}: {sentence}\nHunYuan-MT Answering:\n')

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
        print(tokenizer.decode(max_logit_ids), end="", flush=True)
    print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
else:
    print("\nError: The specified translation language is not supported.")
