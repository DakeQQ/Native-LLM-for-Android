import gc
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = r'/home/DakeQQ/Downloads/Hy-MT2-1.8B'                                    # Set the folder path where the Hunyuan-MT-2-[1.8B, 7B] whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/Hunyuan_Optimized/Hunyuan_MT.onnx'      # Assign a path where the exported Hunyuan-MT-2 model stored.
STOP_TOKEN = [120020, 127960, 127967]                                           # The stop_id in Hunyuan-MT-2-1.8B is"120020"; 127960 & 127967 for 7B
MAX_SEQ_LEN = 4096                                                              # The max context length.
sentence = "May the force be with you"                                          # The test sentence after the export process.
original_language = "English"                                                   # Source language of the text to translate. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.
target_language = "Chinese"                                                     # Target language for translation. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.
OPSET = 17                                                                      # ONNX opset version.


def get_language(language_input):
    """
    Accepts a language identifier (full Chinese name, abbreviation, or full English name)
    and returns the standardized full English and full Chinese names.

    The function is case-insensitive for English names and abbreviations.

    Args:
        language_input (str): The language identifier to look up. e.g., "中文", "zh", "chinese", "French"

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


class HUNYUAN(torch.nn.Module):
    def __init__(self, hunyuan, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(HUNYUAN, self).__init__()
        self.hunyuan = hunyuan
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads
        self.total_qkv_heads = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes = [num_heads, num_key_value_heads]

        # Sum-based RMS norm epsilon (adjusted: eps * dimension_size)
        hidden_rms_norm_eps = float(getattr(hunyuan.model.layers[0].input_layernorm, 'variance_epsilon',
                                            getattr(hunyuan.model.layers[0].input_layernorm, 'eps', 1e-6)))
        qk_rms_norm_eps = float(getattr(hunyuan.model.layers[0].self_attn.query_layernorm, 'variance_epsilon',
                                        getattr(hunyuan.model.layers[0].self_attn.query_layernorm, 'eps', hidden_rms_norm_eps)))
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_size * hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([head_dim * qk_rms_norm_eps], dtype=torch.float32))

        # Embedding quantization
        data = self.hunyuan.model.embed_tokens.weight.data
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        # Rotary embeddings (5D shape for broadcast with qk tensor)
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = position_ids * self.hunyuan.model.rotary_emb.inv_freq
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).view(1, max_seq_len, 1, 1, head_dim).half()
        self.sin_rotary_pos_emb = torch.cat((-sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).view(1, max_seq_len, 1, 1, head_dim).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers

        # Causal attention mask (5D for GQA broadcast)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        # Replace GELU with tanh approximation for ONNX compatibility
        self._replace_gelu_with_tanh_approximation(self.hunyuan)

        # Fuse weights: absorb norms, merge QKV, merge gate_up
        self._fuse_weights(hidden_size)

        # Store o_proj in_features and mlp split sizes
        self.o_proj_in_features = self.hunyuan.model.layers[0].self_attn.o_proj.in_features
        self.mlp_split = [self.hunyuan.model.layers[0].mlp.down_proj.in_features] * 2

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Replace GELU with the tanh approximation for ONNX-friendly export."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                HUNYUAN._replace_gelu_with_tanh_approximation(child)

    def _fuse_weights(self, hidden_size):
        """Fuse QKV, gate_up projections and absorb layer norms."""
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.hunyuan.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Fuse final norm into lm_head
            if getattr(self.hunyuan.config, "tie_word_embeddings", False):
                if self.hunyuan.lm_head.weight.data_ptr() == self.hunyuan.model.embed_tokens.weight.data_ptr():
                    self.hunyuan.lm_head.weight = torch.nn.Parameter(self.hunyuan.lm_head.weight.detach().clone())
            final_norm_weight = self.hunyuan.model.norm.weight.unsqueeze(0) * norm_factor
            self.hunyuan.lm_head.weight.mul_(final_norm_weight)
            del self.hunyuan.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(proj.bias is not None for proj in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        # Fuse QK norm scale (head_dim^-0.25 * head_dim^0.5 = head_dim^0.25)
        combined_scale = scale_factor * norm_factor_qk
        attn.query_layernorm.weight.mul_(combined_scale)
        attn.key_layernorm.weight.mul_(combined_scale)
        q_norm_repeated = attn.query_layernorm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.key_layernorm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim)
        )
        del attn.query_layernorm, attn.key_layernorm

        # Absorb input layernorm into QKV weight
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight,
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _rms_norm(self, x, eps):
        """Sum-based RMS normalization (eps already scaled by dimension size)."""
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size):
        """Flip-based rotate_half for RoPE (more efficient than split+cat in ONNX)."""
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        input_ids = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        batch_size = input_ids.shape[0]
        kv_seq_len = history_len + ids_len

        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = (self.attention_mask[:, :, :, :ids_len, :kv_seq_len] * all_inputs[-1]).float()

        for i, layer in enumerate(self.hunyuan.model.layers):
            # Self-Attention
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            # Fused QKV projection & reshape
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            # Apply RoPE with flip-based rotate_half
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # QK normalization with fused scale
            qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight

            # Split Q and K, reshape Q for GQA
            q, k = torch.split(qk, self.qk_split_sizes, dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)           # (B, KVH, G, S, D)

            # F16 KV cache
            k = k.half().permute(0, 3, 2, 4, 1)    # (B, KVH, 1, D, S)
            v = v.half().transpose(1, 3)            # (B, KVH, 1, S, D)

            # KV cache concatenation
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v

            # Reshape-based GQA attention (no repeat needed)
            attn = torch.matmul(q, k.float()) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v.float())

            # Output projection & residual
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # Feed-Forward Network
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, self.mlp_split, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # Final projection
        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = self.hunyuan.lm_head(hidden_states)
        max_logit_ids = torch.argmax(logits, dim=-1, keepdim=True).int()   # Greedy Search
        return *self.save_key, *self.save_value, max_logit_ids, kv_seq_len


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    hidden_size = model.model.embed_tokens.embedding_dim

    # Build an optimized model
    model = HUNYUAN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size)

    # Generate dummies for torch.onnx.export()
    attention_mask = torch.tensor([0], dtype=torch.int8)
    ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    history_len = torch.zeros(1, dtype=torch.int64)
    past_keys = torch.zeros((1, num_key_value_heads, 1, head_dim, 0), dtype=torch.float16)
    past_values = torch.zeros((1, num_key_value_heads, 1, 0, head_dim), dtype=torch.float16)

    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {4: 'history_len_plus_ids_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'history_len_plus_ids_len'}
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

    # torch.onnx.export(
    #     model,
    #     tuple(all_inputs),
    #     onnx_model_A,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=dynamic_axes,
    #     do_constant_folding=True,
    #     opset_version=OPSET,
    #     dynamo=False
    # )
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
print('\nExport done!\n\nStart running the HunYuan-MT by ONNXRuntime.\nNow loading . . . it could cost minutes.')

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
device = "cpu"
device_id = 0

in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
amount_of_outputs = len(out_name_A)
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs)]

# Pre-process inputs
en_target_language, zh_target_language = get_language(target_language)
is_7B = '7b' in path.lower()
if original_language and target_language:
    if (target_language == 'Chinese') or (original_language == 'Chinese') or ('中文' in target_language) or ('中文' in original_language):
        if is_7B:
            prompt = f"<|startoftext|>将以下文本翻译为{zh_target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{sentence}<|extra_0|>"
        else:
            prompt = f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>将以下文本翻译为{zh_target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{sentence}<｜hy_Assistant｜>"
    else:
        if is_7B:
            prompt = f"<|startoftext|>Translate the following text into {en_target_language}. Note that you should only output the translated result without any additional explanation:\n\n{sentence}<|extra_0|>"
        else:
            prompt = f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into {en_target_language}. Note that you should only output the translated result without any additional explanation:\n\n{sentence}<｜hy_Assistant｜>"
    tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device, device_id)
    ids_len = tokens.shape[-1]
    max_single_chat_length -= ids_len
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([ids_len], dtype=np.int64), device, device_id)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device, device_id)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device, device_id)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device, device_id)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device, device_id)
    past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, num_key_value_heads, 1, head_dim, 0), dtype=np.float16), device, device_id)
    past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, num_key_value_heads, 1, 0, head_dim), dtype=np.float16), device, device_id)
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
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_A[-2]).flat[0]
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
