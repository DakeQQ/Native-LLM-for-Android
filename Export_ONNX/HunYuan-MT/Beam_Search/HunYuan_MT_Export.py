import gc
import time
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from onnxruntime.capi import _pybind_state as C
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════
download_path                  = r'/home/DakeQQ/Downloads/Hy-MT2-1.8B'                       # Set the folder path where the HunYuan-MT2-[1.8B, 7B] project was downloaded.
onnx_model_Embed               = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/LLM_Embed.onnx'       # Output path for the exported ONNX model.
onnx_model_Main                = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/LLM_Main.onnx'
onnx_model_Rotary_Text_Prefill = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Rotary_Text_Prefill.onnx'
onnx_model_Rotary_Text_Decode  = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Rotary_Text_Decode.onnx'
onnx_model_Greedy              = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam          = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam         = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty             = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax              = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/Argmax.onnx'
onnx_model_KV_Slice            = r'/home/DakeQQ/Downloads/Hunyuan_ONNX/KV_Slice.onnx'


# Test input
sentence                       = "May the force be with you"   # Source sentence to translate.
original_language              = "English"                     # Source language label used in the prompt template.  See -> get_language()
target_language                = "Chinese"                     # Target language label used in the prompt template.

# Model Config
DO_EXPORT                      = True                          # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW           = False                         # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN                     = [120020, 127960, 127967]      # HunYuan-MT stop token ids.
MAX_SEQ_LEN                    = 4096                          # Max context length. Can not edit after export.

# KV cache quantization
KV_QUANT_DTYPE                 = "F16"                         # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE            = 32                            # Group size for Q4 and Q8 quantization. Must divide head_dim evenly when grouped.
USE_HADAMARD                   = True                          # Apply enhanced randomized Walsh-Hadamard mixing before grouped quantization.
HADAMARD_RANDOM_SEED           = 9527                          # Seed for the deterministic Rademacher sign pattern used by the Hadamard transform.
USE_CLIP                       = True                          # Clip outliers to mean ± CLIP_SIGMA*std before quantization.
CLIP_SIGMA                     = 3.0                           # Clip threshold in standard deviations when USE_CLIP is enabled.
USE_SHUFFLE                    = True                          # Interleave channels across groups so high-variance channels are spread evenly.
USE_SYM                        = False                         # True: symmetric quantization. False: asymmetric min-max quantization.
USE_FLOAT16_SCALE_BIAS         = True                          # Store quantization scale and bias tensors in float16 when available.

# Decoding strategy
USE_BEAM_SEARCH                = False                         # Use beam search or greedy search.
REPEAT_PENALTY                 = 1.0                           # 0.0 ~ 1.0; no penalty = 1.0.
PENALTY_RANGE                  = 10                            # Recent-token window to apply the repetition penalty.
MAX_BEAM_SIZE                  = 10                            # Max beam size for beam search. Can not edit after export.
TOP_K                          = 3                             # Top-K candidates considered by beam search.
BEAM_SIZE                      = 3                             # Active beam size for beam search. Must be <= MAX_BEAM_SIZE.

# Runtime config
ORT_LOG                        = False                         # Enable ONNX Runtime logging for debugging.
ORT_FP16                       = False                         # Set to True for FP16 ONNX Runtime settings.
ORT_Accelerate_Providers       = []                            # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                    = 0                             # 0 = auto.
DEVICE_ID                      = 0                             # Device ID for GPU or accelerator execution.
OPSET                          = 17                            # ONNX opset version.


if BEAM_SIZE > MAX_BEAM_SIZE:
    raise ValueError(f"BEAM_SIZE ({BEAM_SIZE}) must be <= MAX_BEAM_SIZE ({MAX_BEAM_SIZE}).")


SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA", "Q8", "Q8_CUDA",
    "ROTARY_Q8", "ROTARY_Q8_CUDA", "F16", "F32"
)


# ══════════════════════════════════════════════════════════════════
# Translation Prompt Helpers
# ══════════════════════════════════════════════════════════════════
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


def should_use_chinese_prompt(src_language, dst_language, en_source_language, en_target_language):
    """Choose the Chinese prompt when the task or labels are expressed in Chinese."""
    language_labels = (str(src_language), str(dst_language))
    has_cjk_label = any(any("\u4e00" <= ch <= "\u9fff" for ch in label) for label in language_labels)
    involves_chinese = en_source_language.startswith("Chinese") or en_target_language.startswith("Chinese")
    return has_cjk_label or involves_chinese


def build_translation_prompt(model_path, text, src_language, dst_language):
    """Build the translation prompt while preserving HunYuan-MT prompt variants."""
    if not text or not src_language or not dst_language:
        return None

    en_source_language, _ = get_language(src_language)
    en_target_language, zh_target_language = get_language(dst_language)
    if None in (en_source_language, en_target_language, zh_target_language):
        return None

    is_7b = '7b' in model_path.lower()
    if should_use_chinese_prompt(src_language, dst_language, en_source_language, en_target_language):
        prompt_body = (
            f"将以下文本翻译为{zh_target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
            f"{text}"
        )
        if is_7b:
            return f"<|startoftext|>{prompt_body}<|extra_0|>"
        return f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>{prompt_body}<｜hy_Assistant｜>"

    prompt_body = (
        f"Translate the following text into {en_target_language}. "
        f"Note that you should only output the translated result without any additional explanation:\n\n"
        f"{text}"
    )
    if is_7b:
        return f"<|startoftext|>{prompt_body}<|extra_0|>"

    return f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>{prompt_body}<｜hy_Assistant｜>"


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
    rotary_kv = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes = []

    if KV_QUANT_DTYPE in rotary_kv and head_dim % 2 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")

    if KV_QUANT_DTYPE in quantized_kv:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError(f"KV_QUANT_GROUP_SIZE must be positive, got {KV_QUANT_GROUP_SIZE}.")
        if KV_QUANT_GROUP_SIZE > head_dim:
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim."
            )
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); "
                f"falling back to {KV_QUANT_GROUP_SIZE}."
            )
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(
                f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); "
                "Q8 grouping collapses to per-head quantization."
            )

        if KV_QUANT_DTYPE in q8_kv and KV_QUANT_GROUP_SIZE == head_dim and (USE_HADAMARD or USE_SHUFFLE):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block."
            )
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# ══════════════════════════════════════════════════════════════════
# Decoding Helper Modules
# ══════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: select the token with the highest logit."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers
        # Pre-compute repeat padding tuples for different tensor ranks.
        self._ones_tuple = {d: (1,) * d for d in range(8)}

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        # Compute log-probabilities for the top-k beams.
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        # Replicate KV caches across all beams.
        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *self._ones_tuple[kv.dim() - 1])

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]
        return *self.save_keys_values, save_id, top_beam_prob.transpose(0, 1), top_beam_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        # Compute log-probabilities and accumulate with previous scores.
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        # Select the top beams from all candidates.
        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        # Gather KV caches for surviving beams.
        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return *self.save_keys_values, save_id, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx


class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class KV_SLICE(torch.nn.Module):
    """Apply slice to KV cache tensors."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_layers
                self.save_v_bias = [None] * num_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end = all_inputs[-1]
        for i in range(self.num_layers):
            self.save_key[i] = all_inputs[i][..., slice_start:slice_end]
            self.save_value[i] = all_inputs[i + self.num_layers][..., slice_start:slice_end, :]
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    if self.kv_grouped_6d:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :, :]
                    else:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :]
                elif self.kv_grouped_6d:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    self.save_k_bias[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :, :]
                    self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :, :]
                else:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    self.save_k_bias[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :]
                    self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :]
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation (θ=π/4) to the head_dim axis *before* quantization.
       The rotation spreads outlier energy across dimension pairs, making the
       value distribution more uniform and reducing quantization error.

    2. **Enhanced Hadamard transform** (USE_HADAMARD, Q4 and Q8 modes):
       applies a deterministic randomized Walsh-Hadamard transform within
       each quantization group.

    3. **Channel shuffle** (USE_SHUFFLE, Q4 and Q8 modes): interleaves
       channels across groups so that high-variance channels are evenly
       distributed.

    4. **Residual bias correction** (asymmetric modes): computes the
       mean quantization residual for each block/group and folds it into
       the stored bias.
    """

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, is_q4=False, is_rotary=False, is_q8_cuda=False, use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=2.5, use_shuffle=False):
        super().__init__()
        self.is_rotary = is_rotary
        self.is_q4 = is_q4
        self.is_q8_cuda = is_q8_cuda
        self.use_sym = use_sym
        self.use_hadamard = use_hadamard
        self.use_clip = use_clip
        self.clip_sigma = clip_sigma
        self.use_shuffle = use_shuffle
        self.use_residual_bias_correction = not use_sym
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups

        # ── Quantization range ───────────────────────────────────────
        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX = float(self.SIGNED_QMAX)
            self.ZERO_POINT = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX = 15.0 if is_q4 else 255.0
            self.ZERO_POINT = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        # ── Group parameters (ROTARY_Q4 always grouped; Q8 grouped when hadamard/shuffle enabled) ──
        self.is_grouped = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        # ── Q8_CUDA int32 packing constants ──────────────────────────
        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # ── Rotary transform buffers ─────────────────────────────────
        if is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([
                torch.full((head_dim // 2,), -inv_sqrt2),
                torch.full((head_dim // 2,), inv_sqrt2),
            ])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[:head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # ── Enhanced Hadamard transform buffers ──────────────────────
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            self._hadamard_levels = []
            width = self.hadamard_size
            while width > 1:
                half = width // 2
                self._hadamard_levels.append((width, half))
                width = half

        # ── Clip sigma buffer ────────────────────────────────────────
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # ── Channel shuffle buffers ──────────────────────────────────
        if self.use_shuffle:
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Walsh-Hadamard Helpers
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _next_power_of_two(n):
        """Return the next power of two greater than or equal to `n`."""
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        """Apply a deterministic randomized Walsh-Hadamard transform on the last dim."""
        if not self.use_hadamard:
            return x
        if not inverse:
            x = x * self.hadamard_sign
        if self.hadamard_pad:
            x = F.pad(x, (0, self.hadamard_pad))
        for width, half in self._hadamard_levels:
            x = x.view(*x.shape[:-1], -1, width)
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = x.view(*x.shape[:-2], -1)
        x = x * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            x = x[..., :self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    # ══════════════════════════════════════════════════════════════════
    # Sigma-Based Clipping
    # ══════════════════════════════════════════════════════════════════
    def _clip_to_sigma(self, x, dim):
        """Clip values to mean ± clip_sigma*std per quantization block."""
        mean = x.mean(dim=dim, keepdim=True)
        var = (x - mean).square().mean(dim=dim, keepdim=True)
        std = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    # ══════════════════════════════════════════════════════════════════
    # Rotary Flip Helpers
    # ══════════════════════════════════════════════════════════════════
    def _flip_k(self, k, batch_size):
        """Swap halves along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)."""
        return k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1).flip(-3).view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _flip_v(self, v, batch_size):
        """Swap halves along head_dim (last dim). v: (B, KVH, 1, S, head_dim)."""
        return v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def _flip_q(self, q, batch_size):
        """Swap halves along head_dim (last dim). q: (B, KVH, G, Qlen, head_dim)."""
        return q.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ── Forward rotation (applied during quantization) ──────────────
    def rotate_k(self, k, batch_size):
        """Rotate key pairs along head_dim (dim 3)."""
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        """Rotate value pairs along head_dim (last dim)."""
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    def rotate_q(self, q, batch_size):
        """Forward-rotate query along head_dim for fused rotary attention."""
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    # ── Inverse rotation (fused into attention computation) ─────────
    def inverse_rotate_v(self, v, batch_size):
        """Inverse-rotate dequantized V along head_dim (last dim)."""
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        """Inverse-rotate dequantized K along head_dim (dim 3)."""
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        """Inverse-rotate attention output along head_dim (last dim)."""
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Hadamard Transform Helpers
    # ══════════════════════════════════════════════════════════════════
    def hadamard_k(self, k, batch_size):
        """Apply randomized Walsh-Hadamard mixing within key quantization groups."""
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        """Apply randomized Walsh-Hadamard mixing within value quantization groups."""
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        """Apply the forward randomized Walsh-Hadamard transform to grouped queries."""
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        """Apply the inverse randomized Walsh-Hadamard transform to attention output."""
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Block Quantization
    # ══════════════════════════════════════════════════════════════════
    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        """Finalize asymmetric quantization with optional residual bias correction."""
        if self.use_residual_bias_correction:
            block_residual = x - (x_packed * scale + block_min)
            block_min = block_min + block_residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        """Quantize to signed integers, then encode into the selected storage container."""
        x_quant = torch.round(x / scale).clamp(self.SIGNED_QMIN, self.SIGNED_QMAX).to(torch.int32)
        if self.is_q4:
            return torch.remainder(x_quant, 16).to(torch.uint8)
        if self.is_q8_cuda:
            return torch.remainder(x_quant, 256).to(torch.uint8)
        return x_quant.to(torch.int8)

    @staticmethod
    def _decode_signed_q4_storage(x):
        """Decode two's-complement int4 values stored in uint8 tensors."""
        x = x.to(torch.int16)
        return torch.remainder(x + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8_storage(x):
        """Decode int8 or uint8-backed signed Q8 storage."""
        if x.dtype == torch.int8:
            return x.to(torch.int16)
        x = x.to(torch.int16)
        return torch.remainder(x + 128, 256) - 128

    def _quantize_block(self, x, dim, batch_size=1):
        """Per-block quantization. Symmetric (absmax) or asymmetric (min-max)."""
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim, batch_size)
        if self.use_sym:
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=dim)
            absmax = x.abs().amax(dim=dim, keepdim=True)
            scale = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim, batch_size):
        """Per-group quantization (Q4 or Q8). Symmetric (absmax) or asymmetric (min-max)."""
        if self.use_sym:
            if dim == -2:
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax = x.abs().amax(dim=-2, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax = x.abs().amax(dim=-1, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale

        if dim == -2:
            x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=-2)
            block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
            scale = (block_max - block_min) * self.inv_qmax
            x_packed = torch.round((x - block_min) / scale)
            x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
            x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        else:
            x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=-1)
            block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
            scale = (block_max - block_min) * self.inv_qmax
            x_packed = torch.round((x - block_min) / scale)
            x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
            x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
        return x_packed, scale, block_min

    # ══════════════════════════════════════════════════════════════════
    # CUDA Packing / Unpacking
    # ══════════════════════════════════════════════════════════════════
    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        """Pack 4 uint8 values into a single int32 for CUDA-friendly storage."""
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        """Unpack int32 back into 4 uint8 channels."""
        r3 = x_i32 % self._16777216
        x3 = (x_i32 - r3) // self._16777216 + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return unpacked.reshape(batch_size, num_kv_heads, 1, head_dim, -1)
        return unpacked.reshape(batch_size, num_kv_heads, 1, -1, head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Q4 Packing / Unpacking
    # ══════════════════════════════════════════════════════════════════
    def pack_q4_k(self, x, batch_size):
        """Pack Q4 keys: (B,KVH,1,D,S) -> (B,KVH,1,D//2,S)."""
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        """Pack Q4 values: (B,KVH,1,S,D) -> (B,KVH,1,S,D//2)."""
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        """Unpack Q4 keys: (B,KVH,1,D//2,S) -> (B,KVH,1,D,S)."""
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        """Unpack Q4 values: (B,KVH,1,S,D//2) -> (B,KVH,1,S,D)."""
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ══════════════════════════════════════════════════════════════════
    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        """Quantize and optionally transform KV tensors for cache storage."""
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale

        k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
        v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
        if self.is_q4:
            k_packed = self.pack_q4_k(k_packed, batch_size)
            v_packed = self.pack_q4_v(v_packed, batch_size)
        if self.is_q8_cuda:
            k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
            v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
        return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class LLM_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class ROTARY_MASK_PREFILL(torch.nn.Module):
    """Precompute rotary embeddings and causal mask for the prefill phase."""

    def __init__(self, llm, max_seq_len):
        super().__init__()

        # Causal attention mask: upper triangle -> -128.
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128

        # Precompute rotary embeddings.
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        """Build cosine and sine rotary tables from HunYuan's shared RoPE module."""
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(idx_theta), torch.sin(idx_theta)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):
    """Provide rotary embeddings for a single decode step."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class LLM_MAIN(torch.nn.Module):
    """
    Main transformer module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE)
      - KV cache management with optional quantized storage
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
    """

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super().__init__()
        self.llm = llm

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.head_dim_quarter = head_dim // 4
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads
        self.total_qkv_heads = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes = [num_heads, num_key_value_heads]

        # ── Layer count multipliers (for indexing into flat KV input list) ──
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # ── KV cache dtype flags ─────────────────────────────────────────
        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.kv_q8 = KV_QUANT_DTYPE == "Q8"
        self.kv_q8_cuda = KV_QUANT_DTYPE == "Q8_CUDA"
        self.kv_rotary_q8 = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda = KV_QUANT_DTYPE == "ROTARY_Q8_CUDA"
        self.kv_rotary_q4_cuda = KV_QUANT_DTYPE == "ROTARY_Q4_CUDA"
        self.kv_rotary_cuda = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized = self.kv_quantized or self.kv_rotary
        self.kv_sym = USE_SYM and self.kv_any_quantized
        self.kv_q8_grouped = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_unpack_head_dim = (head_dim // 2) if self.kv_rotary_q4_cuda else head_dim
        self.kv_pack_quarter = (head_dim // 8) if self.kv_rotary_q4_cuda else (head_dim // 4)

        # ── Quantizer & overflow guard ───────────────────────────────────
        self.quantizer = KVQuantizer(
            head_dim=head_dim,
            num_kv_heads=num_key_value_heads,
            num_kv_groups=self.num_key_value_groups,
            is_q4=self.kv_rotary_q4,
            is_rotary=self.kv_rotary,
            is_q8_cuda=self.kv_rotary_cuda or self.kv_q8_cuda,
            use_sym=self.kv_sym,
            use_hadamard=USE_HADAMARD,
            use_clip=USE_CLIP,
            clip_sigma=CLIP_SIGMA,
            use_shuffle=USE_SHUFFLE,
        ).eval()
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        hidden_rms_norm = self.llm.model.layers[0].input_layernorm
        qk_rms_norm = self.llm.model.layers[0].self_attn.query_layernorm
        hidden_rms_norm_eps = float(getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6)))
        qk_rms_norm_eps = float(getattr(qk_rms_norm, "variance_epsilon", getattr(qk_rms_norm, "eps", hidden_rms_norm_eps)))
        self.register_buffer("hidden_rms_norm_eps", torch.tensor(hidden_size * hidden_rms_norm_eps, dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor(self.head_dim * qk_rms_norm_eps, dtype=torch.float32))

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_any_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_layers
                self.save_v_bias = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self._replace_gelu_with_tanh_approximation(self.llm)
        self._fuse_weights(hidden_size)

        # ── Pre-computed per-layer constants (uniform across all layers) ──
        self.o_proj_in_features = self.llm.model.layers[0].self_attn.o_proj.in_features
        self.mlp_split = [self.llm.model.layers[0].mlp.down_proj.in_features] * 2

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP."""
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # HunYuan-MT may tie lm_head to embeddings; detach before in-place fusion.
            if getattr(self.llm.config, "tie_word_embeddings", False):
                if self.llm.lm_head.weight.data_ptr() == self.llm.model.embed_tokens.weight.data_ptr():
                    self.llm.lm_head.weight = torch.nn.Parameter(self.llm.lm_head.weight.detach().clone())
            final_norm_weight = self.llm.model.norm.weight.unsqueeze(0) * norm_factor
            self.llm.lm_head.weight.mul_(final_norm_weight)
            del self.llm.model.norm

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

        attn.q_out_features = int(q_proj.out_features)
        attn.k_out_features = int(k_proj.out_features)
        attn.v_out_features = int(v_proj.out_features)
        attn.qkv_in_features = in_features
        del attn.q_proj, attn.k_proj, attn.v_proj

        combined_scale = scale_factor * norm_factor_qk
        attn.query_layernorm.weight.mul_(combined_scale)
        attn.key_layernorm.weight.mul_(combined_scale)
        q_norm_repeated = attn.query_layernorm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.key_layernorm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.query_layernorm, attn.key_layernorm

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

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Replace GELU with the tanh approximation for ONNX-friendly export."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                LLM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, eps):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)  # Note, not the .mean()

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime."""
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0]

        for i, layer in enumerate(self.llm.model.layers):
            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            # Fused QKV projection & reshape.
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            # QK normalization & rotary embedding.
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
            qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight

            # Split into query and key, reshape query for GQA.
            q, k = torch.split(qk, self.qk_split_sizes, dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)

            if self.kv_f16:
                k = k.half()
                v = v.half()

            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            if self.kv_rotary_q4:
                # ── ROTARY_Q4 ────────────────────────────────────────
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=-3)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                    q_rot = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    attn = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                    attn = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                    v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn = self.quantizer.inverse_rotate_attn(attn, batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-3)
                    v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-3)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i] = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i] = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                    q_rot = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                    attn = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                    attn = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                    v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn = self.quantizer.inverse_rotate_attn(attn, batch_size)

            elif self.kv_rotary:
                # ── ROTARY_Q8 / ROTARY_Q8_CUDA ──────────────────────
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=-2)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_rot = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn = torch.softmax(attn, dim=-1)

                        v_q_g = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot = self.quantizer.rotate_q(q, batch_size)
                        attn_raw = torch.matmul(q_rot, k_signed)
                        attn = attn_raw * k_s + attention_mask
                        attn = torch.softmax(attn, dim=-1)
                        v_scaled = v_signed * v_s
                        attn = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i] = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i] = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                    if self.kv_q8_grouped:
                        q_rot = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                        attn = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn = torch.softmax(attn, dim=-1)

                        v_q_g = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot = self.quantizer.rotate_q(q, batch_size)
                        attn_raw = torch.matmul(q_rot, k.float())
                        q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                        attn_bias = q_bias_factor * k_b + attention_mask
                        attn = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn = torch.softmax(attn, dim=-1)
                        v_scaled = v.float() * v_s
                        bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                        attn = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size) + bias_term

            elif self.kv_quantized:
                # ── Q8 / Q8_CUDA ─────────────────────────────────────
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=-2)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        attn = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn = torch.softmax(attn, dim=-1)

                        v_q_g = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(q, k_signed)
                        attn = attn_raw * k_s + attention_mask
                        attn = torch.softmax(attn, dim=-1)
                        v_scaled = v_signed * v_s
                        attn = torch.matmul(attn, v_scaled)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    k = torch.cat([all_inputs[i], packed_k], dim=-1)
                    v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)

                    self.save_key[i] = k
                    self.save_value[i] = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i] = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i] = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        q_sum_g = q_g.sum(dim=-1, keepdim=True)
                        attn = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn = torch.softmax(attn, dim=-1)

                        v_q_g = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(q, k.float())
                        attn_bias = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                        attn = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn = torch.softmax(attn, dim=-1)
                        v_dequant = torch.addcmul(v_b, v.float(), v_s)
                        attn = torch.matmul(attn, v_dequant)

            else:
                # ── F16 / F32 KV Cache ───────────────────────────────
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v

                if self.kv_f16:
                    k = k.float()
                    v = v.float()

                attn = torch.matmul(q, k) + attention_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)

            # Output projection & residual.
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, self.mlp_split, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = self.llm.lm_head(hidden_states)

        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale, logits
        if self.kv_any_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


# ══════════════════════════════════════════════════════════════════
# ONNX Export
# ══════════════════════════════════════════════════════════════════
if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model = AutoModelForCausalLM.from_pretrained(download_path, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        hidden_size_cfg = getattr(model.config, 'hidden_size', model.model.embed_tokens.embedding_dim)
        head_dim = getattr(model.config, 'head_dim', hidden_size_cfg // num_heads)
        vocab_size = getattr(model.model, 'vocab_size', model.lm_head.out_features)
        hidden_size = model.model.embed_tokens.embedding_dim
        scale_dtype = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

        for note in normalize_kv_quant_settings(head_dim):
            print(f"\n{note}")

        # ══════════════════════════════════════════════════════════════════
        # Build Dummy Tensors for Tracing
        # ══════════════════════════════════════════════════════════════════
        batch_size = BEAM_SIZE
        dummy_ids_len = 10
        dummy_history_len = 0
        ids_len = torch.tensor([dummy_ids_len], dtype=torch.int64)
        history_len = torch.tensor([dummy_history_len], dtype=torch.int64)
        kv_seq_len = ids_len + history_len
        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        logits = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)

        kv_specs = [('key', 4), ('value', 3)]
        _is_rotary = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
        _is_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        _is_quantized = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
        _kv_sym = USE_SYM and (_is_rotary or _is_quantized)
        _q8_grouped = _is_quantized and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        _rotary_q8_grouped = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        _grouped_6d = _is_rotary_q4 or _q8_grouped or _rotary_q8_grouped

        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif _is_quantized:
            if _kv_sym:
                if _q8_grouped:
                    kv_specs.extend([('key_scale', 5), ('value_scale', 3)])
                else:
                    kv_specs.extend([('key_scale', 4), ('value_scale', 3)])
            else:
                if _q8_grouped:
                    kv_specs.extend([('key_scale', 5), ('key_bias', 5), ('value_scale', 3), ('value_bias', 3)])
                else:
                    kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 3), ('value_bias', 3)])
            if KV_QUANT_DTYPE == "Q8_CUDA":
                kv_dtype = torch.int32
            elif _kv_sym:
                kv_dtype = torch.int8
            else:
                kv_dtype = torch.uint8
        elif _is_rotary:
            if _kv_sym:
                if _is_rotary_q4 or _rotary_q8_grouped:
                    kv_specs.extend([('key_scale', 5), ('value_scale', 3)])
                else:
                    kv_specs.extend([('key_scale', 4), ('value_scale', 3)])
            else:
                if _is_rotary_q4 or _rotary_q8_grouped:
                    kv_specs.extend([('key_scale', 5), ('key_bias', 5), ('value_scale', 3), ('value_bias', 3)])
                else:
                    kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 3), ('value_bias', 3)])
            if KV_QUANT_DTYPE in ("ROTARY_Q4_CUDA", "ROTARY_Q8_CUDA"):
                kv_dtype = torch.int32
            elif _kv_sym and not _is_rotary_q4:
                kv_dtype = torch.int8
            else:
                kv_dtype = torch.uint8
        else:
            kv_dtype = torch.float32

        if KV_QUANT_DTYPE == "Q8_CUDA":
            k_head = head_dim // 4
            v_head = head_dim // 4
        elif KV_QUANT_DTYPE == "ROTARY_Q8_CUDA":
            k_head = head_dim // 4
            v_head = head_dim // 4
        elif KV_QUANT_DTYPE == "ROTARY_Q4":
            k_head = head_dim // 2
            v_head = head_dim // 2
        elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
            k_head = head_dim // 8
            v_head = head_dim // 8
        else:
            k_head = head_dim
            v_head = head_dim

        kv_tensors = {
            'key': torch.zeros((batch_size, num_kv_heads, 1, k_head, dummy_history_len), dtype=kv_dtype),
            'value': torch.zeros((batch_size, num_kv_heads, 1, dummy_history_len, v_head), dtype=kv_dtype),
        }
        if KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"):
            if _grouped_6d:
                kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE
                kv_tensors.update({
                    'key_scale': torch.ones((batch_size, num_kv_heads, 1, kv_quant_num_groups, 1, dummy_history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_kv_heads, 1, dummy_history_len, kv_quant_num_groups, 1), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias': torch.ones((batch_size, num_kv_heads, 1, kv_quant_num_groups, 1, dummy_history_len), dtype=scale_dtype),
                        'value_bias': torch.ones((batch_size, num_kv_heads, 1, dummy_history_len, kv_quant_num_groups, 1), dtype=scale_dtype),
                    })
            else:
                kv_tensors.update({
                    'key_scale': torch.ones((batch_size, num_kv_heads, 1, 1, dummy_history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_kv_heads, 1, dummy_history_len, 1), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias': torch.ones((batch_size, num_kv_heads, 1, 1, dummy_history_len), dtype=scale_dtype),
                        'value_bias': torch.ones((batch_size, num_kv_heads, 1, dummy_history_len, 1), dtype=scale_dtype),
                    })

        # ══════════════════════════════════════════════════════════════════
        # Helper: Build KV I/O Names
        # ══════════════════════════════════════════════════════════════════
        def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            """Build flat KV input/output names, tensors, and dynamic axes."""
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                tensor = tensors_dict[name]
                for i in range(num_layers):
                    in_name = f'in_{name}_{i}'
                    out_name = f'out_{name}_{i}'
                    inputs.append(tensor)
                    in_names.append(in_name)
                    out_names.append(out_name)
                    axes[in_name] = {0: batch_axis, dim: seq_axis}
                    axes[out_name] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Embed
        # ══════════════════════════════════════════════════════════════════
        input_ids = torch.ones((1, dummy_ids_len), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={'input_ids': {0: 'batch', 1: 'ids_len'}, 'hidden_states': {0: 'batch', 1: 'ids_len'}},
            opset_version=OPSET,
            dynamo=False,
        )
        del input_ids

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Prefill)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Text_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos': {1: 'ids_len'},
                'rotary_sin': {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Decode)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False,
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Main
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)
        hidden_states = torch.ones((batch_size, dummy_ids_len, hidden_size), dtype=torch.float32)
        rotary_cos = torch.zeros((1, dummy_ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin = rotary_cos
        attention_mask = torch.zeros((1, 1, 1, dummy_ids_len, dummy_ids_len + dummy_history_len), dtype=torch.float32)

        all_inputs = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['logits']
        dynamic_axes = {
            **kv_axes,
            'hidden_states': {0: 'batch', 1: 'ids_len'},
            'logits': {0: 'batch'},
            'rotary_cos': {1: 'ids_len'},
            'rotary_sin': {1: 'ids_len'},
            'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
        }

        model_Main = LLM_MAIN(model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
        del model

        torch.onnx.export(
            model_Main,
            tuple(all_inputs),
            onnx_model_Main,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del model_Main, hidden_states, attention_mask, all_inputs
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Greedy Search
        # ══════════════════════════════════════════════════════════════════
        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)
        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'save_id_out': {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: First Beam Search
        # ══════════════════════════════════════════════════════════════════
        num_layers_beam = num_layers * len(kv_specs)
        kv_tensors_Greedy = {k: v[[0]] for k, v in kv_tensors.items()}
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_Greedy)
        kv_input_only_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}

        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits[[0]], save_id_in, beam_size]),
            onnx_model_First_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'beam_size'],
            output_names=['out_' + name[3:] for name in kv_in_names] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={
                **kv_input_only_axes,
                'logits': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'top_beam_prob': {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx': {0: 'batch'},
                'save_id_out': {0: 'batch', 1: 'history_len'},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Second Beam Search
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size, topK]),
            onnx_model_Second_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
            output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={
                **kv_axes,
                'logits': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'previous_prob': {0: 'batch'},
                'save_id_out': {0: 'batch', 1: 'history_len'},
                'top_beam_prob': {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx': {0: 'batch'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del kv_tensors_Greedy, previous_prob, topK

        # ══════════════════════════════════════════════════════════════════
        # Export: Penalty
        # ══════════════════════════════════════════════════════════════════
        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch'},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del penalty_value, penalty_range

        # ══════════════════════════════════════════════════════════════════
        # Export: Argmax
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_Argmax,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={'logits': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            opset_version=OPSET,
            dynamo=False,
        )
        del logits
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: KV_Slice
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='sliced_len')
        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end = torch.tensor([5], dtype=torch.int64)
        torch.onnx.export(
            KV_SLICE(num_layers, head_dim),
            tuple(kv_ins + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=kv_in_names + ['slice_start', 'slice_end'],
            output_names=kv_out_names,
            dynamic_axes=kv_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del slice_start, slice_end, kv_ins, kv_in_names, kv_out_names, kv_axes, kv_tensors, save_id_in
        gc.collect()

    print('\nExport done!\n\nStart running HunYuan-MT by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# ══════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════
def bind_ort_in_buf(binding, names, values):
    """Bind OrtValue inputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out_buf(binding, names, values):
    """Bind OrtValue outputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def bind_ort_out(binding, names, device):
    """Bind outputs by name, letting ORT allocate on `device`."""
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from a Python list/scalar."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers,
    )


def get_in_names(session):
    """Return all input names for an ONNX Runtime session."""
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    """Return all output names for an ONNX Runtime session."""
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    """Execute an ONNX Runtime session with pre-bound I/O."""
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════
# ORT Session & Runtime Options
# ══════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads      = MAX_THREADS
session_opts.intra_op_num_threads      = MAX_THREADS
session_opts.execution_mode            = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level  = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':     'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else '',
}
for key, value in _session_configs.items():
    session_opts.add_session_config_entry(key, value)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')
disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════
# Execution Provider Configuration
# ══════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False,
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 ** 3),
        'arena_extend_strategy':              'kNextPowerOfTwo',
        'cudnn_conv_algo_search':             'EXHAUSTIVE',
        'sdpa_kernel':                        '2',
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0',
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',
        'device_filter':              'gpu',
        'disable_metacommands':       'false',
        'enable_graph_capture':       'false',
        'enable_graph_serialization': 'false',
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

session_providers = ORT_Accelerate_Providers if ORT_Accelerate_Providers else ['CPUExecutionProvider']
packed_settings   = {
    '_session_opts':       session_opts,
    '_providers':          session_providers,
    '_provider_options':   provider_options,
    '_disabled_optimizers': disabled_optimizers,
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device        = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════
# Load ONNX Sessions
# ══════════════════════════════════════════════════════════════════
ort_session_Embed               = create_session(onnx_model_Embed, **packed_settings)
binding_Embed                   = ort_session_Embed.io_binding()
in_name_Embed                   = get_in_names(ort_session_Embed)[0]
out_name_Embed                  = get_out_names(ort_session_Embed)[0]

ort_session_Rotary_Text_Prefill = create_session(onnx_model_Rotary_Text_Prefill, **packed_settings)
binding_Rotary_Text_Prefill     = ort_session_Rotary_Text_Prefill.io_binding()
in_name_Rotary_Text_Prefill     = get_in_names(ort_session_Rotary_Text_Prefill)
out_name_Rotary_Text_Prefill    = get_out_names(ort_session_Rotary_Text_Prefill)

ort_session_Rotary_Text_Decode  = create_session(onnx_model_Rotary_Text_Decode, **packed_settings)
binding_Rotary_Text_Decode      = ort_session_Rotary_Text_Decode.io_binding()
in_name_Rotary_Text_Decode      = get_in_names(ort_session_Rotary_Text_Decode)[0]
out_name_Rotary_Text_Decode     = get_out_names(ort_session_Rotary_Text_Decode)
out_meta_Rotary_Text_Decode     = ort_session_Rotary_Text_Decode._outputs_meta

ort_session_Main                = create_session(onnx_model_Main, **packed_settings)
binding_Main                    = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════
# Main Model Metadata & Index Offsets
# ══════════════════════════════════════════════════════════════════
in_name_Main                    = get_in_names(ort_session_Main)
out_name_Main                   = get_out_names(ort_session_Main)
in_meta_Main                    = ort_session_Main._inputs_meta

num_keys_values_Main            = len(out_name_Main) - 1
num_keys_values_Main_plus_1     = num_keys_values_Main + 1
num_keys_values_Main_plus_2     = num_keys_values_Main + 2
num_keys_values_Main_plus_3     = num_keys_values_Main + 3

in_name_Main_kv                 = in_name_Main[:num_keys_values_Main]
out_name_Main_kv                = out_name_Main[:num_keys_values_Main]
in_name_Main_others             = in_name_Main[num_keys_values_Main:]
out_name_Main_logits            = out_name_Main[num_keys_values_Main]

kv_dtype_str                    = in_meta_Main[0].type
hidden_dtype_Main               = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32
vocab_size                      = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]


# ══════════════════════════════════════════════════════════════════
# KV Cache Setup
# ══════════════════════════════════════════════════════════════════
if 'uint8' in kv_dtype_str or 'int8' in kv_dtype_str or 'int32' in kv_dtype_str:
    if 'int32' in kv_dtype_str:
        kv_dtype_Main = np.int32
    elif 'uint8' in kv_dtype_str:
        kv_dtype_Main = np.uint8
    else:
        kv_dtype_Main = np.int8

    _is_rotary_rt    = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    _is_quantized_rt = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
    _kv_sym_rt       = USE_SYM and (_is_rotary_rt or _is_quantized_rt)
    _num_types       = 4 if _kv_sym_rt else 6
    num_layers_Main  = num_keys_values_Main // _num_types
    scale_dtype      = np.float16 if 'float16' in in_meta_Main[num_layers_Main * 2].type else np.float32

    if _kv_sym_rt:
        k_scale_shape = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_shape = list(in_meta_Main[num_layers_Main * 3].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases = None
        v_scales = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases = None
    else:
        k_scale_shape = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_idx = num_layers_Main * 4
        v_scale_shape = list(in_meta_Main[v_scale_idx].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_scales = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
else:
    kv_dtype_Main   = np.float16 if 'float16' in kv_dtype_str else np.float32
    num_layers_Main = num_keys_values_Main // 2
    k_scales        = None
    k_biases        = None
    v_scales        = None
    v_biases        = None

past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1], 1, in_meta_Main[0].shape[3], 0),                  kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════
# Decoding Strategy Validation
# ══════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE
if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1
USE_PENALTY = REPEAT_PENALTY != 1.0


# ══════════════════════════════════════════════════════════════════
# Tokenizer & Stop Tokens & Prompt
# ══════════════════════════════════════════════════════════════════
tokenizer      = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)
prompt         = build_translation_prompt(download_path, sentence, original_language, target_language)

if prompt is None:
    print("\nError: The specified translation language is not supported.")
else:
    tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
    num_prefill = tokens.shape[-1]
    if num_prefill >= MAX_SEQ_LEN:
        raise ValueError(f"Prompt length ({num_prefill}) must be smaller than MAX_SEQ_LEN ({MAX_SEQ_LEN}).")

    # ══════════════════════════════════════════════════════════════════
    # Shared OrtValue Buffers
    # ══════════════════════════════════════════════════════════════════
    # --- Input OrtValues ---
    input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
    topK             = create_ort_with_data([TOP_K],       np.int64, device_type, DEVICE_ID)
    beam_size        = create_ort_with_data([BEAM_SIZE],   np.int64, device_type, DEVICE_ID)

    # --- Decode-phase placeholder buffers (reused every step) ---
    attention_mask_buf  = create_ort_with_shape((1, 1, 1, 1, 1),                                      hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_cos_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[0].shape,                   hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_sin_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[1].shape,                   hidden_dtype_Main, device_type, DEVICE_ID)
    hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, in_meta_Main[num_keys_values_Main].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
    save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                         np.int32,          device_type, DEVICE_ID)

    # --- Logits & token-index buffers ---
    prefill_logits_buf = create_ort_with_shape((1, vocab_size),        hidden_dtype_Main, device_type, DEVICE_ID)
    decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    max_idx_buf        = create_ort_with_shape((1, 1),                 np.int32,          device_type, DEVICE_ID)
    save_id            = save_id_buf

    # ══════════════════════════════════════════════════════════════════
    # Decode Head Sessions
    # ══════════════════════════════════════════════════════════════════
    if USE_BEAM_SEARCH:
        print("\nBeam Search does not display immediate decoding results...")

        ort_session_First_Beam      = create_session(onnx_model_First_Beam, **packed_settings)
        binding_First_Beam          = ort_session_First_Beam.io_binding()
        in_name_First_Beam          = get_in_names(ort_session_First_Beam)
        out_name_First_Beam         = get_out_names(ort_session_First_Beam)
        in_name_First_Beam_parts    = in_name_First_Beam[:num_keys_values_Main_plus_1]
        out_name_First_Beam_parts   = out_name_First_Beam[:num_keys_values_Main_plus_1]
        out_name_First_Beam_others  = out_name_First_Beam[num_keys_values_Main_plus_1:]

        ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
        binding_Second_Beam         = ort_session_Second_Beam.io_binding()
        in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
        out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
        in_name_Second_Beam_parts   = in_name_Second_Beam[:num_keys_values_Main_plus_1]
        out_name_Second_Beam_parts  = out_name_Second_Beam[:num_keys_values_Main_plus_1]
        out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Main_plus_1:]

        beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,          device_type, DEVICE_ID)
        beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main, device_type, DEVICE_ID)

        bind_ort_in_buf(binding_First_Beam,  in_name_First_Beam[num_keys_values_Main_plus_1:num_keys_values_Main_plus_3], [save_id_buf, beam_size])
        bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam[num_keys_values_Main_plus_3:],                           [beam_size, topK])
    else:
        ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
        binding_Greedy     = ort_session_Greedy.io_binding()
        in_name_Greedy     = get_in_names(ort_session_Greedy)
        out_name_Greedy    = get_out_names(ort_session_Greedy)
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

        ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
        binding_Argmax     = ort_session_Argmax.io_binding()
        in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
        out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
        save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

    # ══════════════════════════════════════════════════════════════════
    # Penalty Session
    # ══════════════════════════════════════════════════════════════════
    if USE_PENALTY:
        ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
        binding_Penalty     = ort_session_Penalty.io_binding()
        in_name_Penalty     = get_in_names(ort_session_Penalty)
        out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

        penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
        penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
        penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)
        bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])

    # ══════════════════════════════════════════════════════════════════
    # Prefill Phase
    # ══════════════════════════════════════════════════════════════════
    is_prefill_step    = True
    prefill_start_time = time.time()
    prefill_elapsed    = 0.0
    decode_start_time  = None

    # --- Step 1: Embed the input tokens ---
    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, binding_Embed)
    hidden_states = binding_Embed.get_outputs()[0]

    # Pre-bind Embed input for decode phase (will read from max_idx_buf).
    binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

    # --- Step 2: Compute rotary embeddings & causal mask (prefill) ---
    bind_ort_in_buf(binding_Rotary_Text_Prefill, in_name_Rotary_Text_Prefill, [ids_len, init_history_len])
    bind_ort_out(binding_Rotary_Text_Prefill, out_name_Rotary_Text_Prefill, _ort_device_type)
    run(ort_session_Rotary_Text_Prefill, binding_Rotary_Text_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Text_Prefill.get_outputs()

    # --- Step 3: Pre-bind decode rotary outputs (reused every decode step) ---
    binding_Rotary_Text_Decode.bind_ortvalue_input(in_name_Rotary_Text_Decode, kv_seq_len)
    bind_ort_out_buf(binding_Rotary_Text_Decode, out_name_Rotary_Text_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])

    # --- Step 4: Bind Main model inputs — non-KV ---
    bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states, rotary_cos, rotary_sin, attention_mask])

    # --- Step 5: Bind Main model inputs — empty KV cache ---
    index = 0
    for _ in range(num_layers_Main):
        binding_Main.bind_ortvalue_input(in_name_Main[index], past_keys_Main)
        index += 1
    for _ in range(num_layers_Main):
        binding_Main.bind_ortvalue_input(in_name_Main[index], past_values_Main)
        index += 1
    if k_scales is not None:
        if k_biases is not None:
            for ortvalue in (k_scales, k_biases, v_scales, v_biases):
                for _ in range(num_layers_Main):
                    binding_Main.bind_ortvalue_input(in_name_Main[index], ortvalue)
                    index += 1
        else:
            for ortvalue in (k_scales, v_scales):
                for _ in range(num_layers_Main):
                    binding_Main.bind_ortvalue_input(in_name_Main[index], ortvalue)
                    index += 1

    # --- Step 6: Bind Main model outputs ---
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
    binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

    # --- Step 7: Bind penalty inputs/outputs ---
    if USE_PENALTY:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
        binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

    # --- Step 8: Bind decode head inputs/outputs ---
    if USE_BEAM_SEARCH:
        binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_Main], prefill_logits_buf)
    elif USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
        binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
    else:
        binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)
        binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

    print(f"\nTest Query: Translate the following {original_language} sentence into {target_language}: {sentence}\nHunYuan-MT Answering:")

    # ══════════════════════════════════════════════════════════════════
    # Decode Loop
    # ══════════════════════════════════════════════════════════════════
    num_decode     = 0
    generate_limit = MAX_SEQ_LEN - num_prefill

    while num_decode < generate_limit:
        # ── 1. Run Main Model ────────────────────────────────────────────────
        run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        # ── 2. Apply Repetition Penalty (if enabled and enough tokens) ───────
        if USE_PENALTY and num_decode >= PENALTY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            run(ort_session_Penalty, binding_Penalty)

        # ── 3. Token Selection ───────────────────────────────────────────────
        if USE_BEAM_SEARCH:
            # ── 3a. Beam Search ──────────────────────────────────────────────
            if is_prefill_step:
                bind_ort_in_buf(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
                bind_ort_out_buf(binding_First_Beam, out_name_First_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_First_Beam, binding_First_Beam)
                outputs_Beam = binding_First_Beam.get_outputs()
            else:
                bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
                if num_decode < 2:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_2], beam_score_buf)
                bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_Second_Beam, binding_Second_Beam)
                outputs_Beam = binding_Second_Beam.get_outputs()

            save_id = outputs_Beam[num_keys_values_Main]
            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break

            bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_1], save_id)
        else:
            # ── 3b. Greedy / Argmax ─────────────────────────────────────────
            if USE_PENALTY:
                binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
                run(ort_session_Greedy, binding_Greedy)
                save_id = binding_Greedy.get_outputs()[1]
            else:
                run(ort_session_Argmax, binding_Argmax)

            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break

            if USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
            else:
                save_id_numpy[num_decode] = max_logits_idx

            bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)
            print(tokenizer.decode(max_logits_idx), end="", flush=True)

        # ── 4. Re-bind Main KV outputs ───────────────────────────────────────
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

        # ── 5. Transition: prefill -> decode (executes once) ─────────────────
        if is_prefill_step:
            bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states_buf, rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
            binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)
            binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

            if USE_PENALTY:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

            if USE_BEAM_SEARCH:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main], decode_logits_buf)
                binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
            elif USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
            else:
                binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

            is_prefill_step = False
            decode_start_time = time.time()
            prefill_elapsed = decode_start_time - prefill_start_time

        # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
        run(ort_session_Embed, binding_Embed)
        run(ort_session_Rotary_Text_Decode, binding_Rotary_Text_Decode)
        num_decode += 1

    # ══════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════
    decode_end_time = time.time()
    if num_decode < 2:
        prefill_elapsed = 0.0
        decode_elapsed = 0.0
    else:
        decode_elapsed = decode_end_time - decode_start_time
    total_elapsed = decode_end_time - prefill_start_time

    prefill_tokens_per_second  = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
    decode_tokens_per_second   = num_decode / decode_elapsed   if decode_elapsed  > 0 else 0.0
    overall_tokens_per_second  = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

    if USE_PENALTY or USE_BEAM_SEARCH:
        result = tokenizer.decode(save_id.numpy().flat[:num_decode], skip_special_tokens=True)
    else:
        result = tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)

    print(
        f"\n\n{'─' * 56}\n"
        f"  📝 Generated Output\n"
        f"{'─' * 56}\n"
        f"{result}\n"
        f"{'─' * 56}\n\n"
        f"  ⚡ Performance Summary\n"
        f"{'─' * 56}\n"
        f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
        f"  {'─' * 48}\n"
        f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
        f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
        f"  {'─' * 48}\n"
        f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
        f"{'─' * 56}\n"
    )
    
