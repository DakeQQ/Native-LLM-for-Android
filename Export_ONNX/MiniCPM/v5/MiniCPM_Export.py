"""Export MiniCPM5 ONNX graphs for optimized on-device inference."""

import gc
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# Settings
MODEL_PATH                     = str(Path.home() / "Downloads" / "MiniCPM5-1B")  # HF checkpoint folder.
DO_EXPORT                      = True                                # True rebuilds ONNX; False only runs the existing export.
USE_BATCH                      = False                               # True = dynamic batch axis; False = fixed batch 1 for CUDA-friendly Reshape nodes.
OPSET                          = 20                                  # Keep aligned with Android ORT support.
MAX_SEQ_LEN                    = 4096                                # Export-time rotary/mask limit.

# Weight reorder is exact; use it to make later Q4/Q8 weight-only quantization easier.
REORDER_DOWNPROJ_FOR_QUANT     = True                                # Recommended for MLP down_proj quantization.
REORDER_OPROJ_FOR_QUANT        = False                               # Optional; may shift quantized-KV accuracy.
REORDER_KEY                    = "absmean"                           # Channel score: "absmean" | "L4" | "rms" | "std".

# KV cache storage. F16/F32 are unquantized; Q8/Q4 variants store KV as integer caches.
KV_QUANT_DTYPE                 = "Q8"                                # ROTARY_Q4[_CUDA] | Q8[_CUDA] | ROTARY_Q8[_CUDA] | F16 | F32.
KV_QUANT_GROUP_SIZE            = 128                                 # Quant group along head_dim; auto-clamped to a divisor.
COMPUTE_IN_F32                 = False                               # F16 KV only: True preserves broad CPU support; False uses native f16 attention.

# Quantized-KV accuracy/storage knobs. Ignored by F16/F32 KV.
USE_HADAMARD                   = False                               # Grouped Q4/Q8 only: rotate channels before quantization.
HADAMARD_RANDOM_SEED           = 9527                                # Deterministic Hadamard sign pattern.
USE_CLIP                       = False                               # Clip outliers before KV quantization.
CLIP_SIGMA                     = 3.0                                 # Sigma bound used when USE_CLIP=True.
USE_SHUFFLE                    = False                               # Grouped Q4/Q8 only: spread channels across groups.
USE_SYM                        = True                                # True=signed absmax/no bias; False=min-max with bias.
USE_FLOAT16_SCALE_BIAS         = True                                # Store quant scales/biases as f16 instead of f32.
USE_QDQ_FRIENDLY_ASYM          = False                               # Asym only: enables blocked Q/DQ rewrite, disables residual correction.


SCRIPT_DIR                     = Path(__file__).resolve().parent
RAW_ONNX_DIR                   = SCRIPT_DIR / "MiniCPM_ONNX_Raw"
ONNX_DIR                       = SCRIPT_DIR / "MiniCPM_ONNX"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_SCRIPT               = SCRIPT_DIR / "Inference_MiniCPM_ONNX.py"
MODEL_FILE_NAMES = {
    "metadata":               "LLM_Metadata.onnx",
    "main":                   "LLM_Main.onnx",
    "rotary_prefill":         "LLM_RotaryPrefill.onnx",
    "rotary_decode":          "LLM_RotaryDecode.onnx",
    "greedy":                 "LLM_Greedy.onnx",
    "sampling":               "LLM_TopKTopPSampling.onnx",
    "penalty_greedy":         "LLM_PenaltyGreedy.onnx",
    "penalty":                "LLM_Penalty.onnx",
    "kv_slice":               "LLM_KV_Slice.onnx",
    "kv_split2":              "LLM_KV_Split2.onnx",
    "kv_concat":              "LLM_KV_Concat.onnx",
    "rope_shift":             "LLM_RopeShift.onnx",
    "prefill_greedy":         "LLM_TextPrefillGreedy.onnx",
    "prefill_penalty_greedy": "LLM_TextPrefillPenaltyGreedy.onnx",
    "prefill_sampling":       "LLM_TextPrefillSampling.onnx",
    "decode_greedy":          "LLM_DecodeGreedy.onnx",
    "decode_penalty_greedy":  "LLM_DecodePenaltyGreedy.onnx",
    "decode_sampling":        "LLM_DecodeSampling.onnx",
    "shared_initializers":    "LLM_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"
MERGED_METADATA_KEY_ALIASES = {
    "prefill_greedy": "text_prefill_greedy",
    "prefill_penalty_greedy": "text_prefill_penalty_greedy",
    "prefill_sampling": "text_prefill_sampling",
    "decode_greedy": "text_decode_greedy",
    "decode_penalty_greedy": "text_decode_penalty_greedy",
    "decode_sampling": "text_decode_sampling",
}
MODEL_FILE_NAME_METADATA = {
    f"model_file_name_{key}": value
    for key, value in MODEL_FILE_NAMES.items()
}
MODEL_FILE_NAME_METADATA.update({
    f"model_file_name_{alias}": MODEL_FILE_NAMES[key]
    for key, alias in MERGED_METADATA_KEY_ALIASES.items()
})

onnx_model_Metadata            = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["metadata"])
onnx_model_Main                = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["main"])
onnx_model_Rotary_Text_Prefill = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["rotary_prefill"])
onnx_model_Rotary_Text_Decode  = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["rotary_decode"])
onnx_model_Greedy              = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["greedy"])
onnx_model_TopKTopP_Sampling   = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["sampling"])
onnx_model_Penalty_Greedy      = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["penalty_greedy"])
onnx_model_Penalty             = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["penalty"])
onnx_model_KV_Slice            = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["kv_slice"])
onnx_model_KV_Split2           = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["kv_split2"])
onnx_model_KV_Concat           = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["kv_concat"])
onnx_model_Rope_Shift          = str(RAW_ONNX_DIR / MODEL_FILE_NAMES["rope_shift"])

# Shared_Merged.py converts the split export into merged strategy graphs plus one shared weight blob.


SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
    "F16",
    "F32"
)


def normalize_kv_quant_settings(head_dim):
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
                f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}."
            )
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(
                f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization."
            )

        if KV_QUANT_DTYPE in q8_kv and KV_QUANT_GROUP_SIZE == head_dim and (USE_HADAMARD or USE_SHUFFLE):
            notes.append(
                "[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block."
            )
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


class PENALTY_GREEDY_SEARCH(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id        = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class APPLY_PENALTY(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class GREEDY_SEARCH(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    NEG_INF    = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
        self.register_buffer("neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False)
        self.register_buffer("gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False)
        self.register_buffer("gumbel_max", torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32), persistent=False)

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        inv_penalty   = torch.reciprocal(repetition_penalty)
        prev_logits   = torch.gather(logits, 1, previous_ids)
        prev_scores   = torch.where(prev_logits < 0.0, prev_logits * repetition_penalty, prev_logits * inv_penalty)
        scores        = torch.scatter(logits, 1, previous_ids, prev_scores)
        scores        = scores * torch.reciprocal(temperature)
        
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)

        sorted_probs  = torch.softmax(sorted_scores, dim=-1)
        sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep_topp     = (sorted_cumsum - sorted_probs) <= top_p
        sorted_scores = torch.where(keep_topp, sorted_scores, self.neg_inf)

        noise  = torch.clamp(torch.rand_like(sorted_scores), self.gumbel_min, self.gumbel_max)
        gumbel = -torch.log(-torch.log(noise))
        winner = torch.argmax(sorted_scores + gumbel, dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        save_id    = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_id


class METADATA_CARRIER(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, marker):
        return marker


class WINDOW_SPLIT_SIZES(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ref, start, end, dim):
        s, e = int(start), int(end)
        return torch.tensor([s, e - s, ref.shape[dim] - e], dtype=torch.int64)

    @staticmethod
    def symbolic(g, ref, start, end, dim):
        shape    = g.op("Shape", ref)
        dim_size = g.op("Gather", shape, g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)), axis_i=0)
        window   = g.op("Sub", end, start)
        tail     = g.op("Sub", dim_size, end)
        return g.op("Concat", start, window, tail, axis_i=0)


class SLICE_KEEP_MIDDLE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, sizes, dim):
        start = int(sizes[0])
        end   = start + int(sizes[1])
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(start, end)
        return x[tuple(idx)].clone()

    @staticmethod
    def symbolic(g, x, sizes, dim):
        return g.op("Split", x, sizes, axis_i=dim, outputs=3)[1]


def window_split_sizes(ref, start, end, dim):
    if dim < 0:
        dim += ref.dim()
    return WINDOW_SPLIT_SIZES.apply(ref, start, end, dim)


def slice_keep_middle(x, sizes, dim):
    if dim < 0:
        dim += x.dim()
    return SLICE_KEEP_MIDDLE.apply(x, sizes, dim)


class KV_SLICE(torch.nn.Module):

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.save_key     = [None] * num_layers
        self.save_value   = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end   = all_inputs[-1]
        sizes = window_split_sizes(all_inputs[0], slice_start, slice_end, -1)
        for i in range(self.num_layers):
            self.save_key[i]   = slice_keep_middle(all_inputs[i],                   sizes, -1)
            self.save_value[i] = slice_keep_middle(all_inputs[i + self.num_layers], sizes, -2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    if self.kv_grouped_6d:
                        self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -3)
                    else:
                        self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -2)
                elif self.kv_grouped_6d:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.save_k_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_4], sizes, -3)
                    self.save_v_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_5], sizes, -3)
                else:
                    self.save_k_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.save_k_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.save_v_scale[i] = slice_keep_middle(all_inputs[i + self.num_layers_4], sizes, -2)
                    self.save_v_bias[i]  = slice_keep_middle(all_inputs[i + self.num_layers_5], sizes, -2)
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


class SPLIT_POINT_SIZES(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ref, split_at, dim):
        sp = int(split_at)
        return torch.tensor([sp, ref.shape[dim] - sp], dtype=torch.int64)

    @staticmethod
    def symbolic(g, ref, split_at, dim):
        shape    = g.op("Shape", ref)
        dim_size = g.op("Gather", shape, g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)), axis_i=0)
        rest     = g.op("Sub", dim_size, split_at)
        return g.op("Concat", split_at, rest, axis_i=0)


class SPLIT_PREFIX_SUFFIX(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, sizes, dim):
        sp = int(sizes[0])
        idx_prefix = [slice(None)] * x.dim()
        idx_suffix = [slice(None)] * x.dim()
        idx_prefix[dim] = slice(None, sp)
        idx_suffix[dim] = slice(sp, None)
        return x[tuple(idx_prefix)].clone(), x[tuple(idx_suffix)].clone()

    @staticmethod
    def symbolic(g, x, sizes, dim):
        return g.op("Split", x, sizes, axis_i=dim, outputs=2)


def split_point_sizes(ref, split_at, dim):
    if dim < 0:
        dim += ref.dim()
    return SPLIT_POINT_SIZES.apply(ref, split_at, dim)


def split_prefix_suffix(x, sizes, dim):
    if dim < 0:
        dim += x.dim()
    return SPLIT_PREFIX_SUFFIX.apply(x, sizes, dim)


class KV_SPLIT2(torch.nn.Module):

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.prefix_key   = [None] * num_layers
        self.prefix_value = [None] * num_layers
        self.window_key   = [None] * num_layers
        self.window_value = [None] * num_layers
        if self.kv_quantized:
            self.prefix_k_scale = [None] * num_layers
            self.prefix_v_scale = [None] * num_layers
            self.window_k_scale = [None] * num_layers
            self.window_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.prefix_k_bias = [None] * num_layers
                self.prefix_v_bias = [None] * num_layers
                self.window_k_bias = [None] * num_layers
                self.window_v_bias = [None] * num_layers

    def forward(self, *all_inputs):
        split_at = all_inputs[-1]
        sizes = split_point_sizes(all_inputs[0], split_at, -1)
        for i in range(self.num_layers):
            self.prefix_key[i],   self.window_key[i]   = split_prefix_suffix(all_inputs[i],                   sizes, -1)
            self.prefix_value[i], self.window_value[i] = split_prefix_suffix(all_inputs[i + self.num_layers], sizes, -2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    if self.kv_grouped_6d:
                        self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -3)
                    else:
                        self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -2)
                elif self.kv_grouped_6d:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.prefix_k_bias[i],  self.window_k_bias[i]  = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_4], sizes, -3)
                    self.prefix_v_bias[i],  self.window_v_bias[i]  = split_prefix_suffix(all_inputs[i + self.num_layers_5], sizes, -3)
                else:
                    self.prefix_k_scale[i], self.window_k_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_2], sizes, -1)
                    self.prefix_k_bias[i],  self.window_k_bias[i]  = split_prefix_suffix(all_inputs[i + self.num_layers_3], sizes, -1)
                    self.prefix_v_scale[i], self.window_v_scale[i] = split_prefix_suffix(all_inputs[i + self.num_layers_4], sizes, -2)
                    self.prefix_v_bias[i],  self.window_v_bias[i]  = split_prefix_suffix(all_inputs[i + self.num_layers_5], sizes, -2)
        if self.kv_sym:
            return (
                *self.prefix_key, *self.prefix_value, *self.prefix_k_scale, *self.prefix_v_scale,
                *self.window_key, *self.window_value, *self.window_k_scale, *self.window_v_scale
            )
        if self.kv_quantized:
            return (
                *self.prefix_key, *self.prefix_value, *self.prefix_k_scale, *self.prefix_k_bias, *self.prefix_v_scale, *self.prefix_v_bias,
                *self.window_key, *self.window_value, *self.window_k_scale, *self.window_k_bias, *self.window_v_scale, *self.window_v_bias
            )
        return *self.prefix_key, *self.prefix_value, *self.window_key, *self.window_value


class KV_CONCAT(torch.nn.Module):

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_q8_grouped = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym        = USE_SYM and self.kv_quantized
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.v_axis       = -3 if self.kv_grouped_6d else -2
        self.save_key     = [None] * num_layers
        self.save_value   = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

    def _concat(self, prefix, suffix, dim):
        return torch.cat([prefix, suffix], dim=dim)

    def forward(self, *all_inputs):
        n = len(all_inputs) // 2
        a = all_inputs[:n]
        b = all_inputs[n:]
        for i in range(self.num_layers):
            self.save_key[i]   = self._concat(a[i],                   b[i],                   dim=-1)
            self.save_value[i] = self._concat(a[i + self.num_layers], b[i + self.num_layers], dim=-2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = self._concat(a[i + self.num_layers_2], b[i + self.num_layers_2], dim=-1)
                    self.save_v_scale[i] = self._concat(a[i + self.num_layers_3], b[i + self.num_layers_3], dim=self.v_axis)
                else:
                    self.save_k_scale[i] = self._concat(a[i + self.num_layers_2], b[i + self.num_layers_2], dim=-1)
                    self.save_k_bias[i]  = self._concat(a[i + self.num_layers_3], b[i + self.num_layers_3], dim=-1)
                    self.save_v_scale[i] = self._concat(a[i + self.num_layers_4], b[i + self.num_layers_4], dim=self.v_axis)
                    self.save_v_bias[i]  = self._concat(a[i + self.num_layers_5], b[i + self.num_layers_5], dim=self.v_axis)
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


class ONNX_STATIC_RESHAPE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, shape):
        eager_shape = tuple(
            x.shape[index] if dim == 0 else dim
            for index, dim in enumerate(shape)
        )
        return x.reshape(eager_shape)

    @staticmethod
    def symbolic(g, x, shape):
        shape_const = g.op(
            "Constant", value_t=torch.tensor(shape, dtype=torch.int64)
        )
        return g.op("Reshape", x, shape_const)


def onnx_static_reshape(x, shape):
    return ONNX_STATIC_RESHAPE.apply(x, shape)


def onnx_reshape_batch(x, shape):
    batch_dim = 0 if USE_BATCH else 1
    return onnx_static_reshape(x, (batch_dim,) + tuple(shape))


def add_batch_dynamic_axes(dynamic_axes, *tensor_names, axis_name='batch'):
    if USE_BATCH:
        for tensor_name in tensor_names:
            dynamic_axes.setdefault(tensor_name, {})[0] = axis_name
    return dynamic_axes


class ROPE_SHIFT(torch.nn.Module):

    def __init__(self, num_layers, head_dim, num_kv_heads, inv_freq, max_seq_len):
        super().__init__()
        self.num_layers    = num_layers
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2
        self.num_kv_heads  = num_kv_heads
        self.compute_in_f32 = COMPUTE_IN_F32
        inv_freq      = inv_freq.detach().float().reshape(-1)
        inv_freq_full = torch.cat([inv_freq, inv_freq], dim=0).view(1, 1, 1, head_dim, 1)
        half_sign     = torch.cat([torch.ones(self.head_dim_half), -torch.ones(self.head_dim_half)], dim=0).view(1, 1, 1, head_dim, 1)
        shifts        = torch.arange(max_seq_len + 1, dtype=torch.float32).view(max_seq_len + 1, 1, 1, 1, 1)
        angle         = shifts * inv_freq_full
        angle         = angle - 6.283185307179586 * torch.round(angle * (1.0 / 6.283185307179586))
        self.register_buffer("cos_shift", torch.cos(angle).half(), persistent=False)
        self.register_buffer("sin_shift", (torch.sin(angle) * half_sign).half(), persistent=False)

    def _flip_k(self, k):
        k = onnx_reshape_batch(k, (self.num_kv_heads, 1, 2, self.head_dim_half, -1))
        return onnx_reshape_batch(k.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1))

    def forward(self, *all_inputs):
        shift     = all_inputs[-1].reshape(-1)  # Keep [1]; scalar Slice exports poorly.
        kv_dtype  = all_inputs[0].dtype         # Rotate in the cache dtype; cast cos/sin to match k.
        force_f32 = self.compute_in_f32 and kv_dtype != torch.float32  # F16 cache but F32 math requested: upcast K, rotate, cast back.
        cos_tab   = self.cos_shift.index_select(0, shift)
        sin_tab   = self.sin_shift.index_select(0, shift)
        if kv_dtype == torch.float32 or force_f32:
            cos_tab = cos_tab.float()
            sin_tab = sin_tab.float()

        outputs = []
        for i in range(self.num_layers):
            k       = all_inputs[i].float() if force_f32 else all_inputs[i]
            k_shift = k * cos_tab + self._flip_k(k) * sin_tab
            if force_f32:
                k_shift = k_shift.to(kv_dtype)
            outputs.append(k_shift)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):

    def __init__(self, num_layers, head_dim, num_kv_heads, inv_freq, max_seq_len, quantizer, is_asym):
        super().__init__()
        self.num_layers    = num_layers
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2
        self.num_kv_heads  = num_kv_heads
        self.quantizer     = quantizer
        self.is_asym       = is_asym
        inv_freq      = inv_freq.detach().float().reshape(-1)
        inv_freq_full = torch.cat([inv_freq, inv_freq], dim=0).view(1, 1, 1, head_dim, 1)
        half_sign     = torch.cat([torch.ones(self.head_dim_half), -torch.ones(self.head_dim_half)], dim=0).view(1, 1, 1, head_dim, 1)
        shifts        = torch.arange(max_seq_len + 1, dtype=torch.float32).view(max_seq_len + 1, 1, 1, 1, 1)
        angle         = shifts * inv_freq_full
        angle         = angle - 6.283185307179586 * torch.round(angle * (1.0 / 6.283185307179586))
        self.register_buffer("cos_shift", torch.cos(angle).half(), persistent=False)
        self.register_buffer("sin_shift", (torch.sin(angle) * half_sign).half(), persistent=False)

    def _flip_k(self, k):
        k = onnx_reshape_batch(k, (self.num_kv_heads, 1, 2, self.head_dim_half, -1))
        return onnx_reshape_batch(k.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1))

    def forward(self, *all_inputs):
        shift     = all_inputs[-1].reshape(-1)  # Keep [1]; scalar Slice exports poorly.
        cos_tab   = self.cos_shift.index_select(0, shift).float()
        sin_tab   = self.sin_shift.index_select(0, shift).float()

        layers    = self.num_layers
        keys_in   = all_inputs[0:layers]
        scales_in = all_inputs[layers:2 * layers]
        biases_in = all_inputs[2 * layers:3 * layers] if self.is_asym else None

        out_keys, out_scales, out_biases = [], [], []
        for i in range(layers):
            k_packed   = keys_in[i]
            k_bias     = biases_in[i] if self.is_asym else None
            k_raw      = self.quantizer.dequantize_key(k_packed, scales_in[i], k_bias)
            k_shift    = k_raw * cos_tab + self._flip_k(k_raw) * sin_tab
            new_packed, new_scale, new_bias = self.quantizer.quantize_key(k_shift)
            out_keys.append(new_packed)
            out_scales.append(new_scale)
            if self.is_asym:
                out_biases.append(new_bias)
        if self.is_asym:
            return (*out_keys, *out_scales, *out_biases)
        return (*out_keys, *out_scales)


class KVQuantizer(torch.nn.Module):

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, is_q4=False, is_rotary=False, is_q8_cuda=False, use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=2.5, use_shuffle=False):
        super().__init__()
        self.is_rotary     = is_rotary
        self.is_q4         = is_q4
        self.is_q8_cuda    = is_q8_cuda
        self.use_sym       = use_sym
        self.use_hadamard  = use_hadamard
        self.use_clip      = use_clip
        self.clip_sigma    = clip_sigma
        self.use_shuffle   = use_shuffle
        self.use_residual_bias_correction = (not use_sym) and not USE_QDQ_FRIENDLY_ASYM
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads  = num_kv_heads
        self.num_kv_groups = num_kv_groups

        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX        = float(self.SIGNED_QMAX)
            self.ZERO_POINT  = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX        = 15.0 if is_q4 else 255.0
            self.ZERO_POINT  = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        if is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,),  inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[:head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

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
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        if self.use_shuffle:
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        if not self.use_hadamard:
            return x

        if not inverse:
            x = x * self.hadamard_sign

        if self.hadamard_pad:
            x = F.pad(x, (0, self.hadamard_pad))

        leading_zeros = (0,) * (x.dim() - 1)
        for width, half in self._hadamard_levels:
            x = onnx_static_reshape(x, leading_zeros + (-1, width))
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = onnx_static_reshape(x, leading_zeros + (-1,))

        x = x * self.hadamard_inv_sqrt

        if self.hadamard_pad:
            x = x[..., :self.kv_quant_group_size]

        if inverse:
            x = x * self.hadamard_sign

        return x

    def _clip_to_sigma(self, x, dim):
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def _flip_k(self, k):
        k = onnx_reshape_batch(k, (self.num_kv_heads, 1, 2, self.head_dim_half, -1))
        return onnx_reshape_batch(k.flip(-3), (self.num_kv_heads, 1, self.head_dim, -1))

    def _flip_v(self, v):
        v = onnx_reshape_batch(v, (self.num_kv_heads, 1, -1, 2, self.head_dim_half))
        return onnx_reshape_batch(v.flip(-2), (self.num_kv_heads, 1, -1, self.head_dim))

    def _flip_q(self, q):
        q = onnx_reshape_batch(q, (self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half))
        return onnx_reshape_batch(q.flip(-2), (self.num_kv_heads, self.num_kv_groups, -1, self.head_dim))

    def rotate_k(self, k):
        return k * self.rot_cos + self._flip_k(k) * self.rot_sin_k

    def rotate_v(self, v):
        return v * self.rot_cos + self._flip_v(v) * self.rot_sin_v

    def rotate_q(self, q):
        return q * self.rot_cos + self._flip_q(q) * self.rot_sin_v

    def inverse_rotate_k(self, k):
        return k * self.rot_cos - self._flip_k(k) * self.rot_sin_k

    def inverse_rotate_attn(self, x):
        return x * self.rot_cos - self._flip_q(x) * self.rot_sin_v

    def hadamard_k(self, k):
        k = onnx_reshape_batch(k, (self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1))
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return onnx_reshape_batch(k, (self.num_kv_heads, 1, self.head_dim, -1))

    def hadamard_v(self, v):
        v = onnx_reshape_batch(v, (self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size))
        v = self._apply_hadamard_last_dim(v)
        return onnx_reshape_batch(v, (self.num_kv_heads, 1, -1, self.head_dim))

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x):
        x = onnx_reshape_batch(x, (self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size))
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return onnx_reshape_batch(x, (self.num_kv_heads, self.num_kv_groups, -1, self.head_dim))

    def inverse_hadamard_k(self, k):
        k = onnx_reshape_batch(k, (self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1))
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2), inverse=True).transpose(-1, -2)
        return onnx_reshape_batch(k, (self.num_kv_heads, 1, self.head_dim, -1))

    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        if self.use_residual_bias_correction:
            block_residual = x - (x_packed * scale + block_min)
            block_min = block_min + block_residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale     = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        x_quant = torch.round(x / scale).clamp(self.SIGNED_QMIN, self.SIGNED_QMAX).to(torch.int32)
        if self.is_q4:
            return torch.remainder(x_quant, 16).to(torch.uint8)
        if self.is_q8_cuda:
            return torch.remainder(x_quant, 256).to(torch.uint8)
        return x_quant.to(torch.int8)

    @staticmethod
    def _decode_signed_q4_storage(x):
        x = x.to(torch.int16)
        return torch.remainder(x + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8_storage(x):
        if x.dtype == torch.int8:
            return x.to(torch.int16)
        x = x.to(torch.int16)
        return torch.remainder(x + 128, 256) - 128

    def _quantize_block(self, x, dim):
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim)
        if self.use_sym:
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=dim)
            absmax = x.abs().amax(dim=dim, keepdim=True)
            scale  = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale        = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed     = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim):
        if self.use_sym:
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = onnx_reshape_batch(x, (self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1))
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = onnx_reshape_batch(x_packed, (self.num_kv_heads, 1, self.head_dim, -1))
            else:          # values: (B, KVH, 1, S, D)
                x = onnx_reshape_batch(x, (self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size))
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax   = x.abs().amax(dim=-1, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = onnx_reshape_batch(x_packed, (self.num_kv_heads, 1, -1, self.head_dim))
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = onnx_reshape_batch(x, (self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1))
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = onnx_reshape_batch(x_packed, (self.num_kv_heads, 1, self.head_dim, -1))
            else:          # values: (B, KVH, 1, S, D)
                x = onnx_reshape_batch(x, (self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size))
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = onnx_reshape_batch(x_packed, (self.num_kv_heads, 1, -1, self.head_dim))
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = onnx_reshape_batch(x_i32, (num_kv_heads, 1, head_dim_quarter, 4, -1))
        else:
            x_i32 = onnx_reshape_batch(x_i32, (num_kv_heads, 1, -1, head_dim_quarter, 4))
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, x_i32, dim, num_kv_heads, head_dim):
        r3 = x_i32 % self._16777216
        x3 = (x_i32 - r3) // self._16777216 + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return onnx_reshape_batch(unpacked, (num_kv_heads, 1, head_dim, -1))
        return onnx_reshape_batch(unpacked, (num_kv_heads, 1, -1, head_dim))

    def pack_q4_k(self, x):
        x = onnx_reshape_batch(x, (self.num_kv_heads, 1, self.head_dim_half, 2, -1))
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x):
        x = onnx_reshape_batch(x, (self.num_kv_heads, 1, -1, self.head_dim_half, 2))
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x):
        low  = x % 16
        high = x // 16
        return onnx_reshape_batch(torch.stack([low, high], dim=-2), (self.num_kv_heads, 1, self.head_dim, -1))

    def unpack_q4_v(self, x):
        low  = x % 16
        high = x // 16
        return onnx_reshape_batch(torch.stack([low, high], dim=-1), (self.num_kv_heads, 1, -1, self.head_dim))

    def quantize_key(self, keys):
        if self.is_rotary:
            keys = self.rotate_k(keys)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys)
        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys, dim=-2)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed)
            return k_packed, k_scale, None
        k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2)
        if self.is_q4:
            k_packed = self.pack_q4_k(k_packed)
        return k_packed, k_scale, k_bias

    def dequantize_key(self, packed_k, k_scale, k_bias):
        if USE_FLOAT16_SCALE_BIAS:
            k_scale = k_scale.float()
            if k_bias is not None:
                k_bias = k_bias.float()
        if self.is_q4:
            k_int = self.unpack_q4_k(packed_k)
            if self.use_sym:
                k_int = self._decode_signed_q4_storage(k_int)
        else:
            k_int = self._decode_signed_q8_storage(packed_k) if self.use_sym else packed_k
        k_float = k_int.float()
        if self.is_grouped:
            k_g  = onnx_reshape_batch(k_float, (self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1))
            keys = (k_g * k_scale) if self.use_sym else (k_g * k_scale + k_bias)
            keys = onnx_reshape_batch(keys, (self.num_kv_heads, 1, self.head_dim, -1))
        else:
            keys = (k_float * k_scale) if self.use_sym else (k_float * k_scale + k_bias)
        if self.use_hadamard:
            keys = self.inverse_hadamard_k(keys)
        if self.use_shuffle:
            keys = keys.index_select(3, self.unshuffle_idx)
        if self.is_rotary:
            keys = self.inverse_rotate_k(keys)
        return keys

    def forward(self, keys, values, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            keys   = self.rotate_k(keys)
            values = self.rotate_v(values)

        if self.use_shuffle:
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys   = self.hadamard_k(keys)
            values = self.hadamard_v(values)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys,   dim=-2)
            v_packed, v_scale = self._quantize_block(values, dim=-1)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed)
                v_packed = self.pack_q4_v(v_packed)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed)
                v_packed = self.pack_q4_v(v_packed)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class ROTARY_MASK_PREFILL(torch.nn.Module):

    def __init__(self, llm, max_seq_len):
        super().__init__()

        value_dtype = torch.float32 if max_seq_len > 65504 else torch.float16
        self.register_buffer("mask_row_pos", torch.arange(max_seq_len, dtype=torch.int32).view(max_seq_len, 1), persistent=False)
        self.register_buffer("mask_col_pos", torch.arange(max_seq_len, dtype=torch.int32).view(1, max_seq_len), persistent=False)
        self.register_buffer("mask_zero", torch.tensor(0, dtype=value_dtype), persistent=False)
        self.register_buffer("mask_neg", torch.tensor(-65504, dtype=value_dtype), persistent=False)

        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = llm.model.rotary_emb.inv_freq
        idx_theta    = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(idx_theta), torch.sin(idx_theta)

    def forward(self, ids_len, history_len, cache_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        mask_len       = cache_len + ids_len
        row_limit      = self.mask_row_pos[:ids_len] + cache_len.to(self.mask_row_pos.dtype)
        col_pos        = self.mask_col_pos[:, :mask_len]
        attn_bias      = torch.where(col_pos <= row_limit, self.mask_zero, self.mask_neg).view(1, 1, 1, ids_len, mask_len)
        attention_mask = attn_bias.float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):

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


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance   = x.float().pow(2).mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x, scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


class LLM_MAIN(torch.nn.Module):

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super().__init__()
        self.llm = llm

        self.head_dim             = head_dim
        self.head_dim_half        = head_dim // 2
        self.head_dim_quarter     = head_dim // 4
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads             = num_heads + num_key_value_heads
        self.total_qkv_heads      = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes      = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes       = [num_heads, num_key_value_heads]

        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        self.kv_f16             = (KV_QUANT_DTYPE == "F16")
        self.kv_q8              = (KV_QUANT_DTYPE == "Q8")
        self.kv_q8_cuda         = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.kv_rotary_q8       = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4       = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q8_CUDA")
        self.kv_rotary_q4_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q4_CUDA")
        self.kv_rotary_cuda     = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary          = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized       = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized   = self.kv_quantized or self.kv_rotary
        self.kv_sym             = USE_SYM and self.kv_any_quantized
        self.compute_in_f32     = COMPUTE_IN_F32

        self.kv_q8_grouped      = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim

        self.kv_unpack_head_dim = (head_dim // 2) if self.kv_rotary_q4_cuda else head_dim
        self.kv_pack_quarter    = (head_dim // 8) if self.kv_rotary_q4_cuda else (head_dim // 4)

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
        hidden_rms_norm = self.llm.model.layers[0].input_layernorm
        hidden_rms_norm_eps = float(getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6)))
        self.hidden_rms_norm_eps = hidden_rms_norm_eps
        self.register_buffer("hidden_norm_scale", torch.full((hidden_size,),    hidden_size ** -0.5, dtype=torch.float32))

        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_any_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

        self._replace_gelu_with_tanh_approximation(self.llm)

        # Merge token embedding into Main without changing MiniCPM5's independent LM head.
        final_norm_weight = self.llm.model.norm.weight.detach().clone().float()
        embed_module = self.llm.model.embed_tokens
        self.embed_tokens = torch.nn.Embedding(embed_module.num_embeddings, embed_module.embedding_dim)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_module.weight.detach().float())

        self._fuse_weights(hidden_size)
        self.register_buffer("final_norm_scale", final_norm_weight, persistent=False)

        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

        self.o_proj_in_features = self.llm.model.layers[0].self_attn.o_proj.in_features
        self.mlp_split          = [self.llm.model.layers[0].mlp.down_proj.in_features] * 2

    def _fuse_weights(self, hidden_size):
        q_scale     = self.head_dim ** -0.5
        norm_factor = hidden_size ** 0.5

        with torch.no_grad():
            for layer in self.llm.model.layers:
                self._fuse_qkv_projection(layer, q_scale, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)

            del self.llm.model.norm

    def _fuse_qkv_projection(self, layer, q_scale, norm_factor):
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        in_features  = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias     = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * q_scale, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:

            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)

            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * q_scale, _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features

        del attn.q_proj, attn.k_proj, attn.v_proj

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up         = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _reorder_downproj_for_quant(self, key):
        with torch.no_grad():
            for layer in self.llm.model.layers:
                W = layer.mlp.down_proj.weight              # (hidden, intermediate)
                a = W.abs()
                if key == "rms":
                    stat = (W * W).mean(0).sqrt()
                elif key == "L4":
                    stat = a.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    stat = W.std(0)
                else:                                       # "absmean" (default / fallback)
                    stat = a.mean(0)
                perm  = torch.argsort(stat)
                inter = layer.mlp.down_proj.in_features
                gu    = layer.mlp.gate_up_proj.weight       # (2*intermediate, hidden): [gate; up]
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([gu[:inter][perm], gu[inter:][perm]], dim=0))
                layer.mlp.down_proj.weight.copy_(W[:, perm])

    def _reorder_oproj_for_quant(self, key):
        H, KVH, Dh, qk_heads = self.num_heads, self.num_key_value_heads, self.head_dim, self.qk_heads
        G = H // KVH
        with torch.no_grad():
            for layer in self.llm.model.layers:
                Wo  = layer.self_attn.o_proj.weight                 # (hidden, H*head_dim)
                Woc = Wo.view(Wo.shape[0], H, Dh)                   # (hidden, H, head_dim)
                perms = []
                for kvh in range(KVH):                              # one order per kv head
                    cols = Woc[:, kvh * G:(kvh + 1) * G, :]         # its G query heads combined
                    a = cols.abs()
                    if key == "rms":
                        stat = (cols * cols).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        stat = cols.reshape(-1, Dh).std(0)
                    elif key == "L4":
                        stat = a.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:                                           # "absmean" (default / fallback)
                        stat = a.mean(dim=(0, 1))
                    perms.append(torch.argsort(stat))
                Woc2 = Woc.clone()
                for h in range(H):
                    Woc2[:, h, :] = Woc2[:, h, perms[h // G]]
                Wo.copy_(Woc2.reshape(Wo.shape[0], H * Dh))
                Wq  = layer.self_attn.qkv.weight                    # (total_heads*head_dim, hidden)
                Wqr = Wq.view(-1, Dh, Wq.shape[1]).clone()          # (total_heads, head_dim, hidden)
                for kvh in range(KVH):
                    Wqr[qk_heads + kvh] = Wqr[qk_heads + kvh][perms[kvh]]
                Wq.copy_(Wqr.reshape(Wq.shape[0], Wq.shape[1]))

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                LLM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, scale, eps):
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x):
        x = onnx_reshape_batch(x, (-1, 1, self.qk_heads, 2, self.head_dim_half))
        x = x.flip(-2)
        return onnx_reshape_batch(x, (-1, 1, self.qk_heads, self.head_dim))

    def forward(self, *all_inputs):
        input_ids          = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        hidden_states      = self.embed_tokens(input_ids)

        # F16 attention keeps the growing cache in F16 unless COMPUTE_IN_F32 is enabled.
        attn_mask_f16 = attention_mask.half() if (self.kv_f16 and not self.compute_in_f32) else None

        for i, layer in enumerate(self.llm.model.layers):

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)

            qkv   = layer.self_attn.qkv(hidden_states)
            qkv   = onnx_reshape_batch(qkv, (-1, 1, self.total_qkv_heads, self.head_dim))
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin

            if self.kv_f16 and not self.compute_in_f32:
                qk_rot = qk_rot.half()   # One F16 cast feeds both Q and K; the split/reshape/view/permute below then run in F16.

            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            q    = onnx_reshape_batch(q, (-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim))
            q    = q.permute(0, 2, 3, 1, 4)

            if self.kv_f16:
                if self.compute_in_f32:
                    k = k.half()   # F16 KV storage while Q stays F32; K is upcast (k.float()) at matmul time.
                v = v.half()

            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            if self.kv_rotary_q4:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k)).float()
                    q_rot      = self.quantizer.rotate_q(q)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = onnx_reshape_batch(q_rot, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = onnx_reshape_batch(k_unpacked, (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v)).float()
                    v_q_g      = onnx_reshape_batch(v_unpacked, (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                    v_dequant  = onnx_reshape_batch(v_q_g * v_s, (self.num_key_value_heads, 1, -1, self.head_dim))
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                    v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],   dim=-3)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_rotary_q4_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer.unpack_q4_k(k).float()
                    q_rot      = self.quantizer.rotate_q(q)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = onnx_reshape_batch(q_rot, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = onnx_reshape_batch(k_unpacked, (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                    attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer.unpack_q4_v(v).float()
                    v_q_g      = onnx_reshape_batch(v_unpacked, (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                    v_dequant  = onnx_reshape_batch(v_q_g * v_s + v_b, (self.num_key_value_heads, 1, -1, self.head_dim))
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn)

            elif self.kv_rotary:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = onnx_reshape_batch(q_rot, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = onnx_reshape_batch(k_signed, (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = onnx_reshape_batch(v_signed, (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        v_dequant  = onnx_reshape_batch(v_q_g * v_s, (self.num_key_value_heads, 1, -1, self.head_dim))
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn)
                    else:
                        q_rot         = self.quantizer.rotate_q(q)
                        attn_raw      = torch.matmul(q_rot, k_signed)
                        attn          = attn_raw * k_s + attention_mask
                        attn          = torch.softmax(attn, dim=-1)

                        v_scaled  = v_signed * v_s
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled))
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, self.num_key_value_heads, self.kv_pack_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-2)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_rotary_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.kv_unpack_head_dim)

                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = onnx_reshape_batch(q_rot, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = onnx_reshape_batch(k.float(), (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = onnx_reshape_batch(v.float(), (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        v_dequant  = onnx_reshape_batch(v_q_g * v_s + v_b, (self.num_key_value_heads, 1, -1, self.head_dim))
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn)
                    else:
                        q_rot         = self.quantizer.rotate_q(q)
                        attn_raw      = torch.matmul(q_rot, k.float())
                        q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                        attn_bias     = q_bias_factor * k_b + attention_mask
                        attn          = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn          = torch.softmax(attn, dim=-1)

                        v_scaled  = v.float() * v_s
                        bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled)) + bias_term

            elif self.kv_quantized:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, self.num_key_value_heads, self.head_dim_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v],  dim=-2)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_v_scale[i] = v_s

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        v_s = v_s.float()

                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = onnx_reshape_batch(q_in, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = onnx_reshape_batch(k_signed, (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = onnx_reshape_batch(v_signed, (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        v_dequant  = onnx_reshape_batch(v_q_g * v_s, (self.num_key_value_heads, 1, -1, self.head_dim))
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(q, k_signed)
                        attn     = attn_raw * k_s + attention_mask
                        attn     = torch.softmax(attn, dim=-1)

                        v_scaled  = v_signed * v_s
                        attn      = torch.matmul(attn, v_scaled)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, self.num_key_value_heads, self.head_dim_quarter)
                    k   = torch.cat([all_inputs[i],                     packed_k], dim=-1)
                    v   = torch.cat([all_inputs[i + self.num_layers],   packed_v], dim=-2)
                    k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k],  dim=-1)
                    k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k],   dim=-1)
                    if self.kv_q8_grouped:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-3)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-3)
                    else:
                        v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v],  dim=-2)
                        v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v],  dim=-2)

                    self.save_key[i]     = k
                    self.save_value[i]   = v
                    self.save_k_scale[i] = k_s
                    self.save_k_bias[i]  = k_b
                    self.save_v_scale[i] = v_s
                    self.save_v_bias[i]  = v_b

                    if USE_FLOAT16_SCALE_BIAS:
                        k_s = k_s.float()
                        k_b = k_b.float()
                        v_s = v_s.float()
                        v_b = v_b.float()

                    if self.kv_q8_cuda:
                        k = self.quantizer.unpack_cuda(k, -2, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = onnx_reshape_batch(q_in, (self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = onnx_reshape_batch(k.float(), (self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1))
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        q_sum_g    = q_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = onnx_reshape_batch(v.float(), (self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size))
                        v_dequant  = onnx_reshape_batch(v_q_g * v_s + v_b, (self.num_key_value_heads, 1, -1, self.head_dim))
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw  = torch.matmul(q, k.float())
                        attn_bias = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                        attn      = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn      = torch.softmax(attn, dim=-1)
                        v_dequant = torch.addcmul(v_b, v.float(), v_s)
                        attn      = torch.matmul(attn, v_dequant)

            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i]   = k
                self.save_value[i] = v

                if self.kv_f16:
                    if self.compute_in_f32:
                        attn = torch.matmul(q, k.float()) + attention_mask
                        attn = torch.softmax(attn, dim=-1)
                        attn = torch.matmul(attn, v.float())
                    else:
                        attn = torch.matmul(q, k) + attn_mask_f16
                        attn = torch.softmax(attn, dim=-1)
                        attn = torch.matmul(attn, v).float()
                else:
                    attn = torch.matmul(q, k) + attention_mask
                    attn = torch.softmax(attn, dim=-1)
                    attn = torch.matmul(attn, v)

            attn          = onnx_reshape_batch(attn.permute(0, 3, 1, 2, 4), (-1, self.o_proj_in_features))
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)

            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, self.mlp_split, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self._rms_norm(hidden_states[:, -1:], self.final_norm_scale, self.hidden_rms_norm_eps)
        logits        = onnx_reshape_batch(self.llm.lm_head(hidden_states), (-1,))

        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale, logits
        elif self.kv_any_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


def collect_special_token_ids(model_path):
    ids = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        def special_id(piece):
            tid = tok.convert_tokens_to_ids(piece)
            return int(tid) if isinstance(tid, int) and tid >= 0 else None

        def single_id(text):
            enc = tok.encode(text, add_special_tokens=False)
            return int(enc[0]) if len(enc) == 1 else None

        candidates = {
            "chat_endoftext_id":      special_id("<|endoftext|>"),
            "chat_im_start_id":       special_id("<|im_start|>"),
            "chat_im_end_id":         special_id("<|im_end|>"),
            "chat_think_start_id":    special_id("<think>"),
            "chat_think_end_id":      special_id("</think>"),
            "chat_system_id":         single_id("system"),
            "chat_user_id":           single_id("user"),
            "chat_assistant_id":      single_id("assistant"),
            "chat_newline_id":        single_id("\n"),
            "chat_double_newline_id": single_id("\n\n"),
        }
        ids = {key: value for key, value in candidates.items() if value is not None}
    except Exception as exc:  # noqa: BLE001
        print(f"[Metadata] Tokenizer-derived token IDs skipped ({exc}).")
    return ids


def build_model_metadata(*sections):
    def _norm(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    merged = {}
    for section in sections:
        for key, value in section.items():
            if value is not None:
                merged[key] = _norm(value)
    return merged


def write_onnx_metadata(onnx_path, metadata):
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    canonical = {prop.key: prop.value for prop in model.metadata_props}
    canonical.update(metadata)
    model.ClearField("metadata_props")
    for key in sorted(canonical):
        model.metadata_props.add(key=key, value=canonical[key])
    onnx.save(model, onnx_path)


RUNTIME_STANDALONE_MODEL_KEYS = (
    "metadata",
    "kv_slice",
    "kv_split2",
    "kv_concat",
    "rope_shift",
)
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = set()


def copy_runtime_standalones(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    copied = []
    for key in RUNTIME_STANDALONE_MODEL_KEYS:
        file_name = MODEL_FILE_NAMES[key]
        source = source_folder / file_name
        target = target_folder / file_name
        target.unlink(missing_ok=True)
        target.with_name(target.name + ".data").unlink(missing_ok=True)
        if not source.exists():
            if key in REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS:
                raise FileNotFoundError(
                    f"Required runtime standalone graph was not exported: {source}"
                )
            continue
        shutil.copy2(source, target)
        source_data = source.with_name(source.name + ".data")
        if source_data.exists():
            shutil.copy2(source_data, target.with_name(target.name + ".data"))
        copied.append(target)
    return copied


if __name__ == "__main__" and DO_EXPORT:
    if RAW_ONNX_DIR.exists():
        shutil.rmtree(RAW_ONNX_DIR)
    RAW_ONNX_DIR.mkdir(parents=True)

    print('Export start ...')
    with (torch.inference_mode()):

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()

        num_layers   = model.config.num_hidden_layers
        num_heads    = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim     = model.config.head_dim
        vocab_size   = int(model.config.vocab_size)
        hidden_size  = model.model.embed_tokens.embedding_dim
        scale_dtype  = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

        rope_inv_freq = model.model.rotary_emb.inv_freq.detach().float().clone()

        _cfg = model.config
        _gen_cfg = getattr(model, "generation_config", None)
        _eos = getattr(_gen_cfg, "eos_token_id", None) if _gen_cfg is not None else None
        if _eos is None:
            _eos = getattr(_cfg, "eos_token_id", None)
        if isinstance(_eos, (list, tuple)):
            _eos_ids = ",".join(str(int(e)) for e in _eos)
        elif _eos is not None:
            _eos_ids = str(int(_eos))
        else:
            _eos_ids = None
        model_config_meta = {
            "model_type":              getattr(_cfg, "model_type", None),
            "intermediate_size":       getattr(_cfg, "intermediate_size", None),
            "max_position_embeddings": getattr(_cfg, "max_position_embeddings", None),
            "rope_theta":              getattr(_cfg, "rope_theta", None),
            "rms_norm_eps":            getattr(_cfg, "rms_norm_eps", None),
            "tie_word_embeddings":     getattr(_cfg, "tie_word_embeddings", None),
            "bos_token_id":            getattr(_cfg, "bos_token_id", None),
            "pad_token_id":            getattr(_cfg, "pad_token_id", None),
            "eos_token_ids":           _eos_ids,
            "stop_token_ids":          _eos_ids,
        }
        chat_token_meta = collect_special_token_ids(MODEL_PATH)

        for note in normalize_kv_quant_settings(head_dim):
            print(f"\n{note}")

        trace_repeat_penalty = 1.0
        trace_penalty_range  = 20
        batch_size  = 1
        ids_len     = torch.tensor([10], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        cache_len   = torch.tensor([0], dtype=torch.int64)
        kv_seq_len  = ids_len + history_len
        logits      = torch.ones((batch_size, vocab_size), dtype=torch.float32)

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
                    kv_specs.extend([
                        ('key_scale', 5), ('key_bias', 5),
                        ('value_scale', 3), ('value_bias', 3)
                    ])
                else:
                    kv_specs.extend([
                        ('key_scale', 4), ('key_bias', 4),
                        ('value_scale', 3), ('value_bias', 3)
                    ])
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
                    kv_specs.extend([
                        ('key_scale', 5), ('key_bias', 5),
                        ('value_scale', 3), ('value_bias', 3)
                    ])
                else:
                    kv_specs.extend([
                        ('key_scale', 4), ('key_bias', 4),
                        ('value_scale', 3), ('value_bias', 3)
                    ])
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
            'key':   torch.zeros((batch_size, num_kv_heads, 1, k_head, history_len), dtype=kv_dtype),
            'value': torch.zeros((batch_size, num_kv_heads, 1, history_len, v_head), dtype=kv_dtype)
        }
        if KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"):
            if _grouped_6d:
                kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE
                scale_k_dim3  = kv_quant_num_groups
                scale_v_dim4  = kv_quant_num_groups
                kv_tensors.update({
                    'key_scale':   torch.ones((batch_size, num_kv_heads, 1, scale_k_dim3, 1, history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_kv_heads, 1, history_len, scale_v_dim4, 1), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias':    torch.ones((batch_size, num_kv_heads, 1, scale_k_dim3, 1, history_len), dtype=scale_dtype),
                        'value_bias':  torch.ones((batch_size, num_kv_heads, 1, history_len, scale_v_dim4, 1), dtype=scale_dtype),
                    })
            else:
                scale_k_dim3  = 1
                scale_v_dim4  = 1
                kv_tensors.update({
                    'key_scale':   torch.ones((batch_size, num_kv_heads, 1, scale_k_dim3, history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_kv_heads, 1, history_len, scale_v_dim4), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias':    torch.ones((batch_size, num_kv_heads, 1, scale_k_dim3, history_len), dtype=scale_dtype),
                        'value_bias':  torch.ones((batch_size, num_kv_heads, 1, history_len, scale_v_dim4), dtype=scale_dtype),
                    })

        def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len', out_batch_axis=None):
            if out_batch_axis is None:
                out_batch_axis = batch_axis
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                tensor = tensors_dict[name]
                for i in range(num_layers):
                    in_n  = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(tensor)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {dim: seq_axis}
                    axes[out_n] = {dim: out_seq_axis}
                    add_batch_dynamic_axes(
                        axes, in_n, axis_name=batch_axis
                    )
                    add_batch_dynamic_axes(
                        axes, out_n, axis_name=out_batch_axis
                    )
            return inputs, in_names, out_names, axes

        torch.onnx.export(
            ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len, cache_len),
            onnx_model_Rotary_Text_Prefill,
            input_names=['ids_len', 'history_len', 'cache_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'mask_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        torch.onnx.export(
            ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)

        input_ids_main = torch.ones((batch_size, ids_len), dtype=torch.int32)
        rotary_cos     = torch.zeros((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin     = rotary_cos
        attention_mask = torch.zeros((1, 1, 1, ids_len, kv_seq_len), dtype=torch.float32)

        all_inputs   = kv_ins + [input_ids_main, rotary_cos, rotary_sin, attention_mask]
        input_names  = kv_in_names + ['input_ids', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['logits']
        dynamic_axes = add_batch_dynamic_axes(
            {
                **kv_axes,
                'input_ids':      {1: 'ids_len'},
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
            },
            'input_ids',
            'logits',
        )

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
            dynamo=False
        )
        kv_quantizer = model_Main.quantizer
        del model_Main, input_ids_main, attention_mask, all_inputs
        gc.collect()

        save_id_in = torch.zeros((batch_size, 10), dtype=torch.int32)

        torch.onnx.export(
            PENALTY_GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Penalty_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes=add_batch_dynamic_axes(
                {
                    'save_id_in':  {1: 'history_len'},
                    'save_id_out': {1: 'history_len'},
                },
                'logits',
                'save_id_in',
                'save_id_out',
                'max_logits_idx',
            ),
            opset_version=OPSET,
            dynamo=False
        )

        penalty_value = torch.tensor([trace_repeat_penalty], dtype=torch.float32)
        penalty_range = torch.tensor([trace_penalty_range],  dtype=torch.int64)

        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes=add_batch_dynamic_axes(
                {'save_id_in': {1: 'history_len'}},
                'logits_in',
                'save_id_in',
                'logits_out',
            ),
            opset_version=OPSET,
            dynamo=False
        )
        del save_id_in, penalty_value, penalty_range

        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits,),
            onnx_model_Greedy,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes=add_batch_dynamic_axes(
                {}, 'logits', 'max_logits_idx'
            ),
            opset_version=OPSET,
            dynamo=False
        )

        sampling_temperature        = torch.tensor([0.8], dtype=torch.float32)
        sampling_top_k              = torch.tensor([50], dtype=torch.int32)
        sampling_top_p              = torch.tensor([0.95], dtype=torch.float32)
        sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
        sampling_previous_ids       = torch.zeros((1, 10), dtype=torch.int32)

        torch.onnx.export(
            TOPK_TOPP_SAMPLING(),
            (logits[[0]], sampling_temperature, sampling_top_k, sampling_top_p, sampling_repetition_penalty, sampling_previous_ids),
            onnx_model_TopKTopP_Sampling,
            input_names=['logits', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'previous_ids'],
            output_names=['sampled_id', 'save_id_out'],
            dynamic_axes={
                'previous_ids': {1: 'history_len'},
                'save_id_out':  {1: 'history_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del sampling_temperature, sampling_top_k, sampling_top_p, sampling_repetition_penalty, sampling_previous_ids

        del logits
        gc.collect()

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='sliced_len')
        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end   = torch.tensor([5], dtype=torch.int64)

        torch.onnx.export(
            KV_SLICE(num_layers, head_dim),
            tuple(kv_ins + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=kv_in_names + ['slice_start', 'slice_end'],
            output_names=kv_out_names,
            dynamic_axes=kv_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del slice_start, slice_end

        split_at = torch.tensor([5], dtype=torch.int64)
        split_prefix_names = [f'prefix_{name}' for name in kv_out_names]
        split_window_names = [f'window_{name}' for name in kv_out_names]
        split_axes = {name: dict(kv_axes[name]) for name in kv_in_names}
        for source_name, prefix_name, window_name in zip(kv_out_names, split_prefix_names, split_window_names):
            source_axes = kv_axes[source_name]
            split_axes[prefix_name] = dict(source_axes)
            split_axes[window_name] = dict(source_axes)
            for axis in split_axes[prefix_name]:
                if axis != 0:
                    split_axes[prefix_name][axis] = 'prefix_len'
                    split_axes[window_name][axis] = 'window_len'

        torch.onnx.export(
            KV_SPLIT2(num_layers, head_dim),
            tuple(kv_ins + [split_at]),
            onnx_model_KV_Split2,
            input_names=kv_in_names + ['split_at'],
            output_names=split_prefix_names + split_window_names,
            dynamic_axes=split_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del split_at, split_prefix_names, split_window_names, split_axes

        cat_a_ins, cat_a_names, cat_b_ins, cat_b_names, cat_out_names, cat_axes = [], [], [], [], [], {}
        for name, dim in kv_specs:
            tensor = kv_tensors[name]
            for i in range(num_layers):
                a_n, b_n, o_n = f'in_a_{name}_{i}', f'in_b_{name}_{i}', f'out_{name}_{i}'
                cat_a_ins.append(tensor);         cat_a_names.append(a_n)
                cat_b_ins.append(tensor.clone()); cat_b_names.append(b_n)
                cat_out_names.append(o_n)
                cat_axes[a_n] = {dim: 'prefix_len'}
                cat_axes[b_n] = {dim: 'suffix_len'}
                cat_axes[o_n] = {dim: 'concat_len'}
                add_batch_dynamic_axes(
                    cat_axes,
                    a_n,
                    b_n,
                    o_n,
                    axis_name='batch_size',
                )

        torch.onnx.export(
            KV_CONCAT(num_layers, head_dim),
            tuple(cat_a_ins + cat_b_ins),
            onnx_model_KV_Concat,
            input_names=cat_a_names + cat_b_names,
            output_names=cat_out_names,
            dynamic_axes=cat_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del cat_a_ins, cat_a_names, cat_b_ins, cat_b_names, cat_out_names, cat_axes

        def _rope_shift_key_io(specs):
            ins, in_names, out_names, axes = [], [], [], {}
            for name, tensor in specs:
                seq_axis = tensor.dim() - 1
                for i in range(num_layers):
                    in_n, out_n = f'in_{name}_{i}', f'out_{name}_{i}'
                    ins.append(tensor)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {seq_axis: 'history_len'}
                    axes[out_n] = {seq_axis: 'history_len'}
                    add_batch_dynamic_axes(
                        axes,
                        in_n,
                        out_n,
                        axis_name='batch_size',
                    )
            return ins, in_names, out_names, axes

        rope_shift_amount = torch.tensor([5], dtype=torch.int64)
        if KV_QUANT_DTYPE in ("F16", "F32"):
            kv_ins, kv_in_names, kv_out_names, kv_axes = _rope_shift_key_io([('key', kv_tensors['key'])])
            torch.onnx.export(
                ROPE_SHIFT(num_layers, head_dim, num_kv_heads, rope_inv_freq, MAX_SEQ_LEN),
                tuple(kv_ins + [rope_shift_amount]),
                onnx_model_Rope_Shift,
                input_names=kv_in_names + ['shift'],
                output_names=kv_out_names,
                dynamic_axes=kv_axes,
                opset_version=OPSET,
                dynamo=False
            )
        elif KV_QUANT_DTYPE in ("Q8", "ROTARY_Q8", "ROTARY_Q4"):
            def _seq4(tensor):
                shape = list(tensor.shape)
                shape[-1] = 4
                return torch.zeros(shape, dtype=tensor.dtype)
            rs_specs = [('key', _seq4(kv_tensors['key'])), ('key_scale', _seq4(kv_tensors['key_scale']))]
            if not _kv_sym:
                rs_specs.append(('key_bias', _seq4(kv_tensors['key_bias'])))
            kv_ins, kv_in_names, kv_out_names, kv_axes = _rope_shift_key_io(rs_specs)
            torch.onnx.export(
                ROPE_SHIFT_QUANT(num_layers, head_dim, num_kv_heads, rope_inv_freq, MAX_SEQ_LEN, kv_quantizer, not _kv_sym),
                tuple(kv_ins + [rope_shift_amount]),
                onnx_model_Rope_Shift,
                input_names=kv_in_names + ['shift'],
                output_names=kv_out_names,
                dynamic_axes=kv_axes,
                opset_version=OPSET,
                dynamo=False
            )
            del rs_specs
        else:
            kv_ins, kv_in_names, kv_out_names, kv_axes = None, None, None, None

        del rope_shift_amount, kv_ins, kv_in_names, kv_out_names, kv_axes, kv_tensors
        gc.collect()

        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        torch.onnx.export(
            METADATA_CARRIER(),
            (metadata_marker,),
            onnx_model_Metadata,
            input_names=['metadata_marker'],
            output_names=['metadata_marker_out'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        del metadata_marker

        _dtype_names = {
            torch.float32: "float32", torch.float16: "float16",
            torch.int8: "int8", torch.uint8: "uint8", torch.int32: "int32", torch.int64: "int64",
        }
        _kv_blocks_per_layer = len(kv_specs)                 # F16/F32=2, sym=4, asym=6
        _kv_is_quant         = _is_quantized or _is_rotary
        onnx_metadata = build_model_metadata(
            {
                "native_llm_metadata_version": 1,
                "producer":                    "MiniCPM_Export.py",
            },
            MODEL_FILE_NAME_METADATA,
            {
                "num_layers":                  num_layers,
                "num_full_attention_layers":   num_layers,   # pure transformer: every layer is full attention
                "num_linear_attention_layers": 0,
                "num_attention_heads":         num_heads,
                "num_key_value_heads":         num_kv_heads,
                "head_dim":                    head_dim,
                "hidden_size":                 hidden_size,
                "vocab_size":                  vocab_size,
                "max_seq_len":                 MAX_SEQ_LEN,
                "activations_fp16":            False,
                "compute_in_f32":              COMPUTE_IN_F32,
                "dynamic_batch":               USE_BATCH,
                "opset":                       OPSET,
                "reorder_downproj":            REORDER_DOWNPROJ_FOR_QUANT,
                "reorder_oproj":               REORDER_OPROJ_FOR_QUANT,
                "embed_merged_into_main":      True,
            },
            {
                "layer_types": ",".join(["full_attention"] * num_layers),
            },
            {
                "kv_quant_dtype":          KV_QUANT_DTYPE,
                "kv_quant_group_size":     KV_QUANT_GROUP_SIZE,
                "kv_blocks_per_layer":     _kv_blocks_per_layer,
                "kv_num_tensors":          num_layers * _kv_blocks_per_layer,
                "kv_symmetric":            _kv_sym,
                "kv_grouped_6d":           _grouped_6d,
                "kv_cache_elem_type":      _dtype_names.get(kv_dtype, str(kv_dtype)),
                "kv_scale_bias_elem_type": _dtype_names.get(scale_dtype) if _kv_is_quant else None,
                "kv_quant_hadamard":       USE_HADAMARD if _kv_is_quant else None,
                "kv_quant_shuffle":        USE_SHUFFLE if _kv_is_quant else None,
                "kv_quant_clip":           USE_CLIP if _kv_is_quant else None,
            },
            model_config_meta,
            chat_token_meta,
        )

        # Raw split graphs are finalized before merged composition and remain immutable afterward.
        _metadata_targets = [
            onnx_model_Metadata,
            onnx_model_Greedy, onnx_model_Penalty_Greedy,
            onnx_model_TopKTopP_Sampling,
            onnx_model_Penalty, onnx_model_KV_Slice,
            onnx_model_KV_Split2, onnx_model_KV_Concat,
            onnx_model_Rope_Shift,
        ]
        _written, _skipped = [], []
        for _target in _metadata_targets:
            if not Path(_target).exists():
                continue
            try:
                write_onnx_metadata(_target, onnx_metadata)
                _written.append(Path(_target).name)
            except Exception as _exc:  # noqa: BLE001
                _skipped.append(f"{Path(_target).name} ({_exc})")

        print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_written)} ONNX graph(s):")
        for _key in sorted(onnx_metadata):
            print(f"    {_key} = {onnx_metadata[_key]}")
        if _skipped:
            print("[Metadata] Skipped (kept usable, metadata not written):")
            for _entry in _skipped:
                print(f"    {_entry}")
        gc.collect()

        import Shared_Merged
        print("\n[SharedMerged] Building merged decode-strategy graphs + LLM_SharedInitializers bundle ...")
        _bundle = Shared_Merged.build_shared_merged_bundle(
            RAW_ONNX_DIR,
            out_folder=ONNX_DIR,
            model_file_names=MODEL_FILE_NAMES,
        )
        _copied_standalones = copy_runtime_standalones(RAW_ONNX_DIR, ONNX_DIR)
        for _name, _path in _bundle["graphs"].items():
            print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
        print(f"    {MODEL_FILE_NAMES['shared_initializers_data']} ({Path(_bundle['shared_data']).stat().st_size} bytes)")
        print(f"    Copied {len(_copied_standalones)} standalone runtime graph(s) into: {ONNX_DIR}")
        # Stamp the same metadata onto every merged graph so any loaded graph reports identical sizing.
        for _target in _bundle["graphs"].values():
            try:
                write_onnx_metadata(str(_target), onnx_metadata)
            except Exception as _exc:  # noqa: BLE001
                print(f"[SharedMerged] Metadata skipped for {Path(_target).name} ({_exc}).")
        for _removed in _bundle.get("removed_constituents", []):
            print(f"[SharedMerged] Deleted absorbed split constituent: {_removed}")
        gc.collect()
        shutil.rmtree(RAW_ONNX_DIR)
        print(f"    Removed temporary raw split export: {RAW_ONNX_DIR}")


def run_inference_demo():
    status = "Export done!" if DO_EXPORT else "Export skipped."
    print(
        f"\n{status}\n\n"
        "Start running the LLM by ONNX Runtime via Inference_MiniCPM_ONNX.py.\n"
        f"Model folder: {ONNX_DIR}\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--model-folder",
            str(ONNX_DIR),
            "--tokenizer-folder",
            MODEL_PATH,
        ],
        check=True,
    )


if __name__ == "__main__":
    run_inference_demo()