import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from PIL import Image
from onnxruntime.capi import _pybind_state as C
from transformers import AutoModelForCausalLM, AutoTokenizer

download_path                    = r"/home/DakeQQ/Downloads/Qwen3.5-0.8B"                  # Set the folder path where the Qwen3.5 dense model project downloaded.
onnx_model_Embed                 = r"/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx"      # Assign a path where the exported Qwen model stored.
onnx_model_Vision                = r"/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Vision.onnx"
onnx_model_Concat                = r"/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Concat.onnx"
onnx_model_Rotary_Vision_Prefill = r"/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Vision_Prefill.onnx"
onnx_model_Rotary_Vision_Decode  = r"/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Vision_Decode.onnx"
onnx_model_Rotary_Text_Prefill   = r"/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Text_Prefill.onnx"
onnx_model_Rotary_Text_Decode    = r"/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Text_Decode.onnx"
onnx_model_Main                  = r"/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx"
onnx_model_Greedy                = r"/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx"
onnx_model_First_Beam            = r"/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx"
onnx_model_Second_Beam           = r"/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx"
onnx_model_Penalty               = r"/home/DakeQQ/Downloads/Qwen_ONNX/Apply_Penalty.onnx"
onnx_model_Argmax                = r"/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx"
onnx_model_KV_Slice              = r"/home/DakeQQ/Downloads/Qwen_ONNX/KV_Slice.onnx"

# Test Input
TEST_IMAGE                       = r"./psyduck.png"                                       # Test image for the exported onnx model.
TEST_QUERY                       = "Describe this image."                                 # Test query for the exported onnx model. Change it for text-only runs as needed.
ENABLE_THINKING                  = False                                                  # Enable thinking mode in generation.

# Model Config
DO_EXPORT                        = True                                                   # Whether to export the ONNX models
PREVENT_F16_OVERFLOW             = False                                                  # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN                       = [248044]                                               # Qwen3.5 stop token ids
MAX_SEQ_LEN                      = 4096                                                   # Max context length. Can not edit after export.

# Vision Config
HEIGHT_FACTOR                    = 25                                                     # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR                     = 25                                                     # Adjust this value to determine the resize shape and vision resolution.
IMAGE_RESIZE                     = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]                # 32 = self.patch_size * self.merge_size
INPUT_IMAGE_SIZE                 = [960, 960]                                             # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
VISION_BATCH_SIZE                = 1                                                      # Set the number of images for the vision LLM, whether DYNAMIC_IMAGE_SHAPE is True or False.
DYNAMIC_IMAGE_SHAPE              = False                                                  # Allow for a dynamic number of image inputs. (Experiment features, may cause errors)
INPUT_IMAGE_DIM                  = 5                                                      # 4 for [batch, 3, height, width]; 5 for [batch, 1, 3, height, width]

# KV cache quantization
KV_QUANT_DTYPE                   = "F16"                                                  # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE              = 32                                                     # Group size for Q4 and Q8 (when USE_HADAMARD or USE_SHUFFLE enabled) per-group quantization. Must divide head_dim evenly.
USE_HADAMARD                     = True                                                   # True = More Accuracy. Apply enhanced randomized Walsh-Hadamard mixing within each group before quantization. Works for Q4 and Q8 modes.
HADAMARD_RANDOM_SEED             = 9527                                                   # Seed for the deterministic Rademacher sign pattern used by the enhanced Hadamard transform.
USE_CLIP                         = True                                                   # Clip outliers to mean ± CLIP_SIGMA*std before quantization. Works for Q4 and Q8 modes.
CLIP_SIGMA                       = 3.0                                                    # Clip threshold in standard deviations. Lower = more aggressive clipping. 2.5-3.5 recommended. Only used when USE_CLIP=True.
USE_SHUFFLE                      = True                                                   # True = More Accuracy. Interleave channels across groups so that high-variance channels are evenly distributed. Works for Q4 and Q8 modes.
USE_SYM                          = False                                                  # True = Less RAM Bandwidth. True: symmetric quantization (no bias, absmax-based); False: asymmetric (min-max with bias). Works for Q4 and Q8 modes.
USE_FLOAT16_SCALE_BIAS           = True                                                   # Whether to use float16 for scale and bias in all quantized KV modes (Q4, Q8, and ROTARY variants).

# Decoding strategy
USE_BEAM_SEARCH                  = False                                                  # Use beam search or greedy search
REPEAT_PENALTY                   = 1.0                                                    # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE                    = 20                                                     # Recent-token window to apply penalty
MAX_BEAM_SIZE                    = 10                                                     # Max beam size for beam search. Can not edit after export.
TOP_K                            = 3                                                      # Top-K for beam search
BEAM_SIZE                        = 3                                                      # Beam size for beam search. Must be <= MAX_BEAM_SIZE

# Runtime config
ORT_LOG                          = False                                                  # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                         = False                                                  # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers         = []                                                     # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                      = 0                                                      # 0 = auto
DEVICE_ID                        = 0                                                      # Device ID for GPU
OPSET                            = 18                                                     # ONNX opset version


# ══════════════════════════════════════════════════════════════════════════════
# Static config facts from config.json
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = json.loads((Path(download_path) / "config.json").read_text(encoding="utf-8"))
TEXT_CONFIG = MODEL_CONFIG["text_config"]
VISION_CONFIG = MODEL_CONFIG["vision_config"]

HIDDEN_SIZE                    = int(TEXT_CONFIG["hidden_size"])
INTERMEDIATE_SIZE              = int(TEXT_CONFIG["intermediate_size"])
NUM_HIDDEN_LAYERS              = int(TEXT_CONFIG["num_hidden_layers"])
NUM_HEADS                      = int(TEXT_CONFIG["num_attention_heads"])
NUM_KEY_VALUE_HEADS            = int(TEXT_CONFIG["num_key_value_heads"])
HEAD_DIM                       = int(TEXT_CONFIG["head_dim"])
VOCAB_SIZE                     = int(TEXT_CONFIG["vocab_size"])
RMS_NORM_EPS                   = float(TEXT_CONFIG["rms_norm_eps"])
ATTN_OUTPUT_GATE               = bool(TEXT_CONFIG.get("attn_output_gate", False))

ROPE_PARAMETERS                = dict(TEXT_CONFIG["rope_parameters"])
PARTIAL_ROTARY_FACTOR          = float(ROPE_PARAMETERS.get("partial_rotary_factor", 1.0))
ROTARY_DIM                     = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
ROPE_THETA                     = float(ROPE_PARAMETERS["rope_theta"])
MROPE_SECTION                  = list(ROPE_PARAMETERS.get("mrope_section", [11, 11, 10]))

LAYER_TYPES                    = list(TEXT_CONFIG["layer_types"])
FULL_ATTENTION_LAYER_INDICES   = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "full_attention"]
LINEAR_ATTENTION_LAYER_INDICES = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "linear_attention"]
NUM_FULL_ATTENTION_LAYERS      = len(FULL_ATTENTION_LAYER_INDICES)
NUM_LINEAR_ATTENTION_LAYERS    = len(LINEAR_ATTENTION_LAYER_INDICES)

LINEAR_CONV_KERNEL_DIM         = int(TEXT_CONFIG["linear_conv_kernel_dim"])
LINEAR_NUM_KEY_HEADS           = int(TEXT_CONFIG["linear_num_key_heads"])
LINEAR_NUM_VALUE_HEADS         = int(TEXT_CONFIG["linear_num_value_heads"])
LINEAR_KEY_HEAD_DIM            = int(TEXT_CONFIG["linear_key_head_dim"])
LINEAR_VALUE_HEAD_DIM          = int(TEXT_CONFIG["linear_value_head_dim"])
LINEAR_KEY_DIM                 = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM
LINEAR_VALUE_DIM               = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
LINEAR_CONV_DIM                = LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM
LINEAR_CONV_STATE_LEN          = LINEAR_CONV_KERNEL_DIM - 1

VISION_HIDDEN_SIZE             = int(VISION_CONFIG["hidden_size"])
VISION_DEPTH                   = int(VISION_CONFIG["depth"])
VISION_NUM_HEADS               = int(VISION_CONFIG["num_heads"])
VISION_HEAD_DIM                = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_PATCH_SIZE              = int(VISION_CONFIG["patch_size"])
VISION_TEMPORAL_PATCH_SIZE     = int(VISION_CONFIG["temporal_patch_size"])
VISION_SPATIAL_MERGE_SIZE      = int(VISION_CONFIG["spatial_merge_size"])
VISION_OUT_HIDDEN_SIZE         = int(VISION_CONFIG["out_hidden_size"])
VISION_EMBED_SIZE              = HEIGHT_FACTOR * WIDTH_FACTOR * VISION_BATCH_SIZE

SCALE_DTYPE_TORCH = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = REPEAT_PENALTY != 1.0


# ══════════════════════════════════════════════════════════════════════════════
# KV Quant Validation
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "Q8",
    "Q8_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
    "F16",
    "F32",
)


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {
        "Q8",
        "Q8_CUDA",
        "ROTARY_Q8",
        "ROTARY_Q8_CUDA",
        "ROTARY_Q4",
        "ROTARY_Q4_CUDA",
    }
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
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim.")
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}.")
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization.")

        if (
            KV_QUANT_DTYPE in q8_kv
            and KV_QUANT_GROUP_SIZE == head_dim
            and (USE_HADAMARD or USE_SHUFFLE)
        ):
            notes.append("[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block.")
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Prompt helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_assistant_prompt_prefix() -> str:
    """Build the assistant prefix, optionally exposing the thinking block."""
    assistant_prefix = "<|im_start|>assistant\n"
    if ENABLE_THINKING:
        return assistant_prefix + "<think>\n"
    return assistant_prefix + "<think>\n\n</think>\n\n"


def build_text_prompt(query: str) -> str:
    """Build a text-only chat prompt."""
    return f"<|im_start|>user\n{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def build_multimodal_prompt(query: str) -> str:
    """Build a multimodal chat prompt with a vision placeholder."""
    return f"<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def compute_prompt_head_len(tokenizer) -> int:
    """Measure the token span that precedes injected vision embeddings."""
    prefix = "<|im_start|>user\n<|vision_start|>"
    tokens = tokenizer(prefix, return_tensors="np")["input_ids"]
    return int(tokens.shape[-1])


_is_rotary_kv = KV_QUANT_DTYPE in (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
)
_is_rotary_q4_kv = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
_is_quantized_kv = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
_kv_sym = USE_SYM and (_is_rotary_kv or _is_quantized_kv)
_q8_grouped = (
    _is_quantized_kv
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_rotary_q8_grouped = (
    KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_grouped_6d = _is_rotary_q4_kv or _q8_grouped or _rotary_q8_grouped

FULL_STATE_SPECS = [("key", 4), ("value", 3)]
if _is_quantized_kv or _is_rotary_kv:
    if _kv_sym:
        if _grouped_6d:
            FULL_STATE_SPECS.extend([("key_scale", 5), ("value_scale", 3)])
        else:
            FULL_STATE_SPECS.extend([("key_scale", 4), ("value_scale", 3)])
    else:
        if _grouped_6d:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 5),
                    ("key_bias", 5),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
        else:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 4),
                    ("key_bias", 4),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
LINEAR_STATE_SPECS = [("conv_state", None), ("recurrent_state", None)]

NUM_FULL_STATE_TENSORS = NUM_FULL_ATTENTION_LAYERS * len(FULL_STATE_SPECS)
NUM_LINEAR_STATE_TENSORS = NUM_LINEAR_ATTENTION_LAYERS * len(LINEAR_STATE_SPECS)
NUM_MAIN_STATE_TENSORS = NUM_FULL_STATE_TENSORS + NUM_LINEAR_STATE_TENSORS


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: append the highest-logit token to the running sequence."""

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_state_tensors: int):
        super().__init__()
        self.total_state_tensors = total_state_tensors

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        # Replicate exported state tensors across all beams.
        save_states = []
        for state_index in range(self.total_state_tensors):
            state = all_inputs[state_index]
            save_states.append(state.repeat(beam_size, *([1] * (state.dim() - 1))))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]

        return (
            *save_states,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx,
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand surviving beams."""

    def __init__(self, total_state_tensors: int):
        super().__init__()
        self.total_state_tensors = total_state_tensors

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        # Gather the state tensors that correspond to the surviving beams.
        save_states = []
        for state_index in range(self.total_state_tensors):
            save_states.append(torch.index_select(all_inputs[state_index], dim=0, index=beam_index))

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *save_states,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Slice
# ══════════════════════════════════════════════════════════════════════════════
class KV_SLICE(torch.nn.Module):
    """Slice exported KV cache tensors along the sequence axis."""

    def __init__(self, num_full_attn_layers: int):
        super().__init__()
        self.num_full_attn_layers = num_full_attn_layers
        self.kv_quantized = KV_QUANT_DTYPE in (
            "Q8",
            "Q8_CUDA",
            "ROTARY_Q8",
            "ROTARY_Q8_CUDA",
            "ROTARY_Q4",
            "ROTARY_Q4_CUDA",
        )
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary = KV_QUANT_DTYPE in (
            "ROTARY_Q8",
            "ROTARY_Q8_CUDA",
            "ROTARY_Q4",
            "ROTARY_Q4_CUDA",
        )
        self.kv_q8_grouped = (
            KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < HEAD_DIM
        )
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_full_attn_layers
        self.num_layers_2 = num_full_attn_layers * 2
        self.num_layers_3 = num_full_attn_layers * 3
        self.num_layers_4 = num_full_attn_layers * 4
        self.num_layers_5 = num_full_attn_layers * 5
        self.save_key = [None] * num_full_attn_layers
        self.save_value = [None] * num_full_attn_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_full_attn_layers
            self.save_v_scale = [None] * num_full_attn_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_full_attn_layers
                self.save_v_bias = [None] * num_full_attn_layers

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
            return (
                *self.save_key,
                *self.save_value,
                *self.save_k_scale,
                *self.save_v_scale,
            )
        if self.kv_quantized:
            return (
                *self.save_key,
                *self.save_value,
                *self.save_k_scale,
                *self.save_k_bias,
                *self.save_v_scale,
                *self.save_v_bias,
            )
        return *self.save_key, *self.save_value


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, ROTARY_Q8_CUDA, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation to the head_dim axis before quantization.

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

    def __init__(
        self,
        head_dim,
        num_kv_heads,
        num_kv_groups,
        is_q4=False,
        is_rotary=False,
        is_q8_cuda=False,
        use_sym=False,
        use_hadamard=False,
        use_clip=False,
        clip_sigma=2.5,
        use_shuffle=False,
    ):
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

        # Quantization range
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

        # Group parameters
        self.is_grouped = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = (head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0)

        # Q8_CUDA int32 packing constants
        if is_q8_cuda:
            for name, val in [
                ("_256", 256),
                ("_128", 128),
                ("_65536", 65536),
                ("_16777216", 16777216),
            ]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # Rotary transform buffers
        if is_rotary:
            sqrt2 = 2.0**0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,), inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[: head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # Enhanced Hadamard transform buffers
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size**-0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(
                0,
                2,
                (self.kv_quant_group_size,),
                generator=sign_generator,
                dtype=torch.int64,
            )
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # Clip sigma buffer
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # Channel shuffle buffers
        if self.use_shuffle:
            perm = (
                torch.arange(head_dim)
                .view(self.kv_quant_num_groups, self.kv_quant_group_size)
                .T.contiguous()
                .view(-1)
            )
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
        for width, half in self._hadamard_levels:
            x = x.view(*x.shape[:-1], -1, width)
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = x.view(*x.shape[:-2], -1)
        x = x * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            x = x[..., : self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    def _clip_to_sigma(self, x, dim):
        mean = x.mean(dim=dim, keepdim=True)
        var = (x - mean).square().mean(dim=dim, keepdim=True)
        std = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def _flip_k(self, k, batch_size):
        return (
            k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1)
            .flip(-3)
            .view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _flip_v(self, v, batch_size):
        return (
            v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half)
            .flip(-2)
            .view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
        )

    def _flip_q(self, q, batch_size):
        return (
            q.view(
                batch_size,
                self.num_kv_heads,
                self.num_kv_groups,
                -1,
                2,
                self.head_dim_half,
            )
            .flip(-2)
            .view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)
        )

    def rotate_k(self, k, batch_size):
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    def rotate_q(self, q, batch_size):
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    def hadamard_k(self, k, batch_size):
        k = k.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
            -1,
        )
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        v = v.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(
            batch_size,
            self.num_kv_heads,
            self.num_kv_groups,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(
            batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim
        )

    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
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
        x_quant = (
            torch.round(x / scale)
            .clamp(self.SIGNED_QMIN, self.SIGNED_QMAX)
            .to(torch.int32)
        )
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

    def _quantize_block(self, x, dim, batch_size=1):
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
        if self.use_sym:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax = x.abs().amax(dim=-2, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax = x.abs().amax(dim=-1, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-2
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-1
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return (
            x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216
        )

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
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

    def pack_q4_k(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
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
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading Helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_tokenizer():
    """Load the multimodal Qwen3.5 model, trying installed and local entry points."""
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    common_kwargs = dict(
        trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )

    def has_multimodal_layout(candidate_model) -> bool:
        return (
            hasattr(candidate_model, "model")
            and hasattr(candidate_model.model, "language_model")
            and hasattr(candidate_model.model, "visual")
        )

    load_errors = []

    try:
        installed_qwen = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        model = installed_qwen.Qwen3_5ForConditionalGeneration.from_pretrained(
            download_path,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        if has_multimodal_layout(model):
            print(
                "Loaded model via transformers.models.qwen3_5.Qwen3_5ForConditionalGeneration."
            )
            model.eval()
            return model, tokenizer
        load_errors.append(
            f"installed Qwen3_5ForConditionalGeneration returned unexpected layout: {type(model.model).__name__}"
        )
    except Exception as exc:
        load_errors.append(f"installed Qwen3_5ForConditionalGeneration failed: {exc}")

    try:
        if str(download_path) not in sys.path:
            sys.path.insert(0, str(download_path))
        local = importlib.import_module("inference_demo")
        model = local.Qwen3_5ForConditionalGeneration.from_pretrained(
            download_path,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        if has_multimodal_layout(model):
            print(
                "Loaded model via local inference_demo.Qwen3_5ForConditionalGeneration."
            )
            model.eval()
            return model, tokenizer
        load_errors.append(
            f"local inference_demo.Qwen3_5ForConditionalGeneration returned unexpected layout: {type(model.model).__name__}"
        )
    except Exception as exc:
        load_errors.append(
            f"local inference_demo.Qwen3_5ForConditionalGeneration failed: {exc}"
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            download_path, low_cpu_mem_usage=True, **common_kwargs
        )
        if has_multimodal_layout(model):
            print("Loaded model via AutoModelForCausalLM.")
            model.eval()
            return model, tokenizer
        load_errors.append(
            f"AutoModelForCausalLM loaded {type(model).__name__} with text-only layout {type(model.model).__name__}"
        )
    except Exception as exc:
        load_errors.append(f"AutoModelForCausalLM failed: {exc}")

    raise RuntimeError(
        "Unable to load Qwen3.5 multimodal model. " + " | ".join(load_errors)
    )


def replace_gelu_with_tanh_approximation(module):
    """Replace exact GELU activations with the tanh approximation recursively."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh_approximation(child)


def effective_qwen_rms_weight(norm_module) -> torch.Tensor:
    """Convert Qwen's stored RMSNorm delta weights into effective scale weights."""
    return 1.0 + norm_module.weight.data.float()


def is_valid_image_path(path):
    """Return True when `path` exists and has an image-like extension."""
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".raw"}
    _, ext = os.path.splitext(path)
    return ext.lower() in valid_extensions


# ══════════════════════════════════════════════════════════════════════════════
# TorchScript helpers for dynamic linear-attention recurrence
# ══════════════════════════════════════════════════════════════════════════════
@torch.jit.script
def recurrent_gated_delta_step(
    qk_rows_t, key_col_t, value_t, alpha_t, beta_t, qk_dot_t, state):
    projected = torch.matmul(qk_rows_t, state)
    state_k, query_state = torch.split(projected, 1, dim=-2)
    delta = value_t - state_k
    scaled_delta = beta_t * delta
    new_state = alpha_t * state + key_col_t * scaled_delta
    output_t = alpha_t * query_state + qk_dot_t * scaled_delta
    return output_t, new_state


@torch.jit.script
def recurrent_gated_delta_prefill(query, key, value, g, beta, initial_state):
    query_row = query.unsqueeze(-2)
    qk_rows = torch.stack((key, query), dim=-2)
    key_col = key.unsqueeze(-1)
    value_exp = value.unsqueeze(-2)
    alpha = torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    beta_exp = beta.unsqueeze(-1).unsqueeze(-1)
    qk_dot = torch.matmul(query_row, key_col)
    state = initial_state
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    seq_len = query.shape[1]
    for token_index in range(seq_len):
        output_t, state = recurrent_gated_delta_step(
            qk_rows.select(1, token_index),
            key_col.select(1, token_index),
            value_exp.select(1, token_index),
            alpha.select(1, token_index),
            beta_exp.select(1, token_index),
            qk_dot.select(1, token_index),
            state
        )
        outputs.append(output_t)
    output = torch.cat(outputs, dim=2).transpose(1, 2)
    return output, state


# ══════════════════════════════════════════════════════════════════════════════
# Export modules
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract the token embedding layer from the language model."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_VISION(torch.nn.Module):
    """Run the Qwen3.5 vision tower and return merged visual embeddings."""

    def __init__(self, llm):
        super().__init__()
        self.visual = llm.model.visual
        replace_gelu_with_tanh_approximation(self.visual)

        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HEAD_DIM
        self.head_dim_half = self.head_dim // 2
        self.patch_size = VISION_PATCH_SIZE
        self.merge_size = VISION_SPATIAL_MERGE_SIZE
        self.image_hidden_len = HEIGHT_FACTOR * self.merge_size * WIDTH_FACTOR * self.merge_size

        self.register_buffer("means", torch.tensor([127.5], dtype=torch.float32).view(1, 1, 1, 1, 1))
        self.register_buffer("inv_std", torch.tensor([1.0 / 127.5], dtype=torch.float32).view(1, 1, 1, 1, 1))

        grid_thw = torch.tensor([[1, HEIGHT_FACTOR * self.merge_size, WIDTH_FACTOR * self.merge_size]], dtype=torch.int32)
        pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw).unsqueeze(0)
        self.register_buffer("pos_embeds", pos_embeds)

        rotary_pos_emb = (
            self.visual.rot_pos_emb(grid_thw)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.register_buffer("rotary_pos_emb_cos", torch.cat([cos, cos], dim=-1))
        self.register_buffer("rotary_pos_emb_sin", torch.cat([-sin, sin], dim=-1))

        scaling = self.head_dim**-0.25
        norm_scale = float(self.inv_std.item())
        with torch.no_grad():
            for block in self.visual.blocks:
                qk_out = block.attn.qkv.out_features - self.visual.patch_embed.embed_dim
                block.attn.qkv.weight.data[:qk_out].mul_(scaling)
                block.attn.qkv.bias.data[:qk_out].mul_(scaling)
                self.fuse_norm(block.norm1, block.attn.qkv)
                self.fuse_norm(block.norm2, block.mlp.linear_fc1)
            self.fuse_norm(self.visual.merger.norm, self.visual.merger.linear_fc1)

            patch_embed = self.visual.patch_embed
            if patch_embed.temporal_patch_size > 1:
                fused_weight = patch_embed.proj.weight.data.sum(dim=2, keepdim=True)
                fused_bias = (
                    patch_embed.proj.bias.data.clone()
                    if patch_embed.proj.bias is not None
                    else None
                )
                fused_proj = torch.nn.Conv3d(
                    patch_embed.in_channels,
                    patch_embed.embed_dim,
                    kernel_size=(1, self.patch_size, self.patch_size),
                    stride=(1, self.patch_size, self.patch_size),
                    bias=fused_bias is not None,
                )
                fused_proj = fused_proj.to(
                    device=patch_embed.proj.weight.device,
                    dtype=patch_embed.proj.weight.dtype,
                )
                fused_proj.weight.data.copy_(fused_weight * norm_scale)
                if fused_bias is not None:
                    fused_proj.bias.data.copy_(fused_bias)
                patch_embed.proj = fused_proj
                patch_embed.temporal_patch_size = 1
            else:
                patch_embed.proj.weight.data.mul_(norm_scale)

    def fuse_norm(self, norm, linear):
        """Absorb an affine norm into the following linear projection."""
        norm_bias = norm.bias.data
        norm_weight = norm.weight.data
        if linear.weight.shape[1] != norm_bias.shape[0]:
            repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
            norm_bias = norm_bias.repeat(repeat_factor)
            norm_weight = norm_weight.repeat(repeat_factor)
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

    def rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def forward(self, pixel_values):
        if INPUT_IMAGE_DIM != 5:
            pixel_values = pixel_values.unsqueeze(1)

        batch_size = pixel_values.shape[0] if DYNAMIC_IMAGE_SHAPE else VISION_BATCH_SIZE
        pixel_values = pixel_values.float()

        if DYNAMIC_IMAGE_SHAPE or list(pixel_values.shape[-2:]) != IMAGE_RESIZE:
            pixel_values = pixel_values.squeeze(1)
            pixel_values = F.interpolate(
                pixel_values, size=IMAGE_RESIZE, mode="bilinear", align_corners=False
            )
            pixel_values = pixel_values.unsqueeze(1)

        pixel_values = pixel_values - self.means
        pixel_values = pixel_values.reshape(
            batch_size,
            1,
            1,
            3,
            HEIGHT_FACTOR,
            self.merge_size,
            self.patch_size,
            WIDTH_FACTOR,
            self.merge_size,
            self.patch_size,
        )
        pixel_values = pixel_values.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = pixel_values.reshape(-1, 3, 1, self.patch_size, self.patch_size)

        vision_hidden_states = self.visual.patch_embed.proj(pixel_values)
        if DYNAMIC_IMAGE_SHAPE:
            vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.visual.patch_embed.embed_dim)
        else:
            vision_hidden_states = vision_hidden_states.view(batch_size, self.image_hidden_len, self.visual.patch_embed.embed_dim)
        if batch_size != 1:
            vision_hidden_states = vision_hidden_states + self.pos_embeds.expand(batch_size, -1, -1)
        else:
            vision_hidden_states = vision_hidden_states + self.pos_embeds

        for block in self.visual.blocks:
            hidden_states_norm = block.norm1(vision_hidden_states)
            qkv = block.attn.qkv(hidden_states_norm)
            qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            qk, value = qkv.split([2, 1], dim=0)
            qk = qk * self.rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * self.rotary_pos_emb_sin
            query, key = qk.split([1, 1], dim=0)
            attn = torch.matmul(query, key.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, value)
            attn = attn.transpose(2, 3).reshape(batch_size, -1, block.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + block.attn.proj(attn)

            mlp_out = block.mlp.linear_fc1(block.norm2(vision_hidden_states))
            mlp_out = block.mlp.act_fn(mlp_out)
            mlp_out = block.mlp.linear_fc2(mlp_out)
            vision_hidden_states = vision_hidden_states + mlp_out

        vision_hidden_states = self.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.visual.merger.hidden_size)
        vision_hidden_states = self.visual.merger.linear_fc1(vision_hidden_states)
        vision_hidden_states = self.visual.merger.act_fn(vision_hidden_states)
        vision_hidden_states = self.visual.merger.linear_fc2(vision_hidden_states)
        return vision_hidden_states


class LLM_CONCAT(torch.nn.Module):
    """Insert vision embeddings between the prompt head and text tail."""

    def __init__(self, prompt_head_len: int):
        super().__init__()
        self.prompt_head_len = prompt_head_len

    def forward(self, text_hidden_states, vision_hidden_states):
        if text_hidden_states.shape[0] != 1:
            vision_hidden_states = vision_hidden_states.expand(text_hidden_states.shape[0], -1, HIDDEN_SIZE)
        return torch.cat(
            [
                text_hidden_states[:, : self.prompt_head_len],
                vision_hidden_states,
                text_hidden_states[:, self.prompt_head_len :],
            ],
            dim=1,
        )


class ROTARY_VISION_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and the causal mask for vision prefill."""

    def __init__(
        self,
        llm,
        width_factor: int,
        height_factor: int,
        prompt_head_len: int,
        max_seq_len: int,
    ):
        super().__init__()
        total_max = max_seq_len + width_factor * height_factor + prompt_head_len
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128

        cos, sin = self._build_rotary_table(llm, width_factor, height_factor, prompt_head_len, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    @staticmethod
    def _build_rotary_table(
        llm,
        width_factor: int,
        height_factor: int,
        prompt_head_len: int,
        max_seq_len: int,
    ):
        vision_embed_size = width_factor * height_factor
        prefix_plus_vision = prompt_head_len + vision_embed_size
        tail_start = prompt_head_len + max(width_factor, height_factor)

        position_ids = torch.arange(prefix_plus_vision, dtype=torch.float32).repeat(
            3, 1, 1
        )
        position_ids[0, :, prompt_head_len:prefix_plus_vision] = prompt_head_len

        row_start = prompt_head_len
        for start in range(prompt_head_len, prefix_plus_vision, width_factor):
            position_ids[1, :, start : start + width_factor] = row_start
            row_start += 1

        width_positions = torch.arange(
            prompt_head_len, prompt_head_len + width_factor, dtype=torch.float32
        )
        for start in range(prompt_head_len, prefix_plus_vision, width_factor):
            position_ids[2, :, start : start + width_factor] = width_positions

        fill_tail_position = torch.arange(
            tail_start, tail_start + max_seq_len, dtype=torch.float32
        ).repeat(3, 1, 1)
        position_ids = torch.cat(
            [position_ids[:, :, :prefix_plus_vision], fill_tail_position], dim=-1
        )

        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = (
            rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
        )
        freqs = inv_freq_expanded @ position_ids
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = rotary_module.apply_interleaved_mrope(
            freqs, rotary_module.mrope_section
        )
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_VISION_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for one decode step with vision context."""

    def __init__(
        self,
        llm,
        width_factor: int,
        height_factor: int,
        prompt_head_len: int,
        max_seq_len: int,
    ):
        super().__init__()
        cos, sin = ROTARY_VISION_PREFILL._build_rotary_table(llm, width_factor, height_factor, prompt_head_len, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class ROTARY_TEXT_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and the causal mask for text prefill."""

    def __init__(self, llm, max_seq_len: int):
        super().__init__()
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128

        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    @staticmethod
    def _build_rotary_table(llm, max_seq_len: int):
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).repeat(3, 1, 1)
        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = (rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1))
        freqs = inv_freq_expanded @ position_ids
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_TEXT_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for one text-only decode step."""

    def __init__(self, llm, max_seq_len: int):
        super().__init__()
        cos, sin = ROTARY_TEXT_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class LLM_MAIN(torch.nn.Module):
    """Main transformer module for Qwen3.5 export and ORT inference.

    Handles full-attention and linear-attention layers, optional KV cache
    quantization, fused projection weights, and final vocabulary projection.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.head_dim = HEAD_DIM
        self.head_dim_half = HEAD_DIM // 2
        self.head_dim_quarter = HEAD_DIM // 4
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size = INTERMEDIATE_SIZE
        self.num_heads = NUM_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KEY_VALUE_HEADS
        self.qk_heads = NUM_HEADS + NUM_KEY_VALUE_HEADS
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.linear_num_key_heads = LINEAR_NUM_KEY_HEADS
        self.linear_num_value_heads = LINEAR_NUM_VALUE_HEADS
        self.linear_key_head_dim = LINEAR_KEY_HEAD_DIM
        self.linear_value_head_dim = LINEAR_VALUE_HEAD_DIM
        self.linear_key_dim = LINEAR_KEY_DIM
        self.linear_value_dim = LINEAR_VALUE_DIM
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("LLM_MAIN reshape-before-split requires matching linear key/value head dims.")
        self.rms_norm_eps = RMS_NORM_EPS
        self.register_buffer("linear_gated_delta_query_scale", torch.tensor([float(self.linear_key_head_dim) ** -0.5], dtype=torch.float32))

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

        # Whether Q8 modes use per-group quantization (enabled by hadamard/shuffle)
        self.kv_q8_grouped = (
            (self.kv_quantized or self.kv_rotary_q8)
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < HEAD_DIM
        )

        # head_dim used for int32 unpack in rotary CUDA modes
        self.kv_unpack_head_dim = (HEAD_DIM // 2) if self.kv_rotary_q4_cuda else HEAD_DIM
        self.kv_pack_quarter = (HEAD_DIM // 8) if self.kv_rotary_q4_cuda else (HEAD_DIM // 4)

        self.num_full_layers = NUM_FULL_ATTENTION_LAYERS
        self.num_linear_layers = NUM_LINEAR_ATTENTION_LAYERS
        self.full_key_offset = 0
        self.full_value_offset = self.num_full_layers
        if self.kv_any_quantized:
            self.full_key_scale_offset = self.full_value_offset + self.num_full_layers
            if self.kv_sym:
                self.full_value_scale_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_scale_offset + self.num_full_layers
                )
            else:
                self.full_key_bias_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.full_value_scale_offset = (
                    self.full_key_bias_offset + self.num_full_layers
                )
                self.full_value_bias_offset = (
                    self.full_value_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_bias_offset + self.num_full_layers
                )
        else:
            self.num_full_state_tensors = self.full_value_offset + self.num_full_layers

        self.conv_state_offset = self.num_full_state_tensors
        self.recurrent_state_offset = self.conv_state_offset + self.num_linear_layers
        self.hidden_input_index = self.recurrent_state_offset + self.num_linear_layers
        self.rotary_cos_index = self.hidden_input_index + 1
        self.rotary_sin_index = self.hidden_input_index + 2
        self.attn_mask_index = self.hidden_input_index + 3

        self.quantizer = KVQuantizer(
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KEY_VALUE_HEADS,
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
        self.register_buffer("overflow_scale", torch.tensor([0.01], dtype=torch.float32))
        hidden_rms_norm_eps = self.hidden_size * self.rms_norm_eps
        qk_rms_norm_eps = self.head_dim * self.rms_norm_eps
        linear_rms_norm_eps = self.linear_value_head_dim * self.rms_norm_eps
        linear_qk_rms_norm_eps = 1e-6
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps *= self.overflow_scale.square()
            linear_rms_norm_eps *= self.overflow_scale.square()
            linear_qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_rms_norm_eps", torch.tensor([linear_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_qk_rms_norm_eps", torch.tensor([linear_qk_rms_norm_eps], dtype=torch.float32))
        self.lm_head_weight = None

        replace_gelu_with_tanh_approximation(self.llm.model.language_model)
        self._fuse_weights()

    def _rms_norm(self, x, eps):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size: int):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.rotary_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.rotary_dim)

    def _fuse_weights(self):
        scale_factor = self.head_dim ** -0.25
        norm_factor = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer_type, layer in zip(
                LAYER_TYPES, self.llm.model.language_model.layers
            ):
                if layer_type == "full_attention":
                    self._fuse_full_qkv_projection(
                        layer, scale_factor, norm_factor, norm_factor_qk
                    )
                else:
                    self._absorb_linear_input_norm(layer, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)

            final_norm_weight = (
                effective_qwen_rms_weight(self.llm.model.language_model.norm).unsqueeze(0) * norm_factor
            )
            fused_lm_head_weight = (
                self.llm.lm_head.weight.data.float() * final_norm_weight
            ).contiguous()
            self.lm_head_weight = torch.nn.Parameter(fused_lm_head_weight)
            del self.llm.model.language_model.norm

    def _fuse_full_qkv_projection(
        self, layer, scale_factor: float, norm_factor: float, norm_factor_qk: float
    ):
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(
            q_proj.out_features + k_proj.out_features + v_proj.out_features
        )
        has_bias = any(proj.bias is not None for proj in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        q_weight = q_proj.weight.data.reshape(self.num_heads, 2, self.head_dim, in_features)
        q_query_weight = q_weight[:, 0].reshape(-1, in_features)
        q_gate_weight = q_weight[:, 1].reshape(-1, in_features)
        k_weight = k_proj.weight.data.reshape(-1, in_features)
        v_weight = v_proj.weight.data.reshape(-1, in_features)
        qkv.weight.data.copy_(
            torch.cat([q_query_weight, k_weight, q_gate_weight, v_weight], dim=0)
        )
        if has_bias:
            q_bias = (
                q_proj.bias
                if q_proj.bias is not None
                else qkv.weight.new_zeros(q_proj.out_features)
            )
            k_bias = (
                k_proj.bias
                if k_proj.bias is not None
                else qkv.weight.new_zeros(k_proj.out_features)
            )
            v_bias = (
                v_proj.bias
                if v_proj.bias is not None
                else qkv.weight.new_zeros(v_proj.out_features)
            )

            q_bias = q_bias.reshape(self.num_heads, 2, self.head_dim)
            q_query_bias = q_bias[:, 0].reshape(-1)
            q_gate_bias = q_bias[:, 1].reshape(-1)
            qkv.bias.data.copy_(
                torch.cat([q_query_bias, k_bias, q_gate_bias, v_bias], dim=0)
            )

        combined_scale = scale_factor * norm_factor_qk
        q_norm_weight = effective_qwen_rms_weight(attn.q_norm) * combined_scale
        k_norm_weight = effective_qwen_rms_weight(attn.k_norm) * combined_scale
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat(
                [
                    q_norm_weight.repeat(self.num_heads),
                    k_norm_weight.repeat(self.num_key_value_heads),
                ],
                dim=0,
            ).view(1, 1, 1, -1, self.head_dim)
        )

        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )
        qkv.weight.data.mul_(input_norm_weight)
        attn.qkv = qkv

        del attn.q_proj, attn.k_proj, attn.v_proj
        del attn.q_norm, attn.k_norm
        del layer.input_layernorm

    def _absorb_linear_input_norm(self, layer, norm_factor: float):
        linear = layer.linear_attn
        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )

        fused_input_projs = (
            linear.in_proj_qkv,
            linear.in_proj_z,
            linear.in_proj_b,
            linear.in_proj_a,
        )
        has_bias = any(proj.bias is not None for proj in fused_input_projs) or (
            linear.dt_bias is not None
        )
        fused_out_features = sum(int(proj.out_features) for proj in fused_input_projs)

        in_proj_all = torch.nn.Linear(
            linear.in_proj_qkv.in_features,
            fused_out_features,
            bias=has_bias,
        )
        in_proj_all = in_proj_all.to(
            device=linear.in_proj_qkv.weight.device,
            dtype=linear.in_proj_qkv.weight.dtype,
        )
        in_proj_all.weight.data.copy_(
            torch.cat([proj.weight.data for proj in fused_input_projs], dim=0)
        )
        in_proj_all.weight.data.mul_(input_norm_weight)

        if has_bias:
            bias_parts = []
            for proj in fused_input_projs:
                if proj.bias is None:
                    bias_parts.append(in_proj_all.weight.new_zeros(proj.out_features))
                else:
                    bias_parts.append(proj.bias.data)
            if linear.dt_bias is not None:
                bias_parts[-1] = bias_parts[-1] + linear.dt_bias.data.to(
                    device=in_proj_all.bias.device,
                    dtype=in_proj_all.bias.dtype,
                )
            in_proj_all.bias.data.copy_(torch.cat(bias_parts, dim=0))

        linear.in_proj_all = in_proj_all
        linear.in_proj_split_sizes = tuple(
            int(proj.out_features) for proj in fused_input_projs
        )
        linear.register_buffer("g_decay_scale", -linear.A_log.data.exp())
        del linear.in_proj_qkv, linear.in_proj_z, linear.in_proj_b, linear.in_proj_a
        del linear.A_log, linear.dt_bias
        del layer.input_layernorm

        # Fuse linear attention output norm weight into out_proj
        out_norm_weight = linear.norm.weight.data.float() * (
            self.linear_value_head_dim ** 0.5
        )
        out_norm_weight = out_norm_weight.repeat(self.linear_num_value_heads)
        linear.out_proj.weight.data.mul_(out_norm_weight.unsqueeze(0))
        del linear.norm

    def _fuse_gate_up_projection(self, layer, norm_factor: float):
        post_norm_weight = (
            effective_qwen_rms_weight(layer.post_attention_layernorm).unsqueeze(0) * norm_factor
        )
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj

        gate_up_proj = torch.nn.Linear(
            gate_proj.in_features,
            gate_proj.out_features + up_proj.out_features,
            bias=False,
        )
        gate_up_proj.weight.data.copy_(
            torch.cat(
                [
                    gate_proj.weight.data * post_norm_weight,
                    up_proj.weight.data * post_norm_weight,
                ],
                dim=0,
            )
        )
        layer.mlp.gate_up_proj = gate_up_proj
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _linear_full_state_input(self, all_inputs, linear_index: int):
        conv_state = all_inputs[self.conv_state_offset + linear_index].float()
        recurrent_state = all_inputs[self.recurrent_state_offset + linear_index].float()
        return conv_state, recurrent_state

    def forward(self, *all_inputs):
        hidden_states = all_inputs[self.hidden_input_index]
        rotary_pos_emb_cos = all_inputs[self.rotary_cos_index]
        rotary_pos_emb_sin = all_inputs[self.rotary_sin_index]
        attention_mask = all_inputs[self.attn_mask_index]
        batch_size = hidden_states.shape[0]

        save_full_keys = []
        save_full_values = []
        save_key_scales = []
        save_key_biases = []
        save_value_scales = []
        save_value_biases = []
        save_conv_states = []
        save_recurrent_states = []

        full_layer_index = 0
        linear_layer_index = 0
        for layer_type, layer in zip(LAYER_TYPES, self.llm.model.language_model.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            if layer_type == "full_attention":
                attn = layer.self_attn
                qkv = attn.qkv(hidden_states)
                qkv = qkv.reshape(batch_size, -1, 2, self.qk_heads, self.head_dim)
                qk, gate_value = torch.split(qkv, 1, dim=2)

                qk = self._rms_norm(qk, self.qk_rms_norm_eps) * attn.qk_norm_weight
                if self.full_rotary:
                    qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
                else:
                    qk_rot, qk_pass = torch.split(qk, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
                    qk = torch.cat([qk_rot * rotary_pos_emb_cos + self._rotate_half(qk_rot, batch_size) * rotary_pos_emb_sin, qk_pass], dim=-1)

                query, key = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate, value = torch.split(gate_value, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate = gate.reshape(batch_size, -1, self.num_heads * self.head_dim)
                query = query.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                query = query.permute(0, 2, 3, 1, 4)

                if self.kv_f16:
                    key = key.half()
                    value = value.half()

                key = key.permute(0, 3, 2, 4, 1)
                value = value.transpose(1, 3)

                if self.kv_rotary_q4:
                    # ── ROTARY_Q4 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                        attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)

                elif self.kv_rotary:
                    # ── ROTARY_Q8 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                        v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k_signed)
                            attn_output = attn_raw * k_s + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_scaled = v_signed * v_s
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_scaled), batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                            attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k.float())
                            q_sum = query.sum(dim=-1, keepdim=True)
                            attn_output = attn_raw * k_s + q_sum * k_b + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_dequant = v.float() * v_s + v_b
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_dequant), batch_size)

                elif self.kv_quantized:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                    value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                    key_scale = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                    key_bias = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                    value_scale = torch.cat(
                        [all_inputs[self.full_value_scale_offset + full_layer_index], scale_v],
                        dim=3 if self.kv_q8_grouped else -2,
                    )
                    value_bias = torch.cat(
                        [all_inputs[self.full_value_bias_offset + full_layer_index], bias_v],
                        dim=3 if self.kv_q8_grouped else -2,
                    )

                    save_full_keys.append(key_cache)
                    save_full_values.append(value_cache)
                    save_key_scales.append(key_scale)
                    save_key_biases.append(key_bias)
                    save_value_scales.append(value_scale)
                    save_value_biases.append(value_bias)

                    if USE_FLOAT16_SCALE_BIAS:
                        key_scale = key_scale.float()
                        key_bias = key_bias.float()
                        value_scale = value_scale.float()
                        value_bias = value_bias.float()

                    if self.kv_q8_cuda:
                        key_cache = self.quantizer.unpack_cuda(key_cache, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        value_cache = self.quantizer.unpack_cuda(value_cache, -1, batch_size, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        q_grouped = query
                        if self.quantizer.use_shuffle:
                            q_grouped = q_grouped.index_select(-1, self.quantizer.shuffle_idx)
                        q_grouped = q_grouped.view(
                            batch_size,
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            -1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                        )
                        q_grouped = q_grouped.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_grouped = self.quantizer.hadamard_q(q_grouped)

                        k_grouped = key_cache.float().view(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                            -1,
                        )
                        attn_raw_grouped = torch.matmul(q_grouped, k_grouped)
                        q_sum_grouped = q_grouped.sum(dim=-1, keepdim=True)
                        attn_output = (
                            attn_raw_grouped * key_scale + q_sum_grouped * key_bias
                        ).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_grouped = value_cache.float().view(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            -1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                        )
                        value_dequant = (
                            v_grouped * value_scale + value_bias
                        ).reshape(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            -1,
                            self.head_dim,
                        )
                        attn_output = torch.matmul(attn_output, value_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(query, key_cache.float())
                        attn_bias = query.sum(dim=-1, keepdim=True) * key_bias + attention_mask
                        attn_output = torch.addcmul(attn_bias, attn_raw, key_scale)
                        attn_output = torch.softmax(attn_output, dim=-1)
                        value_dequant = torch.addcmul(value_bias, value_cache.float(), value_scale)
                        attn_output = torch.matmul(attn_output, value_dequant)
                else:
                    key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], key], dim=-1)
                    value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], value], dim=-2)
                    save_full_keys.append(key_cache)
                    save_full_values.append(value_cache)

                    if self.kv_f16:
                        key_cache = key_cache.float()
                        value_cache = value_cache.float()

                    attn_output = torch.matmul(query, key_cache) + attention_mask
                    attn_output = torch.softmax(attn_output, dim=-1)
                    attn_output = torch.matmul(attn_output, value_cache)

                attn_output = attn_output.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, attn.o_proj.in_features)
                if ATTN_OUTPUT_GATE:
                    attn_output = attn_output * torch.sigmoid(gate)
                hidden_states = residual + attn.o_proj(attn_output)
                full_layer_index += 1
            else:
                linear = layer.linear_attn
                conv_state, recurrent_state = self._linear_full_state_input(all_inputs, linear_layer_index)

                linear_inputs = linear.in_proj_all(hidden_states)
                mixed_qkv, z, beta_logits, g_logits = torch.split(linear_inputs, linear.in_proj_split_sizes, dim=-1)
                beta = torch.sigmoid(beta_logits)
                g = linear.g_decay_scale * F.softplus(g_logits)

                conv_input = torch.cat([conv_state, mixed_qkv], dim=1)
                conv_state_out = conv_input[:, -LINEAR_CONV_STATE_LEN:]
                conv_output = F.conv1d(conv_input.transpose(1, 2), linear.conv1d.weight, linear.conv1d.bias, padding=0, groups=LINEAR_CONV_DIM)
                conv_output = F.silu(conv_output).transpose(1, 2)
                conv_output = conv_output.reshape(batch_size, -1, self.linear_num_key_heads * 2 + self.linear_num_value_heads, self.linear_key_head_dim)

                qk, value = torch.split(conv_output, [self.linear_num_key_heads * 2, self.linear_num_value_heads], dim=2)
                qk = self._rms_norm(qk, self.linear_qk_rms_norm_eps)
                query, key = torch.split(qk, [self.linear_num_key_heads, self.linear_num_key_heads], dim=2)
                query = query * self.linear_gated_delta_query_scale

                core_attn_out, recurrent_state_out = recurrent_gated_delta_prefill(query, key, value, g, beta, recurrent_state)
                core_attn_out = self._rms_norm(core_attn_out, self.linear_rms_norm_eps)
                core_attn_out = core_attn_out.reshape(batch_size, -1, self.linear_value_dim)
                core_attn_out = core_attn_out * F.silu(z)

                hidden_states = residual + linear.out_proj(core_attn_out)

                save_conv_states.append(conv_state_out.half())
                save_recurrent_states.append(recurrent_state_out.half())

                linear_layer_index += 1

            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate_part, up_part = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate_part) * up_part)

        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = F.linear(hidden_states, self.lm_head_weight)

        if self.kv_any_quantized:
            if self.kv_sym:
                return (
                    *save_full_keys,
                    *save_full_values,
                    *save_key_scales,
                    *save_value_scales,
                    *save_conv_states,
                    *save_recurrent_states,
                    logits,
                )
            return (
                *save_full_keys,
                *save_full_values,
                *save_key_scales,
                *save_key_biases,
                *save_value_scales,
                *save_value_biases,
                *save_conv_states,
                *save_recurrent_states,
                logits,
            )

        return (
            *save_full_keys,
            *save_full_values,
            *save_conv_states,
            *save_recurrent_states,
            logits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Export Helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_full_state_tensors(batch_size: int):
    """Build empty full-attention KV tensors with the exported cache layout."""
    if KV_QUANT_DTYPE == "F16":
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE == "F32":
        kv_dtype = torch.float32
    elif KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"):
        kv_dtype = torch.int32
    elif _kv_sym and not _is_rotary_q4_kv:
        kv_dtype = torch.int8
    else:
        kv_dtype = torch.uint8

    # Determine KV tensor shapes based on quantization mode
    if KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA"):
        key_cache_head_dim = HEAD_DIM // 4
        value_cache_head_dim = HEAD_DIM // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4":
        key_cache_head_dim = HEAD_DIM // 2
        value_cache_head_dim = HEAD_DIM // 2
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        key_cache_head_dim = HEAD_DIM // 8
        value_cache_head_dim = HEAD_DIM // 8
    else:
        key_cache_head_dim = HEAD_DIM
        value_cache_head_dim = HEAD_DIM

    tensors = {
        "key": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, key_cache_head_dim, 0), dtype=kv_dtype
        ),
        "value": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, value_cache_head_dim),
            dtype=kv_dtype,
        ),
    }
    if _is_quantized_kv or _is_rotary_kv:
        if _grouped_6d:
            kv_quant_num_groups = HEAD_DIM // KV_QUANT_GROUP_SIZE
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                dtype=SCALE_DTYPE_TORCH,
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                dtype=SCALE_DTYPE_TORCH,
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                    dtype=SCALE_DTYPE_TORCH,
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                    dtype=SCALE_DTYPE_TORCH,
                )
        else:
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
                )
    return tensors


def build_linear_state_tensors(batch_size: int):
    """Build empty linear-attention state tensors for export and runtime."""
    return {
        "conv_state": torch.zeros(
            (batch_size, LINEAR_CONV_STATE_LEN, LINEAR_CONV_DIM), dtype=torch.float16
        ),
        "recurrent_state": torch.zeros(
            (
                batch_size,
                LINEAR_NUM_VALUE_HEADS,
                LINEAR_KEY_HEAD_DIM,
                LINEAR_VALUE_HEAD_DIM,
            ),
            dtype=torch.float16,
        ),
    }


def get_full_kv_io(
    tensors_dict,
    batch_axis="batch_size",
    seq_axis="history_len",
    out_seq_axis="kv_seq_len",
):
    """Create ordered ONNX IO lists and dynamic axes for full-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, dim in FULL_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis, dim: seq_axis}
            axes[output_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, axes


def get_linear_state_io(tensors_dict, batch_axis="batch_size"):
    """Create ordered ONNX IO lists and dynamic axes for linear-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, _ in LINEAR_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_LINEAR_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis}
            axes[output_name] = {0: batch_axis}
    return inputs, input_names, output_names, axes


# ══════════════════════════════════════════════════════════════════════════════
# Export driver
# ══════════════════════════════════════════════════════════════════════════════
if DO_EXPORT:
    print("Export start ...")

    with torch.inference_mode():
        model, tokenizer = load_model_and_tokenizer()
        prompt_head_len = compute_prompt_head_len(tokenizer)

        for note in normalize_kv_quant_settings(HEAD_DIM):
            print(f"\n{note}")

        batch_size = BEAM_SIZE
        ids_len_dummy = 10
        history_len_dummy = 0
        ids_len = torch.tensor([ids_len_dummy], dtype=torch.int64)
        history_len = torch.tensor([history_len_dummy], dtype=torch.int64)
        kv_seq_len = ids_len + history_len
        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        top_k = torch.tensor([TOP_K], dtype=torch.int64)
        logits = torch.ones((BEAM_SIZE, VOCAB_SIZE), dtype=torch.float32)

        full_state_tensors = build_full_state_tensors(batch_size)
        linear_state_tensors = build_linear_state_tensors(batch_size)
        full_inputs, full_input_names, full_output_names, full_dynamic_axes = (
            get_full_kv_io(full_state_tensors)
        )
        linear_inputs, linear_input_names, linear_output_names, linear_dynamic_axes = (
            get_linear_state_io(linear_state_tensors)
        )
        state_inputs = full_inputs + linear_inputs
        state_input_names = full_input_names + linear_input_names
        state_output_names = full_output_names + linear_output_names
        state_dynamic_axes = {**full_dynamic_axes, **linear_dynamic_axes}
        total_state_tensors = len(state_input_names)

        input_ids = torch.ones((1, ids_len_dummy), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model).eval(),
            (input_ids,),
            onnx_model_Embed,
            input_names=["input_ids"],
            output_names=["text_hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "ids_len"},
                "text_hidden_states": {0: "batch", 1: "ids_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del input_ids
        gc.collect()

        pixel_values = torch.randint(
            0,
            255,
            size=(VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]),
            dtype=torch.uint8,
        )
        if INPUT_IMAGE_DIM != 4:
            pixel_values = pixel_values.unsqueeze(1)
        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", -2: "height", -1: "width"},
            "vision_hidden_states": {1: "vision_embed_len"},
        }
        torch.onnx.export(
            LLM_VISION(model).eval(),
            (pixel_values,),
            onnx_model_Vision,
            input_names=["pixel_values"],
            output_names=["vision_hidden_states"],
            dynamic_axes=vision_dynamic_axes if DYNAMIC_IMAGE_SHAPE else None,
            opset_version=OPSET,
            dynamo=False,
        )
        del pixel_values
        gc.collect()

        text_hidden_states = torch.ones(
            (1, ids_len_dummy, HIDDEN_SIZE), dtype=torch.float32
        )
        vision_hidden_states = torch.ones(
            (1, VISION_EMBED_SIZE, HIDDEN_SIZE), dtype=torch.float32
        )
        concat_dynamic_axes = {
            "text_hidden_states": {1: "ids_len"},
            "concat_hidden_states": {1: "total_len"},
        }
        if DYNAMIC_IMAGE_SHAPE:
            concat_dynamic_axes["vision_hidden_states"] = {1: "vision_embed_len"}
        torch.onnx.export(
            LLM_CONCAT(prompt_head_len).eval(),
            (text_hidden_states, vision_hidden_states),
            onnx_model_Concat,
            input_names=["text_hidden_states", "vision_hidden_states"],
            output_names=["concat_hidden_states"],
            dynamic_axes=concat_dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del text_hidden_states, vision_hidden_states
        gc.collect()

        torch.onnx.export(
            ROTARY_VISION_PREFILL(
                model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, MAX_SEQ_LEN
            ).eval(),
            (ids_len, history_len),
            onnx_model_Rotary_Vision_Prefill,
            input_names=["ids_len", "history_len"],
            output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
            dynamic_axes={
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_VISION_DECODE(
                model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, MAX_SEQ_LEN
            ).eval(),
            (kv_seq_len,),
            onnx_model_Rotary_Vision_Decode,
            input_names=["kv_seq_len"],
            output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_TEXT_PREFILL(model, MAX_SEQ_LEN).eval(),
            (ids_len, history_len),
            onnx_model_Rotary_Text_Prefill,
            input_names=["ids_len", "history_len"],
            output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
            dynamic_axes={
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_TEXT_DECODE(model, MAX_SEQ_LEN).eval(),
            (kv_seq_len,),
            onnx_model_Rotary_Text_Decode,
            input_names=["kv_seq_len"],
            output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
            opset_version=OPSET,
            dynamo=False,
        )

        ids_len_main_dummy = ids_len_dummy + VISION_EMBED_SIZE
        hidden_states = torch.ones(
            (batch_size, ids_len_main_dummy, HIDDEN_SIZE), dtype=torch.float32
        )
        rotary_cos = torch.zeros(
            (1, ids_len_main_dummy, 1, 1, ROTARY_DIM), dtype=torch.float32
        )
        rotary_sin = torch.zeros(
            (1, ids_len_main_dummy, 1, 1, ROTARY_DIM), dtype=torch.float32
        )
        attention_mask = torch.zeros(
            (1, 1, 1, ids_len_main_dummy, ids_len_main_dummy), dtype=torch.float32
        )

        model_main = LLM_MAIN(model).eval()
        del model
        gc.collect()

        torch.onnx.export(
            model_main,
            tuple(
                state_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]
            ),
            onnx_model_Main,
            input_names=state_input_names
            + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
            output_names=state_output_names + ["logits"],
            dynamic_axes={
                **state_dynamic_axes,
                "hidden_states": {0: "batch", 1: "ids_len"},
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del model_main, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        save_id_in = torch.zeros((BEAM_SIZE, 4), dtype=torch.int32)
        torch.onnx.export(
            GREEDY_SEARCH().eval(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=["logits", "save_id_in"],
            output_names=["max_logits_idx", "save_id_out"],
            dynamic_axes={
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "max_logits_idx": {0: "batch"},
                "save_id_out": {0: "batch", 1: "history_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        single_full_state_tensors = build_full_state_tensors(1)
        single_linear_state_tensors = build_linear_state_tensors(1)
        first_full_inputs, first_full_input_names, _, first_full_axes = get_full_kv_io(
            single_full_state_tensors
        )
        first_linear_inputs, first_linear_input_names, _, first_linear_axes = (
            get_linear_state_io(single_linear_state_tensors)
        )
        first_state_inputs = first_full_inputs + first_linear_inputs
        first_state_input_names = first_full_input_names + first_linear_input_names
        first_state_output_names = [
            name.replace("in_", "out_", 1) for name in first_state_input_names
        ]

        torch.onnx.export(
            FIRST_BEAM_SEARCH(total_state_tensors).eval(),
            tuple(first_state_inputs + [logits[[0]], save_id_in, beam_size]),
            onnx_model_First_Beam,
            input_names=first_state_input_names + ["logits", "save_id_in", "beam_size"],
            output_names=first_state_output_names
            + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
            dynamic_axes={
                **first_full_axes,
                **first_linear_axes,
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "save_id_out": {0: "batch", 1: "history_len"},
                "top_beam_prob": {0: "batch"},
                "top_beam_indices": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        torch.onnx.export(
            SECOND_BEAM_SEARCH(total_state_tensors).eval(),
            tuple(state_inputs + [logits, save_id_in, previous_prob, beam_size, top_k]),
            onnx_model_Second_Beam,
            input_names=state_input_names
            + ["logits", "save_id_in", "previous_prob", "beam_size", "topK"],
            output_names=state_output_names
            + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
            dynamic_axes={
                **state_dynamic_axes,
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "previous_prob": {0: "batch"},
                "save_id_out": {0: "batch", 1: "history_len"},
                "top_beam_prob": {0: "batch"},
                "top_beam_indices": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del previous_prob

        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY().eval(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=["logits_in", "save_id_in", "penalty_value", "penalty_range"],
            output_names=["logits_out"],
            dynamic_axes={
                "logits_in": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "logits_out": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del penalty_value, penalty_range

        torch.onnx.export(
            ARGMAX().eval(),
            (logits,),
            onnx_model_Argmax,
            input_names=["logits"],
            output_names=["max_logits_idx"],
            dynamic_axes={
                "logits": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end = torch.tensor([5], dtype=torch.int64)
        torch.onnx.export(
            KV_SLICE(NUM_FULL_ATTENTION_LAYERS).eval(),
            tuple(full_inputs + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=full_input_names + ["slice_start", "slice_end"],
            output_names=full_output_names,
            dynamic_axes=full_dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )

        del (
            tokenizer,
            logits,
            save_id_in,
            beam_size,
            top_k,
            full_state_tensors,
            linear_state_tensors,
        )
        del (
            single_full_state_tensors,
            single_linear_state_tensors,
            first_state_inputs,
            state_inputs,
        )
        gc.collect()

    print("Export done. Starting ORT runtime.")


# ══════════════════════════════════════════════════════════════════════════════
# ORT Runtime Helpers
# ══════════════════════════════════════════════════════════════════════════════
def bind_ort_in_buf(binding, names, values):
    """Bind a sequence of ORT input buffers by name."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def bind_ort_out_buf(binding, names, values):
    """Bind a sequence of preallocated ORT output buffers by name."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_output(name, value)


def bind_ort_out(binding, names, device):
    """Bind output names while letting ORT allocate storage on `device`."""
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from Python data using the requested device and dtype."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.array(data, dtype=dtype), device, device_id
    )


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the provided shape and dtype."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype), device, device_id
    )


def create_session(
    model_path, _session_opts, _providers, _provider_options, _disabled_optimizers
):
    """Create a standard ORT InferenceSession for one exported model."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers,
    )


def get_in_names(session):
    return [item.name for item in session.get_inputs()]


def get_out_names(session):
    return [item.name for item in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def select_state_input_names(names, prefix, exclude_substrings=()):
    """Filter exported state input names by prefix while excluding sub-patterns."""
    return [
        name
        for name in names
        if name.startswith(prefix)
        and all(token not in name for token in exclude_substrings)
    ]


def create_ort_with_meta_shape(
    meta, dtype, device, device_id, batch_size=1, seq_axis=None, seq_len=0
):
    """Create a zero OrtValue from ORT metadata, resolving symbolic dims locally."""
    shape = list(meta.shape)
    rank = len(shape)

    if seq_axis is not None and seq_axis < 0:
        seq_axis += rank

    for idx, dim in enumerate(shape):
        if idx == 0:
            shape[idx] = batch_size
        elif seq_axis is not None and idx == seq_axis:
            shape[idx] = seq_len
        elif not isinstance(dim, (int, np.integer)):
            shape[idx] = 1

    return create_ort_with_shape(tuple(shape), dtype, device, device_id)


# ══════════════════════════════════════════════════════════════════════════════
# ORT Session & Runtime Options
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()

for options in (session_opts, run_options):
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4

session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session_config_entries = {
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
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for key, value in session_config_entries.items():
    session_opts.add_session_config_entry(key, value)

run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16
    else None
)


# ══════════════════════════════════════════════════════════════════════════════
# Execution Provider Configuration
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '1',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '1',
        'enable_cuda_graph':                  '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(
    _ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID
)
kv_device = "cpu" if device_type == "dml" else device_type


# ══════════════════════════════════════════════════════════════════════════════
# Load ONNX Sessions
# ══════════════════════════════════════════════════════════════════════════════
# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Vision ---
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
binding_Vision     = ort_session_Vision.io_binding()
in_name_Vision     = get_in_names(ort_session_Vision)[0]
out_name_Vision    = get_out_names(ort_session_Vision)[0]
vision_dtype = np.float16 if "float16" in ort_session_Vision._outputs_meta[0].type else np.float32

# --- Concat ---
ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
binding_Concat     = ort_session_Concat.io_binding()
in_name_Concat     = get_in_names(ort_session_Concat)
out_name_Concat    = get_out_names(ort_session_Concat)[0]

# --- Rotary Vision (Prefill + Decode) ---
ort_session_Rotary_Vision_Prefill = create_session(onnx_model_Rotary_Vision_Prefill, **packed_settings)
binding_Rotary_Vision_Prefill  = ort_session_Rotary_Vision_Prefill.io_binding()
in_name_Rotary_Vision_Prefill  = get_in_names(ort_session_Rotary_Vision_Prefill)
out_name_Rotary_Vision_Prefill = get_out_names(ort_session_Rotary_Vision_Prefill)

ort_session_Rotary_Vision_Decode = create_session(onnx_model_Rotary_Vision_Decode, **packed_settings)
binding_Rotary_Vision_Decode  = ort_session_Rotary_Vision_Decode.io_binding()
in_name_Rotary_Vision_Decode  = get_in_names(ort_session_Rotary_Vision_Decode)[0]
out_name_Rotary_Vision_Decode = get_out_names(ort_session_Rotary_Vision_Decode)

# --- Rotary Text (Prefill + Decode) ---
ort_session_Rotary_Text_Prefill = create_session(onnx_model_Rotary_Text_Prefill, **packed_settings)
binding_Rotary_Text_Prefill  = ort_session_Rotary_Text_Prefill.io_binding()
in_name_Rotary_Text_Prefill  = get_in_names(ort_session_Rotary_Text_Prefill)
out_name_Rotary_Text_Prefill = get_out_names(ort_session_Rotary_Text_Prefill)

ort_session_Rotary_Text_Decode = create_session(onnx_model_Rotary_Text_Decode, **packed_settings)
binding_Rotary_Text_Decode  = ort_session_Rotary_Text_Decode.io_binding()
in_name_Rotary_Text_Decode  = get_in_names(ort_session_Rotary_Text_Decode)[0]
out_name_Rotary_Text_Decode = get_out_names(ort_session_Rotary_Text_Decode)
out_meta_Rotary_Text_Decode = ort_session_Rotary_Text_Decode._outputs_meta

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"Usable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Model Metadata & Index Offsets
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main         = get_in_names(ort_session_Main)
out_name_Main        = get_out_names(ort_session_Main)
in_meta_Main         = ort_session_Main._inputs_meta
in_meta_Main_by_name = {name: meta for name, meta in zip(in_name_Main, in_meta_Main)}

num_keys_values_Main        = len(out_name_Main) - 1
num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3

# Main model non-state input indices
# Layout: [state_tensors..., hidden_states, rotary_cos, rotary_sin, attention_mask]
idx_rotary_cos = num_keys_values_Main + 1

# Partitioned name lists
in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
out_name_Main_logits = out_name_Main[num_keys_values_Main]

# State layout derived from exported ONNX input names
in_name_Main_keys = select_state_input_names(in_name_Main_kv, "in_key_", exclude_substrings=("scale", "bias"))
in_name_Main_values = select_state_input_names(in_name_Main_kv, "in_value_", exclude_substrings=("scale", "bias"))
in_name_Main_key_scales   = select_state_input_names(in_name_Main_kv, "in_key_scale_")
in_name_Main_key_biases   = select_state_input_names(in_name_Main_kv, "in_key_bias_")
in_name_Main_value_scales = select_state_input_names(in_name_Main_kv, "in_value_scale_")
in_name_Main_value_biases = select_state_input_names(in_name_Main_kv, "in_value_bias_")
in_name_Main_conv_states  = select_state_input_names(in_name_Main_kv, "in_conv_state_")
in_name_Main_recurrent_states = select_state_input_names(in_name_Main_kv, "in_recurrent_state_")

num_full_layers_Main   = len(in_name_Main_keys)
num_linear_layers_Main = len(in_name_Main_conv_states)

kv_dtype_str = in_meta_Main[0].type
hidden_dtype_Main = np.float16 if "float16" in in_meta_Main[num_keys_values_Main].type else np.float32

vocab_size = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]
vision_embed_size = ort_session_Vision._outputs_meta[0].shape[1]
if not isinstance(vision_embed_size, (int, np.integer)):
    vision_embed_size = VISION_BATCH_SIZE * WIDTH_FACTOR * HEIGHT_FACTOR


# ══════════════════════════════════════════════════════════════════════════════
# State buffer setup
# ══════════════════════════════════════════════════════════════════════════════
if "uint8" in kv_dtype_str or "int8" in kv_dtype_str or "int32" in kv_dtype_str:
    if "int32" in kv_dtype_str:
        kv_dtype_Main = np.int32
    elif "uint8" in kv_dtype_str:
        kv_dtype_Main = np.uint8
    else:
        kv_dtype_Main = np.int8

    if in_name_Main_key_scales:
        scale_dtype_Main = (
            np.float16
            if "float16" in in_meta_Main_by_name[in_name_Main_key_scales[0]].type
            else np.float32
        )
        k_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_scales[0]],
            scale_dtype_Main,
            kv_device,
            DEVICE_ID,
            seq_axis=-1,
        )
        v_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_scales[0]],
            scale_dtype_Main,
            kv_device,
            DEVICE_ID,
            seq_axis=3,
        )
        k_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_key_biases[0]],
                scale_dtype_Main,
                kv_device,
                DEVICE_ID,
                seq_axis=-1,
            )
            if in_name_Main_key_biases
            else None
        )
        v_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_value_biases[0]],
                scale_dtype_Main,
                kv_device,
                DEVICE_ID,
                seq_axis=3,
            )
            if in_name_Main_value_biases
            else None
        )
    else:
        k_scales_Main = None
        k_biases_Main = None
        v_scales_Main = None
        v_biases_Main = None
else:
    kv_dtype_Main = np.float16 if "float16" in kv_dtype_str else np.float32
    k_scales_Main = None
    k_biases_Main = None
    v_scales_Main = None
    v_biases_Main = None

past_keys_Main = create_ort_with_meta_shape(
    in_meta_Main_by_name[in_name_Main_keys[0]],
    kv_dtype_Main,
    kv_device,
    DEVICE_ID,
    seq_axis=-1,
)
past_values_Main = create_ort_with_meta_shape(
    in_meta_Main_by_name[in_name_Main_values[0]],
    kv_dtype_Main,
    kv_device,
    DEVICE_ID,
    seq_axis=3,
)
past_conv_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_conv_states[0]],
        np.float16,
        kv_device,
        DEVICE_ID,
    )
    if in_name_Main_conv_states
    else None
)
past_recurrent_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_recurrent_states[0]],
        np.float16,
        kv_device,
        DEVICE_ID,
    )
    if in_name_Main_recurrent_states
    else None
)


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer and prompt construction
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)

if is_valid_image_path(TEST_IMAGE):
    prompt = build_multimodal_prompt(TEST_QUERY)
else:
    prompt = build_text_prompt(TEST_QUERY)

tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
num_prefill = tokens.shape[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Shared OrtValue buffers
# ══════════════════════════════════════════════════════════════════════════════
input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
topK             = create_ort_with_data([TOP_K],       np.int64, device_type, DEVICE_ID)
beam_size        = create_ort_with_data([BEAM_SIZE],   np.int64, device_type, DEVICE_ID)

attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf = create_ort_with_shape(out_meta_Rotary_Text_Decode[0].shape, hidden_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf = create_ort_with_shape(out_meta_Rotary_Text_Decode[1].shape, hidden_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf = create_ort_with_meta_shape(
    in_meta_Main[num_keys_values_Main],
    hidden_dtype_Main,
    device_type,
    DEVICE_ID,
    batch_size=BEAM_SIZE,
    seq_axis=1,
    seq_len=1,
)
save_id_buf = create_ort_with_shape((BEAM_SIZE, 0), np.int32, device_type, DEVICE_ID)
prefill_logits_buf = create_ort_with_shape((1, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
decode_logits_buf = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
max_idx_buf = create_ort_with_shape((1, 1), np.int32, device_type, DEVICE_ID)

if isinstance(ort_session_Vision._outputs_meta[0].shape[1], int):
    fixed_vision_shape = True
    vision_hidden_states_buf = create_ort_with_shape(ort_session_Vision._outputs_meta[0].shape, vision_dtype, device_type, DEVICE_ID)
    binding_Vision.bind_ortvalue_output(out_name_Vision, vision_hidden_states_buf)
    binding_Concat.bind_ortvalue_input(in_name_Concat[1], vision_hidden_states_buf)
else:
    fixed_vision_shape = False


# ══════════════════════════════════════════════════════════════════════════════
# Decode head sessions
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam         = ort_session_First_Beam.io_binding()
    in_name_First_Beam         = get_in_names(ort_session_First_Beam)
    out_name_First_Beam        = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts   = in_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_parts  = out_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Main_plus_1:]

    ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam         = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts   = in_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_parts  = out_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Main_plus_1:]

    beam_ids_buf = create_ort_with_shape((BEAM_SIZE, 1), np.int32, device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main, device_type, DEVICE_ID)
    bind_ort_in_buf(
        binding_First_Beam,
        in_name_First_Beam[num_keys_values_Main_plus_1:num_keys_values_Main_plus_3],
        [save_id_buf, beam_size],
    )
    bind_ort_in_buf(
        binding_Second_Beam,
        in_name_Second_Beam[num_keys_values_Main_plus_3:],
        [beam_size, topK],
    )
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
    save_id_list       = []


# ══════════════════════════════════════════════════════════════════════════════
# Penalty session
# ══════════════════════════════════════════════════════════════════════════════
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]
    penalty_dtype = (
        np.float16
        if "float16" in ort_session_Penalty._inputs_meta[2].type
        else np.float32
    )
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE], np.int64, device_type, DEVICE_ID)
    bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])


# ══════════════════════════════════════════════════════════════════════════════
# Image validation and vision input preparation
# ══════════════════════════════════════════════════════════════════════════════
if is_valid_image_path(TEST_IMAGE):
    image = Image.open(TEST_IMAGE)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    if len(ort_session_Vision._inputs_meta[0].shape) != 4:
        axis = (0, 1)
    else:
        axis = 0
    pixel_values = np.expand_dims(pixel_values, axis=axis)
    use_vision = True
    print("\nChat with image.")
else:
    pixel_values = None
    use_vision = False
    print("\nChat without image.")


# ══════════════════════════════════════════════════════════════════════════════
# Prefill phase
# ══════════════════════════════════════════════════════════════════════════════
is_prefill_step    = True
prefill_start_time = time.time()
prefill_elapsed    = 0.0
decode_start_time  = prefill_start_time

binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
run(ort_session_Embed, binding_Embed)
hidden_states = binding_Embed.get_outputs()[0]
binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

generate_limit = MAX_SEQ_LEN - num_prefill
if use_vision:
    print("\nStart to Process the Image...")
    vision_start_time = time.time()
    binding_Vision.bind_ortvalue_input(
        in_name_Vision,
        onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID),
    )
    if not fixed_vision_shape:
        bind_ort_out(binding_Vision, [out_name_Vision], _ort_device_type)
    run(ort_session_Vision, binding_Vision)
    print(
        f"\nImage Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds"
    )

    if not fixed_vision_shape:
        vision_hidden_states = binding_Vision.get_outputs()[0]
        binding_Concat.bind_ortvalue_input(in_name_Concat[1], vision_hidden_states)
    num_prefill += vision_embed_size
    ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    generate_limit -= vision_embed_size
    binding_Concat.bind_ortvalue_input(in_name_Concat[0], hidden_states)
    bind_ort_out(binding_Concat, [out_name_Concat], _ort_device_type)
    run(ort_session_Concat, binding_Concat)
    concat_hidden_states = binding_Concat.get_outputs()[0]

    bind_ort_in_buf(
        binding_Rotary_Vision_Prefill,
        in_name_Rotary_Vision_Prefill,
        [ids_len, init_history_len],
    )
    bind_ort_out(binding_Rotary_Vision_Prefill, out_name_Rotary_Vision_Prefill, _ort_device_type)
    run(ort_session_Rotary_Vision_Prefill, binding_Rotary_Vision_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Vision_Prefill.get_outputs()
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], concat_hidden_states)
else:
    bind_ort_in_buf(
        binding_Rotary_Text_Prefill,
        in_name_Rotary_Text_Prefill,
        [ids_len, init_history_len],
    )
    bind_ort_out(binding_Rotary_Text_Prefill, out_name_Rotary_Text_Prefill, _ort_device_type)
    run(ort_session_Rotary_Text_Prefill, binding_Rotary_Text_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Text_Prefill.get_outputs()
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states)

if use_vision:
    binding_Rotary_Vision_Decode.bind_ortvalue_input(in_name_Rotary_Vision_Decode, kv_seq_len)
    bind_ort_out_buf(
        binding_Rotary_Vision_Decode,
        out_name_Rotary_Vision_Decode,
        [rotary_cos_buf, rotary_sin_buf, kv_seq_len],
    )
    ort_session_Rotary_Decode = ort_session_Rotary_Vision_Decode
    binding_Rotary_Decode = binding_Rotary_Vision_Decode
else:
    binding_Rotary_Text_Decode.bind_ortvalue_input(in_name_Rotary_Text_Decode, kv_seq_len)
    bind_ort_out_buf(
        binding_Rotary_Text_Decode,
        out_name_Rotary_Text_Decode,
        [rotary_cos_buf, rotary_sin_buf, kv_seq_len],
    )
    ort_session_Rotary_Decode = ort_session_Rotary_Text_Decode
    binding_Rotary_Decode = binding_Rotary_Text_Decode

bind_ort_in_buf(
    binding_Main,
    in_name_Main[idx_rotary_cos:],
    [rotary_cos, rotary_sin, attention_mask],
)

for name in in_name_Main_keys:
    binding_Main.bind_ortvalue_input(name, past_keys_Main)
for name in in_name_Main_values:
    binding_Main.bind_ortvalue_input(name, past_values_Main)
if k_scales_Main is not None:
    for name in in_name_Main_key_scales:
        binding_Main.bind_ortvalue_input(name, k_scales_Main)
    for name in in_name_Main_value_scales:
        binding_Main.bind_ortvalue_input(name, v_scales_Main)
if k_biases_Main is not None:
    for name in in_name_Main_key_biases:
        binding_Main.bind_ortvalue_input(name, k_biases_Main)
    for name in in_name_Main_value_biases:
        binding_Main.bind_ortvalue_input(name, v_biases_Main)
if past_conv_states_Main is not None:
    for name in in_name_Main_conv_states:
        binding_Main.bind_ortvalue_input(name, past_conv_states_Main)
if past_recurrent_states_Main is not None:
    for name in in_name_Main_recurrent_states:
        binding_Main.bind_ortvalue_input(name, past_recurrent_states_Main)

bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

if USE_PENALTY:
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
    binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

if USE_BEAM_SEARCH:
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_Main], prefill_logits_buf)
elif USE_PENALTY:
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
else:
    binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)


# ══════════════════════════════════════════════════════════════════════════════
# Decode loop
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nTest Question: {TEST_QUERY}\nLLM Answering:")

num_decode = 0
save_id    = None

while num_decode < generate_limit:

    # ── 1. Run Main Model ────────────────────────────────────────────
    run(ort_session_Main, binding_Main)
    outputs_Main = binding_Main.get_outputs()

    # ── 2. Apply Repetition Penalty (if enabled) ─────────────────────
    if USE_PENALTY and num_decode >= PENALTY_RANGE:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        run(ort_session_Penalty, binding_Penalty)

    # ── 3. Token Selection ───────────────────────────────────────────
    if USE_BEAM_SEARCH:

        # ── 3a. Beam Search ──────────────────────────────────────────
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

        max_logits_idx = int(max_idx_buf.numpy().flat[0])
        if max_logits_idx in STOP_TOKEN_SET:
            break

        save_id = outputs_Beam[num_keys_values_Main]
        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_1], save_id)
    else:

        # ── 3b. Greedy / Argmax ──────────────────────────────────────
        if USE_PENALTY:
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            greedy_outputs = binding_Greedy.get_outputs()
            save_id = greedy_outputs[1]
        else:
            run(ort_session_Argmax, binding_Argmax)

        max_logits_idx = int(max_idx_buf.numpy().flat[0])
        if max_logits_idx in STOP_TOKEN_SET:
            break

        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_list.append(max_logits_idx)

        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)
        print(tokenizer.decode(max_logits_idx), end="", flush=True)

    # ── 4. Re-bind Main KV outputs (fresh allocation each step) ──────
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

    # ── 5. Transition: prefill → decode (executes once) ──────────────
    if is_prefill_step:
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states_buf)
        bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
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

    # ── 6. Prepare next step: Embed + Rotary ─────────────────────────
    run(ort_session_Embed, binding_Embed)
    run(ort_session_Rotary_Decode, binding_Rotary_Decode)
    num_decode += 1


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════
decode_end_time = time.time()

if num_decode < 2:
    prefill_elapsed = 0.0
    decode_elapsed = 0.0
else:
    decode_elapsed = decode_end_time - decode_start_time

total_elapsed = decode_end_time - prefill_start_time

prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0


if USE_PENALTY or USE_BEAM_SEARCH:
    result = (
        tokenizer.decode(save_id.numpy().flat[:num_decode], skip_special_tokens=True)
        if save_id is not None
        else ""
    )
else:
    result = tokenizer.decode(save_id_list, skip_special_tokens=True)

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