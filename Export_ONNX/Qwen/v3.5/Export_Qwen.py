"""Export Qwen3.5-VL multimodal (image/video/text) hybrid-attention split ONNX graphs for on-device inference and a self-test decode."""

import gc
import importlib
import json
import os
import subprocess
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

download_path                    = r"/home/DakeQQ/Downloads/Qwen3.5-0.8B"                  # Set the folder path where the Qwen3.5 multimodal model project downloaded.
SCRIPT_DIR                       = Path(__file__).resolve().parent                         # Folder containing this export script.

ONNX_DIR                         = SCRIPT_DIR / "Qwen_ONNX"                                # Exported ONNX graphs are written here, next to this script.
ONNX_DIR.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata              = str(ONNX_DIR / "LLM_Metadata.onnx")
onnx_model_Embed                 = str(ONNX_DIR / "LLM_Embed.onnx")
onnx_model_Vision                = str(ONNX_DIR / "LLM_Vision.onnx")
onnx_model_Image_Preprocess      = str(ONNX_DIR / "LLM_Image_Preprocess.onnx")
onnx_model_Video_Preprocess      = str(ONNX_DIR / "LLM_Video_Preprocess.onnx")
onnx_model_Concat_Image          = str(ONNX_DIR / "LLM_Concat_Image.onnx")
onnx_model_Concat_Video          = str(ONNX_DIR / "LLM_Concat_Video.onnx")
onnx_model_Rotary_Image_Prefill  = str(ONNX_DIR / "LLM_Rotary_Image_Prefill.onnx")
onnx_model_Rotary_Image_Decode   = str(ONNX_DIR / "LLM_Rotary_Image_Decode.onnx")
onnx_model_Rotary_Video_Prefill  = str(ONNX_DIR / "LLM_Rotary_Video_Prefill.onnx")
onnx_model_Rotary_Video_Decode   = str(ONNX_DIR / "LLM_Rotary_Video_Decode.onnx")
onnx_model_Rotary_Text_Prefill   = str(ONNX_DIR / "LLM_RotaryPrefill.onnx")     
onnx_model_Rotary_Text_Decode    = str(ONNX_DIR / "LLM_RotaryDecode.onnx")      
onnx_model_Main                  = str(ONNX_DIR / "LLM_Main.onnx")
onnx_model_Greedy                = str(ONNX_DIR / "LLM_Greedy.onnx")            
onnx_model_First_Beam            = str(ONNX_DIR / "LLM_FirstBeam.onnx")         
onnx_model_Second_Beam           = str(ONNX_DIR / "LLM_SecondBeam.onnx")        
onnx_model_Penalty               = str(ONNX_DIR / "LLM_Penalty.onnx")          
onnx_model_Argmax                = str(ONNX_DIR / "LLM_Argmax.onnx")
onnx_model_KV_Slice              = str(ONNX_DIR / "LLM_KV_Slice.onnx")
onnx_model_KV_Split2             = str(ONNX_DIR / "LLM_KV_Split2.onnx")
onnx_model_KV_Concat             = str(ONNX_DIR / "LLM_KV_Concat.onnx")
onnx_model_Rope_Shift            = str(ONNX_DIR / "LLM_RopeShift.onnx")

# Test Input
TEST_IMAGE                       = [str(SCRIPT_DIR / "psyduck.png")]                      # List of image paths for multi-image support. Use [] for text-only.
TEST_VIDEO                       = str(SCRIPT_DIR / "test_video_8s.mp4")                  # Path to test video file. Leave empty to disable video mode.
TEST_QUERY                       = ["Describe this image.", "Describe this video."]       # [image_query, video_query]
ENABLE_THINKING                  = False                                                  # Enable thinking mode in generation.

# Model Config
DO_EXPORT                        = True                                                   # Whether to export the ONNX models
PREVENT_F16_OVERFLOW             = False                                                  # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
COMPUTE_IN_F32                   = False                                                  # F16-KV compute precision. False (default): run the F16 KV-cache attention and RopeShift in float16 (downcast Q so the growing F16 cache is never upcast). True: upcast the stored F16 cache to float32 at each use for float32 attention. No effect unless KV_QUANT_DTYPE == "F16".
STOP_TOKEN                       = [248044]                                               # Qwen3.5 stop token ids
MAX_SEQ_LEN                      = 4096                                                   # Max context length. Can not edit after export.
ORT_LINEAR_ATTENTION_CHUNK_SIZE  = 128                                                     # Internal chunk size hint for com.microsoft::LinearAttention prefill. This is inside one LLM_Main graph, not runtime subgraph fragmentation.

# Image Vision Config
HEIGHT_FACTOR                    = 25                                                     # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR                     = 25                                                     # Adjust this value to determine the resize shape and vision resolution.
IMAGE_RESIZE                     = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]                # 32 = self.patch_size * self.merge_size
INPUT_IMAGE_SIZE                 = [960, 960]                                             # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
VISION_BATCH_SIZE                = 1                                                      # Maximum number of images supported in multi-image mode.
DYNAMIC_IMAGE_SHAPE              = False                                                  # Allow for a dynamic number of image inputs. (Experiment features, may cause errors)

# Video Vision Config
VIDEO_FPS                        = 2.0                                                    # Frame sampling rate from source video.
VIDEO_MAX_FRAMES                 = 768                                                    # Max frames before temporal patching.
VIDEO_MIN_FRAMES                 = 4                                                      # Min frames.
VIDEO_NUM_FRAMES                 = 8                                                      # Actual frames used. Must be even and divisible by TEMPORAL_PATCH_SIZE.
VIDEO_HEIGHT_FACTOR              = 10                                                     # Video height factor (grid_h = factor * merge_size).
VIDEO_WIDTH_FACTOR               = 18                                                     # Video width factor (grid_w = factor * merge_size).
VIDEO_RESIZE                     = [VIDEO_HEIGHT_FACTOR * 32, VIDEO_WIDTH_FACTOR * 32]    # Target frame spatial size.
INPUT_VIDEO_SIZE                 = [720, 1280]                                            # Input video frame shape (H, W).
DYNAMIC_VIDEO_SHAPE              = False                                                  # Allow dynamic video frame count/spatial.

# KV cache quantization
KV_QUANT_DTYPE                   = "Q8"                                                   # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE              = 256                                                    # Group size for Q4 and Q8 (when USE_HADAMARD or USE_SHUFFLE enabled) per-group quantization. Must divide head_dim evenly.
USE_HADAMARD                     = True                                                   # True = More Accuracy. Apply enhanced randomized Walsh-Hadamard mixing within each group before quantization. Works for Q4 and Q8 modes.
HADAMARD_RANDOM_SEED             = 9527                                                   # Seed for the deterministic Rademacher sign pattern used by the enhanced Hadamard transform.
USE_CLIP                         = True                                                   # Clip outliers to mean ± CLIP_SIGMA*std before quantization. Works for Q4 and Q8 modes.
CLIP_SIGMA                       = 3.0                                                    # Clip threshold in standard deviations. Lower = more aggressive clipping. 2.5-3.5 recommended. Only used when USE_CLIP=True.
USE_SHUFFLE                      = True                                                   # True = More Accuracy. Interleave channels across groups so that high-variance channels are evenly distributed. Works for Q4 and Q8 modes.
USE_SYM                          = False                                                  # True = Less RAM Bandwidth. True: symmetric quantization (no bias, absmax-based); False: asymmetric (min-max with bias). Works for Q4 and Q8 modes.
USE_FLOAT16_SCALE_BIAS           = True                                                   # Whether to use float16 for scale and bias in all quantized KV modes (Q4, Q8, and ROTARY variants).
USE_QDQ_FRIENDLY_ASYM            = False                                                  # Asym only: disables residual bias correction for Q/DQ-friendly blocked rewrites.

# Exact channel reorders copied/adapted from Qwen3 for weight-only/KV quant accuracy.
REORDER_DOWNPROJ_FOR_QUANT       = True                                                   # Exact MLP intermediate-channel reorder; improves quantized down_proj behavior.
REORDER_OPROJ_FOR_QUANT          = False                                                  # Exact full-attention output-channel reorder; may shift quantized-KV accuracy.
REORDER_KEY                      = "absmean"                                              # "absmean" | "L4" | "rms" | "std".
REORDER_VISION_MLP_FOR_QUANT     = True                                                   # Exact vision MLP/merger intermediate-channel reorder.
REORDER_VISION_OPROJ_FOR_QUANT   = False                                                  # Exact vision attention value/o_proj channel reorder.

# Decoding strategy
USE_BEAM_SEARCH                  = False                                                  # Use beam search or greedy search
REPEAT_PENALTY                   = 1.0                                                    # 0.0 ~ 1.0; No penalty = 1.0. Traced into Apply_Penalty; live in the inline runtime.
PENALTY_RANGE                    = 20                                                     # Recent-token window to apply penalty. Traced into Apply_Penalty; live in the inline runtime.
MAX_BEAM_SIZE                    = 10                                                     # Max beam size for beam search. Can not edit after export.
TOP_K                            = 3                                                      # Top-K for beam search. Traced into the beam graphs; live in the inline runtime.
BEAM_SIZE                        = 3                                                      # Beam size for beam search. Must be <= MAX_BEAM_SIZE. Traced (batch) into graphs; live in the inline runtime.

# Runtime config
ORT_LOG                          = False                                                  # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                         = False                                                  # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers         = []                                                     # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                      = 0                                                      # 0 = auto
DEVICE_ID                        = 0                                                      # Device ID for GPU
OPSET                            = 20                                                     # ONNX opset version (matches v3; all multimodal graphs verified to export cleanly at 20)


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
LINEAR_GATED_DELTA_QUERY_SCALE = float(LINEAR_KEY_HEAD_DIM) ** -0.5           # d^-1/2 gated-delta query scale, folded into com.microsoft::LinearAttention scale_f

VISION_HIDDEN_SIZE             = int(VISION_CONFIG["hidden_size"])
VISION_DEPTH                   = int(VISION_CONFIG["depth"])
VISION_NUM_HEADS               = int(VISION_CONFIG["num_heads"])
VISION_HEAD_DIM                = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_PATCH_SIZE              = int(VISION_CONFIG["patch_size"])
VISION_TEMPORAL_PATCH_SIZE     = int(VISION_CONFIG["temporal_patch_size"])
VISION_SPATIAL_MERGE_SIZE      = int(VISION_CONFIG["spatial_merge_size"])
VISION_OUT_HIDDEN_SIZE         = int(VISION_CONFIG["out_hidden_size"])

# Multimodal token IDs
IMAGE_TOKEN_ID                 = int(MODEL_CONFIG.get("image_token_id", 248056))
VIDEO_TOKEN_ID                 = int(MODEL_CONFIG.get("video_token_id", 248057))
VISION_START_TOKEN_ID          = int(MODEL_CONFIG.get("vision_start_token_id", 248053))
VISION_END_TOKEN_ID            = int(MODEL_CONFIG.get("vision_end_token_id", 248054))

# Derived vision constants
IMAGE_SEQLEN_PER_IMAGE         = HEIGHT_FACTOR * WIDTH_FACTOR
VISION_EMBED_SIZE              = IMAGE_SEQLEN_PER_IMAGE * VISION_BATCH_SIZE
TEMPORAL_PATCH_SIZE            = VISION_TEMPORAL_PATCH_SIZE

# Video derived constants
VIDEO_GRID_T                   = VIDEO_NUM_FRAMES // VISION_TEMPORAL_PATCH_SIZE
VIDEO_FRAME_SEQLEN             = VIDEO_HEIGHT_FACTOR * VIDEO_WIDTH_FACTOR
VIDEO_TOTAL_VISION_TOKENS      = VIDEO_GRID_T * VIDEO_FRAME_SEQLEN

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


def build_image_prompt(query: str, num_images: int = 1) -> str:
    """Build a compact multi-image chat prompt for static image concat."""
    vision_placeholders = "<|vision_start|><|vision_end|>" * num_images
    return f"<|im_start|>user\n{vision_placeholders}{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def build_video_timestamps(num_frames: int, fps: float = 2.0, frame_indices=None) -> List[float]:
    """Build deterministic timestamps for each exported video segment."""
    if frame_indices is not None:
        indices = list(frame_indices)
        if not indices:
            return [0.0] * num_frames
        if len(indices) < num_frames * VISION_TEMPORAL_PATCH_SIZE:
            indices.extend([indices[-1]] * (num_frames * VISION_TEMPORAL_PATCH_SIZE - len(indices)))
        timestamps = []
        for frame_index in range(num_frames):
            start = frame_index * VISION_TEMPORAL_PATCH_SIZE
            end = start + VISION_TEMPORAL_PATCH_SIZE
            window = indices[start:end]
            timestamps.append(sum(window) / max(len(window), 1) / max(fps, 1e-6))
        return timestamps

    return [frame_index * VISION_TEMPORAL_PATCH_SIZE / fps for frame_index in range(num_frames)]


def build_video_prompt(
    query: str,
    num_frames: int,
    frame_seqlen: int,
    fps: float = 2.0,
    frame_indices=None,
) -> str:
    """Build a video chat prompt with explicit video_pad placeholder spans."""
    parts = ["<|im_start|>user\n"]
    for timestamp in build_video_timestamps(num_frames, fps, frame_indices):
        parts.append(f"<{timestamp:.1f} seconds>")
        parts.append("<|vision_start|>")
        parts.append("<|video_pad|>" * frame_seqlen)
        parts.append("<|vision_end|>")
    parts.append(f"{query}<|im_end|>\n{build_assistant_prompt_prefix()}")
    return "".join(parts)


def build_mm_token_type_ids(token_ids_flat, mode: str | None = None):
    """Build mm_token_type_ids from token IDs. 0=text, 1=image, 2=video."""
    mm_types = []
    for tid in token_ids_flat:
        tid_int = int(tid)
        if tid_int == IMAGE_TOKEN_ID and mode in (None, "image"):
            mm_types.append(1)
        elif tid_int == VIDEO_TOKEN_ID and mode in (None, "video"):
            mm_types.append(2)
        else:
            mm_types.append(0)
    return mm_types


def build_video_segment_offsets(token_ids_flat, frame_seqlen: int):
    """Find the token index of the first video_pad token in each frame block."""
    offsets = []
    i = 0
    while i < len(token_ids_flat):
        if int(token_ids_flat[i]) == VIDEO_TOKEN_ID:
            offsets.append(i)
            i += frame_seqlen
        else:
            i += 1
    return offsets


def build_prompt_and_mm_types(
    query,
    mode,
    num_frames=0,
    frame_seqlen=0,
    fps=2.0,
    frame_indices=None,
    num_images=1,
    image_seqlen=0,
):
    """Build runtime prompt tokens plus modality metadata for text, image, and video."""
    del image_seqlen

    if mode == "text":
        prompt = build_text_prompt(query)
        tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
        return tokens, [0] * tokens.shape[-1]

    if mode == "image":
        prompt = build_image_prompt(query, num_images)
        tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
        return tokens, [0] * tokens.shape[-1]

    if mode == "video":
        prompt = build_video_prompt(query, num_frames, frame_seqlen, fps, frame_indices)
        tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
        token_ids_flat = tokens[0].tolist()
        mm_token_type_ids = build_mm_token_type_ids(token_ids_flat, "video")
        return tokens, mm_token_type_ids

    raise ValueError(f"Unsupported mode: {mode}")


def compute_prompt_head_len(tokenizer) -> int:
    """Measure the token span that precedes injected vision embeddings."""
    prefix = "<|im_start|>user\n<|vision_start|>"
    tokens = tokenizer(prefix, return_tensors="np")["input_ids"]
    return int(tokens.shape[-1])


def compute_between_len(tokenizer) -> int:
    """Measure the token span between consecutive images."""
    between = "<|vision_end|><|vision_start|>"
    tokens = tokenizer(between, return_tensors="np")["input_ids"]
    return int(tokens.shape[-1])


def build_export_image_prompt(query: str, num_images: int, image_seqlen: int) -> str:
    """Build export-time representative image prompt with expanded image_pad tokens."""
    parts = ["<|im_start|>user\n"]
    for i in range(num_images):
        parts.append("<|vision_start|>")
        parts.append("<|image_pad|>" * image_seqlen)
        parts.append("<|vision_end|>")
    parts.append(f"{query}<|im_end|>\n{build_assistant_prompt_prefix()}")
    return "".join(parts)


def build_export_video_prompt(query: str, num_frames: int, frame_seqlen: int, fps: float = 2.0) -> str:
    """Build export-time representative video prompt (same format as runtime)."""
    return build_video_prompt(query, num_frames, frame_seqlen, fps)


def register_dynamic_causal_mask_buffers(module, max_len: int):
    """Register v3-style causal-mask buffers without materializing a full tril matrix."""
    value_dtype = torch.float32 if max_len > 65504 else torch.float16
    module.register_buffer("mask_row_pos", torch.arange(max_len, dtype=torch.int32).view(max_len, 1), persistent=False)
    module.register_buffer("mask_col_pos", torch.arange(max_len, dtype=torch.int32).view(1, max_len), persistent=False)
    module.register_buffer("mask_zero", torch.tensor(0, dtype=value_dtype), persistent=False)
    module.register_buffer("mask_neg", torch.tensor(-65504, dtype=value_dtype), persistent=False)


def dynamic_causal_attention_mask(module, ids_len, cache_len):
    """Windowed causal mask: `ids_len` new queries attend to `cache_len` cached tokens + themselves.

    `cache_len` is the width of the cached KV window fed to LLM_Main and may be smaller than the
    absolute history length when the runtime uses prefill-window decoupling. Callers that do not
    decouple pass cache_len == history_len, which reproduces the plain causal mask.
    """
    mask_len = cache_len + ids_len
    row_limit = module.mask_row_pos[:ids_len] + cache_len.to(module.mask_row_pos.dtype)
    col_pos = module.mask_col_pos[:, :mask_len]
    attn_bias = torch.where(col_pos <= row_limit, module.mask_zero, module.mask_neg)
    return attn_bias.view(1, 1, 1, ids_len, mask_len).float()



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


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph whose only job is to carry model metadata_props."""

    def forward(self, marker):
        return marker


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Management (full-attention state only)
# ══════════════════════════════════════════════════════════════════════════════
# These graphs are architecture-agnostic pure tensor slicing/splitting/concat and are
# wired to the FULL-attention KV block only (FULL_STATE_SPECS + NUM_FULL_ATTENTION_LAYERS).
# The fixed-size linear-attention conv/recurrent states never pass through them.
class WINDOW_SPLIT_SIZES(torch.autograd.Function):
    """Compute [start, window, tail] split sizes along `dim` for a Split op."""

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
    """Keep the middle window of a 3-way Split along `dim`."""

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
    """Slice full-attention KV cache tensors to keep the [start:end] window."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary     = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
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
    """Compute [split_at, rest] split sizes along `dim` for a 2-way Split op."""

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
    """Split a tensor into (prefix, suffix) at the given point along `dim`."""

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
    """Split full-attention KV cache tensors into prefix/window halves."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized  = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q4  = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary     = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
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
    """Concatenate two full-attention KV cache halves along the sequence axis."""

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

    def forward(self, *all_inputs):
        n = len(all_inputs) // 2
        a = all_inputs[:n]
        b = all_inputs[n:]
        for i in range(self.num_layers):
            self.save_key[i]   = torch.cat([a[i],                   b[i]],                   dim=-1)
            self.save_value[i] = torch.cat([a[i + self.num_layers], b[i + self.num_layers]], dim=-2)
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = torch.cat([a[i + self.num_layers_2], b[i + self.num_layers_2]], dim=-1)
                    self.save_v_scale[i] = torch.cat([a[i + self.num_layers_3], b[i + self.num_layers_3]], dim=self.v_axis)
                else:
                    self.save_k_scale[i] = torch.cat([a[i + self.num_layers_2], b[i + self.num_layers_2]], dim=-1)
                    self.save_k_bias[i]  = torch.cat([a[i + self.num_layers_3], b[i + self.num_layers_3]], dim=-1)
                    self.save_v_scale[i] = torch.cat([a[i + self.num_layers_4], b[i + self.num_layers_4]], dim=self.v_axis)
                    self.save_v_bias[i]  = torch.cat([a[i + self.num_layers_5], b[i + self.num_layers_5]], dim=self.v_axis)
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


def build_mrope_shift_tables(rotary_module, max_shift: int):
    """Build relative mRoPE tables for sliding cached keys left by `shift`."""
    shift_ids = torch.arange(max_shift + 1, dtype=torch.float32).view(1, 1, -1).expand(3, 1, -1)
    inv_freq_expanded = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
    freqs = inv_freq_expanded @ shift_ids
    freqs = freqs.transpose(-1, -2).unsqueeze(1)
    freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).squeeze(0).squeeze(1)
    sin = torch.cat([freqs.sin(), -freqs.sin()], dim=-1).squeeze(0).squeeze(1)
    return cos.half().view(max_shift + 1, 1, 1, ROTARY_DIM, 1), sin.half().view(max_shift + 1, 1, 1, ROTARY_DIM, 1)


class ROPE_SHIFT(torch.nn.Module):
    """Shift cached full-attention keys by a scalar mRoPE position delta.

    `shift` is an mRoPE base-position delta, not necessarily a raw token count.
    For text-only/generated tokens those are identical. For multimodal cache
    eviction, callers should pass the amount by which the mRoPE base position
    moves left after removing the prefix.
    """

    def __init__(self, num_layers, num_kv_heads, rotary_module, max_shift):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.compute_in_f32 = COMPUTE_IN_F32
        cos_shift, sin_shift = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, k, batch_size):
        return (
            k.view(batch_size, self.num_kv_heads, 1, 2, self.rotary_dim_half, -1)
            .flip(-3)
            .view(batch_size, self.num_kv_heads, 1, self.rotary_dim, -1)
        )

    def _shift_key(self, key, cos_tab, sin_tab, batch_size):
        if self.full_rotary:
            return key * cos_tab + self._flip_k(key, batch_size) * sin_tab
        key_rot, key_pass = torch.split(key, [self.rotary_dim, HEAD_DIM - self.rotary_dim], dim=-2)
        key_rot = key_rot * cos_tab + self._flip_k(key_rot, batch_size) * sin_tab
        return torch.cat([key_rot, key_pass], dim=-2)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        kv_dtype = all_inputs[0].dtype
        # compute_in_f32 upcasts the stored F16 keys to float32 for the shift, then stores back as F16.
        force_f32 = self.compute_in_f32 and kv_dtype != torch.float32
        cos_tab = self.cos_shift.index_select(0, shift).squeeze(0)
        sin_tab = self.sin_shift.index_select(0, shift).squeeze(0)
        if kv_dtype == torch.float32 or force_f32:
            cos_tab = cos_tab.float()
            sin_tab = sin_tab.float()

        batch_size = all_inputs[0].shape[0]
        outputs = []
        for layer_index in range(self.num_layers):
            key = all_inputs[layer_index]
            if force_f32:
                key = key.float()
            shifted = self._shift_key(key, cos_tab, sin_tab, batch_size)
            if force_f32:
                shifted = shifted.to(kv_dtype)
            outputs.append(shifted)
        return tuple(outputs)


class ROPE_SHIFT_QUANT(torch.nn.Module):
    """Dequantize, mRoPE-shift, and re-quantize cached full-attention keys."""

    def __init__(self, num_layers, num_kv_heads, rotary_module, max_shift, quantizer, is_asym):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.quantizer = quantizer
        self.is_asym = is_asym
        cos_shift, sin_shift = build_mrope_shift_tables(rotary_module, max_shift)
        self.register_buffer("cos_shift", cos_shift, persistent=False)
        self.register_buffer("sin_shift", sin_shift, persistent=False)

    def _flip_k(self, k, batch_size):
        return (
            k.view(batch_size, self.num_kv_heads, 1, 2, self.rotary_dim_half, -1)
            .flip(-3)
            .view(batch_size, self.num_kv_heads, 1, self.rotary_dim, -1)
        )

    def _shift_key(self, key, cos_tab, sin_tab, batch_size):
        if self.full_rotary:
            return key * cos_tab + self._flip_k(key, batch_size) * sin_tab
        key_rot, key_pass = torch.split(key, [self.rotary_dim, HEAD_DIM - self.rotary_dim], dim=-2)
        key_rot = key_rot * cos_tab + self._flip_k(key_rot, batch_size) * sin_tab
        return torch.cat([key_rot, key_pass], dim=-2)

    def forward(self, *all_inputs):
        shift = all_inputs[-1].reshape(-1)
        cos_tab = self.cos_shift.index_select(0, shift).squeeze(0).float()
        sin_tab = self.sin_shift.index_select(0, shift).squeeze(0).float()

        layers = self.num_layers
        keys_in = all_inputs[:layers]
        scales_in = all_inputs[layers:2 * layers]
        biases_in = all_inputs[2 * layers:3 * layers] if self.is_asym else None

        batch_size = all_inputs[0].shape[0]
        out_keys, out_scales, out_biases = [], [], []
        for layer_index in range(layers):
            key_bias = biases_in[layer_index] if self.is_asym else None
            key = self.quantizer.dequantize_key(keys_in[layer_index], scales_in[layer_index], key_bias, batch_size)
            key = self._shift_key(key, cos_tab, sin_tab, batch_size)
            new_key, new_scale, new_bias = self.quantizer.quantize_key(key, batch_size)
            out_keys.append(new_key)
            out_scales.append(new_scale)
            if self.is_asym:
                out_biases.append(new_bias)
        if self.is_asym:
            return (*out_keys, *out_scales, *out_biases)
        return (*out_keys, *out_scales)


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
        self.use_residual_bias_correction = (not use_sym) and not USE_QDQ_FRIENDLY_ASYM
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

    def inverse_hadamard_k(self, k, batch_size):
        k = k.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
            -1,
        )
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2), inverse=True).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

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

    def quantize_key(self, keys, batch_size):
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)

        if self.use_sym:
            key_packed, key_scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            key_bias = None
        else:
            key_packed, key_scale, key_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)

        if self.is_q4:
            key_packed = self.pack_q4_k(key_packed, batch_size)
        if self.is_q8_cuda:
            packed_head_dim = (self.head_dim // 8) if self.is_q4 else (self.head_dim // 4)
            key_packed = self.pack_cuda(key_packed, -2, batch_size, self.num_kv_heads, packed_head_dim)
        return key_packed, key_scale, key_bias

    def dequantize_key(self, packed_key, key_scale, key_bias, batch_size):
        if USE_FLOAT16_SCALE_BIAS:
            key_scale = key_scale.float()
            if key_bias is not None:
                key_bias = key_bias.float()
        if self.is_q8_cuda:
            unpack_head_dim = (self.head_dim // 2) if self.is_q4 else self.head_dim
            packed_key = self.unpack_cuda(packed_key, -2, batch_size, self.num_kv_heads, unpack_head_dim)
        if self.is_q4:
            key_int = self.unpack_q4_k(packed_key, batch_size)
            if self.use_sym:
                key_int = self._decode_signed_q4_storage(key_int)
        else:
            key_int = self._decode_signed_q8_storage(packed_key) if self.use_sym else packed_key
        key_float = key_int.float()
        if self.is_grouped:
            key_grouped = key_float.view(
                batch_size,
                self.num_kv_heads,
                1,
                self.kv_quant_num_groups,
                self.kv_quant_group_size,
                -1,
            )
            keys = key_grouped * key_scale if self.use_sym else key_grouped * key_scale + key_bias
            keys = keys.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        else:
            keys = key_float * key_scale if self.use_sym else key_float * key_scale + key_bias
        if self.use_hadamard:
            keys = self.inverse_hadamard_k(keys, batch_size)
        if self.use_shuffle:
            keys = keys.index_select(3, self.unshuffle_idx)
        if self.is_rotary:
            keys = self.inverse_rotate_k(keys, batch_size)
        return keys

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


def load_image_letterbox(path, target_h, target_w):
    """Load one image into a fixed-size canvas without stretching its aspect ratio."""
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        src_w, src_h = image.size
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        resize_w = max(1, min(target_w, int(round(src_w * scale))))
        resize_h = max(1, min(target_h, int(round(src_h * scale))))
        if image.size != (resize_w, resize_h):
            image = image.resize((resize_w, resize_h), resampling)

        # Use a neutral gray canvas so padding stays near zero after fused 127.5-based normalization.
        canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
        offset_x = (target_w - resize_w) // 2
        offset_y = (target_h - resize_h) // 2
        canvas.paste(image, (offset_x, offset_y))

    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1))

def sample_video_frames(
    video_path,
    target_fps,
    num_frames,
    min_frames,
    max_frames,
    frame_size,
):
    """Sample frames at the configured FPS, then clamp or pad to the export contract."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video mode requires opencv-python (cv2). Install with: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Unable to determine frame count for video: {video_path}")
        if src_fps <= 0:
            src_fps = float(target_fps)

        frame_step = max(src_fps / max(float(target_fps), 1e-6), 1.0)
        candidate_indices = np.arange(0, total_frames, frame_step, dtype=np.float64).astype(np.int64)
        if candidate_indices.size == 0:
            candidate_indices = np.array([0], dtype=np.int64)
        if candidate_indices.size < min_frames:
            candidate_indices = np.linspace(0, total_frames - 1, min_frames, dtype=np.int64)
        if candidate_indices.size > max_frames:
            candidate_indices = candidate_indices[:max_frames]

        if candidate_indices.size >= num_frames:
            sampled_indices = candidate_indices[
                np.linspace(0, candidate_indices.size - 1, num_frames, dtype=np.int64)
            ]
        else:
            sampled_indices = np.concatenate(
                [
                    candidate_indices,
                    np.full(num_frames - candidate_indices.size, candidate_indices[-1], dtype=np.int64),
                ]
            )

        frames = []
        for frame_index in sampled_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            frames.append(np.transpose(frame.astype(np.uint8), (2, 0, 1)))

        if not frames:
            raise RuntimeError(f"Unable to decode any frames from video: {video_path}")
        while len(frames) < num_frames:
            frames.append(frames[-1])

        return np.stack(frames[:num_frames], axis=0), sampled_indices.tolist()
    finally:
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# ORT official LinearAttention helper for linear-attention recurrence
# ══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime 1.27 exposes com.microsoft::LinearAttention with update_rule=
# "gated_delta". Its schema exactly matches the canonical Qwen3.5 recurrence:
# S_t = exp(g_t) * S_{t-1} + beta_t * k_t ⊗ (v_t - exp(g_t) * S_{t-1}^T k_t),
# o_t = scale * q_t^T S_t. The gated-delta d^-1/2 query scale is folded into the
# op's scale_f attribute (q is only read in the output), replacing the hand-built
# sequential/triangular recurrence subgraph with one official ORT CUDA op.
# ══════════════════════════════════════════════════════════════════════════════
class ORT_LINEAR_ATTENTION(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, initial_state, g, beta):
        output = query.new_zeros(query.shape[0], query.shape[1], LINEAR_NUM_KEY_HEADS * LINEAR_VALUE_HEAD_DIM)
        state = initial_state.clone()
        return output, state

    @staticmethod
    def symbolic(g, query, key, value, initial_state, decay, beta):
        return g.op(
            "com.microsoft::LinearAttention",
            query,
            key,
            value,
            initial_state,
            decay,
            beta,
            q_num_heads_i=LINEAR_NUM_KEY_HEADS,
            kv_num_heads_i=LINEAR_NUM_VALUE_HEADS,
            update_rule_s="gated_delta",
            scale_f=LINEAR_GATED_DELTA_QUERY_SCALE,
            chunk_size_i=ORT_LINEAR_ATTENTION_CHUNK_SIZE,
            outputs=2,
        )


def recurrent_gated_delta_prefill(query, key, value, g, beta, initial_state):
    batch_size = query.shape[0]
    query_packed = query.reshape(batch_size, -1, LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM)
    key_packed = key.reshape(batch_size, -1, LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM)
    value_packed = value.reshape(batch_size, -1, LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM)
    output_packed, state = ORT_LINEAR_ATTENTION.apply(query_packed, key_packed, value_packed, initial_state, g, beta)
    output = output_packed.reshape(batch_size, -1, LINEAR_NUM_KEY_HEADS, LINEAR_VALUE_HEAD_DIM)
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


class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Preprocess N images into packed patches ready for the shared vision encoder.
    Accepts: pixel_values [N, 1, 3, H, W] or [N, 3, H, W] uint8/float (0-255 range)
    Returns: (pixel_values_patches, pos_embeds, rotary_cos, rotary_sin, attention_mask)
    """

    def __init__(self, visual, height_factor, width_factor, num_images, dynamic_shape=False):
        super().__init__()
        self.patch_size = VISION_PATCH_SIZE
        self.merge_size = VISION_SPATIAL_MERGE_SIZE
        self.temporal_patch_size = VISION_TEMPORAL_PATCH_SIZE // 2
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.num_images = num_images
        self.dynamic_shape = dynamic_shape
        self.target_h = height_factor * self.patch_size * self.merge_size
        self.target_w = width_factor * self.patch_size * self.merge_size
        self.grid_h = height_factor * self.merge_size
        self.grid_w = width_factor * self.merge_size
        self.seq_per_image = self.grid_h * self.grid_w

        # Precompute position embeddings for all images
        grid_thw = torch.tensor([[1, self.grid_h, self.grid_w]], dtype=torch.int32)
        single_pos = visual.fast_pos_embed_interpolate(grid_thw)  # [seq_per_image, embed_dim]
        # Tile for num_images
        pos_embeds = single_pos.unsqueeze(0).repeat(1, num_images, 1)
        self.register_buffer("pos_embeds", pos_embeds.half())

        # Precompute rotary for vision encoder
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).float()  # [seq_per_image, head_dim/2]
        # Tile for num_images (each image gets same rotary)
        rotary_pos_emb = rotary_pos_emb.repeat(num_images, 1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.register_buffer("rotary_cos", torch.cat([cos, cos], dim=-1).half())
        self.register_buffer("rotary_sin", torch.cat([-sin, sin], dim=-1).half())

        # Block-diagonal attention mask: images don't attend to each other
        total_seq = self.seq_per_image * num_images
        mask = torch.zeros(1, 1, total_seq, total_seq, dtype=torch.int8)
        for img_idx in range(num_images):
            start = img_idx * self.seq_per_image
            end = start + self.seq_per_image
            mask[..., start:end, :start] = -128
            mask[..., start:end, end:] = -128
        self.register_buffer("attention_mask", mask)

    def forward(self, pixel_values):
        # Handle 5D input [N, 1, 3, H, W]
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        # pixel_values: [N, 3, H, W]
        batch_size = pixel_values.shape[0] if self.dynamic_shape else self.num_images
        pixel_values = pixel_values.float()

        # Resize if needed
        if self.dynamic_shape or list(pixel_values.shape[-2:]) != [self.target_h, self.target_w]:
            pixel_values = F.interpolate(
                pixel_values, size=[self.target_h, self.target_w],
                mode="bilinear", align_corners=False
            )

        # Spatial patching: [N, 3, H, W] -> patches
        pixel_values = pixel_values.reshape(
            batch_size, 3,
            self.height_factor, self.merge_size, self.patch_size,
            self.width_factor, self.merge_size, self.patch_size,
        )
        pixel_values = pixel_values.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # [N, hf, wf, merge, merge, 3, ps, ps]
        pixel_values = pixel_values.reshape(-1, 3, self.temporal_patch_size, self.patch_size, self.patch_size)
        # Duplicate temporal dimension for images (temporal_patch_size=2, single frame duplicated)
        pixel_values = torch.cat([pixel_values, pixel_values], dim=2)
        # [total_patches, 3, tps, ps, ps]

        if self.dynamic_shape:
            seq_len = batch_size * self.seq_per_image
            pos_embeds = self.pos_embeds[:, :seq_len]
            rotary_cos = self.rotary_cos[..., :seq_len]
            rotary_sin = self.rotary_sin[..., :seq_len]
            attention_mask = self.attention_mask[..., :seq_len, :seq_len]
        else:
            pos_embeds = self.pos_embeds
            rotary_cos = self.rotary_cos
            rotary_sin = self.rotary_sin
            attention_mask = self.attention_mask

        return pixel_values, pos_embeds, rotary_cos, rotary_sin, attention_mask


class LLM_VIDEO_PREPROCESS(torch.nn.Module):
    """Preprocess video frames into packed patches ready for the shared vision encoder.
    Accepts: video_frames [num_frames, 3, H, W] uint8/float (0-255 range)
    Returns: (pixel_values_patches, pos_embeds, rotary_cos, rotary_sin, attention_mask)
    """

    def __init__(self, visual, num_frames, height_factor, width_factor, dynamic_shape=False):
        super().__init__()
        self.patch_size = VISION_PATCH_SIZE
        self.merge_size = VISION_SPATIAL_MERGE_SIZE
        self.temporal_patch_size = VISION_TEMPORAL_PATCH_SIZE
        self.num_frames = num_frames
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.dynamic_shape = dynamic_shape
        self.target_h = height_factor * self.patch_size * self.merge_size
        self.target_w = width_factor * self.patch_size * self.merge_size
        self.grid_h = height_factor * self.merge_size
        self.grid_w = width_factor * self.merge_size
        self.grid_t = num_frames // self.temporal_patch_size
        self.frame_seqlen = self.grid_h * self.grid_w
        self.total_seq = self.grid_t * self.frame_seqlen

        # Precompute position embeddings for video
        grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], dtype=torch.int32)
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw).unsqueeze(0)
        self.register_buffer("pos_embeds", pos_embeds.half())

        # Precompute rotary for vision encoder
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).float()
        rotary_pos_emb = rotary_pos_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.register_buffer("rotary_cos", torch.cat([cos, cos], dim=-1).half())
        self.register_buffer("rotary_sin", torch.cat([-sin, sin], dim=-1).half())

        # Block-diagonal attention mask: each temporal group attends only to itself
        mask = torch.zeros(1, 1, self.total_seq, self.total_seq, dtype=torch.int8)
        for t in range(self.grid_t):
            start = t * self.frame_seqlen
            end = start + self.frame_seqlen
            mask[:, :, start:end, :start] = -128
            mask[:, :, start:end, end:] = -128
        self.register_buffer("attention_mask", mask)

    def forward(self, video_frames):
        # video_frames: [num_frames, 3, H, W]
        video_frames = video_frames.float()

        # Resize if needed
        if list(video_frames.shape[-2:]) != [self.target_h, self.target_w]:
            video_frames = F.interpolate(
                video_frames, size=[self.target_h, self.target_w],
                mode="bilinear", align_corners=False
            )

        # Compute actual grid_t for dynamic shape
        grid_t = video_frames.shape[0] // self.temporal_patch_size if self.dynamic_shape else self.grid_t

        # Spatial decomposition
        # Must keep the dims <= 8 or ONNX RUntime-CUDA will error out
        video_frames = video_frames.reshape(
            grid_t, self.temporal_patch_size * 3,
            self.height_factor, self.merge_size, self.patch_size,
            self.width_factor, self.merge_size, self.patch_size,
        )
        # Permute: [grid_t, hf, wf, merge, merge, tps, 3, ps, ps]
        video_frames = video_frames.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # Reshape to patches: [total_patches, 3, tps, ps, ps]
        video_frames = video_frames.reshape(-1, self.temporal_patch_size, 3, self.patch_size, self.patch_size).transpose(1, 2)

        if self.dynamic_shape:
            total_seq = grid_t * self.frame_seqlen
            pos_embeds = self.pos_embeds[:, :total_seq]
            rotary_cos = self.rotary_cos[..., :total_seq]
            rotary_sin = self.rotary_sin[..., :total_seq]
            attention_mask = self.attention_mask[..., :total_seq, :total_seq]
        else:
            pos_embeds = self.pos_embeds
            rotary_cos = self.rotary_cos
            rotary_sin = self.rotary_sin
            attention_mask = self.attention_mask

        return video_frames, pos_embeds, rotary_cos, rotary_sin, attention_mask


class LLM_VISION(torch.nn.Module):
    """Shared vision encoder for both image and video inputs.
    Inputs:
        pixel_values: [total_patches, 3, temporal_patch_size, patch_size, patch_size]
        pos_embeds: [1, seq_len, embed_dim]
        rotary_cos: [1, 1, 1, seq_len, head_dim*2]
        rotary_sin: [1, 1, 1, seq_len, head_dim*2]
        attention_mask: [1, 1, seq_len, seq_len]
    Returns:
        vision_hidden_states: [1, seq_len, out_hidden_size]
    """

    def __init__(self, llm):
        super().__init__()
        self.visual = llm.model.visual
        replace_gelu_with_tanh_approximation(self.visual)

        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HEAD_DIM
        self.head_dim_half = self.head_dim // 2
        self.patch_size = VISION_PATCH_SIZE
        self.merge_size = VISION_SPATIAL_MERGE_SIZE

        # Fuse normalization into Conv3d weights and bias
        norm_scale = 1.0 / 127.5
        scaling = self.head_dim**-0.25
        with torch.no_grad():
            patch_embed = self.visual.patch_embed
            # Fuse pixel normalization (divide by 127.5) into Conv3d weights
            patch_embed.proj.weight.data.mul_(norm_scale)
            # Fuse mean subtraction (127.5) into Conv3d bias
            if patch_embed.proj.bias is not None:
                patch_embed.proj.bias.data -= 127.5 * patch_embed.proj.weight.data.flatten(1).sum(dim=1)
            else:
                bias_val = -127.5 * patch_embed.proj.weight.data.flatten(1).sum(dim=1)
                patch_embed.proj.bias = torch.nn.Parameter(bias_val)

            for block in self.visual.blocks:
                qk_out = block.attn.qkv.out_features - self.visual.patch_embed.embed_dim
                block.attn.qkv.weight.data[:qk_out].mul_(scaling)
                block.attn.qkv.bias.data[:qk_out].mul_(scaling)
                self.fuse_norm(block.norm1, block.attn.qkv)
                self.fuse_norm(block.norm2, block.mlp.linear_fc1)
            self.fuse_norm(self.visual.merger.norm, self.visual.merger.linear_fc1)
            if REORDER_VISION_MLP_FOR_QUANT:
                self.reorder_mlp_for_quant(REORDER_KEY)
            if REORDER_VISION_OPROJ_FOR_QUANT:
                self.reorder_oproj_for_quant(REORDER_KEY)

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

    @staticmethod
    def _channel_perm(weight, key: str):
        abs_weight = weight.abs()
        if key == "rms":
            stat = (weight * weight).mean(0).sqrt()
        elif key == "L4":
            stat = abs_weight.pow(4).mean(0).pow(0.25)
        elif key == "std":
            stat = weight.std(0)
        else:
            stat = abs_weight.mean(0)
        return torch.argsort(stat)

    @staticmethod
    def _reorder_mlp_pair(linear_fc1, linear_fc2, key: str):
        perm = LLM_VISION._channel_perm(linear_fc2.weight.data, key)
        linear_fc1.weight.data.copy_(linear_fc1.weight.data[perm])
        if linear_fc1.bias is not None:
            linear_fc1.bias.data.copy_(linear_fc1.bias.data[perm])
        linear_fc2.weight.data.copy_(linear_fc2.weight.data[:, perm])

    def reorder_mlp_for_quant(self, key: str):
        for block in self.visual.blocks:
            self._reorder_mlp_pair(block.mlp.linear_fc1, block.mlp.linear_fc2, key)
        self._reorder_mlp_pair(self.visual.merger.linear_fc1, self.visual.merger.linear_fc2, key)

    def reorder_oproj_for_quant(self, key: str):
        for block in self.visual.blocks:
            out_proj_weight = block.attn.proj.weight
            out_proj_by_head = out_proj_weight.view(out_proj_weight.shape[0], self.num_heads, self.head_dim)
            qkv_weight = block.attn.qkv.weight
            qkv_by_group = qkv_weight.view(3, self.num_heads, self.head_dim, qkv_weight.shape[1]).clone()
            qkv_bias = block.attn.qkv.bias
            qkv_bias_by_group = qkv_bias.view(3, self.num_heads, self.head_dim).clone() if qkv_bias is not None else None
            reordered_out = out_proj_by_head.clone()
            for head in range(self.num_heads):
                perm = self._channel_perm(out_proj_by_head[:, head, :], key)
                reordered_out[:, head, :] = reordered_out[:, head, perm]
                qkv_by_group[2, head] = qkv_by_group[2, head][perm]
                if qkv_bias_by_group is not None:
                    qkv_bias_by_group[2, head] = qkv_bias_by_group[2, head][perm]
            out_proj_weight.data.copy_(reordered_out.reshape(out_proj_weight.shape[0], self.num_heads * self.head_dim))
            qkv_weight.data.copy_(qkv_by_group.reshape(qkv_weight.shape[0], qkv_weight.shape[1]))
            if qkv_bias is not None:
                qkv_bias.data.copy_(qkv_bias_by_group.reshape(qkv_bias.shape[0]))

    def rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def forward(self, pixel_values, pos_embeds, rotary_cos, rotary_sin, attention_mask):
        batch_size = 1  # Vision encoder always processes as batch=1

        # Cast float16 inputs back to float32
        pos_embeds = pos_embeds.float()
        rotary_cos = rotary_cos.float()
        rotary_sin = rotary_sin.float()
        attention_mask = attention_mask.float()

        # Conv3d patch embedding (normalization already fused into weights/bias)
        vision_hidden_states = self.visual.patch_embed.proj(pixel_values)
        vision_hidden_states = vision_hidden_states.view(1, -1, self.visual.patch_embed.embed_dim)

        # Add position embeddings
        vision_hidden_states = vision_hidden_states + pos_embeds

        # Transformer blocks
        for block in self.visual.blocks:
            hidden_states_norm = block.norm1(vision_hidden_states)
            qkv = block.attn.qkv(hidden_states_norm)
            qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            qk, value = qkv.split([2, 1], dim=0)
            qk = qk * rotary_cos + self.rotate_half(qk, batch_size) * rotary_sin
            query, key = qk.split([1, 1], dim=0)
            attn = torch.matmul(query, key.transpose(-1, -2))
            attn = attn + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, value)
            attn = attn.transpose(2, 3).reshape(batch_size, -1, block.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + block.attn.proj(attn)

            mlp_out = block.mlp.linear_fc1(block.norm2(vision_hidden_states))
            mlp_out = block.mlp.act_fn(mlp_out)
            mlp_out = block.mlp.linear_fc2(mlp_out)
            vision_hidden_states = vision_hidden_states + mlp_out

        # Merger
        vision_hidden_states = self.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.visual.merger.hidden_size)
        vision_hidden_states = self.visual.merger.linear_fc1(vision_hidden_states)
        vision_hidden_states = self.visual.merger.act_fn(vision_hidden_states)
        vision_hidden_states = self.visual.merger.linear_fc2(vision_hidden_states)
        return vision_hidden_states


class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Insert image vision embeddings into text embeddings at known positions.
    Layout: [head] [vis_0] [between] [vis_1] ... [vis_N-1] [tail]
    """

    def __init__(self, num_images: int, image_seqlen: int, prompt_head_len: int = 4, between_len: int = 2):
        super().__init__()
        self.num_images = num_images
        self.image_seqlen = image_seqlen
        self.prompt_head_len = prompt_head_len
        self.between_len = between_len
        # Pre-compute vision slice boundaries per image
        self._vis_starts = tuple(i * image_seqlen for i in range(num_images))
        self._vis_ends = tuple(i * image_seqlen + image_seqlen for i in range(num_images))
        # Pre-compute text between-slice boundaries
        self._text_starts = tuple(prompt_head_len + i * between_len for i in range(max(num_images - 1, 0)))
        self._text_ends = tuple(prompt_head_len + i * between_len + between_len for i in range(max(num_images - 1, 0)))
        # Pre-compute tail start position
        self._tail_start = prompt_head_len + max(num_images - 1, 0) * between_len

    def forward(self, text_hidden_states, vision_hidden_states):
        pieces = [text_hidden_states[:, :self.prompt_head_len]]
        for img_idx in range(self.num_images):
            pieces.append(vision_hidden_states[:, self._vis_starts[img_idx]:self._vis_ends[img_idx]])
            if img_idx < self.num_images - 1:
                pieces.append(text_hidden_states[:, self._text_starts[img_idx]:self._text_ends[img_idx]])
        pieces.append(text_hidden_states[:, self._tail_start:])
        return torch.cat(pieces, dim=1)


class LLM_CONCAT_VIDEO(torch.nn.Module):
    """Replace fixed exported video_pad placeholder spans with vision features.

    The exported video prompt places all frame placeholders before the free-form
    text tail, so the insertion offsets are deterministic for a given export
    configuration. Baking those offsets into the module removes a traced tensor
    input and avoids Python branching on tensor values during export.
    """

    def __init__(self, segment_offsets: List[int], frame_seqlen: int):
        super().__init__()
        self.segment_offsets = tuple(int(offset) for offset in segment_offsets)
        self.num_temporal_groups = len(self.segment_offsets)
        self.frame_seqlen = frame_seqlen
        # Pre-compute vision slice boundaries per temporal group
        self._vis_starts = tuple(f * frame_seqlen for f in range(self.num_temporal_groups))
        self._vis_ends = tuple(f * frame_seqlen + frame_seqlen for f in range(self.num_temporal_groups))
        # Pre-compute text cursor positions after each frame insertion
        self._text_cursors = tuple(offset + frame_seqlen for offset in self.segment_offsets)

    def forward(self, text_hidden_states, vision_hidden_states):
        pieces = []
        text_cursor = 0
        for f in range(self.num_temporal_groups):
            offset = self.segment_offsets[f]
            if offset > text_cursor:
                pieces.append(text_hidden_states[:, text_cursor:offset])
            pieces.append(vision_hidden_states[:, self._vis_starts[f]:self._vis_ends[f]])
            text_cursor = self._text_cursors[f]
        # Remaining text after all frames. A zero-length tail is safe and keeps
        # the exported path shape-driven instead of branch-driven.
        pieces.append(text_hidden_states[:, text_cursor:])
        return torch.cat(pieces, dim=1)


class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for image prefill."""

    def __init__(self, llm, tokenizer, num_images, image_seqlen, max_seq_len):
        super().__init__()
        # Build export-time representative prompt
        query = "Q"
        export_prompt = build_export_image_prompt(query, num_images, image_seqlen)
        token_ids = tokenizer(export_prompt, return_tensors="np")["input_ids"][0]
        mm_types = build_mm_token_type_ids(token_ids, "image")

        cos, sin = self._build_rotary_table(llm, mm_types, num_images, image_seqlen, max_seq_len)
        total_max = len(mm_types) + max_seq_len
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))
        register_dynamic_causal_mask_buffers(self, total_max)

    @staticmethod
    def _build_rotary_table(llm, mm_types, num_images, image_seqlen, max_seq_len):
        """Build 3-channel position IDs following upstream get_rope_index semantics."""
        seq_len = len(mm_types)
        merge_size = VISION_SPATIAL_MERGE_SIZE
        grid_h = HEIGHT_FACTOR  # after merge
        grid_w = WIDTH_FACTOR   # after merge

        # Build position_ids [3, 1, total_len]
        total_len = seq_len + max_seq_len
        position_ids = torch.zeros(3, 1, total_len, dtype=torch.float32)

        current_pos = 0
        i = 0
        while i < seq_len:
            if mm_types[i] == 0:
                # Text token
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1
            elif mm_types[i] == 1:
                # Image segment
                img_start = i
                img_end = img_start + image_seqlen
                # Temporal channel: constant
                position_ids[0, 0, img_start:img_end] = current_pos
                # Height channel
                h_positions = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w) + current_pos
                position_ids[1, 0, img_start:img_end] = h_positions
                # Width channel
                w_positions = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h) + current_pos
                position_ids[2, 0, img_start:img_end] = w_positions
                current_pos += max(grid_h, grid_w)
                i = img_end
            else:
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1

        # Fill decode tail
        tail_positions = torch.arange(max_seq_len, dtype=torch.float32) + current_pos
        position_ids[:, 0, seq_len:] = tail_positions

        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = inv_freq_expanded @ position_ids
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        # Vision prefill has no window-decoupling yet: window width == absolute history length.
        attention_mask = dynamic_causal_attention_mask(self, ids_len, history_len)
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for one decode step with image context."""

    def __init__(self, llm, tokenizer, num_images, image_seqlen, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_IMAGE_PREFILL._build_rotary_table(
            llm, build_mm_token_type_ids(
                tokenizer(build_export_image_prompt("Q", num_images, image_seqlen), return_tensors="np")["input_ids"][0],
                "image"
            ), num_images, image_seqlen, max_seq_len
        )
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class ROTARY_VIDEO_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for video prefill."""

    def __init__(self, llm, tokenizer, num_frames, frame_seqlen, max_seq_len, fps=2.0):
        super().__init__()
        query = "Q"
        export_prompt = build_export_video_prompt(query, num_frames, frame_seqlen, fps)
        token_ids = tokenizer(export_prompt, return_tensors="np")["input_ids"][0]
        mm_types = build_mm_token_type_ids(token_ids, "video")

        cos, sin = self._build_rotary_table(llm, mm_types, num_frames, frame_seqlen, max_seq_len)
        total_max = len(mm_types) + max_seq_len
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))
        register_dynamic_causal_mask_buffers(self, total_max)

    @staticmethod
    def _build_rotary_table(llm, mm_types, num_frames, frame_seqlen, max_seq_len):
        """Build 3-channel position IDs for video following upstream get_rope_index."""
        seq_len = len(mm_types)
        merge_size = VISION_SPATIAL_MERGE_SIZE
        grid_h = VIDEO_HEIGHT_FACTOR  # after merge
        grid_w = VIDEO_WIDTH_FACTOR   # after merge

        total_len = seq_len + max_seq_len
        position_ids = torch.zeros(3, 1, total_len, dtype=torch.float32)

        current_pos = 0
        i = 0
        while i < seq_len:
            if mm_types[i] == 0:
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1
            elif mm_types[i] == 2:
                # Video frame segment
                frame_start = i
                frame_end = frame_start + frame_seqlen
                # Temporal channel: constant for this frame
                position_ids[0, 0, frame_start:frame_end] = current_pos
                # Height channel
                h_positions = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w) + current_pos
                position_ids[1, 0, frame_start:frame_end] = h_positions
                # Width channel
                w_positions = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h) + current_pos
                position_ids[2, 0, frame_start:frame_end] = w_positions
                current_pos += max(grid_h, grid_w)
                i = frame_end
            else:
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1

        # Fill decode tail
        tail_positions = torch.arange(max_seq_len, dtype=torch.float32) + current_pos
        position_ids[:, 0, seq_len:] = tail_positions

        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = inv_freq_expanded @ position_ids
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        # Vision prefill has no window-decoupling yet: window width == absolute history length.
        attention_mask = dynamic_causal_attention_mask(self, ids_len, history_len)
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_VIDEO_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for one decode step with video context."""

    def __init__(self, llm, tokenizer, num_frames, frame_seqlen, max_seq_len, fps=2.0):
        super().__init__()
        query = "Q"
        export_prompt = build_export_video_prompt(query, num_frames, frame_seqlen, fps)
        token_ids = tokenizer(export_prompt, return_tensors="np")["input_ids"][0]
        mm_types = build_mm_token_type_ids(token_ids, "video")
        cos, sin = ROTARY_VIDEO_PREFILL._build_rotary_table(llm, mm_types, num_frames, frame_seqlen, max_seq_len)
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
        register_dynamic_causal_mask_buffers(self, max_seq_len)

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

    def forward(self, ids_len, history_len, cache_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        # cache_len decouples the attended KV window from the absolute rotary position so the
        # Android runtime can feed Main a shorter prefill window (prefill-window decoupling).
        attention_mask = dynamic_causal_attention_mask(self, ids_len, cache_len)
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


# ══════════════════════════════════════════════════════════════════════════════
# ORT-fused RMS normalization (SimplifiedLayerNormalization)
# ══════════════════════════════════════════════════════════════════════════════
class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Export an ORT-fused SimplifiedLayerNormalization (fp32-accumulated RMS)."""

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance = x.float().pow(2).mean(dim=axis, keepdim=True)
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
        # The gated-delta query d^-1/2 scale is folded into com.microsoft::LinearAttention's
        # scale_f attribute (o = scale * q^T S); query is only read in the output, so this is
        # exactly equivalent to pre-scaling query and removes one Mul per linear layer.

        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.compute_in_f32 = COMPUTE_IN_F32
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
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))
        # Constant scales for ORT-fused SimplifiedLayerNormalization on the hidden and
        # full-attention QK norms. scale = N^-0.5 cancels the sqrt(N) that _fuse_weights
        # folds into the following linear, so net output == the manual sum-based RMS norm
        # (x * rsqrt(mean(x^2) + raw_eps)). SimplifiedLayerNorm accumulates in fp32
        # (stash_type=1), so no PREVENT_F16_OVERFLOW prescale is needed here.
        self.register_buffer("hidden_norm_scale", torch.full((self.hidden_size,), self.hidden_size ** -0.5, dtype=torch.float32))
        self.register_buffer("qk_norm_scale", torch.full((self.head_dim,), self.head_dim ** -0.5, dtype=torch.float32))
        # Linear-attention qk l2-norm and gated-delta output RMS norm reuse the same ORT-fused
        # SimplifiedLayerNormalization. scale = head_dim^-0.5 reproduces the sum-based norm
        # (x * rsqrt(sum(x^2) + N*epsilon)); the output-norm sqrt(N) + weight stay folded into
        # out_proj (see _absorb_linear_input_norm), so passing rms_norm_eps there is exact, and
        # the qk l2-norm keeps its raw 1e-6 via epsilon = 1e-6 / head_dim.
        self.register_buffer("linear_head_norm_scale", torch.full((self.linear_key_head_dim,), self.linear_key_head_dim ** -0.5, dtype=torch.float32))
        self.linear_qk_norm_eps = 1e-6 / self.linear_key_head_dim
        self.lm_head_weight = None

        replace_gelu_with_tanh_approximation(self.llm.model.language_model)
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

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

    def _reorder_downproj_for_quant(self, key: str):
        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                weight = layer.mlp.down_proj.weight
                abs_weight = weight.abs()
                if key == "rms":
                    stat = (weight * weight).mean(0).sqrt()
                elif key == "L4":
                    stat = abs_weight.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    stat = weight.std(0)
                else:
                    stat = abs_weight.mean(0)
                perm = torch.argsort(stat)
                inter = layer.mlp.down_proj.in_features
                gate_up = layer.mlp.gate_up_proj.weight
                layer.mlp.gate_up_proj.weight.data.copy_(torch.cat([gate_up[:inter][perm], gate_up[inter:][perm]], dim=0))
                layer.mlp.down_proj.weight.data.copy_(weight[:, perm])

    def _reorder_oproj_for_quant(self, key: str):
        heads = self.num_heads
        kv_heads = self.num_key_value_heads
        head_dim = self.head_dim
        groups = heads // kv_heads
        with torch.no_grad():
            for layer_type, layer in zip(LAYER_TYPES, self.llm.model.language_model.layers):
                if layer_type != "full_attention":
                    continue
                out_proj_weight = layer.self_attn.o_proj.weight
                out_proj_by_head = out_proj_weight.view(out_proj_weight.shape[0], heads, head_dim)
                perms = []
                for kv_head in range(kv_heads):
                    cols = out_proj_by_head[:, kv_head * groups:(kv_head + 1) * groups, :]
                    abs_cols = cols.abs()
                    if key == "rms":
                        stat = (cols * cols).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        stat = cols.reshape(-1, head_dim).std(0)
                    elif key == "L4":
                        stat = abs_cols.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:
                        stat = abs_cols.mean(dim=(0, 1))
                    perms.append(torch.argsort(stat))

                reordered_out = out_proj_by_head.clone()
                for head in range(heads):
                    reordered_out[:, head, :] = reordered_out[:, head, perms[head // groups]]
                out_proj_weight.data.copy_(reordered_out.reshape(out_proj_weight.shape[0], heads * head_dim))

                qkv_weight = layer.self_attn.qkv.weight
                qkv_by_group = qkv_weight.view(2, self.qk_heads, head_dim, qkv_weight.shape[1]).clone()
                qkv_bias = layer.self_attn.qkv.bias
                qkv_bias_by_group = qkv_bias.view(2, self.qk_heads, head_dim).clone() if qkv_bias is not None else None
                for kv_head, perm in enumerate(perms):
                    qkv_by_group[1, heads + kv_head] = qkv_by_group[1, heads + kv_head][perm]
                    if qkv_bias_by_group is not None:
                        qkv_bias_by_group[1, heads + kv_head] = qkv_bias_by_group[1, heads + kv_head][perm]
                    for group_index in range(groups):
                        query_head = kv_head * groups + group_index
                        qkv_by_group[1, query_head] = qkv_by_group[1, query_head][perm]
                        if qkv_bias_by_group is not None:
                            qkv_bias_by_group[1, query_head] = qkv_bias_by_group[1, query_head][perm]
                qkv_weight.data.copy_(qkv_by_group.reshape(qkv_weight.shape[0], qkv_weight.shape[1]))
                if qkv_bias is not None:
                    qkv_bias.data.copy_(qkv_bias_by_group.reshape(qkv_bias.shape[0]))

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
        # F16-KV OFF mode runs attention in float16; cast the shared causal mask once per forward.
        attn_mask_f16 = attention_mask.half() if (self.kv_f16 and not self.compute_in_f32) else None

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
            hidden_states = simplified_layer_norm(hidden_states, self.hidden_norm_scale, self.rms_norm_eps)

            if layer_type == "full_attention":
                attn = layer.self_attn
                qkv = attn.qkv(hidden_states)
                qkv = qkv.reshape(batch_size, -1, 2, self.qk_heads, self.head_dim)
                qk, gate_value = torch.split(qkv, 1, dim=2)

                qk = simplified_layer_norm(qk, self.qk_norm_scale, self.rms_norm_eps) * attn.qk_norm_weight
                if self.full_rotary:
                    qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
                else:
                    qk_rot, qk_pass = torch.split(qk, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
                    qk = torch.cat([qk_rot * rotary_pos_emb_cos + self._rotate_half(qk_rot, batch_size) * rotary_pos_emb_sin, qk_pass], dim=-1)

                # F16-KV OFF mode: cast qk to float16 at the earliest clean point (post-RoPE, pre-split)
                # so the shared cast feeds both query and key and the following split/reshape/permute run
                # in float16. The growing F16 KV cache is then never upcast at the attention matmul.
                if self.kv_f16 and not self.compute_in_f32:
                    qk = qk.half()

                query, key = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate, value = torch.split(gate_value, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate = gate.reshape(batch_size, -1, self.num_heads * self.head_dim)
                query = query.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                query = query.permute(0, 2, 3, 1, 4)

                if self.kv_f16:
                    if self.compute_in_f32:
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

                    if self.kv_f16 and self.compute_in_f32:
                        # F16 cache stays F16 for storage; upcast local copies for float32 attention.
                        attn_output = torch.matmul(query, key_cache.float()) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)
                        attn_output = torch.matmul(attn_output, value_cache.float())
                    elif self.kv_f16:
                        # OFF: fully float16 attention (query is already F16 from the early qk cast);
                        # the cache is never upcast. Only the final result is raised to float32 for o_proj.
                        attn_output = torch.matmul(query, key_cache) + attn_mask_f16
                        attn_output = torch.softmax(attn_output, dim=-1)
                        attn_output = torch.matmul(attn_output, value_cache).float()
                    else:
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
                qk = simplified_layer_norm(qk, self.linear_head_norm_scale, self.linear_qk_norm_eps)
                query, key = torch.split(qk, [self.linear_num_key_heads, self.linear_num_key_heads], dim=2)

                core_attn_out, recurrent_state_out = recurrent_gated_delta_prefill(query, key, value, g, beta, recurrent_state)
                core_attn_out = simplified_layer_norm(core_attn_out, self.linear_head_norm_scale, self.rms_norm_eps)
                core_attn_out = core_attn_out.reshape(batch_size, -1, self.linear_value_dim)
                core_attn_out = core_attn_out * F.silu(z)

                hidden_states = residual + linear.out_proj(core_attn_out)

                save_conv_states.append(conv_state_out.half())
                save_recurrent_states.append(recurrent_state_out.half())

                linear_layer_index += 1

            residual = hidden_states
            hidden_states = simplified_layer_norm(hidden_states, self.hidden_norm_scale, self.rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate_part, up_part = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate_part) * up_part)

        hidden_states = simplified_layer_norm(hidden_states[:, -1], self.hidden_norm_scale, self.rms_norm_eps)
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
# Metadata helpers
# ══════════════════════════════════════════════════════════════════════════════
def collect_special_token_ids(model_path):
    """Best-effort chat/special token IDs from the tokenizer; never fatal."""
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
            "chat_vision_start_id":   special_id("<|vision_start|>"),
            "chat_vision_end_id":     special_id("<|vision_end|>"),
            "chat_image_pad_id":      special_id("<|image_pad|>"),
            "chat_video_pad_id":      special_id("<|video_pad|>"),
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
    """Merge metadata sections into a flat str->str map, dropping None values."""
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
    """Stamp metadata_props without re-serializing external *.data weights."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def run_exported_onnx_inference():
    """Run the standalone ONNX inference script against this export folder."""
    inference_script = SCRIPT_DIR / "Inference_Qwen_ONNX.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found: {inference_script}")

    command = [
        sys.executable,
        str(inference_script),
        "--model-folder",
        str(ONNX_DIR.resolve()),
        "--tokenizer-folder",
        str(Path(download_path).expanduser().resolve()),
    ]
    print("\nRunning exported ONNX inference ...")
    subprocess.run(command, check=True)


# ══════════════════════════════════════════════════════════════════════════════
# Export driver
# ══════════════════════════════════════════════════════════════════════════════
if DO_EXPORT:
    print("Export start ...")

    with torch.inference_mode():
        model, tokenizer = load_model_and_tokenizer()
        prompt_head_len = compute_prompt_head_len(tokenizer)
        between_len = compute_between_len(tokenizer)

        for note in normalize_kv_quant_settings(HEAD_DIM):
            print(f"\n{note}")

        # Capture config/generation facts BEFORE `del model` for later metadata stamping.
        _gen_cfg = getattr(model, "generation_config", None)
        _eos = getattr(_gen_cfg, "eos_token_id", None) if _gen_cfg is not None else None
        if _eos is None:
            _eos = TEXT_CONFIG.get("eos_token_id", MODEL_CONFIG.get("eos_token_id"))
        if isinstance(_eos, (list, tuple)):
            _eos_ids = ",".join(str(int(e)) for e in _eos)
        elif _eos is not None:
            _eos_ids = str(int(_eos))
        else:
            _eos_ids = None
        model_config_meta = {
            "model_type":              MODEL_CONFIG.get("model_type"),
            "intermediate_size":       INTERMEDIATE_SIZE,
            "max_position_embeddings": TEXT_CONFIG.get("max_position_embeddings"),
            "rope_theta":              ROPE_THETA,
            "rms_norm_eps":            RMS_NORM_EPS,
            "tie_word_embeddings":     TEXT_CONFIG.get("tie_word_embeddings", MODEL_CONFIG.get("tie_word_embeddings")),
            "bos_token_id":            TEXT_CONFIG.get("bos_token_id", MODEL_CONFIG.get("bos_token_id")),
            "pad_token_id":            TEXT_CONFIG.get("pad_token_id", MODEL_CONFIG.get("pad_token_id")),
            "eos_token_ids":           _eos_ids,
        }
        chat_token_meta = collect_special_token_ids(download_path)

        batch_size = BEAM_SIZE
        ids_len_dummy = 10
        history_len_dummy = 0
        ids_len = torch.tensor([ids_len_dummy], dtype=torch.int64)
        history_len = torch.tensor([history_len_dummy], dtype=torch.int64)
        cache_len = torch.tensor([history_len_dummy], dtype=torch.int64)   # prefill-window width fed to Main
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

        # ── Export LLM_Embed ──
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

        # ── Export LLM_Image_Preprocess ──
        pixel_values_img = torch.randint(
            0, 255,
            size=(VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]),
            dtype=torch.uint8,
        )
        img_preprocess = LLM_IMAGE_PREPROCESS(
            model.model.visual, HEIGHT_FACTOR, WIDTH_FACTOR,
            VISION_BATCH_SIZE, DYNAMIC_IMAGE_SHAPE
        ).eval()
        torch.onnx.export(
            img_preprocess,
            (pixel_values_img,),
            onnx_model_Image_Preprocess,
            input_names=["pixel_values"],
            output_names=["pixel_values_patches", "pos_embeds", "rotary_cos", "rotary_sin", "attention_mask"],
            dynamic_axes={"pixel_values": {0: "num_images"}} if DYNAMIC_IMAGE_SHAPE else None,
            opset_version=OPSET,
            dynamo=False,
        )
        del pixel_values_img
        gc.collect()

        # ── Export LLM_Video_Preprocess ──
        video_frames = torch.randint(
            0, 255,
            size=(VIDEO_NUM_FRAMES, 3, INPUT_VIDEO_SIZE[0], INPUT_VIDEO_SIZE[1]),
            dtype=torch.uint8,
        )
        vid_preprocess = LLM_VIDEO_PREPROCESS(
            model.model.visual, VIDEO_NUM_FRAMES, VIDEO_HEIGHT_FACTOR, VIDEO_WIDTH_FACTOR, DYNAMIC_VIDEO_SHAPE
        ).eval()
        torch.onnx.export(
            vid_preprocess,
            (video_frames,),
            onnx_model_Video_Preprocess,
            input_names=["video_frames"],
            output_names=["pixel_values_patches", "pos_embeds", "rotary_cos", "rotary_sin", "attention_mask"],
            opset_version=OPSET,
            dynamo=False,
        )
        del video_frames, vid_preprocess
        gc.collect()

        # ── Export shared LLM_Vision ──
        # Use image patches as representative input
        img_preprocess_for_vis = LLM_IMAGE_PREPROCESS(
            model.model.visual, HEIGHT_FACTOR, WIDTH_FACTOR,
            VISION_BATCH_SIZE, False
        ).eval()
        dummy_img = torch.randint(0, 255, size=(VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
        with torch.no_grad():
            pv_patches, pos_emb, rot_cos, rot_sin, attn_mask = img_preprocess_for_vis(dummy_img)
        vision_model = LLM_VISION(model).eval()
        torch.onnx.export(
            vision_model,
            (pv_patches, pos_emb, rot_cos, rot_sin, attn_mask),
            onnx_model_Vision,
            input_names=["pixel_values", "pos_embeds", "rotary_cos", "rotary_sin", "attention_mask"],
            output_names=["vision_hidden_states"],
            dynamic_axes={
                "pixel_values": {0: "total_patches"},
                "pos_embeds": {1: "vision_seq_len"},
                "rotary_cos": {3: "vision_seq_len"},
                "rotary_sin": {3: "vision_seq_len"},
                "attention_mask": {2: "vision_seq_len", 3: "vision_seq_len"},
                "vision_hidden_states": {1: "vision_seq_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del pv_patches, pos_emb, rot_cos, rot_sin, attn_mask, dummy_img, img_preprocess_for_vis, vision_model
        gc.collect()

        # ── Export LLM_Concat_Image ──
        text_hidden_states = torch.ones((1, ids_len_dummy, HIDDEN_SIZE), dtype=torch.float32)
        vision_hidden_states = torch.ones((1, VISION_EMBED_SIZE, HIDDEN_SIZE), dtype=torch.float32)
        concat_image = LLM_CONCAT_IMAGE(VISION_BATCH_SIZE, IMAGE_SEQLEN_PER_IMAGE, prompt_head_len, between_len).eval()
        torch.onnx.export(
            concat_image,
            (text_hidden_states, vision_hidden_states),
            onnx_model_Concat_Image,
            input_names=["text_hidden_states", "vision_hidden_states"],
            output_names=["concat_hidden_states"],
            dynamic_axes={
                "text_hidden_states": {1: "ids_len"},
                "concat_hidden_states": {1: "total_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del concat_image
        gc.collect()

        # ── Export LLM_Concat_Video ──
        # Build a representative tokenized video prompt so traced offsets match
        # the in-place placeholder replacement semantics used at runtime.
        video_export_prompt = build_export_video_prompt("Describe this video.", VIDEO_GRID_T, VIDEO_FRAME_SEQLEN, VIDEO_FPS)
        video_prompt_tokens = tokenizer(video_export_prompt, return_tensors="np")["input_ids"][0]
        video_text_len = int(video_prompt_tokens.shape[0])
        text_hidden_video = torch.ones((1, video_text_len, HIDDEN_SIZE), dtype=torch.float32)
        vision_hidden_video = torch.ones((1, VIDEO_TOTAL_VISION_TOKENS, HIDDEN_SIZE), dtype=torch.float32)
        video_segment_offsets = build_video_segment_offsets(
            video_prompt_tokens.tolist(), VIDEO_FRAME_SEQLEN
        )
        concat_video = LLM_CONCAT_VIDEO(video_segment_offsets, VIDEO_FRAME_SEQLEN).eval()
        torch.onnx.export(
            concat_video,
            (text_hidden_video, vision_hidden_video),
            onnx_model_Concat_Video,
            input_names=["text_hidden_states", "vision_hidden_states"],
            output_names=["concat_hidden_states"],
            dynamic_axes={
                "text_hidden_states": {1: "ids_len"},
                "concat_hidden_states": {1: "total_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del text_hidden_states, vision_hidden_states, text_hidden_video, vision_hidden_video, video_segment_offsets, concat_video, video_prompt_tokens
        gc.collect()

        # ── Export Rotary modules ──
        torch.onnx.export(
            ROTARY_IMAGE_PREFILL(model, tokenizer, VISION_BATCH_SIZE, IMAGE_SEQLEN_PER_IMAGE, MAX_SEQ_LEN).eval(),
            (ids_len, history_len),
            onnx_model_Rotary_Image_Prefill,
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
            ROTARY_IMAGE_DECODE(model, tokenizer, VISION_BATCH_SIZE, IMAGE_SEQLEN_PER_IMAGE, MAX_SEQ_LEN).eval(),
            (kv_seq_len,),
            onnx_model_Rotary_Image_Decode,
            input_names=["kv_seq_len"],
            output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_VIDEO_PREFILL(model, tokenizer, VIDEO_GRID_T, VIDEO_FRAME_SEQLEN, MAX_SEQ_LEN, VIDEO_FPS).eval(),
            (ids_len, history_len),
            onnx_model_Rotary_Video_Prefill,
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
            ROTARY_VIDEO_DECODE(model, tokenizer, VIDEO_GRID_T, VIDEO_FRAME_SEQLEN, MAX_SEQ_LEN, VIDEO_FPS).eval(),
            (kv_seq_len,),
            onnx_model_Rotary_Video_Decode,
            input_names=["kv_seq_len"],
            output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_TEXT_PREFILL(model, MAX_SEQ_LEN).eval(),
            (ids_len, history_len, cache_len),
            onnx_model_Rotary_Text_Prefill,
            input_names=["ids_len", "history_len", "cache_len"],
            output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
            dynamic_axes={
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "mask_len"},
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

        # ── Export LLM_Main ──
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
        kv_quantizer = model_main.quantizer
        rope_shift_rotary_module = model_main.llm.model.language_model.rotary_emb
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
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )
        del model_main, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        # ── Export decoding modules ──
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
            KV_SLICE(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM).eval(),
            tuple(full_inputs + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=full_input_names + ["slice_start", "slice_end"],
            output_names=full_output_names,
            dynamic_axes=full_dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del slice_start, slice_end

        # ── Export KV_Split2 (full-attention prefix/window split) ──
        split_at = torch.tensor([5], dtype=torch.int64)
        split_prefix_names = [f"prefix_{name}" for name in full_output_names]
        split_window_names = [f"window_{name}" for name in full_output_names]
        split_axes = {name: dict(full_dynamic_axes[name]) for name in full_input_names}
        for source_name, prefix_name, window_name in zip(full_output_names, split_prefix_names, split_window_names):
            source_axes = full_dynamic_axes[source_name]
            split_axes[prefix_name] = dict(source_axes)
            split_axes[window_name] = dict(source_axes)
            for axis in split_axes[prefix_name]:
                if axis != 0:
                    split_axes[prefix_name][axis] = "prefix_len"
                    split_axes[window_name][axis] = "window_len"
        torch.onnx.export(
            KV_SPLIT2(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM).eval(),
            tuple(full_inputs + [split_at]),
            onnx_model_KV_Split2,
            input_names=full_input_names + ["split_at"],
            output_names=split_prefix_names + split_window_names,
            dynamic_axes=split_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del split_at, split_prefix_names, split_window_names, split_axes

        # ── Export KV_Concat (full-attention prefix + suffix concat) ──
        cat_a_ins, cat_a_names, cat_b_ins, cat_b_names, cat_out_names, cat_axes = [], [], [], [], [], {}
        for name, dim in FULL_STATE_SPECS:
            tensor = full_state_tensors[name]
            for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
                a_n, b_n, o_n = f"in_a_{name}_{layer_index}", f"in_b_{name}_{layer_index}", f"out_{name}_{layer_index}"
                cat_a_ins.append(tensor);         cat_a_names.append(a_n)
                cat_b_ins.append(tensor.clone()); cat_b_names.append(b_n)
                cat_out_names.append(o_n)
                cat_axes[a_n] = {0: "batch_size", dim: "prefix_len"}
                cat_axes[b_n] = {0: "batch_size", dim: "suffix_len"}
                cat_axes[o_n] = {0: "batch_size", dim: "concat_len"}
        torch.onnx.export(
            KV_CONCAT(NUM_FULL_ATTENTION_LAYERS, HEAD_DIM).eval(),
            tuple(cat_a_ins + cat_b_ins),
            onnx_model_KV_Concat,
            input_names=cat_a_names + cat_b_names,
            output_names=cat_out_names,
            dynamic_axes=cat_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del cat_a_ins, cat_a_names, cat_b_ins, cat_b_names, cat_out_names, cat_axes

        # ── Export RopeShift (mRoPE key re-rotation after sliding-window KV eviction) ──
        def _seq4(tensor):
            shape = list(tensor.shape)
            shape[-1] = 4
            return torch.zeros(shape, dtype=tensor.dtype)

        def _rope_shift_key_io(specs):
            inputs, input_names, output_names, axes = [], [], [], {}
            for name, tensor in specs:
                seq_axis = tensor.dim() - 1
                for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
                    input_name = f"in_{name}_{layer_index}"
                    output_name = f"out_{name}_{layer_index}"
                    inputs.append(tensor)
                    input_names.append(input_name)
                    output_names.append(output_name)
                    axes[input_name] = {0: "batch_size", seq_axis: "history_len"}
                    axes[output_name] = {0: "batch_size", seq_axis: "history_len"}
            return inputs, input_names, output_names, axes

        rope_shift_amount = torch.tensor([5], dtype=torch.int64)
        if KV_QUANT_DTYPE in ("F16", "F32"):
            rope_specs = [("key", _seq4(full_state_tensors["key"]))]
            rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_key_io(rope_specs)
            torch.onnx.export(
                ROPE_SHIFT(NUM_FULL_ATTENTION_LAYERS, NUM_KEY_VALUE_HEADS, rope_shift_rotary_module, MAX_SEQ_LEN).eval(),
                tuple(rope_inputs + [rope_shift_amount]),
                onnx_model_Rope_Shift,
                input_names=rope_input_names + ["shift"],
                output_names=rope_output_names,
                dynamic_axes=rope_axes,
                opset_version=OPSET,
                dynamo=False,
            )
        elif _is_quantized_kv or _is_rotary_kv:
            rope_specs = [
                ("key", _seq4(full_state_tensors["key"])),
                ("key_scale", _seq4(full_state_tensors["key_scale"])),
            ]
            if not _kv_sym:
                rope_specs.append(("key_bias", _seq4(full_state_tensors["key_bias"])))
            rope_inputs, rope_input_names, rope_output_names, rope_axes = _rope_shift_key_io(rope_specs)
            torch.onnx.export(
                ROPE_SHIFT_QUANT(
                    NUM_FULL_ATTENTION_LAYERS,
                    NUM_KEY_VALUE_HEADS,
                    rope_shift_rotary_module,
                    MAX_SEQ_LEN,
                    kv_quantizer,
                    not _kv_sym,
                ).eval(),
                tuple(rope_inputs + [rope_shift_amount]),
                onnx_model_Rope_Shift,
                input_names=rope_input_names + ["shift"],
                output_names=rope_output_names,
                dynamic_axes=rope_axes,
                opset_version=OPSET,
                dynamo=False,
            )
        del _seq4, _rope_shift_key_io, rope_shift_amount, rope_specs, rope_inputs, rope_input_names, rope_output_names, rope_axes

        # ── Export LLM_Metadata (metadata carrier graph) ──
        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        torch.onnx.export(
            METADATA_CARRIER().eval(),
            (metadata_marker,),
            onnx_model_Metadata,
            input_names=["metadata_marker"],
            output_names=["metadata_marker_out"],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False,
        )
        del metadata_marker

        # ── Build and stamp model metadata into every exported graph ──
        _dtype_names = {
            torch.float32: "float32", torch.float16: "float16",
            torch.int8: "int8", torch.uint8: "uint8",
            torch.int32: "int32", torch.int64: "int64",
        }
        if KV_QUANT_DTYPE == "F16":
            _kv_cache_dtype = torch.float16
        elif KV_QUANT_DTYPE == "F32":
            _kv_cache_dtype = torch.float32
        elif KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"):
            _kv_cache_dtype = torch.int32
        elif _kv_sym and not _is_rotary_q4_kv:
            _kv_cache_dtype = torch.int8
        else:
            _kv_cache_dtype = torch.uint8
        _kv_is_quant = _is_quantized_kv or _is_rotary_kv
        _kv_blocks_per_layer = len(FULL_STATE_SPECS)

        onnx_metadata = build_model_metadata(
            {
                "native_llm_metadata_version": 1,
                "producer":                    "Export_Qwen.py",
            },
            {
                "num_layers":                  len(LAYER_TYPES),
                "num_full_attention_layers":   NUM_FULL_ATTENTION_LAYERS,
                "num_linear_attention_layers": NUM_LINEAR_ATTENTION_LAYERS,
                "num_attention_heads":         NUM_HEADS,
                "num_key_value_heads":         NUM_KEY_VALUE_HEADS,
                "head_dim":                    HEAD_DIM,
                "hidden_size":                 HIDDEN_SIZE,
                "intermediate_size":           INTERMEDIATE_SIZE,
                "vocab_size":                  VOCAB_SIZE,
                "rms_norm_eps":                RMS_NORM_EPS,
                "rope_theta":                  ROPE_THETA,
                "partial_rotary_factor":       PARTIAL_ROTARY_FACTOR,
                "rotary_dim":                  ROTARY_DIM,
                "mrope_section":               ",".join(str(int(s)) for s in MROPE_SECTION),
                "max_position_embeddings":     TEXT_CONFIG.get("max_position_embeddings"),
                "max_seq_len":                 MAX_SEQ_LEN,
                "opset":                       OPSET,
                "activations_fp16":            False,
                "compute_in_f32":              COMPUTE_IN_F32,
                "rope_shift_mrope":            True,
                "reorder_downproj":            REORDER_DOWNPROJ_FOR_QUANT,
                "reorder_oproj":               REORDER_OPROJ_FOR_QUANT,
                "linear_recurrence_impl":      "ort_linear_attention",
                "linear_attention_op_domain":  "com.microsoft",
                "linear_attention_update_rule": "gated_delta",
                "linear_attention_chunk_size": ORT_LINEAR_ATTENTION_CHUNK_SIZE,
            },
            {
                "layer_types":            ",".join(LAYER_TYPES),
                "linear_conv_kernel_dim": LINEAR_CONV_KERNEL_DIM,
                "linear_num_key_heads":   LINEAR_NUM_KEY_HEADS,
                "linear_num_value_heads": LINEAR_NUM_VALUE_HEADS,
                "linear_key_head_dim":    LINEAR_KEY_HEAD_DIM,
                "linear_value_head_dim":  LINEAR_VALUE_HEAD_DIM,
                "linear_conv_dim":        LINEAR_CONV_DIM,
                "attn_output_gate":       ATTN_OUTPUT_GATE,
            },
            {
                "vision_hidden_size":    VISION_HIDDEN_SIZE,
                "vision_depth":          VISION_DEPTH,
                "vision_num_heads":      VISION_NUM_HEADS,
                "patch_size":            VISION_PATCH_SIZE,
                "temporal_patch_size":   VISION_TEMPORAL_PATCH_SIZE,
                "spatial_merge_size":    VISION_SPATIAL_MERGE_SIZE,
                "out_hidden_size":       VISION_OUT_HIDDEN_SIZE,
                "image_token_id":        IMAGE_TOKEN_ID,
                "video_token_id":        VIDEO_TOKEN_ID,
                "vision_start_token_id": VISION_START_TOKEN_ID,
                "vision_end_token_id":   VISION_END_TOKEN_ID,
                "vision_reorder_mlp":    REORDER_VISION_MLP_FOR_QUANT,
                "vision_reorder_oproj":  REORDER_VISION_OPROJ_FOR_QUANT,
            },
            {
                "video_fps":        VIDEO_FPS,
                "video_min_frames": VIDEO_MIN_FRAMES,
                "video_max_frames": VIDEO_MAX_FRAMES,
                "input_image_size": ",".join(str(int(s)) for s in INPUT_IMAGE_SIZE),
                "input_video_size": ",".join(str(int(s)) for s in INPUT_VIDEO_SIZE),
            },
            {
                "kv_quant_dtype":          KV_QUANT_DTYPE,
                "kv_quant_group_size":     KV_QUANT_GROUP_SIZE,
                "kv_blocks_per_layer":     _kv_blocks_per_layer,
                "kv_num_tensors":          NUM_FULL_ATTENTION_LAYERS * _kv_blocks_per_layer,
                "kv_symmetric":            _kv_sym,
                "kv_grouped_6d":           _grouped_6d,
                "kv_cache_elem_type":      _dtype_names.get(_kv_cache_dtype, str(_kv_cache_dtype)),
                "kv_scale_bias_elem_type": _dtype_names.get(SCALE_DTYPE_TORCH) if _kv_is_quant else None,
                "kv_quant_hadamard":       USE_HADAMARD if _kv_is_quant else None,
                "kv_quant_shuffle":        USE_SHUFFLE if _kv_is_quant else None,
                "kv_quant_clip":           USE_CLIP if _kv_is_quant else None,
                "kv_qdq_friendly_asym":    USE_QDQ_FRIENDLY_ASYM if _kv_is_quant and not _kv_sym else None,
            },
            {
                "stop_token_ids": ",".join(str(int(t)) for t in STOP_TOKEN) if STOP_TOKEN else None,
            },
            model_config_meta,
            chat_token_meta,
        )

        _metadata_targets = [
            onnx_model_Metadata, onnx_model_Embed, onnx_model_Vision,
            onnx_model_Image_Preprocess, onnx_model_Video_Preprocess,
            onnx_model_Concat_Image, onnx_model_Concat_Video,
            onnx_model_Rotary_Image_Prefill, onnx_model_Rotary_Image_Decode,
            onnx_model_Rotary_Video_Prefill, onnx_model_Rotary_Video_Decode,
            onnx_model_Rotary_Text_Prefill, onnx_model_Rotary_Text_Decode,
            onnx_model_Main, onnx_model_Greedy, onnx_model_First_Beam,
            onnx_model_Second_Beam, onnx_model_Penalty, onnx_model_Argmax,
            onnx_model_KV_Slice, onnx_model_KV_Split2, onnx_model_KV_Concat,
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
            kv_quantizer,
            rope_shift_rotary_module,
        )
        gc.collect()

    print("Export done.")
    run_exported_onnx_inference()
