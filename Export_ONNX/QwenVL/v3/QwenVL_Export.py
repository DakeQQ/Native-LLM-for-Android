import os
import gc
import time
import itertools
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding


download_path                   = r'/home/DakeQQ/Downloads/Qwen3-VL-2B-Instruct'          # Set the folder path where the Qwen3-VL whole project downloaded.
onnx_model_Embed                = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx'      # Assign a path where the exported QwenVL model stored.
onnx_model_Vision               = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Vision.onnx'
onnx_model_Concat_Image         = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Concat_Image.onnx'
onnx_model_Concat_Video         = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Concat_Video.onnx'
onnx_model_Rotary_Image_Prefill = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Image_Prefill.onnx'
onnx_model_Rotary_Image_Decode  = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Image_Decode.onnx'
onnx_model_Rotary_Video_Prefill = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Video_Prefill.onnx'
onnx_model_Rotary_Video_Decode  = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Video_Decode.onnx'
onnx_model_Rotary_Text_Prefill  = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Text_Prefill.onnx'
onnx_model_Rotary_Text_Decode   = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Text_Decode.onnx'
onnx_model_Main                 = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx'
onnx_model_Greedy               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_Penalty              = r'/home/DakeQQ/Downloads/Qwen_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'
onnx_model_KV_Slice             = r'/home/DakeQQ/Downloads/Qwen_ONNX/KV_Slice.onnx'
onnx_model_Video_Preprocess     = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Video_Preprocess.onnx'

# Test Input
TEST_IMAGE               = ["../psyduck.png"]                                   # List of test images for the exported onnx model. Supports multi-image: [r"img1.png", r"img2.png"]
TEST_VIDEO               = r"../test_video_8s.mp4"                              # Test video path. Set non-empty to enable video mode.
DEFAULT_IMAGE_QUERY      = "Describe this image."
DEFAULT_MULTI_IMAGE_QUERY = "Treat each image as a separate photo and describe them one by one."
TEST_QUERY               = [DEFAULT_IMAGE_QUERY, "Describe this video."]        # Test query for the exported onnx model.

# Model Config
DO_EXPORT                = True                                     # Whether to export the ONNX models
PREVENT_F16_OVERFLOW     = False                                    # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN               = [151643, 151645]                         # Qwen stop token ids
MAX_SEQ_LEN              = 4096                                     # Max context length. Can not edit after export.

# Vision Config
HEIGHT_FACTOR            = 20                                       # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR             = 20                                       # Adjust this value to determine the resize shape and vision resolution.
IMAGE_RESIZE             = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]  # 32 = self.patch_size * self.merge_size
INPUT_IMAGE_SIZE         = [960, 960]                               # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
VISION_BATCH_SIZE        = 1                                        # Fixed at export time. Number of images in multi-image mode. Set >1 for multi-image support. Each image uses HEIGHT_FACTOR * WIDTH_FACTOR vision tokens.
DYNAMIC_IMAGE_SHAPE      = False                                    # Allow for a dynamic number of image inputs (1..VISION_BATCH_SIZE). When False, exactly VISION_BATCH_SIZE images required.
INPUT_IMAGE_DIM          = 5                                        # 4 for [batch, 3, height, width]; 5 for [batch, 1, 3, height, width]

# Video Config
VIDEO_FPS                = 2.0                                      # Frames per second to sample from video.
VIDEO_MAX_FRAMES         = 768                                      # Maximum total frames before temporal patching.
VIDEO_MIN_FRAMES         = 4                                        # Minimum total frames.
VIDEO_NUM_FRAMES         = 8                                        # Actual number of frames used for export (determines rotary table shape). Must be even (divisible by temporal_patch_size=2).
VIDEO_HEIGHT_FACTOR      = 10                                       # Video spatial height factor. grid_h = VIDEO_HEIGHT_FACTOR * 2
VIDEO_WIDTH_FACTOR       = 18                                       # Video spatial width factor. grid_w = VIDEO_WIDTH_FACTOR * 2
TEMPORAL_PATCH_SIZE      = 2                                        # From model config.vision_config.temporal_patch_size. Do not edit.
VIDEO_RESIZE             = [VIDEO_HEIGHT_FACTOR * 32, VIDEO_WIDTH_FACTOR * 32]  # Input frame shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
INPUT_VIDEO_SIZE         = [720, 1280]                              # Input video frame shape [height, width]. Used as static shape when DYNAMIC_VIDEO_SHAPE is False.
DYNAMIC_VIDEO_SHAPE      = False                                    # Allow dynamic video frame count and spatial size. Set False for static shape (faster on some backends).

# KV cache quantization
KV_QUANT_DTYPE           = "F16"                                    # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE      = 32                                       # Group size for Q4 and Q8 (when USE_HADAMARD or USE_SHUFFLE enabled) per-group quantization. Smaller = more accurate. Must divide head_dim evenly.
USE_HADAMARD             = True                                     # True = More Accuracy. Apply enhanced randomized Walsh-Hadamard mixing within each group before quantization. Works for Q4 and Q8 modes.
HADAMARD_RANDOM_SEED     = 9527                                     # Seed for the deterministic Rademacher sign pattern used by the enhanced Hadamard transform.
USE_CLIP                 = True                                     # Clip outliers to mean ± CLIP_SIGMA*std before quantization. Works for Q4 and Q8 modes. For Q8 without hadamard/shuffle, clips per-head; with grouping, clips per-group.
CLIP_SIGMA               = 3.5                                      # Clip threshold in standard deviations. Lower = more aggressive clipping. 2.5-3.5 recommended. Only used when USE_CLIP=True.
USE_SHUFFLE              = True                                     # True = More Accuracy. Interleave channels across groups so that high-variance channels are evenly distributed. Works for Q4 and Q8 modes.
USE_SYM                  = False                                    # True = Less RAM Bandwidth. True: symmetric quantization (no bias, absmax-based); False: asymmetric (min-max with bias). Works for Q4 and Q8 modes.
USE_FLOAT16_SCALE_BIAS   = True                                     # Whether to use float16 for scale and bias in all quantized KV modes (Q4, Q8, and ROTARY variants).

# Decoding strategy
REPEAT_PENALTY           = 1.0                                      # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE            = 20                                       # Recent-token window to apply penalty

# Runtime config
ORT_LOG                  = False                                    # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                 = False                                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []                                       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                                        # 0 = auto
DEVICE_ID                = 0                                        # Device ID for GPU
OPSET                    = 18                                       # ONNX opset version


# ══════════════════════════════════════════════════════════════════════════════
# KV Quant Validation
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA",
    "Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA",
    "F16",
    "F32"
)


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


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: select the token with the highest logit."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id        = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Slice
# ══════════════════════════════════════════════════════════════════════════════
class KV_SLICE(torch.nn.Module):
    """Apply slice to KV cache tensors."""

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
        for i in range(self.num_layers):
            self.save_key[i]   = all_inputs[i][..., slice_start: slice_end]
            self.save_value[i] = all_inputs[i + self.num_layers][..., slice_start: slice_end, :]
            if self.kv_quantized:
                if self.kv_sym:
                    # Symmetric: 4 types (key, value, k_scale, v_scale) — no bias
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start: slice_end]
                    if self.kv_grouped_6d:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start: slice_end, :, :]
                    else:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start: slice_end, :]
                elif self.kv_grouped_6d:
                    # Asymmetric ROTARY_Q4 / Q8 grouped: 6 types with 6D scale/bias dims
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start: slice_end]
                    self.save_k_bias[i]  = all_inputs[i + self.num_layers_3][..., slice_start: slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start: slice_end, :, :]
                    self.save_v_bias[i]  = all_inputs[i + self.num_layers_5][..., slice_start: slice_end, :, :]
                else:
                    # Asymmetric Q8/ROTARY_Q8 (non-grouped): 6 types with 5D scale/bias dims
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start: slice_end]
                    self.save_k_bias[i]  = all_inputs[i + self.num_layers_3][..., slice_start: slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start: slice_end, :]
                    self.save_v_bias[i]  = all_inputs[i + self.num_layers_5][..., slice_start: slice_end, :]
        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation (θ=π/4) to the head_dim axis *before* quantization.
       The rotation spreads outlier energy across dimension pairs, making the
       value distribution more uniform and reducing quantization error —
       especially at 4-bit.  During attention the inverse rotation is fused
       algebraically so that no full dequant + inverse-rotate is needed.

     2. **Enhanced Hadamard transform** (USE_HADAMARD, Q4 and Q8 modes):
         applies a deterministic randomized Walsh-Hadamard transform within
         each quantization group.  A fixed Rademacher sign pattern is applied
         before the transform, and non-power-of-two groups are zero-padded to
         the next power of two and cropped back.  This keeps the transform
         orthogonal on the active channels while improving energy spreading
         versus a plain fixed Hadamard block.

    3. **Channel shuffle** (USE_SHUFFLE, Q4 and Q8 modes): interleaves
       channels across groups so that high-variance channels are evenly
       distributed.  Like Hadamard, this also enables per-group Q8
       quantization.

     4. **Residual bias correction** (asymmetric modes): computes the
         mean quantization residual for each block/group and folds it into
         the stored bias.  This reduces systematic dequantization drift for
         Q4 without changing the KV cache layout.
    """

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
        self.use_residual_bias_correction = not use_sym
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads  = num_kv_heads
        self.num_kv_groups = num_kv_groups

        # ── Quantization range ───────────────────────────────────────
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

        # ── Group parameters ──────────────────────────────────────────
        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
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

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,),  inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[:head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # ── Enhanced Hadamard transform buffers ───────────────────────
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            # Pre-compute Hadamard butterfly level widths
            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # ── Clip sigma buffer ─────────────────────────────────────────
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
    # Enhanced Walsh-Hadamard helpers
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        """Apply a deterministic randomized Walsh-Hadamard transform on the last dim.

        The butterfly levels are pre-computed at init time (_hadamard_levels).
        view(*x.shape[:-1], ...) generates Shape nodes for dynamic leading dims,
        but these are constant-folded by ORT when the leading dims are known.
        """
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
    # Sigma-based clipping
    # ══════════════════════════════════════════════════════════════════
    def _clip_to_sigma(self, x, dim):
        """Clip values to mean ± clip_sigma*std per quantization block."""
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    # ══════════════════════════════════════════════════════════════════
    # Rotary flip helpers
    # ══════════════════════════════════════════════════════════════════
    def _flip_k(self, k, batch_size):
        """Swap halves along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)"""
        return k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1).flip(-3).view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _flip_v(self, v, batch_size):
        """Swap halves along head_dim (last dim). v: (B, KVH, 1, S, head_dim)"""
        return v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def _flip_q(self, q, batch_size):
        """Swap halves along head_dim (last dim). q: (B, KVH, G, Qlen, head_dim)"""
        return q.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ── Forward rotation (applied during quantization) ───────────────
    def rotate_k(self, k, batch_size):
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    # ── Inverse rotation (fused into attention computation) ──────────
    def rotate_q(self, q, batch_size):
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Hadamard transform helpers (within quantization groups)
    # ══════════════════════════════════════════════════════════════════
    def hadamard_k(self, k, batch_size):
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Block quantization
    # ══════════════════════════════════════════════════════════════════
    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        """Finalize asymmetric quantization with optional residual bias correction."""
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
        """Quantize to signed integers, then encode into the selected storage container."""
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

    def _quantize_block(self, x, dim, batch_size=1):
        """Per-block quantization. Symmetric (absmax) or asymmetric (min-max)."""
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim, batch_size)
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

    def _quantize_block_grouped(self, x, dim, batch_size):
        """Per-group quantization (Q4 or Q8). Symmetric (absmax) or asymmetric (min-max)."""
        if self.use_sym:
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax   = x.abs().amax(dim=-1, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            return x_packed, scale, block_min

    # ══════════════════════════════════════════════════════════════════
    # CUDA packing / unpacking (4 uint8 → 1 int32)
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
    # Q4 packing / unpacking (2 nibbles → 1 byte)
    # ══════════════════════════════════════════════════════════════════
    def pack_q4_k(self, x, batch_size):
        """Pack Q4 keys: (B,KVH,1, D, S) → (B,KVH,1, D//2, S)."""
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        """Pack Q4 values: (B,KVH,1, S, D) → (B,KVH,1, S, D//2)."""
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        """Unpack Q4 keys: (B,KVH,1, D//2, S) → (B,KVH,1, D, S)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        """Unpack Q4 values: (B,KVH,1, S, D//2) → (B,KVH,1, S, D)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════
    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            keys   = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys   = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Image Preprocessing Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_IMAGE_PREPROCESS(torch.nn.Module):
    """Preprocess N images into packed patches matching the vision encoder input format.

    Accepts: pixel_values [N, 1, 3, H, W] or [N, 3, H, W] uint8/float32 (0-255 range)
             where N is the number of images (1..VISION_BATCH_SIZE)
    Returns: (pixel_values_patches, pos_embeds, rotary_cos, rotary_sin, attention_mask)

    Operations: resize → normalize → duplicate temporal dim → spatial patching.
    For multi-image, produces block-diagonal attention mask preventing cross-image attention.
    The auxiliary vision tensors are pre-computed for VISION_BATCH_SIZE images.
    In dynamic mode, they are sliced at runtime to match the actual image count.
    """

    def __init__(self, patch_size, temporal_patch_size, merge_size, height_factor, width_factor,
                 pos_embeds, rotary_cos, rotary_sin, attention_mask, dynamic_shape=False,
                 num_images=1):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size // 2
        self.merge_size = merge_size
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.dynamic_shape = dynamic_shape
        self.num_images = num_images
        self.target_h = height_factor * patch_size * merge_size  # height_factor * 32
        self.target_w = width_factor * patch_size * merge_size   # width_factor * 32
        self.grid_h = self.target_h // patch_size
        self.grid_w = self.target_w // patch_size
        self.grid_h_merged = self.grid_h // merge_size
        self.grid_w_merged = self.grid_w // merge_size
        self.seq_per_image = height_factor * width_factor  # merged token count per image
        # Vision auxiliary tensors (pre-computed for num_images images, block-diagonal)
        self.register_buffer("pos_embeds", pos_embeds.half())
        self.register_buffer("rotary_cos", rotary_cos.half())
        self.register_buffer("rotary_sin", rotary_sin.half())
        self.register_buffer("attention_mask", attention_mask.to(torch.int8))

    def forward(self, pixel_values):
        # pixel_values: [N, 1, 3, H, W] or [N, 3, H, W], values 0-255
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)  # [N, 3, H, W]

        num_images = pixel_values.shape[0]
        pixel_values = pixel_values.float()

        # Resize to target spatial dims: only when dynamic mode and shape mismatches
        if self.dynamic_shape or (pixel_values.shape[-2] != self.target_h or pixel_values.shape[-1] != self.target_w):
            pixel_values = F.interpolate(
                pixel_values,
                size=[self.target_h, self.target_w],
                mode='bilinear',
                align_corners=False
            )

        # Spatial patching: [N, 3, grid_h//merge * merge * ps, grid_w//merge * merge * ps]
        pixel_values = pixel_values.reshape(
            num_images, 3,
            self.grid_h_merged, self.merge_size, self.patch_size,
            self.grid_w_merged, self.merge_size, self.patch_size
        )
        # Permute to: [N, grid_h//merge, grid_w//merge, merge, merge, 3, ps, ps]
        pixel_values = pixel_values.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(-1, 3, self.temporal_patch_size, self.patch_size, self.patch_size)
        # Duplicate temporal dim: [total_patches_all_images, 3, tps, ps, ps] -> [..., 3, tps*2, ps, ps]
        # Concat of self maps to a single ONNX Concat node (more efficient than Tile)
        pixel_values = torch.cat([pixel_values, pixel_values], dim=2)

        if self.dynamic_shape:
            # Slice pre-baked aux tensors to match actual num_images
            total_seq = num_images * self.seq_per_image
            pos_embeds = self.pos_embeds[:, :total_seq]
            rotary_cos = self.rotary_cos[..., :total_seq]
            rotary_sin = self.rotary_sin[..., :total_seq]
            attention_mask = self.attention_mask[..., :total_seq, :total_seq]
        else:
            pos_embeds = self.pos_embeds
            rotary_cos = self.rotary_cos
            rotary_sin = self.rotary_sin
            attention_mask = self.attention_mask

        return pixel_values, pos_embeds, rotary_cos, rotary_sin, attention_mask


# ══════════════════════════════════════════════════════════════════════════════
# Video Preprocessing Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_VIDEO_PREPROCESS(torch.nn.Module):
    """Preprocess video frames into packed patches ready for the vision encoder.

    Accepts: video_frames [num_frames, 3, H, W] uint8 (0-255 range)
    Returns: (pixel_values_videos, pos_embeds, rotary_cos, rotary_sin, attention_mask)

    All operations (resize, normalize, temporal grouping, spatial patching) are done
    inside the ONNX graph — no numpy at inference time.
    The auxiliary vision tensors (pos_embeds, rotary_cos, rotary_sin, attention_mask)
    are registered as buffers and returned directly as ORT tensors.
    """

    def __init__(self, patch_size, temporal_patch_size, merge_size, target_h, target_w, pos_embeds, rotary_cos, rotary_sin, attention_mask, dynamic_shape=False):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.target_h = target_h
        self.target_w = target_w
        self.dynamic_shape = dynamic_shape
        self.grid_h = target_h // patch_size
        self.grid_w = target_w // patch_size
        self.grid_h_merged = self.grid_h // merge_size
        self.grid_w_merged = self.grid_w // merge_size
        self.frame_seqlen = self.grid_h_merged * self.grid_w_merged
        # Vision auxiliary tensors (pre-computed, constant for this config)
        self.register_buffer("pos_embeds", pos_embeds.half())
        self.register_buffer("rotary_cos", rotary_cos.half())
        self.register_buffer("rotary_sin", rotary_sin.half())
        self.register_buffer("attention_mask", attention_mask.to(torch.int8))

    def forward(self, video_frames):
        # video_frames: [num_frames, 3, H, W] uint8 or float32 (0-255)
        video_frames = video_frames.float()

        # Resize to target spatial dims: only when dynamic mode and shape mismatches
        if self.dynamic_shape or (video_frames.shape[2] != self.target_h or video_frames.shape[3] != self.target_w):
            video_frames = F.interpolate(
                video_frames,
                size=[self.target_h, self.target_w],
                mode='bilinear',
                align_corners=False
            )

        # Temporal + spatial patching (fused, max 8D to stay within CUDA Transpose limit):
        # [num_frames, 3, H, W] -> [grid_t, tps*3, grid_h//merge, merge, ps, grid_w//merge, merge, ps]
        video_frames = video_frames.reshape(
            -1, self.temporal_patch_size * 3,
            self.grid_h_merged, self.merge_size, self.patch_size,
            self.grid_w_merged, self.merge_size, self.patch_size
        )
        # Permute (8D): [grid_t, grid_h//merge, grid_w//merge, merge, merge, tps*3, ps, ps]
        video_frames = video_frames.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # Reshape to split tps*3 back into [tps, 3] and flatten spatial batch
        video_frames = video_frames.reshape(-1, self.temporal_patch_size, 3, self.patch_size, self.patch_size)
        # Swap to: [total_patches, 3, temporal_patch_size, patch_size, patch_size]
        video_frames = video_frames.transpose(1, 2)

        if self.dynamic_shape:
            grid_t = video_frames.shape[0] // (self.frame_seqlen)
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


# ══════════════════════════════════════════════════════════════════════════════
class LLM_VISION(torch.nn.Module):
    """Unified vision encoder for image AND video inputs (shared weights).

    Accepts pre-processed patches and mode-specific auxiliary tensors as inputs,
    allowing a single ONNX model to serve both image and video modes.

    Inputs:
        pixel_values:   [total_patches, 3, temporal_patch_size, patch_size, patch_size]
        pos_embeds:     [1, seq_len, embed_dim]
        rotary_cos:     [1, 1, 1, seq_len, head_dim*2]  (cos with doubled last dim)
        rotary_sin:     [1, 1, 1, seq_len, head_dim*2]  (sin with sign-flip pattern)
        attention_mask: [1, 1, seq_len, seq_len]         (0=attend, -1e9=block)

    For images: seq_len = grid_h * grid_w, attention_mask = zeros (global attention)
    For video:  seq_len = grid_t * grid_h * grid_w, attention_mask = block-diagonal
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        visual_model   = self.llm.model.visual
        vision_config  = self.llm.config.vision_config
        self.num_heads = vision_config.num_heads
        self.qk_heads  = self.num_heads // 3 * 2
        self.head_dim  = vision_config.hidden_size // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self.patch_size = visual_model.patch_size
        self.merge_size = visual_model.spatial_merge_size
        self.embed_dim  = visual_model.patch_embed.embed_dim
        self.batch_size = 1

        # Pre-build deepstack layer index mapping for O(1) lookup
        deepstack_indices = visual_model.deepstack_visual_indexes
        self._deepstack_map = {}  # layer_num -> (index, module)
        for idx, layer_num in enumerate(deepstack_indices):
            self._deepstack_map[layer_num] = idx

        # Fuse weight optimizations (norm folding, QK scaling)
        scaling = self.head_dim ** -0.25
        embed_dim_patch = visual_model.patch_embed.embed_dim
        for blk in visual_model.blocks:
            blk.attn.qkv.weight.data[:-embed_dim_patch] *= scaling
            blk.attn.qkv.bias.data[:-embed_dim_patch] *= scaling
            self.fuse_norm(blk.norm1, blk.attn.qkv)
            self.fuse_norm(blk.norm2, blk.mlp.linear_fc1)
        for deepstack_layer in visual_model.deepstack_merger_list:
            self.fuse_norm(deepstack_layer.norm, deepstack_layer.linear_fc1)
        self.fuse_norm(visual_model.merger.norm, visual_model.merger.linear_fc1)

        # Fuse pixel normalization ((x/255 - mean) / std) into Conv3d patch_embed.proj
        # Reads image_mean and image_std from vision config, falling back to preprocessor defaults
        image_mean = torch.tensor(getattr(llm.config.vision_config, 'image_mean', [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1, 1)
        image_std = torch.tensor(getattr(llm.config.vision_config, 'image_std', [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1, 1)
        # Fused: y = W/(255*std) * x + (b - W*mean/std summed over input dims)
        # Conv3d weight shape: [out_channels, in_channels=3, T, H, W]
        proj = visual_model.patch_embed.proj
        proj.weight.data.div_(255.0 * image_std)
        bias_offset = (proj.weight.data * (255.0 * image_mean)).sum(dim=[1, 2, 3, 4])
        if proj.bias is not None:
            proj.bias.data.sub_(bias_offset)
        else:
            proj.bias = torch.nn.Parameter(-bias_offset)

    def fuse_norm(self, norm, linear):
        norm_bias   = norm.bias.data
        norm_weight = norm.weight.data
        if linear.weight.shape[1] != norm_bias.shape[0]:
            repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
            norm_bias     = norm_bias.repeat(repeat_factor)
            norm_weight   = norm_weight.repeat(repeat_factor)
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias   = None

    def rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, pixel_values, pos_embeds, rotary_cos, rotary_sin, attention_mask):
        """
        Args:
            pixel_values:   [total_patches, 3, temporal_patch_size, patch_size, patch_size]
            pos_embeds:     [1, seq_len, embed_dim]
            rotary_cos:     [1, 1, 1, seq_len, head_dim*2]
            rotary_sin:     [1, 1, 1, seq_len, head_dim*2]
            attention_mask: [1, 1, seq_len, seq_len]
        """

        # Cast float16 inputs back to float32
        pos_embeds = pos_embeds.float()
        rotary_cos = rotary_cos.float()
        rotary_sin = rotary_sin.float()
        attention_mask = attention_mask.float()

        # Patch embedding via Conv3d
        vision_hidden_states = self.llm.model.visual.patch_embed.proj(pixel_values)
        vision_hidden_states = vision_hidden_states.view(self.batch_size, -1, self.embed_dim)

        # Add position embeddings
        vision_hidden_states = vision_hidden_states + pos_embeds

        # Run transformer blocks
        deepstack_features = []
        deepstack_modules = self.llm.model.visual.deepstack_merger_list
        for layer_num, blk in enumerate(self.llm.model.visual.blocks):
            hidden_states_norm = blk.norm1(vision_hidden_states)
            qkv      = blk.attn.qkv(hidden_states_norm)
            qkv      = qkv.reshape(self.batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv      = qkv.permute(2, 0, 3, 1, 4)
            qk, v    = qkv.split([2, 1], dim=0)
            qk_rot   = qk * rotary_cos + self.rotate_half(qk, self.batch_size) * rotary_sin
            q_rot, k_rot = qk_rot.split([1, 1], dim=0)
            attn     = torch.matmul(q_rot, k_rot.transpose(-1, -2))
            attn     = attn + attention_mask
            attn     = torch.softmax(attn, dim=-1)
            attn     = torch.matmul(attn, v)
            attn     = attn.transpose(2, 3).reshape(self.batch_size, -1, blk.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + blk.attn.proj(attn)
            mlp_out  = blk.mlp.linear_fc1(blk.norm2(vision_hidden_states))
            mlp_out  = blk.mlp.act_fn(mlp_out)
            mlp_out  = blk.mlp.linear_fc2(mlp_out)
            vision_hidden_states = vision_hidden_states + mlp_out
            if layer_num in self._deepstack_map:
                idx      = self._deepstack_map[layer_num]
                ds_layer = deepstack_modules[idx]
                x_ds     = vision_hidden_states.view(self.batch_size, -1, ds_layer.hidden_size)
                x_ds     = ds_layer.norm(x_ds)
                x_ds     = ds_layer.linear_fc1(x_ds)
                x_ds     = ds_layer.act_fn(x_ds)
                x_ds     = ds_layer.linear_fc2(x_ds)
                deepstack_features.append(x_ds)

        # Merger
        vision_hidden_states = self.llm.model.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(self.batch_size, -1, self.llm.model.visual.merger.hidden_size)
        vision_hidden_states = self.llm.model.visual.merger.linear_fc1(vision_hidden_states)
        vision_hidden_states = self.llm.model.visual.merger.act_fn(vision_hidden_states)
        vision_hidden_states = self.llm.model.visual.merger.linear_fc2(vision_hidden_states)
        return tuple(deepstack_features), vision_hidden_states


# ══════════════════════════════════════════════════════════════════════════════
# Vision-Text Concatenation Module (Multi-Image)
# ══════════════════════════════════════════════════════════════════════════════
class LLM_CONCAT_IMAGE(torch.nn.Module):
    """Concatenates vision features into text hidden states using a pre-computed static pattern.

    The prompt structure is deterministic for a fixed VISION_BATCH_SIZE:
      [head=4 text tokens] [img0: image_seqlen] [2 between tokens] [img1: image_seqlen] ... [tail text]

    Since the insertion pattern is known at export time, all slice offsets are constant —
    no segment_offsets input needed. The ONNX graph uses only constant-offset slices + cat.

    Inputs:
      - deepstack features: per-image deepstack from vision encoder
      - text_hidden_states: embeddings of all text tokens
      - vision_hidden_states: all image vision features concatenated [1, total_vision_tokens, hidden]

    Output layout (final sequence):
      [head(4)] [vis_0(M)] [between(2)] [vis_1(M)] [between(2)] ... [vis_N-1(M)] [tail(variable)]
    """

    def __init__(self, num_images, image_seqlen, hidden_size, deepstack_features_len, max_seq_len,
                 prompt_head_len=4, between_len=2):
        super().__init__()
        self.num_images = num_images
        self.image_seqlen = image_seqlen
        self.hidden_size = hidden_size
        self.deepstack_features_len = deepstack_features_len
        self.prompt_head_len = prompt_head_len
        self.between_len = between_len
        # Total fixed text tokens consumed by the structural markers (head + betweens)
        self.fixed_text_consumed = prompt_head_len + max(0, num_images - 1) * between_len
        self.total_vision_tokens = num_images * image_seqlen
        # Pre-allocate constant-size zero buffers for deepstack (no slicing needed in forward)
        self.register_buffer("zeros_head", torch.zeros(1, prompt_head_len, hidden_size))
        self.register_buffer("zeros_between", torch.zeros(1, between_len, hidden_size))
        # Large zeros buffer for tail (dynamic length, sliced once per forward call)
        self.register_buffer("zeros_tail_buf", torch.zeros(1, max_seq_len, hidden_size))
        self.register_buffer("zeros_tail_pad", torch.zeros(1, max_seq_len, hidden_size, dtype=torch.int8))
        # Pre-compute constant slice indices (all known at export time)
        self._vis_slices = [(i * image_seqlen, (i + 1) * image_seqlen) for i in range(num_images)]
        self._between_slices = [(prompt_head_len + i * between_len, prompt_head_len + (i + 1) * between_len) for i in range(max(0, num_images - 1))]
        self._tail_start = self.fixed_text_consumed
        self._ds_range = range(deepstack_features_len)
        self._img_range = range(num_images)

    def forward(self, *all_inputs):
        # Inputs: [ds_feat_0, ..., ds_feat_N-1, text_hidden_states, vision_hidden_states]
        text_hidden_states   = all_inputs[self.deepstack_features_len]
        vision_hidden_states = all_inputs[self.deepstack_features_len + 1]

        # Static slicing: all offsets are constants known at export time
        pieces = []

        # Head: first prompt_head_len text tokens
        pieces.append(text_hidden_states[:, :self.prompt_head_len])

        for img_idx in self._img_range:
            # Vision segment (constant offset slice)
            vis_start, vis_end = self._vis_slices[img_idx]
            pieces.append(vision_hidden_states[:, vis_start:vis_end])

            # Between text tokens (except after last image)
            if img_idx < self.num_images - 1:
                bt_start, bt_end = self._between_slices[img_idx]
                pieces.append(text_hidden_states[:, bt_start:bt_end])

        # Tail: remaining text tokens (only dynamic part — variable query length)
        tail_len = text_hidden_states.shape[1] - self._tail_start
        pieces.append(text_hidden_states[:, self._tail_start:])

        concat_hidden_states = torch.cat(pieces, dim=1)

        # Deepstack: pre-allocated zeros with vision features
        # Structure per d: [zeros(head)] [vis_d_0] [zeros(between)] [vis_d_1] ... [vis_d_N-1] [zeros(tail)]
        # Hoist static slices out of the d-loop: zeros_head/zeros_between are fixed buffers,
        # zeros_tail is sliced once here for all iterations.
        zeros_tail = self.zeros_tail_buf[:, :tail_len]
        deepstack_features = []
        for d in self._ds_range:
            ds_parts = [self.zeros_head]
            for img_idx in self._img_range:
                vis_start, vis_end = self._vis_slices[img_idx]
                ds_parts.append(all_inputs[d][:, vis_start:vis_end])
                if img_idx < self.num_images - 1:
                    ds_parts.append(self.zeros_between)
            ds_parts.append(zeros_tail)
            deepstack_features.append(torch.cat(ds_parts, dim=1))
        return deepstack_features, concat_hidden_states


# ══════════════════════════════════════════════════════════════════════════════
# Vision-Text Concatenation Module (Video — multi-segment)
# ══════════════════════════════════════════════════════════════════════════════
class LLM_CONCAT_VIDEO(torch.nn.Module):
    """Replaces fixed exported video placeholder spans with vision features.

    For video, vision tokens are interleaved with timestamp text tokens:
      [text_before] [ts1_text] [frame1_vision] [ts2_text] [frame2_vision] ... [text_after]

    This module takes:
      - text_hidden_states: embeddings of all text tokens (including timestamps)
      - vision_hidden_states: all vision features concatenated [total_vision_tokens, hidden]

    The tokenized export prompt places all video placeholder spans at deterministic
    offsets for a fixed video configuration. Baking those offsets and the derived
    text slice ranges into the module keeps `forward()` close to the simpler
    Qwen3.5 static-slice concat path while preserving the extra deepstack outputs
    that QwenVL needs.
    """

    def __init__(self, segment_offsets, frame_seqlen, hidden_size, deepstack_features_len, max_seq_len):
        super().__init__()
        self.segment_offsets = tuple(int(offset) for offset in segment_offsets)
        self.num_frames = len(self.segment_offsets)
        self.frame_seqlen = frame_seqlen
        self.hidden_size = hidden_size
        self.deepstack_features_len = deepstack_features_len
        self.total_vision_tokens = self.num_frames * frame_seqlen
        # Pre-compute constant vision slice indices and range objects
        self._vis_slices = [(f * frame_seqlen, f * frame_seqlen + frame_seqlen) for f in range(self.num_frames)]
        text_cursor = 0
        self._text_slices = []
        for offset in self.segment_offsets:
            self._text_slices.append((text_cursor, offset))
            text_cursor = offset + self.frame_seqlen
        self._text_chunk_lens = tuple(text_end - text_start for text_start, text_end in self._text_slices)
        self._tail_start = text_cursor
        self.register_buffer("zeros_seq_pad", torch.zeros(1, max_seq_len, hidden_size, dtype=torch.int8))
        self._ds_range = range(deepstack_features_len)
        self._frame_range = range(self.num_frames)

    def forward(self, *all_inputs):
        # Inputs: [ds_feat_0, ..., ds_feat_N-1, text_hidden_states, vision_hidden_states]
        deepstack_inputs = all_inputs[:self.deepstack_features_len]
        text_hidden_states = all_inputs[self.deepstack_features_len]
        vision_hidden_states = all_inputs[self.deepstack_features_len + 1]
        zero_dtype = text_hidden_states.dtype
        static_zero_chunks = {
            text_len: self.zeros_seq_pad[:, :text_len].to(zero_dtype)
            for text_len in self._text_chunk_lens
        }

        # Replace each [<|video_pad|>] * frame_seqlen span with the matching vision segment.
        pieces = []
        ds_pieces = [[] for _ in self._ds_range]

        for f in self._frame_range:
            # Text tokens before this vision segment
            text_start, text_end = self._text_slices[f]
            text_chunk = text_hidden_states[:, text_start:text_end]
            pieces.append(text_chunk)
            zeros_chunk = static_zero_chunks[self._text_chunk_lens[f]]
            for d in self._ds_range:
                ds_pieces[d].append(zeros_chunk)

            # Vision segment for this frame (pre-computed slice indices)
            vis_start, vis_end = self._vis_slices[f]
            vis_chunk = vision_hidden_states[:, vis_start:vis_end]
            pieces.append(vis_chunk)
            for d in self._ds_range:
                ds_pieces[d].append(deepstack_inputs[d][:, vis_start:vis_end])

        # Remaining text tokens after all replaced placeholder spans.
        tail_chunk = text_hidden_states[:, self._tail_start:]
        pieces.append(tail_chunk)
        tail_len = text_hidden_states.shape[1] - self._tail_start
        tail_zeros = self.zeros_seq_pad[:, :tail_len].to(zero_dtype)
        for d in self._ds_range:
            ds_pieces[d].append(tail_zeros)

        concat_hidden_states = torch.cat(pieces, dim=1)
        deepstack_features = [torch.cat(ds_pieces[d], dim=1) for d in self._ds_range]
        return deepstack_features, concat_hidden_states


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask (Image Context — Multi-Image)
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_IMAGE_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for the multi-image+text prefill phase.

    Builds the full multimodal 3D position table at export time using the same logic as
    upstream `get_rope_index`:
      - Iterate over mm_token_type_ids segments
      - For text: linear positions on all 3 channels
      - For image: use 3D spatial positions (temporal=const, height grid, width grid)
                   with current_pos advancing by max(H,W)//merge per image
    """

    def __init__(self, llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        super().__init__()
        total_len = len(mm_token_type_ids)
        total_max = max_seq_len + total_len

        self.register_buffer("attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128)

        cos, sin = self._build_image_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    @staticmethod
    def _build_image_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        """Build multi-image-aware mRoPE position table with mm_token_type_ids-driven position assignment.

        For each image segment, assigns 3D positions:
          - Temporal: constant (=current_pos, since images have T=1)
          - Height:   arange(H//merge) + current_pos
          - Width:    arange(W//merge) + current_pos
        After each image, current_pos advances by max(H, W) // spatial_merge_size.
        """
        spatial_merge_size = llm.config.vision_config.spatial_merge_size

        # Group mm_token_type_ids by modality
        input_type_group = []
        for key, group in itertools.groupby(enumerate(mm_token_type_ids), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        # Build position IDs mimicking get_rope_index
        current_pos = 0
        llm_pos_ids_list = []
        image_iter = iter(image_grid_thw)

        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                # Text segment
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                # Image segment (modality_type == 1)
                grid_thw = next(image_iter)
                llm_grid_h = grid_thw[1].item() // spatial_merge_size
                llm_grid_w = grid_thw[2].item() // spatial_merge_size
                image_seqlen = llm_grid_h * llm_grid_w

                # Temporal positions (constant for images, T=1)
                position_temporal = torch.full((image_seqlen,), float(current_pos))
                # Height positions
                position_height = (torch.arange(llm_grid_h, dtype=torch.float32) + current_pos).repeat_interleave(llm_grid_w)
                # Width positions
                position_width = (torch.arange(llm_grid_w, dtype=torch.float32) + current_pos).repeat(llm_grid_h)

                vision_pos = torch.stack([position_temporal, position_height, position_width], dim=0)
                llm_pos_ids_list.append(vision_pos)
                current_pos += max(llm_grid_h, llm_grid_w)

        # Assemble prefill position IDs
        prefill_positions = torch.cat(llm_pos_ids_list, dim=1)

        # Append decode tail positions (linear from current_pos)
        tail_len = max_seq_len
        tail_positions = torch.arange(tail_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos
        position_ids = torch.cat([prefill_positions, tail_positions], dim=-1)

        # Compute rotary embeddings
        position_ids = position_ids.unsqueeze(1)  # [3, 1, total_len]
        inv_freq_expanded = llm.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(
            llm.model.language_model.rotary_emb, freqs, llm.model.language_model.rotary_emb.mrope_section
        )
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len     = ids_len + history_len
        rotary_cos     = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin     = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_IMAGE_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for a single decode step (multi-image context)."""

    def __init__(self, llm, mm_token_type_ids, image_grid_thw, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_IMAGE_PREFILL._build_image_rotary_table(llm, mm_token_type_ids, image_grid_thw, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos      = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin      = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask (Video Context)
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_VIDEO_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for the video+text prefill phase.

    Builds the full multimodal 3D position table at export time using the same logic as
    upstream `get_rope_index`:
      - Iterate over mm_token_type_ids segments
      - For text: linear positions on all 3 channels
      - For video frames: use get_vision_position_ids with current_pos advancing by max(H,W)//merge
    """

    def __init__(self, llm, mm_token_type_ids, video_grid_thw, max_seq_len):
        super().__init__()
        total_len = len(mm_token_type_ids)
        total_max = max_seq_len + total_len

        self.register_buffer("attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128)

        cos, sin = self._build_video_rotary_table(llm, mm_token_type_ids, video_grid_thw, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    @staticmethod
    def _build_video_rotary_table(llm, mm_token_type_ids, video_grid_thw, max_seq_len):
        """Build video-aware mRoPE position table with mm_token_type_ids-driven position assignment."""
        spatial_merge_size = llm.config.vision_config.spatial_merge_size

        # Expand video_grid_thw per-frame (same as upstream get_rope_index)
        expanded_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        expanded_grid_thw[:, 0] = 1

        # Group mm_token_type_ids by modality
        input_type_group = []
        for key, group in itertools.groupby(enumerate(mm_token_type_ids), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        # Build position IDs mimicking get_rope_index
        current_pos = 0
        llm_pos_ids_list = []
        video_frame_iter = iter(expanded_grid_thw)

        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                # Text segment
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                # Video segment (modality_type == 2)
                grid_thw = next(video_frame_iter)
                llm_grid_h = grid_thw[1].item() // spatial_merge_size
                llm_grid_w = grid_thw[2].item() // spatial_merge_size
                frame_seqlen = llm_grid_h * llm_grid_w

                # Temporal positions (constant for single-frame grids after expansion)
                position_temporal = torch.full((frame_seqlen,), float(current_pos))
                # Height positions
                position_height = (torch.arange(llm_grid_h, dtype=torch.float32) + current_pos).repeat_interleave(llm_grid_w)
                # Width positions
                position_width = (torch.arange(llm_grid_w, dtype=torch.float32) + current_pos).repeat(llm_grid_h)

                vision_pos = torch.stack([position_temporal, position_height, position_width], dim=0)
                llm_pos_ids_list.append(vision_pos)
                current_pos += max(llm_grid_h, llm_grid_w)

        # Assemble prefill position IDs
        prefill_positions = torch.cat(llm_pos_ids_list, dim=1)

        # Append decode tail positions (linear from current_pos)
        tail_len = max_seq_len
        tail_positions = torch.arange(tail_len, dtype=torch.float32).view(1, -1).expand(3, -1) + current_pos
        position_ids = torch.cat([prefill_positions, tail_positions], dim=-1)

        # Compute rotary embeddings
        position_ids = position_ids.unsqueeze(1)  # [3, 1, total_len]
        inv_freq_expanded = llm.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(
            llm.model.language_model.rotary_emb, freqs, llm.model.language_model.rotary_emb.mrope_section
        )
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len     = ids_len + history_len
        rotary_cos     = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin     = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_VIDEO_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for a single decode step (video context)."""

    def __init__(self, llm, mm_token_type_ids, video_grid_thw, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_VIDEO_PREFILL._build_video_rotary_table(llm, mm_token_type_ids, video_grid_thw, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos      = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin      = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask (Text-Only Context)
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_TEXT_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for the text-only prefill phase."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        self.register_buffer("attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128)

        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos,  cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        """Build text-only mRoPE position table (all 3 channels use linear positions)."""
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).repeat(3, 1, 1)
        inv_freq_expanded = llm.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(llm.model.language_model.rotary_emb, freqs, llm.model.language_model.rotary_emb.mrope_section)
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len     = ids_len + history_len
        rotary_cos     = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin     = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_TEXT_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for a single decode step (text-only context)."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_TEXT_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos,  cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos      = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin      = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_MAIN(torch.nn.Module):
    """
    Main transformer module for QwenVL that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (mRoPE)
      - KV cache management with optional Q8/Q8_CUDA quantization
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
      - Deepstack feature injection from the vision encoder
    """

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size, deepstack_features_len):
        super().__init__()
        self.llm = llm

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim             = head_dim
        self.head_dim_half        = head_dim // 2
        self.head_dim_quarter     = head_dim // 4
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads             = num_heads + num_key_value_heads

        # ── Layer count multipliers (for indexing into flat KV input list) ──
        self.num_layers   = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # ── Deepstack features ───────────────────────────────────────────
        self.deepstack_features_len = deepstack_features_len
        self._ds_offset             = 3 + deepstack_features_len  # 3 trailing args: cos, sin, attention_mask

        # ── KV cache dtype flags ─────────────────────────────────────────
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

        # Whether Q8 modes use per-group quantization (enabled by hadamard/shuffle)
        self.kv_q8_grouped      = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim

        # head_dim used for int32 unpack in rotary CUDA modes
        self.kv_unpack_head_dim = (head_dim // 2) if self.kv_rotary_q4_cuda else head_dim
        self.kv_pack_quarter    = (head_dim // 8) if self.kv_rotary_q4_cuda else (head_dim // 4)

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
        self.register_buffer("overflow_scale", torch.tensor([0.01], dtype=torch.float32))
        rms_norm_eps = llm.config.text_config.rms_norm_eps
        rms_eps_hidden = rms_norm_eps * hidden_size
        rms_eps_head = rms_norm_eps * head_dim
        if PREVENT_F16_OVERFLOW:
            rms_eps_hidden *= 0.0001  # overflow_scale^2
            rms_eps_head *= 0.0001
        self.register_buffer("rms_eps_hidden", torch.tensor([rms_eps_hidden], dtype=torch.float32))
        self.register_buffer("rms_eps_head", torch.tensor([rms_eps_head], dtype=torch.float32))

        # ── Pre-compute MLP split size (constant across all layers) ───────
        self.mlp_intermediate_size = llm.model.language_model.layers[0].mlp.down_proj.in_features
        # ── Pre-compute attention output projection size ─────────────────
        self.attn_o_proj_in_features = llm.model.language_model.layers[0].self_attn.o_proj.in_features

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_any_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias  = [None] * num_layers
                self.save_v_bias  = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self._replace_gelu_with_tanh_approximation(self.llm)
        self._fuse_weights(hidden_size)

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Absorb final RMSNorm into lm_head
            final_norm_weight = self.llm.model.language_model.norm.weight.unsqueeze(0) * norm_factor
            self.llm.lm_head.weight.mul_(final_norm_weight)
            del self.llm.model.language_model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features  = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias     = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:

            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)

            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        attn.q_out_features = int(q_proj.out_features)
        attn.k_out_features = int(k_proj.out_features)
        attn.v_out_features = int(v_proj.out_features)

        del attn.q_proj, attn.k_proj, attn.v_proj

        # ── Fuse QK norms (absorb scale factors) ────────────────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)

        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        # ── Absorb input LayerNorm into QKV weights ─────────────────
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
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                LLM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, eps):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)  # Note, not the .mean()

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        # Input layout: [kv_caches..., hidden_states, ds_feat_0..N, rotary_cos, rotary_sin, attention_mask]
        hidden_states      = all_inputs[-(self._ds_offset + 1)]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]

        for i, layer in enumerate(self.llm.model.language_model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps_hidden)

            # Fused QKV projection & reshape
            qkv   = layer.self_attn.qkv(hidden_states)
            qkv   = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)

            # QK normalization & rotary embedding
            qk     = self._rms_norm(qk, self.rms_eps_head) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Split into query and key, reshape query for GQA
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q    = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q    = q.permute(0, 2, 3, 1, 4)

            # Optional FP16 cast for KV
            if self.kv_f16:
                k = k.half()
                v = v.half()

            # Transpose K and V into cache layout
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            if self.kv_rotary_q4:
                # ── ROTARY_Q4 ────────────────────────────────────────
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                    q_rot      = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                    v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                    q_rot      = self.quantizer.rotate_q(q, batch_size)
                    if self.quantizer.use_shuffle:
                        q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                    q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    q_rot_g    = q_rot_g.transpose(-2, -3)
                    if self.quantizer.use_hadamard:
                        q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                    k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                    attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                    q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                    attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                    attn       = torch.softmax(attn, dim=-1)

                    v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                    v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                    v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                    attn       = torch.matmul(attn, v_dequant)
                    if self.quantizer.use_hadamard:
                        attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                    if self.quantizer.use_shuffle:
                        attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)

            elif self.kv_rotary:
                # ── ROTARY_Q8 ────────────────────────────────────────
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k_signed)
                        attn          = attn_raw * k_s + attention_mask
                        attn          = torch.softmax(attn, dim=-1)

                        v_scaled  = v_signed * v_s
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                    if self.kv_q8_grouped:
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        q_rot         = self.quantizer.rotate_q(q, batch_size)
                        attn_raw      = torch.matmul(q_rot, k.float())
                        q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                        attn_bias     = q_bias_factor * k_b + attention_mask
                        attn          = torch.addcmul(attn_bias, attn_raw, k_s)
                        attn          = torch.softmax(attn, dim=-1)

                        v_scaled  = v.float() * v_s
                        bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                        attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size) + bias_term

            elif self.kv_quantized:
                if self.kv_sym:
                    packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                    k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                    v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(q, k_signed)
                        attn     = attn_raw * k_s + attention_mask
                        attn     = torch.softmax(attn, dim=-1)

                        v_scaled  = v_signed * v_s
                        attn      = torch.matmul(attn, v_scaled)
                else:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
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
                        k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        q_in = q
                        if self.quantizer.use_shuffle:
                            q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                        q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_g    = q_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_g = self.quantizer.hadamard_q(q_g)
                        k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_g, k_q_g)
                        q_sum_g    = q_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)

                        v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
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
                # Concatenate with cached K/V (F16 or F32)
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i]   = k
                self.save_value[i] = v

                if self.kv_f16:
                    k = k.float()
                    v = v.float()

                attn = torch.matmul(q, k) + attention_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)

            # Output projection & residual
            attn          = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.attn_o_proj_in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps_hidden)

            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [self.mlp_intermediate_size, self.mlp_intermediate_size], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

            # ── Apply deepstack features from vision encoder ─────────
            if i < self.deepstack_features_len:
                hidden_states = all_inputs[i - self._ds_offset] + hidden_states

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1], self.rms_eps_hidden)
        logits        = self.llm.lm_head(hidden_states)

        if self.kv_sym:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale, logits
        elif self.kv_any_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model = Qwen3VLForConditionalGeneration.from_pretrained(download_path, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()

        num_heads              = model.config.text_config.num_attention_heads
        num_key_value_heads    = model.config.text_config.num_key_value_heads
        head_dim               = model.config.text_config.head_dim
        num_layers             = model.config.text_config.num_hidden_layers
        hidden_size            = model.config.text_config.hidden_size
        vocab_size             = model.model.language_model.vocab_size
        deepstack_features_len = len(model.model.visual.deepstack_visual_indexes)
        scale_dtype            = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

        for note in normalize_kv_quant_settings(head_dim):
            print(f"\n{note}")

        # ══════════════════════════════════════════════════════════════════
        # Build Dummy Tensors for Tracing
        # ══════════════════════════════════════════════════════════════════
        batch_size  = 1
        ids_len     = torch.tensor([10], dtype=torch.int64)
        history_len = torch.tensor([0],  dtype=torch.int64)
        kv_seq_len  = ids_len + history_len
        logits      = torch.ones((batch_size, vocab_size), dtype=torch.float32)

        # KV cache spec: list of (name, concat_dim)
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

        # Determine KV tensor shapes based on quantization mode
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
            'key':   torch.zeros((batch_size, num_key_value_heads, 1, k_head, history_len), dtype=kv_dtype),
            'value': torch.zeros((batch_size, num_key_value_heads, 1, history_len, v_head), dtype=kv_dtype)
        }
        if KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"):
            if _grouped_6d:
                kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE
                scale_k_dim3  = kv_quant_num_groups
                scale_v_dim4  = kv_quant_num_groups
                kv_tensors.update({
                    'key_scale':   torch.ones((batch_size, num_key_value_heads, 1, scale_k_dim3, 1, history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_key_value_heads, 1, history_len, scale_v_dim4, 1), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias':    torch.ones((batch_size, num_key_value_heads, 1, scale_k_dim3, 1, history_len), dtype=scale_dtype),
                        'value_bias':  torch.ones((batch_size, num_key_value_heads, 1, history_len, scale_v_dim4, 1), dtype=scale_dtype),
                    })
            else:
                scale_k_dim3  = 1
                scale_v_dim4  = 1
                kv_tensors.update({
                    'key_scale':   torch.ones((batch_size, num_key_value_heads, 1, scale_k_dim3, history_len), dtype=scale_dtype),
                    'value_scale': torch.ones((batch_size, num_key_value_heads, 1, history_len, scale_v_dim4), dtype=scale_dtype),
                })
                if not _kv_sym:
                    kv_tensors.update({
                        'key_bias':    torch.ones((batch_size, num_key_value_heads, 1, scale_k_dim3, history_len), dtype=scale_dtype),
                        'value_bias':  torch.ones((batch_size, num_key_value_heads, 1, history_len, scale_v_dim4), dtype=scale_dtype),
                    })

        # ══════════════════════════════════════════════════════════════════
        # Helper: Build KV I/O names, tensors, and dynamic axes
        # ══════════════════════════════════════════════════════════════════
        def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                tensor = tensors_dict[name]
                for i in range(num_layers):
                    in_n  = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(tensor)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {0: batch_axis, dim: seq_axis}
                    axes[out_n] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Embed
        # ══════════════════════════════════════════════════════════════════
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['text_hidden_states'],
            dynamic_axes={
                'input_ids':          {0: 'batch', 1: 'ids_len'},
                'text_hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del input_ids
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Image_Preprocess
        # ══════════════════════════════════════════════════════════════════
        visual_model = model.model.visual

        # --- Pre-compute auxiliary tensors for MULTI-IMAGE config ---
        # Single-image grid: [1, H_grid, W_grid]
        single_image_grid_thw = torch.tensor([[1, HEIGHT_FACTOR * 2, WIDTH_FACTOR * 2]], dtype=torch.int32)
        single_image_seq_len = int(single_image_grid_thw[0, 1] * single_image_grid_thw[0, 2])
        image_seq_per_image = HEIGHT_FACTOR * WIDTH_FACTOR  # merged token count per image

        # Multi-image grid: [N, 1, H_grid, W_grid] — N identical images for position/rotary computation
        multi_image_grid_thw = single_image_grid_thw.repeat(VISION_BATCH_SIZE, 1)
        multi_image_total_seq = single_image_seq_len * VISION_BATCH_SIZE

        # Position embeddings: compute per-image and tile for N images
        single_pos_embeds = Qwen3VLVisionModel.fast_pos_embed_interpolate(visual_model, single_image_grid_thw)
        image_pos_embeds = single_pos_embeds.repeat(VISION_BATCH_SIZE, 1).unsqueeze(0)  # [1, N*seq_per_image, embed_dim]

        # Rotary: compute per-image and tile for N images
        single_rotary_raw = Qwen3VLVisionModel.rot_pos_emb(visual_model, single_image_grid_thw).float()
        multi_rotary_raw = single_rotary_raw.repeat(VISION_BATCH_SIZE, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        image_rotary_cos = torch.cat([multi_rotary_raw.cos(), multi_rotary_raw.cos()], dim=-1)
        image_rotary_sin = torch.cat([-multi_rotary_raw.sin(), multi_rotary_raw.sin()], dim=-1)

        # Block-diagonal attention mask for multi-image (each image attends only to itself)
        if VISION_BATCH_SIZE > 1:
            image_attn_mask = torch.full((1, 1, multi_image_total_seq, multi_image_total_seq), -1e9, dtype=torch.float32)
            for img_idx in range(VISION_BATCH_SIZE):
                start = img_idx * single_image_seq_len
                end = start + single_image_seq_len
                image_attn_mask[:, :, start:end, start:end] = 0.0
        else:
            image_attn_mask = torch.zeros(1, 1, single_image_seq_len, single_image_seq_len, dtype=torch.float32)

        # --- Pre-compute auxiliary tensors for VIDEO config ---
        video_grid_t = VIDEO_NUM_FRAMES // TEMPORAL_PATCH_SIZE
        video_grid_thw = torch.tensor([[video_grid_t, VIDEO_HEIGHT_FACTOR * 2, VIDEO_WIDTH_FACTOR * 2]], dtype=torch.int32)
        video_seq_len = int(video_grid_thw[0, 0] * video_grid_thw[0, 1] * video_grid_thw[0, 2])
        video_frame_seqlen = int(video_grid_thw[0, 1] * video_grid_thw[0, 2])
        video_pos_embeds = Qwen3VLVisionModel.fast_pos_embed_interpolate(visual_model, video_grid_thw).unsqueeze(0)
        video_rotary_raw = Qwen3VLVisionModel.rot_pos_emb(visual_model, video_grid_thw).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        video_rotary_cos = torch.cat([video_rotary_raw.cos(), video_rotary_raw.cos()], dim=-1)
        video_rotary_sin = torch.cat([-video_rotary_raw.sin(), video_rotary_raw.sin()], dim=-1)
        # Block-diagonal attention mask for video (per-frame independent attention)
        video_attn_mask = torch.full((1, 1, video_seq_len, video_seq_len), -1e9, dtype=torch.float32)
        for f in range(video_grid_t):
            start = f * video_frame_seqlen
            end = start + video_frame_seqlen
            video_attn_mask[:, :, start:end, start:end] = 0.0

        onnx_model_Image_Preprocess = onnx_model_Vision.replace('LLM_Vision.onnx', 'LLM_Image_Preprocess.onnx')
        image_preprocess_module = LLM_IMAGE_PREPROCESS(
            patch_size=16,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            merge_size=2,
            height_factor=HEIGHT_FACTOR,
            width_factor=WIDTH_FACTOR,
            pos_embeds=image_pos_embeds,
            rotary_cos=image_rotary_cos,
            rotary_sin=image_rotary_sin,
            attention_mask=image_attn_mask,
            dynamic_shape=DYNAMIC_IMAGE_SHAPE,
            num_images=VISION_BATCH_SIZE
        )
        pixel_values_dummy = torch.randint(0, 255, (VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
        if INPUT_IMAGE_DIM != 4:
            pixel_values_dummy = pixel_values_dummy.unsqueeze(1)
        image_preprocess_dynamic_axes = None
        if DYNAMIC_IMAGE_SHAPE:
            image_preprocess_dynamic_axes = {
                'pixel_values': {0: 'num_images', -2: 'height', -1: 'width'},
                'pixel_values_patches': {0: 'total_patches'},
                'pos_embeds': {1: 'total_vision_seq'},
                'rotary_cos': {3: 'total_vision_seq'},
                'rotary_sin': {3: 'total_vision_seq'},
                'attention_mask': {2: 'total_vision_seq', 3: 'total_vision_seq'}
            }
        torch.onnx.export(
            image_preprocess_module,
            (pixel_values_dummy,),
            onnx_model_Image_Preprocess,
            input_names=['pixel_values'],
            output_names=['pixel_values_patches', 'pos_embeds', 'rotary_cos', 'rotary_sin', 'attention_mask'],
            dynamic_axes=image_preprocess_dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del pixel_values_dummy, image_preprocess_module
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Vision (Unified — shared for image & video)
        # ══════════════════════════════════════════════════════════════════

        # Export unified vision model using multi-image config as example (dynamic axes handle both)
        vision_model = LLM_VISION(model)
        image_total_patches = VISION_BATCH_SIZE * HEIGHT_FACTOR * 2 * WIDTH_FACTOR * 2
        dummy_patches = torch.randn(image_total_patches, 3, TEMPORAL_PATCH_SIZE, 16, 16)

        vision_output_names = []
        vision_dynamic_axes = {
            'pixel_values':   {0: 'total_patches'},
            'pos_embeds':     {1: 'seq_len'},
            'rotary_cos':     {3: 'seq_len'},
            'rotary_sin':     {3: 'seq_len'},
            'attention_mask': {2: 'seq_len', 3: 'seq_len'},
            'vision_hidden_states': {1: 'seq_len'}
        }
        for i in range(deepstack_features_len):
            name = f'deepstack_feature_{i}'
            vision_output_names.append(name)
            vision_dynamic_axes[name] = {1: 'seq_len'}
        vision_output_names.append('vision_hidden_states')

        torch.onnx.export(
            vision_model,
            (dummy_patches, image_pos_embeds.half(), image_rotary_cos.half(), image_rotary_sin.half(), image_attn_mask.to(torch.int8)),
            onnx_model_Vision,
            input_names=['pixel_values', 'pos_embeds', 'rotary_cos', 'rotary_sin', 'attention_mask'],
            output_names=vision_output_names,
            dynamic_axes=vision_dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del dummy_patches, vision_model
        del image_pos_embeds, image_rotary_cos, image_rotary_sin, multi_rotary_raw
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Video_Preprocess
        # ══════════════════════════════════════════════════════════════════
        video_target_h = VIDEO_RESIZE[0]  # patch_size * merge_size = 32
        video_target_w = VIDEO_RESIZE[1]
        video_preprocess_module = LLM_VIDEO_PREPROCESS(
            patch_size=16,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            merge_size=2,
            target_h=video_target_h,
            target_w=video_target_w,
            pos_embeds=video_pos_embeds,
            rotary_cos=video_rotary_cos,
            rotary_sin=video_rotary_sin,
            attention_mask=video_attn_mask,
            dynamic_shape=DYNAMIC_VIDEO_SHAPE
        )
        video_frames_dummy = torch.randint(0, 255, (VIDEO_NUM_FRAMES, 3, INPUT_VIDEO_SIZE[0], INPUT_VIDEO_SIZE[1]), dtype=torch.uint8)
        torch.onnx.export(
            video_preprocess_module,
            (video_frames_dummy,),
            onnx_model_Video_Preprocess,
            input_names=['video_frames'],
            output_names=['pixel_values_videos', 'pos_embeds', 'rotary_cos', 'rotary_sin', 'attention_mask'],
            dynamic_axes={
                'video_frames': {0: 'num_frames', 2: 'in_height', 3: 'in_width'},
                'pixel_values_videos': {0: 'total_patches'}
            } if DYNAMIC_VIDEO_SHAPE else None,
            opset_version=OPSET,
            dynamo=False
        )
        del video_frames_dummy, video_preprocess_module
        del video_pos_embeds, video_rotary_cos, video_rotary_sin, video_rotary_raw
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Concat (Multi-Image mode — static graph, no segment_offsets)
        # ══════════════════════════════════════════════════════════════════
        prompt_head_len      = 4  # <|im_start|>user\n<|vision_start|>
        between_len          = 2  # <|vision_end|><|vision_start|> between images
        image_seqlen         = WIDTH_FACTOR * HEIGHT_FACTOR  # merged vision tokens per image
        vision_embed_size    = image_seqlen * VISION_BATCH_SIZE
        text_hidden_states   = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        vision_hidden_states = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
        deepstack_features   = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)

        concat_all_inputs  = []
        concat_input_names = []
        concat_output_names = []
        concat_dynamic_axes = {
            'text_hidden_states':   {1: 'ids_len'},
            'concat_hidden_states': {1: 'total_len'}
        }
        if DYNAMIC_IMAGE_SHAPE:
            concat_dynamic_axes['vision_hidden_states'] = {1: 'vision_embed_len'}
        for i in range(deepstack_features_len):
            in_name  = f'in_deepstack_feature_{i}'
            out_name = f'out_deepstack_feature_{i}'
            concat_input_names.append(in_name)
            concat_all_inputs.append(deepstack_features)
            if DYNAMIC_IMAGE_SHAPE:
                concat_dynamic_axes[in_name] = {1: 'vision_embed_len'}
            concat_dynamic_axes[out_name] = {1: 'total_len'}
            concat_output_names.append(out_name)
        concat_input_names.extend(['text_hidden_states', 'vision_hidden_states'])
        concat_all_inputs.extend([text_hidden_states, vision_hidden_states])
        concat_output_names.append('concat_hidden_states')

        torch.onnx.export(
            LLM_CONCAT_IMAGE(VISION_BATCH_SIZE, image_seqlen, hidden_size, deepstack_features_len, MAX_SEQ_LEN,
                       prompt_head_len=prompt_head_len, between_len=between_len),
            tuple(concat_all_inputs),
            onnx_model_Concat_Image,
            input_names=concat_input_names,
            output_names=concat_output_names,
            dynamic_axes=concat_dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del text_hidden_states, vision_hidden_states, deepstack_features, concat_all_inputs
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Concat_Video (Video mode)
        # ══════════════════════════════════════════════════════════════════
        # Compute video frame_seqlen: grid_h * grid_w / merge_size^2
        video_frame_seqlen = (VIDEO_HEIGHT_FACTOR * 2 * VIDEO_WIDTH_FACTOR * 2) // (2 * 2)
        video_num_temporal_frames = VIDEO_NUM_FRAMES // TEMPORAL_PATCH_SIZE  # grid_t
        video_total_vision_tokens = video_num_temporal_frames * video_frame_seqlen

        # Build a representative tokenized video prompt so traced offsets match
        # the in-place placeholder replacement semantics used at runtime.
        export_tokenizer_video_concat = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
        concat_video_pad_token_id = export_tokenizer_video_concat.convert_tokens_to_ids("<|video_pad|>")
        video_placeholder = ""
        for f in range(video_num_temporal_frames):
            timestamp = (f * TEMPORAL_PATCH_SIZE + TEMPORAL_PATCH_SIZE / 2) / VIDEO_FPS
            video_placeholder += f"<{timestamp:.1f} seconds>"
            video_placeholder += "<|vision_start|>" + "<|video_pad|>" * video_frame_seqlen + "<|vision_end|>"
        video_prompt = f"<|im_start|>user\n{video_placeholder}Describe this video.<|im_end|>\n<|im_start|>assistant\n"
        video_prompt_tokens = export_tokenizer_video_concat(video_prompt, return_tensors='np')['input_ids'][0]
        video_text_dummy_len = int(video_prompt_tokens.shape[0])
        video_text_hidden = torch.ones((1, video_text_dummy_len, hidden_size), dtype=torch.float32)
        video_vision_hidden = torch.ones((1, video_total_vision_tokens, hidden_size), dtype=torch.float32)
        video_deepstack_feat = torch.ones((1, video_total_vision_tokens, hidden_size), dtype=torch.float32)
        video_segment_offsets = []
        video_pad_count = 0
        for idx, tid in enumerate(video_prompt_tokens.tolist()):
            if tid == concat_video_pad_token_id:
                if video_pad_count % video_frame_seqlen == 0:
                    video_segment_offsets.append(idx)
                video_pad_count += 1

        video_concat_inputs = []
        video_concat_input_names = []
        video_concat_output_names = []
        video_concat_dynamic_axes = {
            'text_hidden_states':   {1: 'text_len'},
            'concat_hidden_states': {1: 'total_len'}
        }
        for i in range(deepstack_features_len):
            video_concat_input_names.append(f'in_deepstack_feature_{i}')
            video_concat_inputs.append(video_deepstack_feat)
            video_concat_output_names.append(f'out_deepstack_feature_{i}')
            video_concat_dynamic_axes[f'out_deepstack_feature_{i}'] = {1: 'total_len'}
        video_concat_input_names.extend(['text_hidden_states', 'vision_hidden_states'])
        video_concat_inputs.extend([video_text_hidden, video_vision_hidden])
        video_concat_output_names.append('concat_hidden_states')

        torch.onnx.export(
            LLM_CONCAT_VIDEO(video_segment_offsets, video_frame_seqlen, hidden_size, deepstack_features_len, MAX_SEQ_LEN),
            tuple(video_concat_inputs),
            onnx_model_Concat_Video,
            input_names=video_concat_input_names,
            output_names=video_concat_output_names,
            dynamic_axes=video_concat_dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del video_text_hidden, video_vision_hidden, video_deepstack_feat, video_segment_offsets, video_concat_inputs
        del export_tokenizer_video_concat, video_prompt, video_prompt_tokens, video_placeholder
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Image Prefill (Multi-Image aware)
        # ══════════════════════════════════════════════════════════════════
        # Build mm_token_type_ids for the multi-image prompt at export time
        # Structure: <|im_start|>user\n<|vision_start|><|image_pad|>*M<|vision_end|>[<|vision_start|><|image_pad|>*M<|vision_end|>]*(N-1) query<|im_end|>\n<|im_start|>assistant\n
        export_tokenizer_img = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
        image_pad_token_id = export_tokenizer_img.convert_tokens_to_ids("<|image_pad|>")

        def build_export_image_mm_types(num_images, image_seqlen_per_image):
            """Build mm_token_type_ids using the actual tokenizer for multi-image rotary table."""
            image_placeholder = ""
            for _ in range(num_images):
                image_placeholder += "<|vision_start|>" + "<|image_pad|>" * image_seqlen_per_image + "<|vision_end|>"
            prompt = f"<|im_start|>user\n{image_placeholder}Describe this image.<|im_end|>\n<|im_start|>assistant\n"
            toks = export_tokenizer_img(prompt, return_tensors='np')['input_ids'][0].tolist()
            mm_types = []
            for tid in toks:
                if tid == image_pad_token_id:
                    mm_types.append(1)
                else:
                    mm_types.append(0)
            return mm_types

        image_mm_types = build_export_image_mm_types(VISION_BATCH_SIZE, image_seqlen)
        # image_grid_thw for rotary: each image is [1, H_grid, W_grid]
        image_grid_thw_export = torch.tensor([[1, HEIGHT_FACTOR * 2, WIDTH_FACTOR * 2]], dtype=torch.int64).repeat(VISION_BATCH_SIZE, 1)
        del export_tokenizer_img

        torch.onnx.export(
            ROTARY_IMAGE_PREFILL(model, image_mm_types, image_grid_thw_export, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Image_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Image Decode
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_IMAGE_DECODE(model, image_mm_types, image_grid_thw_export, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Image_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Video Prefill
        # ══════════════════════════════════════════════════════════════════
        # Build mm_token_type_ids for the video prompt at export time
        # We use the actual tokenizer to build a representative prompt matching the configured video grid
        video_grid_thw_export = torch.tensor([[video_num_temporal_frames, VIDEO_HEIGHT_FACTOR * 2, VIDEO_WIDTH_FACTOR * 2]], dtype=torch.int64)

        # Use the actual tokenizer to generate the correct mm_token_type_ids
        export_tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
        video_pad_token_id = export_tokenizer.convert_tokens_to_ids("<|video_pad|>")

        # Build a representative video prompt with default timestamps
        def build_export_video_mm_types(num_frames, frame_seqlen, fps=2.0):
            """Build mm_token_type_ids using the actual tokenizer for accurate rotary table."""
            timestamps = [(i * TEMPORAL_PATCH_SIZE + TEMPORAL_PATCH_SIZE / 2) / fps for i in range(num_frames)]
            video_placeholder = ""
            for f in range(num_frames):
                video_placeholder += f"<{timestamps[f]:.1f} seconds>"
                video_placeholder += "<|vision_start|>" + "<|video_pad|>" * frame_seqlen + "<|vision_end|>"
            prompt = f"<|im_start|>user\n{video_placeholder}Describe this video.<|im_end|>\n<|im_start|>assistant\n"
            toks = export_tokenizer(prompt, return_tensors='np')['input_ids'][0].tolist()

            mm_types = []
            for tid in toks:
                if tid == video_pad_token_id:
                    mm_types.append(2)
                else:
                    mm_types.append(0)
            return mm_types

        video_mm_types = build_export_video_mm_types(video_num_temporal_frames, video_frame_seqlen)
        del export_tokenizer

        torch.onnx.export(
            ROTARY_VIDEO_PREFILL(model, video_mm_types, video_grid_thw_export, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Video_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Video Decode
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_VIDEO_DECODE(model, video_mm_types, video_grid_thw_export, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Video_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Text Prefill
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_TEXT_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Text_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary Text Decode
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_TEXT_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Main (Transformer Layers)
        # ══════════════════════════════════════════════════════════════════
        ids_len_main = ids_len + vision_embed_size
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)

        hidden_states     = torch.ones((batch_size, ids_len_main, hidden_size), dtype=torch.float32)
        ds_features_dummy = torch.ones((1, ids_len_main, hidden_size), dtype=torch.float32)
        rotary_cos        = torch.zeros((1, ids_len_main, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin        = rotary_cos
        attention_mask    = torch.zeros((1, 1, 1, ids_len_main, kv_seq_len + vision_embed_size), dtype=torch.float32)

        all_inputs  = kv_ins + [hidden_states]
        input_names = kv_in_names + ['hidden_states']
        dynamic_axes = {
            **kv_axes,
            'hidden_states':  {0: 'batch', 1: 'ids_len'},
            'logits':         {0: 'batch'},
            'rotary_cos':     {1: 'ids_len'},
            'rotary_sin':     {1: 'ids_len'},
            'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
        }

        for i in range(deepstack_features_len):
            name = f'deepstack_features_{i}'
            input_names.append(name)
            all_inputs.append(ds_features_dummy)
            dynamic_axes[name] = {1: 'total_len'}

        all_inputs.extend([rotary_cos, rotary_sin, attention_mask])
        input_names.extend(['rotary_cos', 'rotary_sin', 'attention_mask'])
        output_names = kv_out_names + ['logits']

        model_Main = LLM_MAIN(
            model, num_heads, num_key_value_heads, head_dim,
            num_layers, hidden_size, deepstack_features_len
        )
        del model
        gc.collect()

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
        del model_Main, hidden_states, ds_features_dummy, rotary_cos, rotary_sin, attention_mask, all_inputs
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Greedy Search
        # ══════════════════════════════════════════════════════════════════
        save_id_in = torch.zeros((batch_size, 10), dtype=torch.int32)  # 10 is a dummy value.

        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={
                'logits':         {0: 'batch'},
                'save_id_in':     {0: 'batch', 1: 'history_len'},
                'save_id_out':    {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Apply Penalty
        # ══════════════════════════════════════════════════════════════════
        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE],  dtype=torch.int64)

        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in':  {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del save_id_in, penalty_value, penalty_range

        # ══════════════════════════════════════════════════════════════════
        # Export: Argmax
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_Argmax,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits':         {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del logits
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: KV Slice
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(
            kv_tensors, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='sliced_len'
        )
        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end   = torch.tensor([5], dtype=torch.int64)  # 5 is a dummy value.

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
        del slice_start, slice_end, kv_ins, kv_in_names, kv_out_names, kv_axes, kv_tensors
        gc.collect()

    print(
        '\nExport done!\n\n'
        'Start running the QwenVL by ONNX Runtime.\n'
        'Now loading . . . it could cost minutes.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def is_valid_image_path(path):
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.raw'}
    _, ext = os.path.splitext(path)
    return ext.lower() in valid_extensions


def load_image_letterbox(path, target_h, target_w):
    """Load one image into a fixed-size canvas without stretching its aspect ratio."""
    resampling = getattr(getattr(Image, 'Resampling', Image), 'BICUBIC')
    with Image.open(path) as image:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        src_w, src_h = image.size
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
        resize_w = max(1, min(target_w, int(round(src_w * scale))))
        resize_h = max(1, min(target_h, int(round(src_h * scale))))
        if image.size != (resize_w, resize_h):
            image = image.resize((resize_w, resize_h), resampling)

        # Use a neutral background so padding stays near zero after the fused (x / 255 - 0.5) / 0.5 normalization.
        canvas = Image.new('RGB', (target_w, target_h), (127, 127, 127))
        offset_x = (target_w - resize_w) // 2
        offset_y = (target_h - resize_h) // 2
        canvas.paste(image, (offset_x, offset_y))

    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1))


def normalize_image_query(query, num_images):
    if num_images > 1 and query.strip() == DEFAULT_IMAGE_QUERY:
        return DEFAULT_MULTI_IMAGE_QUERY
    return query


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
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

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
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
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
        'gpu_mem_limit':                      16 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
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

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Vision (unified — shared for image & video) ---
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
binding_Vision     = ort_session_Vision.io_binding()
in_name_Vision     = get_in_names(ort_session_Vision)
out_name_Vision    = get_out_names(ort_session_Vision)
deepstack_features_len = len(out_name_Vision) - 1
vision_dtype = np.float16 if 'float16' in ort_session_Vision._outputs_meta[0].type else np.float32

# --- Image Preprocess (outputs: pixel_values_patches, pos_embeds, rotary_cos, rotary_sin, attention_mask) ---
onnx_model_Image_Preprocess = onnx_model_Vision.replace('LLM_Vision.onnx', 'LLM_Image_Preprocess.onnx')
ort_session_Image_Preprocess = create_session(onnx_model_Image_Preprocess, **packed_settings)
binding_Image_Preprocess     = ort_session_Image_Preprocess.io_binding()
in_name_Image_Preprocess     = get_in_names(ort_session_Image_Preprocess)[0]
out_name_Image_Preprocess    = get_out_names(ort_session_Image_Preprocess)

# --- Video Preprocess (for metadata reading) ---
ort_session_Video_Preprocess_meta = create_session(onnx_model_Video_Preprocess, **packed_settings)

# --- Concat (Multi-Image mode — static graph, no segment_offsets) ---
ort_session_Concat = create_session(onnx_model_Concat_Image, **packed_settings)
binding_Concat     = ort_session_Concat.io_binding()
in_name_Concat     = get_in_names(ort_session_Concat)
out_name_Concat    = get_out_names(ort_session_Concat)

# Read vision config from exported ONNX model metadata
vision_batch_size = ort_session_Image_Preprocess._inputs_meta[0].shape[0]
vision_embed_size_from_concat = ort_session_Concat._inputs_meta[-1].shape[1]
image_seqlen_from_meta = vision_embed_size_from_concat // vision_batch_size if isinstance(vision_embed_size_from_concat, int) and isinstance(vision_batch_size, int) else WIDTH_FACTOR * HEIGHT_FACTOR

_img_meta_shape = ort_session_Image_Preprocess._inputs_meta[0].shape
_img_h, _img_w = _img_meta_shape[-2], _img_meta_shape[-1]
if isinstance(_img_h, int) and isinstance(_img_w, int):
    input_image_size = [_img_h, _img_w]
else:
    _test_imgs = TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]
    _first_valid = next((p for p in _test_imgs if is_valid_image_path(p)), None)
    if _first_valid:
        with Image.open(_first_valid) as _img:
            input_image_size = [_img.height, _img.width]
    else:
        input_image_size = INPUT_IMAGE_SIZE

_vid_meta_shape = ort_session_Video_Preprocess_meta._inputs_meta[0].shape
_vid_h, _vid_w = _vid_meta_shape[2], _vid_meta_shape[3]
if isinstance(_vid_h, int) and isinstance(_vid_w, int):
    input_video_size = [_vid_h, _vid_w]
else:
    if TEST_VIDEO and os.path.exists(TEST_VIDEO):
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(TEST_VIDEO)
        input_video_size = [int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)), int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))]
        _cap.release()
    else:
        input_video_size = INPUT_VIDEO_SIZE
del ort_session_Video_Preprocess_meta

# --- Rotary Image (Prefill + Decode) ---
ort_session_Rotary_Image_Prefill = create_session(onnx_model_Rotary_Image_Prefill, **packed_settings)
binding_Rotary_Image_Prefill     = ort_session_Rotary_Image_Prefill.io_binding()
in_name_Rotary_Image_Prefill     = get_in_names(ort_session_Rotary_Image_Prefill)
out_name_Rotary_Image_Prefill    = get_out_names(ort_session_Rotary_Image_Prefill)

ort_session_Rotary_Image_Decode = create_session(onnx_model_Rotary_Image_Decode, **packed_settings)
binding_Rotary_Image_Decode     = ort_session_Rotary_Image_Decode.io_binding()
in_name_Rotary_Image_Decode     = get_in_names(ort_session_Rotary_Image_Decode)[0]
out_name_Rotary_Image_Decode    = get_out_names(ort_session_Rotary_Image_Decode)

# --- Rotary Text (Prefill + Decode) ---
ort_session_Rotary_Text_Prefill = create_session(onnx_model_Rotary_Text_Prefill, **packed_settings)
binding_Rotary_Text_Prefill     = ort_session_Rotary_Text_Prefill.io_binding()
in_name_Rotary_Text_Prefill     = get_in_names(ort_session_Rotary_Text_Prefill)
out_name_Rotary_Text_Prefill    = get_out_names(ort_session_Rotary_Text_Prefill)

ort_session_Rotary_Text_Decode = create_session(onnx_model_Rotary_Text_Decode, **packed_settings)
binding_Rotary_Text_Decode     = ort_session_Rotary_Text_Decode.io_binding()
in_name_Rotary_Text_Decode     = get_in_names(ort_session_Rotary_Text_Decode)[0]
out_name_Rotary_Text_Decode    = get_out_names(ort_session_Rotary_Text_Decode)
out_meta_Rotary_Text_Decode    = ort_session_Rotary_Text_Decode._outputs_meta

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main  = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main  = ort_session_Main._inputs_meta

# Derived index offsets
num_keys_values_Main        = len(out_name_Main)   - 1
num_keys_values_Main_plus_1 = num_keys_values_Main + 1

# Main model non-KV input indices
# Layout: [KV_caches..., hidden_states, deepstack_0..N-1, rotary_cos, rotary_sin, attention_mask]
idx_rotary_cos = num_keys_values_Main_plus_1 + deepstack_features_len

# Partitioned name lists
in_name_Main_kv        = in_name_Main[:num_keys_values_Main]
out_name_Main_kv       = out_name_Main[:num_keys_values_Main]
out_name_Main_logits   = out_name_Main[num_keys_values_Main]
deepstack_in_name_Main = in_name_Main[num_keys_values_Main_plus_1: idx_rotary_cos]

# Dtype introspection
kv_dtype_str      = in_meta_Main[0].type
hidden_dtype_Main = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32
vocab_size        = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE SETUP
# ══════════════════════════════════════════════════════════════════════════════

if 'uint8' in kv_dtype_str or 'int8' in kv_dtype_str or 'int32' in kv_dtype_str:
    if 'int32' in kv_dtype_str:
        kv_dtype_Main = np.int32
    elif 'uint8' in kv_dtype_str:
        kv_dtype_Main = np.uint8
    else:
        kv_dtype_Main = np.int8
    _is_rotary_rt   = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    _is_quantized_rt = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
    _kv_sym_rt      = USE_SYM and (_is_rotary_rt or _is_quantized_rt)

    # Determine number of tensor types to find num_layers_Main
    if _kv_sym_rt:
        _num_types = 4
    else:
        _num_types = 6

    num_layers_Main = num_keys_values_Main // _num_types
    scale_dtype     = np.float16 if 'float16' in in_meta_Main[num_layers_Main * 2].type else np.float32

    if _kv_sym_rt:
        # Symmetric: scale only, no bias
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_shape   = list(in_meta_Main[num_layers_Main * 3].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = None
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = None
    else:
        # Asymmetric: scale + bias
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_idx     = num_layers_Main * 4
        v_scale_shape   = list(in_meta_Main[v_scale_idx].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
else:
    kv_dtype_Main   = np.float16 if 'float16' in kv_dtype_str else np.float32
    num_layers_Main = num_keys_values_Main // 2
    k_scales        = None

past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1],                    1, in_meta_Main[0].shape[3],          0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & STOP TOKENS & PROMPT
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

STOP_TOKEN_SET = set(STOP_TOKEN)


def is_valid_video_path(path):
    """Check if the path points to a valid video file."""
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(path)
    return ext.lower() in valid_extensions


def build_prompt_and_mm_types(query, mode, num_frames=0, frame_seqlen=0, fps=2.0, frame_indices=None, num_images=1, image_seqlen=0):
    """Build the prompt string, tokenize it, and compute mm_token_type_ids.

    Args:
        query: Text query string
        mode: 'text' | 'image' | 'video'
        num_frames: Number of temporal frames (grid_t) for video
        frame_seqlen: Number of vision tokens per frame for video
        fps: Frames per second for timestamp calculation
        frame_indices: Actual frame indices from video (for timestamp computation)
        num_images: Number of images for multi-image mode
        image_seqlen: Number of vision tokens per image (after merge)

    Returns:
        tokens: int32 numpy array [1, seq_len]
        mm_token_type_ids: list of int (0=text, 1=image, 2=video)
    """
    if mode == 'text':
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
        mm_token_type_ids = [0] * tokens.shape[-1]
        return tokens, mm_token_type_ids

    elif mode == 'image':
        # Multi-image prompt: <|im_start|>user\n[<|vision_start|><|vision_end|>]*N query<|im_end|>\n<|im_start|>assistant\n
        query = normalize_image_query(query, num_images)
        vision_markers = "<|vision_start|><|vision_end|>" * num_images
        prompt = f"<|im_start|>user\n{vision_markers}{query}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
        # mm_token_type_ids: all text (image tokens are inserted at runtime via concat, not in token stream)
        mm_token_type_ids = [0] * tokens.shape[-1]
        return tokens, mm_token_type_ids

    elif mode == 'video':
        # Build video prompt with timestamp-separated frames
        # Structure: <|im_start|>user\n<ts><|vision_start|><|video_pad|>*N<|vision_end|>...<query><|im_end|>\n<|im_start|>assistant\n

        # Compute timestamps (average of temporal patch boundaries)
        if frame_indices is not None:
            timestamps = []
            indices = list(frame_indices)
            if len(indices) % TEMPORAL_PATCH_SIZE != 0:
                indices.extend([indices[-1]] * (TEMPORAL_PATCH_SIZE - len(indices) % TEMPORAL_PATCH_SIZE))
            for i in range(0, len(indices), TEMPORAL_PATCH_SIZE):
                t_start = indices[i] / fps
                t_end = indices[i + TEMPORAL_PATCH_SIZE - 1] / fps
                timestamps.append((t_start + t_end) / 2.0)
        else:
            # Default: evenly spaced timestamps
            timestamps = [(i * TEMPORAL_PATCH_SIZE + TEMPORAL_PATCH_SIZE / 2) / fps for i in range(num_frames)]

        # Build the video placeholder text
        video_token = "<|video_pad|>"
        vision_start = "<|vision_start|>"
        vision_end = "<|vision_end|>"

        video_placeholder = ""
        for f in range(num_frames):
            video_placeholder += f"<{timestamps[f]:.1f} seconds>"
            video_placeholder += vision_start + video_token * frame_seqlen + vision_end

        prompt = f"<|im_start|>user\n{video_placeholder}{query}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)

        # Build mm_token_type_ids by scanning token IDs
        token_ids = tokens[0].tolist()
        video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        mm_token_type_ids = []
        for tid in token_ids:
            if tid == video_pad_id:
                mm_token_type_ids.append(2)  # video
            else:
                mm_token_type_ids.append(0)  # text (including vision_start/end markers)

        return tokens, mm_token_type_ids


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINE TEST MODES
# ══════════════════════════════════════════════════════════════════════════════
test_modes = []
# Support TEST_IMAGE as a list of paths for multi-image
test_image_paths = TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]
valid_image_paths = [p for p in test_image_paths if is_valid_image_path(p)]
if valid_image_paths:
    test_modes.append(('image', TEST_QUERY[0]))
if is_valid_video_path(TEST_VIDEO):
    test_modes.append(('video', TEST_QUERY[1]))
if not test_modes:
    test_modes.append(('text', "Hello! How are you?"))

num_test_images = len(valid_image_paths)
image_seqlen_runtime = image_seqlen_from_meta


for INPUT_MODE, current_query in test_modes:
    print(f"\n{'═' * 56}")
    print(f"  Running test: mode={INPUT_MODE}, query=\"{current_query}\"")
    print(f"{'═' * 56}")

    # Build prompt based on mode
    if INPUT_MODE == 'video':
        video_grid_t_runtime = VIDEO_NUM_FRAMES // TEMPORAL_PATCH_SIZE
        video_frame_seqlen_runtime = (VIDEO_HEIGHT_FACTOR * 2 * VIDEO_WIDTH_FACTOR * 2) // (2 * 2)
        tokens, mm_token_type_ids = build_prompt_and_mm_types(
            current_query, 'video',
            num_frames=video_grid_t_runtime,
            frame_seqlen=video_frame_seqlen_runtime,
            fps=VIDEO_FPS
        )
    elif INPUT_MODE == 'image':
        tokens, mm_token_type_ids = build_prompt_and_mm_types(
            current_query, 'image',
            num_images=num_test_images,
            image_seqlen=image_seqlen_runtime
        )
    else:
        tokens, mm_token_type_ids = build_prompt_and_mm_types(current_query, 'text')

    num_prefill = tokens.shape[-1]



    # ══════════════════════════════════════════════════════════════════════════════
    # SHARED ORTVALUE BUFFERS
    # ══════════════════════════════════════════════════════════════════════════════
    vision_embed_size = num_test_images * image_seqlen_from_meta

    # --- Recreate IO bindings for a fresh run ---
    binding_Embed  = ort_session_Embed.io_binding()
    binding_Main   = ort_session_Main.io_binding()
    binding_Vision = ort_session_Vision.io_binding()
    binding_Concat = ort_session_Concat.io_binding()
    binding_Image_Preprocess = ort_session_Image_Preprocess.io_binding()
    binding_Rotary_Image_Prefill = ort_session_Rotary_Image_Prefill.io_binding()
    binding_Rotary_Image_Decode  = ort_session_Rotary_Image_Decode.io_binding()
    binding_Rotary_Text_Prefill  = ort_session_Rotary_Text_Prefill.io_binding()
    binding_Rotary_Text_Decode   = ort_session_Rotary_Text_Decode.io_binding()

    # --- Input OrtValues ---
    input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens,   device_type, DEVICE_ID)
    ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)

    # --- Decode-phase placeholder buffers (reused every step) ---
    attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                             hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_cos_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[0].shape,                              hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_sin_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[1].shape,                              hidden_dtype_Main, device_type, DEVICE_ID)
    hidden_states_buf  = create_ort_with_shape((1, 1, in_meta_Main[num_keys_values_Main].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
    save_id_buf        = create_ort_with_shape((1, 0),                                              np.int32,          device_type, DEVICE_ID)

    # --- Init deepstack features (zeros for decode and text-only mode) ---
    init_deepstack_features = [create_ort_with_shape((1, 1, ort_session_Vision._outputs_meta[0].shape[2]), vision_dtype, device_type, DEVICE_ID)] * deepstack_features_len  # Same memory address buff

    # With unified vision model (dynamic seq_len), outputs are always dynamically bound
    fixed_vision_shape = False

    # --- Logits & token-index buffers ---
    prefill_logits_buf = create_ort_with_shape((1, vocab_size),         hidden_dtype_Main, device_type, DEVICE_ID)
    decode_logits_buf  = create_ort_with_shape((1, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)


    # ══════════════════════════════════════════════════════════════════════════════
    # DECODE HEAD SESSIONS
    # ══════════════════════════════════════════════════════════════════════════════
    # --- Greedy + generated-ID accumulation ---
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    # --- Plain greedy argmax ---
    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


    # ══════════════════════════════════════════════════════════════════════════════
    # PENALTY SESSION (optional)
    # ══════════════════════════════════════════════════════════════════════════════
    if USE_PENALTY:
        ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
        binding_Penalty     = ort_session_Penalty.io_binding()
        in_name_Penalty     = get_in_names(ort_session_Penalty)
        out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

        penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
        penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
        penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

        bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])


    # ══════════════════════════════════════════════════════════════════════════════
    # INPUT VALIDATION & VISION/VIDEO PROCESSING
    # ══════════════════════════════════════════════════════════════════════════════
    use_vision = False
    use_video = False

    if INPUT_MODE == 'video':
        use_video = True
        print('\nChat with video.')

        # Decode video frames using av (libavcodec) with built-in libswscale resize
        import av
        container = av.open(TEST_VIDEO)
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        video_fps_actual = float(stream.average_rate) if stream.average_rate else 24.0

        # If container doesn't report frame count, count them manually
        if total_frames <= 0:
            container.seek(0)
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)
        if total_frames <= 0:
            raise ValueError(f"Video has no decodable frames: {TEST_VIDEO}")

        # Sample frames uniformly, then pad to exactly VIDEO_NUM_FRAMES
        num_sample = min(max(int(total_frames / video_fps_actual * VIDEO_FPS), VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, total_frames)
        if num_sample > VIDEO_NUM_FRAMES:
            num_sample = VIDEO_NUM_FRAMES
        num_sample = max(num_sample, 1)
        frame_indices = np.linspace(0, total_frames - 1, num_sample).round().astype(int)

        # Pad to exactly VIDEO_NUM_FRAMES (must match export-time frame count for rotary table alignment)
        if num_sample < VIDEO_NUM_FRAMES:
            pad_count = VIDEO_NUM_FRAMES - num_sample
            frame_indices = np.concatenate([frame_indices, np.full(pad_count, frame_indices[-1], dtype=frame_indices.dtype)])

        # Determine output spatial dimensions (apply resize target if static shape)
        if not DYNAMIC_VIDEO_SHAPE:
            out_h, out_w = input_video_size
        else:
            out_h, out_w = stream.codec_context.height, stream.codec_context.width

        # Pre-allocate contiguous output buffer [VIDEO_NUM_FRAMES, 3, H, W]
        video_frames_np = np.empty((VIDEO_NUM_FRAMES, 3, out_h, out_w), dtype=np.uint8)

        # Build sorted unique-index → slot mapping for O(1) decode-time lookup
        # frame_indices is already sorted (from linspace + pad), deduplicate for decode
        unique_indices, inverse_map = np.unique(frame_indices, return_inverse=True)
        last_needed = int(unique_indices[-1])

        # Check if resize is needed during decode
        need_resize = (stream.codec_context.height != out_h or stream.codec_context.width != out_w)

        # Decode only up to last_needed frame; pointer-based extraction into pre-allocated buffer
        container.seek(0)
        unique_pos = 0  # pointer into unique_indices
        frame_idx = 0
        temp_decoded = [None] * len(unique_indices)  # temporary refs for unique frames

        for frame in container.decode(video=0):
            if frame_idx == unique_indices[unique_pos]:
                # Fused resize + format conversion via libswscale in one reformat() call
                if need_resize:
                    frame = frame.reformat(width=out_w, height=out_h, format='rgb24')
                # Extract as [H, W, 3] uint8 directly into numpy — single allocation per unique frame
                rgb = frame.to_ndarray(format='rgb24')
                temp_decoded[unique_pos] = rgb
                unique_pos += 1
                if unique_pos >= len(unique_indices):
                    break
            frame_idx += 1
            if frame_idx > last_needed:
                break
        container.close()

        # Fill any missing unique frames (short video) by repeating last decoded
        last_valid = None
        for i in range(len(unique_indices)):
            if temp_decoded[i] is not None:
                last_valid = temp_decoded[i]
            else:
                temp_decoded[i] = last_valid

        # Scatter unique frames into the pre-allocated [N, 3, H, W] buffer via inverse_map
        # Use np.transpose once per unique frame, then copy into all slots that reference it
        for uid in range(len(unique_indices)):
            chw = np.ascontiguousarray(temp_decoded[uid].transpose(2, 0, 1))  # [3, H, W]
            # Find all output slots mapping to this unique frame
            slots = np.where(inverse_map == uid)[0]
            for s in slots:
                video_frames_np[s] = chw
        del temp_decoded, unique_indices, inverse_map

        video_frames_ort = onnxruntime.OrtValue.ortvalue_from_numpy(video_frames_np, device_type, DEVICE_ID)
        del video_frames_np

        # Re-build prompt with fixed timestamps (must match export-time rotary table structure)
        video_grid_t_runtime = len(frame_indices) // TEMPORAL_PATCH_SIZE
        video_frame_seqlen_runtime = (VIDEO_HEIGHT_FACTOR * 2 * VIDEO_WIDTH_FACTOR * 2) // (2 * 2)
        tokens, mm_token_type_ids = build_prompt_and_mm_types(
            current_query, 'video',
            num_frames=video_grid_t_runtime,
            frame_seqlen=video_frame_seqlen_runtime,
            fps=VIDEO_FPS,
            frame_indices=None
        )
        num_prefill = tokens.shape[-1]
        input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)

        # Release image-specific sessions to free GPU memory (not needed in video mode)
        del ort_session_Image_Preprocess, binding_Image_Preprocess
        del ort_session_Concat, binding_Concat
        del ort_session_Rotary_Image_Prefill, binding_Rotary_Image_Prefill
        del ort_session_Rotary_Image_Decode, binding_Rotary_Image_Decode
        gc.collect()

        # Load video-specific sessions (preprocess, concat, rotary — vision is SHARED)
        ort_session_Video_Preprocess = create_session(onnx_model_Video_Preprocess, **packed_settings)
        binding_Video_Preprocess = ort_session_Video_Preprocess.io_binding()
        in_name_Video_Preprocess = get_in_names(ort_session_Video_Preprocess)[0]
        out_name_Video_Preprocess = get_out_names(ort_session_Video_Preprocess)

        ort_session_Concat_Video = create_session(onnx_model_Concat_Video, **packed_settings)
        binding_Concat_Video = ort_session_Concat_Video.io_binding()
        in_name_Concat_Video = get_in_names(ort_session_Concat_Video)
        out_name_Concat_Video = get_out_names(ort_session_Concat_Video)

        ort_session_Rotary_Video_Prefill = create_session(onnx_model_Rotary_Video_Prefill, **packed_settings)
        binding_Rotary_Video_Prefill = ort_session_Rotary_Video_Prefill.io_binding()
        in_name_Rotary_Video_Prefill = get_in_names(ort_session_Rotary_Video_Prefill)
        out_name_Rotary_Video_Prefill = get_out_names(ort_session_Rotary_Video_Prefill)

        ort_session_Rotary_Video_Decode = create_session(onnx_model_Rotary_Video_Decode, **packed_settings)
        binding_Rotary_Video_Decode = ort_session_Rotary_Video_Decode.io_binding()
        in_name_Rotary_Video_Decode = get_in_names(ort_session_Rotary_Video_Decode)[0]
        out_name_Rotary_Video_Decode = get_out_names(ort_session_Rotary_Video_Decode)

        # Compute vision_embed_size for video
        vision_embed_size = video_grid_t_runtime * video_frame_seqlen_runtime

    elif INPUT_MODE == 'image':
        # Load images into a fixed canvas without stretching aspect ratio.
        target_h, target_w = input_image_size
        if INPUT_IMAGE_DIM != 4:
            pixel_values = np.empty((num_test_images, 1, 3, target_h, target_w), dtype=np.uint8)
        else:
            pixel_values = np.empty((num_test_images, 3, target_h, target_w), dtype=np.uint8)

        for i, img_path in enumerate(valid_image_paths):
            chw = load_image_letterbox(img_path, target_h, target_w)
            if INPUT_IMAGE_DIM != 4:
                pixel_values[i, 0] = chw
            else:
                pixel_values[i] = chw
        use_vision = True
        print(f'\nChat with {num_test_images} image(s).')
    else:
        print('\nChat without image.')


    # ══════════════════════════════════════════════════════════════════════════════
    # PREFILL PHASE
    # ══════════════════════════════════════════════════════════════════════════════
    is_prefill_step = True
    prefill_start_time = time.time()

    # --- Step 1: Embed the input tokens ---
    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, binding_Embed)
    hidden_states = binding_Embed.get_outputs()[0]

    # Pre-bind Embed input for decode phase (will read from max_idx_buf)
    binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

    # --- Step 2: Vision/Video processing & rotary (if applicable) ---
    generate_limit = MAX_SEQ_LEN - num_prefill

    if use_video:
        # Video prompts already include one placeholder token per vision token.
        # Concat replaces those spans in-place, so the prefill length stays unchanged.
        ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)

        print('\nStart to Process the Video...')
        vision_start_time = time.time()

        # Step 2a: Video preprocessing (ONNX) — resize, normalize, patch + aux tensors
        binding_Video_Preprocess.bind_ortvalue_input(in_name_Video_Preprocess, video_frames_ort)
        bind_ort_out(binding_Video_Preprocess, out_name_Video_Preprocess, _ort_device_type)
        run(ort_session_Video_Preprocess, binding_Video_Preprocess)
        outputs_Video_Preprocess = binding_Video_Preprocess.get_outputs()
        video_patches = outputs_Video_Preprocess[0]

        # Step 2b: Vision encoder (shared ONNX, aux inputs from preprocess)
        binding_Vision.bind_ortvalue_input(in_name_Vision[0], video_patches)
        binding_Vision.bind_ortvalue_input(in_name_Vision[1], outputs_Video_Preprocess[1])  # pos_embeds
        binding_Vision.bind_ortvalue_input(in_name_Vision[2], outputs_Video_Preprocess[2])  # rotary_cos
        binding_Vision.bind_ortvalue_input(in_name_Vision[3], outputs_Video_Preprocess[3])  # rotary_sin
        binding_Vision.bind_ortvalue_input(in_name_Vision[4], outputs_Video_Preprocess[4])  # attention_mask
        bind_ort_out(binding_Vision, out_name_Vision, _ort_device_type)
        run(ort_session_Vision, binding_Vision)
        outputs_Vision = binding_Vision.get_outputs()
        print(f'\nVideo Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds')

        # Step 2c: Concat (Video mode) — merge text embeddings + vision features at segment offsets
        for i in range(deepstack_features_len):
            binding_Concat_Video.bind_ortvalue_input(in_name_Concat_Video[i], outputs_Vision[i])
        binding_Concat_Video.bind_ortvalue_input(in_name_Concat_Video[deepstack_features_len], hidden_states)
        binding_Concat_Video.bind_ortvalue_input(in_name_Concat_Video[deepstack_features_len + 1], outputs_Vision[deepstack_features_len])
        bind_ort_out(binding_Concat_Video, out_name_Concat_Video, _ort_device_type)
        run(ort_session_Concat_Video, binding_Concat_Video)
        outputs_Concat = binding_Concat_Video.get_outputs()

        # Step 2d: Rotary Video Prefill
        bind_ort_in_buf(binding_Rotary_Video_Prefill, in_name_Rotary_Video_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Video_Prefill, out_name_Rotary_Video_Prefill, _ort_device_type)
        run(ort_session_Rotary_Video_Prefill, binding_Rotary_Video_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Video_Prefill.get_outputs()

        # Bind Main: concat hidden_states + deepstack features from video
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], outputs_Concat[deepstack_features_len])
        bind_ort_in_buf(binding_Main, deepstack_in_name_Main, outputs_Concat[:deepstack_features_len])

    elif use_vision:
        num_prefill += vision_embed_size
        ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
        generate_limit -= vision_embed_size

        print('\nStart to Process the Image...')
        vision_start_time = time.time()

        # Step 2a: Image preprocessing (ONNX) — resize, normalize, duplicate temporal, patch + aux tensors
        binding_Image_Preprocess.bind_ortvalue_input(in_name_Image_Preprocess, onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID))
        bind_ort_out(binding_Image_Preprocess, out_name_Image_Preprocess, _ort_device_type)
        run(ort_session_Image_Preprocess, binding_Image_Preprocess)
        outputs_Image_Preprocess = binding_Image_Preprocess.get_outputs()
        image_patches = outputs_Image_Preprocess[0]

        # Step 2b: Vision encoder (shared ONNX, aux inputs from preprocess)
        binding_Vision.bind_ortvalue_input(in_name_Vision[0], image_patches)
        binding_Vision.bind_ortvalue_input(in_name_Vision[1], outputs_Image_Preprocess[1])  # pos_embeds
        binding_Vision.bind_ortvalue_input(in_name_Vision[2], outputs_Image_Preprocess[2])  # rotary_cos
        binding_Vision.bind_ortvalue_input(in_name_Vision[3], outputs_Image_Preprocess[3])  # rotary_sin
        binding_Vision.bind_ortvalue_input(in_name_Vision[4], outputs_Image_Preprocess[4])  # attention_mask
        bind_ort_out(binding_Vision, out_name_Vision, _ort_device_type)
        run(ort_session_Vision, binding_Vision)
        outputs_Vision = binding_Vision.get_outputs()
        print(f'\nImage Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds')

        # Run Concat: merge text embeddings + vision features (static graph, no segment_offsets)
        bind_ort_in_buf(binding_Concat, in_name_Concat[:deepstack_features_len], outputs_Vision[:deepstack_features_len])
        binding_Concat.bind_ortvalue_input(in_name_Concat[deepstack_features_len], hidden_states)
        binding_Concat.bind_ortvalue_input(in_name_Concat[deepstack_features_len + 1], outputs_Vision[deepstack_features_len])
        bind_ort_out(binding_Concat, out_name_Concat, _ort_device_type)
        run(ort_session_Concat, binding_Concat)
        outputs_Concat = binding_Concat.get_outputs()

        # Rotary Image Prefill
        bind_ort_in_buf(binding_Rotary_Image_Prefill, in_name_Rotary_Image_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Image_Prefill, out_name_Rotary_Image_Prefill, _ort_device_type)
        run(ort_session_Rotary_Image_Prefill, binding_Rotary_Image_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Image_Prefill.get_outputs()

        # Bind Main: concat hidden_states + deepstack features from vision
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], outputs_Concat[deepstack_features_len])
        bind_ort_in_buf(binding_Main, deepstack_in_name_Main, outputs_Concat[:deepstack_features_len])

    else:
        # Rotary Text Prefill
        bind_ort_in_buf(binding_Rotary_Text_Prefill, in_name_Rotary_Text_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Text_Prefill, out_name_Rotary_Text_Prefill, _ort_device_type)
        run(ort_session_Rotary_Text_Prefill, binding_Rotary_Text_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Text_Prefill.get_outputs()

        # Bind Main: text-only hidden_states + zero deepstack features
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states)
        bind_ort_in_buf(binding_Main, deepstack_in_name_Main, init_deepstack_features)

    # --- Step 3: Pre-bind decode rotary outputs (reused every decode step) ---
    if use_video:
        binding_Rotary_Video_Decode.bind_ortvalue_input(in_name_Rotary_Video_Decode, kv_seq_len)
        bind_ort_out_buf(binding_Rotary_Video_Decode, out_name_Rotary_Video_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])
        ort_session_Rotary_Decode = ort_session_Rotary_Video_Decode
        binding_Rotary_Decode     = binding_Rotary_Video_Decode
    elif use_vision:
        binding_Rotary_Image_Decode.bind_ortvalue_input(in_name_Rotary_Image_Decode, kv_seq_len)
        bind_ort_out_buf(binding_Rotary_Image_Decode, out_name_Rotary_Image_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])
        ort_session_Rotary_Decode = ort_session_Rotary_Image_Decode
        binding_Rotary_Decode     = binding_Rotary_Image_Decode
    else:
        binding_Rotary_Text_Decode.bind_ortvalue_input(in_name_Rotary_Text_Decode, kv_seq_len)
        bind_ort_out_buf(binding_Rotary_Text_Decode, out_name_Rotary_Text_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])
        ort_session_Rotary_Decode = ort_session_Rotary_Text_Decode
        binding_Rotary_Decode     = binding_Rotary_Text_Decode

    # --- Step 4: Bind Main model inputs — rotary & attention mask (prefill) ---
    bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos, rotary_sin, attention_mask])

    # --- Step 5: Bind Main model inputs — empty KV cache (keys, values, optional scales/biases) ---
    i = 0
    for _ in range(num_layers_Main):
        binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
        i += 1
    for _ in range(num_layers_Main):
        binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
        i += 1
    if k_scales is not None:
        if k_biases is not None:
            for ortval in (k_scales, k_biases, v_scales, v_biases):
                for _ in range(num_layers_Main):
                    binding_Main.bind_ortvalue_input(in_name_Main[i], ortval)
                    i += 1
        else:
            for ortval in (k_scales, v_scales):
                for _ in range(num_layers_Main):
                    binding_Main.bind_ortvalue_input(in_name_Main[i], ortval)
                    i += 1

    # --- Step 6: Bind Main model outputs ---
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
    binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

    # --- Step 7: Bind penalty inputs/outputs to prefill logits buffer ---
    if USE_PENALTY:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
        binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

    # --- Step 8: Bind decode head inputs/outputs to prefill logits buffer ---
    if USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
        binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
    else:
        binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
        binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)


    # ══════════════════════════════════════════════════════════════════════════════
    # DECODE LOOP
    # ══════════════════════════════════════════════════════════════════════════════
    print(f'\nTest Question: {current_query}\nLLM Answering:')

    num_decode = 0

    while num_decode < generate_limit:

        # ── 1. Run Main Model ────────────────────────────────────────────────
        run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        # ── 2. Apply Repetition Penalty (if enabled and enough tokens) ───────
        if USE_PENALTY and num_decode >= PENALTY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            run(ort_session_Penalty, binding_Penalty)

        # ── 3. Greedy Token Selection ────────────────────────────────────────
        if USE_PENALTY:
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            save_id = binding_Greedy.get_outputs()[1]
        else:
            run(ort_session_Argmax, binding_Argmax)

        # Stop-token check
        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        # Track generated token IDs
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx

        # Feed greedy KV outputs back into Main
        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)

        # Streaming print
        print(tokenizer.decode(max_logits_idx), end="", flush=True)

        # ── 4. Re-bind Main KV outputs (ORT allocates fresh each step) ───────
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

        # ── 5. Transition: prefill → decode (executes once) ──────────────────
        if is_prefill_step:

            # Switch Main to decode-sized non-KV inputs
            binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states_buf)
            bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
            binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)

            # Switch deepstack features to zeros for decode
            bind_ort_in_buf(binding_Main, deepstack_in_name_Main, init_deepstack_features)

            # Switch Embed to write into decode hidden_states buffer
            binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

            # Switch Penalty to decode logits buffer
            if USE_PENALTY:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                binding_Penalty.bind_ortvalue_output(out_name_Penalty,  decode_logits_buf)

            # Switch decode head to decode logits buffer
            if USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
            else:
                binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

            is_prefill_step = False

            # Record prefill time and start decode timer
            decode_start_time = time.time()
            prefill_elapsed = decode_start_time - prefill_start_time

        # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
        run(ort_session_Embed, binding_Embed)
        run(ort_session_Rotary_Decode, binding_Rotary_Decode)
        num_decode += 1


    # ══════════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════════
    decode_end_time = time.time()

    # Handle edge case where generation stopped at prefill (0 decode tokens after first)
    if num_decode < 2:
        prefill_elapsed = 0.0
        decode_elapsed = 0.0
    else:
        decode_elapsed = decode_end_time - decode_start_time

    total_elapsed = decode_end_time - prefill_start_time

    # Prefill speed: tokens processed per second
    prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0

    # Decode speed: tokens generated per second (excluding the first token from prefill)
    decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0

    # Overall speed
    overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

    if USE_PENALTY:
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
    
