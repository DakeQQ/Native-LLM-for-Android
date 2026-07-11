import argparse
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from PIL import Image
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Qwen3.5-VL split-ONNX multimodal inference demo.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen_Optimized",
        help="Folder containing the split ONNX graphs exported by Export_Qwen.py.",
    )
    parser.add_argument(
        "--tokenizer-folder",
        type=Path,
        default=Path(r"/home/DakeQQ/Downloads/Qwen3.5-0.8B"),
        help="HF checkpoint/tokenizer folder used to tokenize the demo prompt.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "image", "video", "text"),
        default="auto",
        help="Limit the demo to one input mode. Default auto runs available image/video tests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional cap for generated tokens per test.",
    )
    return parser.parse_args()


args = parse_args()

download_path = str(args.tokenizer_folder.expanduser().resolve())
onnx_folder   = args.model_folder.expanduser().resolve()
RUN_MODE      = args.mode
MAX_NEW_TOKENS = args.max_new_tokens

onnx_model_Metadata             = str(onnx_folder / "LLM_Metadata.onnx")
onnx_model_Embed                = str(onnx_folder / "LLM_Embed.onnx")
onnx_model_Vision               = str(onnx_folder / "LLM_Vision.onnx")
onnx_model_Image_Preprocess     = str(onnx_folder / "LLM_Image_Preprocess.onnx")
onnx_model_Video_Preprocess     = str(onnx_folder / "LLM_Video_Preprocess.onnx")
onnx_model_Concat_Image         = str(onnx_folder / "LLM_Concat_Image.onnx")
onnx_model_Concat_Video         = str(onnx_folder / "LLM_Concat_Video.onnx")
onnx_model_Rotary_Image_Prefill = str(onnx_folder / "LLM_Rotary_Image_Prefill.onnx")
onnx_model_Rotary_Image_Decode  = str(onnx_folder / "LLM_Rotary_Image_Decode.onnx")
onnx_model_Rotary_Video_Prefill = str(onnx_folder / "LLM_Rotary_Video_Prefill.onnx")
onnx_model_Rotary_Video_Decode  = str(onnx_folder / "LLM_Rotary_Video_Decode.onnx")
onnx_model_Rotary_Text_Prefill  = str(onnx_folder / "LLM_RotaryPrefill.onnx")
onnx_model_Rotary_Text_Decode   = str(onnx_folder / "LLM_RotaryDecode.onnx")
onnx_model_Main                 = str(onnx_folder / "LLM_Main.onnx")
onnx_model_Greedy               = str(onnx_folder / "LLM_Greedy.onnx")
onnx_model_First_Beam           = str(onnx_folder / "LLM_FirstBeam.onnx")
onnx_model_Second_Beam          = str(onnx_folder / "LLM_SecondBeam.onnx")
onnx_model_Penalty              = str(onnx_folder / "LLM_Penalty.onnx")
onnx_model_Argmax               = str(onnx_folder / "LLM_Argmax.onnx")


# ══════════════════════════════════════════════════════════════════════════════
# Runtime / test configuration
# ══════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = Path(__file__).resolve().parent

# Test Input
TEST_IMAGE      = [str(_SCRIPT_DIR / "psyduck.png")]                # List of image paths for multi-image support. Use [] for text-only.
TEST_VIDEO      = str(_SCRIPT_DIR / "test_video_8s.mp4")            # Path to test video file. Leave empty to disable video mode.
TEST_QUERY      = ["Describe this image.", "Describe this video."]  # [image_query, video_query]
ENABLE_THINKING = False                                             # Enable thinking mode in generation.

# Decoding strategy
USE_BEAM_SEARCH = False                                         # Use beam search or greedy search
REPEAT_PENALTY  = 1.0                                           # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE   = 20                                            # Recent-token window to apply penalty.
TOP_K           = 3                                             # Top-K for beam search.
BEAM_SIZE       = 3                                             # Beam size; must be <= export MAX_BEAM_SIZE.

# Runtime config
ORT_LOG                  = False                                # Enable ONNX Runtime logging for debugging.
ORT_FP16                 = False                                # Auto-set from LLM_Metadata.onnx (activations_fp16).
ORT_Accelerate_Providers = []                                   # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                                    # 0 = auto
DEVICE_ID                = 0                                    # Device ID for GPU


if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = REPEAT_PENALTY != 1.0


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


# ══════════════════════════════════════════════════════════════════════════════
# Image / video loading helpers
# ══════════════════════════════════════════════════════════════════════════════
def is_valid_image_path(path):
    """Return True when `path` exists and has an image-like extension."""
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".raw"}
    _, ext = os.path.splitext(path)
    return ext.lower() in valid_extensions


def is_valid_video_path(path):
    """Return True when path exists and has a video-like extension."""
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
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


def load_metadata_carrier(model_path):
    """Load Native-LLM metadata_props stamped by Export_Qwen.py into LLM_Metadata.onnx."""
    meta_opts = onnxruntime.SessionOptions()
    meta_opts.log_severity_level = 0 if ORT_LOG else 4
    meta_opts.log_verbosity_level = 4
    meta_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        meta_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=meta_opts,
            providers=['CPUExecutionProvider'],
        )
    except Exception as exc:
        raise RuntimeError(
            "LLM_Metadata.onnx is required to preload Native-LLM metadata. "
            "Re-export with Export_Qwen.py or copy the metadata carrier into the model folder."
        ) from exc

    metadata = meta_session.get_modelmeta().custom_metadata_map
    if not metadata.get('native_llm_metadata_version'):
        raise RuntimeError(
            "LLM_Metadata.onnx carries no Native-LLM metadata_props. Re-export the model with Export_Qwen.py."
        )
    return metadata


# Read metadata before creating real sessions so ORT_FP16/session settings follow the exported graphs.
model_meta = load_metadata_carrier(onnx_model_Metadata)
ORT_FP16 = model_meta.get('activations_fp16') == '1'

IMAGE_TOKEN_ID             = int(model_meta['image_token_id'])
VIDEO_TOKEN_ID             = int(model_meta['video_token_id'])
VISION_TEMPORAL_PATCH_SIZE = int(model_meta['temporal_patch_size'])
MAX_SEQ_LEN                = int(model_meta['max_seq_len'])

# Vision sampling / input geometry stamped into LLM_Metadata.onnx by Export_Qwen.py.
VIDEO_FPS        = float(model_meta['video_fps'])
VIDEO_MIN_FRAMES = int(model_meta['video_min_frames'])
VIDEO_MAX_FRAMES = int(model_meta['video_max_frames'])
INPUT_IMAGE_SIZE = [int(s) for s in model_meta['input_image_size'].split(',')]
INPUT_VIDEO_SIZE = [int(s) for s in model_meta['input_video_size'].split(',')]

STOP_TOKEN = [int(t) for t in model_meta.get('stop_token_ids', '').split(',') if t]
if not STOP_TOKEN:
    STOP_TOKEN = [int(t) for t in model_meta.get('eos_token_ids', '').split(',') if t]


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
        'do_copy_in_default_stream':          '0',
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


print(
    '\nStart running the Qwen3.5-VL LLM by ONNXRuntime.\n'
    'Now loading . . . it could cost minutes.'
)


# ══════════════════════════════════════════════════════════════════════════════
# Load ONNX Sessions
# ══════════════════════════════════════════════════════════════════════════════
# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Image Preprocess ---
ort_session_Image_Preprocess = create_session(onnx_model_Image_Preprocess, **packed_settings)
binding_Image_Preprocess     = ort_session_Image_Preprocess.io_binding()
in_name_Image_Preprocess     = get_in_names(ort_session_Image_Preprocess)[0]
out_name_Image_Preprocess    = get_out_names(ort_session_Image_Preprocess)

# --- Video Preprocess ---
ort_session_Video_Preprocess = create_session(onnx_model_Video_Preprocess, **packed_settings)
binding_Video_Preprocess     = ort_session_Video_Preprocess.io_binding()
in_name_Video_Preprocess     = get_in_names(ort_session_Video_Preprocess)[0]
out_name_Video_Preprocess    = get_out_names(ort_session_Video_Preprocess)

# --- Vision (shared) ---
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
binding_Vision     = ort_session_Vision.io_binding()
in_name_Vision     = get_in_names(ort_session_Vision)
out_name_Vision    = get_out_names(ort_session_Vision)[0]

# --- Concat Image ---
ort_session_Concat_Image = create_session(onnx_model_Concat_Image, **packed_settings)
binding_Concat_Image     = ort_session_Concat_Image.io_binding()
in_name_Concat_Image     = get_in_names(ort_session_Concat_Image)
out_name_Concat_Image    = get_out_names(ort_session_Concat_Image)[0]

# Read vision config from exported ONNX model metadata
vision_batch_size = ort_session_Image_Preprocess._inputs_meta[0].shape[0]
vision_embed_size = ort_session_Concat_Image._inputs_meta[1].shape[1]

# Vision input geometry. LLM_Metadata.onnx carries the export-time defaults, which are exact
# whenever the exported graph keeps its spatial axes STATIC. When the graph was exported with
# DYNAMIC spatial axes (symbolic H/W), we no longer stay locked to those defaults: the
# preprocess graph interpolates to its own target size, so we can feed the runtime input at its
# native resolution. Resolve the real size from the actual input, keeping the metadata value
# only as a last-resort fallback when no runtime input is available.
_img_shape = ort_session_Image_Preprocess._inputs_meta[0].shape  # [N, 1, 3, H, W]
if isinstance(_img_shape[3], int) and isinstance(_img_shape[4], int):
    input_image_size = INPUT_IMAGE_SIZE
else:
    _test_imgs = TEST_IMAGE if isinstance(TEST_IMAGE, list) else ([TEST_IMAGE] if TEST_IMAGE else [])
    _first_valid = next((p for p in _test_imgs if is_valid_image_path(p)), None)
    if _first_valid:
        with Image.open(_first_valid) as _img:
            input_image_size = [_img.height, _img.width]
    else:
        input_image_size = INPUT_IMAGE_SIZE

_vid_shape = ort_session_Video_Preprocess._inputs_meta[0].shape  # [num_frames, 3, H, W]
if isinstance(_vid_shape[2], int) and isinstance(_vid_shape[3], int):
    input_video_size = INPUT_VIDEO_SIZE
else:
    if TEST_VIDEO and os.path.exists(TEST_VIDEO):
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(TEST_VIDEO)
        input_video_size = [int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)), int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))]
        _cap.release()
    else:
        input_video_size = INPUT_VIDEO_SIZE

# --- Concat Video ---
ort_session_Concat_Video = create_session(onnx_model_Concat_Video, **packed_settings)
binding_Concat_Video     = ort_session_Concat_Video.io_binding()
in_name_Concat_Video     = get_in_names(ort_session_Concat_Video)
out_name_Concat_Video    = get_out_names(ort_session_Concat_Video)[0]

# Derive vision-token geometry from the exported (static) graph shapes so the runtime
# prompt matches the offsets baked into LLM_Concat_Image / LLM_Concat_Video.
_video_num_frames         = ort_session_Video_Preprocess._inputs_meta[0].shape[0]
_video_total_vision_tokens = ort_session_Concat_Video._inputs_meta[1].shape[1]
if isinstance(_video_num_frames, int) and isinstance(_video_total_vision_tokens, int):
    VIDEO_NUM_FRAMES   = _video_num_frames
    VIDEO_GRID_T       = _video_num_frames // VISION_TEMPORAL_PATCH_SIZE
    VIDEO_FRAME_SEQLEN = _video_total_vision_tokens // VIDEO_GRID_T
else:
    VIDEO_NUM_FRAMES   = 0
    VIDEO_GRID_T       = 0
    VIDEO_FRAME_SEQLEN = 0

if isinstance(vision_embed_size, int) and isinstance(vision_batch_size, int) and vision_batch_size:
    IMAGE_SEQLEN_PER_IMAGE = vision_embed_size // vision_batch_size
else:
    IMAGE_SEQLEN_PER_IMAGE = 0

# --- Rotary Image (Prefill + Decode) ---
ort_session_Rotary_Image_Prefill = create_session(onnx_model_Rotary_Image_Prefill, **packed_settings)
binding_Rotary_Image_Prefill  = ort_session_Rotary_Image_Prefill.io_binding()
in_name_Rotary_Image_Prefill  = get_in_names(ort_session_Rotary_Image_Prefill)
out_name_Rotary_Image_Prefill = get_out_names(ort_session_Rotary_Image_Prefill)

ort_session_Rotary_Image_Decode = create_session(onnx_model_Rotary_Image_Decode, **packed_settings)
binding_Rotary_Image_Decode  = ort_session_Rotary_Image_Decode.io_binding()
in_name_Rotary_Image_Decode  = get_in_names(ort_session_Rotary_Image_Decode)[0]
out_name_Rotary_Image_Decode = get_out_names(ort_session_Rotary_Image_Decode)

# --- Rotary Video (Prefill + Decode) ---
ort_session_Rotary_Video_Prefill = create_session(onnx_model_Rotary_Video_Prefill, **packed_settings)
binding_Rotary_Video_Prefill  = ort_session_Rotary_Video_Prefill.io_binding()
in_name_Rotary_Video_Prefill  = get_in_names(ort_session_Rotary_Video_Prefill)
out_name_Rotary_Video_Prefill = get_out_names(ort_session_Rotary_Video_Prefill)

ort_session_Rotary_Video_Decode = create_session(onnx_model_Rotary_Video_Decode, **packed_settings)
binding_Rotary_Video_Decode  = ort_session_Rotary_Video_Decode.io_binding()
in_name_Rotary_Video_Decode  = get_in_names(ort_session_Rotary_Video_Decode)[0]
out_name_Rotary_Video_Decode = get_out_names(ort_session_Rotary_Video_Decode)

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

# Cross-check the graph-derived state layout against the exported metadata. The hybrid
# Main graph emits full-attention KV blocks plus 2 tensors (conv + recurrent) per linear layer.
_meta_full_layers   = int(model_meta['num_full_attention_layers'])
_meta_linear_layers = int(model_meta['num_linear_attention_layers'])
_meta_kv_tensors    = int(model_meta['kv_num_tensors'])
_expected_state_tensors = _meta_kv_tensors + 2 * _meta_linear_layers
if (num_full_layers_Main != _meta_full_layers
        or num_linear_layers_Main != _meta_linear_layers
        or num_keys_values_Main != _expected_state_tensors):
    raise RuntimeError(
        "LLM_Metadata.onnx disagrees with LLM_Main.onnx state layout: "
        f"metadata full={_meta_full_layers}, linear={_meta_linear_layers}, "
        f"kv_tensors={_meta_kv_tensors} (=> {_expected_state_tensors} state tensors); "
        f"graph full={num_full_layers_Main}, linear={num_linear_layers_Main}, "
        f"state tensors={num_keys_values_Main}."
    )
if len(in_name_Main) != num_keys_values_Main + 4 or len(out_name_Main) != num_keys_values_Main + 1:
    raise RuntimeError(
        "LLM_Main.onnx I/O arity unexpected: "
        f"{len(in_name_Main)} inputs / {len(out_name_Main)} outputs for {num_keys_values_Main} state tensors "
        "(expected state+4 inputs, state+1 outputs)."
    )

kv_dtype_str = in_meta_Main[0].type
hidden_dtype_Main = np.float16 if "float16" in in_meta_Main[num_keys_values_Main].type else np.float32

vocab_size = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]


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
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )
        k_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_key_biases[0]],
                scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
            ) if in_name_Main_key_biases else None
        )
        v_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_value_biases[0]],
                scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
            ) if in_name_Main_value_biases else None
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
    in_meta_Main_by_name[in_name_Main_keys[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
)
past_values_Main = create_ort_with_meta_shape(
    in_meta_Main_by_name[in_name_Main_values[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
)
past_conv_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_conv_states[0]], np.float16, kv_device, DEVICE_ID,
    ) if in_name_Main_conv_states else None
)
past_recurrent_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_recurrent_states[0]], np.float16, kv_device, DEVICE_ID,
    ) if in_name_Main_recurrent_states else None
)


# ══════════════════════════════════════════════════════════════════════════════
# Determine mode and build prompt
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)


# Determine test modes
test_image_paths = TEST_IMAGE if isinstance(TEST_IMAGE, list) else ([TEST_IMAGE] if TEST_IMAGE else [])
valid_images = [p for p in test_image_paths if is_valid_image_path(p)]
num_runtime_images = min(len(valid_images), vision_batch_size)

test_modes = []
if RUN_MODE == "image":
    if not valid_images:
        raise RuntimeError("--mode image requested, but no valid TEST_IMAGE path is configured.")
    test_modes.append(("image", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
elif RUN_MODE == "video":
    if not is_valid_video_path(TEST_VIDEO):
        raise RuntimeError("--mode video requested, but TEST_VIDEO is not valid.")
    test_modes.append(("video", TEST_QUERY[1] if len(TEST_QUERY) > 1 else TEST_QUERY[0]))
elif RUN_MODE == "text":
    test_modes.append(("text", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
else:
    if valid_images:
        test_modes.append(("image", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
    if is_valid_video_path(TEST_VIDEO):
        test_modes.append(("video", TEST_QUERY[1] if len(TEST_QUERY) > 1 else TEST_QUERY[0]))
    if not test_modes:
        test_modes.append(("text", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))


for INPUT_MODE, current_query in test_modes:
    print(f"\n{'═' * 56}")
    print(f"  Running test: mode={INPUT_MODE}, query=\"{current_query}\"")
    print(f"{'═' * 56}")

    if INPUT_MODE == "video":
        mode = "video"
        query = current_query
        tokens, mm_token_type_ids = build_prompt_and_mm_types(
            query,
            "video",
            num_frames=VIDEO_GRID_T,
            frame_seqlen=VIDEO_FRAME_SEQLEN,
            fps=VIDEO_FPS,
        )
    elif INPUT_MODE == "image":
        mode = "image"
        query = current_query
        tokens, mm_token_type_ids = build_prompt_and_mm_types(
            query,
            "image",
            num_images=vision_batch_size,
            image_seqlen=IMAGE_SEQLEN_PER_IMAGE,
        )
    else:
        mode = "text"
        query = current_query
        tokens, mm_token_type_ids = build_prompt_and_mm_types(query, "text")

    num_prefill = tokens.shape[-1]

    # ══════════════════════════════════════════════════════════════════════════════
    # Shared OrtValue buffers (recreated for each test mode)
    # ══════════════════════════════════════════════════════════════════════════════
    input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
    topK             = create_ort_with_data([TOP_K],       np.int64, device_type, DEVICE_ID)
    beam_size_ort    = create_ort_with_data([BEAM_SIZE],   np.int64, device_type, DEVICE_ID)

    attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_cos_buf = create_ort_with_shape(out_meta_Rotary_Text_Decode[0].shape, hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_sin_buf = create_ort_with_shape(out_meta_Rotary_Text_Decode[1].shape, hidden_dtype_Main, device_type, DEVICE_ID)
    hidden_states_buf = create_ort_with_meta_shape(
        in_meta_Main[num_keys_values_Main],
        hidden_dtype_Main, device_type, DEVICE_ID,
        batch_size=BEAM_SIZE, seq_axis=1, seq_len=1,
    )
    save_id_buf = create_ort_with_shape((BEAM_SIZE, 0), np.int32, device_type, DEVICE_ID)
    prefill_logits_buf = create_ort_with_shape((1, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    decode_logits_buf = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    max_idx_buf = create_ort_with_shape((1, 1), np.int32, device_type, DEVICE_ID)

    # Reset KV state buffers for fresh generation
    past_keys_Main = create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_keys[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
    )
    past_values_Main = create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_values[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
    )
    if in_name_Main_conv_states:
        past_conv_states_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_conv_states[0]], np.float16, kv_device, DEVICE_ID,
        )
    if in_name_Main_recurrent_states:
        past_recurrent_states_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_recurrent_states[0]], np.float16, kv_device, DEVICE_ID,
        )
    if k_scales_Main is not None:
        k_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )
    if k_biases_Main is not None:
        k_biases_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_biases[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_biases_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_biases[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )

    # Recreate IO bindings for a fresh run
    binding_Embed = ort_session_Embed.io_binding()
    binding_Main = ort_session_Main.io_binding()
    binding_Vision = ort_session_Vision.io_binding()
    binding_Image_Preprocess = ort_session_Image_Preprocess.io_binding()
    binding_Video_Preprocess = ort_session_Video_Preprocess.io_binding()
    binding_Concat_Image = ort_session_Concat_Image.io_binding()
    binding_Concat_Video = ort_session_Concat_Video.io_binding()
    binding_Rotary_Image_Prefill = ort_session_Rotary_Image_Prefill.io_binding()
    binding_Rotary_Image_Decode = ort_session_Rotary_Image_Decode.io_binding()
    binding_Rotary_Video_Prefill = ort_session_Rotary_Video_Prefill.io_binding()
    binding_Rotary_Video_Decode = ort_session_Rotary_Video_Decode.io_binding()
    binding_Rotary_Text_Prefill = ort_session_Rotary_Text_Prefill.io_binding()
    binding_Rotary_Text_Decode = ort_session_Rotary_Text_Decode.io_binding()


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
            [save_id_buf, beam_size_ort],
        )
        bind_ort_in_buf(
            binding_Second_Beam,
            in_name_Second_Beam[num_keys_values_Main_plus_3:],
            [beam_size_ort, topK],
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
        penalty_range_ort = create_ort_with_data([PENALTY_RANGE], np.int64, device_type, DEVICE_ID)
        bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range_ort])


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

    if mode == "image":
        print("\nStart to Process the Image(s)...")
        vision_start_time = time.time()

        # Load and preprocess images
        images = []
        for img_path in valid_images[:num_runtime_images]:
            images.append(
                load_image_letterbox(
                    img_path,
                    input_image_size[0],
                    input_image_size[1],
                )
            )

        # Keep the exported fixed-slot image contract stable by padding unused image
        # slots with a neutral gray image that stays close to zero after normalization.
        blank_image = np.full((3, input_image_size[0], input_image_size[1]), 128, dtype=np.uint8)
        while len(images) < vision_batch_size:
            images.append(blank_image)
        pixel_values = np.stack(images, axis=0)  # [N, 3, H, W]
        pixel_values = np.expand_dims(pixel_values, axis=1)  # [N, 1, 3, H, W]

        # Run image preprocess
        binding_Image_Preprocess.bind_ortvalue_input(
            in_name_Image_Preprocess,
            onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID),
        )
        bind_ort_out(binding_Image_Preprocess, out_name_Image_Preprocess, _ort_device_type)
        run(ort_session_Image_Preprocess, binding_Image_Preprocess)
        preprocess_outputs = binding_Image_Preprocess.get_outputs()

        # Run shared vision encoder
        for i, name in enumerate(in_name_Vision):
            binding_Vision.bind_ortvalue_input(name, preprocess_outputs[i])
        bind_ort_out(binding_Vision, [out_name_Vision], _ort_device_type)
        run(ort_session_Vision, binding_Vision)
        vision_hidden_states = binding_Vision.get_outputs()[0]

        print(f"\nImage Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds")

        # Image concat: adds vision tokens to prefill
        num_prefill += vision_embed_size
        ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
        generate_limit -= vision_embed_size

        binding_Concat_Image.bind_ortvalue_input(in_name_Concat_Image[0], hidden_states)
        binding_Concat_Image.bind_ortvalue_input(in_name_Concat_Image[1], vision_hidden_states)
        bind_ort_out(binding_Concat_Image, [out_name_Concat_Image], _ort_device_type)
        run(ort_session_Concat_Image, binding_Concat_Image)
        concat_hidden_states = binding_Concat_Image.get_outputs()[0]

        # Image rotary prefill
        bind_ort_in_buf(binding_Rotary_Image_Prefill, in_name_Rotary_Image_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Image_Prefill, out_name_Rotary_Image_Prefill, _ort_device_type)
        run(ort_session_Rotary_Image_Prefill, binding_Rotary_Image_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Image_Prefill.get_outputs()
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], concat_hidden_states)

        # Set decode rotary (session/names only; ping-pong counter bound after the branches)
        ort_session_Rotary_Decode = ort_session_Rotary_Image_Decode
        binding_Rotary_Decode = binding_Rotary_Image_Decode
        in_name_Rotary_Decode = in_name_Rotary_Image_Decode
        out_name_Rotary_Decode = out_name_Rotary_Image_Decode

    elif mode == "video":
        print("\nStart to Process the Video...")
        vision_start_time = time.time()

        video_frames, sampled_frame_indices = sample_video_frames(
            TEST_VIDEO,
            VIDEO_FPS,
            VIDEO_NUM_FRAMES,
            VIDEO_MIN_FRAMES,
            VIDEO_MAX_FRAMES,
            input_video_size,
        )
        del sampled_frame_indices

        # Run video preprocess
        binding_Video_Preprocess.bind_ortvalue_input(
            in_name_Video_Preprocess,
            onnxruntime.OrtValue.ortvalue_from_numpy(video_frames, device_type, DEVICE_ID),
        )
        bind_ort_out(binding_Video_Preprocess, out_name_Video_Preprocess, _ort_device_type)
        run(ort_session_Video_Preprocess, binding_Video_Preprocess)
        preprocess_outputs = binding_Video_Preprocess.get_outputs()

        # Run shared vision encoder
        for i, name in enumerate(in_name_Vision):
            binding_Vision.bind_ortvalue_input(name, preprocess_outputs[i])
        bind_ort_out(binding_Vision, [out_name_Vision], _ort_device_type)
        run(ort_session_Vision, binding_Vision)
        vision_hidden_states = binding_Vision.get_outputs()[0]

        print(f"\nVideo Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds")

        # Video concat: replaces pads in-place, NO change to num_prefill
        binding_Concat_Video.bind_ortvalue_input(in_name_Concat_Video[0], hidden_states)
        binding_Concat_Video.bind_ortvalue_input(in_name_Concat_Video[1], vision_hidden_states)
        bind_ort_out(binding_Concat_Video, [out_name_Concat_Video], _ort_device_type)
        run(ort_session_Concat_Video, binding_Concat_Video)
        concat_hidden_states = binding_Concat_Video.get_outputs()[0]

        # Video rotary prefill
        bind_ort_in_buf(binding_Rotary_Video_Prefill, in_name_Rotary_Video_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Video_Prefill, out_name_Rotary_Video_Prefill, _ort_device_type)
        run(ort_session_Rotary_Video_Prefill, binding_Rotary_Video_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Video_Prefill.get_outputs()
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], concat_hidden_states)

        # Set decode rotary (session/names only; ping-pong counter bound after the branches)
        ort_session_Rotary_Decode = ort_session_Rotary_Video_Decode
        binding_Rotary_Decode = binding_Rotary_Video_Decode
        in_name_Rotary_Decode = in_name_Rotary_Video_Decode
        out_name_Rotary_Decode = out_name_Rotary_Video_Decode

    else:
        # Text-only mode
        bind_ort_in_buf(binding_Rotary_Text_Prefill, in_name_Rotary_Text_Prefill, [ids_len, init_history_len])
        bind_ort_out(binding_Rotary_Text_Prefill, out_name_Rotary_Text_Prefill, _ort_device_type)
        run(ort_session_Rotary_Text_Prefill, binding_Rotary_Text_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Text_Prefill.get_outputs()
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states)

        ort_session_Rotary_Decode = ort_session_Rotary_Text_Decode
        binding_Rotary_Decode = binding_Rotary_Text_Decode
        in_name_Rotary_Decode = in_name_Rotary_Text_Decode
        out_name_Rotary_Decode = out_name_Rotary_Text_Decode

    bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos, rotary_sin, attention_mask])

    kv_seq_len_next = create_ort_with_shape(tuple(kv_seq_len.shape()), np.int64, device_type, DEVICE_ID)
    binding_Rotary_Decode.bind_ortvalue_input(in_name_Rotary_Decode, kv_seq_len)
    bind_ort_out_buf(binding_Rotary_Decode, out_name_Rotary_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len_next])

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

    if MAX_NEW_TOKENS is not None:
        generate_limit = min(generate_limit, max(0, int(MAX_NEW_TOKENS)))

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
    print(f"\nTest Question: {query}\nLLM Answering:")

    num_decode = 0
    save_id    = None

    while num_decode < generate_limit:

        # ── 1. Run Main Model ──────────────────────────────────────────────────────
        run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        # ── 2. Apply Repetition Penalty (if enabled) ─────────────────────
        if USE_PENALTY and num_decode >= PENALTY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            run(ort_session_Penalty, binding_Penalty)

        # ── 3. Token Selection ─────────────────────────────────────────────────────
        if USE_BEAM_SEARCH:
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

        # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
        run(ort_session_Embed, binding_Embed)
        run(ort_session_Rotary_Decode, binding_Rotary_Decode)

        kv_seq_len.update_inplace(kv_seq_len_next)
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
        f"\n\n{chr(9472) * 56}\n"
        f"  Generated Output\n"
        f"{chr(9472) * 56}\n"
        f"{result}\n"
        f"{chr(9472) * 56}\n\n"
        f"  Performance Summary\n"
        f"{chr(9472) * 56}\n"
        f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
        f"  {chr(9472) * 48}\n"
        f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
        f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
        f"  {chr(9472) * 48}\n"
        f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
        f"{chr(9472) * 56}\n"
    )
