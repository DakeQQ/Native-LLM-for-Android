import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnxruntime.capi import _pybind_state as C
from PIL import Image
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a merged-ONNX Qwen multimodal inference demo.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen_Optimized",
        help="Folder containing merged ONNX graphs and LLM_SharedInitializers.onnx(.data).",
    )
    parser.add_argument(
        "--tokenizer-folder",
        type=Path,
        default=Path.home() / "Downloads" / "Qwen3.5-0.8B",
        help="HF checkpoint/tokenizer folder used to tokenize the demo prompt.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "image", "video", "text"),
        default="auto",
        help="Limit the demo to one input mode. Default auto runs available image/video tests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional cap for generated tokens per test.")
    return parser.parse_args()


args = parse_args()

download_path = str(args.tokenizer_folder.expanduser().resolve())
onnx_folder = args.model_folder.expanduser().resolve()
RUN_MODE = args.mode
MAX_NEW_TOKENS = args.max_new_tokens

METADATA_MODEL_NAME = "LLM_Metadata.onnx"
onnx_model_Metadata = str(onnx_folder / METADATA_MODEL_NAME)

_SCRIPT_DIR = Path(__file__).resolve().parent

TEST_IMAGE = [str(_SCRIPT_DIR / "psyduck.png")]
TEST_VIDEO = str(_SCRIPT_DIR / "test_video_8s.mp4")
TEST_QUERY = ["Describe this image.", "Describe this video."]
ENABLE_THINKING = False

USE_SAMPLING = True
TEMPERATURE = 0.8
TOP_P = 0.95
REPEAT_PENALTY = 1.0
PENALTY_RANGE = 20
TOP_K = 3

ORT_LOG = False
ORT_FP16 = False
ORT_Accelerate_Providers = []
MAX_THREADS = 0
DEVICE_ID = 0

TOP_K = max(1, TOP_K)
USE_PENALTY = (REPEAT_PENALTY != 1.0) and not USE_SAMPLING


def build_assistant_prompt_prefix() -> str:
    assistant_prefix = "<|im_start|>assistant\n"
    if ENABLE_THINKING:
        return assistant_prefix + "<think>\n"
    return assistant_prefix + "<think>\n\n</think>\n\n"


def build_text_prompt(query: str) -> str:
    return f"<|im_start|>user\n{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def build_image_prompt(query: str, num_images: int = 1) -> str:
    vision_placeholders = "<|vision_start|><|vision_end|>" * num_images
    return f"<|im_start|>user\n{vision_placeholders}{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def build_video_timestamps(num_frames: int, fps: float = 2.0, frame_indices=None) -> List[float]:
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


def build_video_prompt(query: str, num_frames: int, frame_seqlen: int, fps: float = 2.0, frame_indices=None) -> str:
    parts = ["<|im_start|>user\n"]
    for timestamp in build_video_timestamps(num_frames, fps, frame_indices):
        parts.append(f"<{timestamp:.1f} seconds>")
        parts.append("<|vision_start|>")
        parts.append("<|video_pad|>" * frame_seqlen)
        parts.append("<|vision_end|>")
    parts.append(f"{query}<|im_end|>\n{build_assistant_prompt_prefix()}")
    return "".join(parts)


def build_prompt(query, mode, num_frames=0, frame_seqlen=0, fps=2.0, num_images=1):
    if mode == "text":
        return build_text_prompt(query)
    if mode == "image":
        return build_image_prompt(query, num_images)
    if mode == "video":
        return build_video_prompt(query, num_frames, frame_seqlen, fps)
    raise ValueError(f"Unsupported mode: {mode}")


def is_valid_image_path(path):
    if not path or not os.path.exists(path):
        return False
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".raw"}


def is_valid_video_path(path):
    if not path or not os.path.exists(path):
        return False
    return os.path.splitext(path)[1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def load_image_letterbox(path, target_h, target_w):
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
        canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
        canvas.paste(image, ((target_w - resize_w) // 2, (target_h - resize_h) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1))


def sample_video_frames(video_path, target_fps, num_frames, min_frames, max_frames, frame_size):
    try:
        import cv2
    except ImportError:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,nb_frames,duration:format=duration",
                "-of", "json", video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe_data = json.loads(probe.stdout)
        stream = probe_data.get("streams", [{}])[0]
        rate_text = stream.get("avg_frame_rate", "0/1")
        numerator, denominator = (float(value) for value in rate_text.split("/", 1))
        source_fps = numerator / denominator if denominator else float(target_fps)
        duration = float(
            stream.get("duration")
            or probe_data.get("format", {}).get("duration")
            or 0.0
        )
        frame_count_text = stream.get("nb_frames")
        total_frames = (
            int(frame_count_text)
            if frame_count_text not in (None, "N/A")
            else int(round(duration * source_fps))
        )
        if source_fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Unable to probe video timing: {video_path}")

        frame_step = max(source_fps / max(float(target_fps), 1e-6), 1.0)
        candidate_indices = np.arange(
            0, total_frames, frame_step, dtype=np.float64
        ).astype(np.int64)
        if candidate_indices.size == 0:
            candidate_indices = np.array([0], dtype=np.int64)
        if candidate_indices.size < min_frames:
            candidate_indices = np.linspace(
                0, total_frames - 1, min_frames, dtype=np.int64
            )
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
                    np.full(
                        num_frames - candidate_indices.size,
                        candidate_indices[-1],
                        dtype=np.int64,
                    ),
                ]
            )

        target_height, target_width = frame_size
        expected_bytes = target_height * target_width * 3
        frames = []
        for frame_index in sampled_indices.tolist():
            timestamp = frame_index / source_fps
            decoded = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-ss", f"{timestamp:.9f}",
                    "-i", video_path, "-frames:v", "1",
                    "-vf", f"scale={target_width}:{target_height}:flags=bicubic",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            if len(decoded) != expected_bytes:
                continue
            frame = np.frombuffer(decoded, dtype=np.uint8).reshape(
                target_height, target_width, 3
            )
            frames.append(np.transpose(frame, (2, 0, 1)).copy())
        if not frames:
            raise RuntimeError(f"Unable to decode any frames from video: {video_path}")
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return np.stack(frames[:num_frames], axis=0), sampled_indices.tolist()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(target_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Unable to determine frame count for video: {video_path}")
        frame_step = max(src_fps / max(float(target_fps), 1e-6), 1.0)
        candidate_indices = np.arange(0, total_frames, frame_step, dtype=np.float64).astype(np.int64)
        if candidate_indices.size == 0:
            candidate_indices = np.array([0], dtype=np.int64)
        if candidate_indices.size < min_frames:
            candidate_indices = np.linspace(0, total_frames - 1, min_frames, dtype=np.int64)
        if candidate_indices.size > max_frames:
            candidate_indices = candidate_indices[:max_frames]
        if candidate_indices.size >= num_frames:
            sampled_indices = candidate_indices[np.linspace(0, candidate_indices.size - 1, num_frames, dtype=np.int64)]
        else:
            sampled_indices = np.concatenate([candidate_indices, np.full(num_frames - candidate_indices.size, candidate_indices[-1], dtype=np.int64)])

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


_DEFAULT_MODEL_FILE_NAMES = {
    "shared_initializers": "LLM_SharedInitializers.onnx",
    "shared_initializers_data": "LLM_SharedInitializers.onnx.data",
    "vision": "LLM_Vision.onnx",
    "image_preprocess": "LLM_Image_Preprocess.onnx",
    "video_preprocess": "LLM_Video_Preprocess.onnx",
}
for _modality in ("text", "image", "video"):
    _title = _modality.capitalize()
    _DEFAULT_MODEL_FILE_NAMES.update({
        f"{_modality}_prefill_greedy": f"LLM_{_title}PrefillGreedy.onnx",
        f"{_modality}_prefill_penalty_greedy": f"LLM_{_title}PrefillPenaltyGreedy.onnx",
        f"{_modality}_prefill_sampling": f"LLM_{_title}PrefillSampling.onnx",
        f"{_modality}_decode_greedy": f"LLM_{_title}DecodeGreedy.onnx",
        f"{_modality}_decode_penalty_greedy": f"LLM_{_title}DecodePenaltyGreedy.onnx",
        f"{_modality}_decode_sampling": f"LLM_{_title}DecodeSampling.onnx",
    })

_MODEL_FILE_METADATA_KEYS = {
    "shared_initializers": ("model_file_name_shared_initializers", None),
    "shared_initializers_data": ("model_file_name_shared_initializers_data", None),
    "vision": ("model_file_name_vision", _DEFAULT_MODEL_FILE_NAMES["vision"]),
    "image_preprocess": ("model_file_name_image_preprocess", _DEFAULT_MODEL_FILE_NAMES["image_preprocess"]),
    "video_preprocess": ("model_file_name_video_preprocess", _DEFAULT_MODEL_FILE_NAMES["video_preprocess"]),
}
for _key, _default in _DEFAULT_MODEL_FILE_NAMES.items():
    if _key not in _MODEL_FILE_METADATA_KEYS:
        _MODEL_FILE_METADATA_KEYS[_key] = (f"model_file_name_{_key}", _default)

_UNSHAREABLE_INIT_TYPES = frozenset(getattr(TensorProto, name) for name in ("UINT4", "INT4", "FLOAT4E2M1") if hasattr(TensorProto, name))


def _external_data_map(init):
    return {entry.key: entry.value for entry in init.external_data}


def attach_shared_initializers(session_options, shared_model_path):
    shared_model_path = Path(shared_model_path)
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays = {}
    ort_values = []
    for init in shared_model.graph.initializer:
        if init.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        ext = _external_data_map(init)
        location = ext.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {init.name!r} is not stored as external data.")
        data_path = shared_model_path.parent / location
        offset = int(ext.get("offset", "0"))
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        shape = tuple(int(dim) for dim in init.dims)
        array = np.memmap(data_path, dtype=np_dtype, mode="r", offset=offset, shape=shape)
        arrays[init.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(init.name, ort_value)
    return arrays, ort_values


def load_metadata_carrier(model_path):
    meta_opts = onnxruntime.SessionOptions()
    meta_opts.log_severity_level = 0 if ORT_LOG else 4
    meta_opts.log_verbosity_level = 4
    meta_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        meta_session = onnxruntime.InferenceSession(model_path, sess_options=meta_opts, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError("LLM_Metadata.onnx is required to preload Native-LLM metadata. Re-export with Export_Qwen.py.") from exc
    metadata = meta_session.get_modelmeta().custom_metadata_map
    if not metadata.get("native_llm_metadata_version"):
        raise RuntimeError("LLM_Metadata.onnx carries no Native-LLM metadata_props. Re-export the model with Export_Qwen.py.")
    return metadata


def load_model_file_names(meta):
    missing = [key for _, (key, default_file_name) in _MODEL_FILE_METADATA_KEYS.items() if not meta.get(key) and default_file_name is None]
    if missing:
        raise RuntimeError("LLM_Metadata.onnx is missing model file-name metadata. Missing: " + ", ".join(missing))
    file_names = {}
    for role, (key, default_file_name) in _MODEL_FILE_METADATA_KEYS.items():
        value = meta.get(key) or default_file_name
        path = Path(value)
        if path.is_absolute() or path.name != value:
            raise RuntimeError(f"Metadata key {key!r} must contain a file name, got {value!r}.")
        file_names[role] = value
    return file_names


model_meta = load_metadata_carrier(onnx_model_Metadata)
MODEL_FILE_NAMES = load_model_file_names(model_meta)
ORT_FP16 = model_meta.get("activations_fp16") == "1"

IMAGE_TOKEN_ID = int(model_meta["image_token_id"])
VIDEO_TOKEN_ID = int(model_meta["video_token_id"])
VISION_TEMPORAL_PATCH_SIZE = int(model_meta["temporal_patch_size"])
MAX_SEQ_LEN = int(model_meta["max_seq_len"])
VIDEO_FPS = float(model_meta["video_fps"])
VIDEO_MIN_FRAMES = int(model_meta["video_min_frames"])
VIDEO_MAX_FRAMES = int(model_meta["video_max_frames"])
INPUT_IMAGE_SIZE = [int(s) for s in model_meta["input_image_size"].split(",")]
INPUT_VIDEO_SIZE = [int(s) for s in model_meta["input_video_size"].split(",")]
STOP_TOKEN = [int(t) for t in model_meta.get("stop_token_ids", "").split(",") if t]
if not STOP_TOKEN:
    STOP_TOKEN = [int(t) for t in model_meta.get("eos_token_ids", "").split(",") if t]
STOP_TOKEN_SET = set(STOP_TOKEN)


def create_session_options():
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 0 if ORT_LOG else 4
    session_opts.log_verbosity_level = 4
    session_opts.inter_op_num_threads = MAX_THREADS
    session_opts.intra_op_num_threads = MAX_THREADS
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else "",
    }
    for key, value in entries.items():
        session_opts.add_session_config_entry(key, value)
    return session_opts


def create_run_options():
    options = onnxruntime.RunOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def resolve_execution_provider():
    if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
        return "cpu", C.OrtDevice.cpu(), [{"device_type": "CPU", "precision": "ACCURACY", "num_of_threads": MAX_THREADS if MAX_THREADS != 0 else 8, "num_streams": 1, "enable_opencl_throttling": False, "enable_qdq_optimizer": False, "disable_dynamic_shapes": False}]
    if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        return "cuda", C.OrtDevice.cuda(), [{"device_id": DEVICE_ID, "gpu_mem_limit": 24 * (1024 ** 3), "arena_extend_strategy": "kNextPowerOfTwo", "cudnn_conv_algo_search": "EXHAUSTIVE", "sdpa_kernel": "2", "use_tf32": "1", "fuse_conv_bias": "1", "cudnn_conv_use_max_workspace": "1", "cudnn_conv1d_pad_to_nc1d": "0", "tunable_op_enable": "0", "tunable_op_tuning_enable": "0", "tunable_op_max_tuning_duration_ms": 10, "do_copy_in_default_stream": "0", "enable_cuda_graph": "0", "prefer_nhwc": "0", "enable_skip_layer_norm_strict_mode": "0", "use_ep_level_unified_stream": "0"}]
    if "DmlExecutionProvider" in ORT_Accelerate_Providers:
        return "dml", C.OrtDevice.dml(), [{"device_id": DEVICE_ID, "performance_preference": "high_performance", "device_filter": "gpu", "disable_metacommands": "false", "enable_graph_capture": "false", "enable_graph_serialization": "false"}]
    return "cpu", C.OrtDevice.cpu(), None


run_options = create_run_options()
device_type, _ort_device_type, provider_options = resolve_execution_provider()
disabled_optimizers = ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
ORT_DEVICE = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = "cpu" if device_type == "dml" else device_type


def create_merged_session(model_path, shared_path):
    session_opts = create_session_options()
    shared_initializer_refs = attach_shared_initializers(session_opts, shared_path)
    session = onnxruntime.InferenceSession(str(model_path), sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, disabled_optimizers=disabled_optimizers)
    session._native_llm_shared_initializers = shared_initializer_refs
    return session


def create_plain_session(model_path):
    return onnxruntime.InferenceSession(str(model_path), sess_options=create_session_options(), providers=ORT_Accelerate_Providers, provider_options=provider_options, disabled_optimizers=disabled_optimizers)


def run_iobinding(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def _np_dtype(type_name):
    for key, dtype in (("float16", np.float16), ("float", np.float32), ("uint8", np.uint8), ("int8", np.int8), ("int32", np.int32), ("int64", np.int64)):
        if key in type_name:
            return dtype
    raise ValueError(f"Unsupported ORT tensor type: {type_name}")


def _session_input_dtypes(sess):
    return {meta.name: _np_dtype(meta.type) for meta in sess.get_inputs()}


def _state_seq_axis(value_meta):
    symbolic_axes = [index for index, dim in enumerate(value_meta.shape) if index != 0 and not isinstance(dim, int)]
    if len(symbolic_axes) == 1:
        return symbolic_axes[0]
    return None


def _zero_from_meta(meta, batch_size=1):
    shape = list(meta.shape)
    seq_axis = _state_seq_axis(meta)
    for index, dim in enumerate(shape):
        if index == 0:
            shape[index] = batch_size
        elif seq_axis is not None and index == seq_axis:
            shape[index] = 0
        elif not isinstance(dim, int):
            shape[index] = 1
    return np.zeros(tuple(shape), dtype=_np_dtype(meta.type))


def _ov(arr, device=None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), device or device_type, DEVICE_ID)


def _bind_outputs_device(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, ORT_DEVICE)


def _decode_dynamic_output_names(io_plan, output_meta_by_name):
    names = [
        name for name in io_plan["state_out"]
        if _state_seq_axis(output_meta_by_name[name]) is not None
    ]
    if io_plan["save_id_out"] is not None:
        names.append(io_plan["save_id_out"])
    return names


def _save_id_input_names(strategy, is_decode, inputs):
    if strategy == "greedy":
        candidates = []
    elif strategy == "sampling":
        candidates = ["sampling_previous_ids"]
    elif strategy == "penalty_greedy":
        candidates = ["penalty_greedy_save_id_in"] if not is_decode else ["penalty_save_id_in", "penalty_greedy_save_id_in"]
    else:
        candidates = []
    missing = [name for name in candidates if name not in inputs]
    if missing:
        raise RuntimeError(f"Merged graph is missing expected save-id input(s): {missing}")
    return candidates


def _vision_input_map(inputs):
    """Map standalone Vision output names to their merged-prefill input names."""
    mapped = {}
    for input_name in inputs:
        if input_name.endswith("vision_hidden_states"):
            mapped["vision_hidden_states"] = input_name
            continue
        marker = "in_deepstack_feature_"
        if marker in input_name:
            index = input_name.rsplit("_", 1)[-1]
            if index.isdigit():
                mapped[f"deepstack_feature_{index}"] = input_name
    return mapped


def plan_merged_io(sess, strategy, state_count, is_decode):
    inputs = [item.name for item in sess.get_inputs()]
    outputs = [item.name for item in sess.get_outputs()]
    if len(inputs) < state_count or len(outputs) < state_count:
        raise RuntimeError(f"Merged graph has too few state tensors: inputs={len(inputs)} outputs={len(outputs)} expected={state_count}.")
    state_in = inputs[:state_count]
    state_out = outputs[:state_count]
    if any(not name.startswith("in_") for name in state_in):
        raise RuntimeError(f"Merged graph state inputs are not leading as expected: {state_in[:3]}")
    if any(not name.startswith("out_") for name in state_out):
        raise RuntimeError(f"Merged graph state outputs are not leading as expected: {state_out[:3]}")

    tail = outputs[state_count:]
    if strategy == "greedy":
        max_idx_out, kv_seq_out = tail[:2]
        save_id_out = None
    elif strategy == "sampling":
        max_idx_out, save_id_out, kv_seq_out = tail[:3]
    else:
        max_idx_out, save_id_out, kv_seq_out = tail[:3]

    token_in = next((name for name in ("embed_input_ids", "input_ids") if name in inputs), None)
    if token_in is None:
        raise RuntimeError(f"Merged graph is missing the token input (embed_input_ids/input_ids); saw {inputs[state_count:state_count + 4]!r}.")
    kv_seq_in = None
    if is_decode:
        kv_seq_in = next((name for name in inputs if name.startswith("decode_kv_seq_len")), None)
        if kv_seq_in is None:
            raise RuntimeError(f"Decode graph is missing decode_kv_seq_len input; saw {inputs[state_count:state_count + 4]!r}.")
    return {
        "in_names": inputs,
        "token_in": token_in,
        "out_names": outputs,
        "state_in": state_in,
        "state_out": state_out,
        "max_idx_out": max_idx_out,
        "kv_seq_out": kv_seq_out,
        "kv_seq_in": kv_seq_in,
        "save_id_in": _save_id_input_names(strategy, is_decode, set(inputs)),
        "save_id_out": save_id_out,
        "vision_inputs": _vision_input_map(inputs),
    }


def _resolve_strategy():
    if USE_SAMPLING:
        return "sampling"
    if USE_PENALTY:
        return "penalty_greedy"
    return "greedy"


def _graph_role(modality, phase, strategy):
    return f"{modality}_{phase}_{strategy}"


def _generation_limit(prompt_tokens):
    limit = max(0, MAX_SEQ_LEN - prompt_tokens)
    if MAX_NEW_TOKENS is not None:
        limit = min(limit, max(0, int(MAX_NEW_TOKENS)))
    return limit


def _decode_static_inputs(strategy, input_dtypes):
    static_inputs = []
    if strategy == "penalty_greedy":
        static_inputs.extend([
            ("penalty_penalty_value", _ov(np.array([REPEAT_PENALTY], input_dtypes["penalty_penalty_value"]))),
            ("penalty_penalty_range", _ov(np.array([PENALTY_RANGE], input_dtypes["penalty_penalty_range"]))),
        ])
    if strategy == "sampling":
        static_inputs.extend([
            ("sampling_temperature", _ov(np.array([TEMPERATURE], input_dtypes["sampling_temperature"]))),
            ("sampling_top_k", _ov(np.array([TOP_K], input_dtypes["sampling_top_k"]))),
            ("sampling_top_p", _ov(np.array([TOP_P], input_dtypes["sampling_top_p"]))),
            ("sampling_repetition_penalty", _ov(np.array([REPEAT_PENALTY], input_dtypes["sampling_repetition_penalty"]))),
        ])
    return static_inputs


def _infer_vision_geometry():
    image_pre = create_plain_session(onnx_folder / MODEL_FILE_NAMES["image_preprocess"])
    vision_batch = image_pre._inputs_meta[0].shape[0]
    image_shape = image_pre._inputs_meta[0].shape
    input_image_size = INPUT_IMAGE_SIZE if isinstance(image_shape[3], int) and isinstance(image_shape[4], int) else INPUT_IMAGE_SIZE

    video_pre = create_plain_session(onnx_folder / MODEL_FILE_NAMES["video_preprocess"])
    video_shape = video_pre._inputs_meta[0].shape
    input_video_size = INPUT_VIDEO_SIZE if isinstance(video_shape[2], int) and isinstance(video_shape[3], int) else INPUT_VIDEO_SIZE
    video_num_frames = int(video_shape[0]) if isinstance(video_shape[0], int) else 0

    vision = create_plain_session(onnx_folder / MODEL_FILE_NAMES["vision"])
    return image_pre, video_pre, vision, int(vision_batch), input_image_size, input_video_size, video_num_frames


def _run_vision(mode, image_pre, video_pre, vision_sess, vision_batch, input_image_size, input_video_size, video_num_frames):
    if mode == "text":
        return None
    if mode == "image":
        valid_images = [path for path in (TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]) if is_valid_image_path(path)]
        images = [load_image_letterbox(path, input_image_size[0], input_image_size[1]) for path in valid_images[:vision_batch]]
        blank = np.full((3, input_image_size[0], input_image_size[1]), 128, dtype=np.uint8)
        while len(images) < vision_batch:
            images.append(blank)
        preprocess_array = np.expand_dims(np.stack(images, axis=0), axis=1)
        preprocess_sess = image_pre
    else:
        preprocess_array, _ = sample_video_frames(TEST_VIDEO, VIDEO_FPS, video_num_frames, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES, input_video_size)
        preprocess_sess = video_pre

    preprocess_meta = preprocess_sess.get_inputs()[0]
    preprocess_input = _ov(preprocess_array.astype(_np_dtype(preprocess_meta.type), copy=False))
    preprocess_binding = preprocess_sess.io_binding()
    preprocess_binding.bind_ortvalue_input(preprocess_meta.name, preprocess_input)
    _bind_outputs_device(preprocess_binding, [item.name for item in preprocess_sess.get_outputs()])
    run_iobinding(preprocess_sess, preprocess_binding)
    preprocess_outputs = preprocess_binding.get_outputs()

    vision_binding = vision_sess.io_binding()
    for meta, value in zip(vision_sess.get_inputs(), preprocess_outputs):
        vision_binding.bind_ortvalue_input(meta.name, value)
    vision_output_names = [item.name for item in vision_sess.get_outputs()]
    _bind_outputs_device(vision_binding, vision_output_names)
    run_iobinding(vision_sess, vision_binding)
    return dict(zip(vision_output_names, vision_binding.get_outputs()))


def run_merged_iobinding(mode, query, vision_outputs=None):
    strategy = _resolve_strategy()
    prefill_role = _graph_role(mode, "prefill", strategy)
    decode_role = _graph_role(mode, "decode", strategy)
    shared_path = onnx_folder / MODEL_FILE_NAMES["shared_initializers"]
    shared_data_path = onnx_folder / MODEL_FILE_NAMES["shared_initializers_data"]
    if not shared_path.exists() or not shared_data_path.exists():
        raise RuntimeError(f"Merged runtime requires {shared_path.name} and {shared_data_path.name} in {onnx_folder}.")
    prefill_path = onnx_folder / MODEL_FILE_NAMES[prefill_role]
    decode_path = onnx_folder / MODEL_FILE_NAMES[decode_role]
    if not prefill_path.exists() or not decode_path.exists():
        raise RuntimeError(f"Missing merged graph(s) for {mode}/{strategy}: {prefill_path.name}, {decode_path.name}")

    prefill_sess = create_merged_session(prefill_path, shared_path)
    decode_sess = create_merged_session(decode_path, shared_path)
    print(f"Usable Providers: {decode_sess.get_providers()}")

    state_count = int(model_meta["kv_num_tensors"]) + 2 * int(model_meta["num_linear_attention_layers"])
    prefill_io_plan = plan_merged_io(prefill_sess, strategy, state_count, is_decode=False)
    decode_io_plan = plan_merged_io(decode_sess, strategy, state_count, is_decode=True)
    decode_input_dtypes = _session_input_dtypes(decode_sess)
    vision_hidden_states = (
        vision_outputs.get("vision_hidden_states")
        if vision_outputs is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    if mode == "video":
        video_total_tokens = int(vision_hidden_states.shape()[1]) if vision_hidden_states is not None else 0
        video_tokens_per_prompt_frame = max(video_total_tokens // max(video_num_frames // VISION_TEMPORAL_PATCH_SIZE, 1), 1)
        prompt = build_prompt(query, mode, num_frames=max(video_num_frames // VISION_TEMPORAL_PATCH_SIZE, 1), frame_seqlen=video_tokens_per_prompt_frame, fps=VIDEO_FPS)
    else:
        prompt = build_prompt(query, mode, num_images=vision_batch_size)
    tokens = tokenizer(prompt, return_tensors="np")["input_ids"]
    embed_token_count = int(tokens.shape[-1])
    # Image concat GROWS the sequence (short text embeds + inserted vision rows), so the
    # rotary/mask/Main length must be the post-concat length. Video embeds already include
    # video_pad rows (concat replaces in place) and text has no vision, so they stay as-is.
    vision_growth = int(vision_hidden_states.shape()[1]) if mode == "image" and vision_hidden_states is not None else 0
    num_prefill = embed_token_count + vision_growth
    generate_limit = _generation_limit(num_prefill)

    prefill_input_meta_by_name = {item.name: item for item in prefill_sess.get_inputs()}
    prefill_input_dtypes = _session_input_dtypes(prefill_sess)
    prefill_binding = prefill_sess.io_binding()
    prefill_input_values = []

    def bind_prefill_input(name, arr, device=None):
        # Cast every runtime-built input to the dtype the graph declares (metadata-driven,
        # never hard-coded) so an export dtype change can't silently mismatch the binding.
        value = _ov(np.asarray(arr).astype(prefill_input_dtypes[name], copy=False), device)
        prefill_input_values.append(value)
        prefill_binding.bind_ortvalue_input(name, value)

    bind_prefill_input(prefill_io_plan["token_in"], tokens)
    if "prefill_ids_len" in prefill_io_plan["in_names"]:
        bind_prefill_input("prefill_ids_len", np.array([num_prefill]))
    if "prefill_history_len" in prefill_io_plan["in_names"]:
        bind_prefill_input("prefill_history_len", np.array([0]))
    if "prefill_cache_len" in prefill_io_plan["in_names"]:
        bind_prefill_input("prefill_cache_len", np.array([0]))
    for output_name, input_name in prefill_io_plan["vision_inputs"].items():
        if vision_outputs is None or output_name not in vision_outputs:
            raise RuntimeError(
                f"{mode} prefill graph requires Vision output {output_name!r}."
            )
        prefill_binding.bind_ortvalue_input(input_name, vision_outputs[output_name])
    for name in prefill_io_plan["state_in"]:
        bind_prefill_input(name, _zero_from_meta(prefill_input_meta_by_name[name], batch_size=1), kv_device)
    for name in prefill_io_plan["save_id_in"]:
        bind_prefill_input(name, np.zeros((1, 0)))
    if strategy == "sampling":
        bind_prefill_input("sampling_temperature", np.array([TEMPERATURE]))
        bind_prefill_input("sampling_top_k", np.array([TOP_K]))
        bind_prefill_input("sampling_top_p", np.array([TOP_P]))
        bind_prefill_input("sampling_repetition_penalty", np.array([REPEAT_PENALTY]))

    _bind_outputs_device(prefill_binding, prefill_io_plan["out_names"])
    prefill_start_time = time.time()
    run_iobinding(prefill_sess, prefill_binding)
    prefill_elapsed = time.time() - prefill_start_time
    # Outputs are bound in out_names order, so get_outputs() aligns 1:1 with out_names. Gather
    # positionally (integer index / list slice) instead of rebuilding a name->value dict: the
    # KV + linear-attention state block leads the outputs ([0, state_count)); tails are at fixed offsets.
    prefill_outputs = prefill_binding.get_outputs()
    prefill_out_names = prefill_io_plan["out_names"]
    prefill_max_idx_pos = prefill_out_names.index(prefill_io_plan["max_idx_out"])
    prefill_kv_seq_pos = prefill_out_names.index(prefill_io_plan["kv_seq_out"])
    prefill_next_token_pos = prefill_max_idx_pos
    prefill_save_id_pos = prefill_out_names.index(prefill_io_plan["save_id_out"]) if prefill_io_plan["save_id_out"] else -1

    # Hoist every decode-graph plan field into a local so the per-token loop touches no dict.
    decode_out_names = decode_io_plan["out_names"]
    decode_token_in = decode_io_plan["token_in"]
    decode_kv_seq_in = decode_io_plan["kv_seq_in"]
    decode_state_in = decode_io_plan["state_in"]
    decode_save_id_in = decode_io_plan["save_id_in"]
    decode_max_idx_pos = decode_out_names.index(decode_io_plan["max_idx_out"])
    decode_kv_seq_pos = decode_out_names.index(decode_io_plan["kv_seq_out"])
    decode_next_token_pos = decode_max_idx_pos
    decode_save_id_pos = decode_out_names.index(decode_io_plan["save_id_out"]) if decode_io_plan["save_id_out"] else -1
    decode_output_meta_by_name = {item.name: item for item in decode_sess.get_outputs()}
    decode_dynamic_out_names = _decode_dynamic_output_names(decode_io_plan, decode_output_meta_by_name)

    generated = []
    selected_token_id = prefill_outputs[prefill_max_idx_pos].numpy().flat[0]
    next_token_tensor = prefill_outputs[prefill_next_token_pos]
    cached_state_tensors = prefill_outputs[:state_count]
    kv_sequence_length = prefill_outputs[prefill_kv_seq_pos]
    saved_token_ids = prefill_outputs[prefill_save_id_pos] if prefill_save_id_pos >= 0 else None
    generated_count = 0

    print(f"\nTest Question: {query}\nLLM Answering:")
    if selected_token_id not in STOP_TOKEN_SET and generated_count < generate_limit:
        generated.append(selected_token_id)
        generated_count += 1
        print(tokenizer.decode(selected_token_id), end="", flush=True)

    static_inputs = _decode_static_inputs(strategy, decode_input_dtypes)
    decode_bindings = [decode_sess.io_binding(), decode_sess.io_binding()]
    for binding in decode_bindings:
        for name, value in static_inputs:
            binding.bind_ortvalue_input(name, value)
        _bind_outputs_device(binding, decode_out_names)

    # Two-binding ping-pong: each step's device-auto outputs feed the *other* binding on the
    # next step. KV/linear state and saved_token_ids GROW every step, so ORT re-allocates them
    # (fresh handle) and they -- plus their device outputs -- must be rebound every step.
    # kv_seq_len / next_token are fixed-shape outputs bound once, so each binding
    # keeps reading the *same* peer buffer (overwritten in place); their source only shifts
    # while the ping-pong warms up (prefill -> peer), so bind them on a binding's first two
    # uses and skip the otherwise-redundant per-step rebind afterward.
    control_rebinds_left = [2, 2]

    decode_step = 0
    decode_start_time = time.time()
    while generated_count < generate_limit and selected_token_id not in STOP_TOKEN_SET:
        binding_index = decode_step & 1
        binding = decode_bindings[binding_index]
        if control_rebinds_left[binding_index]:
            control_rebinds_left[binding_index] -= 1
            binding.bind_ortvalue_input(decode_kv_seq_in, kv_sequence_length)
            binding.bind_ortvalue_input(decode_token_in, next_token_tensor)
        for name, value in zip(decode_state_in, cached_state_tensors):
            binding.bind_ortvalue_input(name, value)
        for name in decode_save_id_in:
            binding.bind_ortvalue_input(name, saved_token_ids)
        _bind_outputs_device(binding, decode_dynamic_out_names)
        run_iobinding(decode_sess, binding)
        decode_outputs = binding.get_outputs()

        cached_state_tensors = decode_outputs[:state_count]
        selected_token_id = decode_outputs[decode_max_idx_pos].numpy().flat[0]
        if decode_save_id_pos != -1:
            saved_token_ids = decode_outputs[decode_save_id_pos]
        # kv_seq_len / next_token feed ONLY the warm-up rebinds at the top of the
        # loop; once both bindings lock onto their peer's fixed buffers (overwritten in place)
        # nothing reads a freshly fetched copy again. A step's fetch feeds the NEXT step's
        # rebind, so keep fetching while any binding still has a warm-up rebind pending.
        if any(control_rebinds_left):
            kv_sequence_length = decode_outputs[decode_kv_seq_pos]
            next_token_tensor = decode_outputs[decode_next_token_pos]
        if selected_token_id not in STOP_TOKEN_SET:
            generated.append(selected_token_id)
            generated_count += 1
            print(tokenizer.decode(selected_token_id), end="", flush=True)
        decode_step += 1
    decode_elapsed = time.time() - decode_start_time

    text = tokenizer.decode(generated, skip_special_tokens=True)
    total_elapsed = prefill_elapsed + decode_elapsed
    prefill_tps = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
    decode_tps = decode_step / decode_elapsed if decode_elapsed > 0 else 0.0
    overall_tps = (num_prefill + generated_count) / total_elapsed if total_elapsed > 0 else 0.0
    print(
        "\n\n--------------------------------------------------------\n"
        "  Generated Output\n"
        "--------------------------------------------------------\n"
        f"{text}\n"
        "--------------------------------------------------------\n\n"
        "  Performance Summary\n"
        "--------------------------------------------------------\n"
        f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
        "  ------------------------------------------------\n"
        f"  {'Prefill':<12} {prefill_tps:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
        f"  {'Decode':<12} {decode_tps:>10.2f} t/s {generated_count:>8d} {decode_elapsed:>8.3f}s\n"
        "  ------------------------------------------------\n"
        f"  {'Overall':<12} {overall_tps:>10.2f} t/s {generated_count:>8d} {total_elapsed:>8.3f}s\n"
        "--------------------------------------------------------\n"
    )
    return text


print("\nStart running the Qwen3.5-VL LLM by merged ONNXRuntime.\nNow loading . . . it could cost minutes.")
image_preprocess_sess, video_preprocess_sess, vision_sess, vision_batch_size, input_image_size, input_video_size, video_num_frames = _infer_vision_geometry()

valid_images = [path for path in (TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]) if is_valid_image_path(path)]
test_modes = []
if RUN_MODE == "image":
    if not valid_images:
        raise RuntimeError("--mode image requested, but no valid TEST_IMAGE path is configured.")
    test_modes.append(("image", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
elif RUN_MODE == "video":
    if not is_valid_video_path(TEST_VIDEO):
        raise RuntimeError("--mode video requested, but TEST_VIDEO is not valid.")
    test_modes.append(("video", TEST_QUERY[1] if isinstance(TEST_QUERY, list) and len(TEST_QUERY) > 1 else TEST_QUERY[0]))
elif RUN_MODE == "text":
    test_modes.append(("text", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
else:
    if valid_images:
        test_modes.append(("image", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))
    if is_valid_video_path(TEST_VIDEO):
        test_modes.append(("video", TEST_QUERY[1] if isinstance(TEST_QUERY, list) and len(TEST_QUERY) > 1 else TEST_QUERY[0]))
    if not test_modes:
        test_modes.append(("text", TEST_QUERY[0] if isinstance(TEST_QUERY, list) else TEST_QUERY))

for input_mode, current_query in test_modes:
    print(f"\n{'=' * 56}\n  Running test: mode={input_mode}, query=\"{current_query}\"\n{'=' * 56}")
    vision_outputs = _run_vision(input_mode, image_preprocess_sess, video_preprocess_sess, vision_sess, vision_batch_size, input_image_size, input_video_size, video_num_frames)
    run_merged_iobinding(input_mode, current_query, vision_outputs)