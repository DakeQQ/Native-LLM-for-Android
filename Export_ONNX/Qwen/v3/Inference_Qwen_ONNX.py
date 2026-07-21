import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Qwen v3 merged-ONNX inference demo.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen_Optimized",
        help="Folder containing merged ONNX graphs and LLM_SharedInitializers.onnx(.data).",
    )
    parser.add_argument(
        "--tokenizer-folder",
        type=Path,
        default=Path.home() / "Downloads" / "Qwen3-0.6B",
        help="HF checkpoint/tokenizer folder used to tokenize the demo prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional cap for generated tokens.",
    )
    return parser.parse_args()


args = parse_args()

METADATA_MODEL_NAME = "LLM_Metadata.onnx"


download_path            = str(args.tokenizer_folder.expanduser().resolve())
onnx_folder              = args.model_folder.expanduser().resolve()
onnx_model_Metadata      = str(onnx_folder / METADATA_MODEL_NAME)
MAX_NEW_TOKENS           = args.max_new_tokens

TEST_THINK_MODE          = False
TEST_QUERY               = "地球最高的山峰是什么？"

USE_SAMPLING             = False
TEMPERATURE              = 0.8
TOP_P                    = 0.95
# Greedy decode: direct repeated-token logit multiplier, 0.0 ~ 1.0; no penalty = 1.0.
# USE_SAMPLING=True: standard repetition penalty, >= 1.0; no penalty = 1.0.
REPEAT_PENALTY           = 0.9
PENALTY_RANGE            = 20
TOP_K                    = 10

ORT_LOG                  = False
ORT_FP16                 = False                   # CPU FP16 requires ARM64-v8.2a+.
ORT_Accelerate_Providers = []                      # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                       # 0 lets ORT choose the thread count.
DEVICE_ID                = 0

TOP_K = max(1, TOP_K)

USE_PENALTY = REPEAT_PENALTY != 1.0


_MODEL_FILE_METADATA_KEYS = {
    "prefill_greedy":           ("model_file_name_prefill_greedy", None),
    "prefill_penalty_greedy":   ("model_file_name_prefill_penalty_greedy", None),
    "prefill_sampling":         ("model_file_name_prefill_sampling", "LLM_TextPrefillSampling.onnx"),
    "decode_greedy":            ("model_file_name_decode_greedy", None),
    "decode_penalty_greedy":    ("model_file_name_decode_penalty_greedy", None),
    "decode_sampling":          ("model_file_name_decode_sampling", "LLM_DecodeSampling.onnx"),
    "shared_initializers":      ("model_file_name_shared_initializers", None),
    "shared_initializers_data": ("model_file_name_shared_initializers_data", None),
}

_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)


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


def load_model_file_names(meta):
    missing = [
        key for role, (key, default_file_name) in _MODEL_FILE_METADATA_KEYS.items()
        if not meta.get(key) and default_file_name is None
    ]
    if missing:
        raise RuntimeError(
            "LLM_Metadata.onnx is missing model file-name metadata. "
            "Re-export with Export_Qwen.py or re-quantize with Optimize_ONNX.py. Missing: "
            + ", ".join(missing)
        )

    file_names = {}
    for role, (key, default_file_name) in _MODEL_FILE_METADATA_KEYS.items():
        value = meta.get(key) or default_file_name
        path = Path(value)
        if path.is_absolute() or path.name != value:
            raise RuntimeError(f"Metadata key {key!r} must contain a file name, got {value!r}.")
        file_names[role] = value
    return file_names


# Read metadata before creating real sessions so ORT_FP16/session settings follow the exported graphs.
model_meta = load_metadata_carrier(onnx_model_Metadata)
MODEL_FILE_NAMES = load_model_file_names(model_meta)
ORT_FP16 = model_meta.get('activations_fp16') == '1'


def create_session_options():
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 0 if ORT_LOG else 4
    session_opts.log_verbosity_level = 4
    session_opts.inter_op_num_threads = MAX_THREADS
    session_opts.intra_op_num_threads = MAX_THREADS
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

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

    return session_opts


def create_run_options():
    options = onnxruntime.RunOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def resolve_execution_provider():
    if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
        return (
            'cpu',
            C.OrtDevice.cpu(),
            [{
                'device_type':              'CPU',
                'precision':                'ACCURACY',
                'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
                'num_streams':              1,
                'enable_opencl_throttling': False,
                'enable_qdq_optimizer':     False,
                'disable_dynamic_shapes':   False
            }],
        )

    if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        return (
            'cuda',
            C.OrtDevice.cuda(),
            [{
                'device_id':                          DEVICE_ID,
                'gpu_mem_limit':                      24 * (1024 ** 3),
                'arena_extend_strategy':              'kNextPowerOfTwo',
                'cudnn_conv_algo_search':             'EXHAUSTIVE',
                'sdpa_kernel':                        '2',
                'use_tf32':                           '1',
                'fuse_conv_bias':                     '1',
                'cudnn_conv_use_max_workspace':       '1',
                'cudnn_conv1d_pad_to_nc1d':           '0',
                'tunable_op_enable':                  '0',
                'tunable_op_tuning_enable':           '0',
                'tunable_op_max_tuning_duration_ms':  10,
                'do_copy_in_default_stream':          '0',
                'enable_cuda_graph':                  '0',
                'prefer_nhwc':                        '0',
                'enable_skip_layer_norm_strict_mode': '0',
                'use_ep_level_unified_stream':        '0'
            }],
        )

    if "DmlExecutionProvider" in ORT_Accelerate_Providers:
        return (
            'dml',
            C.OrtDevice.dml(),
            [{
                'device_id':                  DEVICE_ID,
                'performance_preference':     'high_performance',
                'device_filter':              'gpu',
                'disable_metacommands':       'false',
                'enable_graph_capture':       'false',
                'enable_graph_serialization': 'false'
            }],
        )

    return 'cpu', C.OrtDevice.cpu(), None


run_options = create_run_options()
device_type, _ort_device_type, provider_options = resolve_execution_provider()
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16
    else None
)
ORT_DEVICE = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = "cpu" if device_type == "dml" else device_type


def create_merged_session(model_path, shared_path):
    session_opts = create_session_options()
    shared_initializer_refs = attach_shared_initializers(session_opts, shared_path)
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_opts,
        providers=ORT_Accelerate_Providers,
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers,
    )
    session._native_llm_shared_initializers = shared_initializer_refs
    return session


def run_iobinding(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


print(
    '\nStart running the Qwen v3 LLM by ONNXRuntime.\n'
    'Now loading . . . it could cost minutes.'
)


def _merged_stop_tokens(meta):
    stops = [int(t) for t in meta.get('eos_token_ids', '').split(',') if t]
    if not stops:
        stops = [int(v) for k in ('chat_endoftext_id', 'chat_im_end_id') for v in (meta.get(k),) if v]
    return set(stops)


def _meta_int(meta, key):
    value = meta.get(key)
    if value is None:
        raise RuntimeError(f"LLM_Metadata.onnx is missing required key {key!r}.")
    return int(value)


def _kv_num_tensors(meta):
    kv_num = _meta_int(meta, 'kv_num_tensors')
    expected = _meta_int(meta, 'num_layers') * _meta_int(meta, 'kv_blocks_per_layer')
    if kv_num != expected:
        raise RuntimeError(f"Metadata KV tensor count mismatch: kv_num_tensors={kv_num}, expected={expected}.")
    return kv_num


def _generation_limit(meta, prompt_tokens):
    limit = max(0, _meta_int(meta, 'max_seq_len') - prompt_tokens)
    if MAX_NEW_TOKENS is not None:
        limit = min(limit, max(0, int(MAX_NEW_TOKENS)))
    return limit


def _merged_state_seq_axis(value_meta):
    symbolic_axes = [
        index for index, dim in enumerate(value_meta.shape)
        if index != 0 and not isinstance(dim, int)
    ]
    if len(symbolic_axes) == 1:
        return symbolic_axes[0]
    if value_meta.name.startswith('in_') and len(value_meta.shape) > 1:
        return len(value_meta.shape) - 1
    return None


def _merged_np_dtype(type_name):
    for key, dt in (('float16', np.float16), ('float', np.float32), ('uint8', np.uint8),
                    ('int8', np.int8), ('int32', np.int32), ('int64', np.int64)):
        if key in type_name:
            return dt
    raise ValueError(f"Unsupported ORT tensor type: {type_name}")


def _session_input_dtypes(sess):
    return {meta.name: _merged_np_dtype(meta.type) for meta in sess.get_inputs()}


def _merged_zero(meta):
    shape = list(meta.shape)
    seq_axis = _merged_state_seq_axis(meta)
    for i, dim in enumerate(shape):
        if i == 0:
            shape[i] = 1
        elif seq_axis is not None and i == seq_axis:
            shape[i] = 0
        elif not isinstance(dim, int):
            shape[i] = 1
    return np.zeros(tuple(shape), dtype=_merged_np_dtype(meta.type))


def _ov(arr, device=None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(arr), device or device_type, DEVICE_ID
    )


def _bind_outputs_device(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, ORT_DEVICE)


def _decode_dynamic_output_names(io_plan):
    names = list(io_plan['state_out'])
    if io_plan['save_id_out'] is not None:
        names.append(io_plan['save_id_out'])
    return names


def _save_id_in_names(strategy, is_decode, inputs):
    if strategy == 'greedy':
        candidates = []
    elif strategy == 'sampling':
        candidates = ['sampling_previous_ids']
    elif strategy == 'penalty_greedy' and is_decode:
        candidates = ['penalty_save_id_in', 'penalty_greedy_save_id_in']
    elif strategy == 'penalty_greedy':
        candidates = ['penalty_greedy_save_id_in']
    else:
        raise ValueError(f"Unknown decode strategy {strategy!r}.")
    missing = [name for name in candidates if name not in inputs]
    if missing:
        raise RuntimeError(f"Merged graph is missing expected save_id input(s): {missing}")
    return candidates


def plan_merged_io(sess, strategy, kv_num_tensors, is_decode):
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    if len(ins) < kv_num_tensors or len(outs) < kv_num_tensors:
        raise RuntimeError(
            f"Merged graph has too few I/O values for {kv_num_tensors} KV tensors: "
            f"inputs={len(ins)}, outputs={len(outs)}."
        )
    state_in = ins[:kv_num_tensors]
    state_out = outs[:kv_num_tensors]
    bad_inputs = [name for name in state_in if not name.startswith('in_')]
    bad_outputs = [name for name in state_out if not name.startswith('out_')]
    if bad_inputs or bad_outputs:
        raise RuntimeError(
            "Merged graph KV block is not leading/positional as expected: "
            f"bad inputs={bad_inputs[:3]}, bad outputs={bad_outputs[:3]}."
        )

    tail = outs[kv_num_tensors:]
    if strategy == 'greedy':
        if len(tail) < 2:
            raise RuntimeError(f"Greedy merged graph output tail is too short: {tail}")
        max_idx_out, kv_seq_out = tail[:2]
        save_id_out = None
    elif strategy == 'sampling':
        if len(tail) < 3:
            raise RuntimeError(f"Sampling merged graph output tail is too short: {tail}")
        max_idx_out, save_id_out, kv_seq_out = tail[:3]
    elif strategy == 'penalty_greedy':
        if len(tail) < 3:
            raise RuntimeError(f"Penalty-greedy merged graph output tail is too short: {tail}")
        max_idx_out, save_id_out, kv_seq_out = tail[:3]
    else:
        raise ValueError(f"Unknown decode strategy {strategy!r}.")

    kv_seq_in = None
    if is_decode:
        kv_seq_in = ins[kv_num_tensors]
        if not kv_seq_in.startswith('decode_kv_seq_len'):
            raise RuntimeError(f"Decode graph expected decode_kv_seq_len after KV inputs, got {kv_seq_in!r}.")

    return {
        'in_names': ins,
        'out_names': outs,
        'state_in': state_in,
        'state_out': state_out,
        'max_idx_out': max_idx_out,
        'kv_seq_out': kv_seq_out,
        'kv_seq_in': kv_seq_in,
        'save_id_in': _save_id_in_names(strategy, is_decode, set(ins)),
        'save_id_out': save_id_out,
    }


def _decode_static_inputs(strategy, input_dtypes):
    static_inputs = []
    if strategy == 'penalty_greedy':
        static_inputs.extend([
            ('penalty_penalty_value', _ov(np.array([REPEAT_PENALTY], input_dtypes['penalty_penalty_value']))),
            ('penalty_penalty_range', _ov(np.array([PENALTY_RANGE], input_dtypes['penalty_penalty_range']))),
        ])
    if strategy == 'sampling':
        static_inputs.extend([
            ('sampling_temperature', _ov(np.array([TEMPERATURE], input_dtypes['sampling_temperature']))),
            ('sampling_top_k', _ov(np.array([TOP_K], input_dtypes['sampling_top_k']))),
            ('sampling_top_p', _ov(np.array([TOP_P], input_dtypes['sampling_top_p']))),
            ('sampling_repetition_penalty', _ov(np.array([REPEAT_PENALTY], input_dtypes['sampling_repetition_penalty']))),
        ])
    return static_inputs


def run_merged_iobinding(folder, meta, strategy, model_file_names):
    """Run a merged prefill graph once, then ping-pong two decode bindings per token."""
    graph_pair = {
        'greedy':         (model_file_names['prefill_greedy'],         model_file_names['decode_greedy']),
        'penalty_greedy': (model_file_names['prefill_penalty_greedy'], model_file_names['decode_penalty_greedy']),
        'sampling':       (model_file_names['prefill_sampling'],       model_file_names['decode_sampling']),
    }
    if strategy not in graph_pair:
        raise ValueError(f"Unknown decode strategy {strategy!r}.")
    prefill_name, decode_name = graph_pair[strategy]
    is_sampling = strategy == 'sampling'
    shared_path = folder / model_file_names['shared_initializers']
    shared_data_path = folder / model_file_names['shared_initializers_data']
    if not shared_path.exists() or not shared_data_path.exists():
        raise RuntimeError(
            f"Merged runtime requires {model_file_names['shared_initializers']} and "
            f"{model_file_names['shared_initializers_data']} in {folder}."
        )
    for graph in (prefill_name, decode_name):
        if not (folder / graph).exists():
            raise RuntimeError(f"Merged graph {graph} missing for strategy {strategy!r} in {folder}.")

    prefill_sess = create_merged_session(folder / prefill_name, shared_path)
    decode_sess = create_merged_session(folder / decode_name, shared_path)
    print(f"Usable Providers: {decode_sess.get_providers()}")

    kv_num_tensors = _kv_num_tensors(meta)
    prefill_io_plan = plan_merged_io(prefill_sess, strategy, kv_num_tensors, is_decode=False)
    decode_io_plan = plan_merged_io(decode_sess, strategy, kv_num_tensors, is_decode=True)
    if decode_io_plan['kv_seq_in'] is None:
        raise RuntimeError("Decode graph is missing its decode_kv_seq_len input.")
    if 'attention_mask' in decode_io_plan['in_names']:
        raise RuntimeError("Merged decode graph unexpectedly exposes an attention_mask input.")
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    prompt = (
        f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
        if TEST_THINK_MODE else
        f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )
    tokens = tokenizer(prompt, return_tensors='np')['input_ids']
    num_prefill = int(tokens.shape[-1])
    generate_limit = _generation_limit(meta, num_prefill)
    stop_set = _merged_stop_tokens(meta)

    decode_input_dtypes = _session_input_dtypes(decode_sess)

    prefill_input_meta_by_name = {i.name: i for i in prefill_sess.get_inputs()}
    prefill_input_dtypes = _session_input_dtypes(prefill_sess)
    prefill_input_values = []
    prefill_binding = prefill_sess.io_binding()

    def bind_prefill_input(name, arr, device=None):
        # Cast every runtime-built input to the dtype the graph declares (metadata-driven,
        # never hard-coded) so an export dtype change can't silently mismatch the binding.
        value = _ov(np.asarray(arr).astype(prefill_input_dtypes[name], copy=False), device)
        prefill_input_values.append(value)
        prefill_binding.bind_ortvalue_input(name, value)

    bind_prefill_input('input_ids', tokens)
    bind_prefill_input('prefill_ids_len', np.array([num_prefill]))
    bind_prefill_input('prefill_history_len', np.array([0]))
    bind_prefill_input('prefill_cache_len', np.array([0]))
    for name in prefill_io_plan['state_in']:
        bind_prefill_input(name, _merged_zero(prefill_input_meta_by_name[name]), kv_device)
    for name in prefill_io_plan['save_id_in']:
        bind_prefill_input(name, np.zeros((1, 0)))
    if is_sampling:
        bind_prefill_input('sampling_temperature', np.array([TEMPERATURE]))
        bind_prefill_input('sampling_top_k', np.array([TOP_K]))
        bind_prefill_input('sampling_top_p', np.array([TOP_P]))
        bind_prefill_input('sampling_repetition_penalty', np.array([REPEAT_PENALTY]))
    _bind_outputs_device(prefill_binding, prefill_io_plan['out_names'])

    prefill_start_time = time.time()
    run_iobinding(prefill_sess, prefill_binding)
    prefill_elapsed = time.time() - prefill_start_time
    # Outputs are bound in out_names order, so get_outputs() aligns 1:1 with out_names. Gather
    # them positionally (integer index / list slice) instead of rebuilding a name->value dict:
    # the KV state block always leads the outputs, and the tails sit at fixed offsets.
    prefill_outputs = prefill_binding.get_outputs()
    prefill_out_names = prefill_io_plan['out_names']
    prefill_max_idx_pos = prefill_out_names.index(prefill_io_plan['max_idx_out'])
    prefill_kv_seq_pos = prefill_out_names.index(prefill_io_plan['kv_seq_out'])
    prefill_next_token_pos = prefill_max_idx_pos
    prefill_save_id_pos = prefill_out_names.index(prefill_io_plan['save_id_out']) if prefill_io_plan['save_id_out'] else -1

    # Hoist every decode-graph plan field into a local so the per-token loop touches no dict.
    decode_out_names = decode_io_plan['out_names']
    decode_token_in = 'input_ids'
    decode_kv_seq_in = decode_io_plan['kv_seq_in']
    decode_state_in = decode_io_plan['state_in']
    decode_save_id_in = decode_io_plan['save_id_in']
    decode_max_idx_pos = decode_out_names.index(decode_io_plan['max_idx_out'])
    decode_kv_seq_pos = decode_out_names.index(decode_io_plan['kv_seq_out'])
    decode_next_token_pos = decode_max_idx_pos
    decode_save_id_pos = decode_out_names.index(decode_io_plan['save_id_out']) if decode_io_plan['save_id_out'] else -1
    decode_dynamic_out_names = _decode_dynamic_output_names(decode_io_plan)

    generated = []
    selected_token_id = prefill_outputs[prefill_max_idx_pos].numpy().flat[0]
    next_token_tensor = prefill_outputs[prefill_next_token_pos]
    kv_sequence_length = prefill_outputs[prefill_kv_seq_pos]
    cached_state_tensors = prefill_outputs[:kv_num_tensors]
    saved_token_ids = prefill_outputs[prefill_save_id_pos] if prefill_save_id_pos >= 0 else None
    generated_count = 0
    print(f"\nTest Question: {TEST_QUERY}\nLLM Answering:")
    if selected_token_id not in stop_set and generated_count < generate_limit:
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
    # next step. KV state and saved_token_ids GROW every step, so ORT re-allocates them (fresh
    # handle) and they -- plus their device outputs -- must be rebound every step. kv_seq_len /
    # next_token is a fixed-shape output bound once, so each binding keeps reading
    # the *same* peer buffer (overwritten in place); their source only shifts while the ping-
    # pong warms up (prefill -> peer), so bind them on a binding's first two uses and skip the
    # otherwise-redundant per-step rebind afterward.
    control_rebinds_left = [2, 2]

    decode_step = 0
    decode_start_time = time.time()
    while generated_count < generate_limit and selected_token_id not in stop_set:
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

        cached_state_tensors = decode_outputs[:kv_num_tensors]
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
        if selected_token_id not in stop_set:
            generated.append(selected_token_id)
            generated_count += 1
            print(tokenizer.decode(selected_token_id), end="", flush=True)
        decode_step += 1
    decode_elapsed = time.time() - decode_start_time

    text = tokenizer.decode(generated, skip_special_tokens=True)
    total_elapsed = prefill_elapsed + decode_elapsed
    prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
    decode_tokens_per_second = decode_step / decode_elapsed if decode_elapsed > 0 else 0.0
    overall_tokens_per_second = (num_prefill + generated_count) / total_elapsed if total_elapsed > 0 else 0.0

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
        f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
        f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {generated_count:>8d} {decode_elapsed:>8.3f}s\n"
        "  ------------------------------------------------\n"
        f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {generated_count:>8d} {total_elapsed:>8.3f}s\n"
        "--------------------------------------------------------\n"
    )
    return text


def _resolve_strategy():
    if USE_SAMPLING:
        return 'sampling'
    if USE_PENALTY:
        return 'penalty_greedy'
    return 'greedy'


DECODE_STRATEGY = _resolve_strategy()


_required_shared_files = (MODEL_FILE_NAMES['shared_initializers'], MODEL_FILE_NAMES['shared_initializers_data'])
_missing_shared_files = [name for name in _required_shared_files if not (onnx_folder / name).exists()]
if _missing_shared_files:
    raise RuntimeError(
        "Merged shared initializer files not found in "
        f"{onnx_folder}. This runtime is merged-only; re-export with Export_Qwen.py or "
        "re-quantize with Optimize_ONNX.py to produce merged graphs and "
        f"{', '.join(_required_shared_files)}. Missing: {', '.join(_missing_shared_files)}."
    )
run_merged_iobinding(onnx_folder, model_meta, DECODE_STRATEGY, MODEL_FILE_NAMES)